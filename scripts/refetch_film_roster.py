"""Refetch "from scratch" le roster (XUID -> Gamertag) depuis les fichiers film Halo Infinite.

Objectif
- Obtenir des gamertags fiables même quand `HighlightEvents.gamertag` est corrompu.
- Optionnel: patcher `HighlightEvents.ResponseBody` dans la DB pour remplacer `gamertag`.

Principe (inspiré des analyses publiques de film chunks)
- Récupère le manifest: /hi/films/matches/{matchId}/spectate
- Télécharge les chunks (types 1/2: roster)
- Décompresse (zlib)
- Scanne un marqueur binaire (0x2D 0xC0) à tous les offsets de bits (0..7)
- Dans la vue bit-alignée, le pattern observé est: XUID (LE64) puis MARKER à +8 bytes.
- Le gamertag est juste avant le XUID, et se décode en UTF-16BE (empirique sur filmChunk0 type1).

Pré-requis
- Auth headers (le plus simple):
  - SPNKR_SPARTAN_TOKEN
  - SPNKR_CLEARANCE_TOKEN

Exemples
- Extraire le roster et écrire des alias:
  python scripts/refetch_film_roster.py --match-id <MATCH_GUID> --write-aliases

- Extraire le roster et patcher la DB:
  python scripts/refetch_film_roster.py --match-id <MATCH_GUID> --db path\\to\\db.sqlite --patch-highlight-events

Note
- Script expérimental: la structure exacte des chunks peut évoluer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


try:
    import aiohttp
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "aiohttp manquant. Installe-le (souvent via `pip install spnkr aiohttp`) puis relance.\n"
        f"Détail: {e}"
    )


CLEARANCE_COOKIE_RE = re.compile(r"(?:^|[;\s])343-clearance=([^;\s]+)", re.IGNORECASE)
MARKER = b"\x2d\xc0"


def _load_dotenv_if_present() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for name in (".env.local", ".env"):
        p = repo_root / name
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.environ.get(key) is None:
                os.environ[key] = value


def _normalize_token_value(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""

    if ":" in s:
        _, after = s.split(":", 1)
        s = after.strip()

    m = CLEARANCE_COOKIE_RE.search(s)
    if m:
        return (m.group(1) or "").strip().strip('"').strip("'")

    return s


def _build_headers() -> dict[str, str]:
    raise RuntimeError("Use _get_headers(args) instead")


@dataclass(frozen=True)
class Tokens:
    spartan_token: str
    clearance_token: str


async def _get_tokens(args: argparse.Namespace) -> Tokens:
    spartan = _normalize_token_value(getattr(args, "spartan_token", None) or os.environ.get("SPNKR_SPARTAN_TOKEN"))
    clearance = _normalize_token_value(
        getattr(args, "clearance_token", None) or os.environ.get("SPNKR_CLEARANCE_TOKEN")
    )
    if spartan and clearance:
        return Tokens(spartan_token=spartan, clearance_token=clearance)

    # Fallback: si on a les secrets Azure + refresh token, on demande à SPNKr.
    azure_client_id = getattr(args, "azure_client_id", None) or os.environ.get("SPNKR_AZURE_CLIENT_ID")
    azure_client_secret = getattr(args, "azure_client_secret", None) or os.environ.get("SPNKR_AZURE_CLIENT_SECRET")
    azure_redirect_uri = (
        getattr(args, "azure_redirect_uri", None)
        or os.environ.get("SPNKR_AZURE_REDIRECT_URI")
        or "https://localhost"
    )
    oauth_refresh_token = (
        getattr(args, "oauth_refresh_token", None) or os.environ.get("SPNKR_OAUTH_REFRESH_TOKEN")
    )

    if azure_client_id and azure_client_secret and oauth_refresh_token:
        try:
            from aiohttp import ClientSession, ClientTimeout
            from spnkr import AzureApp, refresh_player_tokens
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Dépendance manquante pour l'auth Azure (SPNKr). Installe SPNKr, ex:\n"
                "pip install 'spnkr @ git+https://github.com/acurtis166/SPNKr.git'\n"
                "(et ses dépendances, dont aiohttp)."
            ) from e

        async def _refresh_oauth_access_token_v2(session: ClientSession, refresh_token: str, app: AzureApp) -> str:
            url = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
            data = {
                "client_id": app.client_id,
                "client_secret": app.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": "Xboxlive.signin Xboxlive.offline_access",
            }
            resp = await session.post(url, data=data)
            payload = await resp.json()
            if resp.status >= 400:
                raise SystemExit(
                    "Échec refresh OAuth v2 (consumers). "
                    f"status={resp.status} error={payload.get('error')} desc={payload.get('error_description')}"
                )
            access = payload.get("access_token")
            if not isinstance(access, str) or not access.strip():
                raise SystemExit("OAuth v2: pas de access_token dans la réponse.")
            return access.strip()

        app = AzureApp(azure_client_id, azure_client_secret, azure_redirect_uri)
        async with ClientSession(timeout=ClientTimeout(total=45)) as session:
            try:
                player = await refresh_player_tokens(session, app, oauth_refresh_token)
                return Tokens(
                    spartan_token=str(player.spartan_token.token),
                    clearance_token=str(player.clearance_token.token),
                )
            except Exception as e:
                msg = str(e)
                if "invalid_client" not in msg or "client_secret" not in msg:
                    raise

                # Fallback: endpoint OAuth v2 (consumers) pour obtenir un access_token MSA,
                # puis chain Xbox/XSTS/Halo comme SPNKr.
                from spnkr.auth.xbox import request_user_token, request_xsts_token
                from spnkr.auth.core import XSTS_V3_HALO_AUDIENCE, XSTS_V3_XBOX_AUDIENCE
                from spnkr.auth.halo import request_spartan_token, request_clearance_token

                access_token = await _refresh_oauth_access_token_v2(session, oauth_refresh_token, app)
                user_token = await request_user_token(session, access_token)
                _ = await request_xsts_token(session, user_token.token, XSTS_V3_XBOX_AUDIENCE)
                halo_xsts_token = await request_xsts_token(session, user_token.token, XSTS_V3_HALO_AUDIENCE)
                spartan_token = await request_spartan_token(session, halo_xsts_token.token)
                clearance_token = await request_clearance_token(session, spartan_token.token)

                return Tokens(
                    spartan_token=str(spartan_token.token),
                    clearance_token=str(clearance_token.token),
                )

    raise SystemExit(
        "Tokens manquants. Fournis soit:\n"
        "- SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN (env),\n"
        "- ou --spartan-token/--clearance-token,\n"
        "- ou l'option Azure: --azure-client-id/--azure-client-secret/--oauth-refresh-token (ou via .env.local)."
    )


async def _get_headers(args: argparse.Namespace) -> dict[str, str]:
    t = await _get_tokens(args)
    return {
        "accept": "application/json",
        "x-343-authorization-spartan": t.spartan_token,
        "343-clearance": t.clearance_token,
        "user-agent": "openspartan-graph/film-roster-refetch",
    }


@dataclass(frozen=True)
class FilmFileInfo:
    file_type_id: int
    relative_path: str


@dataclass(frozen=True)
class ScanStats:
    bit_offset: int
    marker_hits: int
    best_pairs_found: int


@dataclass
class Candidate:
    name: str
    best_score: int
    hits: int


def _iter_file_infos(obj: Any) -> Iterable[FilmFileInfo]:
    if isinstance(obj, dict):
        # Selon l'endpoint / la version, le type peut s'appeler FileTypeId ou ChunkType.
        if "FileRelativePath" in obj and ("FileTypeId" in obj or "ChunkType" in obj):
            rel = obj.get("FileRelativePath")
            fti = obj.get("FileTypeId", obj.get("ChunkType"))
            if isinstance(rel, str):
                try:
                    file_type_id = int(fti)
                except Exception:
                    file_type_id = -1
                yield FilmFileInfo(file_type_id=file_type_id, relative_path=rel)

        for v in obj.values():
            yield from _iter_file_infos(v)
        return

    if isinstance(obj, list):
        for it in obj:
            yield from _iter_file_infos(it)


def _shift_bytes(data: bytes, bit_offset: int) -> bytes:
    if bit_offset == 0:
        return data
    if not (0 <= bit_offset <= 7):
        raise ValueError("bit_offset must be 0..7")

    out = bytearray(max(0, len(data) - 1))
    inv = 8 - bit_offset
    for i in range(len(out)):
        out[i] = ((data[i] << bit_offset) & 0xFF) | (data[i + 1] >> inv)
    return bytes(out)


def _looks_like_gamertag(s: str) -> bool:
    v = (s or "").strip()
    if not v:
        return False
    if "\x00" in v:
        return False
    if any(ord(ch) < 32 for ch in v):
        return False
    # Gamertags Xbox peuvent contenir des caractères Unicode (accents, etc.).
    # On garde une heuristique légère: imprimable, taille plausible, au moins un alnum.
    if not v.isprintable():
        return False
    if not (3 <= len(v) <= 20):
        return False
    if not any(ch.isalnum() for ch in v):
        return False
    # Caractères trop “suspects” pour un gamertag.
    if any(ch in "\uFFFD" for ch in v):
        return False
    return True


def _decode_utf16be_best_effort(buf: bytes) -> str:
    try:
        s = buf.decode("utf-16be", errors="ignore")
    except Exception:
        return ""
    return s.replace("\x00", "").strip()


_GT_ALLOWED_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]{2,19}")


def _extract_name_before_xuid(view: bytes, xuid_pos: int) -> str:
    """Extrait un gamertag situé juste avant un XUID (vue déjà bit-alignée).

    Empiriquement (sur chunk type1/filmChunk0):
    - XUID est en little-endian à `xuid_pos`
    - MARKER (0x2D 0xC0) est à `xuid_pos + 8`
    - Le gamertag est stocké juste avant, encodé en UTF-16BE.
    """

    start = max(0, xuid_pos - 160)
    win = view[start:xuid_pos]
    if not win:
        return ""

    # Méthode 1: recule sur des paires 0x00 + ASCII imprimable.
    i = len(win) - 2
    while i >= 0 and win[i : i + 2] == b"\x00\x00":
        i -= 2
    if i >= 1:
        j = i
        while j >= 1:
            hi = win[j - 1]
            lo = win[j]
            if hi == 0x00 and 32 <= lo <= 126:
                j -= 2
                continue
            break
        raw = win[j + 1 : i + 1]
        name = _decode_utf16be_best_effort(raw)
        if _looks_like_gamertag(name):
            return name

    # Méthode 2: décodage large puis regex (fallback quand ce n'est pas strict ASCII).
    decoded = _decode_utf16be_best_effort(win)
    if not decoded:
        return ""
    matches = list(_GT_ALLOWED_RE.finditer(decoded))
    for m in reversed(matches):
        cand = (m.group(0) or "").strip()
        if _looks_like_gamertag(cand):
            return cand
    return ""


def _looks_like_xuid(x: int) -> bool:
    # XUIDs: 12-20 digits en pratique.
    return 10**11 <= x <= 10**20


def extract_roster_from_chunk(chunk: bytes) -> dict[int, str]:
    """Extrait des paires (xuid -> gamertag) depuis un chunk décompressé.

    Implémentation structurelle (rapide): on cherche MARKER, on lit le XUID à marker-8,
    et on extrait le nom juste avant le XUID (UTF-16BE, vue bit-alignée).
    """

    mapping: dict[int, tuple[str, int]] = {}  # xuid -> (name, score)

    for bit_off in range(8):
        view = _shift_bytes(chunk, bit_off)
        start = 0
        while True:
            idx = view.find(MARKER, start)
            if idx < 0:
                break

            x_pos = idx - 8
            if x_pos >= 0 and x_pos + 8 <= len(view):
                xuid = int.from_bytes(view[x_pos : x_pos + 8], "little", signed=False)
                if _looks_like_xuid(xuid):
                    name = _extract_name_before_xuid(view, x_pos)
                    if _looks_like_gamertag(name):
                        # Score simple: privilégie les noms plus longs et bit_off faible (stabilité)
                        score = 100 + min(len(name), 20) * 2 - bit_off
                        prev = mapping.get(xuid)
                        if prev is None or score > prev[1]:
                            mapping[xuid] = (name, score)

            start = idx + 2

    return {xuid: name for xuid, (name, _score) in mapping.items()}


def extract_candidates_from_chunk(chunk: bytes) -> tuple[dict[int, Candidate], list[ScanStats]]:
    """Extraction "investigation": retourne des candidats (xuid -> {name,best_score,hits}).

    Version structurelle (beaucoup moins de faux positifs):
    - Cherche MARKER
    - XUID = bytes(marker-8..marker)
    - Nom = juste avant XUID (UTF-16BE)
    """

    stats: list[ScanStats] = []
    candidates: dict[int, Candidate] = {}

    for bit_off in range(8):
        view = _shift_bytes(chunk, bit_off)
        start = 0
        marker_hits = 0
        best_pairs = 0

        while True:
            idx = view.find(MARKER, start)
            if idx < 0:
                break
            marker_hits += 1

            x_pos = idx - 8
            if x_pos < 0 or x_pos + 8 > len(view):
                start = idx + 2
                continue

            xuid = int.from_bytes(view[x_pos : x_pos + 8], "little", signed=False)
            if not _looks_like_xuid(xuid):
                start = idx + 2
                continue

            name = _extract_name_before_xuid(view, x_pos)
            if not _looks_like_gamertag(name):
                start = idx + 2
                continue

            best_pairs += 1
            score = 100 + min(len(name), 20) * 2 - bit_off
            cur = candidates.get(xuid)
            if cur is None:
                candidates[xuid] = Candidate(name=name, best_score=score, hits=1)
            else:
                cur.hits += 1
                if score > cur.best_score:
                    cur.best_score = score
                    cur.name = name

            start = idx + 2

        stats.append(ScanStats(bit_offset=bit_off, marker_hits=marker_hits, best_pairs_found=best_pairs))

    return candidates, stats


def _merge_candidates(into: dict[int, Candidate], part: dict[int, Candidate]) -> None:
    for xuid, c in part.items():
        cur = into.get(xuid)
        if cur is None:
            into[xuid] = Candidate(name=c.name, best_score=c.best_score, hits=c.hits)
        else:
            cur.hits += c.hits
            if c.best_score > cur.best_score:
                cur.best_score = c.best_score
                cur.name = c.name


def _finalize_roster(
    candidates: dict[int, Candidate],
    *,
    expected_xuids: set[int] | None,
    expected_players: int | None,
) -> dict[int, str]:
    if expected_xuids:
        out: dict[int, str] = {}
        for x in sorted(expected_xuids):
            c = candidates.get(x)
            if c is not None and c.name:
                out[x] = c.name
        return out

    # Sinon: on prend les meilleurs candidats.
    ranked = sorted(
        candidates.items(),
        key=lambda kv: (kv[1].hits, kv[1].best_score, len(kv[1].name)),
        reverse=True,
    )
    limit = expected_players or 24
    return {x: c.name for x, c in ranked[:limit] if c.name}


def _zlib_decompress(data: bytes) -> bytes:
    # Certaines réponses sont en zlib « classique », d'autres en raw deflate.
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(data, wbits=wbits)
        except Exception:
            continue
    # Dernière chance: auto
    return zlib.decompress(data)


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} on {url}: {text[:400]}")
        return await resp.json()


async def _fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} on {url}: {text[:400]}")
        return await resp.read()


async def fetch_roster_for_match(*, match_id: str, headers: dict[str, str]) -> dict[int, str]:
    manifest_url = (
        "https://discovery-infiniteugc.svc.halowaypoint.com"
        f"/hi/films/matches/{match_id}/spectate"
    )

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        manifest = await _fetch_json(session, manifest_url)

        prefix = manifest.get("BlobStoragePathPrefix") if isinstance(manifest, dict) else None
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError("Manifest: BlobStoragePathPrefix manquant.")

        files = list(_iter_file_infos(manifest))
        roster_files = [f for f in files if f.file_type_id in (1, 2)]
        if not roster_files:
            raise RuntimeError("Manifest: aucun chunk roster (type 1/2) trouvé.")

        async def _dl_and_extract(fi: FilmFileInfo) -> dict[int, str]:
            url = prefix.rstrip("/") + "/" + fi.relative_path.lstrip("/")
            raw = await _fetch_bytes(session, url)
            dec = _zlib_decompress(raw)
            return extract_roster_from_chunk(dec)

        parts = await asyncio.gather(*[_dl_and_extract(fi) for fi in roster_files])

    merged: dict[int, str] = {}
    for m in parts:
        for x, gt in m.items():
            if x not in merged:
                merged[x] = gt
    return merged


async def fetch_manifest(*, match_id: str, headers: dict[str, str]) -> Any:
    manifest_url = (
        "https://discovery-infiniteugc.svc.halowaypoint.com"
        f"/hi/films/matches/{match_id}/spectate"
    )
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        return await _fetch_json(session, manifest_url)


async def download_and_decompress_chunks(
    *,
    manifest: Any,
    headers: dict[str, str],
    file_type_ids: set[int],
    out_dir: Path | None,
    max_total_chunks: int | None = None,
    max_type2_chunks: int | None = None,
) -> list[tuple[FilmFileInfo, bytes]]:
    if not isinstance(manifest, dict):
        raise RuntimeError("Manifest invalide (pas un objet JSON).")
    prefix = manifest.get("BlobStoragePathPrefix")
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError("Manifest: BlobStoragePathPrefix manquant.")

    files = [f for f in _iter_file_infos(manifest) if f.file_type_id in file_type_ids]
    if not files:
        raise RuntimeError(f"Manifest: aucun chunk trouvé pour types={sorted(file_type_ids)}")

    # Limites optionnelles pour éviter de télécharger des tonnes de chunks type2.
    if max_type2_chunks is not None and max_type2_chunks >= 0:
        type1 = [f for f in files if f.file_type_id == 1]
        type2 = sorted((f for f in files if f.file_type_id == 2), key=lambda x: x.relative_path)[:max_type2_chunks]
        other = [f for f in files if f.file_type_id not in (1, 2)]
        files = type1 + type2 + other

    if max_total_chunks is not None and max_total_chunks >= 0:
        # Préserve l'ordre d'origine au maximum.
        files = files[:max_total_chunks]

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async def _dl(fi: FilmFileInfo) -> tuple[FilmFileInfo, bytes]:
            url = prefix.rstrip("/") + "/" + fi.relative_path.lstrip("/")
            raw = await _fetch_bytes(session, url)
            dec = _zlib_decompress(raw)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_name = fi.relative_path.replace("/", "_").replace("\\", "_")
                (out_dir / f"type{fi.file_type_id}__{safe_name}.bin").write_bytes(dec)
            return fi, dec

        return await asyncio.gather(*[_dl(fi) for fi in files])


def _load_highlight_xuids(db_path: Path, match_id: str) -> set[int]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT DISTINCT Xuid FROM HighlightEvents WHERE MatchId = ?", (match_id,))
        out: set[int] = set()
        for (x,) in cur.fetchall():
            try:
                out.add(int(str(x)))
            except Exception:
                continue
        return out
    finally:
        con.close()


def _merge_aliases(aliases_path: Path, roster: dict[int, str]) -> int:
    try:
        existing = json.loads(aliases_path.read_text(encoding="utf-8")) if aliases_path.exists() else {}
    except Exception:
        existing = {}

    if not isinstance(existing, dict):
        existing = {}

    changed = 0
    for xuid, gt in roster.items():
        k = str(xuid)
        if existing.get(k) != gt:
            existing[k] = gt
            changed += 1

    aliases_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def _patch_highlight_events(db_path: Path, match_id: str, roster: dict[int, str]) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT rowid, ResponseBody FROM HighlightEvents WHERE MatchId = ?",
            (match_id,),
        )
        rows = cur.fetchall()
        updated = 0
        for rowid, raw_json in rows:
            try:
                payload = json.loads(raw_json)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            xuid = payload.get("xuid")
            try:
                xuid_int = int(str(xuid))
            except Exception:
                continue

            gt = roster.get(xuid_int)
            if not gt:
                continue

            current = payload.get("gamertag")
            if current == gt:
                continue

            # Préserve l'original s'il existe.
            if "gamertag_raw" not in payload and isinstance(current, str):
                payload["gamertag_raw"] = current
            payload["gamertag"] = gt

            cur.execute(
                "UPDATE HighlightEvents SET ResponseBody = ? WHERE rowid = ?",
                (json.dumps(payload, ensure_ascii=False), rowid),
            )
            updated += 1

        con.commit()
        return updated
    finally:
        con.close()


def _load_match_ids_from_db(db_path: Path, *, table: str) -> list[str]:
    if table not in {"MatchStats", "HighlightEvents"}:
        raise ValueError("table must be MatchStats or HighlightEvents")
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        if table == "HighlightEvents":
            cur.execute("SELECT DISTINCT MatchId FROM HighlightEvents")
        else:
            # MatchId est une colonne virtuelle dans le schéma import (json_extract)
            cur.execute("SELECT DISTINCT MatchId FROM MatchStats")
        rows = cur.fetchall()
        out: list[str] = []
        for (mid,) in rows:
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
        return out
    finally:
        con.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refetch film roster (XUID->Gamertag)")
    ap.add_argument("--match-id", default=None, help="Match GUID")
    ap.add_argument("--spartan-token", default=None, help="Valeur brute du header x-343-authorization-spartan")
    ap.add_argument("--clearance-token", default=None, help="Valeur brute du header 343-clearance")
    ap.add_argument("--azure-client-id", default=None, help="Azure App Registration client_id")
    ap.add_argument("--azure-client-secret", default=None, help="Azure App Registration client_secret")
    ap.add_argument(
        "--azure-redirect-uri",
        default=None,
        help="Azure redirect_uri (défaut: SPNKR_AZURE_REDIRECT_URI ou https://localhost)",
    )
    ap.add_argument(
        "--oauth-refresh-token",
        default=None,
        help="OAuth refresh token (défaut: SPNKR_OAUTH_REFRESH_TOKEN)",
    )
    ap.add_argument("--write-aliases", action="store_true", help="Merge dans xuid_aliases.json")
    ap.add_argument(
        "--aliases",
        default=str(Path(__file__).resolve().parent.parent / "xuid_aliases.json"),
        help="Chemin vers xuid_aliases.json",
    )
    ap.add_argument("--db", default=None, help="Chemin vers la DB SQLite (optionnel)")
    ap.add_argument(
        "--all-matches",
        action="store_true",
        help="Traite tous les MatchId présents dans --db (sinon: un seul match via --match-id)",
    )
    ap.add_argument(
        "--db-source-table",
        choices=["HighlightEvents", "MatchStats"],
        default="HighlightEvents",
        help="Table utilisée pour lister les MatchId quand --all-matches est activé",
    )
    ap.add_argument(
        "--patch-highlight-events",
        action="store_true",
        help="Si --db est fourni: remplace gamertag dans HighlightEvents via le roster refetch",
    )
    ap.add_argument(
        "--save-manifest",
        default=None,
        help="Écrit le JSON du manifest dans ce fichier (investigation)",
    )
    ap.add_argument(
        "--manifest-json",
        default=None,
        help="Utilise un manifest JSON local au lieu de le télécharger (investigation offline)",
    )
    ap.add_argument(
        "--save-chunks-dir",
        default=None,
        help="Dossier où écrire les chunks décompressés (investigation)",
    )
    ap.add_argument(
        "--local-chunk",
        action="append",
        default=[],
        help=(
            "Analyse un chunk déjà décompressé depuis un fichier local (peut être répété). "
            "Quand utilisé, aucun réseau n'est nécessaire pour l'extraction." 
        ),
    )
    ap.add_argument(
        "--verbose-scan",
        action="store_true",
        help="Affiche des stats de scan (markers / paires) par bit offset",
    )
    ap.add_argument(
        "--expected-players",
        type=int,
        default=None,
        help="Si pas de filtre XUID: limite le roster aux N meilleurs candidats",
    )
    ap.add_argument(
        "--include-type2",
        action="store_true",
        help="Inclut aussi les chunks type 2 (par défaut: type 1 seulement, moins de faux positifs)",
    )
    ap.add_argument(
        "--max-type2-chunks",
        type=int,
        default=None,
        help=(
            "Limite le nombre de chunks type2 téléchargés (utile en investigation pour éviter les gros downloads). "
            "0 = aucun type2."
        ),
    )
    ap.add_argument(
        "--max-total-chunks",
        type=int,
        default=None,
        help="Limite le nombre total de chunks téléchargés (type1+type2).",
    )
    ap.add_argument(
        "--print-limit",
        type=int,
        default=32,
        help="Limite le nombre de lignes affichées (évite les dumps gigantesques)",
    )
    return ap.parse_args(argv)


async def main_async(argv: list[str]) -> int:
    args = _parse_args(argv)
    _load_dotenv_if_present()

    # Mode offline pur: on analyse les fichiers fournis sans réseau/auth.
    if args.local_chunk:
        db_path = Path(args.db).expanduser().resolve() if args.db else None
        expected_xuids: set[int] | None = None
        if db_path is not None and args.match_id:
            try:
                expected_xuids = _load_highlight_xuids(db_path, args.match_id) or None
                if expected_xuids:
                    print(f"Filtre XUID (HighlightEvents): {len(expected_xuids)} joueurs")
            except Exception:
                expected_xuids = None

        candidates: dict[int, Candidate] = {}
        all_stats: list[ScanStats] = []
        for p in args.local_chunk:
            data = Path(p).expanduser().resolve().read_bytes()
            part, stats = extract_candidates_from_chunk(data)
            _merge_candidates(candidates, part)
            all_stats.extend(stats)

        roster_all = _finalize_roster(
            candidates,
            expected_xuids=expected_xuids,
            expected_players=args.expected_players,
        )

        print(f"Roster extrait (local chunks): {len(roster_all)} joueurs (candidats={len(candidates)})")
        shown = 0
        for xuid in sorted(roster_all.keys()):
            c = candidates.get(xuid)
            meta = f" hits={c.hits} score={c.best_score}" if c else ""
            print(f"- {xuid}: {roster_all[xuid]}{meta}")
            shown += 1
            if shown >= args.print_limit:
                break

        if args.verbose_scan and all_stats:
            print("\nStats scan (local):")
            for s in all_stats:
                print(f"- bit_offset={s.bit_offset}: markers={s.marker_hits} best_pairs={s.best_pairs_found}")

        if args.write_aliases:
            aliases_path = Path(args.aliases).expanduser().resolve()
            changed = _merge_aliases(aliases_path, roster_all)
            print(f"Aliases écrits: {changed} modifications -> {aliases_path}")

        return 0

    headers = await _get_headers(args)

    db_path = Path(args.db).expanduser().resolve() if args.db else None
    if args.all_matches:
        if db_path is None:
            raise SystemExit("--all-matches requiert --db")
        match_ids = _load_match_ids_from_db(db_path, table=args.db_source_table)
        if not match_ids:
            raise SystemExit(f"Aucun MatchId trouvé dans {args.db_source_table}.")
    else:
        if not args.match_id:
            raise SystemExit("Fournis --match-id ou active --all-matches (avec --db).")
        match_ids = [args.match_id]

    aliases_path = Path(args.aliases).expanduser().resolve() if args.write_aliases else None

    total_alias_changes = 0
    total_patched = 0
    for i, match_id in enumerate(match_ids, start=1):
        print(f"\n[{i}/{len(match_ids)}] MatchId={match_id}")

        if args.manifest_json:
            manifest = json.loads(Path(args.manifest_json).expanduser().resolve().read_text(encoding="utf-8"))
        else:
            manifest = await fetch_manifest(match_id=match_id, headers=headers)

        if args.save_manifest:
            out = Path(args.save_manifest).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        out_dir = Path(args.save_chunks_dir).expanduser().resolve() if args.save_chunks_dir else None
        type_ids = {1, 2} if args.include_type2 else {1}
        chunks = await download_and_decompress_chunks(
            manifest=manifest,
            headers=headers,
            file_type_ids=type_ids,
            out_dir=out_dir,
            max_total_chunks=args.max_total_chunks,
            max_type2_chunks=args.max_type2_chunks,
        )

        expected_xuids: set[int] | None = None
        if db_path is not None:
            try:
                hx = _load_highlight_xuids(db_path, match_id)
                expected_xuids = hx or None
                if expected_xuids:
                    print(f"Filtre XUID (HighlightEvents): {len(expected_xuids)} joueurs")
            except Exception:
                expected_xuids = None

        candidates: dict[int, Candidate] = {}
        all_stats: list[ScanStats] = []
        for _fi, dec in chunks:

            part, stats = extract_candidates_from_chunk(dec)
            _merge_candidates(candidates, part)
            all_stats.extend(stats)

        roster = _finalize_roster(
            candidates,
            expected_xuids=expected_xuids,
            expected_players=args.expected_players,
        )

        print(f"Roster extrait: {len(roster)} joueurs (candidats={len(candidates)})")
        shown = 0
        for xuid in sorted(roster.keys()):
            c = candidates.get(xuid)
            meta = f" hits={c.hits} score={c.best_score}" if c else ""
            print(f"- {xuid}: {roster[xuid]}{meta}")
            shown += 1
            if shown >= args.print_limit:
                break

        if aliases_path is not None:
            total_alias_changes += _merge_aliases(aliases_path, roster)

        if args.patch_highlight_events:
            if db_path is None:
                raise SystemExit("--patch-highlight-events requiert --db")
            total_patched += _patch_highlight_events(db_path, match_id, roster)

        if args.verbose_scan and all_stats:
            print("\nStats scan:")
            for s in all_stats:
                print(f"- bit_offset={s.bit_offset}: markers={s.marker_hits} best_pairs={s.best_pairs_found}")

    if aliases_path is not None:
        print(f"\nAliases écrits: {total_alias_changes} modifications -> {aliases_path}")
    if args.patch_highlight_events:
        print(f"HighlightEvents patchés: {total_patched} lignes")
    return 0


def main() -> int:
    return asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
