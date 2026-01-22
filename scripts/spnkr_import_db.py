"""Importe les matchs via SPNKr et génère une DB SQLite compatible OpenSpartan.

But:
- Rendre l'import des matchs plus fiable (source directe Waypoint API via SPNKr)
- Générer une DB lisible par ce projet (tables MatchStats/PlayerMatchStats/Maps/Playlists/...)

Pré-requis:
- `pip install spnkr aiohttp pydantic` (SPNKr installe déjà ses dépendances)

Authentification (deux options):
1) Tokens manuels (le plus simple):
   - définir les variables d'env:
     - SPNKR_SPARTAN_TOKEN
     - SPNKR_CLEARANCE_TOKEN

2) Azure App Registration (automatise le refresh):
   - fournir --azure-client-id/--azure-client-secret/--oauth-refresh-token

Notes:
- API non officielle: peut casser sans préavis.
- On stocke les payloads JSON bruts, au format attendu par OpenSpartan Workshop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


XUID_RE = re.compile(r"(\d{12,20})")
CLEARANCE_COOKIE_RE = re.compile(r"(?:^|[;\s])343-clearance=([^;\s]+)", re.IGNORECASE)


def _load_dotenv_if_present() -> None:
    """Charge un fichier `.env.local` puis `.env` à la racine du repo (si présents).

    Objectif: faciliter l'usage en local sans installer python-dotenv.
    Règles:
    - lignes `KEY=VALUE`
    - ignore lignes vides / commentaires (#)
    - ne remplace pas une variable déjà définie dans l'environnement
    """

    repo_root = Path(__file__).resolve().parent.parent

    for name in (".env.local", ".env"):
        dotenv_path = repo_root / name
        if not dotenv_path.exists():
            continue
        try:
            content = dotenv_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if os.environ.get(key) is None:
                os.environ[key] = value


def _normalize_token_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Autorise un copier-coller depuis l'onglet réseau, ex:
    # "x-343-authorization-spartan: v4=..."
    # "343-clearance: ..."
    if ":" in s:
        _, after = s.split(":", 1)
        s = after.strip()

    # Autorise un copier-coller depuis un header Cookie (fréquent sur certains setups), ex:
    # "Cookie: ...; 343-clearance=eyJ...; ..."
    m = CLEARANCE_COOKIE_RE.search(s)
    if m:
        return m.group(1).strip().strip('"').strip("'") or None

    # SPNKr attend généralement la valeur brute (incluant le préfixe v4= si présent)
    return s or None


def _coerce_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        return int(v)
    except Exception:
        return None


def _extract_xuids_from_match_stats(match_stats_json: dict[str, Any]) -> list[int]:
    players = match_stats_json.get("Players")
    if not isinstance(players, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get("PlayerId")
        s = None
        if isinstance(pid, str):
            s = pid
        else:
            try:
                s = json.dumps(pid)
            except Exception:
                s = None
        if not s:
            continue
        m = XUID_RE.search(s)
        if not m:
            continue
        xi = _coerce_int(m.group(1))
        if xi is None or xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _extract_gamertags_from_match_stats(match_stats_json: dict[str, Any]) -> dict[int, str]:
    """Extrait les paires XUID → Gamertag depuis un match.
    
    Returns:
        Dict {xuid: gamertag} pour tous les joueurs du match.
    """
    players = match_stats_json.get("Players")
    if not isinstance(players, list):
        return {}
    result: dict[int, str] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get("PlayerId")
        gamertag = p.get("PlayerGamertag") or p.get("Gamertag") or ""
        
        # Extraire le XUID
        s = pid if isinstance(pid, str) else None
        if not s:
            try:
                s = json.dumps(pid)
            except Exception:
                continue
        m = XUID_RE.search(s)
        if not m:
            continue
        xi = _coerce_int(m.group(1))
        if xi is None:
            continue
        
        # Nettoyer le gamertag
        gt = str(gamertag).strip() if gamertag else ""
        if gt and xi not in result:
            result[xi] = gt
    return result


def _chunked(lst: list, size: int):
    """Découpe une liste en chunks de taille fixe."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _get_iso_now() -> str:
    """Retourne le timestamp ISO 8601 actuel (UTC)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _update_sync_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    """Met à jour une entrée dans SyncMeta."""
    cur = con.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO SyncMeta (Key, Value, UpdatedAt)
           VALUES (?, ?, ?)""",
        (key, value, _get_iso_now()),
    )
    con.commit()


def _get_sync_meta(con: sqlite3.Connection, key: str) -> Optional[str]:
    """Récupère une valeur depuis SyncMeta."""
    try:
        cur = con.cursor()
        cur.execute("SELECT Value FROM SyncMeta WHERE Key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _create_schema(con: sqlite3.Connection) -> None:
    cur = con.cursor()

    # Schéma minimal mais compatible avec les loaders du projet.
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS MatchStats (
   ResponseBody TEXT,
   MatchId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.MatchId')) VIRTUAL,
   MatchInfo TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.MatchInfo')) VIRTUAL,
   Teams TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Teams')) VIRTUAL,
   Players TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Players')) VIRTUAL
)
"""  # noqa: E501
    )

    cur.execute(
        """
CREATE TABLE IF NOT EXISTS PlayerMatchStats (
   ResponseBody TEXT,
   MatchId TEXT,
   PlayerStats TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Value')) VIRTUAL
)
"""
    )

    # Highlight events (film) — optionnel.
    # Stocke le JSON brut renvoyé par SPNKr (spnkr.film.read_highlight_events).
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS HighlightEvents (
   MatchId TEXT NOT NULL,
   ResponseBody TEXT NOT NULL,
   EventType TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.event_type')) VIRTUAL,
   TimeMs INTEGER GENERATED ALWAYS AS (json_extract(ResponseBody, '$.time_ms')) VIRTUAL,
   Xuid TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.xuid')) VIRTUAL,
   Gamertag TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.gamertag')) VIRTUAL,
   TypeHint INTEGER GENERATED ALWAYS AS (json_extract(ResponseBody, '$.type_hint')) VIRTUAL
)
"""
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_HighlightEvents_MatchId ON HighlightEvents(MatchId)")

    # Index sur MatchStats pour accélérer les requêtes fréquentes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_MatchStats_MatchId ON MatchStats(MatchId)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_MatchStats_StartTime ON MatchStats(json_extract(ResponseBody, '$.MatchInfo.StartTime'))"
    )

    # Table XuidAliases — stocke les correspondances XUID → Gamertag
    # Remplace progressivement le fichier xuid_aliases.json
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS XuidAliases (
   Xuid TEXT PRIMARY KEY,
   Gamertag TEXT NOT NULL,
   LastSeen TEXT,
   Source TEXT,
   UpdatedAt TEXT
)
"""
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_XuidAliases_Gamertag ON XuidAliases(Gamertag COLLATE NOCASE)")

    # Table SyncMeta — métadonnées de synchronisation pour le delta intelligent
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS SyncMeta (
   Key TEXT PRIMARY KEY,
   Value TEXT,
   UpdatedAt TEXT
)
"""
    )

    # Assets — on garde le schéma OpenSpartan (virt cols) car c'est pratique et compatible.
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS Maps (
   ResponseBody TEXT,
   CustomData TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CustomData')) VIRTUAL,
   Tags TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Tags')) VIRTUAL,
   PrefabLinks TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PrefabLinks')) VIRTUAL,
   AssetId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetId')) VIRTUAL,
   VersionId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionId')) VIRTUAL,
   PublicName TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublicName')) VIRTUAL,
   Description TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Description')) VIRTUAL,
   Files TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Files')) VIRTUAL,
   Contributors TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Contributors')) VIRTUAL,
   AssetHome TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetHome')) VIRTUAL,
   AssetStats TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetStats')) VIRTUAL,
   InspectionResult TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.InspectionResult')) VIRTUAL,
   CloneBehavior TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CloneBehavior')) VIRTUAL,
   "Order" TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Order')) VIRTUAL,
   PublishedDate TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublishedDate')) VIRTUAL,
   VersionNumber TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionNumber')) VIRTUAL,
   Admin TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Admin')) VIRTUAL
)
"""  # noqa: E501
    )

    cur.execute(
        """
CREATE TABLE IF NOT EXISTS Playlists (
   ResponseBody TEXT,
   CustomData TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CustomData')) VIRTUAL,
   Tags TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Tags')) VIRTUAL,
   RotationEntries TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.RotationEntries')) VIRTUAL,
   AssetId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetId')) VIRTUAL,
   VersionId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionId')) VIRTUAL,
   PublicName TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublicName')) VIRTUAL,
   Description TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Description')) VIRTUAL,
   Files TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Files')) VIRTUAL,
   Contributors TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Contributors')) VIRTUAL,
   AssetHome TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetHome')) VIRTUAL,
   AssetStats TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetStats')) VIRTUAL,
   InspectionResult TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.InspectionResult')) VIRTUAL,
   CloneBehavior TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CloneBehavior')) VIRTUAL,
   "Order" TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Order')) VIRTUAL,
   PublishedDate TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublishedDate')) VIRTUAL,
   VersionNumber TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionNumber')) VIRTUAL,
   Admin TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Admin')) VIRTUAL
)
"""  # noqa: E501
    )

    cur.execute(
        """
CREATE TABLE IF NOT EXISTS PlaylistMapModePairs (
   ResponseBody TEXT,
   CustomData TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CustomData')) VIRTUAL,
   Tags TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Tags')) VIRTUAL,
   MapLink TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.MapLink')) VIRTUAL,
   UgcGameVariantLink TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.UgcGameVariantLink')) VIRTUAL,
   AssetId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetId')) VIRTUAL,
   VersionId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionId')) VIRTUAL,
   PublicName TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublicName')) VIRTUAL,
   Description TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Description')) VIRTUAL,
   Files TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Files')) VIRTUAL,
   Contributors TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Contributors')) VIRTUAL,
   AssetHome TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetHome')) VIRTUAL,
   AssetStats TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetStats')) VIRTUAL,
   InspectionResult TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.InspectionResult')) VIRTUAL,
   CloneBehavior TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CloneBehavior')) VIRTUAL,
   "Order" TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Order')) VIRTUAL,
   PublishedDate TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublishedDate')) VIRTUAL,
   VersionNumber TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionNumber')) VIRTUAL,
   Admin TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Admin')) VIRTUAL
)
"""  # noqa: E501
    )

    cur.execute(
        """
CREATE TABLE IF NOT EXISTS GameVariants (
   ResponseBody TEXT,
   CustomData TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CustomData')) VIRTUAL,
   Tags TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Tags')) VIRTUAL,
   EngineGameVariantLink TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.EngineGameVariantLink')) VIRTUAL,
   AssetId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetId')) VIRTUAL,
   VersionId TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionId')) VIRTUAL,
   PublicName TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublicName')) VIRTUAL,
   Description TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Description')) VIRTUAL,
   Files TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Files')) VIRTUAL,
   Contributors TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Contributors')) VIRTUAL,
   AssetHome TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetHome')) VIRTUAL,
   AssetStats TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.AssetStats')) VIRTUAL,
   InspectionResult TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.InspectionResult')) VIRTUAL,
   CloneBehavior TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.CloneBehavior')) VIRTUAL,
   "Order" TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Order')) VIRTUAL,
   PublishedDate TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.PublishedDate')) VIRTUAL,
   VersionNumber TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.VersionNumber')) VIRTUAL,
   Admin TEXT GENERATED ALWAYS AS (json_extract(ResponseBody, '$.Admin')) VIRTUAL
)
"""  # noqa: E501
    )

    con.commit()


@dataclass(frozen=True)
class Tokens:
    spartan_token: str
    clearance_token: str


async def _get_tokens_from_env_or_args(args: argparse.Namespace) -> Tokens:
    # Lazy imports (pour ne pas forcer spnkr si on ne lance pas ce script)
    azure_client_id = args.azure_client_id or os.environ.get("SPNKR_AZURE_CLIENT_ID")
    azure_client_secret = args.azure_client_secret or os.environ.get("SPNKR_AZURE_CLIENT_SECRET")
    azure_redirect_uri = args.azure_redirect_uri or os.environ.get("SPNKR_AZURE_REDIRECT_URI") or "https://localhost"
    oauth_refresh_token = args.oauth_refresh_token or os.environ.get("SPNKR_OAUTH_REFRESH_TOKEN")

    if azure_client_id and azure_client_secret and oauth_refresh_token:
        try:
            from aiohttp import ClientSession
            from spnkr import AzureApp, refresh_player_tokens
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Dépendance manquante pour l'import SPNKr. Installe d'abord SPNKr, ex:\n"
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
        from aiohttp import ClientTimeout

        async with ClientSession(timeout=ClientTimeout(total=45)) as session:
            try:
                player = await refresh_player_tokens(session, app, oauth_refresh_token)
                return Tokens(
                    spartan_token=str(player.spartan_token.token),
                    clearance_token=str(player.clearance_token.token),
                )
            except Exception as e:
                # spnkr.errors.OAuth2Error n'expose pas la réponse brute dans cette version;
                # on détecte donc le cas invalid_client via le message.
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

    spartan = _normalize_token_value(args.spartan_token or os.environ.get("SPNKR_SPARTAN_TOKEN"))
    clearance = _normalize_token_value(args.clearance_token or os.environ.get("SPNKR_CLEARANCE_TOKEN"))
    if not spartan or not clearance:
        raise SystemExit(
            "Tokens manquants. Fournis soit:\n"
            "- SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN (env),\n"
            "- ou --spartan-token/--clearance-token,\n"
            "- ou l'option Azure: --azure-client-id/--azure-client-secret/--oauth-refresh-token."
        )
    return Tokens(spartan_token=str(spartan), clearance_token=str(clearance))


async def _request_with_retries(coro_factory, *, tries: int = 4, base_sleep: float = 0.8):
    last_err: Exception | None = None
    for i in range(tries):
        try:
            return await coro_factory()
        except Exception as e:  # réseau / 429 / parse
            # Auth invalide/expirée: inutile de retry.
            try:
                from aiohttp.client_exceptions import ClientResponseError

                if isinstance(e, ClientResponseError) and e.status in (401, 403):
                    raise SystemExit(
                        "Requête non autorisée (401/403). Les tokens sont probablement invalides/expirés.\n"
                        "Vérifie dans `.env.local` (ou `.env`) que: \n"
                        "- SPNKR_SPARTAN_TOKEN est la VALEUR du header spartan (souvent commence par `v4=`), sur une seule ligne\n"
                        "- SPNKR_CLEARANCE_TOKEN est la VALEUR du header clearance (souvent commence par `eyJ...`), sur une seule ligne\n"
                        "Astuce: tu peux coller la ligne entière `x-343-authorization-spartan: ...` / `343-clearance: ...`, le script enlève automatiquement le préfixe avant `:`."
                    ) from e

                # Assets supprimés / non publics: inutile de retry.
                if isinstance(e, ClientResponseError) and e.status in (400, 404, 410):
                    raise
            except ModuleNotFoundError:
                # aiohttp n'est pas importable => on laisse le retry standard.
                pass
            last_err = e
            await asyncio.sleep(base_sleep * (2**i))
    assert last_err is not None
    raise last_err


def _asset_ref(match_info: dict[str, Any], key: str) -> tuple[Optional[str], Optional[str]]:
    obj = match_info.get(key)
    if not isinstance(obj, dict):
        return None, None
    aid = obj.get("AssetId")
    vid = obj.get("VersionId")
    return (aid if isinstance(aid, str) else None, vid if isinstance(vid, str) else None)


def _seed_asset_seen_from_db(con: sqlite3.Connection, tables: list[str]) -> set[tuple[str, str, str]]:
    """Retourne les (table, AssetId, VersionId) déjà présents en DB.

    Permet d'éviter de re-fetch les mêmes assets à chaque refresh.
    """
    cur = con.cursor()
    seen: set[tuple[str, str, str]] = set()
    for table in tables:
        try:
            cur.execute(
                f"SELECT AssetId, VersionId FROM {table} WHERE AssetId IS NOT NULL AND VersionId IS NOT NULL"
            )
        except Exception:
            continue
        for aid, vid in cur.fetchall():
            if isinstance(aid, str) and isinstance(vid, str) and aid and vid:
                seen.add((table, aid, vid))
    return seen


def _load_match_json_from_db(con: sqlite3.Connection, match_id: str) -> Optional[dict[str, Any]]:
    try:
        cur = con.cursor()
        cur.execute("SELECT ResponseBody FROM MatchStats WHERE MatchId = ? LIMIT 1", (match_id,))
        row = cur.fetchone()
        if not row or not isinstance(row[0], str):
            return None
        obj = json.loads(row[0])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


async def _import_assets_for_match_info(
    mi: dict[str, Any],
    *,
    client: Any,
    con: sqlite3.Connection,
    asset_seen: set[tuple[str, str, str]],
    asset_missing: set[tuple[str, str, str]],
) -> None:
    """Importe Maps/Playlists/PlaylistMapModePairs/GameVariants référencés par MatchInfo."""
    cur = con.cursor()

    def _is_missing(exc: Exception) -> bool:
        try:
            from aiohttp.client_exceptions import ClientResponseError

            return isinstance(exc, ClientResponseError) and exc.status in (400, 404, 410)
        except ModuleNotFoundError:
            return False

    # MapVariant (map)
    map_aid, map_vid = _asset_ref(mi, "MapVariant")
    if map_aid and map_vid and ("Maps", map_aid, map_vid) not in asset_seen and ("Maps", map_aid, map_vid) not in asset_missing:
        async def _get_map():
            resp = await client.discovery_ugc.get_map(map_aid, map_vid)
            return await resp.json()

        try:
            obj = await _request_with_retries(_get_map)
            if isinstance(obj, dict):
                cur.execute("INSERT INTO Maps(ResponseBody) VALUES (?)", (json.dumps(obj, ensure_ascii=False),))
                asset_seen.add(("Maps", map_aid, map_vid))
        except Exception as e:
            if _is_missing(e):
                asset_missing.add(("Maps", map_aid, map_vid))
            pass

    # Playlist
    pl_aid, pl_vid = _asset_ref(mi, "Playlist")
    if pl_aid and pl_vid and ("Playlists", pl_aid, pl_vid) not in asset_seen and ("Playlists", pl_aid, pl_vid) not in asset_missing:
        async def _get_playlist():
            resp = await client.discovery_ugc.get_playlist(pl_aid, pl_vid)
            return await resp.json()

        try:
            obj = await _request_with_retries(_get_playlist)
            if isinstance(obj, dict):
                cur.execute("INSERT INTO Playlists(ResponseBody) VALUES (?)", (json.dumps(obj, ensure_ascii=False),))
                asset_seen.add(("Playlists", pl_aid, pl_vid))
        except Exception as e:
            if _is_missing(e):
                asset_missing.add(("Playlists", pl_aid, pl_vid))
            pass

    # Map-mode pair
    mp_aid, mp_vid = _asset_ref(mi, "PlaylistMapModePair")
    if mp_aid and mp_vid and ("PlaylistMapModePairs", mp_aid, mp_vid) not in asset_seen and ("PlaylistMapModePairs", mp_aid, mp_vid) not in asset_missing:
        async def _get_pair():
            resp = await client.discovery_ugc.get_map_mode_pair(mp_aid, mp_vid)
            return await resp.json()

        try:
            obj = await _request_with_retries(_get_pair)
            if isinstance(obj, dict):
                cur.execute(
                    "INSERT INTO PlaylistMapModePairs(ResponseBody) VALUES (?)",
                    (json.dumps(obj, ensure_ascii=False),),
                )
                asset_seen.add(("PlaylistMapModePairs", mp_aid, mp_vid))
        except Exception as e:
            if _is_missing(e):
                asset_missing.add(("PlaylistMapModePairs", mp_aid, mp_vid))
            pass

    # UGC GameVariant
    gv_aid, gv_vid = _asset_ref(mi, "UgcGameVariant")
    if gv_aid and gv_vid and ("GameVariants", gv_aid, gv_vid) not in asset_seen and ("GameVariants", gv_aid, gv_vid) not in asset_missing:
        async def _get_gv():
            resp = await client.discovery_ugc.get_ugc_game_variant(gv_aid, gv_vid)
            return await resp.json()

        try:
            obj = await _request_with_retries(_get_gv)
            if isinstance(obj, dict):
                cur.execute("INSERT INTO GameVariants(ResponseBody) VALUES (?)", (json.dumps(obj, ensure_ascii=False),))
                asset_seen.add(("GameVariants", gv_aid, gv_vid))
        except Exception as e:
            if _is_missing(e):
                asset_missing.add(("GameVariants", gv_aid, gv_vid))
            pass

    con.commit()


def _save_aliases_from_match(
    con: sqlite3.Connection,
    match_json: dict[str, Any],
    *,
    source: str = "match_roster",
) -> int:
    """Sauvegarde les aliases extraits d'un match dans XuidAliases.
    
    Ne met à jour que si le gamertag est nouveau ou différent.
    
    Returns:
        Nombre d'aliases ajoutés/mis à jour.
    """
    gamertags = _extract_gamertags_from_match_stats(match_json)
    if not gamertags:
        return 0
    
    cur = con.cursor()
    now = _get_iso_now()
    updated = 0
    
    for xuid, gamertag in gamertags.items():
        xuid_str = str(xuid)
        # INSERT OR REPLACE met à jour LastSeen à chaque fois
        cur.execute(
            """INSERT INTO XuidAliases (Xuid, Gamertag, LastSeen, Source, UpdatedAt)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(Xuid) DO UPDATE SET
                   Gamertag = CASE 
                       WHEN excluded.Gamertag != '' AND excluded.Gamertag != XuidAliases.Gamertag 
                       THEN excluded.Gamertag 
                       ELSE XuidAliases.Gamertag 
                   END,
                   LastSeen = excluded.LastSeen,
                   UpdatedAt = CASE 
                       WHEN excluded.Gamertag != '' AND excluded.Gamertag != XuidAliases.Gamertag 
                       THEN excluded.UpdatedAt 
                       ELSE XuidAliases.UpdatedAt 
                   END""",
            (xuid_str, gamertag, now, source, now),
        )
        if cur.rowcount > 0:
            updated += 1
    
    return updated


async def _refresh_aliases_via_api(
    client: Any,
    con: sqlite3.Connection,
    xuids: set[int],
    *,
    batch_size: int = 100,
) -> int:
    """Récupère les gamertags via l'API pour les XUIDs non résolus.
    
    Args:
        client: HaloInfiniteClient SPNKr
        con: Connexion SQLite
        xuids: Set de XUIDs à résoudre
        batch_size: Taille des batchs API (max 100)
    
    Returns:
        Nombre d'aliases ajoutés/mis à jour.
    """
    if not xuids:
        return 0
    
    # Filtrer les XUIDs déjà connus avec un gamertag valide
    cur = con.cursor()
    try:
        cur.execute("SELECT Xuid FROM XuidAliases WHERE Gamertag IS NOT NULL AND Gamertag != ''")
        known = {row[0] for row in cur.fetchall()}
    except Exception:
        known = set()
    
    to_fetch = {x for x in xuids if str(x) not in known}
    if not to_fetch:
        return 0
    
    updated = 0
    now = _get_iso_now()
    
    for batch in _chunked(list(to_fetch), batch_size):
        try:
            # SPNKr: récupérer les profils par batch de XUIDs
            # Note: l'API exacte dépend de la version de SPNKr
            # On essaie d'abord get_users_by_id, sinon on fait du best-effort
            try:
                resp = await _request_with_retries(
                    lambda b=batch: client.profile.get_users_by_id(b)
                )
                if hasattr(resp, "parse"):
                    profiles = await resp.parse()
                else:
                    profiles = resp
                    
                for p in profiles:
                    xuid = getattr(p, "xuid", None) or getattr(p, "id", None)
                    gamertag = getattr(p, "gamertag", None) or getattr(p, "gamer_tag", None)
                    if xuid and gamertag:
                        cur.execute(
                            """INSERT OR REPLACE INTO XuidAliases 
                               (Xuid, Gamertag, LastSeen, Source, UpdatedAt)
                               VALUES (?, ?, ?, 'api', ?)""",
                            (str(xuid), str(gamertag), now, now),
                        )
                        updated += 1
            except AttributeError:
                # profile.get_users_by_id n'existe pas dans cette version de SPNKr
                # On skippe silencieusement car les aliases seront extraits des matchs
                pass
        except Exception:
            # Non bloquant: on continue avec les autres batches
            pass
    
    con.commit()
    return updated


async def main_async(argv: list[str]) -> int:
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description="Import SPNKr -> DB compatible OpenSpartan")
    ap.add_argument("--out-db", required=True, help="Chemin de sortie du .db (SQLite)")
    ap.add_argument("--player", required=True, help="XUID (digits) ou gamertag")
    ap.add_argument("--match-type", default="matchmaking", choices=["all", "matchmaking", "custom", "local"], help="Type de matchs")
    ap.add_argument("--max-matches", type=int, default=200, help="Nombre max de matchs à importer")
    ap.add_argument("--start", type=int, default=0, help="Offset dans l'historique (0 = plus récent)")
    ap.add_argument("--requests-per-second", type=int, default=5, help="Rate limit SPNKr (par service)")
    ap.add_argument("--resume", action="store_true", help="Reprend sur une DB existante (sans effacer)")

    ap.add_argument("--no-skill", action="store_true", help="N'importe pas PlayerMatchStats (skill)")
    ap.add_argument("--no-assets", action="store_true", help="N'importe pas les assets UGC (Maps/Playlists/etc.)")

    # Highlight events: activés par défaut, désactivables via --no-highlight-events
    ap.add_argument(
        "--no-highlight-events",
        action="store_true",
        help="Désactive l'import des highlight events (film) — accélère l'import.",
    )
    # Compat legacy: --with-highlight-events est maintenant ignoré (comportement par défaut)
    ap.add_argument(
        "--with-highlight-events",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden, kept for backward compat
    )

    # Aliases: activés par défaut, désactivables via --no-aliases
    ap.add_argument(
        "--no-aliases",
        action="store_true",
        help="Désactive le refresh des aliases (XUID → Gamertag) pour les nouveaux joueurs.",
    )

    # Tokens manuels
    ap.add_argument("--spartan-token", default=None)
    ap.add_argument("--clearance-token", default=None)

    # Azure app (optionnel)
    ap.add_argument("--azure-client-id", default=None)
    ap.add_argument("--azure-client-secret", default=None)
    ap.add_argument("--azure-redirect-uri", default="https://localhost")
    ap.add_argument("--oauth-refresh-token", default=None)

    args = ap.parse_args(argv)

    # Highlight events: ON par défaut, OFF si --no-highlight-events
    fetch_highlight_events = not args.no_highlight_events
    # Aliases: ON par défaut, OFF si --no-aliases
    fetch_aliases = not args.no_aliases

    out_db = str(args.out_db)
    _ensure_parent_dir(out_db)

    if os.path.exists(out_db) and not args.resume:
        os.remove(out_db)

    con = sqlite3.connect(out_db)
    try:
        _create_schema(con)

        existing_match_ids: set[str] = set()
        if args.resume:
            try:
                cur = con.cursor()
                cur.execute("SELECT MatchId FROM MatchStats WHERE MatchId IS NOT NULL")
                existing_match_ids = {str(r[0]) for r in cur.fetchall() if r and r[0]}
            except Exception:
                existing_match_ids = set()

        tokens = await _get_tokens_from_env_or_args(args)

        try:
            from aiohttp import ClientSession
            from spnkr import HaloInfiniteClient
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Dépendance manquante pour l'import SPNKr. Installe d'abord SPNKr, ex:\n"
                "pip install 'spnkr @ git+https://github.com/acurtis166/SPNKr.git'\n"
                "(et ses dépendances, dont aiohttp)."
            ) from e

        from aiohttp import ClientTimeout

        async with ClientSession(timeout=ClientTimeout(total=45)) as session:
            client = HaloInfiniteClient(
                session=session,
                spartan_token=tokens.spartan_token,
                clearance_token=tokens.clearance_token,
                requests_per_second=int(args.requests_per_second),
            )

            film_mod = None
            if fetch_highlight_events:
                try:
                    from spnkr import film as film_mod  # type: ignore
                except Exception:
                    film_mod = None

            player = args.player
            # `--player` peut être "123..." ou gamertag. SPNKr accepte les deux.

            inserted = 0
            aliases_updated = 0
            all_xuids_seen: set[int] = set()  # Collecte tous les XUIDs rencontrés
            asset_tables = ["Maps", "Playlists", "PlaylistMapModePairs", "GameVariants"]
            asset_seen: set[tuple[str, str, str]] = _seed_asset_seen_from_db(con, asset_tables)
            asset_missing: set[tuple[str, str, str]] = set()

            start = int(args.start)
            remaining = int(args.max_matches)

            while remaining > 0:
                batch = min(25, remaining)

                async def _get_hist():
                    resp = await client.stats.get_match_history(player, start=start, count=batch, match_type=args.match_type)
                    return await resp.parse()

                history = await _request_with_retries(_get_hist)
                if not getattr(history, "results", None):
                    break

                match_ids = [str(r.match_id) for r in history.results]

                for mid in match_ids:
                    if remaining <= 0:
                        break
                    if mid in existing_match_ids:
                        # Backfill: PlayerMatchStats (skill) manquant sur une DB existante.
                        # Cas typique: première import avec --no-skill, puis refresh sans --no-skill.
                        if not args.no_skill:
                            try:
                                cur = con.cursor()
                                cur.execute(
                                    "SELECT 1 FROM PlayerMatchStats WHERE MatchId = ? LIMIT 1",
                                    (mid,),
                                )
                                has_skill = cur.fetchone() is not None
                            except Exception:
                                has_skill = False

                            if not has_skill:
                                try:
                                    existing = _load_match_json_from_db(con, mid)
                                    xuids = _extract_xuids_from_match_stats(existing)
                                    if not xuids:
                                        # fallback: si player est un xuid
                                        m = XUID_RE.search(str(player))
                                        if m:
                                            xi = _coerce_int(m.group(1))
                                            if xi is not None:
                                                xuids = [xi]

                                    if xuids:
                                        async def _get_skill_existing():
                                            resp = await client.skill.get_match_skill(mid, xuids)
                                            return await resp.json()

                                        skill_json = await _request_with_retries(_get_skill_existing)
                                        if isinstance(skill_json, dict):
                                            cur.execute(
                                                "INSERT INTO PlayerMatchStats(ResponseBody, MatchId) VALUES (?, ?)",
                                                (json.dumps(skill_json, ensure_ascii=False), mid),
                                            )
                                            con.commit()
                                except Exception:
                                    # Non bloquant: certains matchs n'ont pas de skill data.
                                    pass

                        # Match déjà en DB: on peut quand même backfill les assets si demandé.
                        if not args.no_assets:
                            existing = _load_match_json_from_db(con, mid)
                            mi = existing.get("MatchInfo") if isinstance(existing, dict) else None
                            if isinstance(mi, dict):
                                await _import_assets_for_match_info(
                                    mi,
                                    client=client,
                                    con=con,
                                    asset_seen=asset_seen,
                                    asset_missing=asset_missing,
                                )

                        # Backfill highlight events si demandé et si absent.
                        if fetch_highlight_events and film_mod is not None:
                            try:
                                cur = con.cursor()
                                cur.execute(
                                    "SELECT 1 FROM HighlightEvents WHERE MatchId = ? LIMIT 1",
                                    (mid,),
                                )
                                has_events = cur.fetchone() is not None
                            except Exception:
                                has_events = False

                            if not has_events:
                                try:
                                    async def _get_events_existing():
                                        return await film_mod.read_highlight_events(client, match_id=mid)

                                    events = await _request_with_retries(_get_events_existing)
                                    if events:
                                        cur = con.cursor()

                                        def _event_to_dict(e: Any) -> dict[str, Any]:
                                            if isinstance(e, dict):
                                                return e
                                            if hasattr(e, "model_dump"):
                                                return e.model_dump()  # type: ignore[no-any-return]
                                            if hasattr(e, "dict"):
                                                return e.dict()  # type: ignore[no-any-return]
                                            if hasattr(e, "_asdict"):
                                                return e._asdict()  # type: ignore[no-any-return]
                                            return {"raw": str(e)}

                                        for e in events:
                                            cur.execute(
                                                "INSERT INTO HighlightEvents(MatchId, ResponseBody) VALUES (?, ?)",
                                                (mid, json.dumps(_event_to_dict(e), ensure_ascii=False)),
                                            )
                                        con.commit()
                                except Exception:
                                    pass
                        start += 1
                        remaining -= 1
                        continue

                    # --- MatchStats ---
                    async def _get_match():
                        resp = await client.stats.get_match_stats(mid)
                        return await resp.json()

                    match_json = await _request_with_retries(_get_match)
                    if not isinstance(match_json, dict):
                        start += 1
                        remaining -= 1
                        continue

                    cur = con.cursor()
                    cur.execute("INSERT INTO MatchStats(ResponseBody) VALUES (?)", (json.dumps(match_json, ensure_ascii=False),))

                    # --- PlayerMatchStats (skill) ---
                    # On essaie de récupérer les xuids du match pour maximiser la qualité des TeamMmrs.
                    xuids = _extract_xuids_from_match_stats(match_json)
                    if not xuids:
                        # fallback: si player est un xuid
                        m = XUID_RE.search(str(player))
                        if m:
                            xi = _coerce_int(m.group(1))
                            if xi is not None:
                                xuids = [xi]

                    if (not args.no_skill) and xuids:
                        async def _get_skill():
                            resp = await client.skill.get_match_skill(mid, xuids)
                            return await resp.json()

                        try:
                            skill_json = await _request_with_retries(_get_skill)
                            if isinstance(skill_json, dict):
                                cur.execute(
                                    "INSERT INTO PlayerMatchStats(ResponseBody, MatchId) VALUES (?, ?)",
                                    (json.dumps(skill_json, ensure_ascii=False), mid),
                                )
                        except Exception:
                            # Non bloquant (certains matchs n'ont pas de skill data)
                            pass

                    # Commit au plus tôt pour éviter de perdre le match si un fetch d'asset hang.
                    con.commit()

                    # --- Sauvegarder les aliases depuis le match ---
                    if fetch_aliases:
                        aliases_updated += _save_aliases_from_match(con, match_json)
                        # Collecter les XUIDs pour un éventuel refresh API
                        all_xuids_seen.update(_extract_xuids_from_match_stats(match_json))

                    # --- Highlight events (par défaut) ---
                    if fetch_highlight_events and film_mod is not None:
                        try:
                            async def _get_events():
                                return await film_mod.read_highlight_events(client, match_id=mid)

                            events = await _request_with_retries(_get_events)
                            if events:
                                cur = con.cursor()

                                def _event_to_dict(e: Any) -> dict[str, Any]:
                                    if isinstance(e, dict):
                                        return e
                                    if hasattr(e, "model_dump"):
                                        return e.model_dump()  # type: ignore[no-any-return]
                                    if hasattr(e, "dict"):
                                        return e.dict()  # type: ignore[no-any-return]
                                    if hasattr(e, "_asdict"):
                                        return e._asdict()  # type: ignore[no-any-return]
                                    return {"raw": str(e)}

                                for e in events:
                                    cur.execute(
                                        "INSERT INTO HighlightEvents(MatchId, ResponseBody) VALUES (?, ?)",
                                        (mid, json.dumps(_event_to_dict(e), ensure_ascii=False)),
                                    )
                                con.commit()
                        except Exception:
                            # Non bloquant: certains matchs/chunks peuvent être manquants.
                            pass

                    # --- Assets (optionnel mais utile pour les noms) ---
                    mi = match_json.get("MatchInfo")
                    if (not args.no_assets) and isinstance(mi, dict):
                        await _import_assets_for_match_info(
                            mi,
                            client=client,
                            con=con,
                            asset_seen=asset_seen,
                            asset_missing=asset_missing,
                        )

                    inserted += 1
                    existing_match_ids.add(mid)
                    remaining -= 1
                    start += 1

                    if inserted % 10 == 0:
                        print(f"Imported {inserted} matches... ({aliases_updated} aliases)")

            # --- Refresh aliases via API pour les XUIDs sans gamertag ---
            if fetch_aliases and all_xuids_seen:
                try:
                    api_aliases = await _refresh_aliases_via_api(client, con, all_xuids_seen)
                    aliases_updated += api_aliases
                except Exception:
                    pass  # Non bloquant

            # --- Mise à jour des métadonnées de sync ---
            _update_sync_meta(con, "last_sync_at", _get_iso_now())
            _update_sync_meta(con, "total_matches", str(len(existing_match_ids)))
            
            # Récupérer le dernier match importé
            try:
                cur = con.cursor()
                cur.execute(
                    """SELECT MatchId, json_extract(ResponseBody, '$.MatchInfo.StartTime')
                       FROM MatchStats 
                       ORDER BY json_extract(ResponseBody, '$.MatchInfo.StartTime') DESC 
                       LIMIT 1"""
                )
                row = cur.fetchone()
                if row:
                    _update_sync_meta(con, "last_match_id", str(row[0]))
                    if row[1]:
                        _update_sync_meta(con, "last_match_time", str(row[1]))
            except Exception:
                pass

            # Stocker le XUID du joueur principal si c'est un XUID
            m = XUID_RE.search(str(player))
            if m:
                _update_sync_meta(con, "player_xuid", m.group(1))

            print(f"✅ OK: {out_db}")
            print(f"   • {inserted} nouveaux matchs importés")
            print(f"   • {aliases_updated} aliases mis à jour")
            print(f"   • {len(existing_match_ids)} matchs au total")
            return 0
    finally:
        con.close()


def main() -> int:
    return asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
