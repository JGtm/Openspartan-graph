"""Helpers pour gérer des assets joueur (bannière/emblème) en cache local.

Contraintes:
- Aucun accès réseau implicite: le téléchargement doit être déclenché explicitement par l'utilisateur.
- Le rendu doit fonctionner offline si les fichiers sont déjà présents en cache ou fournis en chemin local.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional


def get_player_assets_cache_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "data" / "cache" / "player_assets")


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _url_ext(url: str) -> str:
    try:
        p = urllib.parse.urlparse(str(url))
        ext = Path(p.path).suffix.lower()
    except Exception:
        ext = ""
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return ext
    return ".bin"


def _hashed_name(url: str, *, prefix: str) -> str:
    h = hashlib.sha256(str(url).encode("utf-8", errors="ignore")).hexdigest()[:20]
    return f"{prefix}_{h}{_url_ext(url)}"


def is_http_url(value: str | None) -> bool:
    s = str(value or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def resolve_local_image_path(value: str | None) -> str | None:
    """Retourne un chemin image local si possible.

    - si `value` est un chemin existant -> retourne ce chemin
    - si `value` est une URL -> retourne le fichier cache s'il existe déjà
    """
    s = str(value or "").strip()
    if not s:
        return None

    # Chemin local direct
    if os.path.exists(s) and os.path.isfile(s):
        return s

    # URL => on regarde si déjà en cache
    if is_http_url(s):
        cache_dir = get_player_assets_cache_dir()
        # Compat: selon les usages, les téléchargements ont pu être faits
        # avec différents préfixes (banner/emblem/backdrop/nameplate).
        # On tente plusieurs noms déterministes.
        for prefix in ("asset", "banner", "emblem", "backdrop", "nameplate"):
            fname = _hashed_name(s, prefix=prefix)
            cached = os.path.join(cache_dir, fname)
            if os.path.exists(cached) and os.path.isfile(cached):
                return cached

    return None


def download_image_to_cache(url: str, *, prefix: str, timeout_seconds: int = 12) -> tuple[bool, str, str | None]:
    """Télécharge une image depuis une URL dans le cache local.

    Retourne (ok, message, local_path).
    """
    u = str(url or "").strip()
    if not is_http_url(u):
        return False, "URL invalide (http/https attendu).", None

    cache_dir = get_player_assets_cache_dir()
    _safe_mkdir(cache_dir)

    fname = _hashed_name(u, prefix=prefix)
    out_path = os.path.join(cache_dir, fname)

    def _auth_headers_for_url(target_url: str) -> dict[str, str]:
        headers: dict[str, str] = {"User-Agent": "OpenSpartan-Graphs"}

        # Certains endpoints Halo Waypoint (hi/images/file/...) exigent une auth (343-clearance).
        clearance = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()
        spartan = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip()
        if clearance:
            # Les deux existent dans la nature; on fournit les deux.
            headers.setdefault("Cookie", f"343-clearance={clearance}")
            headers.setdefault("343-clearance", clearance)
        if spartan:
            headers.setdefault("x-343-authorization-spartan", spartan)
        return headers

    def _run_sync(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            msg = str(e)
            if "asyncio.run() cannot be called" not in msg:
                raise
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(coro))
                return fut.result(timeout=float(timeout_seconds) + 20.0)

    def _try_spnkr_fetch_bytes(target: str) -> tuple[bytes | None, str | None]:
        """Best-effort: télécharge via SPNKr (auth) pour contourner 401/403.

        `target` peut être:
        - une URL /hi/images/file/<relative>
        - un chemin relatif Inventory/... (ou /Inventory/...)
        """

        st = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip()
        ct = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()
        if not (st and ct):
            return None, None

        rel = str(target or "").strip()
        if not rel:
            return None, None

        # URL -> relative path
        try:
            if rel.startswith("http://") or rel.startswith("https://"):
                p = urllib.parse.urlparse(rel)
                path_lower = (p.path or "").lower()
                marker = "/hi/images/file/"
                if marker in path_lower:
                    rel = (p.path or "")[path_lower.index(marker) + len(marker) :]
        except Exception:
            pass

        rel = rel.lstrip("/")
        if not rel:
            return None, None

        async def _run() -> bytes:
            import aiohttp
            from spnkr.client import HaloInfiniteClient

            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                client = HaloInfiniteClient(
                    session,
                    spartan_token=st,
                    clearance_token=ct,
                    requests_per_second=3,
                )
                img_resp = await client.gamecms_hacs.get_image(rel)
                return await img_resp.read()

        try:
            data = _run_sync(_run())
            return (data if isinstance(data, (bytes, bytearray)) else None), None
        except Exception as e:
            return None, f"SPNKr get_image KO: {e}"

    def _extract_image_url_from_json(obj: object) -> str | None:
        candidates: list[str] = []

        def walk(x: object) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
                return
            if isinstance(x, list):
                for v in x[:200]:
                    walk(v)
                return
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return
                candidates.append(s)

        walk(obj)

        image_exts = (".png", ".jpg", ".jpeg", ".webp")
        # Priorité: URL explicite d'image
        for s in candidates:
            if (s.startswith("http://") or s.startswith("https://")) and s.lower().endswith(image_exts):
                return s

        # Sinon: chemin vers /hi/Waypoint/file/images/ (souvent public)
        host = "https://gamecms-hacs.svc.halowaypoint.com"
        for s in candidates:
            if "/hi/Waypoint/file/images/" in s and s.lower().endswith(image_exts):
                rel = s[s.index("/hi/Waypoint/file/images/") :]
                return f"{host}{rel}"

        # Sinon: chemin vers /hi/images/file/
        for s in candidates:
            s_lower = s.lower()
            marker = "/hi/images/file/"
            if marker in s_lower and s_lower.endswith(image_exts):
                rel = s[s_lower.index(marker) :]
                rel = rel.replace("/hi/images/file/", "/hi/Images/file/")
                return f"{host}{rel}"

        # Dernier recours: chemin relatif "Inventory/...png"
        for s in candidates:
            if s.lower().startswith("inventory/") and s.lower().endswith(image_exts):
                return f"{host}/hi/Images/file/{s.lstrip('/')}"

        return None

    def _download_once(target_url: str) -> tuple[bytes | None, str | None, str | None]:
        try:
            data: bytes | None = None
            content_type = ""

            # Certains endpoints renvoient 401/403 en accès direct; on tente SPNKr en fallback.
            if (
                ("/hi/images/file/" in str(target_url))
                or str(target_url).strip().lower().startswith("inventory/")
                or str(target_url).strip().startswith("/Inventory/")
            ):
                data_spnkr, spnkr_err = _try_spnkr_fetch_bytes(target_url)
                if data_spnkr:
                    data = bytes(data_spnkr)
                else:
                    # Fallback urllib (ex: si SPNKr non installé / tokens absents)
                    req = urllib.request.Request(target_url, headers=_auth_headers_for_url(target_url))
                    with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
                        data = resp.read()
                        content_type = str(resp.headers.get("content-type") or "").lower()
            else:
                # Cas standard (ex: Waypoint/file/images/*.png)
                req = urllib.request.Request(target_url, headers=_auth_headers_for_url(target_url))
                with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
                    data = resp.read()
                    content_type = str(resp.headers.get("content-type") or "").lower()

            if not data:
                return None, None, "Téléchargement vide."
            # On ne se fie pas seulement au content-type (parfois absent)
            if "application/json" in content_type or data.lstrip()[:1] in (b"{", b"["):
                try:
                    obj = json.loads(data.decode("utf-8", errors="ignore"))
                except Exception:
                    return None, None, "Réponse JSON illisible."
                img = _extract_image_url_from_json(obj)
                if img:
                    # Signale au caller qu'il faut re-télécharger cette URL.
                    return None, img, None
                return None, None, "Manifeste JSON sans URL d'image exploitable."
            return data, None, None
        except Exception as e:
            return None, None, f"Téléchargement impossible: {e}"

    # Téléchargement (avec auth si dispo) + suivi de manifestes JSON (max N sauts)
    current = u
    visited: set[str] = set()
    data: bytes | None = None
    last_err: str | None = None

    for _ in range(5):
        if current in visited:
            return False, "Boucle de redirection inattendue.", None
        visited.add(current)

        data, next_url, err = _download_once(current)
        last_err = err
        if next_url:
            current = next_url
            continue
        if data is not None:
            break
        # Erreur terminale sans redirection: inutile de boucler.
        break

    if data is None:
        return False, last_err or "Téléchargement impossible.", None

    # Si on a suivi une redirection, on cache avec l'URL finale (extension + hash)
    if current != u:
        u = current
        fname = _hashed_name(u, prefix=prefix)
        out_path = os.path.join(cache_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(data)
        return True, "", out_path
    except Exception as e:
        return False, f"Écriture cache impossible: {e}", None


def ensure_local_image_path(
    value: str | None,
    *,
    prefix: str,
    download_enabled: bool,
    auto_refresh_hours: int = 0,
    timeout_seconds: int = 12,
) -> str | None:
    """Assure qu'une image est disponible localement et retourne son chemin.

    - Si `value` est un chemin local existant => retourne ce chemin.
    - Si `value` est une URL:
      - si un fichier cache existe déjà et est "récent" => retourne ce cache
      - si `download_enabled` => télécharge automatiquement (et remplace le cache)
    - `auto_refresh_hours`:
      - 0 => ne force jamais de re-téléchargement si le cache existe
      - >0 => re-télécharge si le fichier a plus de N heures

    Cette fonction respecte la contrainte "pas d'accès réseau implicite" en
    ne téléchargeant que si `download_enabled=True`.
    """

    s = str(value or "").strip()
    if not s:
        return None

    # Chemin local direct
    if os.path.exists(s) and os.path.isfile(s):
        return s

    # URL: on calcule le chemin cache déterministe
    if not is_http_url(s):
        return None

    cache_dir = get_player_assets_cache_dir()
    fname = _hashed_name(s, prefix=prefix)
    cached = os.path.join(cache_dir, fname)

    if os.path.exists(cached) and os.path.isfile(cached):
        if int(auto_refresh_hours) <= 0:
            return cached
        try:
            age_s = time.time() - float(os.path.getmtime(cached))
            if age_s < float(auto_refresh_hours) * 3600.0:
                return cached
        except Exception:
            return cached

    if not download_enabled:
        # Pas de réseau: on renvoie le cache si trouvé (y compris anciens préfixes)
        return resolve_local_image_path(s)

    ok, _err, out_path = download_image_to_cache(s, prefix=prefix, timeout_seconds=timeout_seconds)
    if ok and out_path:
        return out_path

    # Fallback: si un cache existe malgré tout, ou un cache ancien d'un autre préfixe
    if os.path.exists(cached) and os.path.isfile(cached):
        return cached
    return resolve_local_image_path(s)


def file_to_data_url(path: str | None, *, max_bytes: int = 3 * 1024 * 1024) -> Optional[str]:
    """Encode un fichier image local en data URL pour intégration HTML.

    Limite la taille pour éviter des payloads énormes.
    """
    p = str(path or "").strip()
    if not p or not os.path.exists(p) or not os.path.isfile(p):
        return None

    try:
        size = os.path.getsize(p)
        if size <= 0 or size > int(max_bytes):
            return None
        raw = Path(p).read_bytes()
    except Exception:
        return None

    mime, _ = mimetypes.guess_type(p)
    if not mime:
        ext = Path(p).suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"
