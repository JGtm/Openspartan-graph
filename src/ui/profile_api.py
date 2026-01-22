"""Récupération automatique d'assets profil (opt-in).

Objectif:
- Récupérer automatiquement des éléments d'identité (service tag, emblem, backdrop, nameplate)
  depuis l'API Halo Waypoint via SPNKr.
- Mettre en cache sur disque pour éviter d'appeler l'API à chaque rerun Streamlit.

Contraintes:
- Accès réseau uniquement si l'option est activée.
- Tolérance aux erreurs (pas de crash UI si SPNKr n'est pas installé / tokens manquants).
"""

from __future__ import annotations

import json
import os
import time
import concurrent.futures
import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileAppearance:
    service_tag: str | None = None
    emblem_image_url: str | None = None
    backdrop_image_url: str | None = None
    nameplate_image_url: str | None = None
    rank_label: str | None = None
    rank_subtitle: str | None = None
    rank_image_url: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_if_present() -> None:
    """Charge `.env.local` puis `.env` à la racine du repo (si présents).

    Objectif: permettre à Streamlit (et aux helpers UI) de trouver les variables
    SPNKr Azure sans exiger une export manuelle dans le shell.
    Règles:
    - lignes `KEY=VALUE`
    - ignore lignes vides / commentaires (#)
    - ne remplace pas une variable déjà définie dans l'environnement
    """

    repo_root = _repo_root()
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


def get_profile_api_cache_dir() -> Path:
    return _repo_root() / "data" / "cache" / "profile_api"


def _cache_path(xuid: str) -> Path:
    safe = "".join(ch for ch in str(xuid or "") if ch.isdigit())
    return get_profile_api_cache_dir() / f"appearance_{safe}.json"


def _xuid_cache_path(gamertag: str) -> Path:
    gt = str(gamertag or "").strip().lower()
    h = hashlib.sha256(gt.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return get_profile_api_cache_dir() / f"xuid_gt_{h}.json"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_fresh(fetched_at: float | None, *, refresh_hours: int) -> bool:
    # Convention UI: 0 = désactivé => re-fetch à chaque run (donc jamais "fresh").
    if refresh_hours <= 0:
        return False
    if fetched_at is None:
        return False
    try:
        age_s = time.time() - float(fetched_at)
    except Exception:
        return False
    return age_s < float(refresh_hours) * 3600.0


def _to_image_url(path: str | None) -> str | None:
    p = str(path or "").strip()
    if not p:
        return None
    if p.startswith("http://") or p.startswith("https://"):
        return p

    host = "https://gamecms-hacs.svc.halowaypoint.com"

    # Certains champs retournent déjà un chemin vers /hi/images/file/...
    p_lower = p.lower()
    marker = "/hi/images/file/"
    if marker in p_lower:
        rel = p[p_lower.index(marker) :]
        # Normalise la casse comme observé (hi/Images/file)
        rel = rel.replace("/hi/images/file/", "/hi/Images/file/")
        return f"{host}{rel}"

    rel = p.lstrip("/")
    return f"{host}/hi/Images/file/{rel}"


def _is_probable_auth_error(err: Exception) -> bool:
    msg = str(err or "")
    m = msg.lower()
    # Heuristique: SPNKr/aiohttp remontent souvent 401/unauthorized dans le message.
    return ("401" in m) or ("unauthorized" in m) or ("forbidden" in m and "403" in m)


def _inventory_emblem_to_waypoint_png(emblem_path: str | None, configuration_id: int | None) -> str | None:
    """Best-effort: convertit un chemin Inventory/Spartan/Emblems/<name>.json vers une image PNG Waypoint.

    Pattern observé:
    - Inventory/Spartan/Emblems/<stem>.json + configuration_id ->
      /hi/Waypoint/file/images/emblems/<stem>_<configuration_id>.png
    
    Note: Ce pattern ne fonctionne pas pour tous les emblèmes (ex: ranked, certains events).
    Dans ces cas, il faut appeler l'API progression pour récupérer le PNG via DisplayPath.
    """

    p = str(emblem_path or "").strip()
    if not p or configuration_id is None:
        return None
    try:
        cfg = int(configuration_id)
    except Exception:
        return None
    if cfg <= 0:
        return None

    if "/Spartan/Emblems/" not in p:
        return None

    # Récupère le stem sans extension (.json)
    name = p.split("/Spartan/Emblems/", 1)[1].split("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if not stem:
        return None

    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/Waypoint/file/images/emblems/{stem}_{cfg}.png"


def _inventory_json_to_cms_url(inventory_path: str | None) -> str | None:
    """Construit l'URL vers le fichier JSON CMS pour un chemin d'inventaire.
    
    Ex: Inventory/Spartan/Emblems/104-001-olympus-stuck-3d208338.json
      -> https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/Inventory/Spartan/Emblems/104-001-olympus-stuck-3d208338.json
    """
    p = str(inventory_path or "").strip()
    if not p:
        return None
    
    # Normaliser le chemin
    if p.startswith("/"):
        p = p[1:]
    
    # Vérifier que c'est un chemin d'inventaire
    p_lower = p.lower()
    if not (p_lower.startswith("inventory/") or "/inventory/" in p_lower):
        return None
    
    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/progression/file/{p}"


async def _resolve_inventory_png_via_api(
    session,
    inventory_path: str | None,
    *,
    spartan_token: str,
    clearance_token: str,
) -> str | None:
    """Résout un chemin Inventory/*.json vers l'URL du PNG réel via l'API progression.
    
    Appelle /hi/progression/file/Inventory/... pour récupérer le JSON de métadonnées,
    puis extrait CommonData.DisplayPath.Media.MediaUrl.Path pour construire l'URL du PNG.
    
    Retourne l'URL complète vers le PNG, ou None si non résolu.
    """
    cms_url = _inventory_json_to_cms_url(inventory_path)
    if not cms_url:
        return None
    
    headers = {
        "Accept": "application/json",
        "X-343-Authorization-Spartan": spartan_token,
        "343-Clearance": clearance_token,
    }
    
    try:
        async with session.get(cms_url, headers=headers) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    
    # Extraire CommonData.DisplayPath.Media.MediaUrl.Path
    common_data = data.get("CommonData", {})
    display_path = common_data.get("DisplayPath", {})
    media = display_path.get("Media", {})
    media_url = media.get("MediaUrl", {})
    png_path = media_url.get("Path", "")
    
    if not png_path:
        # Fallback: essayer FolderPath + FileName
        folder = display_path.get("FolderPath", "")
        filename = display_path.get("FileName", "")
        if folder and filename:
            png_path = f"{folder}/{filename}"
    
    if not png_path:
        return None
    
    # Construire l'URL complète
    png_path = png_path.lstrip("/")
    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/images/file/{png_path}"


def _waypoint_nameplate_png_from_emblem(emblem_path: str | None, configuration_id: int | None) -> str | None:
    """Best-effort: construit une URL nameplate Waypoint basée sur l'emblem.

    Pattern observé:
    - /hi/Waypoint/file/images/nameplates/<emblem_stem>_<configuration_id>.png
    
    Note: Le configuration_id est un entier 32 bits signé dans l'API.
    Nous utilisons la valeur signée directement car c'est ce que Waypoint attend.
    Certaines combinaisons emblem + configuration_id n'ont pas de nameplate générée.
    """

    p = str(emblem_path or "").strip()
    if not p or configuration_id is None:
        return None
    try:
        cfg = int(configuration_id)
    except Exception:
        return None
    # Accepter tous les configuration_id non-nuls (positifs ou négatifs)
    if cfg == 0:
        return None

    if "/Spartan/Emblems/" not in p:
        return None

    name = p.split("/Spartan/Emblems/", 1)[1].split("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if not stem:
        return None

    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/Waypoint/file/images/nameplates/{stem}_{cfg}.png"


def _inventory_backdrop_to_waypoint_png(backdrop_path: str | None) -> str | None:
    """Best-effort: convertit un chemin Inventory/Spartan/BackdropImages/<name>.json vers une image PNG Waypoint.

    Pattern observé:
    - Inventory/Spartan/BackdropImages/<stem>.json ->
      /hi/Waypoint/file/images/backdrops/<stem>.png
    
    ATTENTION: Ce pattern ne fonctionne PAS pour la majorité des backdrops !
    Les backdrops utilisent généralement des chemins comme `progression/backgrounds/...`
    qui ne sont pas dans le mapping Waypoint.
    
    Cette fonction ne retourne une URL que si le backdrop est déjà un chemin PNG direct.
    Pour les autres cas, utiliser _resolve_inventory_png_via_api().
    """

    p = str(backdrop_path or "").strip()
    if not p:
        return None

    # Si c'est déjà un PNG/une vraie image, retourner tel quel
    p_lower = p.lower()
    if p_lower.endswith(".png") or p_lower.endswith(".jpg") or p_lower.endswith(".jpeg"):
        return _to_image_url(p)

    # Pour les fichiers JSON, ne pas utiliser le pattern Waypoint (ne fonctionne pas)
    # -> Laisser le caller utiliser _resolve_inventory_png_via_api() à la place
    return None


def ensure_spnkr_tokens(*, timeout_seconds: int = 12) -> tuple[bool, str | None]:
    """Best-effort: s'assure que SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN sont disponibles.

    Utile quand on utilise le cache d'apparence (donc pas d'appel API au run courant),
    mais qu'on veut quand même télécharger des assets /hi/Images/file/ qui exigent une auth.

    Retourne (ok, error_message).
    """

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

    async def _run() -> None:
        import aiohttp

        st = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip() or None
        ct = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip() or None

        timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await _get_tokens(
                session,
                spartan_token=st,
                clearance_token=ct,
                timeout_seconds=int(timeout_seconds),
            )

    try:
        _run_sync(_run())
        st = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip()
        ct = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()
        if st and ct:
            return True, None
        return False, "Tokens SPNKr introuvables (env et Azure refresh non configurés)."
    except Exception as e:
        return False, str(e)


async def _get_tokens(
    session,
    *,
    spartan_token: str | None,
    clearance_token: str | None,
    timeout_seconds: int,
) -> tuple[str, str]:
    _load_dotenv_if_present()
    st = str(spartan_token or "").strip()
    ct = str(clearance_token or "").strip()
    if st and ct:
        # Rend les tokens accessibles aux autres helpers (ex: téléchargement d'assets).
        os.environ["SPNKR_SPARTAN_TOKEN"] = st
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct
        return st, ct

    # Fallback: récupère via Azure refresh token (opt-in)
    azure_client_id = str(os.environ.get("SPNKR_AZURE_CLIENT_ID") or "").strip()
    azure_client_secret = str(os.environ.get("SPNKR_AZURE_CLIENT_SECRET") or "").strip()
    azure_redirect_uri = str(os.environ.get("SPNKR_AZURE_REDIRECT_URI") or "").strip() or "https://localhost"
    oauth_refresh_token = str(os.environ.get("SPNKR_OAUTH_REFRESH_TOKEN") or "").strip()

    if not (azure_client_id and azure_client_secret and oauth_refresh_token):
        raise RuntimeError(
            "Tokens manquants: définis soit SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN, "
            "soit SPNKR_AZURE_CLIENT_ID + SPNKR_AZURE_CLIENT_SECRET + SPNKR_OAUTH_REFRESH_TOKEN."
        )

    from spnkr import AzureApp, refresh_player_tokens

    async def _refresh_oauth_access_token_v2(refresh_token: str, app: AzureApp) -> str:
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
            raise RuntimeError(
                "Échec refresh OAuth v2 (consumers). "
                f"status={resp.status} error={payload.get('error')} desc={payload.get('error_description')}"
            )
        access = payload.get("access_token")
        if not isinstance(access, str) or not access.strip():
            raise RuntimeError("OAuth v2: pas de access_token dans la réponse.")
        return access.strip()

    app = AzureApp(azure_client_id, azure_client_secret, azure_redirect_uri)
    try:
        player = await refresh_player_tokens(session, app, oauth_refresh_token)
        st2, ct2 = str(player.spartan_token.token), str(player.clearance_token.token)
        os.environ["SPNKR_SPARTAN_TOKEN"] = st2
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct2
        return st2, ct2
    except Exception as e:
        msg = str(e)
        if "invalid_client" not in msg or "client_secret" not in msg:
            raise

        # Fallback: endpoint OAuth v2 (consumers) -> chain Xbox/XSTS/Halo
        from spnkr.auth.xbox import request_user_token, request_xsts_token
        from spnkr.auth.core import XSTS_V3_HALO_AUDIENCE, XSTS_V3_XBOX_AUDIENCE
        from spnkr.auth.halo import request_spartan_token, request_clearance_token

        access_token = await _refresh_oauth_access_token_v2(oauth_refresh_token, app)
        user_token = await request_user_token(session, access_token)
        _ = await request_xsts_token(session, user_token.token, XSTS_V3_XBOX_AUDIENCE)
        halo_xsts_token = await request_xsts_token(session, user_token.token, XSTS_V3_HALO_AUDIENCE)
        spartan = await request_spartan_token(session, halo_xsts_token.token)
        clearance = await request_clearance_token(session, spartan.token)
        st3, ct3 = str(spartan.token), str(clearance.token)
        os.environ["SPNKR_SPARTAN_TOKEN"] = st3
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct3
        return st3, ct3


def load_cached_appearance(xuid: str, *, refresh_hours: int) -> ProfileAppearance | None:
    cp = _cache_path(xuid)
    data = _safe_read_json(cp)
    if not data:
        return None
    fetched_at = data.get("fetched_at")
    if not _is_fresh(fetched_at if isinstance(fetched_at, (int, float)) else None, refresh_hours=refresh_hours):
        return None

    return ProfileAppearance(
        service_tag=(str(data.get("service_tag") or "").strip() or None),
        emblem_image_url=(str(data.get("emblem_image_url") or "").strip() or None),
        backdrop_image_url=(str(data.get("backdrop_image_url") or "").strip() or None),
        nameplate_image_url=(str(data.get("nameplate_image_url") or "").strip() or None),
        rank_label=(str(data.get("rank_label") or "").strip() or None),
        rank_subtitle=(str(data.get("rank_subtitle") or "").strip() or None),
        rank_image_url=(str(data.get("rank_image_url") or "").strip() or None),
    )


def save_cached_appearance(xuid: str, appearance: ProfileAppearance) -> None:
    """Sauvegarde l'apparence dans le cache disque."""
    cp = _cache_path(xuid)
    data = {
        "fetched_at": time.time(),
        "service_tag": appearance.service_tag,
        "emblem_image_url": appearance.emblem_image_url,
        "backdrop_image_url": appearance.backdrop_image_url,
        "nameplate_image_url": appearance.nameplate_image_url,
        "rank_label": appearance.rank_label,
        "rank_subtitle": appearance.rank_subtitle,
        "rank_image_url": appearance.rank_image_url,
    }
    _safe_write_json(cp, data)


def save_cached_xuid(gamertag: str, xuid: str) -> None:
    """Sauvegarde le mapping gamertag → xuid dans le cache disque."""
    cp = _xuid_cache_path(gamertag)
    data = {
        "fetched_at": time.time(),
        "gamertag": gamertag,
        "xuid": xuid,
    }
    _safe_write_json(cp, data)


def fetch_appearance_via_spnkr(
    *,
    xuid: str,
    spartan_token: str | None = None,
    clearance_token: str | None = None,
    requests_per_second: int = 3,
    timeout_seconds: int = 12,
) -> ProfileAppearance:
    """Appelle l'API Halo Waypoint via SPNKr pour récupérer l'apparence."""

    def _run_sync(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            # Certains environnements peuvent déjà avoir une loop.
            msg = str(e)
            if "asyncio.run() cannot be called" not in msg:
                raise
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(coro))
                return fut.result(timeout=float(timeout_seconds) + 20.0)

    async def _run() -> ProfileAppearance:
        import aiohttp
        from spnkr.client import HaloInfiniteClient

        timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def _parse(resp: object) -> object:
                if hasattr(resp, "data"):
                    return getattr(resp, "data")
                if hasattr(resp, "parse"):
                    return await resp.parse()  # type: ignore[attr-defined]
                return resp

            async def _call_with_tokens(st_in: str, ct_in: str):
                client = HaloInfiniteClient(
                    session,
                    spartan_token=st_in,
                    clearance_token=ct_in,
                    requests_per_second=int(requests_per_second),
                )
                resp = await client.economy.get_player_customization(str(xuid).strip(), view_type="public")
                customization = await _parse(resp)

                # Best-effort: Career Rank du joueur
                rank_label: str | None = None
                rank_subtitle: str | None = None
                rank_image_url: str | None = None

                async def _get_career_rank_for_player() -> tuple[str | None, str | None, str | None]:
                    """Récupère le Career Rank du joueur via les APIs Halo.
                    
                    1. Appelle gamecms_hacs.get_career_reward_track() pour les métadonnées des rangs
                    2. Appelle l'endpoint Economy pour la progression du joueur
                    """
                    xu = str(xuid).strip()
                    
                    try:
                        # 1. Récupérer les métadonnées des rangs via SPNKr
                        gamecms = getattr(client, "gamecms_hacs", None)
                        if gamecms is None:
                            return None, None, None
                        
                        career_track_resp = await gamecms.get_career_reward_track()
                        career_track = await _parse(career_track_resp)
                        
                        if career_track is None:
                            return None, None, None
                        
                        ranks_list = getattr(career_track, "ranks", None)
                        if not ranks_list:
                            return None, None, None
                        
                        # 2. Appeler l'API Economy pour la progression du joueur
                        # Endpoint documenté par den.dev: https://den.dev/blog/halo-infinite-career-api/
                        economy_host = "https://economy.svc.halowaypoint.com"
                        
                        # Utiliser la session avec les headers d'authentification
                        headers = {
                            "Accept": "application/json",
                        }
                        if st_in:
                            headers["X-343-Authorization-Spartan"] = st_in
                        if ct_in:
                            headers["343-Clearance"] = ct_in
                        
                        import logging
                        logger = logging.getLogger(__name__)
                        
                        career_progress = None
                        response_format = None  # Pour adapter le parsing selon le format
                        
                        # Format 1 (den.dev): GET /hi/players/xuid({XUID})/rewardtracks/careerranks/careerrank1
                        # Réponse directe: { "CurrentProgress": { "Rank": N, "PartialProgress": XP }}
                        try:
                            career_url = f"{economy_host}/hi/players/xuid({xu})/rewardtracks/careerranks/careerrank1"
                            logger.info(f"Career Rank API attempt 1 (den.dev format): {career_url}")
                            async with session.get(career_url, headers=headers) as resp:
                                logger.info(f"Career Rank API response status: {resp.status}")
                                if resp.status == 200:
                                    career_progress = await resp.json()
                                    response_format = "direct"  # CurrentProgress au premier niveau
                                    logger.info(f"Career Rank API success (format 1)")
                        except Exception as e:
                            logger.warning(f"Career Rank GET (den.dev) failed: {e}")
                        
                        # Format 2 (fallback Grunt): POST /hi/rewardtracks/careerRank1 avec body
                        # Réponse encapsulée: { "RewardTracks": [{ "Result": { "CurrentProgress": {...}}}]}
                        if career_progress is None:
                            try:
                                career_url = f"{economy_host}/hi/rewardtracks/careerRank1"
                                body = {"Users": [f"xuid({xu})"]}
                                logger.info(f"Career Rank API attempt 2 (Grunt format POST): {career_url}")
                                async with session.post(career_url, headers=headers, json=body) as resp:
                                    logger.info(f"Career Rank API response status: {resp.status}")
                                    if resp.status == 200:
                                        career_progress = await resp.json()
                                        response_format = "wrapped"  # RewardTracks[0].Result.CurrentProgress
                                        logger.info(f"Career Rank API success (format 2)")
                            except Exception as e:
                                logger.warning(f"Career Rank POST (Grunt) failed: {e}")
                        
                        if career_progress is None:
                            return None, None, None
                        
                        # Extraire la progression selon le format de réponse
                        if response_format == "direct":
                            # Format den.dev: { "CurrentProgress": { "Rank": N, "PartialProgress": XP }}
                            current_progress = career_progress.get("CurrentProgress", {})
                        else:
                            # Format Grunt: { "RewardTracks": [{ "Result": { "CurrentProgress": {...}}}]}
                            reward_tracks = career_progress.get("RewardTracks", [])
                            if not reward_tracks:
                                return None, None, None
                            track0 = reward_tracks[0]
                            result = track0.get("Result", {})
                            current_progress = result.get("CurrentProgress", {})
                        
                        current_rank = current_progress.get("Rank")
                        partial_xp = current_progress.get("PartialProgress", 0)
                        
                        if current_rank is None:
                            return None, None, None
                        
                        # 3. Trouver le stage correspondant dans les métadonnées
                        # Note: rank affiché = rank+1 sauf Hero (272)
                        display_rank = current_rank if current_rank == 272 else current_rank + 1
                        
                        current_stage = None
                        for rank_obj in ranks_list:
                            r = getattr(rank_obj, "rank", None)
                            if r == display_rank:
                                current_stage = rank_obj
                                break
                        
                        if current_stage is None:
                            return f"Career Rank {display_rank}", f"XP {partial_xp}", None
                        
                        # 4. Construire le label du rang
                        tier_type = getattr(current_stage, "tier_type", None)
                        rank_title_obj = getattr(current_stage, "rank_title", None)
                        rank_tier_obj = getattr(current_stage, "rank_tier", None)
                        xp_required = getattr(current_stage, "xp_required_for_rank", None)
                        rank_large_icon = getattr(current_stage, "rank_large_icon", None)
                        
                        # Extraire les valeurs des objets TranslatableString
                        rank_title = getattr(rank_title_obj, "value", None) if rank_title_obj else None
                        rank_tier = getattr(rank_tier_obj, "value", None) if rank_tier_obj else None
                        
                        # Construire le label
                        if current_rank == 272:
                            # Hero rank
                            r_label = rank_title or "Hero"
                            r_subtitle = f"XP {partial_xp}/{xp_required}" if xp_required else f"XP {partial_xp}"
                        else:
                            # Rang normal: "Bronze Private 1" -> "Bronze Private 1"
                            parts = [p for p in [tier_type, rank_title, rank_tier] if p]
                            r_label = " ".join(parts) if parts else f"Rank {display_rank}"
                            r_subtitle = f"XP {partial_xp}/{xp_required}" if xp_required else f"XP {partial_xp}"
                        
                        # 5. Construire l'URL de l'icône
                        r_icon = None
                        if rank_large_icon:
                            # rank_large_icon ex: "career_rank/CelebrationMoment/19_Cadet_Onyx_III.png"
                            host = "https://gamecms-hacs.svc.halowaypoint.com"
                            icon_path = str(rank_large_icon).lstrip("/")
                            r_icon = f"{host}/hi/images/file/{icon_path}"
                        
                        return r_label, r_subtitle, r_icon
                        
                    except Exception:
                        return None, None, None

                rank_label, rank_subtitle, rank_image_url = await _get_career_rank_for_player()

                return customization, rank_label, rank_subtitle, rank_image_url

            st, ct = await _get_tokens(
                session,
                spartan_token=spartan_token,
                clearance_token=clearance_token,
                timeout_seconds=timeout_seconds,
            )

            try:
                customization, rank_label, rank_subtitle, rank_image_url = await _call_with_tokens(st, ct)
            except Exception as e:
                # Si tokens fournis expirés, tente une régénération via Azure refresh (si configuré).
                if (spartan_token and clearance_token) and _is_probable_auth_error(e):
                    st, ct = await _get_tokens(
                        session,
                        spartan_token=None,
                        clearance_token=None,
                        timeout_seconds=timeout_seconds,
                    )
                    customization, rank_label, rank_subtitle, rank_image_url = await _call_with_tokens(st, ct)
                else:
                    raise

            appearance = customization.appearance

            emblem_path = getattr(appearance.emblem, "emblem_path", None)
            emblem_cfg = getattr(appearance.emblem, "configuration_id", None)
            
            # Essayer d'abord le pattern Waypoint (rapide, pas d'appel API supplémentaire)
            emblem_url = _inventory_emblem_to_waypoint_png(emblem_path, emblem_cfg)
            
            # Si le pattern Waypoint ne fonctionne pas, résoudre via l'API progression
            if not emblem_url and emblem_path:
                emblem_url = await _resolve_inventory_png_via_api(
                    session, emblem_path, spartan_token=st, clearance_token=ct
                )
            
            # Dernier fallback: URL vers le JSON (sera résolu au moment du téléchargement)
            if not emblem_url:
                emblem_url = _to_image_url(emblem_path)
            
            # Backdrop: convertir le chemin JSON en URL PNG Waypoint
            backdrop_raw = getattr(appearance, "backdrop_image_path", None)
            backdrop_url = _inventory_backdrop_to_waypoint_png(backdrop_raw)
            
            # Si le pattern Waypoint ne fonctionne pas, résoudre via l'API progression
            if not backdrop_url and backdrop_raw:
                backdrop_url = await _resolve_inventory_png_via_api(
                    session, backdrop_raw, spartan_token=st, clearance_token=ct
                )
            
            # Dernier fallback
            if not backdrop_url:
                backdrop_url = _to_image_url(backdrop_raw)
            
            player_title_path = getattr(appearance, "player_title_path", None)
            nameplate_url = _to_image_url(player_title_path) or _waypoint_nameplate_png_from_emblem(
                emblem_path,
                emblem_cfg,
            )

            return ProfileAppearance(
                service_tag=str(getattr(appearance, "service_tag", "") or "").strip() or None,
                emblem_image_url=emblem_url,
                backdrop_image_url=backdrop_url,
                nameplate_image_url=nameplate_url,
                rank_label=(str(rank_label or "").strip() or None),
                rank_subtitle=(str(rank_subtitle or "").strip() or None),
                rank_image_url=(str(rank_image_url or "").strip() or None),
            )

    return _run_sync(_run())


def load_cached_xuid_for_gamertag(gamertag: str, *, refresh_hours: int) -> str | None:
    cp = _xuid_cache_path(gamertag)
    data = _safe_read_json(cp)
    if not data:
        return None
    fetched_at = data.get("fetched_at")
    if not _is_fresh(fetched_at if isinstance(fetched_at, (int, float)) else None, refresh_hours=refresh_hours):
        return None
    xuid = str(data.get("xuid") or "").strip()
    return xuid if xuid.isdigit() else None


def fetch_xuid_via_spnkr(
    *,
    gamertag: str,
    spartan_token: str | None = None,
    clearance_token: str | None = None,
    requests_per_second: int = 3,
    timeout_seconds: int = 12,
) -> tuple[str, str]:
    """Retourne (xuid_digits, canonical_gamertag) pour un gamertag."""

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

    async def _run() -> tuple[str, str]:
        import aiohttp
        from spnkr.client import HaloInfiniteClient

        gt = str(gamertag or "").strip()
        if not gt:
            raise RuntimeError("Gamertag vide.")

        timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def _call_with_tokens(st_in: str, ct_in: str):
                client = HaloInfiniteClient(
                    session,
                    spartan_token=st_in,
                    clearance_token=ct_in,
                    requests_per_second=int(requests_per_second),
                )
                return await client.profile.get_user_by_gamertag(gt)

            st, ct = await _get_tokens(
                session,
                spartan_token=spartan_token,
                clearance_token=clearance_token,
                timeout_seconds=timeout_seconds,
            )

            try:
                resp = await _call_with_tokens(st, ct)
            except Exception as e:
                if (spartan_token and clearance_token) and _is_probable_auth_error(e):
                    st, ct = await _get_tokens(
                        session,
                        spartan_token=None,
                        clearance_token=None,
                        timeout_seconds=timeout_seconds,
                    )
                    resp = await _call_with_tokens(st, ct)
                else:
                    raise

            # Compat: selon la version SPNKr, JsonResponse expose `data` ou seulement `parse()`.
            if hasattr(resp, "data"):
                user = resp.data
            else:
                user = await resp.parse()

            xuid = str(getattr(user, "xuid", "") or "").strip()
            if not xuid.isdigit():
                raise RuntimeError("Impossible de résoudre le XUID pour ce gamertag.")
            canonical_gt = str(getattr(user, "gamertag", "") or "").strip() or gt
            return xuid, canonical_gt

    return _run_sync(_run())


def get_xuid_for_gamertag(
    *,
    gamertag: str,
    enabled: bool,
    refresh_hours: int,
    force_refresh: bool = False,
    requests_per_second: int = 3,
    timeout_seconds: int = 12,
) -> tuple[str | None, str | None]:
    """Retourne (xuid, error_message)."""

    gt = str(gamertag or "").strip()
    if not gt:
        return None, None

    if not force_refresh:
        cached = load_cached_xuid_for_gamertag(gt, refresh_hours=refresh_hours)
        if cached is not None:
            return cached, None

    if not enabled:
        return None, None

    try:
        xuid, canonical_gt = fetch_xuid_via_spnkr(
            gamertag=gt,
            spartan_token=str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip() or None,
            clearance_token=str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip() or None,
            requests_per_second=requests_per_second,
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        return (
            None,
            (
                "Échec résolution XUID via SPNKr: "
                f"{e} (attendu: SPNKR_AZURE_CLIENT_ID + SPNKR_AZURE_CLIENT_SECRET + SPNKR_OAUTH_REFRESH_TOKEN, "
                "ou SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN)"
            ),
        )

    try:
        _safe_write_json(
            _xuid_cache_path(gt),
            {
                "fetched_at": time.time(),
                "gamertag": canonical_gt,
                "xuid": xuid,
            },
        )
    except Exception:
        pass

    return xuid, None


def get_profile_appearance(
    *,
    xuid: str,
    enabled: bool,
    refresh_hours: int,
    requests_per_second: int = 3,
    timeout_seconds: int = 12,
    force_refresh: bool = False,
) -> tuple[ProfileAppearance | None, str | None]:
    """Retourne (appearance, error_message).

    - Tente d'abord le cache disque.
    - Sinon, appelle l'API si enabled et tokens présents.
    """

    xu = str(xuid or "").strip()
    if not xu or not xu.isdigit():
        return None, None

    if not force_refresh:
        cached = load_cached_appearance(xu, refresh_hours=refresh_hours)
        if cached is not None:
            return cached, None

    if not enabled:
        return None, None

    try:
        appearance = fetch_appearance_via_spnkr(
            xuid=xu,
            spartan_token=str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip() or None,
            clearance_token=str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip() or None,
            requests_per_second=requests_per_second,
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        return (
            None,
            (
                "Échec récupération profil via SPNKr: "
                f"{e} (attendu: SPNKR_AZURE_CLIENT_ID + SPNKR_AZURE_CLIENT_SECRET + SPNKR_OAUTH_REFRESH_TOKEN, "
                "ou SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN)"
            ),
        )

    try:
        _safe_write_json(
            _cache_path(xu),
            {
                "fetched_at": time.time(),
                "service_tag": appearance.service_tag,
                "emblem_image_url": appearance.emblem_image_url,
                "backdrop_image_url": appearance.backdrop_image_url,
                "nameplate_image_url": appearance.nameplate_image_url,
                "rank_label": appearance.rank_label,
                "rank_subtitle": appearance.rank_subtitle,
                "rank_image_url": appearance.rank_image_url,
            },
        )
    except Exception:
        # Cache best-effort
        pass

    return appearance, None
