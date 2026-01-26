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

import asyncio
import concurrent.futures
import os
import time

# Re-export des modules décomposés pour compatibilité
from src.ui.profile_api_cache import (
    ProfileAppearance,
    get_profile_api_cache_dir,
    load_cached_appearance,
    save_cached_appearance,
    load_cached_xuid_for_gamertag,
    save_cached_xuid,
    _cache_path,
    _xuid_cache_path,
    _safe_read_json,
    _safe_write_json,
)
from src.ui.profile_api_urls import (
    _to_image_url,
    _inventory_emblem_to_waypoint_png,
    _inventory_json_to_cms_url,
    _waypoint_nameplate_png_from_emblem,
    _inventory_backdrop_to_waypoint_png,
    resolve_inventory_png_via_api as _resolve_inventory_png_via_api,
)
from src.ui.profile_api_tokens import (
    _load_dotenv_if_present,
    _is_probable_auth_error,
    get_tokens as _get_tokens,
    ensure_spnkr_tokens,
)
from src.ui.career_ranks import format_career_rank_label_fr


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
                rank_label, rank_subtitle, rank_image_url = await _get_career_rank_for_player(
                    client, session, st_in, ct_in, xuid, _parse
                )

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


async def _get_career_rank_for_player(
    client, session, st_in: str, ct_in: str, xuid: str, parse_fn
) -> tuple[str | None, str | None, str | None]:
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
        career_track = await parse_fn(career_track_resp)
        
        if career_track is None:
            return None, None, None
        
        ranks_list = getattr(career_track, "ranks", None)
        if not ranks_list:
            return None, None, None
        
        # 2. Appeler l'API Economy pour la progression du joueur
        economy_host = "https://economy.svc.halowaypoint.com"
        
        headers = {
            "Accept": "application/json",
        }
        if st_in:
            headers["X-343-Authorization-Spartan"] = st_in
        if ct_in:
            headers["343-Clearance"] = ct_in
        
        career_progress = None
        response_format = None
        
        # Format 1 (den.dev): GET /hi/players/xuid({XUID})/rewardtracks/careerranks/careerrank1
        try:
            career_url = f"{economy_host}/hi/players/xuid({xu})/rewardtracks/careerranks/careerrank1"
            async with session.get(career_url, headers=headers) as resp:
                if resp.status == 200:
                    career_progress = await resp.json()
                    response_format = "direct"
        except Exception:
            pass
        
        # Format 2 (fallback Grunt): POST /hi/rewardtracks/careerRank1
        if career_progress is None:
            try:
                career_url = f"{economy_host}/hi/rewardtracks/careerRank1"
                body = {"Users": [f"xuid({xu})"]}
                async with session.post(career_url, headers=headers, json=body) as resp:
                    if resp.status == 200:
                        career_progress = await resp.json()
                        response_format = "wrapped"
            except Exception:
                pass
        
        if career_progress is None:
            return None, None, None
        
        # Extraire la progression selon le format de réponse
        if response_format == "direct":
            current_progress = career_progress.get("CurrentProgress", {})
        else:
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
        display_rank = current_rank if current_rank == 272 else current_rank + 1
        
        current_stage = None
        for rank_obj in ranks_list:
            r = getattr(rank_obj, "rank", None)
            if r == display_rank:
                current_stage = rank_obj
                break
        
        if current_stage is None:
            return f"Rang de carrière {display_rank}", f"XP {partial_xp}", None
        
        # 4. Construire le label du rang
        tier_type = getattr(current_stage, "tier_type", None)
        rank_title_obj = getattr(current_stage, "rank_title", None)
        rank_tier_obj = getattr(current_stage, "rank_tier", None)
        xp_required = getattr(current_stage, "xp_required_for_rank", None)
        rank_large_icon = getattr(current_stage, "rank_large_icon", None)
        
        rank_title = getattr(rank_title_obj, "value", None) if rank_title_obj else None
        rank_tier = getattr(rank_tier_obj, "value", None) if rank_tier_obj else None
        
        if current_rank == 272:
            r_label = format_career_rank_label_fr(tier=None, title=(rank_title or "Hero"), grade=None)
            r_subtitle = f"XP {partial_xp}/{xp_required}" if xp_required else f"XP {partial_xp}"
        else:
            r_label = format_career_rank_label_fr(tier=tier_type, title=rank_title, grade=rank_tier)
            if not r_label:
                r_label = f"Rang {display_rank}"
            r_subtitle = f"XP {partial_xp}/{xp_required}" if xp_required else f"XP {partial_xp}"
        
        # 5. Construire l'URL de l'icône
        r_icon = None
        if rank_large_icon:
            host = "https://gamecms-hacs.svc.halowaypoint.com"
            icon_path = str(rank_large_icon).lstrip("/")
            r_icon = f"{host}/hi/images/file/{icon_path}"
        
        return r_label, r_subtitle, r_icon
        
    except Exception:
        return None, None, None


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
