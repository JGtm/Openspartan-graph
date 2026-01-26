"""Helpers pour le main() de streamlit_app.py.

Fonctions d'initialisation, de validation et de rendu du profil.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from src.config import get_default_db_path
from src.db import resolve_xuid_from_db
from src.db.parsers import parse_xuid_input
from src.ui import (
    display_name_from_xuid,
    get_hero_html,
    get_profile_appearance,
    ensure_spnkr_tokens,
    load_settings,
)
from src.ui.player_assets import download_image_to_cache, ensure_local_image_path
from src.ui.sync import (
    pick_latest_spnkr_db_if_any,
    is_spnkr_db_path,
    render_sync_indicator,
    sync_all_players,
)
from src.ui.multiplayer import render_player_selector
from src.ui.cache import clear_app_caches, load_df_optimized, db_cache_key
from src.analysis import mark_firefight

from src.app.data_loader import default_identity_from_secrets, init_source_state

if TYPE_CHECKING:
    from src.ui import AppSettings


def propagate_identity_to_env() -> None:
    """Propage les defaults depuis secrets vers l'environnement."""
    try:
        xuid_or_gt, xuid_fallback, wp = default_identity_from_secrets()
        if xuid_or_gt and not str(xuid_or_gt).strip().isdigit() and xuid_fallback:
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_GAMERTAG"] = str(xuid_or_gt).strip()
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_XUID"] = str(xuid_fallback).strip()
        if wp and not str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip():
            os.environ["OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER"] = str(wp).strip()
    except Exception:
        pass


def apply_settings_path_overrides(settings: AppSettings) -> None:
    """Applique les overrides de chemins depuis les settings."""
    try:
        aliases_override = str(getattr(settings, "aliases_path", "") or "").strip()
        if aliases_override:
            os.environ["OPENSPARTAN_ALIASES_PATH"] = aliases_override
        else:
            os.environ.pop("OPENSPARTAN_ALIASES_PATH", None)
    except Exception:
        pass
    try:
        profiles_override = str(getattr(settings, "profiles_path", "") or "").strip()
        if profiles_override:
            os.environ["OPENSPARTAN_PROFILES_PATH"] = profiles_override
        else:
            os.environ.pop("OPENSPARTAN_PROFILES_PATH", None)
    except Exception:
        pass


def validate_and_fix_db_path(db_path: str, default_db: str) -> str:
    """Valide le chemin DB et applique un fallback si nécessaire.
    
    Returns:
        Chemin valide ou chaîne vide.
    """
    if db_path and not os.path.exists(db_path):
        return ""
    
    if db_path and os.path.exists(db_path):
        try:
            if os.path.getsize(db_path) <= 0:
                st.warning("La base sélectionnée est vide (0 octet). Basculement automatique vers une DB valide si possible.")
                fallback = ""
                if is_spnkr_db_path(db_path):
                    fallback = pick_latest_spnkr_db_if_any()
                    if fallback and os.path.exists(fallback) and os.path.getsize(fallback) <= 0:
                        fallback = ""
                if not fallback:
                    fallback = str(default_db or "").strip()
                    if not (fallback and os.path.exists(fallback)):
                        fallback = ""
                if fallback and fallback != db_path:
                    st.info(f"DB utilisée: {fallback}")
                    st.session_state["db_path"] = fallback
                    return fallback
                return ""
        except Exception:
            pass
    
    return db_path


def resolve_xuid_from_input(xuid_input: str, db_path: str) -> str:
    """Résout le XUID depuis l'entrée utilisateur.
    
    Args:
        xuid_input: Entrée brute (XUID ou gamertag).
        db_path: Chemin vers la DB.
        
    Returns:
        XUID résolu ou chaîne vide.
    """
    xraw = (xuid_input or "").strip()
    xuid_resolved = parse_xuid_input(xraw) or ""
    
    if not xuid_resolved and xraw and not xraw.isdigit() and db_path:
        xuid_resolved = resolve_xuid_from_db(db_path, xraw) or ""
        # Fallback: secrets/env quand l'entrée correspond au gamertag par défaut
        if not xuid_resolved:
            try:
                xuid_or_gt, xuid_fallback, _wp = default_identity_from_secrets()
                if (
                    xuid_or_gt
                    and xuid_fallback
                    and (not str(xuid_or_gt).strip().isdigit())
                    and str(xuid_or_gt).strip().casefold() == str(xraw).strip().casefold()
                ):
                    xuid_resolved = str(xuid_fallback).strip()
            except Exception:
                pass
    
    if not xuid_resolved and not xraw and db_path:
        xuid_or_gt, xuid_fallback, _wp = default_identity_from_secrets()
        if xuid_or_gt and not xuid_or_gt.isdigit():
            xuid_resolved = resolve_xuid_from_db(db_path, xuid_or_gt) or xuid_fallback
        else:
            xuid_resolved = xuid_or_gt or xuid_fallback
    
    return xuid_resolved or ""


def render_sidebar_header(db_path: str, xuid: str, settings: AppSettings) -> str:
    """Rend le header de la sidebar (brand, sync, player selector).
    
    Returns:
        XUID potentiellement mis à jour.
    """
    st.markdown("<div class='os-sidebar-brand' style='font-size: 2.5em;'>LevelUp</div>", unsafe_allow_html=True)
    st.markdown("<div class='os-sidebar-divider'></div>", unsafe_allow_html=True)

    # Indicateur de dernière synchronisation
    if db_path and os.path.exists(db_path):
        render_sync_indicator(db_path)

    # Sélecteur multi-joueurs (si DB fusionnée)
    if db_path and os.path.exists(db_path):
        new_xuid = render_player_selector(db_path, xuid, key="sidebar_player_selector")
        if new_xuid:
            st.session_state["xuid_input"] = new_xuid
            xuid = new_xuid
            # Reset des filtres au changement de joueur
            for filter_key in ["filter_playlists", "filter_modes", "filter_maps"]:
                if filter_key in st.session_state:
                    del st.session_state[filter_key]
            st.rerun()

    # Bouton Sync pour toutes les DB SPNKr
    if db_path and is_spnkr_db_path(db_path) and os.path.exists(db_path):
        if st.button(
            "🔄 Synchroniser",
            key="sidebar_sync_button",
            help="Synchronise tous les joueurs (nouveaux matchs, highlights, aliases).",
            use_container_width=True,
        ):
            with st.spinner("Synchronisation en cours..."):
                ok, msg = sync_all_players(
                    db_path=db_path,
                    match_type=str(getattr(settings, "spnkr_refresh_match_type", "matchmaking") or "matchmaking"),
                    max_matches=int(getattr(settings, "spnkr_refresh_max_matches", 200) or 200),
                    rps=int(getattr(settings, "spnkr_refresh_rps", 5) or 5),
                    with_highlight_events=True,
                    with_aliases=True,
                    delta=True,
                    timeout_seconds=180,
                )
            if ok:
                st.success(msg)
                clear_app_caches()
                st.rerun()
            else:
                st.error(msg)
    
    return xuid


def load_profile_api(xuid: str, settings: AppSettings) -> tuple[object | None, str | None]:
    """Charge le profil depuis l'API SPNKr si activé.
    
    Returns:
        (api_appearance, error_message)
    """
    api_enabled = bool(getattr(settings, "profile_api_enabled", False))
    api_refresh_h = int(getattr(settings, "profile_api_auto_refresh_hours", 0) or 0)
    api_app = None
    api_err = None
    
    if api_enabled and str(xuid or "").strip():
        try:
            api_app, api_err = get_profile_appearance(
                xuid=str(xuid).strip(),
                enabled=True,
                refresh_hours=api_refresh_h,
            )
        except Exception as e:
            api_app, api_err = None, str(e)
        if api_err:
            st.caption(f"Profil auto (SPNKr): {api_err}")
    
    return api_app, api_err


def _needs_halo_auth(url: str) -> bool:
    """Vérifie si l'URL nécessite une authentification Halo."""
    u = str(url or "").strip().lower()
    if not u:
        return False
    return (
        ("/hi/images/file/" in u)
        or ("/hi/waypoint/file/images/nameplates/" in u)
        or u.startswith("inventory/")
        or u.startswith("/inventory/")
        or ("gamecms-hacs.svc.halowaypoint.com/hi/images/file/" in u)
    )


def render_profile_hero(
    xuid: str,
    settings: AppSettings,
    api_app: object | None,
) -> None:
    """Rend le hero/header du profil joueur."""
    me_name = display_name_from_xuid(xuid.strip()) if str(xuid or "").strip() else "(joueur)"
    
    dl_enabled = bool(getattr(settings, "profile_assets_download_enabled", False)) or bool(getattr(settings, "profile_api_enabled", False))
    refresh_h = int(getattr(settings, "profile_assets_auto_refresh_hours", 0) or 0)

    # Valeurs manuelles (prioritaires) / sinon auto depuis API
    banner_value = str(getattr(settings, "profile_banner", "") or "").strip()
    emblem_value = str(getattr(settings, "profile_emblem", "") or "").strip() or (getattr(api_app, "emblem_image_url", None) if api_app else "")
    backdrop_value = str(getattr(settings, "profile_backdrop", "") or "").strip() or (getattr(api_app, "backdrop_image_url", None) if api_app else "")
    nameplate_value = str(getattr(settings, "profile_nameplate", "") or "").strip() or (getattr(api_app, "nameplate_image_url", None) if api_app else "")
    service_tag_value = str(getattr(settings, "profile_service_tag", "") or "").strip() or (getattr(api_app, "service_tag", None) if api_app else "")
    rank_label_value = str(getattr(settings, "profile_rank_label", "") or "").strip() or (getattr(api_app, "rank_label", None) if api_app else "")
    rank_subtitle_value = str(getattr(settings, "profile_rank_subtitle", "") or "").strip() or (getattr(api_app, "rank_subtitle", None) if api_app else "")
    rank_icon_value = (getattr(api_app, "rank_image_url", None) if api_app else "") or ""

    # Tokens Halo si nécessaire
    if dl_enabled and (not str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()):
        if _needs_halo_auth(backdrop_value) or _needs_halo_auth(rank_icon_value) or _needs_halo_auth(nameplate_value):
            ensure_spnkr_tokens(timeout_seconds=12)

    # Résolution des chemins locaux
    banner_path = ensure_local_image_path(banner_value, prefix="banner", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
    emblem_path = ensure_local_image_path(emblem_value, prefix="emblem", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
    backdrop_path = ensure_local_image_path(backdrop_value, prefix="backdrop", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
    nameplate_path = ensure_local_image_path(nameplate_value, prefix="nameplate", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
    rank_icon_path = ensure_local_image_path(rank_icon_value, prefix="rank", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)

    # Diagnostics non bloquants
    def _warn_asset(prefix: str, url: str, path: str | None) -> None:
        if not dl_enabled:
            return
        u = str(url or "").strip()
        if not u or (not u.startswith("http://") and not u.startswith("https://")):
            return
        if path:
            return
        key = f"_warned_asset_{prefix}_{hash(u)}"
        if st.session_state.get(key):
            return
        st.session_state[key] = True
        ok, err, _out = download_image_to_cache(u, prefix=prefix, timeout_seconds=12)
        if not ok:
            st.caption(f"Asset '{prefix}' non téléchargé: {err}")

    _warn_asset("backdrop", backdrop_value, backdrop_path)
    _warn_asset("rank", rank_icon_value, rank_icon_path)

    st.markdown(
        get_hero_html(
            player_name=me_name,
            service_tag=str(service_tag_value or "").strip() or None,
            rank_label=str(rank_label_value or "").strip() or None,
            rank_subtitle=str(rank_subtitle_value or "").strip() or None,
            rank_icon_path=rank_icon_path,
            banner_path=banner_path,
            backdrop_path=backdrop_path,
            nameplate_path=nameplate_path,
            id_badge_text_color=str(getattr(settings, "profile_id_badge_text_color", "") or "").strip() or None,
            emblem_path=emblem_path,
        ),
        unsafe_allow_html=True,
    )


def load_match_dataframe(db_path: str, xuid: str) -> tuple[pd.DataFrame, tuple[int, int] | None]:
    """Charge le DataFrame des matchs.
    
    Returns:
        (df, db_key)
    """
    from src.ui.perf import perf_section
    
    df = pd.DataFrame()
    db_key = db_cache_key(db_path) if db_path else None
    
    if db_path and os.path.exists(db_path) and str(xuid or "").strip():
        with perf_section("db/load_df_optimized"):
            df = load_df_optimized(db_path, xuid.strip(), db_key=db_key)
        if df.empty:
            st.warning("Aucun match trouvé.")
    else:
        st.info("Configure une DB et un joueur dans Paramètres.")

    if not df.empty:
        with perf_section("analysis/mark_firefight"):
            df = mark_firefight(df)
    
    return df, db_key
