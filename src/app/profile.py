"""Gestion du profil joueur pour l'application Streamlit.

Ce module gère :
- Le chargement et la résolution des identités joueur
- La récupération des assets de profil (banner, emblem, backdrop, etc.)
- L'affichage du header/hero du profil
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import NamedTuple

import streamlit as st

from src.config import (
    DEFAULT_PLAYER_GAMERTAG,
    DEFAULT_PLAYER_XUID,
    DEFAULT_WAYPOINT_PLAYER,
)
from src.db import resolve_xuid_from_db
from src.db.parsers import parse_xuid_input
from src.ui import (
    display_name_from_xuid,
    get_hero_html,
    get_profile_appearance,
    ensure_spnkr_tokens,
    AppSettings,
)
from src.ui.player_assets import download_image_to_cache, ensure_local_image_path


# =============================================================================
# Identity resolution
# =============================================================================


class PlayerIdentity(NamedTuple):
    """Identité d'un joueur (gamertag, xuid, waypoint)."""
    gamertag: str
    xuid: str
    waypoint_player: str


def get_identity_from_secrets() -> PlayerIdentity:
    """Retourne l'identité par défaut depuis secrets/env/constants.
    
    Ordre de priorité :
    1. Secrets Streamlit (.streamlit/secrets.toml)
    2. Variables d'environnement
    3. Constantes du projet
    
    Returns:
        PlayerIdentity avec gamertag, xuid et waypoint_player.
    """
    # Secrets Streamlit
    try:
        player = st.secrets.get("player", {})
        if isinstance(player, Mapping):
            gt = str(player.get("gamertag") or "").strip()
            xu = str(player.get("xuid") or "").strip()
            wp = str(player.get("waypoint_player") or "").strip()
        else:
            gt = xu = wp = ""
    except Exception:
        gt = xu = wp = ""

    # Env vars (utile Docker/CLI)
    gt = gt or str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip()
    xu = xu or str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip()
    wp = wp or str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip() or gt

    # Fallback constants
    gt = gt or str(DEFAULT_PLAYER_GAMERTAG or "").strip()
    xu = xu or str(DEFAULT_PLAYER_XUID or "").strip()
    wp = wp or str(DEFAULT_WAYPOINT_PLAYER or "").strip() or gt

    return PlayerIdentity(gamertag=gt or xu, xuid=xu, waypoint_player=wp)


def resolve_xuid(
    xuid_input: str,
    db_path: str,
    identity: PlayerIdentity | None = None,
) -> str:
    """Résout un XUID à partir d'une entrée utilisateur.
    
    Tente plusieurs stratégies :
    1. Parse direct si c'est un XUID valide
    2. Résolution depuis la DB si c'est un gamertag
    3. Fallback depuis l'identité par défaut si l'entrée correspond
    
    Args:
        xuid_input: Entrée utilisateur (XUID ou gamertag).
        db_path: Chemin vers la base de données.
        identity: Identité par défaut optionnelle.
        
    Returns:
        XUID résolu ou chaîne vide.
    """
    xraw = (xuid_input or "").strip()
    xuid_resolved = parse_xuid_input(xraw) or ""
    
    # Si ce n'est pas un XUID numérique, essayer de résoudre depuis la DB
    if not xuid_resolved and xraw and not xraw.isdigit() and db_path:
        xuid_resolved = resolve_xuid_from_db(db_path, xraw) or ""
        
        # Fallback: si la DB ne permet pas de résoudre,
        # utiliser les defaults si l'entrée correspond au gamertag par défaut
        if not xuid_resolved and identity:
            if (
                identity.gamertag
                and identity.xuid
                and (not str(identity.gamertag).strip().isdigit())
                and str(identity.gamertag).strip().casefold() == str(xraw).strip().casefold()
            ):
                xuid_resolved = identity.xuid
    
    # Si toujours pas de XUID et pas d'entrée, utiliser l'identité par défaut
    if not xuid_resolved and not xraw and db_path and identity:
        if identity.gamertag and not identity.gamertag.isdigit():
            xuid_resolved = resolve_xuid_from_db(db_path, identity.gamertag) or identity.xuid
        else:
            xuid_resolved = identity.gamertag or identity.xuid
    
    return xuid_resolved or ""


def propagate_identity_to_env(identity: PlayerIdentity) -> None:
    """Propage l'identité vers les variables d'environnement.
    
    Utile pour résoudre un XUID quand la DB SPNKr ne contient pas les gamertags.
    
    Args:
        identity: Identité à propager.
    """
    try:
        gt = identity.gamertag
        xuid = identity.xuid
        wp = identity.waypoint_player
        
        if gt and not str(gt).strip().isdigit() and xuid:
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_GAMERTAG"] = str(gt).strip()
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_XUID"] = str(xuid).strip()
        if wp and not str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip():
            os.environ["OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER"] = str(wp).strip()
    except Exception:
        pass


# =============================================================================
# Profile assets
# =============================================================================


def _needs_halo_auth(url: str) -> bool:
    """Vérifie si une URL nécessite une authentification Halo."""
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


class ProfileAssets(NamedTuple):
    """Assets du profil joueur."""
    banner_path: str | None
    emblem_path: str | None
    backdrop_path: str | None
    nameplate_path: str | None
    rank_icon_path: str | None
    service_tag: str | None
    rank_label: str | None
    rank_subtitle: str | None


def load_profile_assets(
    xuid: str,
    settings: AppSettings,
) -> tuple[ProfileAssets, str | None]:
    """Charge les assets du profil joueur.
    
    Combine les valeurs manuelles (settings) avec les données auto (API SPNKr).
    
    Args:
        xuid: XUID du joueur.
        settings: Paramètres de l'application.
        
    Returns:
        Tuple (ProfileAssets, erreur API ou None).
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
    
    # Valeurs manuelles (prioritaires) / sinon auto depuis API
    banner_value = str(getattr(settings, "profile_banner", "") or "").strip()
    emblem_value = (
        str(getattr(settings, "profile_emblem", "") or "").strip()
        or (getattr(api_app, "emblem_image_url", None) if api_app else "")
    )
    backdrop_value = (
        str(getattr(settings, "profile_backdrop", "") or "").strip()
        or (getattr(api_app, "backdrop_image_url", None) if api_app else "")
    )
    nameplate_value = (
        str(getattr(settings, "profile_nameplate", "") or "").strip()
        or (getattr(api_app, "nameplate_image_url", None) if api_app else "")
    )
    service_tag_value = (
        str(getattr(settings, "profile_service_tag", "") or "").strip()
        or (getattr(api_app, "service_tag", None) if api_app else "")
    )
    rank_label_value = (
        str(getattr(settings, "profile_rank_label", "") or "").strip()
        or (getattr(api_app, "rank_label", None) if api_app else "")
    )
    rank_subtitle_value = (
        str(getattr(settings, "profile_rank_subtitle", "") or "").strip()
        or (getattr(api_app, "rank_subtitle", None) if api_app else "")
    )
    rank_icon_value = (getattr(api_app, "rank_image_url", None) if api_app else "") or ""
    
    # Configuration du téléchargement
    dl_enabled = bool(getattr(settings, "profile_assets_download_enabled", False)) or bool(api_enabled)
    refresh_h = int(getattr(settings, "profile_assets_auto_refresh_hours", 0) or 0)
    
    # Authentification si nécessaire pour les assets protégés
    if dl_enabled and (not str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()):
        if _needs_halo_auth(backdrop_value) or _needs_halo_auth(rank_icon_value) or _needs_halo_auth(nameplate_value):
            ensure_spnkr_tokens(timeout_seconds=12)
    
    # Téléchargement/résolution des chemins
    banner_path = ensure_local_image_path(
        banner_value, prefix="banner", download_enabled=dl_enabled, auto_refresh_hours=refresh_h
    )
    emblem_path = ensure_local_image_path(
        emblem_value, prefix="emblem", download_enabled=dl_enabled, auto_refresh_hours=refresh_h
    )
    backdrop_path = ensure_local_image_path(
        backdrop_value, prefix="backdrop", download_enabled=dl_enabled, auto_refresh_hours=refresh_h
    )
    nameplate_path = ensure_local_image_path(
        nameplate_value, prefix="nameplate", download_enabled=dl_enabled, auto_refresh_hours=refresh_h
    )
    rank_icon_path = ensure_local_image_path(
        rank_icon_value, prefix="rank", download_enabled=dl_enabled, auto_refresh_hours=refresh_h
    )
    
    assets = ProfileAssets(
        banner_path=banner_path,
        emblem_path=emblem_path,
        backdrop_path=backdrop_path,
        nameplate_path=nameplate_path,
        rank_icon_path=rank_icon_path,
        service_tag=str(service_tag_value or "").strip() or None,
        rank_label=str(rank_label_value or "").strip() or None,
        rank_subtitle=str(rank_subtitle_value or "").strip() or None,
    )
    
    return assets, api_err


def warn_missing_assets(
    settings: AppSettings,
    backdrop_value: str,
    backdrop_path: str | None,
    rank_icon_value: str,
    rank_icon_path: str | None,
) -> None:
    """Affiche des avertissements pour les assets non téléchargés.
    
    Args:
        settings: Paramètres de l'application.
        backdrop_value: URL du backdrop.
        backdrop_path: Chemin local du backdrop ou None.
        rank_icon_value: URL de l'icône de rang.
        rank_icon_path: Chemin local de l'icône ou None.
    """
    dl_enabled = bool(getattr(settings, "profile_assets_download_enabled", False))
    api_enabled = bool(getattr(settings, "profile_api_enabled", False))
    dl_enabled = dl_enabled or api_enabled
    
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


# =============================================================================
# Profile header rendering
# =============================================================================


def render_profile_header(
    xuid: str,
    settings: AppSettings,
    assets: ProfileAssets,
) -> None:
    """Rend le header/hero du profil joueur.
    
    Args:
        xuid: XUID du joueur.
        settings: Paramètres de l'application.
        assets: Assets du profil.
    """
    me_name = display_name_from_xuid(xuid.strip()) if str(xuid or "").strip() else "(joueur)"
    
    st.markdown(
        get_hero_html(
            player_name=me_name,
            service_tag=assets.service_tag,
            rank_label=assets.rank_label,
            rank_subtitle=assets.rank_subtitle,
            rank_icon_path=assets.rank_icon_path,
            banner_path=assets.banner_path,
            backdrop_path=assets.backdrop_path,
            nameplate_path=assets.nameplate_path,
            id_badge_text_color=str(getattr(settings, "profile_id_badge_text_color", "") or "").strip() or None,
            emblem_path=assets.emblem_path,
        ),
        unsafe_allow_html=True,
    )
