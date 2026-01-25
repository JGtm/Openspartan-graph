"""Page Paramètres (Settings)."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from src.config import get_default_db_path
from src.ui import (
    AppSettings,
    load_settings,
    save_settings,
    directory_input,
)
from src.ui.sections import render_source_section


def render_settings_page(
    settings: AppSettings,
    *,
    get_local_dbs_fn: Callable[[], list[str]],
    on_clear_caches_fn: Callable[[], None],
) -> AppSettings:
    """Rend l'onglet Paramètres et retourne les settings (potentiellement modifiés).

    Parameters
    ----------
    settings : AppSettings
        Paramètres actuels de l'application.
    get_local_dbs_fn : Callable[[], list[str]]
        Fonction pour lister les bases de données locales.
    on_clear_caches_fn : Callable[[], None]
        Fonction pour vider les caches de l'application.

    Returns
    -------
    AppSettings
        Paramètres (modifiés ou non).
    """
    st.subheader("Paramètres")

    with st.expander("Source", expanded=True):
        default_db = get_default_db_path()
        render_source_section(
            default_db,
            get_local_dbs=get_local_dbs_fn,
            on_clear_caches=on_clear_caches_fn,
        )

    with st.expander("SPNKr API", expanded=True):
        st.caption("Optionnel: recharge les derniers matchs via l'API et met à jour la DB SPNKr.")
        prefer_spnkr = bool(getattr(settings, "prefer_spnkr_db_if_available", True))
        spnkr_on_start = st.toggle(
            "Rafraîchir la DB au démarrage",
            value=bool(getattr(settings, "spnkr_refresh_on_start", True)),
        )
        spnkr_on_refresh = st.toggle(
            "Le bouton Actualiser rafraîchit aussi la DB",
            value=bool(getattr(settings, "spnkr_refresh_on_manual_refresh", True)),
        )
        mt = st.selectbox(
            "Type de matchs",
            options=["matchmaking", "all", "custom", "local"],
            index=["matchmaking", "all", "custom", "local"].index(
                str(getattr(settings, "spnkr_refresh_match_type", "matchmaking") or "matchmaking").strip().lower()
                if str(getattr(settings, "spnkr_refresh_match_type", "matchmaking") or "matchmaking").strip().lower()
                in {"matchmaking", "all", "custom", "local"}
                else "matchmaking"
            ),
        )
        max_matches = st.number_input(
            "Max matchs (refresh)",
            min_value=10,
            max_value=5000,
            value=int(getattr(settings, "spnkr_refresh_max_matches", 200) or 200),
            step=10,
        )
        rps = st.number_input(
            "Requêtes / seconde",
            min_value=1,
            max_value=20,
            value=int(getattr(settings, "spnkr_refresh_rps", 3) or 3),
            step=1,
        )
        with_he = st.toggle(
            "Inclure highlight events",
            value=bool(getattr(settings, "spnkr_refresh_with_highlight_events", False)),
        )

    with st.expander("Médias", expanded=True):
        media_enabled = st.toggle("Activer la section Médias", value=bool(settings.media_enabled))
        media_screens_dir = directory_input(
            "Dossier captures (images)",
            value=str(settings.media_screens_dir or ""),
            key="settings_media_screens_dir",
            help="Chemin vers un dossier contenant des captures (png/jpg/webp).",
            placeholder="Ex: C:\\Users\\Guillaume\\Pictures\\Halo",
        )
        media_videos_dir = directory_input(
            "Dossier vidéos",
            value=str(settings.media_videos_dir or ""),
            key="settings_media_videos_dir",
            help="Chemin vers un dossier contenant des vidéos (mp4/webm/mkv).",
            placeholder="Ex: C:\\Users\\Guillaume\\Videos",
        )
        media_tolerance_minutes = st.slider(
            "Tolérance (minutes) autour du match",
            min_value=0,
            max_value=30,
            value=int(settings.media_tolerance_minutes or 0),
            step=1,
        )

    with st.expander("Expérience", expanded=True):
        refresh_clears_caches = st.toggle(
            "Le bouton Actualiser vide aussi les caches",
            value=bool(getattr(settings, "refresh_clears_caches", False)),
            help="Utile si la DB change en dehors de l'app (NAS / import externe).",
        )

    # Section "Fichiers (avancé)" masquée - valeurs conservées depuis settings
    aliases_path = str(getattr(settings, "aliases_path", "") or "").strip()
    profiles_path = str(getattr(settings, "profiles_path", "") or "").strip()

    # Profil joueur (bannière / rang)
    # Par défaut, on masque ces réglages et on garde les valeurs actuelles.
    profile_assets_download_enabled = bool(getattr(settings, "profile_assets_download_enabled", False))
    profile_assets_auto_refresh_hours = int(getattr(settings, "profile_assets_auto_refresh_hours", 24) or 0)
    profile_api_enabled = bool(getattr(settings, "profile_api_enabled", False))
    profile_api_auto_refresh_hours = int(getattr(settings, "profile_api_auto_refresh_hours", 6) or 0)
    profile_banner = str(getattr(settings, "profile_banner", "") or "").strip()
    profile_emblem = str(getattr(settings, "profile_emblem", "") or "").strip()
    profile_backdrop = str(getattr(settings, "profile_backdrop", "") or "").strip()
    profile_nameplate = str(getattr(settings, "profile_nameplate", "") or "").strip()
    profile_service_tag = str(getattr(settings, "profile_service_tag", "") or "").strip()
    profile_id_badge_text_color = str(getattr(settings, "profile_id_badge_text_color", "") or "").strip()
    profile_rank_label = str(getattr(settings, "profile_rank_label", "") or "").strip()
    profile_rank_subtitle = str(getattr(settings, "profile_rank_subtitle", "") or "").strip()

    # Section "Profil joueur (avancé)" masquée - valeurs conservées depuis settings

    cols = st.columns(2)
    if cols[0].button("Enregistrer", width="stretch"):
        new_settings = AppSettings(
            media_enabled=bool(media_enabled),
            media_screens_dir=str(media_screens_dir or "").strip(),
            media_videos_dir=str(media_videos_dir or "").strip(),
            media_tolerance_minutes=int(media_tolerance_minutes),
            refresh_clears_caches=bool(refresh_clears_caches),
            prefer_spnkr_db_if_available=bool(prefer_spnkr),
            spnkr_refresh_on_start=bool(spnkr_on_start),
            spnkr_refresh_on_manual_refresh=bool(spnkr_on_refresh),
            spnkr_refresh_match_type=str(mt),
            spnkr_refresh_max_matches=int(max_matches),
            spnkr_refresh_rps=int(rps),
            spnkr_refresh_with_highlight_events=bool(with_he),
            aliases_path=str(aliases_path or "").strip(),
            profiles_path=str(profiles_path or "").strip(),
            profile_assets_download_enabled=bool(profile_assets_download_enabled),
            profile_assets_auto_refresh_hours=int(profile_assets_auto_refresh_hours),
            profile_api_enabled=bool(profile_api_enabled),
            profile_api_auto_refresh_hours=int(profile_api_auto_refresh_hours),
            profile_banner=str(profile_banner or "").strip(),
            profile_emblem=str(profile_emblem or "").strip(),
            profile_backdrop=str(profile_backdrop or "").strip(),
            profile_nameplate=str(profile_nameplate or "").strip(),
            profile_service_tag=str(profile_service_tag or "").strip(),
            profile_id_badge_text_color=str(profile_id_badge_text_color or "").strip(),
            profile_rank_label=str(profile_rank_label or "").strip(),
            profile_rank_subtitle=str(profile_rank_subtitle or "").strip(),
        )
        ok, err = save_settings(new_settings)
        if ok:
            st.success("Paramètres enregistrés.")
            st.session_state["app_settings"] = new_settings
            st.rerun()
        else:
            st.error(err)
        return new_settings

    if cols[1].button("Recharger depuis fichier", width="stretch"):
        reloaded = load_settings()
        st.session_state["app_settings"] = reloaded
        st.rerun()
        return reloaded

    return settings
