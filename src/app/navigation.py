"""Navigation et rendu des pages pour l'application Streamlit.

Ce module gère :
- La liste des pages disponibles
- Le rendu de la page active
- Les paramètres communs pour les pages
"""

from __future__ import annotations

from typing import Callable, Any

import pandas as pd
import streamlit as st

from src.ui import AppSettings
from src.ui.formatting import (
    format_score_label,
    score_css_color,
    format_datetime_fr_hm,
    PARIS_TZ,
)
from src.ui.cache import (
    cached_compute_sessions_db,
    cached_load_player_match_result,
    cached_load_match_medals_for_player,
    cached_load_match_rosters,
    cached_load_highlight_events_for_match,
    cached_load_match_player_gamertags,
    top_medals_smart,
)
from src.ui.pages import (
    render_session_comparison_page,
    render_timeseries_page,
    render_win_loss_page,
    render_match_history_page,
    render_teammates_page,
    render_citations_page,
    render_settings_page,
    render_match_view,
    render_last_match_page,
    render_match_search_page,
)
from src.visualization import plot_multi_metric_bars_by_match


# =============================================================================
# Liste des pages
# =============================================================================

PAGES = [
    "Séries temporelles",
    "Comparaison de sessions",
    "Dernier match",
    "Match",
    "Citations",
    "Victoires/Défaites",
    "Mes coéquipiers",
    "Historique des parties",
    "Paramètres",
]


# =============================================================================
# Paramètres communs pour les pages de match
# =============================================================================


def get_match_view_params(
    *,
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: tuple[int, int] | None,
    settings: AppSettings,
    df_full: pd.DataFrame,
    normalize_mode_label_fn: Callable[[str | None], str],
) -> dict[str, Any]:
    """Retourne les paramètres communs pour les pages de match.
    
    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        waypoint_player: Gamertag Waypoint.
        db_key: Clé de cache DB.
        settings: Paramètres de l'application.
        df_full: DataFrame complet pour le score relatif.
        normalize_mode_label_fn: Fonction de normalisation des modes.
        
    Returns:
        Dictionnaire des paramètres.
    """
    return {
        "db_path": db_path,
        "xuid": xuid,
        "waypoint_player": waypoint_player,
        "db_key": db_key,
        "settings": settings,
        "df_full": df_full,
        "render_match_view_fn": render_match_view,
        "normalize_mode_label_fn": normalize_mode_label_fn,
        "format_score_label_fn": format_score_label,
        "score_css_color_fn": score_css_color,
        "format_datetime_fn": format_datetime_fr_hm,
        "load_player_match_result_fn": cached_load_player_match_result,
        "load_match_medals_fn": cached_load_match_medals_for_player,
        "load_highlight_events_fn": cached_load_highlight_events_for_match,
        "load_match_gamertags_fn": cached_load_match_player_gamertags,
        "load_match_rosters_fn": cached_load_match_rosters,
        "paris_tz": PARIS_TZ,
    }


# =============================================================================
# Gestion de l'état de navigation
# =============================================================================


def consume_pending_navigation() -> None:
    """Consomme les changements de navigation en attente."""
    pending_page = st.session_state.pop("_pending_page", None)
    if isinstance(pending_page, str) and pending_page in PAGES:
        st.session_state["page"] = pending_page
    if "page" not in st.session_state:
        st.session_state["page"] = "Séries temporelles"

    pending_mid = st.session_state.pop("_pending_match_id", None)
    if isinstance(pending_mid, str) and pending_mid.strip():
        st.session_state["match_id_input"] = pending_mid.strip()


def render_page_navigation() -> str:
    """Rend le sélecteur de pages et retourne la page active.
    
    Returns:
        Nom de la page active.
    """
    return st.segmented_control(
        "Onglets",
        options=PAGES,
        key="page",
        label_visibility="collapsed",
    )


# =============================================================================
# Rendu des pages
# =============================================================================


def render_active_page(
    *,
    page: str,
    df: pd.DataFrame,
    dff: pd.DataFrame,
    base: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
    settings: AppSettings,
    waypoint_player: str,
    picked_session_labels: list[str] | None,
    gap_minutes: int,
    match_view_params: dict[str, Any],
    build_friends_opts_map_fn: Callable,
    assign_player_colors_fn: Callable,
    clear_caches_fn: Callable,
    get_local_dbs_fn: Callable,
) -> None:
    """Rend la page active.
    
    Args:
        page: Nom de la page à rendre.
        df: DataFrame complet.
        dff: DataFrame filtré.
        base: DataFrame de base.
        me_name: Nom du joueur.
        xuid: XUID du joueur.
        db_path: Chemin vers la base de données.
        db_key: Clé de cache DB.
        aliases_key: Clé de cache des alias.
        settings: Paramètres de l'application.
        waypoint_player: Gamertag Waypoint.
        picked_session_labels: Sessions sélectionnées.
        gap_minutes: Écart entre sessions.
        match_view_params: Paramètres pour les pages de match.
        build_friends_opts_map_fn: Fonction pour construire les options d'amis.
        assign_player_colors_fn: Fonction pour assigner les couleurs.
        clear_caches_fn: Fonction pour vider les caches.
        get_local_dbs_fn: Fonction pour lister les DB locales.
    """
    # --------------------------------------------------------------------------
    # Page: Dernier match
    # --------------------------------------------------------------------------
    if page == "Dernier match":
        render_last_match_page(dff=dff, **match_view_params)

    # --------------------------------------------------------------------------
    # Page: Match (recherche)
    # --------------------------------------------------------------------------
    elif page == "Match":
        render_match_search_page(df=df, dff=dff, **match_view_params)

    # --------------------------------------------------------------------------
    # Page: Citations
    # --------------------------------------------------------------------------
    elif page == "Citations":
        render_citations_page(
            dff=dff,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            top_medals_fn=top_medals_smart,
        )

    # --------------------------------------------------------------------------
    # Page: Comparaison de sessions
    # --------------------------------------------------------------------------
    elif page == "Comparaison de sessions":
        all_sessions_df = cached_compute_sessions_db(
            db_path, xuid.strip(), db_key, True, gap_minutes
        )
        render_session_comparison_page(all_sessions_df, df_full=df)

    # --------------------------------------------------------------------------
    # Page: Séries temporelles
    # --------------------------------------------------------------------------
    elif page == "Séries temporelles":
        render_timeseries_page(dff, df_full=df)

    # --------------------------------------------------------------------------
    # Page: Victoires/Défaites
    # --------------------------------------------------------------------------
    elif page == "Victoires/Défaites":
        render_win_loss_page(
            dff=dff,
            base=base,
            picked_session_labels=picked_session_labels,
            db_path=db_path,
            xuid=xuid,
            db_key=db_key,
        )

    # --------------------------------------------------------------------------
    # Page: Mes coéquipiers
    # --------------------------------------------------------------------------
    elif page == "Mes coéquipiers":
        render_teammates_page(
            df=df,
            dff=dff,
            base=base,
            me_name=me_name,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            aliases_key=aliases_key,
            settings=settings,
            picked_session_labels=picked_session_labels,
            include_firefight=True,
            waypoint_player=waypoint_player,
            build_friends_opts_map_fn=build_friends_opts_map_fn,
            assign_player_colors_fn=assign_player_colors_fn,
            plot_multi_metric_bars_fn=plot_multi_metric_bars_by_match,
            top_medals_fn=top_medals_smart,
        )

    # --------------------------------------------------------------------------
    # Page: Historique des parties
    # --------------------------------------------------------------------------
    elif page == "Historique des parties":
        render_match_history_page(
            dff=dff,
            waypoint_player=waypoint_player,
            db_path=db_path,
            xuid=xuid,
            db_key=db_key,
            df_full=df,
        )

    # --------------------------------------------------------------------------
    # Page: Paramètres
    # --------------------------------------------------------------------------
    elif page == "Paramètres":
        render_settings_page(
            settings,
            get_local_dbs_fn=get_local_dbs_fn,
            on_clear_caches_fn=clear_caches_fn,
        )
