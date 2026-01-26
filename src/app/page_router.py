"""Routage des pages extraits de main() pour simplification.

Ce module centralise:
- La liste des pages disponibles
- La construction des paramètres pour les pages de match
- Le dispatch vers les différentes pages
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import streamlit as st

from src.ui.settings import AppSettings


# Liste des pages disponibles
PAGES: list[str] = [
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


def build_match_view_params(
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: str | None,
    settings: AppSettings,
    df_full: pd.DataFrame,
    render_match_view_fn: Callable,
    normalize_mode_label_fn: Callable,
    format_score_label_fn: Callable,
    score_css_color_fn: Callable,
    format_datetime_fn: Callable,
    load_player_match_result_fn: Callable,
    load_match_medals_fn: Callable,
    load_highlight_events_fn: Callable,
    load_match_gamertags_fn: Callable,
    load_match_rosters_fn: Callable,
    paris_tz,
) -> dict:
    """Construit les paramètres communs pour les pages de match."""
    return dict(
        db_path=db_path,
        xuid=xuid,
        waypoint_player=waypoint_player,
        db_key=db_key,
        settings=settings,
        df_full=df_full,
        render_match_view_fn=render_match_view_fn,
        normalize_mode_label_fn=normalize_mode_label_fn,
        format_score_label_fn=format_score_label_fn,
        score_css_color_fn=score_css_color_fn,
        format_datetime_fn=format_datetime_fn,
        load_player_match_result_fn=load_player_match_result_fn,
        load_match_medals_fn=load_match_medals_fn,
        load_highlight_events_fn=load_highlight_events_fn,
        load_match_gamertags_fn=load_match_gamertags_fn,
        load_match_rosters_fn=load_match_rosters_fn,
        paris_tz=paris_tz,
    )


def consume_pending_page() -> None:
    """Consomme la page en attente si définie."""
    pending_page = st.session_state.pop("_pending_page", None)
    if isinstance(pending_page, str) and pending_page in PAGES:
        st.session_state["page"] = pending_page
    if "page" not in st.session_state:
        st.session_state["page"] = "Séries temporelles"


def consume_pending_match_id() -> None:
    """Consomme le match_id en attente si défini."""
    pending_mid = st.session_state.pop("_pending_match_id", None)
    if isinstance(pending_mid, str) and pending_mid.strip():
        st.session_state["match_id_input"] = pending_mid.strip()


def render_page_selector() -> str:
    """Rend le sélecteur de page et retourne la page choisie."""
    return st.segmented_control(
        "Onglets",
        options=PAGES,
        key="page",
        label_visibility="collapsed",
    )


def dispatch_page(
    page: str,
    dff: pd.DataFrame,
    df: pd.DataFrame,
    base: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: str | None,
    aliases_key: str | None,
    settings: AppSettings,
    picked_session_labels: Optional[list[str]],
    waypoint_player: str,
    gap_minutes: int,
    match_view_params: dict,
    # Fonctions de rendu
    render_last_match_page_fn: Callable,
    render_match_search_page_fn: Callable,
    render_citations_page_fn: Callable,
    render_session_comparison_page_fn: Callable,
    render_timeseries_page_fn: Callable,
    render_win_loss_page_fn: Callable,
    render_teammates_page_fn: Callable,
    render_match_history_page_fn: Callable,
    render_settings_page_fn: Callable,
    # Fonctions utilitaires
    cached_compute_sessions_db_fn: Callable,
    top_medals_fn: Callable,
    build_friends_opts_map_fn: Callable,
    assign_player_colors_fn: Callable,
    plot_multi_metric_bars_fn: Callable,
    get_local_dbs_fn: Callable,
    clear_caches_fn: Callable,
) -> None:
    """Dispatch vers la page appropriée."""
    if page == "Dernier match":
        render_last_match_page_fn(dff=dff, **match_view_params)

    elif page == "Match":
        render_match_search_page_fn(df=df, dff=dff, **match_view_params)

    elif page == "Citations":
        render_citations_page_fn(
            dff=dff,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            top_medals_fn=top_medals_fn,
        )

    elif page == "Comparaison de sessions":
        all_sessions_df = cached_compute_sessions_db_fn(
            db_path, xuid.strip(), db_key, True, gap_minutes
        )
        render_session_comparison_page_fn(all_sessions_df, df_full=df)

    elif page == "Séries temporelles":
        render_timeseries_page_fn(dff, df_full=df)

    elif page == "Victoires/Défaites":
        render_win_loss_page_fn(
            dff=dff,
            base=base,
            picked_session_labels=picked_session_labels,
            db_path=db_path,
            xuid=xuid,
            db_key=db_key,
        )

    elif page == "Mes coéquipiers":
        render_teammates_page_fn(
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
            plot_multi_metric_bars_fn=plot_multi_metric_bars_fn,
            top_medals_fn=top_medals_fn,
        )

    elif page == "Historique des parties":
        render_match_history_page_fn(
            dff=dff,
            waypoint_player=waypoint_player,
            db_path=db_path,
            xuid=xuid,
            db_key=db_key,
            df_full=df,
        )

    elif page == "Paramètres":
        render_settings_page_fn(
            settings,
            get_local_dbs_fn=get_local_dbs_fn,
            on_clear_caches_fn=clear_caches_fn,
        )
