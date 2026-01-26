"""LevelUp - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données SPNKr.
"""

import os
import urllib.parse
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

# Imports depuis la nouvelle architecture
from src.config import (
    get_default_db_path,
)
from src.analysis import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
)
from src.analysis.stats import format_mmss
from src.visualization import (
    plot_multi_metric_bars_by_match,
)
from src.ui import (
    display_name_from_xuid,
    load_css,
    translate_playlist_name,
    translate_pair_name,
    AppSettings,
    load_settings,
)
from src.ui.formatting import (
    format_duration_hms,
    format_duration_dhm,
    format_datetime_fr_hm,
    format_score_label,
    score_css_color,
    PARIS_TZ,
)
from src.config import get_aliases_file_path

from src.ui.perf import perf_reset_run, perf_section
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
from src.ui.components import (
    render_kpi_cards,
    render_top_summary,
    render_checkbox_filter,
    render_hierarchical_checkbox_filter,
    get_firefight_playlists,
)
from src.analysis.performance_config import PERFORMANCE_SCORE_FULL_DESC
from src.ui.cache import (
    db_cache_key,
    cached_list_local_dbs,
    cached_compute_sessions_db,
    cached_same_team_match_ids_with_friend,
    cached_load_player_match_result,
    cached_load_match_medals_for_player,
    cached_load_match_rosters,
    cached_load_highlight_events_for_match,
    cached_load_match_player_gamertags,
    top_medals_smart,
    clear_app_caches,
)
from src.ui.sync import (
    is_spnkr_db_path,
    cleanup_orphan_tmp_dbs,
    render_sync_indicator,
    sync_all_players,
)
# Phase 1 refactoring: Import des nouveaux modules app
# Phase 2 refactoring: Helpers et fonctions extraites
from src.app.helpers import (
    clean_asset_label,
    normalize_mode_label,
    normalize_map_label,
    assign_player_colors,
    date_range,
    styler_map,
    compute_session_span_seconds,
    compute_total_play_seconds,
    avg_match_duration_seconds,
)
from src.app.filters import (
    build_friends_opts_map,
)
from src.app.data_loader import (
    default_identity_from_secrets,
    init_source_state,
    ensure_h5g_commendations_repo,
)
# Phase 4 refactoring: Main helpers
from src.app.main_helpers import (
    propagate_identity_to_env,
    apply_settings_path_overrides as apply_settings_overrides_main,
    validate_and_fix_db_path,
    resolve_xuid_from_input,
    load_profile_api,
    render_profile_hero,
    load_match_dataframe,
)
# Phase 4 refactoring: Page router
from src.app.page_router import (
    build_match_view_params,
    consume_pending_page,
    consume_pending_match_id,
    render_page_selector,
    dispatch_page,
)
# Phase 5 refactoring: KPIs et Filtres
from src.app.kpis_render import (
    render_kpis_section,
    render_performance_info,
)
from src.app.filters_render import (
    FilterState,
    render_filters_sidebar,
    apply_filters,
)
from src.ui.multiplayer import (
    render_player_selector,
)


# =============================================================================
# Aliases vers les fonctions extraites (Phase 2)
# =============================================================================
_default_identity_from_secrets = default_identity_from_secrets
_init_source_state = init_source_state
_ensure_h5g_commendations_repo = ensure_h5g_commendations_repo
_clean_asset_label = clean_asset_label
_normalize_mode_label = normalize_mode_label
_normalize_map_label = normalize_map_label
_styler_map = styler_map
_assign_player_colors = assign_player_colors
_compute_session_span_seconds = compute_session_span_seconds
_compute_total_play_seconds = compute_total_play_seconds
_avg_match_duration_seconds = avg_match_duration_seconds
_date_range = date_range
_build_friends_opts_map = build_friends_opts_map


def _qp_first(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    s = str(value)
    return s if s.strip() else None


def _set_query_params(**kwargs: str) -> None:
    clean: dict[str, str] = {k: str(v) for k, v in kwargs.items() if v is not None and str(v).strip()}
    try:
        st.query_params.clear()
        for k, v in clean.items():
            st.query_params[k] = v
    except Exception:
        # Fallback API legacy (compat)
        try:
            st.experimental_set_query_params(**clean)
        except Exception:
            pass


def _app_url(page: str, **params: str) -> str:
    qp: dict[str, str] = {"page": page}
    for k, v in params.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            qp[k] = s
    return "?" + urllib.parse.urlencode(qp)


def _clear_min_matches_maps_auto() -> None:
    st.session_state["_min_matches_maps_auto"] = False


def _clear_min_matches_maps_friends_auto() -> None:
    st.session_state["_min_matches_maps_friends_auto"] = False


# Alias pour les fonctions déplacées vers cache.py
_db_cache_key = db_cache_key
_top_medals = top_medals_smart
_clear_app_caches = clear_app_caches


def _aliases_cache_key() -> int | None:
    try:
        p = get_aliases_file_path()
        st_ = os.stat(p)
        return int(getattr(st_, "st_mtime_ns", int(st_.st_mtime * 1e9)))
    except OSError:
        return None


# =============================================================================
# Application principale
# =============================================================================

def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    st.set_page_config(page_title="LevelUp", layout="wide")

    perf_reset_run()

    # Nettoyage des fichiers temporaires orphelins (une fois par session)
    cleanup_orphan_tmp_dbs()

    with perf_section("css"):
        st.markdown(load_css(), unsafe_allow_html=True)

    # IMPORTANT: aucun accès réseau implicite.
    # La génération du référentiel Citations doit être explicite (opt-in via env).
    if str(os.environ.get("OPENSPARTAN_CITATIONS_AUTOGEN") or "").strip() in {"1", "true", "True"}:
        _ensure_h5g_commendations_repo()

    # Paramètres (persistés)
    settings: AppSettings = load_settings()
    st.session_state["app_settings"] = settings

    # Propage les defaults depuis secrets vers l'env et applique les overrides de chemins
    propagate_identity_to_env()
    apply_settings_overrides_main(settings)

    # ==========================================================================
    # Source (persistée via session_state) — UI dans l'onglet Paramètres
    # ==========================================================================

    DEFAULT_DB = get_default_db_path()
    _init_source_state(DEFAULT_DB, settings)

    # Support liens internes via query params (?page=...&match_id=...)
    try:
        qp = dict(st.query_params)
        qp_page = _qp_first(qp.get("page"))
        qp_mid = _qp_first(qp.get("match_id"))
    except Exception:
        qp_page = None
        qp_mid = None
    qp_token = (str(qp_page or "").strip(), str(qp_mid or "").strip())
    if any(qp_token) and st.session_state.get("_consumed_query_params") != qp_token:
        st.session_state["_consumed_query_params"] = qp_token
        if qp_token[0]:
            st.session_state["_pending_page"] = qp_token[0]
        if qp_token[1]:
            st.session_state["_pending_match_id"] = qp_token[1]
        # Nettoie l'URL après consommation pour ne pas forcer la page en boucle.
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

    db_path = str(st.session_state.get("db_path", "") or "").strip()
    xuid = str(st.session_state.get("xuid_input", "") or "").strip()
    waypoint_player = str(st.session_state.get("waypoint_player", "") or "").strip()

    with st.sidebar:
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
                # (les valeurs de l'ancien joueur peuvent ne pas exister pour le nouveau)
                for filter_key in ["filter_playlists", "filter_modes", "filter_maps"]:
                    if filter_key in st.session_state:
                        del st.session_state[filter_key]
                st.rerun()

        # Bouton Sync pour toutes les DB SPNKr (multi-joueurs si DB fusionnée)
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
                    _clear_app_caches()
                    st.rerun()
                else:
                    st.error(msg)

    # Validation du chemin DB
    db_path = validate_and_fix_db_path(db_path, DEFAULT_DB)

    # Résolution du XUID
    xuid = resolve_xuid_from_input(xuid, db_path)

    me_name = display_name_from_xuid(xuid.strip()) if str(xuid or "").strip() else "(joueur)"
    aliases_key = _aliases_cache_key()

    # Auto-profil (SPNKr) et rendu du hero
    api_app, _api_err = load_profile_api(xuid, settings)
    render_profile_hero(xuid, settings, api_app)

    # ==========================================================================
    # Chargement des données
    # ==========================================================================
    
    df, db_key = load_match_dataframe(db_path, xuid)

    if df.empty:
        st.radio(
            "Navigation",
            options=["Paramètres"],
            horizontal=True,
            key="page",
            label_visibility="collapsed",
        )
        render_settings_page(
            settings,
            get_local_dbs_fn=cached_list_local_dbs,
            on_clear_caches_fn=_clear_app_caches,
        )
        return

    # ==========================================================================
    # Sidebar - Filtres
    # ==========================================================================
    
    with st.sidebar:
        filter_state = render_filters_sidebar(
            df=df,
            db_path=db_path,
            xuid=xuid,
            db_key=db_key,
            aliases_key=aliases_key,
            date_range_fn=_date_range,
            clean_asset_label_fn=_clean_asset_label,
            normalize_mode_label_fn=_normalize_mode_label,
            normalize_map_label_fn=_normalize_map_label,
            build_friends_opts_map_fn=_build_friends_opts_map,
        )

    # Base "globale" : toutes les parties (après inclusion/exclusion Firefight)
    base = df.copy()

    # ==========================================================================
    # Application des filtres
    # ==========================================================================
    
    dff = apply_filters(
        dff=df,
        filter_state=filter_state,
        db_path=db_path,
        xuid=xuid,
        db_key=db_key,
        clean_asset_label_fn=_clean_asset_label,
        normalize_mode_label_fn=_normalize_mode_label,
        normalize_map_label_fn=_normalize_map_label,
    )

    # Variables pour compatibilité avec le dispatch
    gap_minutes = filter_state.gap_minutes
    picked_session_labels = filter_state.picked_session_labels

    # ==========================================================================
    # KPIs
    # ==========================================================================
    
    render_kpis_section(dff)
    render_performance_info()

    # ==========================================================================
    # Pages (navigation)
    # ==========================================================================

    consume_pending_page()
    consume_pending_match_id()
    page = render_page_selector()

    # Paramètres communs pour les pages de match
    _match_view_params = build_match_view_params(
        db_path=db_path,
        xuid=xuid,
        waypoint_player=waypoint_player,
        db_key=db_key,
        settings=settings,
        df_full=df,
        render_match_view_fn=render_match_view,
        normalize_mode_label_fn=_normalize_mode_label,
        format_score_label_fn=format_score_label,
        score_css_color_fn=score_css_color,
        format_datetime_fn=format_datetime_fr_hm,
        load_player_match_result_fn=cached_load_player_match_result,
        load_match_medals_fn=cached_load_match_medals_for_player,
        load_highlight_events_fn=cached_load_highlight_events_for_match,
        load_match_gamertags_fn=cached_load_match_player_gamertags,
        load_match_rosters_fn=cached_load_match_rosters,
        paris_tz=PARIS_TZ,
    )

    # Dispatch vers la page appropriée
    dispatch_page(
        page=page,
        dff=dff,
        df=df,
        base=base,
        me_name=me_name,
        xuid=xuid,
        db_path=db_path,
        db_key=db_key,
        aliases_key=aliases_key,
        settings=settings,
        picked_session_labels=picked_session_labels,
        waypoint_player=waypoint_player,
        gap_minutes=gap_minutes,
        match_view_params=_match_view_params,
        # Fonctions de rendu
        render_last_match_page_fn=render_last_match_page,
        render_match_search_page_fn=render_match_search_page,
        render_citations_page_fn=render_citations_page,
        render_session_comparison_page_fn=render_session_comparison_page,
        render_timeseries_page_fn=render_timeseries_page,
        render_win_loss_page_fn=render_win_loss_page,
        render_teammates_page_fn=render_teammates_page,
        render_match_history_page_fn=render_match_history_page,
        render_settings_page_fn=render_settings_page,
        # Fonctions utilitaires
        cached_compute_sessions_db_fn=cached_compute_sessions_db,
        top_medals_fn=_top_medals,
        build_friends_opts_map_fn=_build_friends_opts_map,
        assign_player_colors_fn=_assign_player_colors,
        plot_multi_metric_bars_fn=plot_multi_metric_bars_by_match,
        get_local_dbs_fn=cached_list_local_dbs,
        clear_caches_fn=_clear_app_caches,
    )


if __name__ == "__main__":
    main()
