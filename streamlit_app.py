"""LevelUp - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données SPNKr.
"""

import os
import json
import html
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone
from collections.abc import Mapping
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Imports depuis la nouvelle architecture
from src.config import (
    get_default_db_path,
    get_default_workshop_exe_path,
    get_repo_root,
    DEFAULT_WAYPOINT_PLAYER,
    HALO_COLORS,
    SESSION_CONFIG,
    OUTCOME_CODES,
    BOT_MAP,
    TEAM_MAP,
)
from src.db import (
    load_matches,
    guess_xuid_from_db_path,
    has_table,
    get_sync_metadata,
)
from src.db.parsers import parse_xuid_input
from src.db import infer_spnkr_player_from_db_path
from src.analysis import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
    compute_sessions,
    compute_map_breakdown,
    mark_firefight,
    is_allowed_playlist_name,
    build_option_map,
    build_xuid_option_map,
    compute_killer_victim_pairs,
    killer_victim_counts_long,
    killer_victim_matrix,
)
from src.analysis.stats import format_selected_matches_summary, format_mmss
from src.visualization import (
    plot_timeseries,
    plot_assists_timeseries,
    plot_per_minute_timeseries,
    plot_average_life,
    plot_spree_headshots_accuracy,
    plot_kda_distribution,
    plot_outcomes_over_time,
    plot_map_comparison,
    plot_map_ratio_with_winloss,
    plot_trio_metric,
    plot_metric_bars_by_match,
    plot_multi_metric_bars_by_match,
)
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom
from src.ui import (
    load_aliases_file,
    save_aliases_file,
    get_xuid_aliases,
    display_name_from_xuid,
    load_css,
    get_hero_html,
    translate_playlist_name,
    translate_pair_name,
    AppSettings,
    load_settings,
    save_settings,
    directory_input,
    file_input,
    get_profile_appearance,
    ensure_spnkr_tokens,
)
from src.ui.player_assets import download_image_to_cache, ensure_local_image_path
from src.ui.medals import (
    load_medal_name_maps,
    medal_has_known_label,
    get_medals_cache_dir,
    medal_label,
    medal_icon_path,
    render_medals_grid,
)
from src.ui.formatting import (
    format_date_fr,
    format_duration_hms,
    format_duration_dhm,
    format_datetime_fr_hm,
    format_score_label,
    score_css_color,
    style_outcome_text,
    style_signed_number,
    style_score_label,
    parse_date_fr_input,
    coerce_int,
    to_paris_naive,
    paris_epoch_seconds,
    PARIS_TZ,
    PARIS_TZ_NAME,
)
from src.db.profiles import (
    PROFILES_PATH,
    load_profiles,
    save_profiles,
    list_local_dbs,
)
from src.config import DEFAULT_PLAYER_GAMERTAG, DEFAULT_PLAYER_XUID, get_aliases_file_path

from src.ui.perf import perf_reset_run, perf_section, render_perf_panel
from src.ui.sections import render_openspartan_tools, render_source_section
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
    compute_session_performance_score,
    render_kpi_cards,
    render_top_summary,
    render_checkbox_filter,
    render_hierarchical_checkbox_filter,
    get_firefight_playlists,
)
from src.ui.cache import (
    load_df,
    db_cache_key,
    cached_list_local_dbs,
    cached_compute_sessions_db,
    cached_same_team_match_ids_with_friend,
    cached_query_matches_with_friend,
    cached_load_player_match_result,
    cached_load_match_medals_for_player,
    cached_load_match_rosters,
    cached_load_highlight_events_for_match,
    cached_load_match_player_gamertags,
    cached_load_top_medals,
    top_medals_smart,
    cached_friend_matches_df,
    clear_app_caches,
    cached_list_other_xuids,
    cached_list_top_teammates,
)
from src.ui.sync import (
    pick_latest_spnkr_db_if_any,
    is_spnkr_db_path,
    cleanup_orphan_tmp_dbs,
    render_sync_indicator,
    refresh_spnkr_db_via_api,
)
from src.ui.multiplayer import (
    is_multi_player_db,
    render_player_selector,
    get_player_display_name,
)


_LABEL_SUFFIX_RE = re.compile(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", re.IGNORECASE)


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


def _default_identity_from_secrets() -> tuple[str, str, str]:
    """Retourne (xuid_or_gamertag, xuid_fallback, waypoint_player) depuis secrets/env/constants."""
    # Secrets Streamlit: .streamlit/secrets.toml
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

    # UI: on préfère afficher le gamertag, tout en conservant xuid en fallback.
    xuid_or_gt = gt or xu
    return xuid_or_gt, xu, wp


def _init_source_state(default_db: str, settings: AppSettings) -> None:
    if "db_path" not in st.session_state:
        chosen = str(default_db or "")
        # Si l'utilisateur force une DB via env (OPENSPARTAN_DB_PATH/OPENSPARTAN_DB),
        # on ne doit pas l'écraser par une auto-sélection SPNKr.
        forced_env_db = str(os.environ.get("OPENSPARTAN_DB") or os.environ.get("OPENSPARTAN_DB_PATH") or "").strip()
        if (not forced_env_db) and bool(getattr(settings, "prefer_spnkr_db_if_available", False)):
            spnkr = pick_latest_spnkr_db_if_any()
            if spnkr and os.path.exists(spnkr) and os.path.getsize(spnkr) > 0:
                chosen = spnkr
        st.session_state["db_path"] = chosen
    if "xuid_input" not in st.session_state:
        legacy = str(st.session_state.get("xuid", "") or "").strip()
        guessed = guess_xuid_from_db_path(st.session_state.get("db_path", "") or "") or ""
        xuid_or_gt, _xuid_fallback, _wp = _default_identity_from_secrets()
        # Pour les DB SPNKr, on pré-remplit avec le joueur déduit du nom de DB (gamertag le plus souvent).
        inferred = infer_spnkr_player_from_db_path(str(st.session_state.get("db_path", "") or "")) or ""
        st.session_state["xuid_input"] = legacy or inferred or guessed or xuid_or_gt
    if "waypoint_player" not in st.session_state:
        _xuid_or_gt, _xuid_fallback, wp = _default_identity_from_secrets()
        st.session_state["waypoint_player"] = wp


def _ensure_h5g_commendations_repo() -> None:
    """Génère automatiquement le référentiel Citations s'il est absent."""
    if st.session_state.get("_h5g_repo_ensured") is True:
        return
    st.session_state["_h5g_repo_ensured"] = True

    try:
        from src.config import get_repo_root

        repo_root = get_repo_root(__file__)
    except Exception:
        repo_root = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(repo_root, "data", "wiki", "halo5_commendations_fr.json")
    html_path = os.path.join(repo_root, "data", "wiki", "halo5_citations_printable.html")
    script_path = os.path.join(repo_root, "scripts", "extract_h5g_commendations_fr.py")

    if os.path.exists(json_path):
        return
    if not (os.path.exists(html_path) and os.path.exists(script_path)):
        return

    with st.spinner("Génération du référentiel Citations (offline)…"):
        try:
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--input-html",
                    html_path,
                    "--clean-output",
                    "--exclude",
                    os.path.join(repo_root, "data", "wiki", "halo5_commendations_exclude.json"),
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            try:
                from src.ui.commendations import load_h5g_commendations_json

                getattr(load_h5g_commendations_json, "clear")()
            except Exception:
                pass
        except Exception:
            return


def _clean_asset_label(s: str | None) -> str | None:
    if s is None:
        return None
    v = str(s).strip()
    if not v:
        return None
    m = _LABEL_SUFFIX_RE.match(v)
    if m:
        v = (m.group(1) or "").strip()
    return v or None


def _normalize_mode_label(pair_name: str | None) -> str | None:
    if pair_name is None:
        return None
    # On traduit d'abord (le mapping complet contient souvent la carte), puis on retire le nom de carte si besoin.
    base = _clean_asset_label(pair_name)
    t = translate_pair_name(base)
    if t is None:
        return None
    s = str(t).strip()
    if " on " in s:
        s = s.split(" on ", 1)[0].strip()
    s = re.sub(r"\s*-\s*Forge\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*-\s*Ranked\b", "", s, flags=re.IGNORECASE).strip()
    return s or None


def _styler_map(styler, func, subset):
    """Compat pandas: Styler.map n'existe pas sur certaines versions.

    - pandas récents: .map(func, subset=...)
    - pandas anciens: .applymap(func, subset=...)
    """
    try:
        if hasattr(styler, "map"):
            return styler.map(func, subset=subset)
    except Exception:
        pass
    # Fallback
    try:
        return styler.applymap(func, subset=subset)
    except Exception:
        return styler


def _assign_player_colors(names: list[str]) -> dict[str, str]:
    palette = HALO_COLORS.as_dict()
    cycle = [
        palette["cyan"],
        palette["violet"],
        palette["amber"],
        palette["red"],
        palette["green"],
        palette["slate"],
    ]
    state_key = "_os_player_colors"
    persisted = st.session_state.get(state_key)
    if not isinstance(persisted, dict):
        persisted = {}

    used = {str(v) for v in persisted.values() if v is not None}

    for n in names:
        key = str(n)
        if not key or key in persisted:
            continue

        chosen = None
        for c in cycle:
            if c not in used:
                chosen = c
                break
        if chosen is None:
            chosen = cycle[len(persisted) % len(cycle)]

        persisted[key] = chosen
        used.add(chosen)

    st.session_state[state_key] = persisted
    return {str(n): persisted[str(n)] for n in names if str(n) in persisted}


def _compute_session_span_seconds(df_: pd.DataFrame) -> float | None:
    if df_ is None or df_.empty or "start_time" not in df_.columns:
        return None
    starts = pd.to_datetime(df_["start_time"], errors="coerce")
    if starts.dropna().empty:
        return None
    t0 = starts.min()
    if "time_played_seconds" in df_.columns:
        durations = pd.to_numeric(df_["time_played_seconds"], errors="coerce")
        ends = starts + pd.to_timedelta(durations.fillna(0), unit="s")
    else:
        ends = starts
    t1 = ends.max()
    if pd.isna(t0) or pd.isna(t1):
        return None
    return float((t1 - t0).total_seconds())


def _compute_total_play_seconds(df_: pd.DataFrame) -> float | None:
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.sum())


def _format_duration_dhm(seconds: float | int | None) -> str:
    if seconds is None or seconds != seconds:
        return "-"
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "-"
    if total < 0:
        return "-"

    minutes, _s = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts: list[str] = []
    if days:
        parts.append(f"{days}j")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}min")
    return " ".join(parts)


def _avg_match_duration_seconds(df_: pd.DataFrame) -> float | None:
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _normalize_map_label(map_name: str | None) -> str | None:
    base = _clean_asset_label(map_name)
    if base is None:
        return None
    s = re.sub(r"\s*-\s*Forge\b", "", base, flags=re.IGNORECASE).strip()
    return s or None

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
# Note: cached_list_other_xuids et cached_list_top_teammates sont importés depuis cache.py
# =============================================================================


# =============================================================================
# Helpers UI
# =============================================================================

def _date_range(df: pd.DataFrame) -> tuple[date, date]:
    """Retourne la plage de dates du DataFrame."""
    dmin = df["date"].min()
    dmax = df["date"].max()
    return dmin, dmax


@st.cache_data(show_spinner=False)
def _build_friends_opts_map(
    db_path: str,
    self_xuid: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
) -> tuple[dict[str, str], list[str]]:
    def _load_local_friends_defaults() -> dict[str, list[str]]:
        """Charge un mapping local {self_xuid: [friend1, friend2, ...]}.

        Fichier local (ignoré git): .streamlit/friends_defaults.json
        Valeurs: gamertags OU XUIDs.
        """
        try:
            p = Path(__file__).resolve().parent / ".streamlit" / "friends_defaults.json"
            if not p.exists():
                return {}
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
        except Exception:
            return {}

        if not isinstance(obj, dict):
            return {}

        out: dict[str, list[str]] = {}
        for k, v in obj.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if not isinstance(v, list):
                continue
            vals: list[str] = []
            for it in v:
                if isinstance(it, str) and it.strip():
                    vals.append(it.strip())
            if vals:
                out[k.strip()] = vals
        return out

    top = cached_list_top_teammates(db_path, self_xuid, db_key=db_key, limit=20)
    default_two = [t[0] for t in top[:2]]
    all_other = cached_list_other_xuids(db_path, self_xuid, db_key=db_key, limit=500)

    ordered: list[str] = []
    seen: set[str] = set()
    for xx, _cnt in top:
        if xx not in seen:
            ordered.append(xx)
            seen.add(xx)
    for xx in all_other:
        if xx not in seen:
            ordered.append(xx)
            seen.add(xx)

    opts_map = build_xuid_option_map(ordered, display_name_fn=display_name_from_xuid)

    # Defaults: top 2, ou override local.
    default_xuids = list(default_two)
    try:
        overrides = _load_local_friends_defaults().get(str(self_xuid).strip())
        if overrides:
            name_to_xuid = {
                str(display_name_from_xuid(xu) or "").strip().casefold(): str(xu).strip() for xu in ordered
            }
            ordered_xuids = [str(xu).strip() for xu in ordered]

            chosen: list[str] = []
            for ident in overrides:
                s = str(ident or "").strip()
                if not s:
                    continue
                if s.isdigit() and s in ordered_xuids:
                    chosen.append(s)
                    continue
                xu = name_to_xuid.get(s.casefold())
                if xu:
                    chosen.append(xu)
            chosen = [x for x in chosen if x]
            if len(chosen) >= 2:
                default_xuids = chosen[:2]
    except Exception:
        pass

    label_by_xuid = {v: k for k, v in opts_map.items()}
    default_labels = [label_by_xuid[x] for x in default_xuids if x in label_by_xuid]
    return opts_map, default_labels


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

    # Propage les defaults depuis secrets vers l'env.
    # Utile notamment pour résoudre un XUID quand la DB SPNKr ne contient pas les gamertags.
    try:
        xuid_or_gt, xuid_fallback, wp = _default_identity_from_secrets()
        if xuid_or_gt and not str(xuid_or_gt).strip().isdigit() and xuid_fallback:
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_GAMERTAG"] = str(xuid_or_gt).strip()
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_XUID"] = str(xuid_fallback).strip()
        if wp and not str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip():
            os.environ["OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER"] = str(wp).strip()
    except Exception:
        pass

    # Applique les overrides de chemins (avant que l'app lise les fichiers).
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
                st.rerun()

        # Bouton Sync rapide pour les DB SPNKr
        if db_path and is_spnkr_db_path(db_path) and os.path.exists(db_path):
            # Déduire le joueur depuis le nom de la DB
            spnkr_player = infer_spnkr_player_from_db_path(db_path) or ""
            
            if spnkr_player:
                sync_col1, sync_col2 = st.columns([1, 1])
                with sync_col1:
                    sync_clicked = st.button(
                        "🔄 Sync",
                        key="quick_sync_button",
                        help="Synchronise les nouveaux matchs (mode delta: arrêt dès match connu).",
                        width="stretch",
                    )
                with sync_col2:
                    full_sync = st.button(
                        "📥 Complète",
                        key="full_sync_button", 
                        help="Synchronisation complète (parcourt tout l'historique).",
                        width="stretch",
                    )
                
                if sync_clicked or full_sync:
                    with st.spinner("Synchronisation en cours..." if sync_clicked else "Sync complète en cours..."):
                        ok, msg = refresh_spnkr_db_via_api(
                            db_path=db_path,
                            player=spnkr_player,
                            match_type="matchmaking",
                            max_matches=200 if sync_clicked else 500,
                            rps=5,
                            with_highlight_events=True,
                            with_aliases=True,
                            delta=sync_clicked,  # Mode delta pour sync rapide
                            timeout_seconds=120 if sync_clicked else 300,
                        )
                    if ok:
                        st.success(msg)
                        _clear_app_caches()
                        st.rerun()
                    else:
                        st.error(msg)

        # Toujours visible
        if st.button(
            "Actualiser",
            width="stretch",
            help="Relance l'app (optionnellement en vidant les caches selon Paramètres).",
        ):
            if bool(getattr(settings, "refresh_clears_caches", False)):
                _clear_app_caches()
                try:
                    getattr(cached_list_local_dbs, "clear")()
                except Exception:
                    pass
            st.rerun()

    # Validation légère (non bloquante)
    from src.db import resolve_xuid_from_db

    if db_path and not os.path.exists(db_path):
        db_path = ""

    # Si la DB existe mais est vide (0 octet), on tente un fallback automatique.
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
                    fallback = str(DEFAULT_DB or "").strip()
                    if not (fallback and os.path.exists(fallback)):
                        fallback = ""
                if fallback and fallback != db_path:
                    st.info(f"DB utilisée: {fallback}")
                    st.session_state["db_path"] = fallback
                    db_path = fallback
                else:
                    db_path = ""
        except Exception:
            pass

    xraw = (xuid or "").strip()
    xuid_resolved = parse_xuid_input(xraw) or ""
    if not xuid_resolved and xraw and not xraw.isdigit() and db_path:
        xuid_resolved = resolve_xuid_from_db(db_path, xraw) or ""
        # Fallback: si la DB ne permet pas de résoudre (pas de gamertags),
        # utilise les defaults secrets/env quand l'entrée correspond au gamertag par défaut.
        if not xuid_resolved:
            try:
                xuid_or_gt, xuid_fallback, _wp = _default_identity_from_secrets()
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
        xuid_or_gt, xuid_fallback, _wp = _default_identity_from_secrets()
        if xuid_or_gt and not xuid_or_gt.isdigit():
            xuid_resolved = resolve_xuid_from_db(db_path, xuid_or_gt) or xuid_fallback
        else:
            xuid_resolved = xuid_or_gt or xuid_fallback

    xuid = xuid_resolved or ""

    me_name = display_name_from_xuid(xuid.strip()) if str(xuid or "").strip() else "(joueur)"
    aliases_key = _aliases_cache_key()

    # Auto-profil (SPNKr) : récupère des URLs d'apparence (service tag / emblem / backdrop / nameplate)
    # et ne remplace pas les champs manuels si déjà remplis.
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

    # Header (profil joueur):
    # - chemins locaux OK
    # - URLs: mise en cache automatique uniquement si l'option est activée
    dl_enabled = bool(getattr(settings, "profile_assets_download_enabled", False)) or bool(api_enabled)
    refresh_h = int(getattr(settings, "profile_assets_auto_refresh_hours", 0) or 0)

    # Valeurs manuelles (prioritaires) / sinon auto depuis API.
    banner_value = str(getattr(settings, "profile_banner", "") or "").strip()
    emblem_value = str(getattr(settings, "profile_emblem", "") or "").strip() or (getattr(api_app, "emblem_image_url", None) if api_app else "")
    backdrop_value = str(getattr(settings, "profile_backdrop", "") or "").strip() or (getattr(api_app, "backdrop_image_url", None) if api_app else "")
    nameplate_value = str(getattr(settings, "profile_nameplate", "") or "").strip() or (getattr(api_app, "nameplate_image_url", None) if api_app else "")
    service_tag_value = str(getattr(settings, "profile_service_tag", "") or "").strip() or (getattr(api_app, "service_tag", None) if api_app else "")
    rank_label_value = str(getattr(settings, "profile_rank_label", "") or "").strip() or (getattr(api_app, "rank_label", None) if api_app else "")
    rank_subtitle_value = str(getattr(settings, "profile_rank_subtitle", "") or "").strip() or (getattr(api_app, "rank_subtitle", None) if api_app else "")
    rank_icon_value = (getattr(api_app, "rank_image_url", None) if api_app else "") or ""

    def _needs_halo_auth(url: str) -> bool:
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

    # Si on doit télécharger des assets protégés et qu'on vient du cache, les tokens peuvent manquer.
    if dl_enabled and (not str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()):
        if _needs_halo_auth(backdrop_value) or _needs_halo_auth(rank_icon_value) or _needs_halo_auth(nameplate_value):
            _ok, _err = ensure_spnkr_tokens(timeout_seconds=12)

    banner_path = ensure_local_image_path(
        banner_value,
        prefix="banner",
        download_enabled=dl_enabled,
        auto_refresh_hours=refresh_h,
    )
    emblem_path = ensure_local_image_path(
        emblem_value,
        prefix="emblem",
        download_enabled=dl_enabled,
        auto_refresh_hours=refresh_h,
    )
    backdrop_path = ensure_local_image_path(
        backdrop_value,
        prefix="backdrop",
        download_enabled=dl_enabled,
        auto_refresh_hours=refresh_h,
    )
    nameplate_path = ensure_local_image_path(
        nameplate_value,
        prefix="nameplate",
        download_enabled=dl_enabled,
        auto_refresh_hours=refresh_h,
    )
    rank_icon_path = ensure_local_image_path(
        rank_icon_value,
        prefix="rank",
        download_enabled=dl_enabled,
        auto_refresh_hours=refresh_h,
    )

    # Diagnostics non bloquants (aide à comprendre pourquoi une image ne s'affiche pas)
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

    # ==========================================================================
    # Chargement des données
    # ==========================================================================
    
    df = pd.DataFrame()
    db_key = _db_cache_key(db_path) if db_path else None
    if db_path and os.path.exists(db_path) and str(xuid or "").strip():
        with perf_section("db/load_df"):
            df = load_df(db_path, xuid.strip(), db_key=db_key)
        if df.empty:
            st.warning("Aucun match trouvé.")
    else:
        st.info("Configure une DB et un joueur dans Paramètres.")

    if not df.empty:
        with perf_section("analysis/mark_firefight"):
            df = mark_firefight(df)

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
        st.header("Filtres")

    # Base pour les filtres : on inclut tout (Firefight sera filtré via checkboxes)
    base_for_filters = df.copy()

    # Base "globale" : toutes les parties (après inclusion/exclusion Firefight)
    base = base_for_filters

    with st.sidebar:
        dmin, dmax = _date_range(base_for_filters)

        # Streamlit interdit de modifier l'état d'un widget après instanciation.
        # On applique donc les changements demandés (depuis d'autres onglets/boutons)
        # ici, avant de créer le radio/selectbox associés.
        pending_mode = st.session_state.pop("_pending_filter_mode", None)
        if pending_mode in ("Période", "Sessions"):
            st.session_state["filter_mode"] = pending_mode

        pending_label = st.session_state.pop("_pending_picked_session_label", None)
        if isinstance(pending_label, str) and pending_label:
            st.session_state["picked_session_label"] = pending_label
        pending_sessions = st.session_state.pop("_pending_picked_sessions", None)
        if isinstance(pending_sessions, list):
            st.session_state["picked_sessions"] = pending_sessions

        if "filter_mode" not in st.session_state:
            st.session_state["filter_mode"] = "Période"
        filter_mode = st.radio(
            "Sélection",
            options=["Période", "Sessions"],
            horizontal=True,
            key="filter_mode",
        )

        # UX: si on a forcé automatiquement le minimum par carte à 1 (via un bouton session),
        # on remet la valeur par défaut quand on revient en mode "Période".
        if filter_mode == "Période" and bool(st.session_state.get("_min_matches_maps_auto")):
            st.session_state["min_matches_maps"] = 5
            st.session_state["_min_matches_maps_auto"] = False

        if filter_mode == "Période" and bool(st.session_state.get("_min_matches_maps_friends_auto")):
            st.session_state["min_matches_maps_friends"] = 5
            st.session_state["_min_matches_maps_friends_auto"] = False

        start_d, end_d = dmin, dmax
        gap_minutes = 35
        picked_session_labels: Optional[list[str]] = None
        
        if filter_mode == "Période":
            cols = st.columns(2)
            with cols[0]:
                start_default = pd.to_datetime(dmin, errors="coerce")
                if pd.isna(start_default):
                    start_default = pd.Timestamp.today().normalize()
                start_default_date = start_default.date()
                end_limit = pd.to_datetime(dmax, errors="coerce")
                if pd.isna(end_limit):
                    end_limit = start_default
                end_limit_date = end_limit.date()
                start_value = st.session_state.get("start_date_cal", start_default_date)
                if not isinstance(start_value, date) or start_value < start_default_date or start_value > end_limit_date:
                    start_value = start_default_date
                start_date = st.date_input(
                    "Début",
                    value=start_value,
                    min_value=start_default_date,
                    max_value=end_limit_date,
                    format="DD/MM/YYYY",
                    key="start_date_cal",
                )
                start_d = start_date
            with cols[1]:
                end_default = pd.to_datetime(dmax, errors="coerce")
                if pd.isna(end_default):
                    end_default = pd.Timestamp.today().normalize()
                end_default_date = end_default.date()
                start_limit = pd.to_datetime(dmin, errors="coerce")
                if pd.isna(start_limit):
                    start_limit = end_default
                start_limit_date = start_limit.date()
                end_value = st.session_state.get("end_date_cal", end_default_date)
                if not isinstance(end_value, date) or end_value < start_limit_date or end_value > end_default_date:
                    end_value = end_default_date
                end_date = st.date_input(
                    "Fin",
                    value=end_value,
                    min_value=start_limit_date,
                    max_value=end_default_date,
                    format="DD/MM/YYYY",
                    key="end_date_cal",
                )
                end_d = end_date
            if start_d > end_d:
                st.warning("La date de début est après la date de fin.")
        else:
            gap_minutes = st.slider(
                "Écart max entre parties (minutes)",
                min_value=15,
                max_value=240,
                value=int(st.session_state.get("gap_minutes", 120)),
                step=5,
                key="gap_minutes",
            )

            base_s_ui = cached_compute_sessions_db(
                db_path,
                xuid.strip(),
                db_key,
                True,  # Inclure Firefight (filtrage via checkboxes)
                gap_minutes,
            )
            session_labels_ui = (
                base_s_ui[["session_id", "session_label"]]
                .drop_duplicates()
                .sort_values("session_id", ascending=False)
            )
            options_ui = session_labels_ui["session_label"].tolist()
            st.session_state["_latest_session_label"] = options_ui[0] if options_ui else None

            def _set_session_selection(label: str) -> None:
                st.session_state.picked_session_label = label
                if label == "(toutes)":
                    st.session_state.picked_sessions = []
                elif label in options_ui:
                    st.session_state.picked_sessions = [label]

            if "picked_session_label" not in st.session_state:
                _set_session_selection(options_ui[0] if options_ui else "(toutes)")
            if "picked_sessions" not in st.session_state:
                st.session_state.picked_sessions = options_ui[:1] if options_ui else []

            cols = st.columns(2)
            if cols[0].button("Dernière session", width="stretch"):
                _set_session_selection(options_ui[0] if options_ui else "(toutes)")
                # UX: en session courte, on veut voir des cartes même jouées 1 fois.
                st.session_state["min_matches_maps"] = 1
                st.session_state["_min_matches_maps_auto"] = True
                st.session_state["min_matches_maps_friends"] = 1
                st.session_state["_min_matches_maps_friends_auto"] = True
            if cols[1].button("Session précédente", width="stretch"):
                current = st.session_state.get("picked_session_label", "(toutes)")
                if not options_ui:
                    _set_session_selection("(toutes)")
                elif current == "(toutes)" or current not in options_ui:
                    _set_session_selection(options_ui[0])
                else:
                    idx = options_ui.index(current)
                    next_idx = min(idx + 1, len(options_ui) - 1)
                    _set_session_selection(options_ui[next_idx])

            # Trio: on calcule le label ici (avant le bouton) à partir de la sélection
            # d'amis, afin que le bouton soit activable sans devoir visiter l'onglet.
            trio_label = None
            try:
                friends_opts_map, friends_default_labels = _build_friends_opts_map(
                    db_path, xuid.strip(), db_key, aliases_key
                )
                picked_friend_labels = st.session_state.get("friends_picked_labels")
                if not isinstance(picked_friend_labels, list) or not picked_friend_labels:
                    picked_friend_labels = friends_default_labels
                picked_xuids = [friends_opts_map[lbl] for lbl in picked_friend_labels if lbl in friends_opts_map]
                if len(picked_xuids) >= 2:
                    f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
                    ids_m = set(
                        cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f1_xuid, db_key=db_key)
                    )
                    ids_c = set(
                        cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f2_xuid, db_key=db_key)
                    )
                    trio_ids = ids_m & ids_c

                    # Sécurité: ne garde que les matchs présents dans la base utilisée pour les sessions.
                    trio_ids = trio_ids & set(base_for_filters["match_id"].astype(str))
                    if trio_ids:
                        trio_rows = base_s_ui.loc[base_s_ui["match_id"].astype(str).isin(trio_ids)].copy()
                        if not trio_rows.empty:
                            latest_sid = int(trio_rows["session_id"].max())
                            latest_labels = (
                                trio_rows.loc[trio_rows["session_id"] == latest_sid, "session_label"]
                                .dropna()
                                .unique()
                                .tolist()
                            )
                            trio_label = latest_labels[0] if latest_labels else None
            except Exception:
                trio_label = None

            st.session_state["_trio_latest_session_label"] = trio_label
            disabled_trio = not isinstance(trio_label, str) or not trio_label
            if st.button("Dernière session en trio", width="stretch", disabled=disabled_trio):
                st.session_state["_pending_filter_mode"] = "Sessions"
                st.session_state["_pending_picked_session_label"] = trio_label
                st.session_state["_pending_picked_sessions"] = [trio_label]
                st.session_state["min_matches_maps"] = 1
                st.session_state["_min_matches_maps_auto"] = True
                st.session_state["min_matches_maps_friends"] = 1
                st.session_state["_min_matches_maps_friends_auto"] = True
                st.rerun()
            if disabled_trio:
                st.caption('Trio : sélectionne 2 amis dans "Avec mes amis" pour activer.')
            else:
                st.caption(f"Trio : {trio_label}")

            picked_one = st.selectbox("Session", options=["(toutes)"] + options_ui, key="picked_session_label")
            picked_session_labels = None if picked_one == "(toutes)" else [picked_one]

        # ------------------------------------------------------------------
        # Filtres en cascade avec checkboxes
        # Playlist -> Mode (pair) -> Carte
        # ------------------------------------------------------------------
        # Le scope des filtres suit les filtres au-dessus (période/sessions).
        dropdown_base = base_for_filters.copy()

        if filter_mode == "Période":
            dropdown_base = dropdown_base.loc[(dropdown_base["date"] >= start_d) & (dropdown_base["date"] <= end_d)].copy()
        else:
            # Sessions: on utilise la sélection de sessions pour limiter les options.
            if picked_session_labels:
                dropdown_base = base_s_ui.loc[base_s_ui["session_label"].isin(picked_session_labels)].copy()
            else:
                dropdown_base = base_s_ui.copy()

        dropdown_base["playlist_ui"] = dropdown_base["playlist_name"].apply(_clean_asset_label).apply(translate_playlist_name)
        dropdown_base["mode_ui"] = dropdown_base["pair_name"].apply(_normalize_mode_label)
        dropdown_base["map_ui"] = dropdown_base["map_name"].apply(_normalize_map_label)

        # --- Playlists (avec Firefight décoché par défaut) ---
        playlist_values = sorted({str(x).strip() for x in dropdown_base["playlist_ui"].dropna().tolist() if str(x).strip()})
        preferred_order = ["Partie rapide", "Arène classée", "Assassin classé"]
        playlist_values = [p for p in preferred_order if p in playlist_values] + [p for p in playlist_values if p not in preferred_order]
        
        firefight_playlists = get_firefight_playlists(playlist_values)
        playlists_selected = render_checkbox_filter(
            label="Playlists",
            options=playlist_values,
            session_key="filter_playlists",
            default_unchecked=firefight_playlists,  # Firefight décoché par défaut
            expanded=False,
        )

        # Scope après filtre playlist
        scope1 = dropdown_base
        if playlists_selected and len(playlists_selected) < len(playlist_values):
            scope1 = scope1.loc[scope1["playlist_ui"].fillna("").isin(playlists_selected)].copy()

        # --- Modes (hiérarchique par catégorie) ---
        mode_values = sorted({str(x).strip() for x in scope1["mode_ui"].dropna().tolist() if str(x).strip()})
        modes_selected = render_hierarchical_checkbox_filter(
            label="Modes",
            options=mode_values,
            session_key="filter_modes",
            expanded=False,
        )

        # Scope après filtre mode
        scope2 = scope1
        if modes_selected and len(modes_selected) < len(mode_values):
            scope2 = scope2.loc[scope2["mode_ui"].fillna("").isin(modes_selected)].copy()

        # --- Cartes ---
        map_values = sorted({str(x).strip() for x in scope2["map_ui"].dropna().tolist() if str(x).strip()})
        maps_selected = render_checkbox_filter(
            label="Cartes",
            options=map_values,
            session_key="filter_maps",
            expanded=False,
        )

        # Paramètres avancés déplacés dans l'onglet Paramètres.

    # ==========================================================================
    # Application des filtres
    # ==========================================================================
    
    with perf_section("filters/apply"):
        if filter_mode == "Sessions":
            # On inclut tout (Firefight sera filtré via checkboxes playlists)
            base_s = cached_compute_sessions_db(db_path, xuid.strip(), db_key, True, gap_minutes)
            dff = (
                base_s.loc[base_s["session_label"].isin(picked_session_labels)].copy()
                if picked_session_labels
                else base_s.copy()
            )
        else:
            dff = base_for_filters.copy()

        if "playlist_fr" not in dff.columns:
            dff["playlist_fr"] = dff["playlist_name"].apply(translate_playlist_name)
        if "pair_fr" not in dff.columns:
            dff["pair_fr"] = dff["pair_name"].apply(translate_pair_name)
            
    if "playlist_ui" not in dff.columns:
        dff["playlist_ui"] = dff["playlist_name"].apply(_clean_asset_label).apply(translate_playlist_name)
    if "mode_ui" not in dff.columns:
        dff["mode_ui"] = dff["pair_name"].apply(_normalize_mode_label)
    if "map_ui" not in dff.columns:
        dff["map_ui"] = dff["map_name"].apply(_normalize_map_label)

    # Application des filtres checkboxes
    if playlists_selected:
        dff = dff.loc[dff["playlist_ui"].fillna("").isin(playlists_selected)]
    if modes_selected:
        dff = dff.loc[dff["mode_ui"].fillna("").isin(modes_selected)]
    if maps_selected:
        dff = dff.loc[dff["map_ui"].fillna("").isin(maps_selected)]

    if filter_mode == "Période":
        mask = (dff["date"] >= start_d) & (dff["date"] <= end_d)
        dff = dff.loc[mask].copy()


    # ==========================================================================
    # KPIs
    # ==========================================================================
    
    with perf_section("kpis"):
        rates = compute_outcome_rates(dff)
        total_outcomes = max(1, rates.total)
        win_rate = rates.wins / total_outcomes
        loss_rate = rates.losses / total_outcomes

        avg_acc = dff["accuracy"].dropna().mean() if not dff.empty else None
        global_ratio = compute_global_ratio(dff)
        avg_life = dff["average_life_seconds"].dropna().mean() if not dff.empty else None

    # ------------------------------------------------------------------
    # Bandeau résumé (en haut du site) — regroupé
    # ------------------------------------------------------------------
    avg_match_seconds = _avg_match_duration_seconds(dff)
    total_play_seconds = _compute_total_play_seconds(dff)
    avg_match_txt = format_duration_hms(avg_match_seconds)
    total_play_txt = format_duration_dhm(total_play_seconds)

    # Stats par minute / totaux
    stats = compute_aggregated_stats(dff)

    # Moyennes par partie
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    st.subheader("Parties")
    render_top_summary(len(dff), rates)
    render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Durée totale", total_play_txt),
        ]
    )

    st.subheader("Carrière")
    render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Frags par partie", f"{kpg:.2f}" if (kpg is not None and pd.notna(kpg)) else "-"),
            ("Morts par partie", f"{dpg:.2f}" if (dpg is not None and pd.notna(dpg)) else "-"),
            ("Assistances par partie", f"{apg:.2f}" if (apg is not None and pd.notna(apg)) else "-"),
        ],
        dense=False,
    )
    render_kpi_cards(
        [
            ("Frags / min", f"{stats.kills_per_minute:.2f}" if stats.kills_per_minute else "-"),
            ("Morts / min", f"{stats.deaths_per_minute:.2f}" if stats.deaths_per_minute else "-"),
            ("Assistances / min", f"{stats.assists_per_minute:.2f}" if stats.assists_per_minute else "-"),
            ("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-"),
            ("Durée de vie moyenne", format_mmss(avg_life)),
            ("Taux de victoire", f"{win_rate*100:.1f}%" if rates.total else "-"),
            ("Taux de défaite", f"{loss_rate*100:.1f}%" if rates.total else "-"),
            ("Ratio", f"{global_ratio:.2f}" if global_ratio is not None else "-"),
        ],
        dense=False,
    )

    # (Résumé déplacé en haut du site)

    # ==========================================================================
    # Pages (navigation)
    # ==========================================================================

    pages = [
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

    pending_page = st.session_state.pop("_pending_page", None)
    if isinstance(pending_page, str) and pending_page in pages:
        st.session_state["page"] = pending_page
    if "page" not in st.session_state:
        st.session_state["page"] = "Séries temporelles"

    pending_mid = st.session_state.pop("_pending_match_id", None)
    if isinstance(pending_mid, str) and pending_mid.strip():
        st.session_state["match_id_input"] = pending_mid.strip()

    page = st.segmented_control(
        "Onglets",
        options=pages,
        key="page",
        label_visibility="collapsed",
    )

    # Paramètres communs pour les pages de match
    _match_view_params = dict(
        db_path=db_path,
        xuid=xuid,
        waypoint_player=waypoint_player,
        db_key=db_key,
        settings=settings,
        df_full=df,  # Historique complet pour le score relatif
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

    # --------------------------------------------------------------------------
    # Page: Dernier match
    # --------------------------------------------------------------------------
    if page == "Dernier match":
        render_last_match_page(dff=dff, **_match_view_params)

    # --------------------------------------------------------------------------
    # Page: Match (recherche)
    # --------------------------------------------------------------------------
    elif page == "Match":
        render_match_search_page(df=df, dff=dff, **_match_view_params)

    # --------------------------------------------------------------------------
    # Page: Citations (ex Médailles)
    # --------------------------------------------------------------------------
    elif page == "Citations":
        render_citations_page(
            dff=dff,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            top_medals_fn=_top_medals,
        )

    # --------------------------------------------------------------------------
    # Page: Comparaison de sessions
    # --------------------------------------------------------------------------
    elif page == "Comparaison de sessions":
        all_sessions_df = cached_compute_sessions_db(
            db_path, xuid.strip(), db_key, True, gap_minutes  # Inclure tout, filtrage via checkboxes
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
    # Page: Mes coéquipiers (fusion)
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
            include_firefight=True,  # Inclure tout, filtrage via checkboxes playlists
            waypoint_player=waypoint_player,
            build_friends_opts_map_fn=_build_friends_opts_map,
            assign_player_colors_fn=_assign_player_colors,
            plot_multi_metric_bars_fn=plot_multi_metric_bars_by_match,
            top_medals_fn=_top_medals,
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
            df_full=df,  # Historique complet pour le score relatif
        )

    # --------------------------------------------------------------------------
    # Page: Paramètres
    # --------------------------------------------------------------------------
    elif page == "Paramètres":
        render_settings_page(
            settings,
            get_local_dbs_fn=cached_list_local_dbs,
            on_clear_caches_fn=_clear_app_caches,
        )


if __name__ == "__main__":
    main()
