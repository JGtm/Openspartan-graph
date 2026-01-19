"""OpenSpartan Graphs - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données OpenSpartan Workshop.
"""

import os
import re
from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Imports depuis la nouvelle architecture
from src.config import (
    get_default_db_path,
    get_default_workshop_exe_path,
    DEFAULT_WAYPOINT_PLAYER,
    HALO_COLORS,
    SESSION_CONFIG,
    OUTCOME_CODES,
)
from src.db import (
    load_matches,
    query_matches_with_friend,
    list_other_player_xuids,
    list_top_teammates,
    guess_xuid_from_db_path,
    load_player_match_result,
    load_match_medals_for_player,
    load_top_medals,
)
from src.db.parsers import parse_xuid_input
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
)
from src.ui.medals import (
    load_medal_name_maps,
    medal_has_known_label,
    get_medals_cache_dir,
    medal_label,
    medal_icon_path,
    render_medals_grid,
)
from src.ui.formatting import format_date_fr
from src.db.profiles import (
    PROFILES_PATH,
    load_profiles,
    save_profiles,
    list_local_dbs,
)
from src.config import get_aliases_file_path

from src.ui.perf import perf_reset_run, perf_section, render_perf_panel
from src.ui.sections import render_openspartan_tools, render_source_section


_LABEL_SUFFIX_RE = re.compile(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", re.IGNORECASE)
_SCORE_LABEL_RE = re.compile(r"^\s*(-?\d+)\s*[-–—]\s*(-?\d+)\s*$")


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


def _format_duration_hms(seconds: float | int | None) -> str:
    if seconds is None or seconds != seconds:
        return "-"
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "-"
    if total < 0:
        return "-"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, hh = divmod(h, 24)
        return f"{d}j {hh:02d}:{m:02d}:{s:02d}"
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _coerce_int(v) -> int | None:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = float(v)
        if x != x:
            return None
        return int(round(x))
    except Exception:
        return None


def _format_score_label(my_team_score, enemy_team_score) -> str:
    my_s = _coerce_int(my_team_score)
    en_s = _coerce_int(enemy_team_score)
    if my_s is None or en_s is None:
        return "-"
    return f"{my_s} - {en_s}"


def _score_css_color(my_team_score, enemy_team_score) -> str:
    colors = HALO_COLORS.as_dict()
    my_s = _coerce_int(my_team_score)
    en_s = _coerce_int(enemy_team_score)
    if my_s is None or en_s is None:
        return colors["slate"]
    if my_s > en_s:
        return colors["green"]
    if my_s < en_s:
        return colors["red"]
    return colors["violet"]


def _style_outcome_text(v: str) -> str:
    s = (v or "").strip().lower()
    if s == "victoire":
        return "color: #1B5E20; font-weight: 700;"
    if s in ("défaite", "defaite"):
        return "color: #B71C1C; font-weight: 700;"
    if s in ("égalité", "egalite"):
        return "color: #8E6CFF; font-weight: 700;"
    if s in ("non terminé", "non termine"):
        return "color: #616161; font-weight: 700;"
    return ""


def _style_signed_number(v) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if x > 0:
        return "color: #1B5E20; font-weight: 700;"
    if x < 0:
        return "color: #B71C1C; font-weight: 700;"
    return "color: #424242;"


def _style_score_label(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s == "-":
        return "color: #616161;"
    m = _SCORE_LABEL_RE.match(s)
    if not m:
        return ""
    try:
        my_s = int(m.group(1))
        en_s = int(m.group(2))
    except Exception:
        return ""
    if my_s > en_s:
        return "color: #1B5E20; font-weight: 800;"
    if my_s < en_s:
        return "color: #B71C1C; font-weight: 800;"
    return "color: #8E6CFF; font-weight: 800;"


def _format_datetime_fr_hm(dt_value) -> str:
    if dt_value is None:
        return "-"
    try:
        ts = pd.to_datetime(dt_value, errors="coerce")
        if pd.isna(ts):
            return "-"
        d = ts.to_pydatetime()
    except Exception:
        return "-"
    return f"{format_date_fr(d)} {d:%H:%M}"


def _plot_metric_bars_by_match(
    df_: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    y_axis_title: str,
    hover_label: str,
    bar_color: str,
    smooth_color: str,
    smooth_window: int = 10,
) -> go.Figure | None:
    if df_ is None or df_.empty:
        return None
    if metric_col not in df_.columns or "start_time" not in df_.columns:
        return None

    d = df_[["start_time", metric_col]].copy()
    d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce")
    d = d.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    if d.empty:
        return None

    y = pd.to_numeric(d[metric_col], errors="coerce")
    x_idx = list(range(len(d)))
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if labels else 1

    w = int(smooth_window) if smooth_window else 0
    smooth = y.rolling(window=max(1, w), min_periods=1).mean() if w and w > 1 else y

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=y,
            name=y_axis_title,
            marker_color=bar_color,
            opacity=0.70,
            hovertemplate=f"{hover_label}=%{{y}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=smooth,
            mode="lines",
            name="Moyenne (lissée)",
            line=dict(width=3, color=smooth_color),
            hovertemplate="moyenne=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_yaxes(title_text=y_axis_title, rangemode="tozero")
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, height=320)


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


def _avg_match_duration_seconds(df_: pd.DataFrame) -> float | None:
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _render_kpi_cards(cards: list[tuple[str, str]], *, dense: bool = True) -> None:
    if not cards:
        return
    grid_class = "os-kpi-grid os-kpi-grid--dense" if dense else "os-kpi-grid"
    items = "".join(
        f"<div class='os-kpi'><div class='os-kpi__label'>{label}</div><div class='os-kpi__value'>{value}</div></div>"
        for (label, value) in cards
    )
    st.markdown(f"<div class='{grid_class}'>{items}</div>", unsafe_allow_html=True)


def _render_top_summary(total_matches: int, rates) -> None:
    if total_matches <= 0:
        st.markdown(
            "<div class='os-top-summary'>"
            "  <div class='os-top-summary__empty'>Aucun match sélectionné</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    wins = int(getattr(rates, "wins", 0) or 0)
    losses = int(getattr(rates, "losses", 0) or 0)
    ties = int(getattr(rates, "ties", 0) or 0)
    no_finish = int(getattr(rates, "no_finish", 0) or 0)

    st.markdown(
        "<div class='os-top-summary'>"
        "  <div class='os-top-summary__row'>"
        "    <div class='os-top-summary__left'>"
        "      <div class='os-top-summary__kicker'>Parties sélectionnées</div>"
        f"      <div class='os-top-summary__count'>{total_matches}</div>"
        "    </div>"
        "    <div class='os-top-summary__chips'>"
        f"      <div class='os-top-chip os-top-chip--win'><span class='os-top-chip__label'>Victoires</span><span class='os-top-chip__value'>{wins}</span></div>"
        f"      <div class='os-top-chip os-top-chip--loss'><span class='os-top-chip__label'>Défaites</span><span class='os-top-chip__value'>{losses}</span></div>"
        f"      <div class='os-top-chip os-top-chip--tie'><span class='os-top-chip__label'>Égalités</span><span class='os-top-chip__value'>{ties}</span></div>"
        f"      <div class='os-top-chip os-top-chip--nf'><span class='os-top-chip__label'>Non terminés</span><span class='os-top-chip__value'>{no_finish}</span></div>"
        "    </div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )


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


def _db_cache_key(db_path: str) -> tuple[int, int] | None:
    """Retourne une signature stable de la DB pour invalider les caches.

    On utilise (mtime_ns, size) : rapide et suffisamment fiable pour détecter
    les mises à jour de la DB OpenSpartan.
    """
    try:
        st_ = os.stat(db_path)
    except OSError:
        return None
    return int(getattr(st_, "st_mtime_ns", int(st_.st_mtime * 1e9))), int(st_.st_size)


@st.cache_data(show_spinner=False, ttl=30)
def cached_list_local_dbs(_refresh_token: int = 0) -> list[str]:
    """Liste des DB locales (TTL court pour éviter un scan disque trop fréquent)."""
    return list_local_dbs()


@st.cache_data(show_spinner=False)
def cached_compute_sessions_db(
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    include_firefight: bool,
    gap_minutes: int,
) -> pd.DataFrame:
    """Compute sessions sur la base (cache)."""
    df0 = load_df(db_path, xuid, db_key=db_key)
    df0 = mark_firefight(df0)
    if (not include_firefight) and ("is_firefight" in df0.columns):
        df0 = df0.loc[~df0["is_firefight"]].copy()
    return compute_sessions(df0, gap_minutes=int(gap_minutes))


@st.cache_data(show_spinner=False)
def cached_same_team_match_ids_with_friend(
    db_path: str,
    self_xuid: str,
    friend_xuid: str,
    db_key: tuple[int, int] | None,
) -> tuple[str, ...]:
    """Retourne les match_id (str) joués dans la même équipe avec un ami (cache)."""
    rows = query_matches_with_friend(db_path, self_xuid, friend_xuid)
    ids = {str(r.match_id) for r in rows if getattr(r, "same_team", False)}
    return tuple(sorted(ids))


@st.cache_data(show_spinner=False)
def cached_query_matches_with_friend(
    db_path: str,
    self_xuid: str,
    friend_xuid: str,
    db_key: tuple[int, int] | None,
):
    return query_matches_with_friend(db_path, self_xuid, friend_xuid)


@st.cache_data(show_spinner=False)
def cached_load_player_match_result(
    db_path: str,
    match_id: str,
    xuid: str,
    db_key: tuple[int, int] | None,
):
    return load_player_match_result(db_path, match_id, xuid)


@st.cache_data(show_spinner=False)
def cached_load_match_medals_for_player(
    db_path: str,
    match_id: str,
    xuid: str,
    db_key: tuple[int, int] | None,
):
    return load_match_medals_for_player(db_path, match_id, xuid)


@st.cache_data(show_spinner=False)
def cached_load_top_medals(
    db_path: str,
    xuid: str,
    match_ids: tuple[str, ...],
    top_n: int,
    db_key: tuple[int, int] | None,
):
    return load_top_medals(db_path, xuid, list(match_ids), top_n=int(top_n))


def _top_medals(
    db_path: str,
    xuid: str,
    match_ids: list[str],
    *,
    top_n: int,
    db_key: tuple[int, int] | None,
):
    # Évite de stocker d'immenses tuples en cache.
    if len(match_ids) > 1500:
        return load_top_medals(db_path, xuid, match_ids, top_n=top_n)
    return cached_load_top_medals(db_path, xuid, tuple(match_ids), top_n, db_key=db_key)


@st.cache_data(show_spinner=False)
def cached_friend_matches_df(
    db_path: str,
    self_xuid: str,
    friend_xuid: str,
    same_team_only: bool,
    db_key: tuple[int, int] | None,
) -> pd.DataFrame:
    rows = cached_query_matches_with_friend(db_path, self_xuid, friend_xuid, db_key=db_key)
    if same_team_only:
        rows = [r for r in rows if r.same_team]
    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "start_time",
                "playlist_name",
                "pair_name",
                "same_team",
                "my_team_id",
                "my_outcome",
                "friend_team_id",
                "friend_outcome",
            ]
        )

    dfr = pd.DataFrame(
        [
            {
                "match_id": r.match_id,
                "start_time": r.start_time,
                "playlist_name": translate_playlist_name(r.playlist_name),
                "pair_name": translate_pair_name(r.pair_name),
                "same_team": r.same_team,
                "my_team_id": r.my_team_id,
                "my_outcome": r.my_outcome,
                "friend_team_id": r.friend_team_id,
                "friend_outcome": r.friend_outcome,
            }
            for r in rows
        ]
    )
    dfr["start_time"] = pd.to_datetime(dfr["start_time"], utc=True).dt.tz_convert(None)
    return dfr.sort_values("start_time", ascending=False)


def _clear_app_caches() -> None:
    """Vide les caches Streamlit (utile si DB/alias/csv changent en dehors de l'app)."""
    try:
        st.cache_data.clear()
    except Exception:
        pass


def _aliases_cache_key() -> int | None:
    try:
        p = get_aliases_file_path()
        st_ = os.stat(p)
        return int(getattr(st_, "st_mtime_ns", int(st_.st_mtime * 1e9)))
    except OSError:
        return None


# =============================================================================
# Chargement des données (avec cache)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_df(db_path: str, xuid: str, db_key: tuple[int, int] | None = None) -> pd.DataFrame:
    """Charge les matchs et les convertit en DataFrame."""
    matches = load_matches(db_path, xuid)
    df = pd.DataFrame(
        {
            "match_id": [m.match_id for m in matches],
            "start_time": [m.start_time for m in matches],
            "map_id": [m.map_id for m in matches],
            "map_name": [m.map_name for m in matches],
            "playlist_id": [m.playlist_id for m in matches],
            "playlist_name": [m.playlist_name for m in matches],
            "pair_id": [m.map_mode_pair_id for m in matches],
            "pair_name": [m.map_mode_pair_name for m in matches],
            "game_variant_id": [m.game_variant_id for m in matches],
            "game_variant_name": [m.game_variant_name for m in matches],
            "outcome": [m.outcome for m in matches],
            "kda": [m.kda for m in matches],
            "my_team_score": [m.my_team_score for m in matches],
            "enemy_team_score": [m.enemy_team_score for m in matches],
            "max_killing_spree": [m.max_killing_spree for m in matches],
            "headshot_kills": [m.headshot_kills for m in matches],
            "average_life_seconds": [m.average_life_seconds for m in matches],
            "time_played_seconds": [m.time_played_seconds for m in matches],
            "kills": [m.kills for m in matches],
            "deaths": [m.deaths for m in matches],
            "assists": [m.assists for m in matches],
            "accuracy": [m.accuracy for m in matches],
            "ratio": [m.ratio for m in matches],
        }
    )
    # Facilite les filtres date
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(None)
    df["date"] = df["start_time"].dt.date

    # Stats par minute
    minutes = (pd.to_numeric(df["time_played_seconds"], errors="coerce") / 60.0).astype(float)
    minutes = minutes.where(minutes > 0)
    df["kills_per_min"] = pd.to_numeric(df["kills"], errors="coerce") / minutes
    df["deaths_per_min"] = pd.to_numeric(df["deaths"], errors="coerce") / minutes
    df["assists_per_min"] = pd.to_numeric(df["assists"], errors="coerce") / minutes
    return df


@st.cache_data(show_spinner=False)
def cached_list_other_xuids(
    db_path: str, self_xuid: str, db_key: tuple[int, int] | None = None, limit: int = 500
) -> list[str]:
    """Version cachée de list_other_player_xuids."""
    return list_other_player_xuids(db_path, self_xuid, limit)


@st.cache_data(show_spinner=False)
def cached_list_top_teammates(
    db_path: str, self_xuid: str, db_key: tuple[int, int] | None = None, limit: int = 20
) -> list[tuple[str, int]]:
    """Version cachée de list_top_teammates."""
    return list_top_teammates(db_path, self_xuid, limit)


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
    default_labels = [k for k, v in opts_map.items() if v in default_two]
    return opts_map, default_labels


# =============================================================================
# Application principale
# =============================================================================

def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    st.set_page_config(page_title="OpenSpartan Graphs", layout="wide")

    perf_reset_run()

    with perf_section("css"):
        st.markdown(load_css(), unsafe_allow_html=True)

    # ==========================================================================
    # Sidebar - Configuration
    # ==========================================================================
    
    DEFAULT_DB = get_default_db_path()

    with st.sidebar:
        render_perf_panel(location="sidebar")
        st.markdown("<div class='os-sidebar-brand'>OpenSpartan Graphs</div>", unsafe_allow_html=True)
        st.markdown("<div class='os-sidebar-divider'></div>", unsafe_allow_html=True)

        with perf_section("sidebar/source"):
            with st.expander("Source", expanded=False):
                db_path, xuid, waypoint_player = render_source_section(
                    DEFAULT_DB,
                    get_local_dbs=cached_list_local_dbs,
                    on_clear_caches=_clear_app_caches,
                )

        with perf_section("sidebar/openspartan"):
            render_openspartan_tools()

        with perf_section("sidebar/validate"):
            if not db_path.strip():
                st.error(
                    "Aucun .db détecté automatiquement. "
                    "Vérifie que OpenSpartan Workshop est installé."
                )
                st.stop()
            if not os.path.exists(db_path):
                st.error("Le fichier .db n'existe pas à ce chemin.")
                st.stop()
            if not xuid.strip().isdigit():
                st.error("XUID invalide (doit être numérique).")
                st.stop()

    me_name = display_name_from_xuid(xuid.strip())
    aliases_key = _aliases_cache_key()

    # ==========================================================================
    # Chargement des données
    # ==========================================================================
    
    db_key = _db_cache_key(db_path)
    with perf_section("db/load_df"):
        df = load_df(db_path, xuid.strip(), db_key=db_key)
    if df.empty:
        st.warning("Aucun match trouvé.")
        st.stop()

    with perf_section("analysis/mark_firefight"):
        df = mark_firefight(df)

    # ==========================================================================
    # Sidebar - Filtres
    # ==========================================================================
    
    with st.sidebar:
        st.header("Filtres")
        if "include_firefight" not in st.session_state:
            st.session_state["include_firefight"] = False
        if "restrict_playlists" not in st.session_state:
            st.session_state["restrict_playlists"] = True

    include_firefight = bool(st.session_state.get("include_firefight", False))

    # Firefight exclu par défaut
    base_for_filters = df.copy()
    if (not include_firefight) and ("is_firefight" in base_for_filters.columns):
        base_for_filters = base_for_filters.loc[~base_for_filters["is_firefight"]].copy()

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
                start_d = st.date_input("Début", value=dmin, min_value=dmin, max_value=dmax)
            with cols[1]:
                end_d = st.date_input("Fin", value=dmax, min_value=dmin, max_value=dmax)
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
                include_firefight,
                gap_minutes,
            )
            session_labels_ui = (
                base_s_ui[["session_id", "session_label"]]
                .drop_duplicates()
                .sort_values("session_id", ascending=False)
            )
            options_ui = session_labels_ui["session_label"].tolist()
            st.session_state["_latest_session_label"] = options_ui[0] if options_ui else None

            compare_multi = st.toggle("Comparer plusieurs sessions", value=False)

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

            if compare_multi:
                picked = st.multiselect("Sessions", options=options_ui, key="picked_sessions")
                picked_session_labels = picked if picked else None
            else:
                picked_one = st.selectbox("Session", options=["(toutes)"] + options_ui, key="picked_session_label")
                picked_session_labels = None if picked_one == "(toutes)" else [picked_one]

        # ------------------------------------------------------------------
        # Filtres en cascade (ne montrent que les valeurs réellement jouées)
        # Playlist -> Mode (pair) -> Carte
        # ------------------------------------------------------------------
        # Le scope des dropdowns suit les filtres au-dessus (période/sessions)
        # + les réglages avancés (Firefight / restriction playlists).
        dropdown_base = base_for_filters.copy()

        restrict_playlists_ui = bool(st.session_state.get("restrict_playlists", True))
        if restrict_playlists_ui:
            pl0 = dropdown_base["playlist_name"].apply(_clean_asset_label).fillna("").astype(str)
            allowed_mask0 = pl0.apply(is_allowed_playlist_name)
            if allowed_mask0.any():
                dropdown_base = dropdown_base.loc[allowed_mask0].copy()

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

        playlist_values = sorted({str(x).strip() for x in dropdown_base["playlist_ui"].dropna().tolist() if str(x).strip()})
        preferred_order = ["Partie rapide", "Arène classée", "Assassin classé"]
        playlist_values = [p for p in preferred_order if p in playlist_values] + [p for p in playlist_values if p not in preferred_order]
        playlist_selected = st.selectbox("Playlist", options=["(toutes)"] + playlist_values, index=0)

        scope1 = dropdown_base
        if playlist_selected != "(toutes)":
            scope1 = scope1.loc[scope1["playlist_ui"].fillna("") == playlist_selected].copy()

        mode_values = sorted({str(x).strip() for x in scope1["mode_ui"].dropna().tolist() if str(x).strip()})
        mode_selected = st.selectbox("Mode", options=["(tous)"] + mode_values, index=0)

        scope2 = scope1
        if mode_selected != "(tous)":
            scope2 = scope2.loc[scope2["mode_ui"].fillna("") == mode_selected].copy()

        map_values = sorted({str(x).strip() for x in scope2["map_ui"].dropna().tolist() if str(x).strip()})
        map_selected = st.selectbox("Carte", options=["(toutes)"] + map_values, index=0)

        last_n_acc = st.slider("Précision: derniers matchs", 5, 50, 20, step=1)

        with st.expander("Paramètres avancés", expanded=False):
            st.toggle("Inclure Firefight (PvE)", key="include_firefight", value=False)
            st.toggle(
                "Limiter aux playlists (Quick Play / Ranked Slayer / Ranked Arena)",
                key="restrict_playlists",
                value=True,
            )

    # ==========================================================================
    # Application des filtres
    # ==========================================================================
    
    with perf_section("filters/apply"):
        if filter_mode == "Sessions":
            base_s = cached_compute_sessions_db(db_path, xuid.strip(), db_key, include_firefight, gap_minutes)
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

    restrict_playlists = bool(st.session_state.get("restrict_playlists", True))
    if restrict_playlists:
        pl = dff["playlist_name"].apply(_clean_asset_label).fillna("").astype(str)
        allowed_mask = pl.apply(is_allowed_playlist_name)
        if allowed_mask.any():
            dff = dff.loc[allowed_mask].copy()
        else:
            st.sidebar.warning(
                "Aucune playlist n'a matché Quick Play / Ranked Slayer / Ranked Arena. "
                "Désactive ce filtre si tes libellés sont différents."
            )
            
    if "playlist_ui" not in dff.columns:
        dff["playlist_ui"] = dff["playlist_name"].apply(_clean_asset_label).apply(translate_playlist_name)
    if "mode_ui" not in dff.columns:
        dff["mode_ui"] = dff["pair_name"].apply(_normalize_mode_label)
    if "map_ui" not in dff.columns:
        dff["map_ui"] = dff["map_name"].apply(_normalize_map_label)

    if playlist_selected != "(toutes)":
        dff = dff.loc[dff["playlist_ui"].fillna("") == playlist_selected]
    if mode_selected != "(tous)":
        dff = dff.loc[dff["mode_ui"].fillna("") == mode_selected]
    if map_selected != "(toutes)":
        dff = dff.loc[dff["map_ui"].fillna("") == map_selected]

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
    # Bandeau résumé (en haut du site)
    # ------------------------------------------------------------------
    _render_top_summary(len(dff), rates)

    avg_match_seconds = _avg_match_duration_seconds(dff)
    span_seconds = _compute_session_span_seconds(dff)
    avg_match_txt = _format_duration_hms(avg_match_seconds)
    span_txt = _format_duration_hms(span_seconds)
    st.markdown(
        "<div class='os-top-kpis'>"
        "  <div class='os-top-kpi'>"
        "    <div class='os-top-kpi__label'>Durée moyenne / match</div>"
        f"    <div class='os-top-kpi__value'>{avg_match_txt}</div>"
        "  </div>"
        "  <div class='os-top-kpi'>"
        "    <div class='os-top-kpi__label'>Durée totale</div>"
        f"    <div class='os-top-kpi__value'>{span_txt}</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Moyennes par partie
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    _render_kpi_cards(
        [
            ("Frags par partie", f"{kpg:.2f}" if (kpg is not None and pd.notna(kpg)) else "-"),
            ("Morts par partie", f"{dpg:.2f}" if (dpg is not None and pd.notna(dpg)) else "-"),
            ("Assistances par partie", f"{apg:.2f}" if (apg is not None and pd.notna(apg)) else "-"),
        ]
    )

    # Stats par minute
    stats = compute_aggregated_stats(dff)

    _render_kpi_cards(
        [
            ("Frags / min", f"{stats.kills_per_minute:.2f}" if stats.kills_per_minute else "-"),
            ("Morts / min", f"{stats.deaths_per_minute:.2f}" if stats.deaths_per_minute else "-"),
            ("Assistances / min", f"{stats.assists_per_minute:.2f}" if stats.assists_per_minute else "-"),
        ]
    )

    _render_kpi_cards(
        [
            ("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-"),
            ("Taux de victoire", f"{win_rate*100:.1f}%" if rates.total else "-"),
            ("Taux de défaite", f"{loss_rate*100:.1f}%" if rates.total else "-"),
            ("Ratio global", f"{global_ratio:.2f}" if global_ratio is not None else "-"),
            ("Durée de vie moyenne", format_mmss(avg_life)),
        ]
    )

    # (Résumé déplacé en haut du site)

    # ==========================================================================
    # Onglets
    # ==========================================================================
    
    tab_series, tab_last, tab_medals, tab_mom, tab_teammates, tab_table = st.tabs(
        [
            "Séries temporelles",
            "Dernier match",
            "Médailles (Top 25)",
            "Victoires/Défaites",
            "Mes coéquipiers",
            "Historique des parties",
        ]
    )

    # --------------------------------------------------------------------------
    # Tab: Dernier match
    # --------------------------------------------------------------------------
    with tab_last:
        st.caption("Dernière partie selon la sélection/filtres actuels.")

        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            last_row = dff.sort_values("start_time").iloc[-1]
            last_match_id = str(last_row.get("match_id", ""))
            last_time = last_row.get("start_time")
            last_map = last_row.get("map_name")
            last_playlist = last_row.get("playlist_name")
            last_pair = last_row.get("pair_name")
            last_mode = last_row.get("game_variant_name")
            last_outcome = last_row.get("outcome")

            last_playlist_fr = translate_playlist_name(str(last_playlist)) if last_playlist else None
            last_pair_fr = translate_pair_name(str(last_pair)) if last_pair else None

            # Résultat + score (pour affichage compact)
            outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
            try:
                outcome_code = int(last_outcome) if last_outcome == last_outcome else None
            except Exception:
                outcome_code = None
            outcome_label = outcome_map.get(outcome_code, "?") if outcome_code is not None else "-"

            colors = HALO_COLORS.as_dict()
            if outcome_code == OUTCOME_CODES.WIN:
                outcome_color = colors["green"]
            elif outcome_code == OUTCOME_CODES.LOSS:
                outcome_color = colors["red"]
            elif outcome_code == OUTCOME_CODES.TIE:
                outcome_color = colors["violet"]
            else:
                outcome_color = colors["slate"]

            last_my_score = last_row.get("my_team_score")
            last_enemy_score = last_row.get("enemy_team_score")
            score_label = _format_score_label(last_my_score, last_enemy_score)
            score_color = _score_css_color(last_my_score, last_enemy_score)

            wp = str(waypoint_player or "").strip()
            match_url = None
            if wp and last_match_id and last_match_id.strip() and last_match_id.strip() != "-":
                match_url = f"https://www.halowaypoint.com/halo-infinite/players/{wp}/matches/{last_match_id.strip()}"

            top_cols = st.columns([2, 3])
            with top_cols[0]:
                top_cols_0 = st.columns([2, 3])
                with top_cols_0[0]:
                    st.metric("Date", format_date_fr(last_time))
                with top_cols_0[1]:
                    st.markdown(
                        "<div style='border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:10px 12px; background: rgba(255,255,255,0.03)'>"
                        "<div style='display:flex; align-items:baseline; gap:10px; justify-content:space-between'>"
                        f"<div style='font-weight:900; font-size:1.05rem; color:{outcome_color}'>{outcome_label}</div>"
                        f"<div style='font-weight:900; font-size:1.05rem; color:{score_color}'>{score_label}</div>"
                        "</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            # Carte + Playlist + Mode sur la même ligne
            last_mode_ui = last_row.get("mode_ui") or _normalize_mode_label(str(last_pair) if last_pair else None)
            row_cols = st.columns(3)
            row_cols[0].metric("Carte", str(last_map) if last_map else "-")
            row_cols[1].metric(
                "Playlist",
                str(last_playlist_fr or last_playlist) if (last_playlist_fr or last_playlist) else "-",
            )
            row_cols[2].metric(
                "Mode",
                str(last_mode_ui or last_pair_fr or last_pair or last_mode)
                if (last_mode_ui or last_pair_fr or last_pair or last_mode)
                else "-",
            )

            with st.spinner("Lecture des stats détaillées (attendu vs réel, médailles)…"):
                pm = cached_load_player_match_result(db_path, last_match_id, xuid.strip(), db_key=db_key)
                medals_last = cached_load_match_medals_for_player(
                    db_path, last_match_id, xuid.strip(), db_key=db_key
                )

            # (Le résultat + score est affiché en haut, compact, près de la date.)

            if not pm:
                st.info(
                    "Stats détaillées indisponibles pour ce match (PlayerMatchStats manquant ou format inattendu)."
                )
            else:
                team_mmr = pm.get("team_mmr")
                enemy_mmr = pm.get("enemy_mmr")
                delta_mmr = (team_mmr - enemy_mmr) if (team_mmr is not None and enemy_mmr is not None) else None

                mmr_cols = st.columns(3)
                mmr_cols[0].metric("MMR d'équipe", f"{team_mmr:.1f}" if team_mmr is not None else "-")
                mmr_cols[1].metric("MMR adverse", f"{enemy_mmr:.1f}" if enemy_mmr is not None else "-")
                if delta_mmr is None:
                    mmr_cols[2].metric("Écart MMR", "-")
                else:
                    mmr_cols[2].metric(
                        "Écart MMR (équipe - adverse)",
                        "-",
                        f"{float(delta_mmr):+.1f}",
                        delta_color="normal",
                    )

                if match_url:
                    with mmr_cols[2]:
                        st.link_button("Ouvrir sur HaloWaypoint", match_url, width="stretch")

                # Attendu vs réel (K / D) + ratios (match uniquement)
                def _metric_expected_vs_actual(title: str, perf: dict, delta_color: str) -> None:
                    count = perf.get("count")
                    expected = perf.get("expected")
                    if count is None or expected is None:
                        st.metric(title, "-")
                        return
                    delta = float(count) - float(expected)
                    st.metric(title, f"{count:.0f} vs {expected:.1f}", f"{delta:+.1f}", delta_color=delta_color)

                perf_k = pm.get("kills") or {}
                perf_d = pm.get("deaths") or {}
                perf_a = pm.get("assists") or {}

                st.subheader("Réel vs attendu")
                av_cols = st.columns(3)
                with av_cols[0]:
                    _metric_expected_vs_actual("Frags", perf_k, delta_color="normal")
                with av_cols[1]:
                    _metric_expected_vs_actual("Morts", perf_d, delta_color="inverse")
                with av_cols[2]:
                    avg_life_last = last_row.get("average_life_seconds")
                    st.metric("Durée de vie moyenne", format_mmss(avg_life_last))

                labels = ["F", "D", "A"]
                actual_vals = [
                    float(last_row.get("kills") or 0.0),
                    float(last_row.get("deaths") or 0.0),
                    float(last_row.get("assists") or 0.0),
                ]
                exp_vals = [
                    perf_k.get("expected"),
                    perf_d.get("expected"),
                    perf_a.get("expected"),
                ]

                show_expected_ratio = all(v is not None for v in exp_vals)

                real_ratio = last_row.get("ratio")
                try:
                    real_ratio_f = float(real_ratio) if real_ratio == real_ratio else None
                except Exception:
                    real_ratio_f = None
                if real_ratio_f is None:
                    denom = max(1.0, float(last_row.get("deaths") or 0.0))
                    real_ratio_f = (float(last_row.get("kills") or 0.0) + float(last_row.get("assists") or 0.0)) / denom

                exp_ratio_f = None
                if show_expected_ratio:
                    denom_e = max(1e-9, float(exp_vals[1] or 0.0))
                    exp_ratio_f = (float(exp_vals[0] or 0.0) + float(exp_vals[2] or 0.0)) / denom_e

                exp_fig = make_subplots(specs=[[{"secondary_y": True}]])

                bar_colors = [HALO_COLORS.green, HALO_COLORS.red, HALO_COLORS.cyan]
                exp_fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=exp_vals,
                        name="Attendu",
                        marker=dict(
                            color=bar_colors,
                            pattern=dict(shape="/", fgcolor="rgba(255,255,255,0.75)", solidity=0.22),
                        ),
                        opacity=0.50,
                        hovertemplate="%{x} (attendu): %{y:.1f}<extra></extra>",
                    ),
                    secondary_y=False,
                )
                exp_fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=actual_vals,
                        name="Réel",
                        marker_color=bar_colors,
                        opacity=0.90,
                        hovertemplate="%{x} (réel): %{y:.0f}<extra></extra>",
                    ),
                    secondary_y=False,
                )
                exp_fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=[real_ratio_f] * len(labels),
                        mode="lines+markers",
                        name="Ratio réel",
                        line=dict(color=HALO_COLORS.amber, width=4),
                        marker=dict(size=7),
                        hovertemplate="ratio (réel): %{y:.2f}<extra></extra>",
                    ),
                    secondary_y=True,
                )
                if exp_ratio_f is not None:
                    exp_fig.add_trace(
                        go.Scatter(
                            x=labels,
                            y=[exp_ratio_f] * len(labels),
                            mode="lines+markers",
                            name="Ratio attendu",
                            line=dict(color=HALO_COLORS.violet, width=3, dash="dot"),
                            marker=dict(size=6),
                            hovertemplate="ratio (attendu): %{y:.2f}<extra></extra>",
                        ),
                        secondary_y=True,
                    )
                else:
                    st.caption("Ratio attendu indisponible (Assists attendues manquantes dans PlayerMatchStats).")

                exp_fig.update_layout(
                    barmode="group",
                    height=360,
                    margin=dict(l=40, r=20, t=30, b=90),
                    legend=get_legend_horizontal_bottom(),
                )
                exp_fig.update_yaxes(title_text="F / D / A", rangemode="tozero", secondary_y=False)
                exp_fig.update_yaxes(title_text="Ratio", secondary_y=True)
                st.plotly_chart(exp_fig, width="stretch")

            # Médailles (match)
            st.subheader("Médailles")
            if not medals_last:
                st.info("Médailles indisponibles pour ce match (ou aucune médaille).")
            else:
                md_df = pd.DataFrame(medals_last)
                md_df["label"] = md_df["name_id"].apply(lambda x: medal_label(int(x)))
                md_df = md_df.sort_values(["count", "label"], ascending=[False, True])
                render_medals_grid(md_df[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)



    # --------------------------------------------------------------------------
    # Tab: Médailles (Top 25)
    # --------------------------------------------------------------------------
    with tab_medals:
        st.caption("Top 25 des médailles sur la sélection/filtres actuels.")

        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            match_ids = [str(x) for x in dff["match_id"].dropna().astype(str).tolist()]

            with st.spinner("Agrégation des médailles…"):
                top = _top_medals(db_path, xuid.strip(), match_ids, top_n=25, db_key=db_key)

            if not top:
                st.info("Aucune médaille trouvée (ou payload médailles absent).")
            else:
                md = pd.DataFrame(top, columns=["name_id", "count"])

                md["label"] = md["name_id"].apply(lambda x: medal_label(int(x)))
                md_desc = md.sort_values("count", ascending=False)
                render_medals_grid(md_desc[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)

    # --------------------------------------------------------------------------
    # Tab: Séries temporelles
    # --------------------------------------------------------------------------
    with tab_series:
        with st.spinner("Génération des graphes…"):
            fig = plot_timeseries(dff, title=f"{me_name}")
            st.plotly_chart(fig, width="stretch")

            st.subheader("FDA")
            valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
            if valid.empty:
                st.info("FDA indisponible sur ce filtre.")
            else:
                m = st.columns(1)
                m[0].metric("Moyenne FDA", f"{valid['kda'].mean():.2f}")
                st.caption("Densité (KDE) + rug : forme de la distribution + position des matchs.")
                st.plotly_chart(plot_kda_distribution(dff), width="stretch")

            st.subheader("Assistances")
            st.plotly_chart(plot_assists_timeseries(dff, title=f"{me_name} — Assistances"), width="stretch")

            st.subheader("Stats par minute")
            st.plotly_chart(
                plot_per_minute_timeseries(dff, title=f"{me_name} — Frags/Morts/Assistances par minute"),
                width="stretch",
            )

            st.subheader("Durée de vie moyenne")
            if dff.dropna(subset=["average_life_seconds"]).empty:
                st.info("Average Life indisponible sur ce filtre.")
            else:
                st.plotly_chart(plot_average_life(dff), width="stretch")

            st.subheader("Folie meurtrière / Tirs à la tête / Précision")
            st.plotly_chart(plot_spree_headshots_accuracy(dff), width="stretch")

    # --------------------------------------------------------------------------
    # Tab: Victoires/Défaites
    # --------------------------------------------------------------------------
    with tab_mom:
        with st.spinner("Calcul des victoires/défaites…"):
            # En mode Sessions, si on a sélectionné une ou plusieurs sessions on raisonne "session".
            # Si on est sur "(toutes)", on revient au bucket temporel normal (jour/semaine/mois...).
            current_mode = st.session_state.get("filter_mode")
            is_session_scope = bool(current_mode == "Sessions" and picked_session_labels)
            fig_out, bucket_label = plot_outcomes_over_time(dff, session_style=is_session_scope)
            st.markdown(
                f"Par **{bucket_label}** : on regroupe les parties par {bucket_label} et on compte le nombre de "
                "victoires/défaites (et autres statuts) pour suivre l'évolution."
            )
            st.caption("Basé sur Players[].Outcome (2=victoire, 3=défaite, 1=égalité, 4=non terminé).")
            st.plotly_chart(fig_out, width="stretch")

            st.subheader("Tableau — par période")
            d = dff.dropna(subset=["outcome"]).copy()
            if d.empty:
                st.info("Aucune donnée pour construire le tableau.")
            else:
                if is_session_scope:
                    d = d.sort_values("start_time").reset_index(drop=True)
                    if len(d.index) <= 20:
                        d["bucket"] = (d.index + 1)
                    else:
                        t = pd.to_datetime(d["start_time"], errors="coerce")
                        d["bucket"] = t.dt.floor("h")
                else:
                    tmin = pd.to_datetime(d["start_time"], errors="coerce").min()
                    tmax = pd.to_datetime(d["start_time"], errors="coerce").max()
                    dt_range = (tmax - tmin) if (tmin == tmin and tmax == tmax) else pd.Timedelta(days=999)
                    days = float(dt_range.total_seconds() / 86400.0) if dt_range is not None else 999.0
                    cfg = SESSION_CONFIG
                    if days < cfg.bucket_threshold_hourly:
                        d = d.sort_values("start_time").reset_index(drop=True)
                        d["bucket"] = (d.index + 1)
                    elif days <= cfg.bucket_threshold_daily:
                        d["bucket"] = d["start_time"].dt.floor("h")
                    elif days <= cfg.bucket_threshold_weekly:
                        d["bucket"] = d["start_time"].dt.to_period("D").astype(str)
                    elif days <= cfg.bucket_threshold_monthly:
                        d["bucket"] = d["start_time"].dt.to_period("W-MON").astype(str)
                    else:
                        d["bucket"] = d["start_time"].dt.to_period("M").astype(str)

                d["my_score"] = pd.to_numeric(d.get("my_team_score"), errors="coerce")
                d["enemy_score"] = pd.to_numeric(d.get("enemy_team_score"), errors="coerce")
                d["score_diff"] = d["my_score"] - d["enemy_score"]

                pivot = (
                    d.pivot_table(index="bucket", columns="outcome", values="match_id", aggfunc="count")
                    .fillna(0)
                    .astype(int)
                    .sort_index()
                )
                out_tbl = pd.DataFrame(index=pivot.index)
                out_tbl["Victoires"] = pivot[2] if 2 in pivot.columns else 0
                out_tbl["Défaites"] = pivot[3] if 3 in pivot.columns else 0
                out_tbl["Égalités"] = pivot[1] if 1 in pivot.columns else 0
                out_tbl["Non terminés"] = pivot[4] if 4 in pivot.columns else 0
                out_tbl["Total"] = out_tbl[["Victoires", "Défaites", "Égalités", "Non terminés"]].sum(axis=1)
                out_tbl["Win rate"] = (
                    100.0
                    * (out_tbl["Victoires"] / out_tbl["Total"].where(out_tbl["Total"] > 0))
                ).fillna(0.0)

                score_diff_avg = d.groupby("bucket")["score_diff"].mean(numeric_only=True)
                out_tbl["Diff score moy."] = score_diff_avg

                out_tbl = out_tbl.reset_index().rename(columns={"bucket": bucket_label.capitalize()})

                def _style_pct(v) -> str:
                    try:
                        x = float(v)
                    except Exception:
                        return ""
                    return "color: #E0E0E0; font-weight: 700;"

                out_styled = out_tbl.style.map(_style_pct, subset=["Win rate"]).map(_style_signed_number, subset=["Diff score moy."])
                st.dataframe(
                    out_styled,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Win rate": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
                        "Diff score moy.": st.column_config.NumberColumn("Diff score moy.", format="%.2f"),
                    },
                )

            st.divider()
            st.subheader("Ratio par cartes")
            st.caption("Compare tes performances par map.")

            scope = st.radio(
                "Scope",
                options=["Moi (filtres actuels)", "Moi (toutes les parties)", "Avec Madina972", "Avec Chocoboflor"],
                horizontal=True,
            )
            min_matches = st.slider(
                "Minimum de matchs par carte",
                1,
                30,
                1,
                step=1,
                key="min_matches_maps",
                on_change=_clear_min_matches_maps_auto,
            )

            if scope == "Moi (toutes les parties)":
                base_scope = base
            elif scope == "Avec Madina972":
                match_ids = set(
                    cached_same_team_match_ids_with_friend(
                        db_path,
                        xuid.strip(),
                        "2533274858283686",
                        db_key=db_key,
                    )
                )
                base_scope = base.loc[base["match_id"].astype(str).isin(match_ids)].copy()
            elif scope == "Avec Chocoboflor":
                match_ids = set(
                    cached_same_team_match_ids_with_friend(
                        db_path,
                        xuid.strip(),
                        "2535469190789936",
                        db_key=db_key,
                    )
                )
                base_scope = base.loc[base["match_id"].astype(str).isin(match_ids)].copy()
            else:
                base_scope = dff

            with st.spinner("Calcul des stats par carte…"):
                breakdown = compute_map_breakdown(base_scope)
                breakdown = breakdown.loc[breakdown["matches"] >= int(min_matches)].copy()

            if breakdown.empty:
                st.warning("Pas assez de matchs par map avec ces filtres.")
            else:
                metric = st.selectbox(
                    "Métrique",
                    options=[
                        ("ratio_global", "Ratio Victoire/défaite"),
                        ("win_rate", "Win rate"),
                        ("accuracy_avg", "Précision moyenne"),
                    ],
                    format_func=lambda x: x[1],
                )
                key, label = metric

                view = breakdown.head(20).iloc[::-1]
                title = f"{label} par carte — {scope} (min {min_matches} matchs)"
                if key == "ratio_global":
                    fig = plot_map_ratio_with_winloss(view, title=title)
                else:
                    fig = plot_map_comparison(view, key, title=title)

                if key in ("win_rate",):
                    fig.update_xaxes(tickformat=".0%")
                if key in ("accuracy_avg",):
                    fig.update_xaxes(ticksuffix="%")

                st.plotly_chart(fig, width="stretch")

                tbl = breakdown.copy()
                tbl["win_rate"] = (tbl["win_rate"] * 100).round(1)
                tbl["loss_rate"] = (tbl["loss_rate"] * 100).round(1)
                tbl["accuracy_avg"] = tbl["accuracy_avg"].round(2)
                tbl["ratio_global"] = tbl["ratio_global"].round(2)

                def _single_or_multi_label(series: pd.Series) -> str:
                    try:
                        vals = sorted({str(x).strip() for x in series.dropna().tolist() if str(x).strip()})
                    except Exception:
                        return "-"
                    if len(vals) == 0:
                        return "-"
                    if len(vals) == 1:
                        return vals[0]
                    return "Plusieurs"

                if "playlist_ui" in base_scope.columns:
                    playlist_ctx = _single_or_multi_label(base_scope["playlist_ui"])
                else:
                    playlist_ctx = _single_or_multi_label(
                        base_scope["playlist_name"].apply(_clean_asset_label).apply(translate_playlist_name)
                    )

                if "mode_ui" in base_scope.columns:
                    mode_ctx = _single_or_multi_label(base_scope["mode_ui"])
                else:
                    mode_ctx = _single_or_multi_label(base_scope["pair_name"].apply(_normalize_mode_label))

                tbl_disp = tbl.copy()
                tbl_disp["playlist_ctx"] = playlist_ctx
                tbl_disp["mode_ctx"] = mode_ctx
                tbl_disp = tbl_disp.rename(
                    columns={
                        "map_name": "Carte",
                        "matches": "Parties",
                        "accuracy_avg": "Précision moy. (%)",
                        "win_rate": "Taux victoire (%)",
                        "loss_rate": "Taux défaite (%)",
                        "ratio_global": "Ratio global",
                        "playlist_ctx": "Playlist",
                        "mode_ctx": "Mode",
                    }
                )
                ordered_cols = [
                    "Carte",
                    "Playlist",
                    "Mode",
                    "Parties",
                    "Précision moy. (%)",
                    "Taux victoire (%)",
                    "Taux défaite (%)",
                    "Ratio global",
                ]
                tbl_disp = tbl_disp[[c for c in ordered_cols if c in tbl_disp.columns]]
                st.dataframe(tbl_disp, width="stretch", hide_index=True)

            # (Le graphique "Net victoires/défaites" a été retiré :
            #  l'onglet se concentre sur le graphe Victoires au-dessus / Défaites en dessous.)

    # --------------------------------------------------------------------------
    # Tab: Mes coéquipiers (fusion)
    # --------------------------------------------------------------------------
    with tab_teammates:
        st.caption("Vue dédiée aux matchs joués avec tes coéquipiers (XUID rencontrés / alias).")

        apply_current_filters_teammates = st.toggle(
            "Appliquer les filtres actuels (période/sessions + map/playlist)",
            value=True,
            key="apply_current_filters_teammates",
        )
        same_team_only_teammates = st.checkbox("Même équipe", value=True, key="teammates_same_team_only")

        opts_map, default_labels = _build_friends_opts_map(db_path, xuid.strip(), db_key, aliases_key)
        picked_labels = st.multiselect(
            "Coéquipiers",
            options=list(opts_map.keys()),
            default=default_labels,
            key="teammates_picked_labels",
        )
        picked_xuids = [opts_map[lbl] for lbl in picked_labels if lbl in opts_map]

        if len(picked_xuids) < 1:
            st.info("Sélectionne au moins un coéquipier.")
        elif len(picked_xuids) == 1:
            friend_xuid = picked_xuids[0]
            with st.spinner("Chargement des matchs avec ce coéquipier…"):
                dfr = cached_friend_matches_df(
                    db_path,
                    xuid.strip(),
                    friend_xuid,
                    same_team_only=bool(same_team_only_teammates),
                    db_key=db_key,
                )
                if dfr.empty:
                    st.warning("Aucun match trouvé avec ce coéquipier (selon le filtre).")
                else:
                    outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
                    dfr["my_outcome_label"] = dfr["my_outcome"].map(outcome_map).fillna("?")
                    counts = dfr["my_outcome_label"].value_counts().reindex(
                        ["Victoire", "Défaite", "Égalité", "Non terminé", "?"], fill_value=0
                    )
                    colors = HALO_COLORS.as_dict()
                    fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color=colors["cyan"])])
                    fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
                    st.plotly_chart(fig, width="stretch")

                    with st.expander("Détails des matchs (joueur vs joueur)", expanded=False):
                        st.dataframe(
                            dfr[
                                [
                                    "start_time",
                                    "playlist_name",
                                    "pair_name",
                                    "same_team",
                                    "my_team_id",
                                    "my_outcome",
                                    "friend_team_id",
                                    "friend_outcome",
                                    "match_id",
                                ]
                            ].reset_index(drop=True),
                            width="stretch",
                            hide_index=True,
                        )

                    base_for_friend = dff if apply_current_filters_teammates else df
                    shared_ids = set(dfr["match_id"].astype(str))
                    sub = base_for_friend.loc[base_for_friend["match_id"].astype(str).isin(shared_ids)].copy()

                    if sub.empty:
                        st.info("Aucun match à afficher avec les filtres actuels (période/sessions + map/playlist).")
                    else:
                        name = display_name_from_xuid(friend_xuid)

                        rates_sub = compute_outcome_rates(sub)
                        total_out = max(1, rates_sub.total)
                        win_rate_sub = rates_sub.wins / total_out
                        loss_rate_sub = rates_sub.losses / total_out
                        global_ratio_sub = compute_global_ratio(sub)

                        k = st.columns(3)
                        k[0].metric("Matchs", f"{len(sub)}")
                        k[1].metric("Win/Loss", f"{win_rate_sub*100:.1f}% / {loss_rate_sub*100:.1f}%")
                        k[2].metric("Ratio global", f"{global_ratio_sub:.2f}" if global_ratio_sub is not None else "-")

                        stats_sub = compute_aggregated_stats(sub)
                        per_min = st.columns(3)
                        per_min[0].metric(
                            "Frags / min",
                            f"{stats_sub.kills_per_minute:.2f}" if stats_sub.kills_per_minute else "-",
                        )
                        per_min[1].metric(
                            "Morts / min",
                            f"{stats_sub.deaths_per_minute:.2f}" if stats_sub.deaths_per_minute else "-",
                        )
                        per_min[2].metric(
                            "Assistances / min",
                            f"{stats_sub.assists_per_minute:.2f}" if stats_sub.assists_per_minute else "-",
                        )

                        friend_df = load_df(db_path, friend_xuid, db_key=db_key)
                        friend_sub = friend_df.loc[friend_df["match_id"].astype(str).isin(shared_ids)].copy()

                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(plot_timeseries(sub, title=f"{me_name} — matchs avec {name}"), width="stretch")
                        with c2:
                            if friend_sub.empty:
                                st.warning("Impossible de charger les stats du coéquipier sur les matchs partagés.")
                            else:
                                st.plotly_chart(
                                    plot_timeseries(friend_sub, title=f"{name} — matchs avec {me_name}"),
                                    width="stretch",
                                )

                        c3, c4 = st.columns(2)
                        with c3:
                            st.plotly_chart(
                                plot_per_minute_timeseries(sub, title=f"{me_name} — stats/min (avec {name})"),
                                width="stretch",
                            )
                        with c4:
                            if not friend_sub.empty:
                                st.plotly_chart(
                                    plot_per_minute_timeseries(
                                        friend_sub,
                                        title=f"{name} — stats/min (avec {me_name})",
                                    ),
                                    width="stretch",
                                )

                        c5, c6 = st.columns(2)
                        with c5:
                            if not sub.dropna(subset=["average_life_seconds"]).empty:
                                st.plotly_chart(
                                    plot_average_life(sub, title=f"{me_name} — Durée de vie (avec {name})"),
                                    width="stretch",
                                )
                        with c6:
                            if not friend_sub.empty and not friend_sub.dropna(subset=["average_life_seconds"]).empty:
                                st.plotly_chart(
                                    plot_average_life(friend_sub, title=f"{name} — Durée de vie (avec {me_name})"),
                                    width="stretch",
                                )

                        # Avant-dernier: Folie meurtrière / Tirs à la tête (style barres + moyenne lissée) puis médailles.
                        st.subheader("Folie meurtrière (max) & tirs à la tête")
                        colors = HALO_COLORS.as_dict()
                        s1, s2 = st.columns(2)
                        with s1:
                            st.caption(f"{me_name}")
                            fig_spree_me = _plot_metric_bars_by_match(
                                sub,
                                metric_col="max_killing_spree",
                                title="Folie meurtrière (max)",
                                y_axis_title="Folie meurtrière (max)",
                                hover_label="folie meurtrière",
                                bar_color=colors["amber"],
                                smooth_color=colors["green"],
                                smooth_window=10,
                            )
                            if fig_spree_me is not None:
                                st.plotly_chart(fig_spree_me, width="stretch")
                            fig_hs_me = _plot_metric_bars_by_match(
                                sub,
                                metric_col="headshot_kills",
                                title="Tirs à la tête",
                                y_axis_title="Tirs à la tête",
                                hover_label="tirs à la tête",
                                bar_color=colors["red"],
                                smooth_color=colors["green"],
                                smooth_window=10,
                            )
                            if fig_hs_me is not None:
                                st.plotly_chart(fig_hs_me, width="stretch")

                        with s2:
                            st.caption(f"{name}")
                            if friend_sub.empty:
                                st.info("Stats du coéquipier indisponibles pour ces matchs.")
                            else:
                                fig_spree_fr = _plot_metric_bars_by_match(
                                    friend_sub,
                                    metric_col="max_killing_spree",
                                    title="Folie meurtrière (max)",
                                    y_axis_title="Folie meurtrière (max)",
                                    hover_label="folie meurtrière",
                                    bar_color=colors["amber"],
                                    smooth_color=colors["green"],
                                    smooth_window=10,
                                )
                                if fig_spree_fr is not None:
                                    st.plotly_chart(fig_spree_fr, width="stretch")
                                fig_hs_fr = _plot_metric_bars_by_match(
                                    friend_sub,
                                    metric_col="headshot_kills",
                                    title="Tirs à la tête",
                                    y_axis_title="Tirs à la tête",
                                    hover_label="tirs à la tête",
                                    bar_color=colors["red"],
                                    smooth_color=colors["green"],
                                    smooth_window=10,
                                )
                                if fig_hs_fr is not None:
                                    st.plotly_chart(fig_hs_fr, width="stretch")

                        st.subheader("Médailles (matchs partagés)")
                        shared_list = sorted({str(x) for x in shared_ids if str(x).strip()})
                        if not shared_list:
                            st.info("Aucun match partagé pour calculer les médailles.")
                        else:
                            with st.spinner("Agrégation des médailles (moi + coéquipier)…"):
                                my_top = _top_medals(db_path, xuid.strip(), shared_list, top_n=12, db_key=db_key)
                                fr_top = _top_medals(db_path, friend_xuid, shared_list, top_n=12, db_key=db_key)

                            m1, m2 = st.columns(2)
                            with m1:
                                st.caption(f"{me_name}")
                                render_medals_grid(
                                    [{"name_id": int(n), "count": int(c)} for n, c in (my_top or [])],
                                    cols_per_row=6,
                                )
                            with m2:
                                st.caption(f"{name}")
                                render_medals_grid(
                                    [{"name_id": int(n), "count": int(c)} for n, c in (fr_top or [])],
                                    cols_per_row=6,
                                )
        else:
            st.subheader("Par carte — avec mes coéquipiers")
            with st.spinner("Calcul du ratio par carte (coéquipiers)…"):
                current_mode = st.session_state.get("filter_mode")
                latest_session_label = st.session_state.get("_latest_session_label")
                trio_latest_label = st.session_state.get("_trio_latest_session_label")

                selected_session = None
                if current_mode == "Sessions" and isinstance(picked_session_labels, list) and len(picked_session_labels) == 1:
                    selected_session = picked_session_labels[0]

                is_last_session = bool(selected_session and selected_session == latest_session_label)
                is_last_trio_session = bool(selected_session and isinstance(trio_latest_label, str) and selected_session == trio_latest_label)

                if is_last_session or is_last_trio_session:
                    last_applied = st.session_state.get("_friends_min_matches_last_session_label")
                    if last_applied != selected_session:
                        st.session_state["min_matches_maps_friends"] = 1
                        st.session_state["_min_matches_maps_friends_auto"] = True
                        st.session_state["_friends_min_matches_last_session_label"] = selected_session

                min_matches_maps_friends = st.slider(
                    "Minimum de matchs par carte",
                    1,
                    30,
                    1,
                    step=1,
                    key="min_matches_maps_friends",
                    on_change=_clear_min_matches_maps_friends_auto,
                )

                base_for_friends_all = dff if apply_current_filters_teammates else df
                all_match_ids: set[str] = set()
                per_friend_ids: dict[str, set[str]] = {}
                for fx in picked_xuids:
                    ids: set[str] = set()
                    if bool(same_team_only_teammates):
                        ids = {str(x) for x in cached_same_team_match_ids_with_friend(db_path, xuid.strip(), fx, db_key=db_key)}
                    else:
                        rows = cached_query_matches_with_friend(db_path, xuid.strip(), fx, db_key=db_key)
                        ids = {str(r.match_id) for r in rows}
                    per_friend_ids[str(fx)] = ids
                    all_match_ids.update(ids)

                sub_all = base_for_friends_all.loc[
                    base_for_friends_all["match_id"].astype(str).isin(all_match_ids)
                ].copy()

                # Graphes demandés: style identique (barres + moyenne lissée), un sous-onglet par coéquipier.
                st.subheader("Folie meurtrière (max) & tirs à la tête")
                colors = HALO_COLORS.as_dict()
                tab_labels = [display_name_from_xuid(fx) for fx in picked_xuids]
                max_tabs = 8
                if len(tab_labels) > max_tabs:
                    st.warning(f"Beaucoup de coéquipiers sélectionnés : affichage limité aux {max_tabs} premiers pour garder l'UI lisible.")
                use_xuids = picked_xuids[:max_tabs]
                use_tabs = tab_labels[:max_tabs]
                friend_tabs = st.tabs(use_tabs)
                for tab_obj, fx in zip(friend_tabs, use_xuids):
                    with tab_obj:
                        ids = per_friend_ids.get(str(fx), set())
                        sub_fx = base_for_friends_all.loc[
                            base_for_friends_all["match_id"].astype(str).isin(ids)
                        ].copy()
                        g1, g2 = st.columns(2)
                        with g1:
                            fig_spree = _plot_metric_bars_by_match(
                                sub_fx,
                                metric_col="max_killing_spree",
                                title="Folie meurtrière (max)",
                                y_axis_title="Folie meurtrière (max)",
                                hover_label="folie meurtrière",
                                bar_color=colors["amber"],
                                smooth_color=colors["green"],
                                smooth_window=10,
                            )
                            if fig_spree is None:
                                st.info("Aucune donnée de folie meurtrière (max) sur ces matchs.")
                            else:
                                st.plotly_chart(fig_spree, width="stretch")

                        with g2:
                            fig_hs = _plot_metric_bars_by_match(
                                sub_fx,
                                metric_col="headshot_kills",
                                title="Tirs à la tête",
                                y_axis_title="Tirs à la tête",
                                hover_label="tirs à la tête",
                                bar_color=colors["red"],
                                smooth_color=colors["green"],
                                smooth_window=10,
                            )
                            if fig_hs is None:
                                st.info("Aucune donnée de tirs à la tête sur ces matchs.")
                            else:
                                st.plotly_chart(fig_hs, width="stretch")

                breakdown_all = compute_map_breakdown(sub_all)
                breakdown_all = breakdown_all.loc[breakdown_all["matches"] >= int(min_matches_maps_friends)].copy()

                if breakdown_all.empty:
                    st.info("Pas assez de matchs avec tes coéquipiers (selon le filtre actuel).")
                else:
                    view_all = breakdown_all.head(20).iloc[::-1]
                    title = f"Ratio global par carte — avec mes coéquipiers (min {min_matches_maps_friends} matchs)"
                    st.plotly_chart(plot_map_ratio_with_winloss(view_all, title=title), width="stretch")

                st.subheader("Historique — matchs avec mes coéquipiers")
                if sub_all.empty:
                    st.info("Aucun match trouvé avec tes coéquipiers (selon le filtre actuel).")
                else:
                    friends_table = sub_all.copy()
                    friends_table["start_time_fr"] = friends_table["start_time"].apply(_format_datetime_fr_hm)
                    if "playlist_fr" not in friends_table.columns:
                        friends_table["playlist_fr"] = friends_table["playlist_name"].apply(translate_playlist_name)
                    if "mode_ui" in friends_table.columns:
                        friends_table["mode"] = friends_table["mode_ui"].apply(
                            lambda x: x if (x is not None and str(x).strip()) else None
                        )
                    else:
                        friends_table["mode"] = None
                    if friends_table["mode"].isna().any() and "pair_name" in friends_table.columns:
                        mask = friends_table["mode"].isna()
                        friends_table.loc[mask, "mode"] = friends_table.loc[mask, "pair_name"].apply(
                            lambda p: _normalize_mode_label(str(p) if p is not None else None)
                        )
                    friends_table["mode"] = friends_table["mode"].fillna("-")
                    friends_table["outcome_label"] = friends_table["outcome"].map({2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}).fillna("-")
                    friends_table["score"] = friends_table.apply(
                        lambda r: _format_score_label(r.get("my_team_score"), r.get("enemy_team_score")), axis=1
                    )
                    wp = str(waypoint_player or "").strip()
                    if wp:
                        friends_table["match_url"] = (
                            "https://www.halowaypoint.com/halo-infinite/players/"
                            + wp
                            + "/matches/"
                            + friends_table["match_id"].astype(str)
                        )
                    else:
                        friends_table["match_url"] = ""

                    friends_show = [
                        "match_url",
                        "start_time_fr",
                        "map_name",
                        "playlist_fr",
                        "mode",
                        "outcome_label",
                        "score",
                        "kda",
                        "kills",
                        "deaths",
                        "assists",
                        "accuracy",
                    ]
                    friends_view = (
                        friends_table.sort_values("start_time", ascending=False)[friends_show]
                        .reset_index(drop=True)
                    )
                    friends_styled = (
                        friends_view.style
                        .map(_style_outcome_text, subset=["outcome_label"])
                        .map(_style_score_label, subset=["score"])
                        .map(_style_signed_number, subset=["kda"])
                    )
                    st.dataframe(
                        friends_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "match_url": st.column_config.LinkColumn("HaloWaypoint", display_text="Ouvrir"),
                            "start_time_fr": st.column_config.TextColumn("Date"),
                            "map_name": st.column_config.TextColumn("Carte"),
                            "playlist_fr": st.column_config.TextColumn("Playlist"),
                            "mode": st.column_config.TextColumn("Mode"),
                            "outcome_label": st.column_config.TextColumn("Résultat"),
                            "score": st.column_config.TextColumn("Score"),
                            "kda": st.column_config.NumberColumn("FDA", format="%.2f"),
                            "kills": st.column_config.NumberColumn("Frags"),
                            "deaths": st.column_config.NumberColumn("Morts"),
                            "assists": st.column_config.NumberColumn("Assistances"),
                            "accuracy": st.column_config.NumberColumn("Précision (%)", format="%.2f"),
                        },
                    )

            # Vue trio (moi + 2 coéquipiers) : uniquement si on a au moins deux personnes.
            if len(picked_xuids) >= 2:
                f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
                f1_name = display_name_from_xuid(f1_xuid)
                f2_name = display_name_from_xuid(f2_xuid)
                st.subheader(f"Tous les trois — {f1_name} + {f2_name}")

                ids_m = set(
                    cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f1_xuid, db_key=db_key)
                )
                ids_c = set(
                    cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f2_xuid, db_key=db_key)
                )
                trio_ids = ids_m & ids_c

                base_for_trio = dff if apply_current_filters_teammates else df
                trio_ids = trio_ids & set(base_for_trio["match_id"].astype(str))

                if not trio_ids:
                    st.warning("Aucun match trouvé où vous êtes tous les trois dans la même équipe (avec les filtres actuels).")
                else:
                    # UX: bouton pour basculer les filtres globaux sur la dernière session où vous êtes tous les trois.
                    # On utilise la logique de sessions globale (mêmes paramètres gap) pour que la session sélectionnée
                    # corresponde bien aux options de la sidebar.
                    trio_ids_set = {str(x) for x in trio_ids}
                    base_for_session_pick = base  # même base que les filtres globaux (Firefight déjà exclu si besoin)
                    try:
                        gm = int(st.session_state.get("gap_minutes", 120))
                    except Exception:
                        gm = 120
                    base_s_trio = cached_compute_sessions_db(
                        db_path,
                        xuid.strip(),
                        db_key,
                        include_firefight,
                        gm,
                    )
                    trio_rows = base_s_trio.loc[base_s_trio["match_id"].astype(str).isin(trio_ids_set)].copy()
                    latest_label = None
                    if not trio_rows.empty:
                        latest_sid = int(trio_rows["session_id"].max())
                        latest_labels = trio_rows.loc[trio_rows["session_id"] == latest_sid, "session_label"].dropna().unique().tolist()
                        latest_label = latest_labels[0] if latest_labels else None

                    st.session_state["_trio_latest_session_label"] = latest_label
                    if latest_label:
                        st.caption(f"Dernière session trio détectée : {latest_label} (gap {gm} min).")
                    else:
                        st.caption("Impossible de déterminer une session trio (données insuffisantes).")

                    # Charge les stats de chacun et aligne par match_id.
                    me_df = base_for_trio.loc[base_for_trio["match_id"].isin(trio_ids)].copy()
                    f1_df = load_df(db_path, f1_xuid, db_key=db_key)
                    f2_df = load_df(db_path, f2_xuid, db_key=db_key)
                    f1_df = f1_df.loc[f1_df["match_id"].isin(trio_ids)].copy()
                    f2_df = f2_df.loc[f2_df["match_id"].isin(trio_ids)].copy()

                    # Aligne sur les mêmes match_id et utilise le start_time de moi comme référence d'axe.
                    me_df = me_df.sort_values("start_time")

                    # Tableau récap: stats/min pour les 3 joueurs sur ces matchs.
                    me_stats = compute_aggregated_stats(me_df)
                    f1_stats = compute_aggregated_stats(f1_df)
                    f2_stats = compute_aggregated_stats(f2_df)
                    trio_per_min = pd.DataFrame(
                        [
                            {
                                "Joueur": me_name,
                                "Frags/min": round(float(me_stats.kills_per_minute), 2) if me_stats.kills_per_minute else None,
                                "Morts/min": round(float(me_stats.deaths_per_minute), 2) if me_stats.deaths_per_minute else None,
                                "Assists/min": round(float(me_stats.assists_per_minute), 2) if me_stats.assists_per_minute else None,
                            },
                            {
                                "Joueur": f1_name,
                                "Frags/min": round(float(f1_stats.kills_per_minute), 2) if f1_stats.kills_per_minute else None,
                                "Morts/min": round(float(f1_stats.deaths_per_minute), 2) if f1_stats.deaths_per_minute else None,
                                "Assists/min": round(float(f1_stats.assists_per_minute), 2) if f1_stats.assists_per_minute else None,
                            },
                            {
                                "Joueur": f2_name,
                                "Frags/min": round(float(f2_stats.kills_per_minute), 2) if f2_stats.kills_per_minute else None,
                                "Morts/min": round(float(f2_stats.deaths_per_minute), 2) if f2_stats.deaths_per_minute else None,
                                "Assists/min": round(float(f2_stats.assists_per_minute), 2) if f2_stats.assists_per_minute else None,
                            },
                        ]
                    )
                    st.subheader("Stats par minute")
                    st.dataframe(trio_per_min, width="stretch", hide_index=True)

                    f1_df = f1_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
                    f2_df = f2_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
                    merged = me_df[["match_id", "start_time", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].merge(
                        f1_df.add_prefix("f1_"), left_on="match_id", right_on="f1_match_id", how="inner"
                    ).merge(
                        f2_df.add_prefix("f2_"), left_on="match_id", right_on="f2_match_id", how="inner"
                    )
                    if merged.empty:
                        st.warning("Impossible d'aligner les stats des 3 joueurs sur ces matchs.")
                    else:
                        merged = merged.sort_values("start_time")
                        # Reconstitue 3 DF alignées pour le plot.
                        d_self = merged[["start_time", "kills", "deaths", "assists", "ratio", "accuracy", "average_life_seconds"]].copy()
                        d_f1 = merged[["start_time", "f1_kills", "f1_deaths", "f1_assists", "f1_ratio", "f1_accuracy", "f1_average_life_seconds"]].rename(
                            columns={
                                "f1_kills": "kills",
                                "f1_deaths": "deaths",
                                "f1_assists": "assists",
                                "f1_ratio": "ratio",
                                "f1_accuracy": "accuracy",
                                "f1_average_life_seconds": "average_life_seconds",
                            }
                        )
                        d_f2 = merged[["start_time", "f2_kills", "f2_deaths", "f2_assists", "f2_ratio", "f2_accuracy", "f2_average_life_seconds"]].rename(
                            columns={
                                "f2_kills": "kills",
                                "f2_deaths": "deaths",
                                "f2_assists": "assists",
                                "f2_ratio": "ratio",
                                "f2_accuracy": "accuracy",
                                "f2_average_life_seconds": "average_life_seconds",
                            }
                        )

                        names = (me_name, f1_name, f2_name)
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="kills", names=names, title="Frags", y_title="Frags"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="deaths", names=names, title="Morts", y_title="Morts"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="assists", names=names, title="Assistances", y_title="Assists"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="ratio", names=names, title="FDA", y_title="FDA", y_format=".3f"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="accuracy", names=names, title="Précision", y_title="%", y_suffix="%", y_format=".2f"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="average_life_seconds", names=names, title="Durée de vie moyenne", y_title="Secondes", y_format=".1f"),
                            width="stretch",
                        )

                        st.subheader("Médailles")
                        trio_match_ids = [str(x) for x in merged["match_id"].dropna().astype(str).tolist()]
                        if not trio_match_ids:
                            st.info("Impossible de déterminer la liste des matchs pour l'agrégation des médailles.")
                        else:
                            with st.spinner("Agrégation des médailles…"):
                                top_self = _top_medals(db_path, xuid.strip(), trio_match_ids, top_n=12, db_key=db_key)
                                top_f1 = _top_medals(db_path, f1_xuid, trio_match_ids, top_n=12, db_key=db_key)
                                top_f2 = _top_medals(db_path, f2_xuid, trio_match_ids, top_n=12, db_key=db_key)

                            c1, c2, c3 = st.columns(3)
                            with c1:
                                with st.expander(f"{me_name}", expanded=True):
                                    render_medals_grid(
                                        [{"name_id": int(n), "count": int(c)} for n, c in (top_self or [])],
                                        cols_per_row=6,
                                    )
                            with c2:
                                with st.expander(f"{f1_name}", expanded=True):
                                    render_medals_grid(
                                        [{"name_id": int(n), "count": int(c)} for n, c in (top_f1 or [])],
                                        cols_per_row=6,
                                    )
                            with c3:
                                with st.expander(f"{f2_name}", expanded=True):
                                    render_medals_grid(
                                        [{"name_id": int(n), "count": int(c)} for n, c in (top_f2 or [])],
                                        cols_per_row=6,
                                    )


    # --------------------------------------------------------------------------
    # Tab: Historique des parties
    # --------------------------------------------------------------------------
    with tab_table:
        st.subheader("Historique des parties")
        dff_table = dff.copy()
        if "playlist_fr" not in dff_table.columns:
            dff_table["playlist_fr"] = dff_table["playlist_name"].apply(translate_playlist_name)
        if "mode_ui" not in dff_table.columns:
            dff_table["mode_ui"] = dff_table["pair_name"].apply(_normalize_mode_label)
        dff_table["match_url"] = (
            "https://www.halowaypoint.com/halo-infinite/players/"
            + waypoint_player.strip()
            + "/matches/"
            + dff_table["match_id"].astype(str)
        )

        outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
        dff_table["outcome_label"] = dff_table["outcome"].map(outcome_map).fillna("-")

        dff_table["score"] = dff_table.apply(
            lambda r: _format_score_label(r.get("my_team_score"), r.get("enemy_team_score")), axis=1
        )

        dff_table["average_life_mmss"] = dff_table["average_life_seconds"].apply(lambda x: format_mmss(x))

        show_cols = [
            "match_url", "start_time", "map_name", "playlist_fr", "mode_ui", "outcome_label", "score",
            "kda", "kills", "deaths", "max_killing_spree", "headshot_kills",
            "average_life_mmss", "assists", "accuracy", "ratio",
        ]
        table = dff_table[show_cols].sort_values("start_time", ascending=False).reset_index(drop=True)

        styled = (
            table.style
            .map(_style_outcome_text, subset=["outcome_label"])
            .map(_style_score_label, subset=["score"])
            .map(_style_signed_number, subset=["kda"])
        )

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "match_url": st.column_config.LinkColumn(
                    "Consulter sur HaloWaypoint",
                    display_text="Ouvrir",
                ),
                "map_name": st.column_config.TextColumn("Carte"),
                "playlist_fr": st.column_config.TextColumn("Playlist"),
                "mode_ui": st.column_config.TextColumn("Mode"),
                "outcome_label": st.column_config.TextColumn("Résultat"),
                "score": st.column_config.TextColumn("Score"),
                "kda": st.column_config.NumberColumn("FDA", format="%.2f"),
                "kills": st.column_config.NumberColumn("Frags"),
                "deaths": st.column_config.NumberColumn("Morts"),
                "max_killing_spree": st.column_config.NumberColumn("Folie meurtrière (max)", format="%d"),
                "headshot_kills": st.column_config.NumberColumn("Tirs à la tête", format="%d"),
                "average_life_mmss": st.column_config.TextColumn("Durée de vie moyenne"),
                "assists": st.column_config.NumberColumn("Assistances"),
                "accuracy": st.column_config.NumberColumn("Précision (%)", format="%.2f"),
            },
        )

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger CSV",
            data=csv,
            file_name="openspartan_matches.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
