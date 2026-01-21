"""OpenSpartan Graphs - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données OpenSpartan Workshop.
"""

import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
import uuid
from datetime import date, datetime, time, timedelta, timezone
from collections.abc import Mapping
from typing import Optional
from zoneinfo import ZoneInfo

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
    load_highlight_events_for_match,
    load_match_player_gamertags,
    has_table,
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
)
from src.ui.medals import (
    load_medal_name_maps,
    medal_has_known_label,
    get_medals_cache_dir,
    medal_label,
    medal_icon_path,
    render_medals_grid,
)
from src.ui.commendations import render_h5g_commendations_section
from src.ui.formatting import format_date_fr
from src.db.profiles import (
    PROFILES_PATH,
    load_profiles,
    save_profiles,
    list_local_dbs,
)
from src.config import DEFAULT_PLAYER_GAMERTAG, DEFAULT_PLAYER_XUID, get_aliases_file_path

from src.ui.perf import perf_reset_run, perf_section, render_perf_panel
from src.ui.sections import render_openspartan_tools, render_source_section


_LABEL_SUFFIX_RE = re.compile(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", re.IGNORECASE)
_SCORE_LABEL_RE = re.compile(r"^\s*(-?\d+)\s*[-–—]\s*(-?\d+)\s*$")


PARIS_TZ_NAME = "Europe/Paris"
PARIS_TZ = ZoneInfo(PARIS_TZ_NAME)


def _to_paris_naive(dt_value) -> datetime | None:
    """Convertit une date en datetime naïf (sans tzinfo) en heure de Paris.

    - tz-aware -> convertit en Europe/Paris puis enlève tzinfo
    - naïf -> supposé déjà en heure de Paris
    """
    if dt_value is None:
        return None
    try:
        ts = pd.to_datetime(dt_value, errors="coerce")
        if pd.isna(ts):
            return None

        # pandas.Timestamp: ts.tz != None si tz-aware
        try:
            if getattr(ts, "tz", None) is not None:
                ts = ts.tz_convert(PARIS_TZ_NAME).tz_localize(None)
        except Exception:
            pass

        d = ts.to_pydatetime()
        if getattr(d, "tzinfo", None) is not None:
            d = d.astimezone(PARIS_TZ).replace(tzinfo=None)
        return d
    except Exception:
        return None


def _paris_epoch_seconds(dt_value) -> float | None:
    """Retourne un timestamp Unix (UTC) pour une date exprimée en heure de Paris."""
    d = _to_paris_naive(dt_value)
    if d is None:
        return None
    try:
        aware = d.replace(tzinfo=PARIS_TZ)
        return float(aware.astimezone(timezone.utc).timestamp())
    except Exception:
        return None


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


def _pick_latest_spnkr_db_if_any() -> str:
    try:
        repo_root = Path(__file__).resolve().parent
        data_dir = repo_root / "data"
        if not data_dir.exists():
            return ""
        candidates = [p for p in data_dir.glob("spnkr*.db") if p.is_file()]
        if not candidates:
            return ""
        # On évite de sélectionner une DB vide (0 octet), ce qui bloque l'app (aucune table).
        non_empty = [p for p in candidates if p.exists() and p.stat().st_size > 0]
        non_empty.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        if non_empty:
            return str(non_empty[0])
        # Fallback: si tout est vide, retourne quand même la plus récente pour debug.
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        return str(candidates[0])
    except Exception:
        return ""


def _is_spnkr_db_path(db_path: str) -> bool:
    try:
        p = Path(db_path)
        return p.suffix.lower() == ".db" and p.name.lower().startswith("spnkr")
    except Exception:
        return False


def _refresh_spnkr_db_via_api(
    *,
    db_path: str,
    player: str,
    match_type: str,
    max_matches: int,
    rps: int,
    with_highlight_events: bool,
    timeout_seconds: int = 180,
) -> tuple[bool, str]:
    """Rafraîchit une DB SPNKr en appelant scripts/spnkr_import_db.py.

    Retourne (ok, message) pour affichage UI.
    """
    repo_root = Path(__file__).resolve().parent
    importer = repo_root / "scripts" / "spnkr_import_db.py"
    if not importer.exists():
        return False, f"Script introuvable: {importer}"

    p = (player or "").strip()
    if not p:
        return False, "Aucun joueur pour SPNKr (gamertag ou XUID)."

    mt = (match_type or "matchmaking").strip().lower()
    if mt not in {"all", "matchmaking", "custom", "local"}:
        mt = "matchmaking"

    target = str(db_path)
    # IMPORTANT: on n'écrit jamais directement dans la DB cible.
    # Si l'import crashe/timeout, SQLite peut laisser un fichier vide/corrompu.
    tmp = f"{target}.tmp.{uuid.uuid4().hex}.db"

    cmd = [
        sys.executable,
        str(importer),
        "--out-db",
        str(tmp),
        "--player",
        p,
        "--match-type",
        mt,
        "--max-matches",
        str(int(max_matches)),
        "--requests-per-second",
        str(int(rps)),
        "--resume",
    ]
    if with_highlight_events:
        cmd.append("--with-highlight-events")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False, f"Timeout après {timeout_seconds}s (import SPNKr trop long)."
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False, f"Erreur au lancement de l'import SPNKr: {e}"

    if int(proc.returncode) != 0:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 1200:
            tail = tail[-1200:]
        return False, f"Import SPNKr en échec (code={proc.returncode}).\n{tail}".strip()

    # Remplace la DB cible uniquement si le tmp semble valide (non vide).
    try:
        if not os.path.exists(tmp) or os.path.getsize(tmp) <= 0:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False, "Import SPNKr terminé mais DB temporaire vide (annulé)."
        os.makedirs(str(Path(target).resolve().parent), exist_ok=True)
        os.replace(tmp, target)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False, f"Import SPNKr OK mais remplacement de la DB a échoué: {e}"

    return True, "DB SPNKr rafraîchie."


def _init_source_state(default_db: str, settings: AppSettings) -> None:
    if "db_path" not in st.session_state:
        chosen = str(default_db or "")
        # Si l'utilisateur force une DB via env (OPENSPARTAN_DB_PATH/OPENSPARTAN_DB),
        # on ne doit pas l'écraser par une auto-sélection SPNKr.
        forced_env_db = str(os.environ.get("OPENSPARTAN_DB") or os.environ.get("OPENSPARTAN_DB_PATH") or "").strip()
        if (not forced_env_db) and bool(getattr(settings, "prefer_spnkr_db_if_available", False)):
            spnkr = _pick_latest_spnkr_db_if_any()
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
                    "--download-images",
                    "--clean-output",
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
        return "color: #8E6CFF; font-weight: 700;"
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


def _format_datetime_fr_hm(dt_value) -> str:
    if dt_value is None:
        return "-"
    d = _to_paris_naive(dt_value)
    if d is None:
        return "-"
    return f"{format_date_fr(d)} {d:%H:%M}"


_DATE_FR_RE = re.compile(r"^\s*(\d{1,2})\s*[\-/]\s*(\d{1,2})\s*[\-/]\s*(\d{4})\s*$")


def _parse_date_fr_input(value: str | None, *, default_value: date) -> date:
    """Parse une date au format dd/mm/yyyy (ou dd-mm-yyyy).

    Retourne default_value si la saisie est invalide.
    """
    s = (value or "").strip()
    if not s:
        return default_value
    m = _DATE_FR_RE.match(s)
    if not m:
        return default_value
    try:
        dd = int(m.group(1))
        mm = int(m.group(2))
        yyyy = int(m.group(3))
        return date(yyyy, mm, dd)
    except Exception:
        return default_value


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


def _plot_multi_metric_bars_by_match(
    series: list[tuple[str, pd.DataFrame]],
    *,
    metric_col: str,
    title: str,
    y_axis_title: str,
    hover_label: str,
    colors: dict[str, str] | list[str] | None,
    smooth_window: int = 10,
    show_smooth_lines: bool = True,
) -> go.Figure | None:
    if not series:
        return None

    prepared: list[tuple[str, pd.DataFrame]] = []
    all_times: list[pd.Timestamp] = []
    for name, df_ in series:
        if df_ is None or df_.empty:
            continue
        if metric_col not in df_.columns or "start_time" not in df_.columns:
            continue
        d = df_[["start_time", metric_col]].copy()
        d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce")
        d = d.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
        if d.empty:
            continue
        prepared.append((str(name), d))
        all_times.extend(d["start_time"].tolist())

    if not prepared or not all_times:
        return None

    # Axe X commun (timeline de tous les joueurs)
    uniq = pd.Series(all_times).dropna().drop_duplicates().sort_values()
    times = uniq.tolist()
    idx_map = {t: i for i, t in enumerate(times)}
    labels = [pd.to_datetime(t).strftime("%d/%m %H:%M") for t in times]
    step = max(1, len(labels) // 10) if labels else 1

    fig = go.Figure()
    w = int(smooth_window) if smooth_window else 0
    for i, (name, d) in enumerate(prepared):
        if isinstance(colors, dict):
            color = colors.get(name) or "#35D0FF"
        elif isinstance(colors, list) and colors:
            color = colors[i % len(colors)]
        else:
            color = "#35D0FF"
        y = pd.to_numeric(d[metric_col], errors="coerce")
        mask = y.notna()
        d2 = d.loc[mask].copy()
        if d2.empty:
            continue
        y2 = pd.to_numeric(d2[metric_col], errors="coerce")
        x = [idx_map.get(t) for t in d2["start_time"].tolist()]
        x = [xi for xi in x if xi is not None]
        if not x:
            continue

        fig.add_trace(
            go.Bar(
                x=x,
                y=y2,
                name=name,
                marker_color=color,
                opacity=0.70,
                hovertemplate=f"{name}<br>{hover_label}=%{{y}}<extra></extra>",
                legendgroup=name,
            )
        )

        if bool(show_smooth_lines):
            smooth = y2.rolling(window=max(1, w), min_periods=1).mean() if w and w > 1 else y2
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=smooth,
                    mode="lines",
                    name=f"{name} — moyenne lissée",
                    line=dict(width=3, color=color),
                    opacity=0.95,
                    hovertemplate=f"{name}<br>moyenne=%{{y:.2f}}<extra></extra>",
                    legendgroup=name,
                )
            )

    if not fig.data:
        return None

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
        barmode="group",
    )
    fig.update_yaxes(title_text=y_axis_title, rangemode="tozero")
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=list(range(len(labels)))[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, height=320)


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
        parts.append(f"{days}D")
    if hours or days:
        parts.append(f"{hours}H")
    parts.append(f"{minutes}M")
    return " ".join(parts)


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
def cached_load_highlight_events_for_match(db_path: str, match_id: str, *, db_key: str | None = None):
    _ = db_key
    return load_highlight_events_for_match(db_path, match_id)


@st.cache_data(show_spinner=False)
def cached_load_match_player_gamertags(db_path: str, match_id: str, *, db_key: str | None = None):
    _ = db_key
    return load_match_player_gamertags(db_path, match_id)


@st.cache_data(show_spinner=False)
def cached_load_top_medals(
    db_path: str,
    xuid: str,
    match_ids: tuple[str, ...],
    top_n: int | None,
    db_key: tuple[int, int] | None,
):
    return load_top_medals(db_path, xuid, list(match_ids), top_n=(int(top_n) if top_n is not None else None))


def _top_medals(
    db_path: str,
    xuid: str,
    match_ids: list[str],
    *,
    top_n: int | None,
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
    dfr["start_time"] = (
        pd.to_datetime(dfr["start_time"], utc=True)
        .dt.tz_convert(PARIS_TZ_NAME)
        .dt.tz_localize(None)
    )
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


def _render_settings_page(settings: AppSettings) -> AppSettings:
    """Rend l'onglet Paramètres et retourne les settings (potentiellement modifiés)."""

    st.subheader("Paramètres")

    with st.expander("Source", expanded=True):
        default_db = get_default_db_path()
        render_source_section(
            default_db,
            get_local_dbs=cached_list_local_dbs,
            on_clear_caches=_clear_app_caches,
        )
        st.caption("Valeurs par défaut: .streamlit/secrets.toml (ignoré par Git).")
        st.code(
            "[player]\n"
            "gamertag=\"...\"\n"
            "xuid=\"2533...\"\n"
            "waypoint_player=\"...\"\n",
            language="toml",
        )

    with st.expander("Paramètres avancés", expanded=False):
        st.toggle(
            "Inclure Firefight (PvE)",
            key="include_firefight",
            value=bool(st.session_state.get("include_firefight", False)),
            help="Impacte les filtres et tous les onglets basés sur les matchs.",
        )
        st.toggle(
            "Limiter aux playlists (Quick Play / Ranked Slayer / Ranked Arena)",
            key="restrict_playlists",
            value=bool(st.session_state.get("restrict_playlists", True)),
            help="Réduit les playlists/modes/cartes aux valeurs attendues (utile si la DB est hétérogène).",
        )

    with st.expander("SPNKr (API → DB)", expanded=False):
        st.caption("Optionnel: recharge les derniers matchs via l'API et met à jour la DB SPNKr.")
        prefer_spnkr = st.toggle(
            "Utiliser SPNKr par défaut (si disponible)",
            value=bool(getattr(settings, "prefer_spnkr_db_if_available", True)),
        )
        spnkr_on_start = st.toggle(
            "Rafraîchir la DB au démarrage (SPNKr)",
            value=bool(getattr(settings, "spnkr_refresh_on_start", True)),
        )
        spnkr_on_refresh = st.toggle(
            "Le bouton Actualiser rafraîchit aussi la DB (SPNKr)",
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
            "Inclure highlight events (plus long)",
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

    with st.expander("Expérience", expanded=False):
        refresh_clears_caches = st.toggle(
            "Le bouton Actualiser vide aussi les caches",
            value=bool(getattr(settings, "refresh_clears_caches", False)),
            help="Utile si la DB change en dehors de l'app (NAS / import externe).",
        )

    with st.expander("Outils", expanded=False):
        if os.name == "nt":
            render_openspartan_tools()
        else:
            st.caption("Outils Windows masqués (environnement non-Windows/NAS).")

    with st.expander("Fichiers (avancé)", expanded=False):
        st.caption("Optionnel: sélectionne les fichiers JSON utilisés par l'app (sinon valeurs par défaut).")
        aliases_path = file_input(
            "Fichier d'alias XUID (json)",
            value=str(getattr(settings, "aliases_path", "") or ""),
            key="settings_aliases_path",
            exts=(".json",),
            help="Override de OPENSPARTAN_ALIASES_PATH. Laisse vide pour utiliser xuid_aliases.json à la racine.",
            placeholder="Ex: C:\\...\\xuid_aliases.json",
        )
        profiles_path = file_input(
            "Fichier de profils DB (json)",
            value=str(getattr(settings, "profiles_path", "") or ""),
            key="settings_profiles_path",
            exts=(".json",),
            help="Override de OPENSPARTAN_PROFILES_PATH. Laisse vide pour utiliser db_profiles.json à la racine.",
            placeholder="Ex: C:\\...\\db_profiles.json",
        )

    with st.expander("NAS / Docker", expanded=False):
        st.caption(
            "Astuce: sur NAS/Docker, monte la DB en volume et définis OPENSPARTAN_DB (et éventuellement OPENSPARTAN_DB_READONLY=1)."
        )
        st.code(
            "OPENSPARTAN_DB=...\nOPENSPARTAN_DB_READONLY=1\nOPENSPARTAN_PROFILES_PATH=...\nOPENSPARTAN_ALIASES_PATH=...\nOPENSPARTAN_SETTINGS_PATH=...",
            language="text",
        )

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


def _request_open_match(match_id: str) -> None:
    mid = str(match_id or "").strip()
    if not mid:
        return
    _set_query_params(page="Match", match_id=mid)
    st.session_state["_pending_page"] = "Match"
    st.session_state["_pending_match_id"] = mid
    st.rerun()


def _safe_dt(v) -> datetime | None:
    return _to_paris_naive(v)


def _match_time_window(row: pd.Series, *, tolerance_minutes: int) -> tuple[datetime | None, datetime | None]:
    start = _safe_dt(row.get("start_time"))
    if start is None:
        return None, None

    dur_s = row.get("time_played_seconds")
    try:
        dur = float(dur_s) if dur_s == dur_s else None
    except Exception:
        dur = None
    if dur is None or dur <= 0:
        end = start + timedelta(minutes=30)
    else:
        end = start + timedelta(seconds=float(dur))

    tol = max(0, int(tolerance_minutes))
    return start - timedelta(minutes=tol), end + timedelta(minutes=tol)


@st.cache_data(show_spinner=False, ttl=120)
def _index_media_dir(dir_path: str, exts: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    p = str(dir_path or "").strip()
    if not p or not os.path.isdir(p):
        return pd.DataFrame(columns=["path", "mtime", "ext"])

    wanted = {e.lower().lstrip(".") for e in (exts or tuple()) if isinstance(e, str) and e.strip()}
    if not wanted:
        return pd.DataFrame(columns=["path", "mtime", "ext"])

    max_files = 12000
    try:
        for root, _dirs, files in os.walk(p):
            for fn in files:
                if len(rows) >= max_files:
                    break
                ext = os.path.splitext(fn)[1].lower().lstrip(".")
                if ext not in wanted:
                    continue
                full = os.path.join(root, fn)
                try:
                    st_ = os.stat(full)
                    mtime = float(st_.st_mtime)
                except Exception:
                    continue
                rows.append({"path": full, "mtime": mtime, "ext": ext})
            if len(rows) >= max_files:
                break
    except Exception:
        return pd.DataFrame(columns=["path", "mtime", "ext"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mtime", ascending=False).reset_index(drop=True)


def _render_media_section(*, row: pd.Series, settings: AppSettings) -> None:
    if not bool(getattr(settings, "media_enabled", True)):
        return

    tol = int(getattr(settings, "media_tolerance_minutes", 0) or 0)
    t0, t1 = _match_time_window(row, tolerance_minutes=tol)
    if t0 is None or t1 is None:
        return

    screens_dir = str(getattr(settings, "media_screens_dir", "") or "").strip()
    videos_dir = str(getattr(settings, "media_videos_dir", "") or "").strip()

    if not screens_dir and not videos_dir:
        return

    st.subheader("Médias")
    st.caption(f"Fenêtre de recherche: {_format_datetime_fr_hm(t0)} → {_format_datetime_fr_hm(t1)}")

    t0_epoch = _paris_epoch_seconds(t0)
    t1_epoch = _paris_epoch_seconds(t1)
    if t0_epoch is None or t1_epoch is None:
        return

    found_any = False

    if screens_dir and os.path.isdir(screens_dir):
        img_df = _index_media_dir(screens_dir, ("png", "jpg", "jpeg", "webp"))
        if not img_df.empty:
            mask = (img_df["mtime"] >= t0_epoch) & (img_df["mtime"] <= t1_epoch)
            hits = img_df.loc[mask].head(24)
            if not hits.empty:
                found_any = True
                st.caption("Captures")
                for p in hits["path"].tolist():
                    try:
                        st.image(p, caption=str(p))
                    except Exception:
                        st.write(str(p))

    if videos_dir and os.path.isdir(videos_dir):
        vid_df = _index_media_dir(videos_dir, ("mp4", "webm", "mkv", "mov"))
        if not vid_df.empty:
            mask = (vid_df["mtime"] >= t0_epoch) & (vid_df["mtime"] <= t1_epoch)
            hits = vid_df.loc[mask].head(10)
            if not hits.empty:
                found_any = True
                st.caption("Vidéos")
                # UX/robustesse: embed d'une seule vidéo à la fois.
                paths = [str(p) for p in hits["path"].tolist() if p]
                if paths:
                    labels = [os.path.basename(p) for p in paths]
                    picked = st.selectbox(
                        "Vidéo",
                        options=list(range(len(paths))),
                        format_func=lambda i: labels[i],
                        index=0,
                        key=f"media_video_pick_{row.get('match_id','')}",
                        label_visibility="collapsed",
                    )
                    p = paths[int(picked)]
                    try:
                        st.video(p)
                        st.caption(str(p))
                    except Exception:
                        st.write(str(p))

    if not found_any:
        st.info("Aucun média trouvé pour ce match.")


def _render_match_view(
    *,
    row: pd.Series,
    match_id: str,
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: tuple[int, int] | None,
    settings: AppSettings,
) -> None:
    match_id = str(match_id or "").strip()
    if not match_id:
        st.info("MatchId manquant.")
        return

    last_time = row.get("start_time")
    last_map = row.get("map_name")
    last_playlist = row.get("playlist_name")
    last_pair = row.get("pair_name")
    last_mode = row.get("game_variant_name")
    last_outcome = row.get("outcome")

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
    elif outcome_code == OUTCOME_CODES.NO_FINISH:
        outcome_color = colors["violet"]
    else:
        outcome_color = colors["slate"]

    last_my_score = row.get("my_team_score")
    last_enemy_score = row.get("enemy_team_score")
    score_label = _format_score_label(last_my_score, last_enemy_score)
    score_color = _score_css_color(last_my_score, last_enemy_score)

    wp = str(waypoint_player or "").strip()
    match_url = None
    if wp and match_id and match_id.strip() and match_id.strip() != "-":
        match_url = f"https://www.halowaypoint.com/halo-infinite/players/{wp}/matches/{match_id.strip()}"

    top_cols = st.columns([3, 4])
    with top_cols[0]:
        st.metric("Date", format_date_fr(last_time))
    with top_cols[1]:
        outcome_border = f"{outcome_color}55" if str(outcome_color).startswith("#") else outcome_color
        outcome_bg = f"{outcome_color}14" if str(outcome_color).startswith("#") else "rgba(255,255,255,0.03)"
        st.markdown(
            "<div style='border:1px solid "
            + str(outcome_border)
            + "; border-radius:14px; padding:12px 14px; background: "
            + str(outcome_bg)
            + "; box-shadow: 0 6px 18px rgba(0,0,0,0.18)'>"
            "<div style='display:flex; align-items:center; gap:12px; justify-content:space-between'>"
            f"<div style='font-weight:900; font-size:1.10rem; color:{outcome_color}; letter-spacing:0.2px'>{outcome_label}</div>"
            f"<div style='font-weight:900; font-size:1.10rem; color:{score_color}; letter-spacing:0.2px'>{score_label}</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    last_mode_ui = row.get("mode_ui") or _normalize_mode_label(str(last_pair) if last_pair else None)
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
        pm = cached_load_player_match_result(db_path, match_id, xuid.strip(), db_key=db_key)
        medals_last = cached_load_match_medals_for_player(db_path, match_id, xuid.strip(), db_key=db_key)

    if not pm:
        st.info("Stats détaillées indisponibles pour ce match (PlayerMatchStats manquant ou format inattendu).")
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
            avg_life_last = row.get("average_life_seconds")
            st.metric("Durée de vie moyenne", format_mmss(avg_life_last))

    st.subheader("Médailles")
    if not medals_last:
        st.info("Médailles indisponibles pour ce match (ou aucune médaille).")
    else:
        md_df = pd.DataFrame(medals_last)
        md_df["label"] = md_df["name_id"].apply(lambda x: medal_label(int(x)))
        md_df = md_df.sort_values(["count", "label"], ascending=[False, True])
        render_medals_grid(md_df[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)

    st.subheader("Némésis / Souffre-douleur")
    if not (match_id and match_id.strip() and has_table(db_path, "HighlightEvents")):
        st.caption(
            "Indisponible: la DB ne contient pas les highlight events. "
            "Si tu utilises une DB SPNKr, relance l'import avec `--with-highlight-events`."
        )
    else:
        with st.spinner("Chargement des highlight events (film)…"):
            he = cached_load_highlight_events_for_match(db_path, match_id.strip(), db_key=db_key)

        # Mapping plus fiable que les gamertags des highlight events
        match_gt_map = cached_load_match_player_gamertags(db_path, match_id.strip(), db_key=db_key)

        pairs = compute_killer_victim_pairs(he, tolerance_ms=5)
        if not pairs:
            st.info("Aucune paire kill/death trouvée (ou match sans timeline exploitable).")
        else:
            kv_long = killer_victim_counts_long(pairs)

            me_xuid = str(xuid).strip()
            killed_me = kv_long[kv_long["victim_xuid"].astype(str) == me_xuid]
            i_killed = kv_long[kv_long["killer_xuid"].astype(str) == me_xuid]

            def _display_name_from_kv(xuid_value, gamertag_value) -> str:
                """Retourne un nom lisible pour l'UI.

                Les highlight events SPNKr peuvent ne pas contenir de gamertag.
                Dans ce cas, on retombe sur un alias local basé sur le XUID.
                """
                gt = str(gamertag_value or "").strip()
                xu_raw = str(xuid_value or "").strip()
                xu = parse_xuid_input(xu_raw) or xu_raw

                # Si on connait le gamertag depuis MatchStats, on le préfère.
                xu_key = str(xu).strip() if xu is not None else ""
                if xu_key and isinstance(match_gt_map, dict):
                    mapped = match_gt_map.get(xu_key)
                    if isinstance(mapped, str) and mapped.strip():
                        return mapped.strip()

                # Si le gamertag ressemble à un XUID / placeholder, on l'ignore.
                if (not gt) or gt == "?" or gt.isdigit() or gt.lower().startswith("xuid("):
                    if xu:
                        return display_name_from_xuid(str(xu).strip())
                    return "-"
                return gt

            nemesis_name = "-"
            nemesis_kills = None
            if not killed_me.empty:
                top = (
                    killed_me[["killer_xuid", "killer_gamertag", "count"]]
                    .rename(columns={"count": "Kills"})
                    .sort_values(["Kills"], ascending=[False])
                    .iloc[0]
                )
                nemesis_name = _display_name_from_kv(top.get("killer_xuid"), top.get("killer_gamertag"))
                nemesis_kills = int(top.get("Kills")) if top.get("Kills") is not None else None

            bully_name = "-"
            bully_kills = None
            if not i_killed.empty:
                top = (
                    i_killed[["victim_xuid", "victim_gamertag", "count"]]
                    .rename(columns={"count": "Kills"})
                    .sort_values(["Kills"], ascending=[False])
                    .iloc[0]
                )
                bully_name = _display_name_from_kv(top.get("victim_xuid"), top.get("victim_gamertag"))
                bully_kills = int(top.get("Kills")) if top.get("Kills") is not None else None

            def _clean_name(v: str) -> str:
                s = str(v or "")
                s = s.replace("\ufffd", "")
                s = re.sub(r"[\x00-\x1f\x7f]", "", s)
                s = re.sub(r"\s+", " ", s).strip()
                return s or "-"

            nemesis_name = _clean_name(nemesis_name)
            bully_name = _clean_name(bully_name)

            c = st.columns(2)
            c[0].metric("Némésis (m'a le plus tué)", nemesis_name, f"{nemesis_kills} kills" if nemesis_kills is not None else None)
            c[1].metric(
                "Souffre-douleur (j'ai le plus tué)",
                bully_name,
                f"{bully_kills} kills" if bully_kills is not None else None,
            )

    _render_media_section(row=row, settings=settings)


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
    df["start_time"] = (
        pd.to_datetime(df["start_time"], utc=True)
        .dt.tz_convert(PARIS_TZ_NAME)
        .dt.tz_localize(None)
    )
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
        render_perf_panel(location="sidebar")
        st.markdown("<div class='os-sidebar-brand'>OpenSpartan Graphs</div>", unsafe_allow_html=True)
        st.markdown("<div class='os-sidebar-divider'></div>", unsafe_allow_html=True)

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
                if _is_spnkr_db_path(db_path):
                    fallback = _pick_latest_spnkr_db_if_any()
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
        _render_settings_page(settings)
        return

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

        # Paramètres avancés déplacés dans l'onglet Paramètres.

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
    # Bandeau résumé (en haut du site) — regroupé
    # ------------------------------------------------------------------
    avg_match_seconds = _avg_match_duration_seconds(dff)
    total_play_seconds = _compute_total_play_seconds(dff)
    avg_match_txt = _format_duration_hms(avg_match_seconds)
    total_play_txt = _format_duration_dhm(total_play_seconds)

    # Stats par minute / totaux
    stats = compute_aggregated_stats(dff)

    # Moyennes par partie
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    st.subheader("Parties")
    _render_top_summary(len(dff), rates)
    _render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
        ]
    )

    st.subheader("Carrière")
    _render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Frags par partie", f"{kpg:.2f}" if (kpg is not None and pd.notna(kpg)) else "-"),
            ("Morts par partie", f"{dpg:.2f}" if (dpg is not None and pd.notna(dpg)) else "-"),
            ("Assistances par partie", f"{apg:.2f}" if (apg is not None and pd.notna(apg)) else "-"),
        ],
        dense=False,
    )
    _render_kpi_cards(
        [
            ("Frags / min", f"{stats.kills_per_minute:.2f}" if stats.kills_per_minute else "-"),
            ("Morts / min", f"{stats.deaths_per_minute:.2f}" if stats.deaths_per_minute else "-"),
            ("Assistances / min", f"{stats.assists_per_minute:.2f}" if stats.assists_per_minute else "-"),
            ("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-"),
            ("Durée totale", total_play_txt),
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

    # --------------------------------------------------------------------------
    # Page: Dernier match
    # --------------------------------------------------------------------------
    if page == "Dernier match":
        st.caption("Dernière partie selon la sélection/filtres actuels.")
        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            last_row = dff.sort_values("start_time").iloc[-1]
            last_match_id = str(last_row.get("match_id", "")).strip()
            _render_match_view(
                row=last_row,
                match_id=last_match_id,
                db_path=db_path,
                xuid=xuid,
                waypoint_player=waypoint_player,
                db_key=db_key,
                settings=settings,
            )

    # --------------------------------------------------------------------------
    # Page: Match (recherche)
    # --------------------------------------------------------------------------
    elif page == "Match":
        st.caption("Afficher un match précis via un MatchId, une date/heure, ou une sélection.")

        # Entrée MatchId
        match_id_input = st.text_input("MatchId", value=str(st.session_state.get("match_id_input", "") or ""), key="match_id_input")

        # Sélection rapide (sur les filtres actuels, triés du plus récent au plus ancien)
        quick_df = dff.sort_values("start_time", ascending=False).head(200).copy()
        quick_df["start_time_fr"] = quick_df["start_time"].apply(_format_datetime_fr_hm)
        if "mode_ui" not in quick_df.columns:
            quick_df["mode_ui"] = quick_df["pair_name"].apply(_normalize_mode_label)
        quick_df["label"] = (
            quick_df["start_time_fr"].astype(str)
            + " — "
            + quick_df["map_name"].astype(str)
            + " — "
            + quick_df["mode_ui"].astype(str)
        )
        opts = {r["label"]: str(r["match_id"]) for _, r in quick_df.iterrows()}
        picked_label = st.selectbox("Sélection rapide (filtres actuels)", options=["(aucun)"] + list(opts.keys()), index=0)
        if st.button("Utiliser ce match", width="stretch") and picked_label != "(aucun)":
            st.session_state["match_id_input"] = opts[picked_label]
            st.rerun()

        # Recherche par date/heure
        with st.expander("Recherche par date/heure", expanded=False):
            dd = st.date_input("Date", value=date.today(), format="DD/MM/YYYY")
            tt = st.time_input("Heure", value=time(20, 0))
            tol_min = st.slider("Tolérance (minutes)", 0, 30, 10, 1)
            if st.button("Rechercher", width="stretch"):
                target = datetime.combine(dd, tt)
                all_df = df.copy()
                all_df["_dt"] = pd.to_datetime(all_df["start_time"], errors="coerce")
                all_df = all_df.dropna(subset=["_dt"]).copy()
                if all_df.empty:
                    st.warning("Aucune date exploitable dans la DB.")
                else:
                    all_df["_diff"] = (all_df["_dt"] - target).abs()
                    best = all_df.sort_values("_diff").iloc[0]
                    diff_min = float(best["_diff"].total_seconds() / 60.0)
                    if diff_min <= float(tol_min):
                        st.session_state["match_id_input"] = str(best.get("match_id") or "").strip()
                        st.rerun()
                    else:
                        st.warning(f"Aucun match trouvé dans ±{tol_min} min (le plus proche est à {diff_min:.1f} min).")

        mid = str(match_id_input or "").strip()
        if not mid:
            st.info("Renseigne un MatchId ou utilise la sélection/recherche ci-dessus.")
        else:
            rows = df.loc[df["match_id"].astype(str) == mid]
            if rows.empty:
                st.warning("MatchId introuvable dans la DB actuelle.")
            else:
                match_row = rows.sort_values("start_time").iloc[-1]
                _render_match_view(
                    row=match_row,
                    match_id=mid,
                    db_path=db_path,
                    xuid=xuid,
                    waypoint_player=waypoint_player,
                    db_key=db_key,
                    settings=settings,
                )

    # --------------------------------------------------------------------------
    # Page: Citations (ex Médailles)
    # --------------------------------------------------------------------------
    elif page == "Citations":
        # 1) Commendations Halo 5 (référentiel offline)
        render_h5g_commendations_section()
        st.divider()

        # 2) Médailles (Halo Infinite) sur la sélection/filtres actuels
        st.caption("Médailles sur la sélection/filtres actuels (non limitées).")
        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            show_all = st.toggle("Afficher toutes les médailles (peut être lent)", value=False)
            top_n = None if show_all else int(st.slider("Nombre de médailles", 25, 500, 100, 25))

            match_ids = [str(x) for x in dff["match_id"].dropna().astype(str).tolist()]
            with st.spinner("Agrégation des médailles…"):
                top = _top_medals(db_path, xuid.strip(), match_ids, top_n=top_n, db_key=db_key)

            if not top:
                st.info("Aucune médaille trouvée (ou payload médailles absent).")
            else:
                md = pd.DataFrame(top, columns=["name_id", "count"])
                md["label"] = md["name_id"].apply(lambda x: medal_label(int(x)))
                md_desc = md.sort_values("count", ascending=False)
                render_medals_grid(md_desc[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)

    # --------------------------------------------------------------------------
    # Page: Séries temporelles
    # --------------------------------------------------------------------------
    elif page == "Séries temporelles":
        with st.spinner("Génération des graphes…"):
            fig = plot_timeseries(dff, title=f"{me_name}")
            st.plotly_chart(fig, width="stretch")

            st.subheader("FDA")
            valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
            if valid.empty:
                st.info("FDA indisponible sur ce filtre.")
            else:
                m = st.columns(1)
                m[0].metric("KDA moyen", f"{valid['kda'].mean():.2f}", label_visibility="collapsed")
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
    # Page: Victoires/Défaites
    # --------------------------------------------------------------------------
    elif page == "Victoires/Défaites":
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

                out_tbl = out_tbl.reset_index().rename(columns={"bucket": bucket_label.capitalize()})

                def _style_pct(v) -> str:
                    try:
                        x = float(v)
                    except Exception:
                        return ""
                    return "color: #E0E0E0; font-weight: 700;"

                out_styled = _styler_map(out_tbl.style, _style_pct, subset=["Win rate"])
                st.dataframe(
                    out_styled,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Win rate": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
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

                # Style: vert/rouge selon l'avantage (win% vs loss%), et violet si égalité.
                # Ratio: >1 vert, <1 rouge, ==1 violet (8E6CFF).
                def _to_float(v: object) -> Optional[float]:
                    try:
                        if v is None:
                            return None
                        x = float(v)
                        return x if x == x else None
                    except Exception:
                        return None

                def _style_map_table_row(row: pd.Series) -> pd.Series:
                    green = str(getattr(HALO_COLORS, "green", "#2ECC71"))
                    red = str(getattr(HALO_COLORS, "red", "#E74C3C"))
                    violet = "#8E6CFF"

                    w = _to_float(row.get("Taux victoire (%)"))
                    l = _to_float(row.get("Taux défaite (%)"))
                    r = _to_float(row.get("Ratio global"))

                    styles: dict[str, str] = {str(c): "" for c in row.index}

                    if w is not None and l is not None:
                        if w > l:
                            styles["Taux victoire (%)"] = f"color: {green}; font-weight: 800;"
                            styles["Taux défaite (%)"] = f"color: {red}; font-weight: 800;"
                        elif w < l:
                            styles["Taux victoire (%)"] = f"color: {red}; font-weight: 800;"
                            styles["Taux défaite (%)"] = f"color: {green}; font-weight: 800;"
                        else:
                            styles["Taux victoire (%)"] = f"color: {violet}; font-weight: 800;"
                            styles["Taux défaite (%)"] = f"color: {violet}; font-weight: 800;"

                    if r is not None:
                        if r > 1.0:
                            styles["Ratio global"] = f"color: {green}; font-weight: 800;"
                        elif r < 1.0:
                            styles["Ratio global"] = f"color: {red}; font-weight: 800;"
                        else:
                            styles["Ratio global"] = f"color: {violet}; font-weight: 800;"

                    return pd.Series(styles)

                tbl_styled = tbl_disp.style.apply(_style_map_table_row, axis=1)
                st.dataframe(
                    tbl_styled,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Parties": st.column_config.NumberColumn("Parties", format="%d"),
                        "Précision moy. (%)": st.column_config.NumberColumn("Précision moy. (%)", format="%.2f"),
                        "Taux victoire (%)": st.column_config.NumberColumn("Taux victoire (%)", format="%.1f"),
                        "Taux défaite (%)": st.column_config.NumberColumn("Taux défaite (%)", format="%.1f"),
                        "Ratio global": st.column_config.NumberColumn("Ratio global", format="%.2f"),
                    },
                )

            # (Le graphique "Net victoires/défaites" a été retiré :
            #  l'onglet se concentre sur le graphe Victoires au-dessus / Défaites en dessous.)

    # --------------------------------------------------------------------------
    # Page: Mes coéquipiers (fusion)
    # --------------------------------------------------------------------------
    elif page == "Mes coéquipiers":
        st.caption("Vue dédiée aux matchs joués avec tes coéquipiers (XUID rencontrés / alias).")

        apply_current_filters_teammates = st.toggle(
            "Appliquer les filtres actuels (période/sessions + map/playlist)",
            value=True,
            key="apply_current_filters_teammates",
        )
        same_team_only_teammates = st.checkbox("Même équipe", value=True, key="teammates_same_team_only")

        show_smooth_teammates = st.toggle(
            "Afficher les courbes lissées",
            value=bool(st.session_state.get("teammates_show_smooth", True)),
            key="teammates_show_smooth",
            help="Active/désactive les courbes de moyenne lissée sur les graphes de cette section.",
        )

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
                            st.plotly_chart(
                                plot_timeseries(sub, title=f"{me_name} — matchs avec {name}"),
                                width="stretch",
                                key=f"friend_ts_me_{friend_xuid}",
                            )
                        with c2:
                            if friend_sub.empty:
                                st.warning("Impossible de charger les stats du coéquipier sur les matchs partagés.")
                            else:
                                st.plotly_chart(
                                    plot_timeseries(friend_sub, title=f"{name} — matchs avec {me_name}"),
                                    width="stretch",
                                    key=f"friend_ts_fr_{friend_xuid}",
                                )

                        c3, c4 = st.columns(2)
                        with c3:
                            st.plotly_chart(
                                plot_per_minute_timeseries(sub, title=f"{me_name} — stats/min (avec {name})"),
                                width="stretch",
                                key=f"friend_pm_me_{friend_xuid}",
                            )
                        with c4:
                            if not friend_sub.empty:
                                st.plotly_chart(
                                    plot_per_minute_timeseries(
                                        friend_sub,
                                        title=f"{name} — stats/min (avec {me_name})",
                                    ),
                                    width="stretch",
                                    key=f"friend_pm_fr_{friend_xuid}",
                                )

                        c5, c6 = st.columns(2)
                        with c5:
                            if not sub.dropna(subset=["average_life_seconds"]).empty:
                                st.plotly_chart(
                                    plot_average_life(sub, title=f"{me_name} — Durée de vie (avec {name})"),
                                    width="stretch",
                                    key=f"friend_life_me_{friend_xuid}",
                                )
                        with c6:
                            if not friend_sub.empty and not friend_sub.dropna(subset=["average_life_seconds"]).empty:
                                st.plotly_chart(
                                    plot_average_life(friend_sub, title=f"{name} — Durée de vie (avec {me_name})"),
                                    width="stretch",
                                    key=f"friend_life_fr_{friend_xuid}",
                                )
                        # Folie meurtrière / Tirs à la tête (tout en bas, avant les médailles) — 1 graphe par ligne.
                        series = [(me_name, sub)]
                        if not friend_sub.empty:
                            series.append((name, friend_sub))
                        colors_by_name = _assign_player_colors([n for n, _ in series])

                        fig_spree = _plot_multi_metric_bars_by_match(
                            series,
                            metric_col="max_killing_spree",
                            title="Folie meurtrière (max)",
                            y_axis_title="Folie meurtrière (max)",
                            hover_label="folie meurtrière",
                            colors=colors_by_name,
                            smooth_window=10,
                            show_smooth_lines=show_smooth_teammates,
                        )
                        if fig_spree is None:
                            st.info("Aucune donnée de folie meurtrière (max) sur ces matchs.")
                        else:
                            st.plotly_chart(fig_spree, width="stretch", key=f"friend_spree_multi_{friend_xuid}")

                        fig_hs = _plot_multi_metric_bars_by_match(
                            series,
                            metric_col="headshot_kills",
                            title="Tirs à la tête",
                            y_axis_title="Tirs à la tête",
                            hover_label="tirs à la tête",
                            colors=colors_by_name,
                            smooth_window=10,
                            show_smooth_lines=show_smooth_teammates,
                        )
                        if fig_hs is None:
                            st.info("Aucune donnée de tirs à la tête sur ces matchs.")
                        else:
                            st.plotly_chart(fig_hs, width="stretch", key=f"friend_hs_multi_{friend_xuid}")

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

                use_xuids = picked_xuids

                series: list[tuple[str, pd.DataFrame]] = [(me_name, sub_all)]
                with st.spinner("Chargement des stats des coéquipiers…"):
                    for fx in use_xuids:
                        ids = per_friend_ids.get(str(fx), set())
                        if not ids:
                            continue
                        try:
                            fr_df = load_df(db_path, str(fx), db_key=db_key)
                        except Exception:
                            continue
                        fr_sub = fr_df.loc[fr_df["match_id"].astype(str).isin(ids)].copy()
                        if fr_sub.empty:
                            continue
                        series.append((display_name_from_xuid(str(fx)), fr_sub))
                colors_by_name = _assign_player_colors([n for n, _ in series])

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

                    # MMR équipe/adverse : même source que l'onglet "Dernier match" (PlayerMatchStats).
                    def _mmr_tuple(match_id: str):
                        pm = cached_load_player_match_result(db_path, str(match_id), xuid.strip(), db_key=db_key)
                        if not isinstance(pm, dict):
                            return (None, None)
                        return (pm.get("team_mmr"), pm.get("enemy_mmr"))

                    mmr_pairs = friends_table["match_id"].astype(str).apply(_mmr_tuple)
                    friends_table["team_mmr"] = mmr_pairs.apply(lambda t: t[0])
                    friends_table["enemy_mmr"] = mmr_pairs.apply(lambda t: t[1])
                    friends_table["delta_mmr"] = friends_table.apply(
                        lambda r: (float(r.get("team_mmr")) - float(r.get("enemy_mmr")))
                        if (r.get("team_mmr") is not None and r.get("enemy_mmr") is not None)
                        else None,
                        axis=1,
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

                    friends_table["app_match_url"] = friends_table["match_id"].astype(str).apply(
                        lambda mid: _app_url("Match", match_id=str(mid))
                    )

                    friends_show = [
                        "app_match_url",
                        "match_url",
                        "start_time_fr",
                        "map_name",
                        "playlist_fr",
                        "mode",
                        "outcome_label",
                        "score",
                        "team_mmr",
                        "enemy_mmr",
                        "delta_mmr",
                    ]
                    friends_view = (
                        friends_table.sort_values("start_time", ascending=False)[friends_show]
                        .reset_index(drop=True)
                    )
                    friends_styled = (
                        friends_view.style
                        .map(_style_outcome_text, subset=["outcome_label"])
                        .map(_style_score_label, subset=["score"])
                        .map(_style_signed_number, subset=["delta_mmr"])
                    )
                    st.dataframe(
                        friends_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "app_match_url": st.column_config.LinkColumn("Match", display_text="Ouvrir"),
                            "match_url": st.column_config.LinkColumn("HaloWaypoint", display_text="Ouvrir"),
                            "start_time_fr": st.column_config.TextColumn("Date"),
                            "map_name": st.column_config.TextColumn("Carte"),
                            "playlist_fr": st.column_config.TextColumn("Playlist"),
                            "mode": st.column_config.TextColumn("Mode"),
                            "outcome_label": st.column_config.TextColumn("Résultat"),
                            "score": st.column_config.TextColumn("Score"),
                            "team_mmr": st.column_config.NumberColumn("MMR d'équipe", format="%.1f"),
                            "enemy_mmr": st.column_config.NumberColumn("MMR adverse", format="%.1f"),
                            "delta_mmr": st.column_config.NumberColumn("Écart MMR", format="%+.1f"),
                        },
                    )

                # Graphes : on les rendra tout en bas (idéalement juste avant les médailles).
                rendered_bottom_charts = False

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
                            key=f"trio_kills_{f1_xuid}_{f2_xuid}",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="deaths", names=names, title="Morts", y_title="Morts"),
                            width="stretch",
                            key=f"trio_deaths_{f1_xuid}_{f2_xuid}",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="assists", names=names, title="Assistances", y_title="Assists"),
                            width="stretch",
                            key=f"trio_assists_{f1_xuid}_{f2_xuid}",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="ratio", names=names, title="FDA", y_title="FDA", y_format=".3f"),
                            width="stretch",
                            key=f"trio_ratio_{f1_xuid}_{f2_xuid}",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="accuracy", names=names, title="Précision", y_title="%", y_suffix="%", y_format=".2f"),
                            width="stretch",
                            key=f"trio_accuracy_{f1_xuid}_{f2_xuid}",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="average_life_seconds", names=names, title="Durée de vie moyenne", y_title="Secondes", y_format=".1f"),
                            width="stretch",
                            key=f"trio_life_{f1_xuid}_{f2_xuid}",
                        )

                        # Graphes (tout en bas, juste avant les médailles) — 1 graphe par ligne.
                        fig_spree = _plot_multi_metric_bars_by_match(
                            series,
                            metric_col="max_killing_spree",
                            title="Folie meurtrière (max)",
                            y_axis_title="Folie meurtrière (max)",
                            hover_label="folie meurtrière",
                            colors=colors_by_name,
                            smooth_window=10,
                            show_smooth_lines=show_smooth_teammates,
                        )
                        if fig_spree is None:
                            st.info("Aucune donnée de folie meurtrière (max) sur ces matchs.")
                        else:
                            st.plotly_chart(fig_spree, width="stretch", key=f"teammates_multi_spree_{len(series)}")

                        fig_hs = _plot_multi_metric_bars_by_match(
                            series,
                            metric_col="headshot_kills",
                            title="Tirs à la tête",
                            y_axis_title="Tirs à la tête",
                            hover_label="tirs à la tête",
                            colors=colors_by_name,
                            smooth_window=10,
                            show_smooth_lines=show_smooth_teammates,
                        )
                        if fig_hs is None:
                            st.info("Aucune donnée de tirs à la tête sur ces matchs.")
                        else:
                            st.plotly_chart(fig_hs, width="stretch", key=f"teammates_multi_hs_{len(series)}")

                        rendered_bottom_charts = True

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

                if not rendered_bottom_charts:
                    # S'il n'y a pas de trio (ou pas assez de données), on affiche quand même les graphes tout en bas.
                    fig_spree = _plot_multi_metric_bars_by_match(
                        series,
                        metric_col="max_killing_spree",
                        title="Folie meurtrière (max)",
                        y_axis_title="Folie meurtrière (max)",
                        hover_label="folie meurtrière",
                        colors=colors_by_name,
                        smooth_window=10,
                        show_smooth_lines=show_smooth_teammates,
                    )
                    if fig_spree is None:
                        st.info("Aucune donnée de folie meurtrière (max) sur ces matchs.")
                    else:
                        st.plotly_chart(fig_spree, width="stretch", key=f"teammates_multi_spree_{len(series)}")

                    fig_hs = _plot_multi_metric_bars_by_match(
                        series,
                        metric_col="headshot_kills",
                        title="Tirs à la tête",
                        y_axis_title="Tirs à la tête",
                        hover_label="tirs à la tête",
                        colors=colors_by_name,
                        smooth_window=10,
                        show_smooth_lines=show_smooth_teammates,
                    )
                    if fig_hs is None:
                        st.info("Aucune donnée de tirs à la tête sur ces matchs.")
                    else:
                        st.plotly_chart(fig_hs, width="stretch", key=f"teammates_multi_hs_{len(series)}")


    # --------------------------------------------------------------------------
    # Page: Historique des parties
    # --------------------------------------------------------------------------
    elif page == "Historique des parties":
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

        dff_table["app_match_url"] = dff_table["match_id"].astype(str).apply(
            lambda mid: _app_url("Match", match_id=str(mid))
        )

        # MMR équipe/adverse pour chaque match (source PlayerMatchStats).
        with st.spinner("Chargement des MMR (équipe/adverse)…"):
            def _mmr_tuple(match_id: str):
                pm = cached_load_player_match_result(db_path, str(match_id), xuid.strip(), db_key=db_key)
                if not isinstance(pm, dict):
                    return (None, None)
                return (pm.get("team_mmr"), pm.get("enemy_mmr"))

            mmr_pairs = dff_table["match_id"].astype(str).apply(_mmr_tuple)
            dff_table["team_mmr"] = mmr_pairs.apply(lambda t: t[0])
            dff_table["enemy_mmr"] = mmr_pairs.apply(lambda t: t[1])
            dff_table["delta_mmr"] = dff_table.apply(
                lambda r: (float(r.get("team_mmr")) - float(r.get("enemy_mmr")))
                if (r.get("team_mmr") is not None and r.get("enemy_mmr") is not None)
                else None,
                axis=1,
            )

        dff_table["start_time_fr"] = dff_table["start_time"].apply(_format_datetime_fr_hm)

        dff_table["average_life_mmss"] = dff_table["average_life_seconds"].apply(lambda x: format_mmss(x))

        show_cols = [
            "app_match_url",
            "match_url", "start_time_fr", "map_name", "playlist_fr", "mode_ui", "outcome_label", "score",
            "team_mmr", "enemy_mmr", "delta_mmr",
            "kda", "kills", "deaths", "max_killing_spree", "headshot_kills",
            "average_life_mmss", "assists", "accuracy", "ratio",
        ]
        table = dff_table[show_cols + ["start_time"]].sort_values("start_time", ascending=False).reset_index(drop=True)
        table = table[show_cols]

        styled = table.style
        styled = _styler_map(styled, _style_outcome_text, subset=["outcome_label"])
        styled = _styler_map(styled, _style_score_label, subset=["score"])
        styled = _styler_map(styled, _style_signed_number, subset=["kda"])
        styled = _styler_map(styled, _style_signed_number, subset=["delta_mmr"])

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "app_match_url": st.column_config.LinkColumn(
                    "Match",
                    display_text="Ouvrir",
                ),
                "match_url": st.column_config.LinkColumn(
                    "Consulter sur HaloWaypoint",
                    display_text="Ouvrir",
                ),
                "start_time_fr": st.column_config.TextColumn("Date de début"),
                "map_name": st.column_config.TextColumn("Carte"),
                "playlist_fr": st.column_config.TextColumn("Playlist"),
                "mode_ui": st.column_config.TextColumn("Mode"),
                "outcome_label": st.column_config.TextColumn("Résultat"),
                "score": st.column_config.TextColumn("Score"),
                "team_mmr": st.column_config.NumberColumn("MMR d'équipe", format="%.1f"),
                "enemy_mmr": st.column_config.NumberColumn("MMR adverse", format="%.1f"),
                "delta_mmr": st.column_config.NumberColumn("Écart MMR", format="%+.1f"),
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

        csv_table = table.rename(columns={"start_time_fr": "Date de début"})
        csv = csv_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger CSV",
            data=csv,
            file_name="openspartan_matches.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------------------------
    # Page: Paramètres
    # --------------------------------------------------------------------------
    elif page == "Paramètres":
        _render_settings_page(settings)


if __name__ == "__main__":
    main()
