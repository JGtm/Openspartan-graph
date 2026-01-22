"""OpenSpartan Graphs - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données OpenSpartan Workshop.
"""

import os
import json
import html
import re
import subprocess
import sys
import time as time_module
import urllib.parse
from pathlib import Path
import uuid
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
)
from src.ui.components import compute_session_performance_score
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


_LABEL_SUFFIX_RE = re.compile(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", re.IGNORECASE)


# Alias pour compatibilité (fonctions déplacées vers src.ui.formatting)
_to_paris_naive = to_paris_naive
_paris_epoch_seconds = paris_epoch_seconds
_format_duration_hms = format_duration_hms
_format_duration_dhm = format_duration_dhm
_format_datetime_fr_hm = format_datetime_fr_hm
_format_score_label = format_score_label
_score_css_color = score_css_color
_style_outcome_text = style_outcome_text
_style_signed_number = style_signed_number
_style_score_label = style_score_label
_parse_date_fr_input = parse_date_fr_input
_coerce_int = coerce_int

# PARIS_TZ importés depuis src.ui.formatting


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


def _cleanup_orphan_tmp_dbs() -> None:
    """Nettoie les fichiers .tmp.*.db orphelins dans le dossier data/.
    
    Ces fichiers peuvent rester si un import SPNKr a été interrompu
    (crash, timeout, fermeture de l'app). On supprime ceux de plus de 1h.
    """
    if st.session_state.get("_tmp_db_cleanup_done"):
        return
    st.session_state["_tmp_db_cleanup_done"] = True
    
    try:
        repo_root = Path(__file__).resolve().parent
        data_dir = repo_root / "data"
        if not data_dir.exists():
            return
        
        now = time_module.time()
        one_hour_ago = now - 3600  # 1 heure
        
        # Pattern: *.tmp.*.db (ex: spnkr_gt_Madina.db.tmp.1234567890.12345.db)
        for tmp_file in data_dir.glob("*.tmp.*.db"):
            try:
                if tmp_file.stat().st_mtime < one_hour_ago:
                    tmp_file.unlink()
            except Exception:
                pass
        
        # Pattern alternatif: *.db.tmp.* sans extension finale
        for tmp_file in data_dir.glob("*.db.tmp.*"):
            try:
                if tmp_file.stat().st_mtime < one_hour_ago:
                    tmp_file.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _render_sync_indicator(db_path: str) -> None:
    """Affiche l'indicateur de dernière synchronisation dans la sidebar.
    
    Couleurs:
    - 🟢 Vert: sync < 1h
    - 🟡 Jaune: sync < 24h  
    - 🔴 Rouge: sync > 24h ou jamais
    """
    if not db_path or not os.path.exists(db_path):
        return
    
    meta = get_sync_metadata(db_path)
    last_sync = meta.get("last_sync_at")
    total_matches = meta.get("total_matches", 0)
    
    now = datetime.now(timezone.utc)
    
    if last_sync:
        delta = now - last_sync
        hours = delta.total_seconds() / 3600
        
        if hours < 1:
            minutes = int(delta.total_seconds() / 60)
            indicator = "🟢"
            time_str = f"il y a {minutes} min" if minutes > 0 else "à l'instant"
        elif hours < 24:
            indicator = "🟡"
            h = int(hours)
            time_str = f"il y a {h}h"
        else:
            indicator = "🔴"
            days = int(hours / 24)
            if days == 1:
                time_str = "il y a 1 jour"
            else:
                time_str = f"il y a {days} jours"
        
        sync_text = f"{indicator} Sync {time_str}"
    else:
        # Pas de métadonnées de sync, on utilise la date de modification du fichier
        try:
            mtime = os.path.getmtime(db_path)
            mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            delta = now - mtime_dt
            hours = delta.total_seconds() / 3600
            
            if hours < 1:
                indicator = "🟢"
                minutes = int(delta.total_seconds() / 60)
                time_str = f"il y a {minutes} min" if minutes > 0 else "à l'instant"
            elif hours < 24:
                indicator = "🟡"
                h = int(hours)
                time_str = f"il y a {h}h"
            else:
                indicator = "🔴"
                days = int(hours / 24)
                time_str = f"il y a {days} jour{'s' if days > 1 else ''}"
            
            sync_text = f"{indicator} Modifié {time_str}"
        except Exception:
            sync_text = "🔴 Sync inconnue"
    
    # Affichage compact
    match_info = f"({total_matches} matchs)" if total_matches > 0 else ""
    st.markdown(
        f"<div style='font-size: 0.85em; color: #888; margin: 4px 0 8px 0;'>"
        f"{sync_text} {match_info}</div>",
        unsafe_allow_html=True,
    )


def _refresh_spnkr_db_via_api(
    *,
    db_path: str,
    player: str,
    match_type: str,
    max_matches: int,
    rps: int,
    with_highlight_events: bool = True,
    with_aliases: bool = True,
    delta: bool = False,
    timeout_seconds: int = 180,
) -> tuple[bool, str]:
    """Rafraîchit une DB SPNKr en appelant scripts/spnkr_import_db.py.

    Retourne (ok, message) pour affichage UI.
    
    Args:
        with_highlight_events: Activer les highlight events (défaut: True)
        with_aliases: Activer le refresh des aliases (défaut: True)
        delta: Mode delta - s'arrête dès qu'un match connu est rencontré (défaut: False)
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
    # Highlight events et aliases sont activés par défaut côté import
    # On n'ajoute les flags --no-* que si explicitement désactivés
    if not with_highlight_events:
        cmd.append("--no-highlight-events")
    if not with_aliases:
        cmd.append("--no-aliases")
    # Mode delta: arrêt dès match connu (sync rapide)
    if delta:
        cmd.append("--delta")

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

# Note: _format_duration_hms, _coerce_int, _format_score_label, _score_css_color,
# _style_outcome_text, _style_signed_number, _style_score_label sont maintenant
# des alias vers les fonctions de src.ui.formatting (voir imports).

# Regex pour _style_score_label (utilisé par l'alias)
_SCORE_LABEL_RE = re.compile(r"^\s*(-?\d+)\s*[-–—]\s*(-?\d+)\s*$")


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


# Note: _format_datetime_fr_hm, _parse_date_fr_input sont maintenant des alias (voir imports).

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
    st.set_page_config(page_title="OpenSpartan Graphs", layout="wide")

    perf_reset_run()

    # Nettoyage des fichiers temporaires orphelins (une fois par session)
    _cleanup_orphan_tmp_dbs()

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
        render_perf_panel(location="sidebar")
        st.markdown("<div class='os-sidebar-brand'>OpenSpartan Graphs</div>", unsafe_allow_html=True)
        st.markdown("<div class='os-sidebar-divider'></div>", unsafe_allow_html=True)

        # Indicateur de dernière synchronisation
        if db_path and os.path.exists(db_path):
            _render_sync_indicator(db_path)

        # Bouton Sync rapide pour les DB SPNKr
        if db_path and _is_spnkr_db_path(db_path) and os.path.exists(db_path):
            # Déduire le joueur depuis le nom de la DB
            spnkr_player = infer_spnkr_player_from_db_path(db_path) or ""
            
            if spnkr_player:
                sync_col1, sync_col2 = st.columns([1, 1])
                with sync_col1:
                    sync_clicked = st.button(
                        "🔄 Sync",
                        key="quick_sync_button",
                        help="Synchronise les nouveaux matchs (mode delta: arrêt dès match connu).",
                        use_container_width=True,
                    )
                with sync_col2:
                    full_sync = st.button(
                        "📥 Full",
                        key="full_sync_button", 
                        help="Synchronisation complète (parcourt tout l'historique).",
                        use_container_width=True,
                    )
                
                if sync_clicked or full_sync:
                    with st.spinner("Synchronisation en cours..." if sync_clicked else "Sync complète en cours..."):
                        ok, msg = _refresh_spnkr_db_via_api(
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
        if "include_firefight" not in st.session_state:
            st.session_state["include_firefight"] = False
        if "restrict_playlists" not in st.session_state:
            st.session_state["restrict_playlists"] = False  # Désactivé par défaut pour voir tous les matchs

        # Filtres de type de partie
        st.toggle(
            "Inclure Firefight (PvE)",
            key="include_firefight",
            help="Inclut les parties Firefight (mode PvE) dans les statistiques.",
        )
        st.toggle(
            "Restreindre aux playlists classiques",
            key="restrict_playlists",
            help="Limite aux playlists Partie rapide, Arène/Assassin classé, Grand combat en équipe.",
        )
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

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
                "Aucune playlist n'a matché Partie rapide / Assassin classé / Arène classée. "
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
            ("Durée totale", total_play_txt),
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
        "Comparaison sessions",
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
            render_match_view(
                row=last_row,
                match_id=last_match_id,
                db_path=db_path,
                xuid=xuid,
                waypoint_player=waypoint_player,
                db_key=db_key,
                settings=settings,
                normalize_mode_label_fn=_normalize_mode_label,
                format_score_label_fn=_format_score_label,
                score_css_color_fn=_score_css_color,
                format_datetime_fn=_format_datetime_fr_hm,
                load_player_match_result_fn=cached_load_player_match_result,
                load_match_medals_fn=cached_load_match_medals_for_player,
                load_highlight_events_fn=cached_load_highlight_events_for_match,
                load_match_gamertags_fn=cached_load_match_player_gamertags,
                load_match_rosters_fn=cached_load_match_rosters,
                paris_tz=PARIS_TZ,
            )

    # --------------------------------------------------------------------------
    # Page: Match (recherche)
    # --------------------------------------------------------------------------
    elif page == "Match":
        st.caption("Afficher un match précis via un MatchId, une date/heure, ou une sélection.")

        # Entrée MatchId
        match_id_input = st.text_input("MatchId", key="match_id_input")

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
        st.selectbox(
            "Sélection rapide (filtres actuels)",
            options=["(aucun)"] + list(opts.keys()),
            index=0,
            key="match_quick_pick_label",
        )

        def _on_use_quick_match() -> None:
            picked = st.session_state.get("match_quick_pick_label")
            if isinstance(picked, str) and picked in opts:
                st.session_state["match_id_input"] = opts[picked]

        st.button("Utiliser ce match", width="stretch", on_click=_on_use_quick_match)

        # Recherche par date/heure
        with st.expander("Recherche par date/heure", expanded=False):
            dd = st.date_input("Date", value=date.today(), format="DD/MM/YYYY")
            tt = st.time_input("Heure", value=time(20, 0))
            tol_min = st.slider("Tolérance (minutes)", 0, 30, 10, 1)

            def _on_search_by_datetime() -> None:
                target = datetime.combine(dd, tt)
                all_df = df.copy()
                all_df["_dt"] = pd.to_datetime(all_df["start_time"], errors="coerce")
                all_df = all_df.dropna(subset=["_dt"]).copy()
                if all_df.empty:
                    st.warning("Aucune date exploitable dans la DB.")
                    return

                all_df["_diff"] = (all_df["_dt"] - target).abs()
                best = all_df.sort_values("_diff").iloc[0]
                diff_min = float(best["_diff"].total_seconds() / 60.0)
                if diff_min <= float(tol_min):
                    st.session_state["match_id_input"] = str(best.get("match_id") or "").strip()
                else:
                    st.warning(f"Aucun match trouvé dans ±{tol_min} min (le plus proche est à {diff_min:.1f} min).")

            st.button("Rechercher", width="stretch", on_click=_on_search_by_datetime)

        mid = str(match_id_input or "").strip()
        if not mid:
            st.info("Renseigne un MatchId ou utilise la sélection/recherche ci-dessus.")
        else:
            rows = df.loc[df["match_id"].astype(str) == mid]
            if rows.empty:
                st.warning("MatchId introuvable dans la DB actuelle.")
            else:
                match_row = rows.sort_values("start_time").iloc[-1]
                render_match_view(
                    row=match_row,
                    match_id=mid,
                    db_path=db_path,
                    xuid=xuid,
                    waypoint_player=waypoint_player,
                    db_key=db_key,
                    settings=settings,
                    normalize_mode_label_fn=_normalize_mode_label,
                    format_score_label_fn=_format_score_label,
                    score_css_color_fn=_score_css_color,
                    format_datetime_fn=_format_datetime_fr_hm,
                    load_player_match_result_fn=cached_load_player_match_result,
                    load_match_medals_fn=cached_load_match_medals_for_player,
                    load_highlight_events_fn=cached_load_highlight_events_for_match,
                    load_match_gamertags_fn=cached_load_match_player_gamertags,
                    load_match_rosters_fn=cached_load_match_rosters,
                    paris_tz=PARIS_TZ,
                )

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
    # Page: Comparaison sessions
    # --------------------------------------------------------------------------
    elif page == "Comparaison sessions":
        all_sessions_df = cached_compute_sessions_db(
            db_path, xuid.strip(), db_key, include_firefight, gap_minutes
        )
        render_session_comparison_page(all_sessions_df)

    # --------------------------------------------------------------------------
    # Page: Séries temporelles
    # --------------------------------------------------------------------------
    elif page == "Séries temporelles":
        render_timeseries_page(dff, me_name)

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
            include_firefight=include_firefight,
            waypoint_player=waypoint_player,
            build_friends_opts_map_fn=_build_friends_opts_map,
            assign_player_colors_fn=_assign_player_colors,
            plot_multi_metric_bars_fn=_plot_multi_metric_bars_by_match,
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
