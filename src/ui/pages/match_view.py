"""Page Match View - Affichage détaillé d'un match."""

from __future__ import annotations

import html
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    get_repo_root,
    HALO_COLORS,
    OUTCOME_CODES,
    BOT_MAP,
    TEAM_MAP,
)
from src.db import has_table
from src.db.parsers import parse_xuid_input
from src.analysis import compute_killer_victim_pairs, killer_victim_counts_long
from src.analysis.stats import format_mmss
from src.ui import (
    translate_playlist_name,
    translate_pair_name,
    display_name_from_xuid,
    AppSettings,
)
from src.ui.formatting import format_date_fr
from src.ui.medals import medal_label, render_medals_grid
from src.ui.components.performance import get_score_class, get_score_color
from src.analysis.performance_score import compute_relative_performance_score
from src.analysis.performance_config import SCORE_THRESHOLDS
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom


# =============================================================================
# Helpers internes
# =============================================================================


def _to_paris_naive_local(dt_value, paris_tz) -> datetime | None:
    """Convertit une date en datetime naïf (sans tzinfo) en heure de Paris."""
    if dt_value is None:
        return None
    try:
        ts = pd.to_datetime(dt_value, errors="coerce")
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            return ts.to_pydatetime()
        return ts.tz_convert(paris_tz).tz_localize(None).to_pydatetime()
    except Exception:
        return None


def _safe_dt(v, paris_tz) -> datetime | None:
    return _to_paris_naive_local(v, paris_tz)


def _match_time_window(
    row: pd.Series, *, tolerance_minutes: int, paris_tz
) -> tuple[datetime | None, datetime | None]:
    start = _safe_dt(row.get("start_time"), paris_tz)
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


def _paris_epoch_seconds_local(dt: datetime | None, paris_tz) -> float | None:
    """Convertit un datetime naïf Paris en epoch seconds."""
    if dt is None:
        return None
    try:
        aware = paris_tz.localize(dt) if dt.tzinfo is None else dt
        return aware.timestamp()
    except Exception:
        return None


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


def _render_media_section(
    *,
    row: pd.Series,
    settings: AppSettings,
    format_datetime_fn: Callable[[datetime | None], str],
    paris_tz,
) -> None:
    """Rend la section médias (captures/vidéos) pour un match."""
    if not bool(getattr(settings, "media_enabled", True)):
        return

    tol = int(getattr(settings, "media_tolerance_minutes", 0) or 0)
    t0, t1 = _match_time_window(row, tolerance_minutes=tol, paris_tz=paris_tz)
    if t0 is None or t1 is None:
        return

    screens_dir = str(getattr(settings, "media_screens_dir", "") or "").strip()
    videos_dir = str(getattr(settings, "media_videos_dir", "") or "").strip()

    if not screens_dir and not videos_dir:
        return

    st.subheader("Médias")
    st.caption(f"Fenêtre de recherche: {format_datetime_fn(t0)} → {format_datetime_fn(t1)}")

    # Note: _paris_epoch_seconds_local nécessite pytz, on utilise timestamp direct
    try:
        t0_epoch = t0.timestamp() if t0 else None
        t1_epoch = t1.timestamp() if t1 else None
    except Exception:
        t0_epoch = t1_epoch = None

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


def _os_card(
    title: str,
    kpi: str,
    sub_html: str | None = None,
    *,
    accent: str | None = None,
    kpi_color: str | None = None,
    sub_style: str | None = None,
    min_h: int = 112,
) -> None:
    """Rend une carte KPI avec style OpenSpartan."""
    t = html.escape(str(title or ""))
    k = html.escape(str(kpi or "-"))
    s = "" if not sub_html else str(sub_html)
    style = "min-height:" + str(int(min_h)) + "px; margin-bottom:10px;"
    if accent and str(accent).startswith("#"):
        style += f"border-color:{accent}66;"
    kpi_style = "" if not (kpi_color and str(kpi_color).startswith("#")) else f" style='color:{kpi_color}'"
    sub_style_attr = "" if not sub_style else " style=\"" + html.escape(str(sub_style), quote=True) + "\""
    st.markdown(
        "<div class='os-card' style='" + style + "'>"
        f"<div class='os-card-title'>{t}</div>"
        f"<div class='os-card-kpi'{kpi_style}>{k}</div>"
        + ("" if not s else f"<div class='os-card-sub'{sub_style_attr}>{s}</div>")
        + "</div>",
        unsafe_allow_html=True,
    )


def _map_thumb_path(row: pd.Series, map_id: str | None) -> str | None:
    """Trouve le chemin vers la miniature de la carte."""
    def _safe_stem_from_name(name: str | None) -> str:
        s = str(name or "").strip()
        if not s:
            return ""
        s = re.sub(r'[<>:"/\\|?*]', " ", s)
        s = re.sub(r"[\x00-\x1f]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    repo = Path(get_repo_root(__file__))
    base_dirs = [repo / "static" / "maps" / "thumbs", repo / "thumbs"]

    candidates: list[str] = []
    mid = str(map_id or "").strip()
    if mid and mid != "-":
        candidates.append(mid)

    safe_name = _safe_stem_from_name(row.get("map_name"))
    if safe_name:
        candidates.append(safe_name)
        candidates.append(safe_name.replace(" ", "_"))

    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)

    for base in base_dirs:
        for stem in uniq:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = base / f"{stem}{ext}"
                if p.exists():
                    return str(p)
    return None


# =============================================================================
# Fonction principale
# =============================================================================


def render_match_view(
    *,
    row: pd.Series,
    match_id: str,
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: tuple[int, int] | None,
    settings: AppSettings,
    df_full: pd.DataFrame | None = None,
    # Fonctions injectées
    normalize_mode_label_fn: Callable[[str | None], str],
    format_score_label_fn: Callable[[Any, Any], str],
    score_css_color_fn: Callable[[Any, Any], str],
    format_datetime_fn: Callable[[datetime | None], str],
    load_player_match_result_fn: Callable,
    load_match_medals_fn: Callable,
    load_highlight_events_fn: Callable,
    load_match_gamertags_fn: Callable,
    load_match_rosters_fn: Callable,
    paris_tz,
) -> None:
    """Rend la vue détaillée d'un match.

    Parameters
    ----------
    row : pd.Series
        Données du match depuis le DataFrame.
    match_id : str
        Identifiant du match.
    db_path : str
        Chemin vers la base de données.
    xuid : str
        XUID du joueur principal.
    waypoint_player : str
        Gamertag pour les liens Waypoint.
    db_key : tuple[int, int] | None
        Clé de cache pour la base de données.
    settings : AppSettings
        Paramètres de l'application.
    df_full : pd.DataFrame | None
        DataFrame complet pour le calcul du score relatif.
    normalize_mode_label_fn, format_score_label_fn, score_css_color_fn, format_datetime_fn
        Fonctions de formatage injectées.
    load_player_match_result_fn, load_match_medals_fn, load_highlight_events_fn,
    load_match_gamertags_fn, load_match_rosters_fn
        Fonctions de chargement de données injectées.
    paris_tz
        Timezone Paris.
    """
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
    score_label = format_score_label_fn(last_my_score, last_enemy_score)
    score_color = score_css_color_fn(last_my_score, last_enemy_score)

    wp = str(waypoint_player or "").strip()
    match_url = None
    if wp and match_id and match_id.strip() and match_id.strip() != "-":
        match_url = f"https://www.halowaypoint.com/halo-infinite/players/{wp}/matches/{match_id.strip()}"

    # Calcul du score de performance RELATIF
    perf_score = None
    if df_full is not None and len(df_full) >= 10:
        perf_score = compute_relative_performance_score(row, df_full)
    perf_display = f"{perf_score:.0f}" if perf_score is not None else "-"
    perf_color = None
    if perf_score is not None:
        if perf_score >= SCORE_THRESHOLDS["excellent"]:
            perf_color = colors["green"]
        elif perf_score >= SCORE_THRESHOLDS["good"]:
            perf_color = colors["cyan"]
        elif perf_score >= SCORE_THRESHOLDS["average"]:
            perf_color = colors["amber"]
        elif perf_score >= SCORE_THRESHOLDS["below_average"]:
            perf_color = colors.get("orange", "#FF8C00")
        else:
            perf_color = colors["red"]

    # Cartes KPI - Date, Résultat, Performance
    top_cols = st.columns(3)
    with top_cols[0]:
        _os_card("Date", format_date_fr(last_time))
    with top_cols[1]:
        # Utiliser des classes CSS pour le résultat
        outcome_class = "text-win" if "victoire" in str(outcome_label).lower() else (
            "text-loss" if "défaite" in str(outcome_label).lower() else "text-tie"
        )
        _os_card(
            "Résultats",
            str(outcome_label),
            f"<span class='{outcome_class} fw-bold'>{html.escape(str(score_label))}</span>",
            accent=str(outcome_color),
            kpi_color=str(outcome_color),
        )
    with top_cols[2]:
        _os_card(
            "Performance",
            perf_display,
            "Relatif à ton historique" if perf_score is not None else "Historique insuffisant",
            accent=perf_color,
            kpi_color=perf_color,
        )

    last_mode_ui = row.get("mode_ui") or normalize_mode_label_fn(str(last_pair) if last_pair else None)
    row_cols = st.columns(3)
    row_cols[0].metric(" ", str(last_map) if last_map else "-")
    row_cols[1].metric(
        " ",
        str(last_playlist_fr or last_playlist) if (last_playlist_fr or last_playlist) else "-",
    )
    row_cols[2].metric(
        " ",
        str(last_mode_ui or last_pair_fr or last_pair or last_mode)
        if (last_mode_ui or last_pair_fr or last_pair or last_mode)
        else "-",
    )

    # Miniature de la carte
    map_id = row.get("map_id")
    thumb = _map_thumb_path(row, str(map_id) if map_id else None)
    if thumb:
        c = st.columns([1, 2, 1])
        with c[1]:
            try:
                st.image(thumb, width=400)
            except Exception:
                pass

    # Stats détaillées
    with st.spinner("Lecture des stats détaillées (attendu vs réel, médailles)…"):
        pm = load_player_match_result_fn(db_path, match_id, xuid.strip(), db_key=db_key)
        medals_last = load_match_medals_fn(db_path, match_id, xuid.strip(), db_key=db_key)

    if not pm:
        st.info("Stats détaillées indisponibles pour ce match (PlayerMatchStats manquant ou format inattendu).")
    else:
        _render_expected_vs_actual(row, pm, colors)

    # Némésis / Souffre-douleur
    _render_nemesis_section(
        match_id=match_id,
        db_path=db_path,
        xuid=xuid,
        db_key=db_key,
        colors=colors,
        load_highlight_events_fn=load_highlight_events_fn,
        load_match_gamertags_fn=load_match_gamertags_fn,
    )

    # Roster
    _render_roster_section(
        match_id=match_id,
        db_path=db_path,
        xuid=xuid,
        db_key=db_key,
        load_match_rosters_fn=load_match_rosters_fn,
        load_match_gamertags_fn=load_match_gamertags_fn,
    )

    # Médailles
    st.subheader("Médailles")
    if not medals_last:
        st.info("Médailles indisponibles pour ce match (ou aucune médaille).")
    else:
        md_df = pd.DataFrame(medals_last)
        md_df["label"] = md_df["name_id"].apply(lambda x: medal_label(int(x)))
        md_df = md_df.sort_values(["count", "label"], ascending=[False, True])
        render_medals_grid(md_df[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)

    # Médias
    _render_media_section(
        row=row,
        settings=settings,
        format_datetime_fn=format_datetime_fn,
        paris_tz=paris_tz,
    )

    # Lien Waypoint
    if match_url:
        st.link_button("Ouvrir sur HaloWaypoint", match_url, width="stretch")


def _render_expected_vs_actual(row: pd.Series, pm: dict, colors: dict) -> None:
    """Rend la section Réel vs Attendu."""
    team_mmr = pm.get("team_mmr")
    enemy_mmr = pm.get("enemy_mmr")
    delta_mmr = (team_mmr - enemy_mmr) if (team_mmr is not None and enemy_mmr is not None) else None

    mmr_cols = st.columns(3)
    with mmr_cols[0]:
        _os_card("MMR d'équipe", f"{team_mmr:.1f}" if team_mmr is not None else "-")
    with mmr_cols[1]:
        _os_card("MMR adverse", f"{enemy_mmr:.1f}" if enemy_mmr is not None else "-")
    with mmr_cols[2]:
        if delta_mmr is None:
            _os_card("Écart MMR", "-")
        else:
            dm = float(delta_mmr)
            # Utiliser les variables CSS pour les couleurs
            col = "var(--color-win)" if dm > 0 else ("var(--color-loss)" if dm < 0 else "var(--color-tie)")
            _os_card("Écart MMR", f"{dm:+.1f}", "équipe - adverse", accent=col, kpi_color=col)

    def _ev_card(title: str, perf: dict, *, mode: str) -> None:
        count = perf.get("count")
        expected = perf.get("expected")
        if count is None or expected is None:
            _os_card(title, "-", "")
            return

        delta = float(count) - float(expected)
        if delta == 0:
            delta_class = "text-neutral"
        else:
            good = delta > 0
            if mode == "inverse":
                good = not good
            delta_class = "text-positive" if good else "text-negative"

        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        sub = f"<span class='{delta_class} fw-bold'>{arrow} {delta:+.1f}</span>"
        _os_card(title, f"{float(count):.0f} vs {float(expected):.1f}", sub)

    perf_k = pm.get("kills") or {}
    perf_d = pm.get("deaths") or {}
    perf_a = pm.get("assists") or {}

    st.subheader("Réel vs attendu")
    av_cols = st.columns(3)
    with av_cols[0]:
        _ev_card("Frags", perf_k, mode="normal")
    with av_cols[1]:
        _ev_card("Morts", perf_d, mode="inverse")
    with av_cols[2]:
        avg_life_last = row.get("average_life_seconds")
        _os_card("Durée de vie moyenne", format_mmss(avg_life_last), "")

    # Graphique F / D / A
    labels = ["F", "D", "A"]
    actual_vals = [
        float(row.get("kills") or 0.0),
        float(row.get("deaths") or 0.0),
        float(row.get("assists") or 0.0),
    ]
    exp_vals = [
        perf_k.get("expected"),
        perf_d.get("expected"),
        perf_a.get("expected"),
    ]

    real_ratio = row.get("ratio")
    try:
        real_ratio_f = float(real_ratio) if real_ratio == real_ratio else None
    except Exception:
        real_ratio_f = None
    if real_ratio_f is None:
        denom = max(1.0, float(row.get("deaths") or 0.0))
        real_ratio_f = (float(row.get("kills") or 0.0) + float(row.get("assists") or 0.0)) / denom

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

    exp_fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=40, r=20, t=30, b=90),
        legend=get_legend_horizontal_bottom(),
    )
    exp_fig.update_yaxes(title_text="F / D / A", rangemode="tozero", secondary_y=False)
    exp_fig.update_yaxes(title_text="Ratio", secondary_y=True)
    st.plotly_chart(exp_fig, width="stretch")

    # Folie meurtrière / Tirs à la tête
    spree_v = pd.to_numeric(row.get("max_killing_spree"), errors="coerce")
    headshots_v = pd.to_numeric(row.get("headshot_kills"), errors="coerce")
    if (spree_v == spree_v) or (headshots_v == headshots_v):
        st.subheader("Folie meurtrière / Tirs à la tête")
        fig_sh = go.Figure()
        fig_sh.add_trace(
            go.Bar(
                x=["Folie meurtrière (max)", "Tirs à la tête"],
                y=[
                    float(spree_v) if (spree_v == spree_v) else 0.0,
                    float(headshots_v) if (headshots_v == headshots_v) else 0.0,
                ],
                marker_color=[HALO_COLORS.violet, HALO_COLORS.cyan],
                opacity=0.85,
                hovertemplate="%{x}: %{y:.0f}<extra></extra>",
            )
        )
        fig_sh.update_layout(
            height=260,
            margin=dict(l=40, r=20, t=30, b=60),
            showlegend=False,
        )
        fig_sh.update_yaxes(rangemode="tozero")
        st.plotly_chart(apply_halo_plot_style(fig_sh, height=260), width="stretch")


def _render_nemesis_section(
    *,
    match_id: str,
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    colors: dict,
    load_highlight_events_fn: Callable,
    load_match_gamertags_fn: Callable,
) -> None:
    """Rend la section Némésis / Souffre-douleur."""
    st.subheader("Antagonistes du match")
    if not (match_id and match_id.strip() and has_table(db_path, "HighlightEvents")):
        st.caption(
            "Indisponible: la DB ne contient pas les highlight events. "
            "Si tu utilises une DB SPNKr, relance l'import avec `--with-highlight-events`."
        )
        return

    with st.spinner("Chargement des highlight events (film)…"):
        he = load_highlight_events_fn(db_path, match_id.strip(), db_key=db_key)

    match_gt_map = load_match_gamertags_fn(db_path, match_id.strip(), db_key=db_key)

    pairs = compute_killer_victim_pairs(he, tolerance_ms=5)
    if not pairs:
        st.info("Aucune paire kill/death trouvée (ou match sans timeline exploitable).")
        return

    kv_long = killer_victim_counts_long(pairs)

    me_xuid = str(xuid).strip()
    killed_me = kv_long[kv_long["victim_xuid"].astype(str) == me_xuid]
    i_killed = kv_long[kv_long["killer_xuid"].astype(str) == me_xuid]

    def _display_name_from_kv(xuid_value, gamertag_value) -> str:
        gt = str(gamertag_value or "").strip()
        xu_raw = str(xuid_value or "").strip()
        xu = parse_xuid_input(xu_raw) or xu_raw

        xu_key = str(xu).strip() if xu is not None else ""
        if xu_key and isinstance(match_gt_map, dict):
            mapped = match_gt_map.get(xu_key)
            if isinstance(mapped, str) and mapped.strip():
                return mapped.strip()

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

    def _count_kills(df_: pd.DataFrame, *, col: str, xuid_value: str) -> int | None:
        if df_ is None or df_.empty or not xuid_value:
            return None
        try:
            mask = df_[col].astype(str) == str(xuid_value)
            hit = df_.loc[mask]
            if hit.empty:
                return None
            return int(hit["count"].iloc[0])
        except Exception:
            return None

    nemesis_xu = ""
    bully_xu = ""
    if not killed_me.empty:
        try:
            nemesis_xu = str(killed_me.sort_values(["count"], ascending=[False]).iloc[0].get("killer_xuid") or "").strip()
        except Exception:
            nemesis_xu = ""
    if not i_killed.empty:
        try:
            bully_xu = str(i_killed.sort_values(["count"], ascending=[False]).iloc[0].get("victim_xuid") or "").strip()
        except Exception:
            bully_xu = ""

    nemesis_killed_me = nemesis_kills
    me_killed_nemesis = _count_kills(i_killed, col="victim_xuid", xuid_value=nemesis_xu)
    me_killed_bully = bully_kills
    bully_killed_me = _count_kills(killed_me, col="killer_xuid", xuid_value=bully_xu)

    def _cmp_color(deaths_: int | None, kills_: int | None) -> str:
        if deaths_ is None or kills_ is None:
            return colors["slate"]
        if int(deaths_) > int(kills_):
            return colors["red"]
        if int(deaths_) < int(kills_):
            return colors["green"]
        return colors["violet"]

    def _fmt_two_lines(deaths_: int | None, kills_: int | None) -> str:
        d = "-" if deaths_ is None else f"{int(deaths_)} morts"
        k = "-" if kills_ is None else f"Tué {int(kills_)} fois"
        return html.escape(d) + "<br/>" + html.escape(k)

    c = st.columns(2)
    with c[0]:
        _os_card(
            "Némésis",
            nemesis_name,
            _fmt_two_lines(nemesis_killed_me, me_killed_nemesis),
            accent=_cmp_color(nemesis_killed_me, me_killed_nemesis),
            sub_style="color: rgba(245, 248, 255, 0.92); font-weight: 800; font-size: 16px; line-height: 1.15;",
            min_h=110,
        )
    with c[1]:
        _os_card(
            "Souffre-douleur",
            bully_name,
            _fmt_two_lines(bully_killed_me, me_killed_bully),
            accent=_cmp_color(bully_killed_me, me_killed_bully),
            sub_style="color: rgba(245, 248, 255, 0.92); font-weight: 800; font-size: 16px; line-height: 1.15;",
            min_h=110,
        )


def _render_roster_section(
    *,
    match_id: str,
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    load_match_rosters_fn: Callable,
    load_match_gamertags_fn: Callable,
) -> None:
    """Rend la section Joueurs (roster)."""
    st.subheader("Joueurs")
    rosters = load_match_rosters_fn(db_path, match_id.strip(), xuid.strip(), db_key=db_key)
    if not rosters:
        st.info("Roster indisponible pour ce match (payload MatchStats manquant ou équipe introuvable).")
        return

    gt_map = load_match_gamertags_fn(db_path, match_id.strip(), db_key=db_key)
    me_xu = str(parse_xuid_input(str(xuid or "").strip()) or str(xuid or "").strip()).strip()

    my_team_id = rosters.get("my_team_id")
    my_team_name = rosters.get("my_team_name")
    enemy_team_ids = rosters.get("enemy_team_ids") or []
    enemy_team_names = rosters.get("enemy_team_names") or []

    def _team_label(team_id_value) -> str:
        try:
            tid = int(team_id_value)
        except Exception:
            return "-"
        return TEAM_MAP.get(tid) or f"Team {tid}"

    def _roster_name(xu: str, gt: str | None) -> str:
        xu_s = str(parse_xuid_input(str(xu or "").strip()) or str(xu or "").strip()).strip()

        if xu_s:
            bot_key = xu_s.strip()
            if bot_key.lower().startswith("bid("):
                bot_name = BOT_MAP.get(bot_key)
                if isinstance(bot_name, str) and bot_name.strip():
                    return bot_name.strip()

        if xu_s and isinstance(gt_map, dict):
            mapped = gt_map.get(xu_s)
            if isinstance(mapped, str) and mapped.strip():
                return mapped.strip()

        g = str(gt or "").strip()
        if g and g != "?" and (not g.isdigit()) and (not g.lower().startswith("xuid(")):
            return g

        if xu_s:
            return display_name_from_xuid(xu_s)
        return "-"

    my_rows = rosters.get("my_team") or []
    en_rows = rosters.get("enemy_team") or []

    my_names: list[tuple[str, bool]] = []
    en_names: list[tuple[str, bool]] = []

    for r in my_rows:
        xu = str(r.get("xuid") or "").strip()
        name = str(r.get("display_name") or "").strip() or _roster_name(xu, r.get("gamertag"))
        is_self = bool(me_xu and xu and (str(parse_xuid_input(xu) or xu).strip() == me_xu)) or bool(r.get("is_me"))
        my_names.append((name, is_self))

    for r in en_rows:
        xu = str(r.get("xuid") or "").strip()
        name = str(r.get("display_name") or "").strip() or _roster_name(xu, r.get("gamertag"))
        en_names.append((name, False))

    rows_n = max(len(my_names), len(en_names), 1)
    my_names += [("", False)] * (rows_n - len(my_names))
    en_names += [("", False)] * (rows_n - len(en_names))

    def _pill_html(name: str, *, side: str, is_self: bool) -> str:
        if not name:
            return "<span class='os-roster-empty'>—</span>"
        safe = html.escape(str(name))
        extra = " os-roster-pill--self" if is_self else ""
        return (
            f"<span class='os-roster-pill os-roster-pill--{side}{extra}'>"
            "<span class='os-roster-pill__dot'></span>"
            f"<span>{safe}</span>"
            "</span>"
        )

    body_rows = []
    for i in range(rows_n):
        n_me, is_self = my_names[i]
        n_en, _ = en_names[i]
        body_rows.append(
            "<tr>"
            f"<td>{_pill_html(n_me, side='me', is_self=is_self)}</td>"
            f"<td>{_pill_html(n_en, side='enemy', is_self=False)}</td>"
            "</tr>"
        )

    st.markdown(
        "<div class='os-table-wrap os-roster-wrap'>"
        "<table class='os-table os-roster'>"
        "<thead><tr>"
        f"<th class='os-roster-th os-roster-th--me'>Mon équipe — {html.escape(str(my_team_name or _team_label(my_team_id)))} ({len([n for n, _ in my_names if n])})</th>"
        f"<th class='os-roster-th os-roster-th--enemy'>Équipe adverse — {html.escape(str(enemy_team_names[0] if (isinstance(enemy_team_names, list) and len(enemy_team_names)==1 and enemy_team_names[0]) else (' / '.join([_team_label(t) for t in enemy_team_ids]) if enemy_team_ids else 'Adversaires')))} ({len([n for n, _ in en_names if n])})</th>"
        "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )
