"""Page Match View - Affichage détaillé d'un match.

Ce module a été refactorisé en sous-modules :
- match_view_helpers.py : Utilitaires date/heure, médias, composants UI
- match_view_charts.py : Graphiques Expected vs Actual
- match_view_players.py : Sections Némésis et Roster
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.config import HALO_COLORS, OUTCOME_CODES
from src.ui import (
    translate_playlist_name,
    translate_pair_name,
    AppSettings,
)
from src.ui.formatting import format_date_fr
from src.ui.medals import medal_label, render_medals_grid
from src.analysis.performance_score import compute_relative_performance_score
from src.analysis.performance_config import SCORE_THRESHOLDS

# Imports depuis les sous-modules
from src.ui.pages.match_view_helpers import (
    os_card,
    map_thumb_path,
    render_media_section,
)
from src.ui.pages.match_view_charts import render_expected_vs_actual
from src.ui.pages.match_view_players import (
    render_nemesis_section,
    render_roster_section,
)


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
        os_card("Date", format_date_fr(last_time))
    with top_cols[1]:
        outcome_class = "text-win" if "victoire" in str(outcome_label).lower() else (
            "text-loss" if "défaite" in str(outcome_label).lower() else "text-tie"
        )
        os_card(
            "Résultats",
            str(outcome_label),
            f"<span class='{outcome_class} fw-bold'>{html.escape(str(score_label))}</span>",
            accent=str(outcome_color),
            kpi_color=str(outcome_color),
        )
    with top_cols[2]:
        os_card(
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
    thumb = map_thumb_path(row, str(map_id) if map_id else None)
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
        render_expected_vs_actual(row, pm, colors, df_full=df_full)

    # Némésis / Souffre-douleur
    render_nemesis_section(
        match_id=match_id,
        db_path=db_path,
        xuid=xuid,
        db_key=db_key,
        colors=colors,
        load_highlight_events_fn=load_highlight_events_fn,
        load_match_gamertags_fn=load_match_gamertags_fn,
    )

    # Roster
    render_roster_section(
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
    render_media_section(
        row=row,
        settings=settings,
        format_datetime_fn=format_datetime_fn,
        paris_tz=paris_tz,
    )

    # Lien Waypoint
    if match_url:
        st.link_button("Ouvrir sur HaloWaypoint", match_url, width="stretch")


# =============================================================================
# Exports publics (rétrocompatibilité)
# =============================================================================

# Réexporter les fonctions helpers pour rétrocompatibilité
from src.ui.pages.match_view_helpers import (
    to_paris_naive_local as _to_paris_naive_local,
    safe_dt as _safe_dt,
    match_time_window as _match_time_window,
    paris_epoch_seconds_local as _paris_epoch_seconds_local,
    index_media_dir as _index_media_dir,
    render_media_section as _render_media_section,
    os_card as _os_card,
    map_thumb_path as _map_thumb_path,
)

from src.ui.pages.match_view_charts import (
    render_expected_vs_actual as _render_expected_vs_actual,
)

from src.ui.pages.match_view_players import (
    render_nemesis_section as _render_nemesis_section,
    render_roster_section as _render_roster_section,
)

__all__ = [
    "render_match_view",
    # Helpers (rétrocompatibilité)
    "_to_paris_naive_local",
    "_safe_dt",
    "_match_time_window",
    "_paris_epoch_seconds_local",
    "_index_media_dir",
    "_render_media_section",
    "_os_card",
    "_map_thumb_path",
    "_render_expected_vs_actual",
    "_render_nemesis_section",
    "_render_roster_section",
]
