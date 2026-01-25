"""Pages Dernier match et Match (recherche).

Ce module contient les fonctions de rendu pour :
- La page "Dernier match" (dernière partie selon les filtres)
- La page "Match" (recherche par MatchId, date/heure ou sélection rapide)
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo
    from src.ui.settings import AppSettings


def render_last_match_page(
    dff: pd.DataFrame,
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: tuple[int, int] | None,
    settings: "AppSettings",
    df_full: pd.DataFrame | None,
    render_match_view_fn: Callable,
    normalize_mode_label_fn: Callable[[str | None], str | None],
    format_score_label_fn: Callable,
    score_css_color_fn: Callable,
    format_datetime_fn: Callable,
    load_player_match_result_fn: Callable,
    load_match_medals_fn: Callable,
    load_highlight_events_fn: Callable,
    load_match_gamertags_fn: Callable,
    load_match_rosters_fn: Callable,
    paris_tz: "ZoneInfo",
) -> None:
    """Rend la page Dernier match.
    
    Affiche la dernière partie selon la sélection/filtres actuels.
    
    Args:
        dff: DataFrame filtré des matchs.
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        waypoint_player: Nom du joueur Waypoint.
        db_key: Clé de cache de la DB.
        settings: Paramètres de l'application.
        df_full: DataFrame complet pour le calcul du score relatif.
        render_match_view_fn: Fonction de rendu du match.
        normalize_mode_label_fn: Fonction de normalisation du label de mode.
        format_score_label_fn: Fonction de formatage du score.
        score_css_color_fn: Fonction de couleur CSS du score.
        format_datetime_fn: Fonction de formatage date/heure.
        load_player_match_result_fn: Fonction de chargement du résultat joueur.
        load_match_medals_fn: Fonction de chargement des médailles.
        load_highlight_events_fn: Fonction de chargement des événements.
        load_match_gamertags_fn: Fonction de chargement des gamertags.
        load_match_rosters_fn: Fonction de chargement des rosters.
        paris_tz: Timezone Paris.
    """
    st.caption("Dernière partie selon la sélection/filtres actuels.")
    
    if dff.empty:
        st.info("Aucun match disponible avec les filtres actuels.")
        return
    
    last_row = dff.sort_values("start_time").iloc[-1]
    last_match_id = str(last_row.get("match_id", "")).strip()
    
    render_match_view_fn(
        row=last_row,
        match_id=last_match_id,
        db_path=db_path,
        xuid=xuid,
        waypoint_player=waypoint_player,
        db_key=db_key,
        settings=settings,
        df_full=df_full,
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


def render_match_search_page(
    df: pd.DataFrame,
    dff: pd.DataFrame,
    db_path: str,
    xuid: str,
    waypoint_player: str,
    db_key: tuple[int, int] | None,
    settings: "AppSettings",
    df_full: pd.DataFrame | None,
    render_match_view_fn: Callable,
    normalize_mode_label_fn: Callable[[str | None], str | None],
    format_score_label_fn: Callable,
    score_css_color_fn: Callable,
    format_datetime_fn: Callable,
    load_player_match_result_fn: Callable,
    load_match_medals_fn: Callable,
    load_highlight_events_fn: Callable,
    load_match_gamertags_fn: Callable,
    load_match_rosters_fn: Callable,
    paris_tz: "ZoneInfo",
) -> None:
    """Rend la page Match (recherche).
    
    Permet de rechercher un match par MatchId, date/heure ou sélection rapide.
    
    Args:
        df: DataFrame complet des matchs (non filtré).
        dff: DataFrame filtré des matchs.
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        waypoint_player: Nom du joueur Waypoint.
        db_key: Clé de cache de la DB.
        settings: Paramètres de l'application.
        df_full: DataFrame complet pour le calcul du score relatif.
        render_match_view_fn: Fonction de rendu du match.
        normalize_mode_label_fn: Fonction de normalisation du label de mode.
        format_score_label_fn: Fonction de formatage du score.
        score_css_color_fn: Fonction de couleur CSS du score.
        format_datetime_fn: Fonction de formatage date/heure.
        load_player_match_result_fn: Fonction de chargement du résultat joueur.
        load_match_medals_fn: Fonction de chargement des médailles.
        load_highlight_events_fn: Fonction de chargement des événements.
        load_match_gamertags_fn: Fonction de chargement des gamertags.
        load_match_rosters_fn: Fonction de chargement des rosters.
        paris_tz: Timezone Paris.
    """
    st.caption("Afficher un match précis via un MatchId, une date/heure, ou une sélection.")

    # Entrée MatchId
    match_id_input = st.text_input("MatchId", key="match_id_input")

    # Sélection rapide (sur les filtres actuels, triés du plus récent au plus ancien)
    quick_df = dff.sort_values("start_time", ascending=False).head(200).copy()
    quick_df["start_time_fr"] = quick_df["start_time"].apply(format_datetime_fn)
    if "mode_ui" not in quick_df.columns:
        quick_df["mode_ui"] = quick_df["pair_name"].apply(normalize_mode_label_fn)
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
            render_match_view_fn(
                row=match_row,
                match_id=mid,
                db_path=db_path,
                xuid=xuid,
                waypoint_player=waypoint_player,
                db_key=db_key,
                settings=settings,
                df_full=df_full,
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
