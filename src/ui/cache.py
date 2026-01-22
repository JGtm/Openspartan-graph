"""Fonctions de cache Streamlit pour le dashboard.

Ce module regroupe toutes les fonctions @st.cache_data utilisées
pour éviter de recharger les données à chaque interaction.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from src.analysis import compute_sessions, mark_firefight
from src.db import (
    load_match_medals_for_player,
    load_match_rosters,
    load_matches,
    load_player_match_result,
    load_top_medals,
    load_highlight_events_for_match,
    load_match_player_gamertags,
    query_matches_with_friend,
    list_other_player_xuids,
    list_top_teammates,
)
from src.db.profiles import list_local_dbs
from src.ui import translate_playlist_name, translate_pair_name

if TYPE_CHECKING:
    pass

# Timezone Paris pour les conversions
PARIS_TZ_NAME = "Europe/Paris"


def db_cache_key(db_path: str) -> tuple[int, int] | None:
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
    """Requête les matchs joués avec un ami (cache)."""
    return query_matches_with_friend(db_path, self_xuid, friend_xuid)


@st.cache_data(show_spinner=False)
def cached_load_player_match_result(
    db_path: str,
    match_id: str,
    xuid: str,
    db_key: tuple[int, int] | None,
):
    """Charge le résultat d'un match pour un joueur (cache)."""
    return load_player_match_result(db_path, match_id, xuid)


@st.cache_data(show_spinner=False)
def cached_load_match_medals_for_player(
    db_path: str,
    match_id: str,
    xuid: str,
    db_key: tuple[int, int] | None,
):
    """Charge les médailles d'un match pour un joueur (cache)."""
    return load_match_medals_for_player(db_path, match_id, xuid)


@st.cache_data(show_spinner=False)
def cached_load_match_rosters(
    db_path: str,
    match_id: str,
    xuid: str,
    db_key: tuple[int, int] | None,
):
    """Charge les rosters d'un match (cache)."""
    _ = db_key
    return load_match_rosters(db_path, match_id, xuid)


@st.cache_data(show_spinner=False)
def cached_load_highlight_events_for_match(
    db_path: str,
    match_id: str,
    *,
    db_key: str | None = None,
):
    """Charge les événements highlight d'un match (cache)."""
    _ = db_key
    return load_highlight_events_for_match(db_path, match_id)


@st.cache_data(show_spinner=False)
def cached_load_match_player_gamertags(
    db_path: str,
    match_id: str,
    *,
    db_key: str | None = None,
):
    """Charge les gamertags des joueurs d'un match (cache)."""
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
    """Charge les top médailles (cache)."""
    return load_top_medals(
        db_path,
        xuid,
        list(match_ids),
        top_n=(int(top_n) if top_n is not None else None),
    )


def top_medals_smart(
    db_path: str,
    xuid: str,
    match_ids: list[str],
    *,
    top_n: int | None,
    db_key: tuple[int, int] | None,
):
    """Charge les top médailles avec gestion intelligente du cache.
    
    Évite de stocker d'immenses tuples en cache pour les grandes listes.
    """
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
    """Retourne un DataFrame des matchs joués avec un ami (cache)."""
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


def clear_app_caches() -> None:
    """Vide les caches Streamlit (utile si DB/alias/csv changent en dehors de l'app)."""
    try:
        st.cache_data.clear()
    except Exception:
        pass


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
            "team_mmr": [m.team_mmr for m in matches],
            "enemy_mmr": [m.enemy_mmr for m in matches],
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
