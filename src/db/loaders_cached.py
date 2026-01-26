"""Loaders optimisés utilisant les tables de cache.

Ce module fournit des fonctions de chargement qui :
1. Utilisent MatchCache au lieu de parser MatchStats JSON
2. Supportent un fallback vers les loaders originaux si le cache n'existe pas
3. Sont compatibles avec l'API existante (mêmes signatures)

Usage:
    from src.db.loaders_cached import load_matches_cached, load_sessions_cached

    # Utilise le cache si disponible, sinon fallback sur load_matches()
    matches = load_matches_cached(db_path, xuid)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from src.db.connection import get_connection
from src.db.parsers import parse_iso_utc
from src.models import MatchRow


def _has_match_cache(con: sqlite3.Connection) -> bool:
    """Vérifie si la table MatchCache existe et contient des données."""
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='MatchCache'")
        if cur.fetchone() is None:
            return False
        cur.execute("SELECT COUNT(*) FROM MatchCache LIMIT 1")
        count = cur.fetchone()[0]
        return count > 0
    except sqlite3.OperationalError:
        return False


def load_matches_cached(
    db_path: str,
    xuid: str,
    *,
    playlist_filter: Optional[str] = None,
    map_mode_pair_filter: Optional[str] = None,
    map_filter: Optional[str] = None,
    game_variant_filter: Optional[str] = None,
    include_firefight: bool = True,
) -> List[MatchRow]:
    """Charge les matchs depuis MatchCache (optimisé).
    
    Signature compatible avec load_matches() mais utilise le cache pré-parsé.
    Si le cache n'existe pas, retourne une liste vide (pas de fallback auto
    pour éviter les surprises de performance).
    
    Args:
        db_path: Chemin vers le fichier .db.
        xuid: XUID du joueur.
        playlist_filter: Filtre optionnel sur playlist_id.
        map_mode_pair_filter: Filtre optionnel sur pair_id.
        map_filter: Filtre optionnel sur map_id.
        game_variant_filter: Filtre optionnel sur game_variant_id.
        include_firefight: Si False, exclut les matchs PvE.
        
    Returns:
        Liste de MatchRow triée par date croissante.
    """
    with get_connection(db_path) as con:
        if not _has_match_cache(con):
            return []
        
        # Construire la requête avec filtres optionnels
        query = """
            SELECT
                match_id, start_time, map_id, map_name, playlist_id, playlist_name,
                pair_id, pair_name, game_variant_id, game_variant_name,
                outcome, last_team_id, kda, max_killing_spree, headshot_kills,
                average_life_seconds, time_played_seconds, kills, deaths, assists,
                accuracy, my_team_score, enemy_team_score, team_mmr, enemy_mmr,
                session_id, session_label, performance_score, is_firefight, teammates_signature,
                known_teammates_count, is_with_friends
            FROM MatchCache
            WHERE xuid = ?
        """
        params: list = [xuid]
        
        if playlist_filter is not None:
            query += " AND playlist_id = ?"
            params.append(playlist_filter)
        
        if map_mode_pair_filter is not None:
            query += " AND pair_id = ?"
            params.append(map_mode_pair_filter)
        
        if map_filter is not None:
            query += " AND map_id = ?"
            params.append(map_filter)
        
        if game_variant_filter is not None:
            query += " AND game_variant_id = ?"
            params.append(game_variant_filter)
        
        if not include_firefight:
            query += " AND is_firefight = 0"
        
        query += " ORDER BY start_time ASC"
        
        cur = con.cursor()
        cur.execute(query, params)
        
        rows: List[MatchRow] = []
        for row in cur.fetchall():
            (
                match_id, start_time_str, map_id, map_name, playlist_id, playlist_name,
                pair_id, pair_name, game_variant_id, game_variant_name,
                outcome, last_team_id, kda, max_spree, headshots,
                avg_life, time_played, kills, deaths, assists,
                accuracy, my_team_score, enemy_team_score, team_mmr, enemy_mmr,
                session_id, session_label, perf_score, is_ff, teammates_sig,
                known_teammates_count, is_with_friends_int, friends_xuids_str,
            ) = row
            
            # Parser le timestamp
            try:
                start_time = parse_iso_utc(start_time_str)
            except Exception:
                start_time = datetime.now(timezone.utc)
            
            rows.append(
                MatchRow(
                    match_id=match_id,
                    start_time=start_time,
                    map_id=map_id,
                    map_name=map_name,
                    playlist_id=playlist_id,
                    playlist_name=playlist_name,
                    map_mode_pair_id=pair_id,
                    map_mode_pair_name=pair_name,
                    game_variant_id=game_variant_id,
                    game_variant_name=game_variant_name,
                    outcome=outcome,
                    last_team_id=last_team_id,
                    kda=kda,
                    max_killing_spree=max_spree,
                    headshot_kills=headshots,
                    average_life_seconds=avg_life,
                    time_played_seconds=time_played,
                    kills=kills or 0,
                    deaths=deaths or 0,
                    assists=assists or 0,
                    accuracy=accuracy,
                    my_team_score=my_team_score,
                    enemy_team_score=enemy_team_score,
                    team_mmr=team_mmr,
                    enemy_mmr=enemy_mmr,
                    known_teammates_count=known_teammates_count or 0,
                    is_with_friends=bool(is_with_friends_int),
                    friends_xuids=friends_xuids_str or "",
                )
            )
        
        return rows


def has_cache_tables(db_path: str) -> bool:
    """Vérifie si les tables de cache existent et sont peuplées."""
    try:
        with get_connection(db_path) as con:
            return _has_match_cache(con)
    except Exception:
        return False


def get_cache_stats(db_path: str, xuid: str) -> dict:
    """Retourne des statistiques sur le cache pour un joueur."""
    with get_connection(db_path) as con:
        if not _has_match_cache(con):
            return {"has_cache": False}
        
        cur = con.cursor()
        
        # Nombre de matchs
        cur.execute("SELECT COUNT(*) FROM MatchCache WHERE xuid = ?", (xuid,))
        match_count = cur.fetchone()[0]
        
        # Nombre de sessions
        cur.execute(
            "SELECT COUNT(DISTINCT session_id) FROM MatchCache WHERE xuid = ? AND session_id IS NOT NULL",
            (xuid,),
        )
        session_count = cur.fetchone()[0]
        
        # Version du schéma
        schema_version = None
        try:
            cur.execute("SELECT value FROM CacheMeta WHERE key = 'schema_version'")
            row = cur.fetchone()
            if row:
                schema_version = row[0]
        except sqlite3.OperationalError:
            pass
        
        # Dernière mise à jour
        cur.execute(
            "SELECT MAX(updated_at) FROM MatchCache WHERE xuid = ?",
            (xuid,),
        )
        last_update = cur.fetchone()[0]
        
        return {
            "has_cache": True,
            "match_count": match_count,
            "session_count": session_count,
            "schema_version": schema_version,
            "last_update": last_update,
        }


def load_sessions_cached(
    db_path: str,
    xuid: str,
    *,
    include_firefight: bool = True,
) -> list[dict]:
    """Charge les sessions pré-calculées depuis MatchCache.
    
    Returns:
        Liste de dicts avec session_id, session_label, match_count, etc.
    """
    with get_connection(db_path) as con:
        if not _has_match_cache(con):
            return []
        
        query = """
            SELECT
                session_id,
                session_label,
                COUNT(*) as match_count,
                MIN(start_time) as first_match,
                MAX(start_time) as last_match,
                SUM(kills) as total_kills,
                SUM(deaths) as total_deaths,
                SUM(assists) as total_assists,
                AVG(performance_score) as avg_performance,
                SUM(CASE WHEN outcome = 2 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 3 THEN 1 ELSE 0 END) as losses
            FROM MatchCache
            WHERE xuid = ? AND session_id IS NOT NULL
        """
        params: list = [xuid]
        
        if not include_firefight:
            query += " AND is_firefight = 0"
        
        query += " GROUP BY session_id ORDER BY first_match DESC"
        
        cur = con.cursor()
        cur.execute(query, params)
        
        sessions = []
        for row in cur.fetchall():
            (
                session_id, session_label, match_count, first_match, last_match,
                total_kills, total_deaths, total_assists, avg_perf, wins, losses,
            ) = row
            
            sessions.append({
                "session_id": session_id,
                "session_label": session_label,
                "match_count": match_count,
                "first_match": first_match,
                "last_match": last_match,
                "total_kills": total_kills or 0,
                "total_deaths": total_deaths or 0,
                "total_assists": total_assists or 0,
                "avg_performance": round(avg_perf, 1) if avg_perf else None,
                "wins": wins or 0,
                "losses": losses or 0,
                "kd_ratio": (total_kills / total_deaths) if total_deaths else None,
            })
        
        return sessions


def load_session_matches_cached(
    db_path: str,
    xuid: str,
    session_id: int,
) -> List[MatchRow]:
    """Charge les matchs d'une session spécifique."""
    with get_connection(db_path) as con:
        if not _has_match_cache(con):
            return []
        
        query = """
            SELECT
                match_id, start_time, map_id, map_name, playlist_id, playlist_name,
                pair_id, pair_name, game_variant_id, game_variant_name,
                outcome, last_team_id, kda, max_killing_spree, headshot_kills,
                average_life_seconds, time_played_seconds, kills, deaths, assists,
                accuracy, my_team_score, enemy_team_score, team_mmr, enemy_mmr,
                session_id, session_label, performance_score, is_firefight, teammates_signature,
                known_teammates_count, is_with_friends, friends_xuids
            FROM MatchCache
            WHERE xuid = ? AND session_id = ?
            ORDER BY start_time ASC
        """
        
        cur = con.cursor()
        cur.execute(query, (xuid, session_id))
        
        rows: List[MatchRow] = []
        for row in cur.fetchall():
            (
                match_id, start_time_str, map_id, map_name, playlist_id, playlist_name,
                pair_id, pair_name, game_variant_id, game_variant_name,
                outcome, last_team_id, kda, max_spree, headshots,
                avg_life, time_played, kills, deaths, assists,
                accuracy, my_team_score, enemy_team_score, team_mmr, enemy_mmr,
                sess_id, sess_label, perf_score, is_ff, teammates_sig,
                known_teammates_count, is_with_friends_int, friends_xuids_str,
            ) = row
            
            try:
                start_time = parse_iso_utc(start_time_str)
            except Exception:
                start_time = datetime.now(timezone.utc)
            
            rows.append(
                MatchRow(
                    match_id=match_id,
                    start_time=start_time,
                    map_id=map_id,
                    map_name=map_name,
                    playlist_id=playlist_id,
                    playlist_name=playlist_name,
                    map_mode_pair_id=pair_id,
                    map_mode_pair_name=pair_name,
                    game_variant_id=game_variant_id,
                    game_variant_name=game_variant_name,
                    outcome=outcome,
                    last_team_id=last_team_id,
                    kda=kda,
                    max_killing_spree=max_spree,
                    headshot_kills=headshots,
                    average_life_seconds=avg_life,
                    time_played_seconds=time_played,
                    kills=kills or 0,
                    deaths=deaths or 0,
                    assists=assists or 0,
                    accuracy=accuracy,
                    my_team_score=my_team_score,
                    enemy_team_score=enemy_team_score,
                    team_mmr=team_mmr,
                    enemy_mmr=enemy_mmr,
                    known_teammates_count=known_teammates_count or 0,
                    is_with_friends=bool(is_with_friends_int),
                    friends_xuids=friends_xuids_str or "",
                )
            )
        
        return rows


def load_top_teammates_cached(
    db_path: str,
    xuid: str,
    limit: int = 20,
) -> list[tuple[str, str | None, int, int, int]]:
    """Charge les top coéquipiers depuis TeammatesAggregate.
    
    Returns:
        Liste de tuples (teammate_xuid, gamertag, matches_together, wins, losses)
    """
    with get_connection(db_path) as con:
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT teammate_xuid, teammate_gamertag, matches_together, wins_together, losses_together
                FROM TeammatesAggregate
                WHERE xuid = ?
                ORDER BY matches_together DESC
                LIMIT ?
                """,
                (xuid, limit),
            )
            return [
                (row[0], row[1], row[2], row[3], row[4])
                for row in cur.fetchall()
            ]
        except sqlite3.OperationalError:
            return []


def load_friends(db_path: str, owner_xuid: str) -> list[dict]:
    """Charge la liste des amis depuis la table Friends.
    
    Returns:
        Liste de dicts avec friend_xuid, friend_gamertag, nickname.
    """
    with get_connection(db_path) as con:
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT friend_xuid, friend_gamertag, nickname, added_at
                FROM Friends
                WHERE owner_xuid = ?
                ORDER BY friend_gamertag
                """,
                (owner_xuid,),
            )
            return [
                {
                    "xuid": row[0],
                    "gamertag": row[1],
                    "nickname": row[2],
                    "added_at": row[3],
                }
                for row in cur.fetchall()
            ]
        except sqlite3.OperationalError:
            return []


def get_match_session_info(db_path: str, match_id: str) -> dict | None:
    """Retourne les infos de session pour un match spécifique."""
    with get_connection(db_path) as con:
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT session_id, session_label, teammates_signature, performance_score
                FROM MatchCache
                WHERE match_id = ?
                """,
                (match_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                "session_id": row[0],
                "session_label": row[1],
                "teammates": row[2].split(",") if row[2] else [],
                "performance_score": row[3],
            }
        except sqlite3.OperationalError:
            return None
