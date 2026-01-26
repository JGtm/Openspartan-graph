#!/usr/bin/env python3
"""Script de migration pour peupler les tables de cache optimisées.

Ce script :
1. Crée les tables de cache si elles n'existent pas
2. Peuple MatchCache depuis MatchStats (parsing JSON → colonnes scalaires)
3. Calcule les sessions avec la nouvelle logique (gap 2h + coéquipiers)
4. Pré-calcule les scores de performance par match et session
5. Agrège les statistiques de coéquipiers
6. Insère les amis prédéfinis

Usage:
    python scripts/migrate_to_cache.py [--db PATH] [--xuid XUID] [--force]

Options:
    --db PATH     Chemin vers la DB (défaut: auto-détecté)
    --xuid XUID   XUID du joueur principal (défaut: depuis env)
    --force       Réinitialise les tables de cache avant migration
    --dry-run     Affiche ce qui serait fait sans modifier la DB
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_default_db_path, DEFAULT_PLAYER_XUID
from src.db.connection import get_connection
from src.db.schema import (
    get_all_cache_table_ddl,
    get_cache_table_names,
    CACHE_SCHEMA_VERSION,
)
from src.db.parsers import coerce_number as _coerce_number_base, parse_iso_utc


def coerce_number(v: Any, type_: type = float, default: Any = None) -> Any:
    """Version étendue de coerce_number avec type et default."""
    result = _coerce_number_base(v)
    if result is None:
        return default
    try:
        return type_(result)
    except (ValueError, TypeError):
        return default


# =============================================================================
# Configuration de la migration
# =============================================================================

# Gap de session en minutes (2h)
SESSION_GAP_MINUTES = 120

# Heure de coupure pour les sessions "en cours" (8h du matin)
SESSION_CUTOFF_HOUR = 8

# Amis prédéfinis (gamertag -> sera résolu en XUID si possible)
PREDEFINED_FRIENDS = [
    {"gamertag": "Madina97294", "nickname": "Madina"},
    {"gamertag": "Chocoboflor", "nickname": "Chocobo"},
]

# Playlists/modes Firefight (PvE)
FIREFIGHT_PATTERNS = [
    "firefight",
    "pve",
    "flood",
    "spartan ops",
]


# =============================================================================
# RÈGLES DE SESSION - FACILE À MODIFIER
# =============================================================================

# XUIDs des amis proches (seuls ceux-ci comptent pour le changement de session)
# Laisser vide pour considérer TOUS les coéquipiers
# NOTE: Sera remplacé par la liste d'amis depuis la table Friends si disponible
FRIENDS_XUIDS: set[str] = {
    "2533274858283686",  # Madina97294
    "2535469190789936",  # Chocoboflor
}


def load_friends_xuids_from_db(con: sqlite3.Connection, owner_xuid: str) -> set[str]:
    """Charge les XUIDs des amis depuis la table Friends.
    
    Args:
        con: Connexion SQLite.
        owner_xuid: XUID du joueur principal.
    
    Returns:
        Set des XUIDs des amis connus.
    """
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT friend_xuid FROM Friends WHERE owner_xuid = ?",
            (owner_xuid,),
        )
        xuids = {row[0] for row in cur.fetchall() if row[0] and not row[0].startswith("unknown:")}
        return xuids
    except sqlite3.OperationalError:
        return set()


def compute_known_teammates(
    teammates_sig: str,
    friends_xuids: set[str],
) -> tuple[int, bool, str]:
    """Calcule le nombre de coéquipiers connus (amis) dans un match.
    
    Args:
        teammates_sig: Signature des coéquipiers (XUIDs séparés par virgule).
        friends_xuids: Set des XUIDs des amis connus.
    
    Returns:
        Tuple (known_teammates_count, is_with_friends, friends_xuids_str).
        friends_xuids_str contient les XUIDs des amis présents, séparés par virgule.
    """
    if not teammates_sig or not friends_xuids:
        return (0, False, "")
    
    teammates = set(teammates_sig.split(","))
    known = teammates & friends_xuids
    count = len(known)
    # Retourner les XUIDs des amis présents, triés pour cohérence
    friends_str = ",".join(sorted(known)) if known else ""
    return (count, count > 0, friends_str)


def should_start_new_session_on_teammate_change(
    prev_teammates: set[str],
    curr_teammates: set[str],
) -> bool:
    """
    Détermine si un changement de coéquipiers déclenche une nouvelle session.
    
    ╔════════════════════════════════════════════════════════════════════════╗
    ║  RÈGLES ACTUELLES (modifiez ici pour changer le comportement) :        ║
    ║                                                                        ║
    ║  Si FRIENDS_XUIDS est défini :                                         ║
    ║  • Seuls les amis proches sont considérés pour le calcul               ║
    ║  • Les joueurs aléatoires matchmaking sont ignorés                     ║
    ║                                                                        ║
    ║  Logique :                                                             ║
    ║  • Nouvelle session si un AMI rejoint                                  ║
    ║  • Nouvelle session si on passe de "avec amis" à SOLO                  ║
    ║  • MÊME session si un ami part (sauf passage à solo)                   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    
    Args:
        prev_teammates: Set des XUIDs des coéquipiers du match précédent.
        curr_teammates: Set des XUIDs des coéquipiers du match actuel.
    
    Returns:
        True si on doit démarrer une nouvelle session.
    
    Exemples (avec amis A, B) :
        [A, B, random] → [A, B]        = False (juste un random part)
        [A, random] → [A, B, random]   = True  (B rejoint)
        [A, B] → [random, random]      = True  (passage à "sans amis")
        [A] → []                       = True  (passage à solo)
    """
    # Si FRIENDS_XUIDS est défini, ne considérer que les amis
    if FRIENDS_XUIDS:
        prev_friends = prev_teammates & FRIENDS_XUIDS
        curr_friends = curr_teammates & FRIENDS_XUIDS
    else:
        # Sinon considérer tous les coéquipiers
        prev_friends = prev_teammates
        curr_friends = curr_teammates
    
    # Cas 1: Passage à "sans amis" (curr_friends vide alors que prev_friends non vide)
    if not curr_friends and prev_friends:
        return True
    
    # Cas 2: Un ami rejoint
    new_friends = curr_friends - prev_friends
    if new_friends:
        return True
    
    # Cas 3: Des amis partent mais aucun nouveau → même session
    return False


# =============================================================================
# Helpers de parsing (copiés/adaptés de loaders.py pour autonomie)
# =============================================================================

def _find_player_in_match(players: list, xuid: str) -> dict | None:
    """Trouve le joueur dans la liste Players du match."""
    target_ids = {
        xuid,
        f"xuid({xuid})",
        xuid.lower(),
        f"xuid({xuid.lower()})",
    }
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get("PlayerId")
        if isinstance(pid, str) and pid.lower() in {t.lower() for t in target_ids}:
            return p
        if isinstance(pid, dict):
            xuid_val = pid.get("Xuid") or pid.get("xuid")
            if str(xuid_val) in target_ids:
                return p
    return None


def _get_team_xuids(players: list, team_id: int, self_xuid: str) -> list[str]:
    """Retourne les XUIDs des coéquipiers (même équipe, hors soi-même)."""
    teammates = []
    self_ids = {self_xuid, f"xuid({self_xuid})"}
    
    for p in players:
        if not isinstance(p, dict):
            continue
        
        # Extraire team_id du joueur
        p_team = p.get("LastTeamId")
        if p_team is None:
            # Essayer dans PlayerTeamStats
            team_stats = p.get("PlayerTeamStats")
            if isinstance(team_stats, list) and team_stats:
                p_team = team_stats[0].get("TeamId")
        
        if p_team != team_id:
            continue
        
        # Extraire XUID
        pid = p.get("PlayerId")
        xuid_str = None
        if isinstance(pid, str):
            # Format "xuid(123456)" ou juste "123456"
            m = re.match(r"xuid\((\d+)\)", pid, re.IGNORECASE)
            if m:
                xuid_str = m.group(1)
            elif pid.isdigit():
                xuid_str = pid
        elif isinstance(pid, dict):
            xuid_str = str(pid.get("Xuid") or pid.get("xuid") or "")
        
        if not xuid_str or xuid_str in self_ids or f"xuid({xuid_str})" in self_ids:
            continue
        
        # Ignorer les bots
        if xuid_str.startswith("bid("):
            continue
        
        teammates.append(xuid_str)
    
    return sorted(teammates)


def _extract_core_stats(me: dict) -> dict[str, Any]:
    """Extrait les stats principales du joueur."""
    stats: dict[str, Any] = {
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "accuracy": None,
        "kda": None,
        "time_played_seconds": None,
        "average_life_seconds": None,
        "max_killing_spree": None,
        "headshot_kills": None,
    }
    
    team_stats = me.get("PlayerTeamStats")
    if not isinstance(team_stats, list) or not team_stats:
        return stats
    
    first_team = team_stats[0]
    if not isinstance(first_team, dict):
        return stats
    
    s = first_team.get("Stats", {})
    core = s.get("CoreStats", {})
    
    stats["kills"] = coerce_number(core.get("Kills"), int, 0)
    stats["deaths"] = coerce_number(core.get("Deaths"), int, 0)
    stats["assists"] = coerce_number(core.get("Assists"), int, 0)
    stats["kda"] = coerce_number(core.get("KDA"), float, None)
    stats["max_killing_spree"] = coerce_number(core.get("MaxKillingSpree"), int, None)
    stats["headshot_kills"] = coerce_number(core.get("HeadshotKills"), int, None)
    stats["average_life_seconds"] = coerce_number(core.get("AverageLifeDuration"), float, None)
    
    # Accuracy
    shots = s.get("ShotStats", {})
    accuracy = shots.get("Accuracy")
    if accuracy is not None:
        stats["accuracy"] = coerce_number(accuracy, float, None)
    
    # Time played
    time_played = core.get("TimePlayed")
    if isinstance(time_played, str) and time_played.startswith("PT"):
        # Format ISO 8601 duration: PT12M34S
        try:
            total_seconds = 0.0
            time_str = time_played[2:]  # Remove 'PT'
            
            # Hours
            if "H" in time_str:
                h_idx = time_str.index("H")
                total_seconds += float(time_str[:h_idx]) * 3600
                time_str = time_str[h_idx + 1:]
            
            # Minutes
            if "M" in time_str:
                m_idx = time_str.index("M")
                total_seconds += float(time_str[:m_idx]) * 60
                time_str = time_str[m_idx + 1:]
            
            # Seconds
            if "S" in time_str:
                s_idx = time_str.index("S")
                total_seconds += float(time_str[:s_idx])
            
            stats["time_played_seconds"] = total_seconds
        except Exception:
            pass
    
    return stats


def _extract_outcome_team(me: dict) -> tuple[int | None, int | None]:
    """Extrait outcome et team_id."""
    outcome = me.get("Outcome")
    if isinstance(outcome, str):
        outcome_map = {"tie": 1, "win": 2, "loss": 3, "dnf": 4, "didnotfinish": 4}
        outcome = outcome_map.get(outcome.lower())
    elif isinstance(outcome, int):
        pass
    else:
        outcome = None
    
    team_id = me.get("LastTeamId")
    if team_id is None:
        team_stats = me.get("PlayerTeamStats")
        if isinstance(team_stats, list) and team_stats:
            team_id = team_stats[0].get("TeamId")
    
    return outcome, team_id


def _extract_team_scores(match_obj: dict, my_team_id: int | None) -> tuple[int | None, int | None]:
    """Extrait les scores d'équipe."""
    teams = match_obj.get("Teams")
    if not isinstance(teams, list) or my_team_id is None:
        return None, None
    
    my_score = None
    enemy_score = None
    
    for t in teams:
        if not isinstance(t, dict):
            continue
        tid = t.get("TeamId")
        stats = t.get("Stats", {})
        core = stats.get("CoreStats", {})
        score = core.get("Score") or core.get("RoundScore") or core.get("PersonalScore")
        
        if tid == my_team_id:
            my_score = coerce_number(score, int, None)
        else:
            # Prend le premier adversaire trouvé
            if enemy_score is None:
                enemy_score = coerce_number(score, int, None)
    
    return my_score, enemy_score


def _is_firefight(playlist_name: str | None, pair_name: str | None) -> bool:
    """Détermine si le match est du Firefight (PvE)."""
    for text in [playlist_name, pair_name]:
        if not text:
            continue
        text_lower = text.lower()
        for pattern in FIREFIGHT_PATTERNS:
            if pattern in text_lower:
                return True
    return False


# =============================================================================
# Logique de session améliorée
# =============================================================================

@dataclass
class MatchForSession:
    """Données minimales pour le calcul de session."""
    match_id: str
    start_time: datetime
    teammates_signature: str  # XUIDs triés, séparés par virgule


def compute_sessions_with_teammates(
    matches: list[MatchForSession],
    gap_minutes: int = SESSION_GAP_MINUTES,
    cutoff_hour: int = SESSION_CUTOFF_HOUR,
) -> dict[str, tuple[int, str]]:
    """Calcule les sessions avec la logique améliorée.
    
    Règles :
    1. Un gap > gap_minutes entre deux matchs = nouvelle session
    2. Changement de coéquipiers selon should_start_new_session_on_teammate_change()
    3. Les matchs avant cutoff_hour ou à J-1 sont considérés comme sessions terminées
    
    Args:
        matches: Liste de matchs triés par start_time croissant.
        gap_minutes: Gap maximum entre matchs d'une même session.
        cutoff_hour: Heure de coupure (avant cette heure = session terminée).
    
    Returns:
        Dict {match_id: (session_id, session_label)}
    """
    if not matches:
        return {}
    
    matches = sorted(matches, key=lambda m: m.start_time)
    result: dict[str, tuple[int, str]] = {}
    
    session_id = 0
    session_matches: list[MatchForSession] = []
    
    now = datetime.now(timezone.utc)
    today_start = datetime.combine(now.date(), time(cutoff_hour, 0), tzinfo=timezone.utc)
    
    def _finalize_session(s_matches: list[MatchForSession], s_id: int) -> None:
        """Génère le label et enregistre la session."""
        if not s_matches:
            return
        
        start_dt = s_matches[0].start_time
        end_dt = s_matches[-1].start_time
        count = len(s_matches)
        
        # Label: "25/01/26 14:30–16:45 (5)"
        label = f"{start_dt:%d/%m/%y %H:%M}–{end_dt:%H:%M} ({count})"
        
        for m in s_matches:
            result[m.match_id] = (s_id, label)
    
    def _parse_teammates(sig: str) -> set[str]:
        """Convertit la signature en set de XUIDs."""
        if not sig:
            return set()
        return set(sig.split(","))
    
    prev_match: MatchForSession | None = None
    prev_teammates: set[str] = set()
    
    for match in matches:
        new_session = False
        curr_teammates = _parse_teammates(match.teammates_signature)
        
        if prev_match is None:
            # Premier match
            new_session = True
        else:
            # Vérifier le gap temporel
            gap = (match.start_time - prev_match.start_time).total_seconds() / 60.0
            if gap > gap_minutes:
                new_session = True
            
            # Vérifier le changement de coéquipiers avec la règle configurable
            elif should_start_new_session_on_teammate_change(prev_teammates, curr_teammates):
                new_session = True
        
        if new_session and session_matches:
            _finalize_session(session_matches, session_id)
            session_id += 1
            session_matches = []
        
        session_matches.append(match)
        prev_match = match
        prev_teammates = curr_teammates
    
    # Finaliser la dernière session
    if session_matches:
        _finalize_session(session_matches, session_id)
    
    return result


# =============================================================================
# Calcul du score de performance par match
# =============================================================================

def compute_match_performance_score(
    kills: int,
    deaths: int,
    assists: int,
    accuracy: float | None,
    kda: float | None,
    outcome: int | None,
) -> float | None:
    """Calcule un score de performance simplifié pour un match unique.
    
    Score 0-100 basé sur:
    - K/D ratio (40%)
    - Outcome (30%)
    - Accuracy (30%)
    """
    components = []
    weights = []
    
    # K/D ratio → score 0-100 (1.0 = 50, 2.0 = 100)
    if deaths > 0:
        kd = kills / deaths
    elif kills > 0:
        kd = float(kills)
    else:
        kd = None
    
    if kd is not None:
        kd_score = min(100.0, kd * 50.0)
        components.append(kd_score)
        weights.append(0.40)
    
    # Outcome → score
    if outcome is not None:
        outcome_scores = {1: 50.0, 2: 100.0, 3: 25.0, 4: 10.0}  # Tie, Win, Loss, DNF
        out_score = outcome_scores.get(outcome, 50.0)
        components.append(out_score)
        weights.append(0.30)
    
    # Accuracy → score (50% = 50 pts, 100% = 100 pts)
    if accuracy is not None and accuracy > 0:
        acc_score = min(100.0, accuracy)
        components.append(acc_score)
        weights.append(0.30)
    
    if not components:
        return None
    
    # Moyenne pondérée normalisée
    total_weight = sum(weights)
    score = sum(c * w for c, w in zip(components, weights)) / total_weight
    
    return round(score, 1)


# =============================================================================
# Migration principale
# =============================================================================

def create_cache_tables(con: sqlite3.Connection) -> None:
    """Crée les tables de cache."""
    cur = con.cursor()
    for ddl in get_all_cache_table_ddl():
        try:
            cur.execute(ddl)
        except sqlite3.OperationalError as e:
            # Index déjà existant, etc.
            if "already exists" not in str(e).lower():
                raise
    
    # Enregistrer la version du schéma
    cur.execute(
        """
        INSERT INTO CacheMeta (key, value, updated_at)
        VALUES ('schema_version', ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = datetime('now')
        """,
        (CACHE_SCHEMA_VERSION, CACHE_SCHEMA_VERSION),
    )
    con.commit()
    print(f"✓ Tables de cache créées (version {CACHE_SCHEMA_VERSION})")


def drop_cache_tables(con: sqlite3.Connection) -> None:
    """Supprime les tables de cache (pour --force)."""
    cur = con.cursor()
    for table in get_cache_table_names():
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    con.commit()
    print("✓ Tables de cache supprimées")


def migrate_match_cache(
    con: sqlite3.Connection,
    xuid: str,
    dry_run: bool = False,
    friends_xuids: set[str] | None = None,
) -> list[MatchForSession]:
    """Peuple MatchCache depuis MatchStats.
    
    Args:
        con: Connexion SQLite.
        xuid: XUID du joueur principal.
        dry_run: Si True, n'effectue pas les modifications.
        friends_xuids: Set des XUIDs des amis connus (pour is_with_friends).
    
    Returns:
        Liste des matchs pour le calcul de session.
    """
    cur = con.cursor()
    
    # Si pas d'amis fournis, charger depuis la DB
    if friends_xuids is None:
        friends_xuids = load_friends_xuids_from_db(con, xuid)
        # Fallback sur les amis prédéfinis si DB vide
        if not friends_xuids:
            friends_xuids = FRIENDS_XUIDS
    
    if friends_xuids:
        print(f"  → {len(friends_xuids)} amis connus pour le calcul is_with_friends")
    
    # Charger les noms d'assets
    asset_names: dict[str, dict[str, str]] = {}
    for table in ["Maps", "Playlists", "PlaylistMapModePairs", "GameVariants"]:
        asset_names[table] = {}
        try:
            cur.execute(f"SELECT json_extract(ResponseBody, '$.AssetId'), json_extract(ResponseBody, '$.PublicName') FROM {table}")
            for (asset_id, name) in cur.fetchall():
                if asset_id and name:
                    asset_names[table][asset_id] = name
        except sqlite3.OperationalError:
            pass
    
    # Charger tous les matchs depuis MatchStats
    cur.execute("SELECT ResponseBody FROM MatchStats")
    rows = cur.fetchall()
    print(f"  → {len(rows)} matchs à traiter...")
    
    matches_for_session: list[MatchForSession] = []
    cache_rows: list[tuple] = []
    
    for (body,) in rows:
        try:
            obj = json.loads(body)
        except Exception:
            continue
        
        match_id = obj.get("MatchId")
        if not isinstance(match_id, str):
            continue
        
        match_info = obj.get("MatchInfo", {})
        start_time_raw = match_info.get("StartTime")
        if not isinstance(start_time_raw, str):
            continue
        
        start_time = parse_iso_utc(start_time_raw)
        
        # Extraire les IDs
        playlist_id = None
        playlist_obj = match_info.get("Playlist")
        if isinstance(playlist_obj, dict):
            playlist_id = playlist_obj.get("AssetId")
        
        map_id = None
        map_variant = match_info.get("MapVariant")
        if isinstance(map_variant, dict):
            map_id = map_variant.get("AssetId")
        
        pair_id = None
        pair_obj = match_info.get("PlaylistMapModePair")
        if isinstance(pair_obj, dict):
            pair_id = pair_obj.get("AssetId")
        
        game_variant_id = None
        ugc_variant = match_info.get("UgcGameVariant")
        if isinstance(ugc_variant, dict):
            game_variant_id = ugc_variant.get("AssetId")
        
        # Trouver le joueur
        players = obj.get("Players", [])
        me = _find_player_in_match(players, xuid)
        if me is None:
            continue
        
        # Extraire les stats
        stats = _extract_core_stats(me)
        outcome, team_id = _extract_outcome_team(me)
        my_score, enemy_score = _extract_team_scores(obj, team_id)
        
        # Coéquipiers
        teammates = _get_team_xuids(players, team_id, xuid) if team_id is not None else []
        teammates_sig = ",".join(teammates)
        
        # Noms
        playlist_name = asset_names["Playlists"].get(playlist_id) or playlist_id
        map_name = asset_names["Maps"].get(map_id) or map_id
        pair_name = asset_names["PlaylistMapModePairs"].get(pair_id) or pair_id
        game_variant_name = asset_names["GameVariants"].get(game_variant_id) or game_variant_id
        
        # Firefight
        is_ff = 1 if _is_firefight(playlist_name, pair_name) else 0
        
        # Score de performance
        perf_score = compute_match_performance_score(
            stats["kills"],
            stats["deaths"],
            stats["assists"],
            stats["accuracy"],
            stats["kda"],
            outcome,
        )
        
        # Calculer le nombre de coéquipiers connus (amis)
        known_teammates_count, is_with_friends, friends_xuids_str = compute_known_teammates(
            teammates_sig, friends_xuids
        )
        
        # Préparer la ligne pour MatchCache
        cache_rows.append((
            match_id,
            xuid,
            start_time.isoformat(),
            playlist_id,
            playlist_name,
            map_id,
            map_name,
            pair_id,
            pair_name,
            game_variant_id,
            game_variant_name,
            outcome,
            team_id,
            stats["kills"],
            stats["deaths"],
            stats["assists"],
            stats["accuracy"],
            stats["kda"],
            stats["time_played_seconds"],
            stats["average_life_seconds"],
            stats["max_killing_spree"],
            stats["headshot_kills"],
            my_score,
            enemy_score,
            None,  # team_mmr (à extraire de PlayerMatchStats si besoin)
            None,  # enemy_mmr
            None,  # session_id (calculé après)
            None,  # session_label
            perf_score,
            is_ff,
            teammates_sig,
            known_teammates_count,
            1 if is_with_friends else 0,
            friends_xuids_str,
        ))
        
        matches_for_session.append(MatchForSession(
            match_id=match_id,
            start_time=start_time,
            teammates_signature=teammates_sig,
        ))
    
    if dry_run:
        print(f"  [DRY-RUN] {len(cache_rows)} lignes à insérer dans MatchCache")
        return matches_for_session
    
    # Insérer dans MatchCache
    cur.executemany(
        """
        INSERT OR REPLACE INTO MatchCache (
            match_id, xuid, start_time, playlist_id, playlist_name,
            map_id, map_name, pair_id, pair_name, game_variant_id, game_variant_name,
            outcome, last_team_id, kills, deaths, assists, accuracy, kda,
            time_played_seconds, average_life_seconds, max_killing_spree, headshot_kills,
            my_team_score, enemy_team_score, team_mmr, enemy_mmr,
            session_id, session_label, performance_score, is_firefight, teammates_signature,
            known_teammates_count, is_with_friends, friends_xuids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        cache_rows,
    )
    con.commit()
    
    # Compter les matchs avec amis
    with_friends_count = sum(1 for row in cache_rows if row[-1] == 1)
    print(f"✓ MatchCache peuplé ({len(cache_rows)} matchs, {with_friends_count} avec amis)")
    
    return matches_for_session


def update_sessions(
    con: sqlite3.Connection,
    matches: list[MatchForSession],
    dry_run: bool = False,
) -> None:
    """Met à jour les session_id et session_label dans MatchCache."""
    if not matches:
        return
    
    sessions = compute_sessions_with_teammates(matches)
    
    if dry_run:
        print(f"  [DRY-RUN] {len(set(s[0] for s in sessions.values()))} sessions détectées")
        return
    
    cur = con.cursor()
    for match_id, (session_id, session_label) in sessions.items():
        cur.execute(
            "UPDATE MatchCache SET session_id = ?, session_label = ? WHERE match_id = ?",
            (session_id, session_label, match_id),
        )
    
    con.commit()
    n_sessions = len(set(s[0] for s in sessions.values()))
    print(f"✓ Sessions calculées ({n_sessions} sessions)")


def migrate_teammates_aggregate(
    con: sqlite3.Connection,
    xuid: str,
    dry_run: bool = False,
    matches_data: list | None = None,
) -> None:
    """Calcule et stocke les statistiques de coéquipiers."""
    cur = con.cursor()
    
    # En dry-run, on utilise les données passées directement
    if dry_run:
        if matches_data:
            # Compter les coéquipiers uniques depuis les données de migration
            teammates_set = set()
            for m in matches_data:
                if m.teammates_signature:
                    for tm in m.teammates_signature.split(","):
                        if tm:
                            teammates_set.add(tm)
            print(f"  [DRY-RUN] {len(teammates_set)} coéquipiers à agréger")
        else:
            print("  [DRY-RUN] Coéquipiers seront agrégés depuis MatchCache")
        return
    
    # Récupérer tous les matchs avec leurs coéquipiers
    cur.execute(
        """
        SELECT match_id, teammates_signature, outcome, start_time
        FROM MatchCache
        WHERE xuid = ? AND teammates_signature IS NOT NULL AND teammates_signature != ''
        ORDER BY start_time
        """,
        (xuid,),
    )
    
    # Agréger par coéquipier
    teammate_stats: dict[str, dict] = {}
    
    for match_id, teammates_sig, outcome, start_time in cur.fetchall():
        for tm_xuid in teammates_sig.split(","):
            if not tm_xuid:
                continue
            
            if tm_xuid not in teammate_stats:
                teammate_stats[tm_xuid] = {
                    "matches_together": 0,
                    "same_team_count": 0,
                    "wins_together": 0,
                    "losses_together": 0,
                    "first_played": start_time,
                    "last_played": start_time,
                }
            
            stats = teammate_stats[tm_xuid]
            stats["matches_together"] += 1
            stats["same_team_count"] += 1  # Par définition, ils sont dans la même équipe
            stats["last_played"] = start_time
            
            if outcome == 2:  # Win
                stats["wins_together"] += 1
            elif outcome == 3:  # Loss
                stats["losses_together"] += 1
    
    if dry_run:
        print(f"  [DRY-RUN] {len(teammate_stats)} coéquipiers à agréger")
        return
    
    # Insérer dans TeammatesAggregate
    for tm_xuid, stats in teammate_stats.items():
        cur.execute(
            """
            INSERT OR REPLACE INTO TeammatesAggregate (
                xuid, teammate_xuid, teammate_gamertag,
                matches_together, same_team_count, opposite_team_count,
                wins_together, losses_together, first_played, last_played, computed_at
            ) VALUES (?, ?, NULL, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
            """,
            (
                xuid,
                tm_xuid,
                stats["matches_together"],
                stats["same_team_count"],
                stats["wins_together"],
                stats["losses_together"],
                stats["first_played"],
                stats["last_played"],
            ),
        )
    
    con.commit()
    print(f"✓ TeammatesAggregate peuplé ({len(teammate_stats)} coéquipiers)")
    
    # Enrichir les gamertags depuis les données brutes
    _enrich_teammate_gamertags(con, xuid)


def _build_xuid_gamertag_mapping(con: sqlite3.Connection) -> dict[str, str]:
    """Construit un mapping XUID → Gamertag depuis les sources disponibles."""
    mapping: dict[str, str] = {}
    
    # 1. D'abord charger depuis xuid_aliases.json si disponible
    aliases_path = Path(__file__).parent.parent / "xuid_aliases.json"
    if aliases_path.exists():
        try:
            with open(aliases_path, encoding="utf-8") as f:
                aliases = json.load(f)
                mapping.update(aliases)
                print(f"  → {len(aliases)} gamertags chargés depuis xuid_aliases.json")
        except Exception as e:
            print(f"  ⚠ Erreur lecture xuid_aliases.json: {e}")
    
    # 2. Compléter depuis les fichiers de cache profile_api
    cache_dir = Path(__file__).parent.parent / "data" / "cache" / "profile_api"
    if cache_dir.exists():
        for f in cache_dir.glob("xuid_gt_*.json"):
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                    xuid = data.get("xuid")
                    gamertag = data.get("gamertag")
                    if xuid and gamertag:
                        mapping[xuid] = gamertag
            except Exception:
                pass
    
    # 3. Compléter depuis MatchStats si des joueurs manquent encore
    cur = con.cursor()
    cur.execute("SELECT ResponseBody FROM MatchStats")
    for (body,) in cur.fetchall():
        try:
            obj = json.loads(body)
            players = obj.get("Players", [])
            for p in players:
                if not isinstance(p, dict):
                    continue
                
                # Extraire gamertag s'il est présent
                gamertag = p.get("PlayerInfo", {}).get("Gamertag") if p.get("PlayerInfo") else None
                if not gamertag:
                    continue
                
                # Extraire XUID
                pid = p.get("PlayerId")
                xuid_str = None
                if isinstance(pid, str):
                    m = re.match(r"xuid\((\d+)\)", pid, re.IGNORECASE)
                    if m:
                        xuid_str = m.group(1)
                    elif pid.isdigit():
                        xuid_str = pid
                elif isinstance(pid, dict):
                    xuid_str = str(pid.get("Xuid") or pid.get("xuid") or "")
                
                if xuid_str and not xuid_str.startswith("bid(") and xuid_str not in mapping:
                    mapping[xuid_str] = gamertag
        except Exception:
            continue
    
    return mapping


def _enrich_teammate_gamertags(con: sqlite3.Connection, xuid: str) -> None:
    """Met à jour les gamertags dans TeammatesAggregate."""
    cur = con.cursor()
    
    # Construire le mapping
    mapping = _build_xuid_gamertag_mapping(con)
    
    # Mettre à jour les gamertags
    updated = 0
    for tm_xuid, gamertag in mapping.items():
        cur.execute(
            """
            UPDATE TeammatesAggregate 
            SET teammate_gamertag = ? 
            WHERE xuid = ? AND teammate_xuid = ? AND teammate_gamertag IS NULL
            """,
            (gamertag, xuid, tm_xuid),
        )
        updated += cur.rowcount
    
    con.commit()
    if updated > 0:
        print(f"  → {updated} gamertags enrichis")


def resolve_gamertag_to_xuid(
    con: sqlite3.Connection,
    gamertag: str,
    matches_data: list[MatchForSession] | None = None,
    xuid_mapping: dict[str, str] | None = None,
) -> str | None:
    """Tente de résoudre un gamertag en XUID depuis les données disponibles."""
    gamertag_lower = gamertag.lower()
    
    # 1. Si on a un mapping (xuid → gamertag), l'inverser pour chercher
    if xuid_mapping:
        for xuid, gt in xuid_mapping.items():
            if gt.lower() == gamertag_lower:
                return xuid
    
    # 2. Charger depuis xuid_aliases.json
    aliases_path = Path(__file__).parent.parent / "xuid_aliases.json"
    if aliases_path.exists():
        try:
            with open(aliases_path, encoding="utf-8") as f:
                aliases = json.load(f)
                for xuid, gt in aliases.items():
                    if gt.lower() == gamertag_lower:
                        return xuid
        except Exception:
            pass
    
    # 3. Chercher dans XuidAliases si la table existe
    cur = con.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='XuidAliases'"
    )
    if cur.fetchone():
        cur.execute(
            "SELECT Xuid FROM XuidAliases WHERE Gamertag = ? COLLATE NOCASE LIMIT 1",
            (gamertag,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
    
    return None


def insert_predefined_friends(
    con: sqlite3.Connection,
    owner_xuid: str,
    dry_run: bool = False,
    matches_data: list[MatchForSession] | None = None,
    xuid_mapping: dict[str, str] | None = None,
) -> None:
    """Insère les amis prédéfinis dans la table Friends."""
    cur = con.cursor()
    
    for friend in PREDEFINED_FRIENDS:
        gamertag = friend["gamertag"]
        nickname = friend.get("nickname")
        
        # Essayer de résoudre le XUID
        friend_xuid = resolve_gamertag_to_xuid(con, gamertag, matches_data)
        
        if dry_run:
            print(f"  [DRY-RUN] Ami: {gamertag} → XUID={friend_xuid or '(non résolu)'}")
            continue
        
        if friend_xuid:
            cur.execute(
                """
                INSERT OR REPLACE INTO Friends (owner_xuid, friend_xuid, friend_gamertag, nickname, added_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (owner_xuid, friend_xuid, gamertag, nickname),
            )
        else:
            # Insérer avec XUID placeholder (à résoudre plus tard)
            cur.execute(
                """
                INSERT OR IGNORE INTO Friends (owner_xuid, friend_xuid, friend_gamertag, nickname, added_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (owner_xuid, f"unknown:{gamertag}", gamertag, nickname),
            )
            print(f"  ⚠ XUID non résolu pour {gamertag}, à compléter manuellement")
    
    if not dry_run:
        con.commit()
        print(f"✓ {len(PREDEFINED_FRIENDS)} amis ajoutés")


def update_is_with_friends(
    con: sqlite3.Connection,
    xuid: str,
    dry_run: bool = False,
) -> None:
    """Met à jour les colonnes known_teammates_count et is_with_friends pour les matchs existants.
    
    Cette fonction est utile pour :
    1. Mettre à jour les matchs après ajout de nouveaux amis
    2. Recalculer après migration depuis une ancienne version du schéma
    
    Args:
        con: Connexion SQLite.
        xuid: XUID du joueur principal.
        dry_run: Si True, n'effectue pas les modifications.
    """
    cur = con.cursor()
    
    # Vérifier que la table MatchCache existe
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='MatchCache'"
    )
    if not cur.fetchone():
        print("  ❌ Table MatchCache introuvable. Exécutez d'abord la migration complète :")
        print(f"     python scripts/migrate_to_cache.py --db <path> --xuid {xuid}")
        return
    
    # Vérifier si les colonnes existent, sinon les ajouter
    cur.execute("PRAGMA table_info(MatchCache)")
    columns = {row[1] for row in cur.fetchall()}
    
    if "known_teammates_count" not in columns:
        print("  → Ajout colonne known_teammates_count...")
        if not dry_run:
            cur.execute("ALTER TABLE MatchCache ADD COLUMN known_teammates_count INTEGER NOT NULL DEFAULT 0")
    
    if "is_with_friends" not in columns:
        print("  → Ajout colonne is_with_friends...")
        if not dry_run:
            cur.execute("ALTER TABLE MatchCache ADD COLUMN is_with_friends INTEGER NOT NULL DEFAULT 0")
    
    if "friends_xuids" not in columns:
        print("  → Ajout colonne friends_xuids...")
        if not dry_run:
            cur.execute("ALTER TABLE MatchCache ADD COLUMN friends_xuids TEXT DEFAULT ''")
    
    if not dry_run:
        con.commit()
    
    # Charger les amis depuis la DB
    friends_xuids = load_friends_xuids_from_db(con, xuid)
    if not friends_xuids:
        friends_xuids = FRIENDS_XUIDS
    
    if not friends_xuids:
        print("  ⚠ Aucun ami trouvé, is_with_friends sera 0 pour tous les matchs")
        return
    
    print(f"  → {len(friends_xuids)} amis connus")
    
    # Charger tous les matchs avec leurs teammates_signature
    cur.execute(
        """
        SELECT match_id, teammates_signature 
        FROM MatchCache 
        WHERE xuid = ?
        """,
        (xuid,),
    )
    
    updates: list[tuple[int, int, str, str]] = []
    with_friends_count = 0
    
    for (match_id, teammates_sig) in cur.fetchall():
        count, is_with, friends_str = compute_known_teammates(teammates_sig or "", friends_xuids)
        updates.append((count, 1 if is_with else 0, friends_str, match_id))
        if is_with:
            with_friends_count += 1
    
    if dry_run:
        print(f"  [DRY-RUN] {len(updates)} matchs à mettre à jour ({with_friends_count} avec amis)")
        return
    
    cur.executemany(
        """
        UPDATE MatchCache 
        SET known_teammates_count = ?, is_with_friends = ?, friends_xuids = ?
        WHERE match_id = ?
        """,
        updates,
    )
    con.commit()
    print(f"✓ {len(updates)} matchs mis à jour ({with_friends_count} avec amis)")


def run_migration(
    db_path: str,
    xuid: str,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Exécute la migration complète."""
    print(f"\n{'='*60}")
    print("Migration vers les tables de cache optimisées")
    print(f"{'='*60}")
    print(f"DB: {db_path}")
    print(f"XUID: {xuid}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'RÉEL'}")
    print(f"{'='*60}\n")
    
    with get_connection(db_path) as con:
        # Étape 1: Créer/réinitialiser les tables
        if force:
            print("[1/6] Suppression des tables existantes...")
            if not dry_run:
                drop_cache_tables(con)
        
        print("[1/6] Création des tables de cache...")
        if not dry_run:
            create_cache_tables(con)
        
        # Étape 2: Insérer les amis AVANT le calcul de is_with_friends
        print("\n[2/6] Insertion des amis prédéfinis...")
        # On fait une première passe pour insérer les amis (sans matches_data)
        insert_predefined_friends(con, xuid, dry_run, matches_data=None)
        
        # Étape 3: Peupler MatchCache (avec is_with_friends correct)
        print("\n[3/6] Migration de MatchStats → MatchCache...")
        matches = migrate_match_cache(con, xuid, dry_run)
        
        # Étape 4: Calculer les sessions
        print("\n[4/6] Calcul des sessions...")
        update_sessions(con, matches, dry_run)
        
        # Étape 5: Agréger les coéquipiers
        print("\n[5/6] Agrégation des statistiques de coéquipiers...")
        migrate_teammates_aggregate(con, xuid, dry_run, matches_data=matches)
        
        # Étape 6: Mettre à jour les amis avec les XUID résolus + recalculer is_with_friends
        print("\n[6/6] Finalisation des amis et is_with_friends...")
        insert_predefined_friends(con, xuid, dry_run, matches_data=matches)
        # Recalculer is_with_friends maintenant que les amis ont leurs vrais XUIDs
        update_is_with_friends(con, xuid, dry_run)
    
    print(f"\n{'='*60}")
    print("✓ Migration terminée avec succès !" if not dry_run else "✓ Dry-run terminé")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Migre les données vers les tables de cache optimisées.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Chemin vers la DB SQLite (défaut: auto-détecté)",
    )
    parser.add_argument(
        "--xuid",
        type=str,
        default=None,
        help="XUID du joueur principal (défaut: depuis OPENSPARTAN_DEFAULT_XUID)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Supprime et recrée les tables de cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait fait sans modifier la DB",
    )
    parser.add_argument(
        "--update-friends",
        action="store_true",
        help="Met à jour uniquement is_with_friends (après ajout d'amis)",
    )
    
    args = parser.parse_args()
    
    # Résoudre le chemin de la DB
    db_path = args.db or get_default_db_path()
    if not db_path or not os.path.exists(db_path):
        print(f"❌ DB non trouvée: {db_path or '(aucun chemin)'}")
        print("   Spécifiez --db ou définissez OPENSPARTAN_DB")
        sys.exit(1)
    
    # Résoudre le XUID
    xuid = args.xuid or DEFAULT_PLAYER_XUID
    if not xuid:
        print("❌ XUID non spécifié")
        print("   Spécifiez --xuid ou définissez OPENSPARTAN_DEFAULT_XUID")
        sys.exit(1)
    
    # Mode spécial: mise à jour is_with_friends uniquement
    if args.update_friends:
        print(f"\n{'='*60}")
        print("Mise à jour is_with_friends")
        print(f"{'='*60}")
        print(f"DB: {db_path}")
        print(f"XUID: {xuid}")
        print(f"Mode: {'DRY-RUN' if args.dry_run else 'RÉEL'}")
        print(f"{'='*60}\n")
        
        with get_connection(db_path) as con:
            update_is_with_friends(con, xuid, dry_run=args.dry_run)
        
        print(f"\n✓ Terminé !")
    else:
        run_migration(db_path, xuid, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
