"""Schéma des tables de cache et d'optimisation.

Ce module définit les tables supplémentaires pour :
- Éviter le parsing JSON répétitif (MatchCache)
- Pré-calculer les sessions (colonnes dans MatchCache)
- Stocker les scores de performance
- Gérer les amis et coéquipiers

Convention: les tables originales (MatchStats, PlayerMatchStats, etc.)
restent la "source de vérité". Les tables de cache sont dérivées.
"""

from __future__ import annotations

# =============================================================================
# Table MatchCache - Données de match dé-normalisées
# =============================================================================

CREATE_MATCH_CACHE = """
CREATE TABLE IF NOT EXISTS MatchCache (
    match_id TEXT PRIMARY KEY,
    xuid TEXT NOT NULL,
    start_time TEXT NOT NULL,
    playlist_id TEXT,
    playlist_name TEXT,
    map_id TEXT,
    map_name TEXT,
    pair_id TEXT,
    pair_name TEXT,
    game_variant_id TEXT,
    game_variant_name TEXT,
    outcome INTEGER,
    last_team_id INTEGER,
    kills INTEGER NOT NULL DEFAULT 0,
    deaths INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    accuracy REAL,
    kda REAL,
    time_played_seconds REAL,
    average_life_seconds REAL,
    max_killing_spree INTEGER,
    headshot_kills INTEGER,
    my_team_score INTEGER,
    enemy_team_score INTEGER,
    team_mmr REAL,
    enemy_mmr REAL,
    -- Colonnes calculées
    session_id INTEGER,
    session_label TEXT,
    performance_score REAL,
    is_firefight INTEGER NOT NULL DEFAULT 0,
    -- Coéquipiers (XUIDs séparés par virgule, triés)
    teammates_signature TEXT,
    -- Amis détectés dans l'équipe
    known_teammates_count INTEGER NOT NULL DEFAULT 0,
    is_with_friends INTEGER NOT NULL DEFAULT 0,
    friends_xuids TEXT DEFAULT '',
    -- Métadonnées
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
"""

CREATE_MATCH_CACHE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_xuid ON MatchCache(xuid)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_start_time ON MatchCache(start_time)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_session ON MatchCache(xuid, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_playlist ON MatchCache(xuid, playlist_id)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_map ON MatchCache(xuid, map_id)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_teammates ON MatchCache(xuid, teammates_signature)",
    "CREATE INDEX IF NOT EXISTS idx_MatchCache_friends ON MatchCache(xuid, is_with_friends)",
]


# =============================================================================
# Table Friends - Liste des amis (manuelle)
# =============================================================================

CREATE_FRIENDS = """
CREATE TABLE IF NOT EXISTS Friends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_xuid TEXT NOT NULL,
    friend_xuid TEXT NOT NULL,
    friend_gamertag TEXT,
    nickname TEXT,
    added_at TEXT DEFAULT (datetime('now')),
    UNIQUE(owner_xuid, friend_xuid)
)
"""

CREATE_FRIENDS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_Friends_owner ON Friends(owner_xuid)",
]


# =============================================================================
# Table PerformanceScores - Scores pré-calculés
# =============================================================================

CREATE_PERFORMANCE_SCORES = """
CREATE TABLE IF NOT EXISTS PerformanceScores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    xuid TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT,
    score_version TEXT NOT NULL DEFAULT 'v2',
    score REAL,
    components TEXT,
    confidence REAL,
    match_count INTEGER,
    computed_at TEXT DEFAULT (datetime('now')),
    UNIQUE(xuid, scope_type, scope_id, score_version)
)
"""

CREATE_PERFORMANCE_SCORES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_PerfScores_xuid_scope ON PerformanceScores(xuid, scope_type)",
    "CREATE INDEX IF NOT EXISTS idx_PerfScores_session ON PerformanceScores(xuid, scope_id) WHERE scope_type = 'session'",
]


# =============================================================================
# Table TeammatesAggregate - Stats coéquipiers agrégées
# =============================================================================

CREATE_TEAMMATES_AGGREGATE = """
CREATE TABLE IF NOT EXISTS TeammatesAggregate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    xuid TEXT NOT NULL,
    teammate_xuid TEXT NOT NULL,
    teammate_gamertag TEXT,
    matches_together INTEGER NOT NULL DEFAULT 0,
    same_team_count INTEGER NOT NULL DEFAULT 0,
    opposite_team_count INTEGER NOT NULL DEFAULT 0,
    wins_together INTEGER NOT NULL DEFAULT 0,
    losses_together INTEGER NOT NULL DEFAULT 0,
    first_played TEXT,
    last_played TEXT,
    computed_at TEXT DEFAULT (datetime('now')),
    UNIQUE(xuid, teammate_xuid)
)
"""

CREATE_TEAMMATES_AGGREGATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_TeammatesAgg_xuid ON TeammatesAggregate(xuid)",
    "CREATE INDEX IF NOT EXISTS idx_TeammatesAgg_same_team ON TeammatesAggregate(xuid, same_team_count DESC)",
]


# =============================================================================
# Table MedalsAggregate - Totaux de médailles pré-calculés
# =============================================================================

CREATE_MEDALS_AGGREGATE = """
CREATE TABLE IF NOT EXISTS MedalsAggregate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    xuid TEXT NOT NULL,
    scope_type TEXT NOT NULL DEFAULT 'global',
    scope_id TEXT,
    medal_id INTEGER NOT NULL,
    total_count INTEGER NOT NULL DEFAULT 0,
    computed_at TEXT DEFAULT (datetime('now')),
    UNIQUE(xuid, scope_type, scope_id, medal_id)
)
"""

CREATE_MEDALS_AGGREGATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_MedalsAgg_xuid ON MedalsAggregate(xuid, scope_type)",
]


# =============================================================================
# Table CacheMeta - Métadonnées de cache (version, timestamps)
# =============================================================================

CREATE_CACHE_META = """
CREATE TABLE IF NOT EXISTS CacheMeta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
)
"""

# Version du schéma de cache (à incrémenter si changement majeur)
CACHE_SCHEMA_VERSION = "1.1"


# =============================================================================
# Helpers
# =============================================================================

def get_all_cache_table_ddl() -> list[str]:
    """Retourne toutes les instructions DDL pour créer les tables de cache."""
    ddl = [
        CREATE_MATCH_CACHE,
        CREATE_FRIENDS,
        CREATE_PERFORMANCE_SCORES,
        CREATE_TEAMMATES_AGGREGATE,
        CREATE_MEDALS_AGGREGATE,
        CREATE_CACHE_META,
    ]
    ddl.extend(CREATE_MATCH_CACHE_INDEXES)
    ddl.extend(CREATE_FRIENDS_INDEXES)
    ddl.extend(CREATE_PERFORMANCE_SCORES_INDEXES)
    ddl.extend(CREATE_TEAMMATES_AGGREGATE_INDEXES)
    ddl.extend(CREATE_MEDALS_AGGREGATE_INDEXES)
    return ddl


def get_cache_table_names() -> list[str]:
    """Retourne les noms des tables de cache."""
    return [
        "MatchCache",
        "Friends",
        "PerformanceScores",
        "TeammatesAggregate",
        "MedalsAggregate",
        "CacheMeta",
    ]
