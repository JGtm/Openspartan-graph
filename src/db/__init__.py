"""Module de gestion de la base de données."""

from src.db.connection import get_connection, DatabaseConnection
from src.db.loaders import (
    load_matches,
    load_asset_name_map,
    load_player_match_result,
    load_top_medals,
    load_match_medals_for_player,
    has_table,
    load_highlight_events_for_match,
    query_matches_with_friend,
    list_other_player_xuids,
    list_top_teammates,
)
from src.db.parsers import (
    guess_xuid_from_db_path,
    parse_iso_utc,
    resolve_xuid_from_db,
)
from src.db.profiles import (
    PROFILES_PATH,
    load_profiles,
    save_profiles,
    list_local_dbs,
)

__all__ = [
    # connection
    "get_connection",
    "DatabaseConnection",
    # loaders
    "load_matches",
    "load_asset_name_map",
    "load_player_match_result",
    "load_top_medals",
    "load_match_medals_for_player",
    "has_table",
    "load_highlight_events_for_match",
    "query_matches_with_friend",
    "list_other_player_xuids",
    "list_top_teammates",
    # parsers
    "guess_xuid_from_db_path",
    "parse_iso_utc",
    "resolve_xuid_from_db",
    # profiles
    "PROFILES_PATH",
    "load_profiles",
    "save_profiles",
    "list_local_dbs",
]
