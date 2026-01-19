"""Module de gestion de la base de données."""

from src.db.connection import get_connection, DatabaseConnection
from src.db.loaders import (
    load_matches,
    load_asset_name_map,
    load_player_match_result,
    load_top_medals,
    load_match_medals_for_player,
    query_matches_with_friend,
    list_other_player_xuids,
    list_top_teammates,
)
from src.db.parsers import (
    guess_xuid_from_db_path,
    parse_iso_utc,
)

__all__ = [
    "get_connection",
    "DatabaseConnection",
    "load_matches",
    "load_asset_name_map",
    "load_player_match_result",
    "load_top_medals",
    "load_match_medals_for_player",
    "query_matches_with_friend",
    "list_other_player_xuids",
    "list_top_teammates",
    "guess_xuid_from_db_path",
    "parse_iso_utc",
]
