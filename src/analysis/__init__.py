"""Module d'analyse des données."""

from src.analysis.stats import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
)
from src.analysis.sessions import compute_sessions
from src.analysis.maps import compute_map_breakdown
from src.analysis.filters import (
    mark_firefight,
    is_allowed_playlist_name,
    build_option_map,
    build_xuid_option_map,
)
from src.analysis.killer_victim import (
    KVPair,
    compute_killer_victim_pairs,
    killer_victim_counts_long,
    killer_victim_matrix,
)

__all__ = [
    "compute_aggregated_stats",
    "compute_outcome_rates",
    "compute_global_ratio",
    "compute_sessions",
    "compute_map_breakdown",
    "mark_firefight",
    "is_allowed_playlist_name",
    "build_option_map",
    "build_xuid_option_map",
    "KVPair",
    "compute_killer_victim_pairs",
    "killer_victim_counts_long",
    "killer_victim_matrix",
]
