"""Module d'analyse des données."""

from src.analysis.stats import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
)
from src.analysis.sessions import (
    compute_sessions,
    compute_sessions_with_context,
    is_session_potentially_active,
    DEFAULT_SESSION_GAP_MINUTES,
    SESSION_CUTOFF_HOUR,
)
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
from src.analysis.performance_score import (
    compute_match_performance_from_row,
    compute_relative_performance_score,
    compute_performance_series,
)
from src.analysis.performance_config import MIN_MATCHES_FOR_RELATIVE

__all__ = [
    "compute_aggregated_stats",
    "compute_outcome_rates",
    "compute_global_ratio",
    "compute_sessions",
    "compute_sessions_with_context",
    "is_session_potentially_active",
    "DEFAULT_SESSION_GAP_MINUTES",
    "SESSION_CUTOFF_HOUR",
    "compute_map_breakdown",
    "mark_firefight",
    "is_allowed_playlist_name",
    "build_option_map",
    "build_xuid_option_map",
    "KVPair",
    "compute_killer_victim_pairs",
    "killer_victim_counts_long",
    "killer_victim_matrix",
    "compute_match_performance_from_row",
    "compute_relative_performance_score",
    "compute_performance_series",
    "MIN_MATCHES_FOR_RELATIVE",
]
