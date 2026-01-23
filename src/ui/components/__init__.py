"""Composants UI réutilisables pour le dashboard."""

from src.ui.components.checkbox_filter import (
    render_checkbox_filter,
    render_hierarchical_checkbox_filter,
    get_firefight_playlists,
)
from src.ui.components.performance import (
    compute_session_performance_score,
    compute_session_performance_score_v2_ui,
    get_score_color,
    get_score_label,
    render_performance_score_card,
    render_metric_comparison_row,
)
from src.ui.components.kpi import (
    render_kpi_cards,
    render_top_summary,
)

__all__ = [
    "render_checkbox_filter",
    "render_hierarchical_checkbox_filter",
    "get_firefight_playlists",
    "compute_session_performance_score",
    "compute_session_performance_score_v2_ui",
    "get_score_color",
    "get_score_label",
    "render_performance_score_card",
    "render_metric_comparison_row",
    "render_kpi_cards",
    "render_top_summary",
]
