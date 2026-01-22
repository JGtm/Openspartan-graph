"""Pages UI du dashboard."""

from src.ui.pages.session_compare import render_session_comparison_page
from src.ui.pages.timeseries import render_timeseries_page
from src.ui.pages.win_loss import render_win_loss_page
from src.ui.pages.match_history import render_match_history_page
from src.ui.pages.teammates import render_teammates_page
from src.ui.pages.citations import render_citations_page
from src.ui.pages.settings import render_settings_page
from src.ui.pages.match_view import render_match_view

__all__ = [
    "render_session_comparison_page",
    "render_timeseries_page",
    "render_win_loss_page",
    "render_match_history_page",
    "render_teammates_page",
    "render_citations_page",
    "render_settings_page",
    "render_match_view",
]
