"""Module UI - Gestion des alias, médailles et helpers d'interface."""

from src.ui.aliases import (
    load_aliases_file,
    save_aliases_file,
    get_xuid_aliases,
    display_name_from_xuid,
)
from src.ui.styles import load_css, get_hero_html
from src.ui.translations import translate_playlist_name, translate_pair_name
from src.ui.medals import (
    load_medal_name_maps,
    medal_has_known_label,
    get_medals_cache_dir,
    get_local_medals_icons_dir,
    medal_label,
    medal_icon_path,
    render_medals_grid,
)
from src.ui.formatting import format_date_fr, format_mmss

from src.ui.settings import AppSettings, load_settings, save_settings
from src.ui.path_picker import directory_input, file_input
from src.ui.profile_api import ProfileAppearance, get_profile_appearance, get_xuid_for_gamertag, ensure_spnkr_tokens

__all__ = [
    # aliases
    "load_aliases_file",
    "save_aliases_file",
    "get_xuid_aliases",
    "display_name_from_xuid",
    # styles
    "load_css",
    "get_hero_html",
    # translations
    "translate_playlist_name",
    "translate_pair_name",
    # medals
    "load_medal_name_maps",
    "medal_has_known_label",
    "get_medals_cache_dir",
    "get_local_medals_icons_dir",
    "medal_label",
    "medal_icon_path",
    "render_medals_grid",
    # formatting
    "format_date_fr",
    "format_mmss",
    # settings
    "AppSettings",
    "load_settings",
    "save_settings",
    # profile api
    "ProfileAppearance",
    "get_profile_appearance",
    "get_xuid_for_gamertag",
    "ensure_spnkr_tokens",
    # path picker
    "directory_input",
    "file_input",
]
