"""Gestion des styles CSS."""

from __future__ import annotations

import os
from functools import lru_cache
import html

from src.ui.player_assets import file_to_data_url


def get_css_path() -> str:
    """Retourne le chemin du fichier CSS."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(
        repo_root,
        "static",
        "styles.css",
    )


def load_css() -> str:
    """Charge le contenu du fichier CSS.
    
    Returns:
        Contenu CSS avec balises <style>.
    """
    css_path = get_css_path()

    mtime: float | None
    try:
        mtime = os.path.getmtime(css_path)
    except OSError:
        mtime = None

    return _load_css_cached(css_path, mtime)


@lru_cache(maxsize=8)
def _load_css_cached(css_path: str, mtime: float | None) -> str:
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>\n{css_content}\n</style>"
    except FileNotFoundError:
        # Fallback: CSS minimal si le fichier n'existe pas
        return """
        <style>
            .hero { padding: 18px; margin-bottom: 14px; }
            .hero .title { font-size: 28px; font-weight: 700; }
            .hero .subtitle { color: #aaa; font-size: 14px; }
        </style>
        """


def get_hero_html(
    *,
    player_name: str | None = None,
    service_tag: str | None = None,
    rank_label: str | None = None,
    rank_subtitle: str | None = None,
    rank_icon_path: str | None = None,
    banner_path: str | None = None,
    backdrop_path: str | None = None,
    nameplate_path: str | None = None,
    id_badge_text_color: str | None = None,
    emblem_path: str | None = None,
) -> str:
    """Retourne le HTML du banner hero.

    - Si `player_name` est fourni, affiche un header "profil" (bannière/emblème/rang) si possible.
    - Aucun accès réseau ici: on travaille uniquement avec des chemins locaux déjà résolus.
    """

    p = (player_name or "").strip()
    if not p:
        return """
        <div class="wp-notch-top"></div>
        <div class="wp-notch-bottom"></div>
        <div class="hero">
            <div class="title">OpenSpartan Graphs</div>
            <div class="subtitle">Analyse tes parties Halo Infinite depuis la DB OpenSpartan Workshop — filtres, séries temporelles, amis, maps.</div>
        </div>
        """

    # Images locales (déjà résolues en dehors de ce module)
    backdrop_data = file_to_data_url(backdrop_path, max_bytes=8 * 1024 * 1024)
    banner_data = file_to_data_url(banner_path, max_bytes=8 * 1024 * 1024)
    emblem_data = file_to_data_url(emblem_path)
    nameplate_data = file_to_data_url(nameplate_path)
    rank_icon_data = file_to_data_url(rank_icon_path)

    # Priorité au backdrop si disponible, sinon banner.
    bg_data = backdrop_data or banner_data
    bg_style = "" if not bg_data else f" background-image: url('{bg_data}');"

    r = (rank_label or "").strip()
    rs = (rank_subtitle or "").strip()
    st = (service_tag or "").strip()
    badge_color = (id_badge_text_color or "").strip()

    safe_player = html.escape(p)
    safe_service_tag = html.escape(st)
    safe_rank = html.escape(r)
    safe_rank_sub = html.escape(rs)
    safe_badge_color = html.escape(badge_color)

    emblem_html = "" if not emblem_data else (
        "<div class='hero-player__emblem'>"
        f"<img src='{emblem_data}' alt='emblem' />"
        "</div>"
    )

    nameplate_style = "" if not nameplate_data else f" style=\"background-image: url('{nameplate_data}');\""
    badge_style = "" if not safe_badge_color else f" style=\"color: {safe_badge_color};\""

    id_line = (
        "<div class='hero-player__id'>"
        f"  <div class='hero-player__gamertag'{badge_style}>{safe_player}</div>"
        f"  <div class='hero-player__servicetag'{badge_style}>{('[' + safe_service_tag + ']') if safe_service_tag else ''}</div>"
        "</div>"
    )

    nameplate_html = f"<div class='hero-player__nameplate'{nameplate_style}></div>"
    chips = ""
    if safe_rank:
        if rank_icon_data:
            chips += (
                "<span class='chip'>"
                f"<img src='{rank_icon_data}' alt='rank' "
                "style='height:16px;width:16px;vertical-align:-3px;margin-right:6px;'/>"
                f"{safe_rank}"
                "</span>"
            )
        else:
            chips += f"<span class='chip'>{safe_rank}</span>"
    if safe_rank_sub:
        chips += f"<span class='chip'>{safe_rank_sub}</span>"
    chips_html = "" if not chips else f"<div class='chips hero-player__chips'>{chips}</div>"

    return (
        "<div class='wp-notch-top'></div>"
        "<div class='wp-notch-bottom'></div>"
        f"<div class='hero hero--player' style='{bg_style}'>"
        "  <div class='hero__overlay'></div>"
        "  <div class='hero-player'>"
        f"    {emblem_html}"
        "    <div class='hero-player__text'>"
        "      <div class='hero-player__badge'>"
        f"        {nameplate_html}"
        f"        {id_line}"
        "      </div>"
        f"      {chips_html}"
        "    </div>"
        "  </div>"
        "</div>"
    )


def get_notches_html() -> str:
    """Retourne le HTML des découpes haut/bas."""
    return """
    <div class="wp-notch-top"></div>
    <div class="wp-notch-bottom"></div>
    """
