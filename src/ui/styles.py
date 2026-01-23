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
    grid_mode: bool = False,
) -> str:
    """Retourne le HTML du banner hero (Spartan ID card style).

    Structure visuelle (de l'arrière vers l'avant):
    1. Career Rank icon (à gauche, aligné horizontalement)
    2. Spartan ID card:
       - Backdrop (240px, 2/3 de la nameplate, centré)
       - Nameplate (360px)
       - Emblème (par-dessus, aligné à gauche avec padding 10px)
       - Gamertag + Service tag (à droite de l'emblème)
    
    Args:
        grid_mode: Si True, utilise un style compact pour les grilles (sans margin-top, centré).
    """

    p = (player_name or "").strip()
    if not p:
        return """
        <div class="wp-notch-top"></div>
        <div class="wp-notch-bottom"></div>
        <div class="hero">
            <div class="title">LevelUp</div>
            <div class="subtitle">Analyse tes parties Halo Infinite depuis la DB SPNKr — filtres, séries temporelles, amis, maps.</div>
        </div>
        """

    # Images locales (déjà résolues en dehors de ce module)
    backdrop_data = file_to_data_url(backdrop_path, max_bytes=8 * 1024 * 1024)
    emblem_data = file_to_data_url(emblem_path)
    nameplate_data = file_to_data_url(nameplate_path)
    rank_icon_data = file_to_data_url(rank_icon_path)

    st = (service_tag or "").strip()

    safe_player = html.escape(p)
    safe_service_tag = html.escape(st)

    # Backdrop (arrière-plan, 2/3 de la largeur)
    backdrop_html = ""
    if backdrop_data:
        backdrop_html = f"<div class='spartan-id__backdrop'><img src='{backdrop_data}' alt='' /></div>"

    # Nameplate
    nameplate_html = ""
    if nameplate_data:
        nameplate_html = f"<div class='spartan-id__nameplate'><img src='{nameplate_data}' alt='' /></div>"

    # Emblème
    emblem_html = ""
    if emblem_data:
        emblem_html = f"<div class='spartan-id__emblem'><img src='{emblem_data}' alt='emblem' /></div>"

    # Service tag
    service_tag_html = ""
    if safe_service_tag:
        service_tag_html = f"<div class='spartan-id__servicetag'>{safe_service_tag}</div>"

    # Classe wrapper (avec ou sans --grid)
    wrapper_class = "spartan-id-wrapper spartan-id-wrapper--grid" if grid_mode else "spartan-id-wrapper"
    
    # Notches uniquement en mode normal (pas en grille)
    notches = "" if grid_mode else "<div class='wp-notch-top'></div><div class='wp-notch-bottom'></div>"

    # Career Rank (icône + label, à gauche du Spartan ID)
    rank_html = ""
    if rank_icon_data or rank_label:
        safe_rank_label = html.escape(rank_label or "") if rank_label else ""
        safe_rank_subtitle = html.escape(rank_subtitle or "") if rank_subtitle else ""
        
        rank_icon_html = ""
        if rank_icon_data:
            rank_icon_html = f"<img src='{rank_icon_data}' alt='rank' class='career-rank__icon' />"
        
        rank_label_html = ""
        if safe_rank_label:
            rank_label_html = f"<div class='career-rank__label'>{safe_rank_label}</div>"
        
        rank_subtitle_html = ""
        if safe_rank_subtitle:
            rank_subtitle_html = f"<div class='career-rank__subtitle'>{safe_rank_subtitle}</div>"
        
        rank_html = (
            "<div class='career-rank'>"
            f"  {rank_icon_html}"
            "  <div class='career-rank__text'>"
            f"    {rank_label_html}"
            f"    {rank_subtitle_html}"
            "  </div>"
            "</div>"
        )

    return (
        f"{notches}"
        f"<div class='{wrapper_class}'>"
        f"  {rank_html}"
        "  <div class='spartan-id'>"
        f"    {backdrop_html}"
        f"    {nameplate_html}"
        f"    {emblem_html}"
        "    <div class='spartan-id__text'>"
        f"      <div class='spartan-id__gamertag'>{safe_player}</div>"
        f"      {service_tag_html}"
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
