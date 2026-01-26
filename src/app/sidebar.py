"""Logique et rendu de la sidebar.

Ce module centralise :
- Le rendu de la sidebar (brand, navigation, filtres)
- Le bouton de synchronisation
- Le sélecteur de joueur (multi-joueurs)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import streamlit as st

from src.app.state import get_db_cache_key
from src.ui.sync import (
    is_spnkr_db_path,
    render_sync_indicator,
    sync_all_players,
)
from src.ui.multiplayer import (
    is_multi_player_db,
    render_player_selector,
    get_player_display_name,
)

if TYPE_CHECKING:
    from src.ui.settings import AppSettings


def render_sidebar(
    *,
    db_path: str,
    xuid: str,
    settings: "AppSettings",
    on_player_change: Callable[[str], None] | None = None,
    on_sync_complete: Callable[[], None] | None = None,
) -> str:
    """Rend la sidebar complète.

    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur courant.
        settings: Paramètres de l'application.
        on_player_change: Callback appelé quand le joueur change.
        on_sync_complete: Callback appelé après une sync réussie.

    Returns:
        Le XUID potentiellement mis à jour.
    """
    with st.sidebar:
        # Brand
        st.markdown(
            "<div class='os-sidebar-brand' style='font-size: 2.5em;'>LevelUp</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='os-sidebar-divider'></div>", unsafe_allow_html=True)

        # Indicateur de dernière synchronisation
        if db_path and os.path.exists(db_path):
            render_sync_indicator(db_path)

        # Sélecteur multi-joueurs (si DB fusionnée)
        new_xuid = render_player_selector_sidebar(
            db_path=db_path,
            xuid=xuid,
            on_change=on_player_change,
        )
        if new_xuid and new_xuid != xuid:
            xuid = new_xuid

        # Bouton Sync
        render_sync_button(
            db_path=db_path,
            settings=settings,
            on_complete=on_sync_complete,
        )

    return xuid


def render_player_selector_sidebar(
    *,
    db_path: str,
    xuid: str,
    on_change: Callable[[str], None] | None = None,
) -> str | None:
    """Rend le sélecteur de joueur dans la sidebar.

    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur courant.
        on_change: Callback appelé quand le joueur change.

    Returns:
        Le nouveau XUID si changé, None sinon.
    """
    if not (db_path and os.path.exists(db_path)):
        return None

    new_xuid = render_player_selector(db_path, xuid, key="sidebar_player_selector")

    if new_xuid and new_xuid != xuid:
        st.session_state["xuid_input"] = new_xuid

        # Reset des filtres au changement de joueur
        for filter_key in ["filter_playlists", "filter_modes", "filter_maps"]:
            if filter_key in st.session_state:
                del st.session_state[filter_key]

        if on_change:
            on_change(new_xuid)

        return new_xuid

    return None


def render_sync_button(
    *,
    db_path: str,
    settings: "AppSettings",
    on_complete: Callable[[], None] | None = None,
) -> bool:
    """Rend le bouton de synchronisation.

    Args:
        db_path: Chemin vers la base de données.
        settings: Paramètres de l'application.
        on_complete: Callback appelé après une sync réussie.

    Returns:
        True si une sync a été effectuée avec succès.
    """
    if not (db_path and is_spnkr_db_path(db_path) and os.path.exists(db_path)):
        return False

    if st.button(
        "🔄 Synchroniser",
        key="sidebar_sync_button",
        help="Synchronise tous les joueurs (nouveaux matchs, highlights, aliases).",
        use_container_width=True,
    ):
        with st.spinner("Synchronisation en cours..."):
            ok, msg = sync_all_players(
                db_path=db_path,
                match_type=str(
                    getattr(settings, "spnkr_refresh_match_type", "matchmaking")
                    or "matchmaking"
                ),
                max_matches=int(
                    getattr(settings, "spnkr_refresh_max_matches", 200) or 200
                ),
                rps=int(getattr(settings, "spnkr_refresh_rps", 5) or 5),
                with_highlight_events=True,
                with_aliases=True,
                delta=True,
                timeout_seconds=180,
            )

        if ok:
            st.success(msg)
            if on_complete:
                on_complete()
            return True
        else:
            st.error(msg)

    return False


def render_navigation_tabs(
    *,
    pages: list[str],
    current_page: str,
    on_change: Callable[[str], None] | None = None,
) -> str:
    """Rend les onglets de navigation.

    Args:
        pages: Liste des noms de pages.
        current_page: Page courante.
        on_change: Callback appelé quand la page change.

    Returns:
        La page sélectionnée.
    """
    # Trouver l'index de la page courante
    try:
        current_index = pages.index(current_page)
    except ValueError:
        current_index = 0

    # Utiliser st.tabs ou st.radio selon le nombre de pages
    if len(pages) <= 8:
        tabs = st.tabs(pages)
        for i, tab in enumerate(tabs):
            with tab:
                if i != current_index and on_change:
                    on_change(pages[i])
        return pages[current_index]
    else:
        selected = st.radio(
            "Navigation",
            options=pages,
            index=current_index,
            horizontal=True,
            label_visibility="collapsed",
        )
        if selected != current_page and on_change:
            on_change(selected)
        return selected


def render_db_info(db_path: str) -> None:
    """Affiche les informations sur la DB dans la sidebar.

    Args:
        db_path: Chemin vers la base de données.
    """
    if not db_path:
        st.caption("Aucune base de données sélectionnée")
        return

    if not os.path.exists(db_path):
        st.warning(f"Base introuvable: {db_path}")
        return

    try:
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        st.caption(f"📁 {os.path.basename(db_path)} ({size_mb:.1f} MB)")
    except Exception:
        st.caption(f"📁 {os.path.basename(db_path)}")


def render_quick_filters(
    *,
    playlists: list[str],
    selected_playlists: list[str],
    on_change: Callable[[list[str]], None] | None = None,
) -> list[str]:
    """Rend les filtres rapides de playlist.

    Args:
        playlists: Liste des playlists disponibles.
        selected_playlists: Playlists actuellement sélectionnées.
        on_change: Callback appelé quand la sélection change.

    Returns:
        Liste des playlists sélectionnées.
    """
    if not playlists:
        return selected_playlists

    with st.expander("🎮 Filtres rapides", expanded=False):
        new_selection = st.multiselect(
            "Playlists",
            options=playlists,
            default=selected_playlists,
            key="quick_filter_playlists",
        )

        if new_selection != selected_playlists and on_change:
            on_change(new_selection)

        return new_selection

    return selected_playlists
