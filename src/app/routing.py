"""Gestion de la navigation entre pages.

Ce module centralise :
- Le routage entre pages du dashboard
- La gestion des query params pour les liens profonds
- La construction d'URLs internes
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import streamlit as st


class Page(str, Enum):
    """Pages disponibles dans l'application."""

    ACCUEIL = "Accueil"
    DERNIER_MATCH = "Dernier match"
    HISTORIQUE = "Historique"
    SESSIONS = "Sessions"
    CARTES = "Cartes"
    COEQUIPIERS = "Coéquipiers"
    VICTOIRES = "Victoires / Défaites"
    SERIES = "Séries temporelles"
    CITATIONS = "Citations & Médailles"
    RECHERCHE = "Recherche match"
    PARAMETRES = "Paramètres"
    MATCH_VIEW = "match_view"  # Page interne (non listée dans nav)

    @classmethod
    def from_string(cls, value: str) -> "Page":
        """Convertit une chaîne en Page (case insensitive)."""
        value_lower = value.lower().strip()
        for page in cls:
            if page.value.lower() == value_lower:
                return page
        return cls.ACCUEIL

    @classmethod
    def navigable_pages(cls) -> list["Page"]:
        """Retourne les pages affichées dans la navigation."""
        return [
            cls.ACCUEIL,
            cls.DERNIER_MATCH,
            cls.HISTORIQUE,
            cls.SESSIONS,
            cls.CARTES,
            cls.COEQUIPIERS,
            cls.VICTOIRES,
            cls.SERIES,
            cls.CITATIONS,
            cls.RECHERCHE,
            cls.PARAMETRES,
        ]


@dataclass
class Router:
    """Routeur pour la navigation entre pages.

    Gère l'état de navigation et les transitions entre pages.
    """

    current_page: Page = Page.ACCUEIL
    pending_page: Page | None = None
    pending_match_id: str | None = None
    _page_handlers: dict[Page, Callable[[], None]] = field(default_factory=dict)

    def register(self, page: Page, handler: Callable[[], None]) -> None:
        """Enregistre un handler pour une page.

        Args:
            page: Page à gérer.
            handler: Fonction à appeler pour rendre la page.
        """
        self._page_handlers[page] = handler

    def navigate_to(self, page: Page, *, match_id: str | None = None) -> None:
        """Navigue vers une page.

        Args:
            page: Page cible.
            match_id: ID de match optionnel (pour match_view).
        """
        self.current_page = page
        st.session_state["current_page"] = page.value

        if match_id:
            st.session_state["_pending_match_id"] = match_id

        # Clear query params après navigation
        _clear_query_params()

    def render_current_page(self) -> None:
        """Rend la page courante."""
        handler = self._page_handlers.get(self.current_page)
        if handler:
            handler()
        else:
            st.warning(f"Page non implémentée: {self.current_page.value}")

    @classmethod
    def from_session(cls) -> "Router":
        """Charge le routeur depuis session_state."""
        current_str = str(st.session_state.get("current_page", "Accueil") or "Accueil")
        pending_str = st.session_state.get("_pending_page")

        return cls(
            current_page=Page.from_string(current_str),
            pending_page=Page.from_string(pending_str) if pending_str else None,
            pending_match_id=st.session_state.get("_pending_match_id"),
        )

    def consume_pending(self) -> bool:
        """Consomme la navigation en attente.

        Returns:
            True si une navigation a été effectuée.
        """
        if self.pending_page:
            self.current_page = self.pending_page
            st.session_state["current_page"] = self.current_page.value
            self.pending_page = None
            st.session_state.pop("_pending_page", None)
            return True
        return False


def get_current_page() -> Page:
    """Retourne la page courante."""
    current_str = str(st.session_state.get("current_page", "Accueil") or "Accueil")
    return Page.from_string(current_str)


def navigate_to(page: Page | str, *, match_id: str | None = None) -> None:
    """Navigue vers une page.

    Args:
        page: Page cible (Page ou string).
        match_id: ID de match optionnel.
    """
    if isinstance(page, str):
        page = Page.from_string(page)

    st.session_state["current_page"] = page.value

    if match_id:
        st.session_state["_pending_match_id"] = match_id

    _clear_query_params()


def consume_query_params() -> tuple[str | None, str | None]:
    """Consomme les query params et retourne (page, match_id).

    Cette fonction lit les query params, les stocke en session_state,
    puis les nettoie de l'URL.

    Returns:
        Tuple (page, match_id) ou (None, None).
    """
    try:
        qp = dict(st.query_params)
        qp_page = _qp_first(qp.get("page"))
        qp_mid = _qp_first(qp.get("match_id"))
    except Exception:
        return None, None

    qp_token = (str(qp_page or "").strip(), str(qp_mid or "").strip())

    # Déjà consommé ?
    if not any(qp_token):
        return None, None

    if st.session_state.get("_consumed_query_params") == qp_token:
        return None, None

    # Marquer comme consommé
    st.session_state["_consumed_query_params"] = qp_token

    if qp_token[0]:
        st.session_state["_pending_page"] = qp_token[0]
    if qp_token[1]:
        st.session_state["_pending_match_id"] = qp_token[1]

    # Nettoyer l'URL
    _clear_query_params()

    return qp_token[0] or None, qp_token[1] or None


def build_app_url(page: Page | str, **params: str) -> str:
    """Construit une URL interne vers une page.

    Args:
        page: Page cible.
        **params: Paramètres additionnels (match_id, etc.).

    Returns:
        URL relative avec query params.
    """
    if isinstance(page, Page):
        page_str = page.value
    else:
        page_str = str(page)

    qp: dict[str, str] = {"page": page_str}
    for k, v in params.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            qp[k] = s

    return "?" + urllib.parse.urlencode(qp)


def _qp_first(value) -> str | None:
    """Extrait la première valeur d'un query param."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    s = str(value)
    return s if s.strip() else None


def _clear_query_params() -> None:
    """Nettoie les query params de l'URL."""
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def _set_query_params(**kwargs: str) -> None:
    """Définit les query params (utile pour liens partageables)."""
    clean: dict[str, str] = {
        k: str(v) for k, v in kwargs.items() if v is not None and str(v).strip()
    }
    try:
        st.query_params.clear()
        for k, v in clean.items():
            st.query_params[k] = v
    except Exception:
        try:
            st.experimental_set_query_params(**clean)
        except Exception:
            pass
