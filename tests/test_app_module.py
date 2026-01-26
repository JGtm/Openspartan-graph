"""Tests pour le module src.app (Phase 1 refactoring)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.app.state import (
    PlayerIdentity,
    AppState,
    get_db_cache_key,
    get_aliases_cache_key,
)
from src.app.routing import (
    Page,
    build_app_url,
)


class TestPlayerIdentity:
    """Tests pour PlayerIdentity."""

    def test_display_name_with_gamertag(self):
        """Test avec gamertag."""
        identity = PlayerIdentity(
            xuid_or_gamertag="Spartan117",
            xuid_fallback="1234567890",
            waypoint_player="Spartan117",
        )
        assert identity.display_name == "Spartan117"

    def test_display_name_with_xuid_only(self):
        """Test avec XUID uniquement."""
        identity = PlayerIdentity(
            xuid_or_gamertag="",
            xuid_fallback="1234567890",
            waypoint_player="",
        )
        assert identity.display_name == "1234567890"

    def test_display_name_empty(self):
        """Test sans identité."""
        identity = PlayerIdentity()
        assert identity.display_name == "Joueur"

    def test_xuid_from_gamertag(self):
        """Test extraction XUID quand gamertag fourni."""
        identity = PlayerIdentity(
            xuid_or_gamertag="Spartan117",
            xuid_fallback="1234567890",
        )
        assert identity.xuid == "1234567890"

    def test_xuid_from_numeric(self):
        """Test extraction XUID quand xuid_or_gamertag est numérique."""
        identity = PlayerIdentity(
            xuid_or_gamertag="1234567890",
            xuid_fallback="",
        )
        assert identity.xuid == "1234567890"


class TestAppState:
    """Tests pour AppState."""

    def test_clear_filters(self):
        """Test réinitialisation des filtres."""
        state = AppState(
            filter_playlists=["Quick Play"],
            filter_modes=["Slayer"],
            filter_maps=["Aquarius"],
        )
        state.clear_filters()
        assert state.filter_playlists == []
        assert state.filter_modes == []
        assert state.filter_maps == []


class TestPage:
    """Tests pour l'enum Page."""

    def test_from_string_exact(self):
        """Test conversion exacte."""
        assert Page.from_string("Accueil") == Page.ACCUEIL
        assert Page.from_string("Paramètres") == Page.PARAMETRES

    def test_from_string_case_insensitive(self):
        """Test conversion case insensitive."""
        assert Page.from_string("accueil") == Page.ACCUEIL
        assert Page.from_string("ACCUEIL") == Page.ACCUEIL

    def test_from_string_unknown(self):
        """Test conversion avec valeur inconnue."""
        assert Page.from_string("PageInconnue") == Page.ACCUEIL

    def test_navigable_pages(self):
        """Test liste des pages navigables."""
        pages = Page.navigable_pages()
        assert Page.ACCUEIL in pages
        assert Page.PARAMETRES in pages
        assert Page.MATCH_VIEW not in pages  # Page interne


class TestBuildAppUrl:
    """Tests pour build_app_url."""

    def test_simple_page(self):
        """Test URL simple."""
        url = build_app_url(Page.ACCUEIL)
        assert "page=Accueil" in url

    def test_with_params(self):
        """Test URL avec paramètres."""
        url = build_app_url(Page.MATCH_VIEW, match_id="abc123")
        assert "page=match_view" in url
        assert "match_id=abc123" in url

    def test_string_page(self):
        """Test avec page en string."""
        url = build_app_url("Historique")
        assert "page=Historique" in url


class TestGetDbCacheKey:
    """Tests pour get_db_cache_key."""

    def test_nonexistent_file(self):
        """Test avec fichier inexistant."""
        result = get_db_cache_key("/nonexistent/path.db")
        assert result is None

    def test_existing_file(self, tmp_path):
        """Test avec fichier existant."""
        db_file = tmp_path / "test.db"
        db_file.write_text("test content")
        
        result = get_db_cache_key(str(db_file))
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)  # mtime_ns
        assert isinstance(result[1], int)  # size
