"""Tests pour le refactoring de spnkr_import_db.py."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestExtractXuidsFromMatchStats:
    """Tests pour _extract_xuids_from_match_stats."""

    def test_extracts_xuids_from_players(self):
        """Test extraction des XUIDs depuis la liste Players."""
        # Import dynamique pour éviter les dépendances SPNKr
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _extract_xuids_from_match_stats
        
        match_json = {
            "Players": [
                {"PlayerId": "xuid(2535445291321133)", "PlayerGamertag": "Player1"},
                {"PlayerId": "xuid(2533274880629884)", "PlayerGamertag": "Player2"},
            ]
        }
        
        xuids = _extract_xuids_from_match_stats(match_json)
        assert 2535445291321133 in xuids
        assert 2533274880629884 in xuids
        assert len(xuids) == 2

    def test_returns_empty_for_no_players(self):
        """Test retourne liste vide si pas de Players."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _extract_xuids_from_match_stats
        
        assert _extract_xuids_from_match_stats({}) == []
        assert _extract_xuids_from_match_stats({"Players": None}) == []
        assert _extract_xuids_from_match_stats({"Players": []}) == []


class TestExtractGamertagsFromMatchStats:
    """Tests pour _extract_gamertags_from_match_stats."""

    def test_extracts_gamertags(self):
        """Test extraction des paires XUID -> Gamertag."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _extract_gamertags_from_match_stats
        
        match_json = {
            "Players": [
                {"PlayerId": "xuid(2535445291321133)", "PlayerGamertag": "TestPlayer"},
                {"PlayerId": "xuid(2533274880629884)", "Gamertag": "AnotherPlayer"},
            ]
        }
        
        gamertags = _extract_gamertags_from_match_stats(match_json)
        assert gamertags.get(2535445291321133) == "TestPlayer"
        assert gamertags.get(2533274880629884) == "AnotherPlayer"

    def test_handles_missing_gamertag(self):
        """Test gère les joueurs sans gamertag."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _extract_gamertags_from_match_stats
        
        match_json = {
            "Players": [
                {"PlayerId": "xuid(2535445291321133)"},  # Pas de gamertag
            ]
        }
        
        gamertags = _extract_gamertags_from_match_stats(match_json)
        # Ne devrait pas inclure ce joueur car pas de gamertag
        assert 2535445291321133 not in gamertags


class TestImportContext:
    """Tests pour ImportContext et MatchResult dataclasses."""

    def test_import_context_creation(self):
        """Test création d'ImportContext."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import ImportContext
        
        ctx = ImportContext(
            client=MagicMock(),
            con=MagicMock(),
            existing_match_ids={"match1", "match2"},
            asset_seen=set(),
            asset_missing=set(),
            film_mod=None,
            fetch_skill=True,
            fetch_assets=True,
            fetch_highlight_events=True,
            fetch_aliases=True,
            delta_mode=False,
            player="TestPlayer",
        )
        
        assert ctx.delta_mode is False
        assert "match1" in ctx.existing_match_ids
        assert ctx.fetch_skill is True

    def test_match_result_creation(self):
        """Test création de MatchResult."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import MatchResult
        
        result = MatchResult(
            match_id="test-match-id",
            inserted=True,
            aliases_updated=5,
            xuids_seen={123, 456},
            skipped_delta=False,
        )
        
        assert result.match_id == "test-match-id"
        assert result.inserted is True
        assert 123 in result.xuids_seen


class TestAssetRef:
    """Tests pour _asset_ref."""

    def test_extracts_asset_ids(self):
        """Test extraction AssetId et VersionId."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _asset_ref
        
        match_info = {
            "MapVariant": {
                "AssetId": "map-asset-id",
                "VersionId": "map-version-id",
            }
        }
        
        aid, vid = _asset_ref(match_info, "MapVariant")
        assert aid == "map-asset-id"
        assert vid == "map-version-id"

    def test_returns_none_for_missing(self):
        """Test retourne None si asset absent."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _asset_ref
        
        aid, vid = _asset_ref({}, "MapVariant")
        assert aid is None
        assert vid is None

    def test_returns_none_for_non_dict(self):
        """Test retourne None si pas un dict."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _asset_ref
        
        aid, vid = _asset_ref({"MapVariant": "not-a-dict"}, "MapVariant")
        assert aid is None
        assert vid is None


class TestEventToDict:
    """Tests pour _event_to_dict."""

    def test_dict_passthrough(self):
        """Test que les dicts passent tels quels."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _event_to_dict
        
        event = {"type": "kill", "value": 1}
        result = _event_to_dict(event)
        assert result == event

    def test_model_dump(self):
        """Test avec un objet ayant model_dump (Pydantic v2)."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _event_to_dict
        
        mock_event = MagicMock()
        mock_event.model_dump.return_value = {"key": "value"}
        
        result = _event_to_dict(mock_event)
        assert result == {"key": "value"}

    def test_fallback_to_raw(self):
        """Test fallback vers str() si aucune méthode disponible."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import _event_to_dict
        
        class SimpleEvent:
            pass
        
        event = SimpleEvent()
        result = _event_to_dict(event)
        assert "raw" in result


class TestDefaultParallelMatches:
    """Tests pour la constante de parallélisation."""

    def test_default_parallel_matches_is_reasonable(self):
        """Test que la constante est raisonnable (1-10)."""
        import sys
        sys.path.insert(0, "scripts")
        from spnkr_import_db import DEFAULT_PARALLEL_MATCHES
        
        assert 1 <= DEFAULT_PARALLEL_MATCHES <= 10
        # Actuellement configuré à 3
        assert DEFAULT_PARALLEL_MATCHES == 3
