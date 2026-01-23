"""Tests pour les fonctions d'analyse."""

import pandas as pd
import pytest

from src.analysis.stats import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
    format_selected_matches_summary,
    format_mmss,
)
from src.analysis.filters import (
    is_allowed_playlist_name,
    build_option_map,
)
from src.models import OutcomeRates

from src.analysis.killer_victim import (
    compute_killer_victim_pairs,
    killer_victim_counts_long,
)


class TestComputeGlobalRatio:
    """Tests pour compute_global_ratio."""

    def test_normal_values(self):
        """Test avec des valeurs normales."""
        df = pd.DataFrame({
            "kills": [10, 15, 20],
            "deaths": [5, 10, 10],
            "assists": [4, 6, 8],
        })
        # Total: kills=45, deaths=25, assists=18
        # Ratio = (45 + 18/2) / 25 = 54 / 25 = 2.16
        result = compute_global_ratio(df)
        assert result == pytest.approx(2.16)

    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        df = pd.DataFrame({"kills": [], "deaths": [], "assists": []})
        assert compute_global_ratio(df) is None

    def test_zero_deaths(self):
        """Test avec zéro deaths."""
        df = pd.DataFrame({
            "kills": [10],
            "deaths": [0],
            "assists": [4],
        })
        assert compute_global_ratio(df) is None


class TestComputeOutcomeRates:
    """Tests pour compute_outcome_rates."""

    def test_normal_values(self):
        """Test avec des valeurs normales."""
        df = pd.DataFrame({
            "outcome": [2, 2, 3, 1, 2, 3, 4],
        })
        rates = compute_outcome_rates(df)
        assert rates.wins == 3
        assert rates.losses == 2
        assert rates.ties == 1
        assert rates.no_finish == 1
        assert rates.total == 7

    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        df = pd.DataFrame({"outcome": []})
        rates = compute_outcome_rates(df)
        assert rates.total == 0


class TestIsAllowedPlaylistName:
    """Tests pour is_allowed_playlist_name."""

    def test_quick_play(self):
        """Test Quick Play."""
        assert is_allowed_playlist_name("Quick Play") is True
        assert is_allowed_playlist_name("quick play") is True
        assert is_allowed_playlist_name("QuickPlay") is True

    def test_ranked_slayer(self):
        """Test Ranked Slayer."""
        assert is_allowed_playlist_name("Ranked Slayer") is True
        assert is_allowed_playlist_name("Ranked: Slayer") is True

    def test_ranked_arena(self):
        """Test Ranked Arena."""
        assert is_allowed_playlist_name("Ranked Arena") is True

    def test_french_labels(self):
        """Test libellés FR (UI)."""
        assert is_allowed_playlist_name("Partie rapide") is True
        assert is_allowed_playlist_name("Arène classée") is True
        assert is_allowed_playlist_name("Assassin classé") is True

    def test_not_allowed(self):
        """Test playlists non autorisées."""
        assert is_allowed_playlist_name("Custom Game") is False
        assert is_allowed_playlist_name("Firefight") is False
        assert is_allowed_playlist_name("") is False


class TestBuildOptionMap:
    """Tests pour build_option_map."""

    def test_normal_values(self):
        """Test avec des valeurs normales."""
        names = pd.Series(["Map A", "Map B", "Map C"])
        ids = pd.Series(["id-a", "id-b", "id-c"])
        result = build_option_map(names, ids)
        
        assert result["Map A"] == "id-a"
        assert result["Map B"] == "id-b"
        assert result["Map C"] == "id-c"

    def test_with_uuid_suffix(self):
        """Test avec suffixe UUID à nettoyer."""
        names = pd.Series(["Streets - abc12345"])
        ids = pd.Series(["id-streets"])
        result = build_option_map(names, ids)
        
        assert "Streets" in result
        assert result["Streets"] == "id-streets"

    def test_empty_values(self):
        """Test avec valeurs vides."""
        names = pd.Series(["", "Map A", None])
        ids = pd.Series(["id-empty", "id-a", "id-none"])
        result = build_option_map(names, ids)
        
        assert len(result) == 1
        assert "Map A" in result


class TestFormatMmss:
    """Tests pour format_mmss."""

    def test_normal(self):
        """Test avec valeur normale."""
        # Note: format_mmss utilise le format "mm:ss" avec zero-padding sur les minutes
        assert format_mmss(90.0) == "01:30"
        assert format_mmss(65.0) == "01:05"
        assert format_mmss(30.0) == "00:30"

    def test_none(self):
        """Test avec None."""
        assert format_mmss(None) == "-"

    def test_nan(self):
        """Test avec NaN."""
        assert format_mmss(float("nan")) == "-"


class TestFormatSelectedMatchesSummary:
    """Tests pour format_selected_matches_summary."""

    def test_normal(self):
        """Test avec valeurs normales."""
        rates = OutcomeRates(wins=5, losses=3, ties=1, no_finish=1, total=10)
        result = format_selected_matches_summary(10, rates)
        
        assert "10" in result
        assert "Victoires: 5" in result
        assert "Défaites: 3" in result

    def test_zero_matches(self):
        """Test avec 0 matchs."""
        rates = OutcomeRates()
        result = format_selected_matches_summary(0, rates)
        assert "Aucun match" in result

    def test_singular(self):
        """Test du singulier."""
        rates = OutcomeRates(wins=1, losses=1, ties=1, no_finish=1, total=4)
        result = format_selected_matches_summary(1, rates)
        
        assert "Partie sélectionnée: 1" in result
        assert "Victoire: 1" in result


class TestKillerVictim:
    def test_join_kill_death_with_tolerance(self):
        events = [
            {"event_type": "kill", "time_ms": 1000, "xuid": "1", "gamertag": "Alice"},
            {"event_type": "death", "time_ms": 1003, "xuid": "2", "gamertag": "Bob"},
        ]

        pairs = compute_killer_victim_pairs(events, tolerance_ms=5)
        assert len(pairs) == 1
        assert pairs[0].killer_gamertag == "Alice"
        assert pairs[0].victim_gamertag == "Bob"

    def test_no_match_outside_tolerance(self):
        events = [
            {"event_type": "kill", "time_ms": 1000, "xuid": "1", "gamertag": "Alice"},
            {"event_type": "death", "time_ms": 1010, "xuid": "2", "gamertag": "Bob"},
        ]
        pairs = compute_killer_victim_pairs(events, tolerance_ms=5)
        assert pairs == []

    def test_infer_event_type_from_type_hint(self):
        # Fallback: type_hint 50=kill, 20=death
        events = [
            {"type_hint": 50, "time_ms": 2000, "xuid": "1", "gamertag": "Alice"},
            {"type_hint": 20, "time_ms": 2000, "xuid": "3", "gamertag": "Cara"},
        ]
        pairs = compute_killer_victim_pairs(events, tolerance_ms=0)
        assert len(pairs) == 1
        assert pairs[0].victim_gamertag == "Cara"

    def test_aggregate_counts(self):
        events = [
            {"event_type": "kill", "time_ms": 1000, "xuid": "1", "gamertag": "Alice"},
            {"event_type": "death", "time_ms": 1000, "xuid": "2", "gamertag": "Bob"},
            {"event_type": "kill", "time_ms": 2000, "xuid": "1", "gamertag": "Alice"},
            {"event_type": "death", "time_ms": 2000, "xuid": "2", "gamertag": "Bob"},
        ]
        pairs = compute_killer_victim_pairs(events, tolerance_ms=0)
        df = killer_victim_counts_long(pairs)
        assert int(df.iloc[0]["count"]) == 2
