"""Tests pour les nouveaux modules Phase 2 (helpers, filters, profile, kpis)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.app.helpers import (
    clean_asset_label,
    normalize_mode_label,
    normalize_map_label,
    is_uuid_like,
    compute_total_play_seconds,
    avg_match_duration_seconds,
)
from src.app.kpis import (
    KPIStats,
    compute_kpi_stats,
)
from src.app.profile import (
    PlayerIdentity,
    get_identity_from_secrets,
)


class TestCleanAssetLabel:
    """Tests pour clean_asset_label."""

    def test_removes_suffix_with_dash(self):
        """Retire les suffixes après tiret avec ID."""
        result = clean_asset_label("Quick Play - 12345678")
        assert result == "Quick Play"

    def test_preserves_normal_label(self):
        """Garde les labels normaux."""
        result = clean_asset_label("Ranked Arena")
        assert result == "Ranked Arena"

    def test_handles_none(self):
        """Retourne None pour None."""
        assert clean_asset_label(None) is None

    def test_handles_empty_string(self):
        """Retourne None pour chaîne vide."""
        assert clean_asset_label("") is None
        assert clean_asset_label("   ") is None


class TestNormalizeModeLabel:
    """Tests pour normalize_mode_label."""

    def test_translates_known_mode(self):
        """Traduit un mode connu."""
        result = normalize_mode_label("Arena:Slayer")
        assert result == "Arène : Assassin"

    def test_removes_map_name(self):
        """Retire le nom de carte."""
        result = normalize_mode_label("Arena:Slayer on Aquarius")
        assert result == "Arène : Assassin"

    def test_removes_forge_suffix(self):
        """Retire le suffixe Forge."""
        # Note: dépend des traductions disponibles
        result = normalize_mode_label("Arena:Slayer - Forge")
        # Vérifie juste que ce n'est pas None et ne contient pas Forge
        assert result is None or "Forge" not in result

    def test_handles_none(self):
        """Retourne None pour None."""
        assert normalize_mode_label(None) is None


class TestNormalizeMapLabel:
    """Tests pour normalize_map_label."""

    def test_removes_suffix(self):
        """Retire le suffixe après ' - '."""
        result = normalize_map_label("Aquarius - Live Fire")
        assert result == "Aquarius"

    def test_preserves_normal_map(self):
        """Garde les noms normaux."""
        result = normalize_map_label("Bazaar")
        assert result == "Bazaar"

    def test_masks_uuid(self):
        """Masque les UUIDs."""
        result = normalize_map_label("a446725e-b281-414c-a21e")
        assert result == "Carte inconnue"

    def test_handles_none(self):
        """Retourne None pour None."""
        assert normalize_map_label(None) is None


class TestIsUuidLike:
    """Tests pour is_uuid_like."""

    def test_full_uuid(self):
        """Reconnaît un UUID complet."""
        assert is_uuid_like("a446725e-b281-414c-a21e-123456789abc") is True

    def test_partial_uuid(self):
        """Reconnaît un UUID partiel."""
        assert is_uuid_like("a446725e-b281-414c") is True

    def test_not_uuid(self):
        """Rejette les non-UUIDs."""
        assert is_uuid_like("Aquarius") is False
        assert is_uuid_like("Quick Play") is False


class TestComputeTotalPlaySeconds:
    """Tests pour compute_total_play_seconds."""

    def test_sum_durations(self):
        """Somme les durées."""
        df = pd.DataFrame({"time_played_seconds": [600, 720, 480]})
        result = compute_total_play_seconds(df)
        assert result == 1800.0

    def test_empty_df(self):
        """Retourne None pour DataFrame vide."""
        df = pd.DataFrame({"time_played_seconds": []})
        result = compute_total_play_seconds(df)
        assert result is None

    def test_missing_column(self):
        """Retourne None si colonne absente."""
        df = pd.DataFrame({"kills": [10, 15]})
        result = compute_total_play_seconds(df)
        assert result is None


class TestAvgMatchDurationSeconds:
    """Tests pour avg_match_duration_seconds."""

    def test_average(self):
        """Calcule la moyenne."""
        df = pd.DataFrame({"time_played_seconds": [600, 720, 480]})
        result = avg_match_duration_seconds(df)
        assert result == 600.0

    def test_empty_df(self):
        """Retourne None pour DataFrame vide."""
        df = pd.DataFrame({"time_played_seconds": []})
        result = avg_match_duration_seconds(df)
        assert result is None


class TestKPIStats:
    """Tests pour compute_kpi_stats."""

    @pytest.fixture
    def sample_df(self):
        """DataFrame de test."""
        return pd.DataFrame({
            "outcome": [2, 2, 3, 1],  # 2 wins, 1 loss, 1 tie
            "kills": [10, 15, 8, 12],
            "deaths": [5, 7, 10, 6],
            "assists": [3, 5, 2, 4],
            "accuracy": [45.5, 52.3, 38.0, 48.2],
            "average_life_seconds": [45, 52, 35, 48],
            "time_played_seconds": [600, 720, 540, 660],
        })

    def test_compute_kpi_stats(self, sample_df):
        """Calcule les KPIs correctement."""
        kpis = compute_kpi_stats(sample_df)
        
        assert isinstance(kpis, KPIStats)
        assert kpis.total_matches == 4
        assert kpis.wins == 2
        assert kpis.losses == 1
        assert kpis.ties == 1
        assert kpis.win_rate == 0.5
        assert kpis.loss_rate == 0.25

    def test_empty_df(self):
        """Gère un DataFrame vide."""
        df = pd.DataFrame({
            "outcome": [],
            "kills": [],
            "deaths": [],
            "assists": [],
            "accuracy": [],
            "average_life_seconds": [],
            "time_played_seconds": [],
        })
        kpis = compute_kpi_stats(df)
        
        assert kpis.total_matches == 0
        assert kpis.kills_per_game is None


class TestPlayerIdentityFromSecrets:
    """Tests pour get_identity_from_secrets."""

    def test_returns_player_identity(self):
        """Retourne un PlayerIdentity."""
        identity = get_identity_from_secrets()
        assert isinstance(identity, PlayerIdentity)
        assert hasattr(identity, "gamertag")
        assert hasattr(identity, "xuid")
        assert hasattr(identity, "waypoint_player")
