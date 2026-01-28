from __future__ import annotations

import pandas as pd

from src.ui.pages.session_compare import compute_similar_sessions_average


class TestSessionCompareHistoricalAverageByCategory:
    """Tests pour le filtrage de la moyenne historique par catégorie dominante."""

    def test_compute_similar_sessions_average_filters_by_mode_category(self) -> None:
        """Ne conserve que les sessions dont la catégorie dominante correspond."""
        # Session 1: Assassin (solo)
        # Session 2: BTB (solo)
        # Session 3: Assassin (solo)
        df = pd.DataFrame(
            [
                {
                    "session_id": 1,
                    "pair_name": "Arena:Slayer on Aquarius",
                    "is_with_friends": 0,
                    "kills": 10,
                    "deaths": 5,
                    "assists": 2,
                    "outcome": 2,
                    "average_life_seconds": 30.0,
                    "accuracy": 45.0,
                },
                {
                    "session_id": 1,
                    "pair_name": "Community:Slayer on Aquarius",
                    "is_with_friends": 0,
                    "kills": 8,
                    "deaths": 4,
                    "assists": 1,
                    "outcome": 3,
                    "average_life_seconds": 25.0,
                    "accuracy": 50.0,
                },
                {
                    "session_id": 2,
                    "pair_name": "BTB:CTF on Highpower",
                    "is_with_friends": 0,
                    "kills": 20,
                    "deaths": 10,
                    "assists": 5,
                    "outcome": 2,
                    "average_life_seconds": 40.0,
                    "accuracy": 55.0,
                },
                {
                    "session_id": 3,
                    "pair_name": "Arena:Slayer on Aquarius",
                    "is_with_friends": 0,
                    "kills": 12,
                    "deaths": 6,
                    "assists": 3,
                    "outcome": 2,
                    "average_life_seconds": 35.0,
                    "accuracy": 60.0,
                },
            ]
        )

        hist = compute_similar_sessions_average(
            df,
            is_with_friends=False,
            exclude_session_ids=None,
            same_friends_xuids=None,
            mode_category="Assassin",
        )

        # Doit garder session_id 1 et 3 (2 sessions)
        assert hist.get("session_count") == 2
        assert hist.get("kd_ratio") is not None
        assert hist.get("win_rate") is not None
        assert hist.get("accuracy") is not None

        # Contrôle: catégorie BTB -> uniquement session_id 2
        hist_btb = compute_similar_sessions_average(
            df,
            is_with_friends=False,
            exclude_session_ids=None,
            same_friends_xuids=None,
            mode_category="BTB",
        )
        assert hist_btb.get("session_count") == 1

    def test_compute_similar_sessions_average_without_is_with_friends(self) -> None:
        """Fallback: si is_with_friends est absent, comparer sur toutes les sessions."""
        df = pd.DataFrame(
            [
                {
                    "session_id": 10,
                    "pair_name": "Arena:Slayer on Aquarius",
                    "kills": 10,
                    "deaths": 5,
                    "assists": 2,
                    "outcome": 2,
                    "average_life_seconds": 30.0,
                    "accuracy": 45.0,
                },
                {
                    "session_id": 11,
                    "pair_name": "BTB:CTF on Highpower",
                    "kills": 20,
                    "deaths": 10,
                    "assists": 5,
                    "outcome": 2,
                    "average_life_seconds": 40.0,
                    "accuracy": 55.0,
                },
            ]
        )

        hist = compute_similar_sessions_average(
            df,
            is_with_friends=False,
            exclude_session_ids=None,
            same_friends_xuids=None,
            mode_category="BTB",
        )
        assert hist.get("session_count") == 1
        assert hist.get("win_rate") is not None
