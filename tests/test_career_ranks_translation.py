from __future__ import annotations

from src.ui.career_ranks import format_career_rank_label_fr


class TestCareerRanksTranslation:
    """Tests pour la traduction FR des rangs Career."""

    def test_translates_private_silver(self):
        """Traduction d'un rang classique: Silver Private 2 -> Argent Soldat 2."""
        assert (
            format_career_rank_label_fr(tier="Silver", title="Private", grade="2")
            == "Argent Soldat 2"
        )

    def test_translates_lance_corporal_bronze(self):
        """Traduction avec titre composé: Bronze Lance Corporal 3 -> Bronze Caporal suppléant 3."""
        assert (
            format_career_rank_label_fr(tier="Bronze", title="Lance Corporal", grade="3")
            == "Bronze Caporal suppléant 3"
        )

    def test_translates_lt_colonel(self):
        """Lt Colonel doit être traduit en Lieutenant-colonel."""
        assert (
            format_career_rank_label_fr(tier="Gold", title="Lt Colonel", grade="1")
            == "Or Lieutenant-colonel 1"
        )

    def test_translates_recruit_without_tier(self):
        """Le grade initial doit s'afficher sans tier: Recruit -> Recrue."""
        assert format_career_rank_label_fr(tier="Bronze", title="Recruit", grade=None) == "Recrue"

    def test_translates_hero(self):
        """Le grade final doit s'afficher en français: Hero -> Héros."""
        assert format_career_rank_label_fr(tier=None, title="Hero", grade=None) == "Héros"
