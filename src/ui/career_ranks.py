"""Helper pour gérer les Career Ranks Halo Infinite.

Ce module charge les métadonnées des rangs depuis le cache local
et fournit des fonctions pour afficher le rang d'un joueur.

Données requises:
- data/cache/career_ranks_metadata.json (métadonnées des 272 rangs)
- data/cache/career_ranks/ (icônes PNG optionnelles)

Usage:
    from src.ui.career_ranks import get_rank_info, get_rank_icon_path
    
    # Si on connaît le numéro de rang du joueur (1-272):
    info = get_rank_info(150)
    print(f"{info.full_label_fr}: {info.xp_required} XP")
    
    # Récupérer le chemin local de l'icône:
    icon_path = get_rank_icon_path(150)
    if icon_path:
        st.image(str(icon_path), width=64)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_CAREER_RANK_TIER_FR: dict[str, str] = {
    "Bronze": "Bronze",
    "Silver": "Argent",
    "Gold": "Or",
    "Platinum": "Platine",
    "Diamond": "Diamant",
    "Onyx": "Onyx",
}


_CAREER_RANK_TITLE_FR: dict[str, str] = {
    "Recruit": "Recrue",
    "Cadet": "Cadet",
    "Private": "Soldat",
    "Lance Corporal": "Caporal suppléant",
    "Corporal": "Caporal",
    "Sergeant": "Sergent",
    "Staff Sergeant": "Sergent-chef",
    "Gunnery Sergeant": "Sergent d'artillerie",
    "Master Sergeant": "Adjudant",
    "Lieutenant": "Lieutenant",
    "Captain": "Capitaine",
    "Major": "Lieutenant-major",
    "Lt Colonel": "Lieutenant-colonel",
    "Colonel": "Colonel",
    "Brigadier General": "Général de brigade",
    "General": "Général",
    "Hero": "Héros",
}


def format_career_rank_label_fr(*, tier: str | None, title: str | None, grade: str | None) -> str:
    """Formate un libellé de rang Career en français.

    Args:
        tier: Tier/type de rang (ex: "Silver", "Gold")
        title: Titre du rang (ex: "Private", "Lt Colonel")
        grade: Sous-grade ("1"/"2"/"3") ou None

    Returns:
        Libellé FR (ex: "Argent Soldat 2", "Or Lieutenant-colonel 1", "Héros").
    """
    raw_title = (title or "").strip()
    raw_tier = (tier or "").strip()
    raw_grade = (grade or "").strip()

    title_fr = _CAREER_RANK_TITLE_FR.get(raw_title, raw_title)
    tier_fr = _CAREER_RANK_TIER_FR.get(raw_tier, raw_tier)

    # Cas spéciaux: grade initial et grade final
    if title_fr in ("Recrue", "Héros"):
        return title_fr

    parts: list[str] = []
    if tier_fr:
        parts.append(tier_fr)
    if title_fr:
        parts.append(title_fr)
    if raw_grade:
        parts.append(raw_grade)
    return " ".join(parts).strip()


@dataclass(frozen=True)
class CareerRankInfo:
    """Informations sur un rang Career."""
    
    rank_number: int
    title: str
    subtitle: str | None
    tier: str | None
    xp_required: int
    icon_path_remote: str  # Chemin relatif pour l'API CMS
    
    @property
    def full_label(self) -> str:
        """Retourne le label complet du rang (ex: 'Gold Lance Corporal 3')."""
        parts = []
        if self.subtitle:
            parts.append(self.subtitle)
        parts.append(self.title)
        if self.tier:
            parts.append(self.tier)
        return " ".join(parts)

    @property
    def full_label_fr(self) -> str:
        """Retourne le label complet du rang en français (ex: 'Or Caporal suppléant 1')."""
        return format_career_rank_label_fr(tier=self.subtitle, title=self.title, grade=self.tier)
    
    @property
    def display_label(self) -> str:
        """Retourne un label compact (ex: 'Lance Corporal Gold 3')."""
        if self.subtitle and self.tier:
            return f"{self.title} {self.subtitle} {self.tier}"
        elif self.subtitle:
            return f"{self.title} {self.subtitle}"
        return self.title

    @property
    def display_label_fr(self) -> str:
        """Retourne un label compact en français (ex: 'Caporal suppléant Bronze 1')."""
        title_fr = _CAREER_RANK_TITLE_FR.get(self.title, self.title)
        tier_fr = _CAREER_RANK_TIER_FR.get(self.subtitle or "", self.subtitle or "")
        if tier_fr and self.tier:
            return f"{title_fr} {tier_fr} {self.tier}".strip()
        if tier_fr:
            return f"{title_fr} {tier_fr}".strip()
        return title_fr.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_metadata_path() -> Path:
    return _repo_root() / "data" / "cache" / "career_ranks_metadata.json"


def _get_icons_dir() -> Path:
    return _repo_root() / "data" / "cache" / "career_ranks"


@lru_cache(maxsize=1)
def _load_ranks_metadata() -> dict[str, Any]:
    """Charge les métadonnées des rangs depuis le cache."""
    path = _get_metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _build_ranks_lookup() -> dict[int, CareerRankInfo]:
    """Construit un dict de lookup rank_number -> CareerRankInfo."""
    metadata = _load_ranks_metadata()
    ranks = metadata.get("Ranks", [])
    
    lookup: dict[int, CareerRankInfo] = {}
    
    for rank_data in ranks:
        rank_num = rank_data.get("Rank", 0)
        
        title_obj = rank_data.get("RankTitle", {})
        title = title_obj.get("value", "") if isinstance(title_obj, dict) else str(title_obj or "")
        
        subtitle_obj = rank_data.get("RankSubTitle", {})
        subtitle = subtitle_obj.get("value", "") if isinstance(subtitle_obj, dict) else str(subtitle_obj or "")
        
        tier_obj = rank_data.get("RankTier", {})
        tier = tier_obj.get("value", "") if isinstance(tier_obj, dict) else str(tier_obj or "")
        
        icon_large = rank_data.get("RankLargeIcon", "")
        xp_required = rank_data.get("XpRequiredForRank", 0)
        
        info = CareerRankInfo(
            rank_number=rank_num,
            title=title,
            subtitle=subtitle if subtitle else None,
            tier=tier if tier else None,
            xp_required=xp_required,
            icon_path_remote=icon_large,
        )
        lookup[rank_num] = info
    
    return lookup


def get_rank_info(rank_number: int) -> CareerRankInfo | None:
    """Retourne les informations d'un rang par son numéro (1-272).
    
    Args:
        rank_number: Numéro du rang (1 = Recruit, 272 = Hero)
    
    Returns:
        CareerRankInfo ou None si le rang n'existe pas
    """
    lookup = _build_ranks_lookup()
    return lookup.get(rank_number)


def get_all_ranks() -> list[CareerRankInfo]:
    """Retourne la liste de tous les rangs triés par numéro."""
    lookup = _build_ranks_lookup()
    return sorted(lookup.values(), key=lambda r: r.rank_number)


def get_rank_icon_path(rank_number: int) -> Path | None:
    """Retourne le chemin local de l'icône d'un rang si elle existe.
    
    Args:
        rank_number: Numéro du rang (1-272)
    
    Returns:
        Path vers le fichier PNG ou None si non téléchargé
    """
    icons_dir = _get_icons_dir()
    icon_path = icons_dir / f"rank_{rank_number:03d}_large.png"
    
    if icon_path.exists():
        return icon_path
    return None


def get_rank_icon_url(rank_number: int) -> str | None:
    """Retourne l'URL CMS pour télécharger l'icône d'un rang.
    
    Args:
        rank_number: Numéro du rang (1-272)
    
    Returns:
        URL complète ou None si le rang n'existe pas
    """
    info = get_rank_info(rank_number)
    if not info or not info.icon_path_remote:
        return None
    
    return f"https://gamecms-hacs.svc.halowaypoint.com/hi/images/file/{info.icon_path_remote}"


def get_rank_for_xp(total_xp: int) -> CareerRankInfo | None:
    """Détermine le rang correspondant à un total d'XP.
    
    Args:
        total_xp: Total d'XP Career du joueur
    
    Returns:
        Le rang le plus élevé atteint avec cet XP
    """
    ranks = get_all_ranks()
    
    current_rank = None
    for rank in ranks:
        if total_xp >= rank.xp_required:
            current_rank = rank
        else:
            break
    
    return current_rank


def count_cached_icons() -> int:
    """Compte le nombre d'icônes téléchargées en local."""
    icons_dir = _get_icons_dir()
    if not icons_dir.exists():
        return 0
    return len(list(icons_dir.glob("rank_*_large.png")))


def is_metadata_available() -> bool:
    """Vérifie si les métadonnées des rangs sont disponibles."""
    return _get_metadata_path().exists()
