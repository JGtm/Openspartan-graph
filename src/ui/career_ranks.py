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
    print(f"{info.full_label}: {info.xp_required} XP")
    
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
        """Retourne le label complet du rang (ex: 'Gold Lance Corporal III')."""
        parts = []
        if self.subtitle:
            parts.append(self.subtitle)
        parts.append(self.title)
        if self.tier:
            parts.append(self.tier)
        return " ".join(parts)
    
    @property
    def display_label(self) -> str:
        """Retourne un label compact (ex: 'Lance Corporal Gold III')."""
        if self.subtitle and self.tier:
            return f"{self.title} {self.subtitle} {self.tier}"
        elif self.subtitle:
            return f"{self.title} {self.subtitle}"
        return self.title


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
