"""Gestion des alias XUID -> Gamertag."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict

from src.config import XUID_ALIASES_DEFAULT, get_aliases_file_path
from src.db.parsers import parse_xuid_input


def _safe_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def load_aliases_file(path: str | None = None) -> Dict[str, str]:
    """Charge les alias depuis un fichier JSON.
    
    Args:
        path: Chemin du fichier (default: xuid_aliases.json à la racine).
        
    Returns:
        Dictionnaire {xuid: gamertag}.
    """
    if path is None:
        path = get_aliases_file_path()

    return dict(_load_aliases_cached(path, _safe_mtime(path)))


@lru_cache(maxsize=16)
def _load_aliases_cached(path: str, mtime: float | None) -> Dict[str, str]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        cleaned: Dict[str, str] = {}
        for k, v in raw.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                cleaned[kk] = vv
        return cleaned
    except Exception:
        return {}


def save_aliases_file(aliases: Dict[str, str], path: str | None = None) -> None:
    """Sauvegarde les alias dans un fichier JSON.
    
    Args:
        aliases: Dictionnaire {xuid: gamertag}.
        path: Chemin du fichier (default: xuid_aliases.json à la racine).
    """
    if path is None:
        path = get_aliases_file_path()
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(aliases.items())), f, ensure_ascii=False, indent=2)

    # Invalide le cache (le contenu a changé)
    _load_aliases_cached.cache_clear()


def get_xuid_aliases() -> Dict[str, str]:
    """Retourne les alias fusionnés (par défaut + fichier).
    
    Returns:
        Dictionnaire {xuid: gamertag}.
    """
    merged = dict(XUID_ALIASES_DEFAULT)
    merged.update(load_aliases_file())
    return merged


def display_name_from_xuid(xuid: str) -> str:
    """Convertit un XUID en nom d'affichage.
    
    Args:
        xuid: XUID du joueur.
        
    Returns:
        Gamertag si un alias existe, sinon le XUID tel quel.
    """
    raw = str(xuid or "").strip()
    # SPNKr/OpenSpartan stockent souvent l'identifiant sous forme "xuid(2533...)".
    # Normaliser ici permet aux alias (clés = XUID numérique) de fonctionner partout.
    normalized = parse_xuid_input(raw) or raw
    return get_xuid_aliases().get(normalized, normalized)
