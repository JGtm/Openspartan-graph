"""Cache disque pour les données de profil Halo Waypoint.

Gestion du cache local (JSON) pour éviter les appels API répétés.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileAppearance:
    """Apparence du joueur Halo Infinite."""

    service_tag: str | None = None
    emblem_image_url: str | None = None
    backdrop_image_url: str | None = None
    nameplate_image_url: str | None = None
    rank_label: str | None = None
    rank_subtitle: str | None = None
    rank_image_url: str | None = None


def _repo_root() -> Path:
    """Retourne la racine du repository."""
    return Path(__file__).resolve().parents[2]


def get_profile_api_cache_dir() -> Path:
    """Retourne le répertoire de cache pour les données de profil."""
    return _repo_root() / "data" / "cache" / "profile_api"


def _cache_path(xuid: str) -> Path:
    """Chemin du fichier cache pour un XUID donné."""
    safe = "".join(ch for ch in str(xuid or "") if ch.isdigit())
    return get_profile_api_cache_dir() / f"appearance_{safe}.json"


def _xuid_cache_path(gamertag: str) -> Path:
    """Chemin du fichier cache pour un gamertag donné."""
    gt = str(gamertag or "").strip().lower()
    h = hashlib.sha256(gt.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return get_profile_api_cache_dir() / f"xuid_gt_{h}.json"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    """Lecture sécurisée d'un fichier JSON."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Écriture sécurisée d'un fichier JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_fresh(fetched_at: float | None, *, refresh_hours: int) -> bool:
    """Vérifie si le cache est encore frais."""
    # Convention UI: 0 = désactivé => re-fetch à chaque run (donc jamais "fresh").
    if refresh_hours <= 0:
        return False
    if fetched_at is None:
        return False
    try:
        age_s = time.time() - float(fetched_at)
    except Exception:
        return False
    return age_s < float(refresh_hours) * 3600.0


def load_cached_appearance(xuid: str, *, refresh_hours: int) -> ProfileAppearance | None:
    """Charge l'apparence depuis le cache disque.
    
    Args:
        xuid: XUID du joueur.
        refresh_hours: Durée de validité du cache en heures.
        
    Returns:
        ProfileAppearance si le cache est valide, None sinon.
    """
    cp = _cache_path(xuid)
    data = _safe_read_json(cp)
    if not data:
        return None
    fetched_at = data.get("fetched_at")
    if not _is_fresh(fetched_at if isinstance(fetched_at, (int, float)) else None, refresh_hours=refresh_hours):
        return None

    return ProfileAppearance(
        service_tag=(str(data.get("service_tag") or "").strip() or None),
        emblem_image_url=(str(data.get("emblem_image_url") or "").strip() or None),
        backdrop_image_url=(str(data.get("backdrop_image_url") or "").strip() or None),
        nameplate_image_url=(str(data.get("nameplate_image_url") or "").strip() or None),
        rank_label=(str(data.get("rank_label") or "").strip() or None),
        rank_subtitle=(str(data.get("rank_subtitle") or "").strip() or None),
        rank_image_url=(str(data.get("rank_image_url") or "").strip() or None),
    )


def save_cached_appearance(xuid: str, appearance: ProfileAppearance) -> None:
    """Sauvegarde l'apparence dans le cache disque."""
    cp = _cache_path(xuid)
    data = {
        "fetched_at": time.time(),
        "service_tag": appearance.service_tag,
        "emblem_image_url": appearance.emblem_image_url,
        "backdrop_image_url": appearance.backdrop_image_url,
        "nameplate_image_url": appearance.nameplate_image_url,
        "rank_label": appearance.rank_label,
        "rank_subtitle": appearance.rank_subtitle,
        "rank_image_url": appearance.rank_image_url,
    }
    _safe_write_json(cp, data)


def load_cached_xuid_for_gamertag(gamertag: str, *, refresh_hours: int) -> str | None:
    """Charge le XUID depuis le cache pour un gamertag donné.
    
    Args:
        gamertag: Gamertag du joueur.
        refresh_hours: Durée de validité du cache en heures.
        
    Returns:
        XUID si le cache est valide, None sinon.
    """
    cp = _xuid_cache_path(gamertag)
    data = _safe_read_json(cp)
    if not data:
        return None
    fetched_at = data.get("fetched_at")
    if not _is_fresh(fetched_at if isinstance(fetched_at, (int, float)) else None, refresh_hours=refresh_hours):
        return None
    xuid = str(data.get("xuid") or "").strip()
    return xuid if xuid.isdigit() else None


def save_cached_xuid(gamertag: str, xuid: str) -> None:
    """Sauvegarde le mapping gamertag → xuid dans le cache disque."""
    cp = _xuid_cache_path(gamertag)
    data = {
        "fetched_at": time.time(),
        "gamertag": gamertag,
        "xuid": xuid,
    }
    _safe_write_json(cp, data)
