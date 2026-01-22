"""Gestion des paramètres utilisateur (persistés).

Objectif:
- Déplacer les "paramètres" hors sidebar (onglet Paramètres)
- Permettre une exécution sur NAS/Docker avec un fichier de config monté

Le chemin est configurable via OPENSPARTAN_SETTINGS_PATH.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


def get_settings_path() -> str:
    override = os.environ.get("OPENSPARTAN_SETTINGS_PATH")
    if override and str(override).strip():
        return str(override).strip()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(repo_root, "app_settings.json")


@dataclass
class AppSettings:
    # Médias
    media_enabled: bool = True
    media_screens_dir: str = ""
    media_videos_dir: str = ""
    media_tolerance_minutes: int = 3

    # UX
    refresh_clears_caches: bool = False

    # Source
    prefer_spnkr_db_if_available: bool = True

    # SPNKr (API → DB) : rafraîchissement automatique
    spnkr_refresh_on_start: bool = True
    spnkr_refresh_on_manual_refresh: bool = True
    spnkr_refresh_match_type: str = "matchmaking"  # all|matchmaking|custom|local
    spnkr_refresh_max_matches: int = 200
    spnkr_refresh_rps: int = 3
    spnkr_refresh_with_highlight_events: bool = False

    # Fichiers (overrides optionnels)
    aliases_path: str = ""
    profiles_path: str = ""

    # Profil joueur (bannière/rang) — aucun accès réseau implicite
    profile_assets_download_enabled: bool = False
    profile_assets_auto_refresh_hours: int = 24  # 0 = désactivé (ne rafraîchit pas)

    # Profil joueur (auto depuis API Waypoint via SPNKr) — opt-in
    profile_api_enabled: bool = False
    profile_api_auto_refresh_hours: int = 6  # cache disque des URLs/paths d'apparence

    profile_banner: str = ""  # URL https://... ou chemin local
    profile_emblem: str = ""  # URL https://... ou chemin local
    profile_backdrop: str = ""  # URL https://... ou chemin local
    profile_nameplate: str = ""  # URL https://... ou chemin local
    profile_service_tag: str = ""  # ex: "JGTM"
    profile_id_badge_text_color: str = ""  # ex: "#FFFFFF"
    profile_rank_label: str = ""  # ex: "Diamond III" / "Héros" / etc.
    profile_rank_subtitle: str = ""  # ex: "CSR 1540" / "Saison 5" / etc.


def _coerce_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return default
        x = int(v)
        return x
    except Exception:
        return default


def load_settings() -> AppSettings:
    path = get_settings_path()
    if not os.path.exists(path):
        return AppSettings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f) or {}
    except Exception:
        return AppSettings()

    if not isinstance(obj, dict):
        return AppSettings()

    s = AppSettings()
    s.media_enabled = _coerce_bool(obj.get("media_enabled"), s.media_enabled)
    s.media_screens_dir = str(obj.get("media_screens_dir") or "").strip()
    s.media_videos_dir = str(obj.get("media_videos_dir") or "").strip()
    s.media_tolerance_minutes = max(0, _coerce_int(obj.get("media_tolerance_minutes"), s.media_tolerance_minutes))
    s.refresh_clears_caches = _coerce_bool(obj.get("refresh_clears_caches"), s.refresh_clears_caches)
    s.prefer_spnkr_db_if_available = _coerce_bool(
        obj.get("prefer_spnkr_db_if_available"), s.prefer_spnkr_db_if_available
    )

    s.spnkr_refresh_on_start = _coerce_bool(obj.get("spnkr_refresh_on_start"), s.spnkr_refresh_on_start)
    s.spnkr_refresh_on_manual_refresh = _coerce_bool(
        obj.get("spnkr_refresh_on_manual_refresh"), s.spnkr_refresh_on_manual_refresh
    )
    mt = str(obj.get("spnkr_refresh_match_type") or s.spnkr_refresh_match_type).strip().lower()
    if mt in {"all", "matchmaking", "custom", "local"}:
        s.spnkr_refresh_match_type = mt
    s.spnkr_refresh_max_matches = max(1, _coerce_int(obj.get("spnkr_refresh_max_matches"), s.spnkr_refresh_max_matches))
    s.spnkr_refresh_rps = max(1, _coerce_int(obj.get("spnkr_refresh_rps"), s.spnkr_refresh_rps))
    s.spnkr_refresh_with_highlight_events = _coerce_bool(
        obj.get("spnkr_refresh_with_highlight_events"), s.spnkr_refresh_with_highlight_events
    )

    s.aliases_path = str(obj.get("aliases_path") or "").strip()
    s.profiles_path = str(obj.get("profiles_path") or "").strip()

    s.profile_assets_download_enabled = _coerce_bool(
        obj.get("profile_assets_download_enabled"), s.profile_assets_download_enabled
    )
    s.profile_assets_auto_refresh_hours = max(
        0, _coerce_int(obj.get("profile_assets_auto_refresh_hours"), s.profile_assets_auto_refresh_hours)
    )

    s.profile_api_enabled = _coerce_bool(obj.get("profile_api_enabled"), s.profile_api_enabled)
    s.profile_api_auto_refresh_hours = max(
        0, _coerce_int(obj.get("profile_api_auto_refresh_hours"), s.profile_api_auto_refresh_hours)
    )

    s.profile_banner = str(obj.get("profile_banner") or "").strip()
    s.profile_emblem = str(obj.get("profile_emblem") or "").strip()
    s.profile_backdrop = str(obj.get("profile_backdrop") or "").strip()
    s.profile_nameplate = str(obj.get("profile_nameplate") or "").strip()
    s.profile_service_tag = str(obj.get("profile_service_tag") or "").strip()
    s.profile_id_badge_text_color = str(obj.get("profile_id_badge_text_color") or "").strip()
    s.profile_rank_label = str(obj.get("profile_rank_label") or "").strip()
    s.profile_rank_subtitle = str(obj.get("profile_rank_subtitle") or "").strip()
    return s


def save_settings(settings: AppSettings) -> tuple[bool, str]:
    path = get_settings_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
        return True, ""
    except Exception as e:
        return False, f"Impossible d'écrire {path}: {e}"
