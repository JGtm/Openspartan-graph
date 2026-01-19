# -*- coding: utf-8 -*-
"""Gestion des profils multi-base de données.

Ce module permet de sauvegarder et charger plusieurs configurations
de bases de données (profils), chacune avec son chemin DB, XUID et
identifiant Waypoint.
"""
from __future__ import annotations

import json
import os
from typing import Any

__all__ = [
    "PROFILES_PATH",
    "load_profiles",
    "save_profiles",
    "list_local_dbs",
]

# Chemin du fichier de profils (à côté du script principal)
_DEFAULT_PROFILES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "db_profiles.json",
)
PROFILES_PATH = os.environ.get("OPENSPARTAN_PROFILES_PATH") or _DEFAULT_PROFILES_PATH


def load_profiles() -> dict[str, dict[str, str]]:
    """Charge les profils depuis le fichier JSON.

    Returns:
        Dictionnaire {nom_profil: {db_path, xuid, waypoint_player}}.
        Retourne un dict vide si le fichier n'existe pas ou est invalide.
    """
    if not os.path.exists(PROFILES_PATH):
        return {}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            obj: Any = json.load(f) or {}
    except Exception:
        return {}

    profiles = obj.get("profiles") if isinstance(obj, dict) else None
    if not isinstance(profiles, dict):
        return {}

    out: dict[str, dict[str, str]] = {}
    for name, v in profiles.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(v, dict):
            continue
        p: dict[str, str] = {}
        for k in ("db_path", "xuid", "waypoint_player"):
            val = v.get(k)
            if isinstance(val, str) and val.strip():
                p[k] = val.strip()
        if p:
            out[name.strip()] = p
    return out


def save_profiles(profiles: dict[str, dict[str, str]]) -> tuple[bool, str]:
    """Sauvegarde les profils dans le fichier JSON.

    Args:
        profiles: Dictionnaire des profils à sauvegarder.

    Returns:
        Tuple (succès, message_erreur). succès=True si OK, sinon message d'erreur.
    """
    try:
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump({"profiles": profiles}, f, ensure_ascii=False, indent=2)
        return True, ""
    except Exception as e:
        return False, f"Impossible d'écrire {PROFILES_PATH}: {e}"


def list_local_dbs() -> list[str]:
    """Liste les fichiers .db dans le dossier OpenSpartan.Workshop.

    Returns:
        Liste des chemins absolus vers les fichiers .db, triés par date
        de modification décroissante. Liste vide si aucun trouvé.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return []
    base = os.path.join(local, "OpenSpartan.Workshop", "data")
    if not os.path.isdir(base):
        return []
    try:
        dbs = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(".db")]
    except Exception:
        return []
    dbs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dbs
