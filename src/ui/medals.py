# -*- coding: utf-8 -*-
"""Gestion des médailles : labels, icônes et affichage en grille.

Ce module centralise les fonctions liées aux médailles Halo Infinite :
- Chargement des fichiers de traduction (FR/EN)
- Récupération des icônes depuis le cache OpenSpartan.Workshop
- Affichage d'une grille de médailles dans Streamlit
"""
from __future__ import annotations

import json
import os
from typing import Optional

import streamlit as st

__all__ = [
    "load_medal_name_maps",
    "medal_has_known_label",
    "get_medals_cache_dir",
    "medal_label",
    "medal_icon_path",
    "render_medals_grid",
]


@st.cache_data(show_spinner=False)
def load_medal_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Charge les labels de médailles.

    Returns:
        Tuple (fr_map, en_map) où chaque map est {str(NameId): "Label"}.

    Notes:
        - Priorité au fichier FR: static/medals/medals_fr.json
        - Fallback: static/medals/medals_en.json

    Formats acceptés:
        {"<NameId>": "Nom"}
        {"<NameId>": {"name": "...", "label": "...", "fr": "...", "name_fr": "..."}}
    """

    def _load(path: str) -> dict[str, str]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}

        out: dict[str, str] = {}
        for k, v in raw.items():
            kid = str(k)
            if isinstance(v, str) and v.strip():
                out[kid] = v.strip()
                continue
            if isinstance(v, dict):
                for key in ("fr", "name_fr", "nameFr", "label_fr", "labelFr", "label", "name"):
                    val = v.get(key)
                    if isinstance(val, str) and val.strip():
                        out[kid] = val.strip()
                        break
        return out

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fr_map = _load(os.path.join(repo_root, "static", "medals", "medals_fr.json"))
    # Fallback EN : nouveau fichier généré (et compat avec un éventuel ancien medals.json)
    en_map = _load(os.path.join(repo_root, "static", "medals", "medals_en.json"))
    if not en_map:
        en_map = _load(os.path.join(repo_root, "static", "medals", "medals.json"))
    return fr_map, en_map


def get_local_medals_icons_dir() -> str:
    """Retourne le dossier d'icônes médailles embarquées dans le repo.

    Returns:
        Chemin absolu vers static/medals/icons
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "static", "medals", "icons")


def medal_has_known_label(nid: int) -> bool:
    """Vérifie si une médaille a un label connu (FR ou EN).

    Args:
        nid: NameId de la médaille.

    Returns:
        True si un label existe, False sinon.
    """
    fr_map, en_map = load_medal_name_maps()
    key = str(int(nid))
    return key in fr_map or key in en_map


def get_medals_cache_dir() -> str:
    """Retourne le dossier des icônes médailles (OpenSpartan.Workshop).

    Le chemin peut être surchargé via la variable d'environnement
    OPENSPARTAN_MEDALS_CACHE.

    Returns:
        Chemin absolu vers le dossier de cache des médailles.
    """
    override = os.environ.get("OPENSPARTAN_MEDALS_CACHE")
    if override:
        return override

    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        return os.path.join(localappdata, "OpenSpartan.Workshop", "imagecache", "medals")

    # fallback (utile hors Windows, mais probablement vide)
    return os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "OpenSpartan.Workshop", "imagecache", "medals"
    )


def medal_label(nid: int) -> str:
    """Retourne le label d'une médaille (FR prioritaire, sinon EN, sinon générique).

    Args:
        nid: NameId de la médaille.

    Returns:
        Label de la médaille ou "Médaille #<nid>" si inconnu.
    """
    fr_map, en_map = load_medal_name_maps()
    key = str(int(nid))
    return fr_map.get(key) or en_map.get(key) or f"Médaille #{nid}"


def medal_icon_path(nid: int) -> Optional[str]:
    """Retourne le chemin de l'icône PNG d'une médaille si elle existe.

    Args:
        nid: NameId de la médaille.

    Returns:
        Chemin absolu vers l'icône ou None si introuvable.
    """
    local_p = os.path.join(get_local_medals_icons_dir(), f"{int(nid)}.png")
    if os.path.exists(local_p):
        return local_p

    p = os.path.join(get_medals_cache_dir(), f"{int(nid)}.png")
    return p if os.path.exists(p) else None


def render_medals_grid(
    medals: list[dict[str, int]],
    cols_per_row: int = 8,
    deltas: dict[int, int] | None = None,
) -> None:
    """Affiche une grille de médailles dans Streamlit.

    Args:
        medals: Liste de dicts avec 'name_id' et 'count'.
        cols_per_row: Nombre de colonnes (3-12, défaut 8).
        deltas: Dict {medal_id: delta_count} pour afficher +XXX à côté du compteur.
    """
    if not medals:
        st.info("Aucune médaille.")
        return

    # Note: on n'affiche plus de warning pour les médailles inconnues
    # (certaines comme #590706932 sont des médailles internes/test à ignorer)

    local_dir = get_local_medals_icons_dir()
    cache_dir = get_medals_cache_dir()
    has_local = os.path.isdir(local_dir)
    has_cache = os.path.isdir(cache_dir)
    if not (has_local or has_cache):
        st.caption(
            "Icônes de médailles introuvables. "
            "Utilise scripts/sync_medal_icons.py pour copier les PNG en local, "
            "ou définis OPENSPARTAN_MEDALS_CACHE / installe OpenSpartan.Workshop."
        )

    cols_per_row = max(3, min(int(cols_per_row), 12))
    cols = st.columns(cols_per_row)
    for i, m in enumerate(medals):
        col = cols[i % cols_per_row]
        nid = int(m.get("name_id", 0))
        cnt = int(m.get("count", 0))
        name = medal_label(nid)
        icon = medal_icon_path(nid)

        if icon:
            col.image(icon, width="stretch")
        else:
            col.markdown(
                f"<div class='os-medal-missing'>#{nid}</div>",
                unsafe_allow_html=True,
            )

        # Afficher le delta si fourni
        delta_html = ""
        if deltas is not None and nid in deltas:
            delta_val = deltas[nid]
            if delta_val > 0:
                delta_html = f" <span style='color: #4CAF50; font-weight: bold;'>+{delta_val}</span>"
        
        col.markdown(
            "<div class='os-medal-caption'>"
            + "<div class='os-medal-name'>" + name + "</div>"
            + "<div class='os-medal-count'>x" + str(cnt) + delta_html + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )
