"""Composant de filtre par checkboxes dans un expander.

Remplace les selectbox par des listes de checkboxes groupées,
plus pratiques quand il y a beaucoup de valeurs à filtrer.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Optional

import streamlit as st


def _set_filter_all(session_key: str, options: list[str]) -> None:
    """Callback pour sélectionner toutes les options."""
    st.session_state[session_key] = set(options)
    # Supprimer les clés individuelles des checkboxes pour forcer leur reset
    prefixes = (f"{session_key}_cb_", f"{session_key}_cat_", f"{session_key}_mode_")
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(prefixes)]
    for k in keys_to_delete:
        del st.session_state[k]


def _set_filter_none(session_key: str) -> None:
    """Callback pour désélectionner toutes les options."""
    st.session_state[session_key] = set()
    # Supprimer les clés individuelles des checkboxes pour forcer leur reset
    prefixes = (f"{session_key}_cb_", f"{session_key}_cat_", f"{session_key}_mode_")
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(prefixes)]
    for k in keys_to_delete:
        del st.session_state[k]


# Cache pour les catégories de modes
_MODE_CATEGORIES_CACHE: dict[str, str] | None = None


def _load_mode_categories() -> dict[str, str]:
    """Charge le mapping mode -> catégorie depuis le JSON.
    
    Returns:
        Dict {mode_fr: category} ex: {"Arène : Assassin": "Arena"}
    """
    global _MODE_CATEGORIES_CACHE
    if _MODE_CATEGORIES_CACHE is not None:
        return _MODE_CATEGORIES_CACHE
    
    json_path = Path(__file__).parent.parent.parent.parent / "Playlist_modes_translations.json"
    if not json_path.exists():
        _MODE_CATEGORIES_CACHE = {}
        return _MODE_CATEGORIES_CACHE
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        _MODE_CATEGORIES_CACHE = {}
        for mode in data.get("modes", []):
            fr_name = mode.get("fr", "")
            category = mode.get("category", "Autre")
            if fr_name and fr_name not in _MODE_CATEGORIES_CACHE:
                _MODE_CATEGORIES_CACHE[fr_name] = category
        return _MODE_CATEGORIES_CACHE
    except Exception:
        _MODE_CATEGORIES_CACHE = {}
        return _MODE_CATEGORIES_CACHE


# Mapping préfixe -> catégorie pour inférer la catégorie des modes non traduits
# Catégories simplifiées: Assassin, Fiesta, BTB, Ranked, Firefight, Other
PREFIX_TO_CATEGORY: dict[str, str] = {
    # Assassin (Arena, Tactical, Community, etc.)
    "Arena": "Assassin",
    "Arène": "Assassin",
    "Tactical": "Assassin",
    "Tactique": "Assassin",
    "Community": "Assassin",
    "Communauté": "Assassin",
    "Assault": "Assassin",
    # Fiesta (Super Fiesta, Husky Raid, Castle Wars, etc.)
    "Fiesta": "Fiesta",
    "Super Fiesta": "Fiesta",
    "Husky Raid": "Fiesta",
    "Super Husky Raid": "Fiesta",
    "Castle Wars": "Fiesta",
    # BTB
    "BTB": "BTB",
    "BTB Heavies": "BTB",
    # Ranked
    "Ranked": "Ranked",
    "Classé": "Ranked",
    # Firefight
    "Firefight": "Firefight",
    "Gruntpocalypse": "Firefight",
    # Autre
    "Event": "Other",
}


def _infer_category(mode_name: str) -> str:
    """Infère la catégorie d'un mode à partir de son préfixe ou contenu.
    
    Catégories: Assassin, Fiesta, BTB, Ranked, Firefight, Other
    
    Exemples:
        "Arène : Assassin" -> "Assassin"
        "BTB : CTF" -> "BTB"
        "Super Fiesta : Assassin" -> "Fiesta"
        "Communauté : Fiesta Assassin" -> "Fiesta" (contient Fiesta)
    """
    # Détecter les modes Fiesta par leur contenu (pas seulement le préfixe)
    mode_lower = mode_name.lower()
    if "fiesta" in mode_lower or "husky raid" in mode_lower or "castle wars" in mode_lower:
        return "Fiesta"
    
    # Extraire le préfixe (avant ":" ou " : ")
    prefix = None
    if " : " in mode_name:
        prefix = mode_name.split(" : ", 1)[0].strip()
    elif ":" in mode_name:
        prefix = mode_name.split(":", 1)[0].strip()
    
    if prefix:
        # Vérifier si le préfixe correspond à une catégorie connue
        if prefix in PREFIX_TO_CATEGORY:
            return PREFIX_TO_CATEGORY[prefix]
        # Essayer en ignorant la casse
        for p, cat in PREFIX_TO_CATEGORY.items():
            if prefix.lower() == p.lower():
                return cat
    
    return "Other"


# Traduction des catégories en français
CATEGORY_FR: dict[str, str] = {
    "Assassin": "Assassin",
    "Fiesta": "Fiesta",
    "BTB": "Grande bataille en équipe",
    "Ranked": "Classé",
    "Firefight": "Baptême du feu",
    "Other": "Autre",
}


def _translate_category(cat: str) -> str:
    """Traduit une catégorie en français."""
    return CATEGORY_FR.get(cat, cat)


def render_checkbox_filter(
    *,
    label: str,
    options: list[str],
    session_key: str,
    default_unchecked: Optional[set[str]] = None,
    show_select_buttons: bool = True,
    expanded: bool = False,
) -> set[str]:
    """Affiche un expander avec checkboxes pour filtrer une liste de valeurs.

    Utilise un système de "version" pour les boutons Tout/Aucun :
    - Chaque clic incrémente une version
    - Les checkboxes ont une clé qui inclut la version
    - Ainsi, après Tout/Aucun, de nouvelles checkboxes sont créées
      avec le bon état initial, sans rerun ni conflit de state.

    Args:
        label: Titre de l'expander (ex: "Playlists").
        options: Liste des valeurs disponibles à cocher.
        session_key: Clé session_state pour persister la sélection.
        default_unchecked: Valeurs décochées par défaut (ex: Firefight).
            Si None, tout est coché par défaut.
        show_select_buttons: Afficher les boutons Tout/Aucun.
        expanded: Si l'expander est ouvert par défaut.

    Returns:
        Ensemble des valeurs sélectionnées (cochées).
    """
    if not options:
        return set()

    version_key = f"{session_key}_version"
    
    # Initialisation session_state si nécessaire
    if session_key not in st.session_state:
        if default_unchecked:
            st.session_state[session_key] = set(options) - default_unchecked
        else:
            st.session_state[session_key] = set(options)
    
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    # Nettoyer les valeurs obsolètes (plus dans options)
    current_selection: set[str] = st.session_state[session_key]
    current_selection = current_selection & set(options)
    st.session_state[session_key] = current_selection

    selected_count = len(current_selection)
    total_count = len(options)
    version = st.session_state[version_key]

    # Titre avec compteur
    if selected_count == total_count:
        title = f"{label} (tous)"
    elif selected_count == 0:
        title = f"{label} (aucun)"
    else:
        title = f"{label} ({selected_count}/{total_count})"

    with st.expander(title, expanded=expanded):
        # Boutons Tout / Aucun
        if show_select_buttons and len(options) > 1:
            cols = st.columns(2)
            if cols[0].button(
                "✓ Tout",
                key=f"{session_key}_all_v{version}",
                use_container_width=True,
            ):
                st.session_state[session_key] = set(options)
                st.session_state[version_key] = version + 1
                st.rerun()
            if cols[1].button(
                "✗ Aucun",
                key=f"{session_key}_none_v{version}",
                use_container_width=True,
            ):
                st.session_state[session_key] = set()
                st.session_state[version_key] = version + 1
                st.rerun()

        # Checkboxes avec version dans la clé
        for opt in options:
            checked = opt in st.session_state[session_key]
            cb_key = f"{session_key}_cb_{opt}_v{version}"
            new_val = st.checkbox(
                opt,
                value=checked,
                key=cb_key,
            )
            # Synchroniser avec le set si l'utilisateur a cliqué
            if new_val and opt not in st.session_state[session_key]:
                st.session_state[session_key] = st.session_state[session_key] | {opt}
            elif not new_val and opt in st.session_state[session_key]:
                st.session_state[session_key] = st.session_state[session_key] - {opt}

    return st.session_state[session_key]


def _extract_mode_name(full_mode: str) -> str:
    """Extrait le nom du mode sans le préfixe de catégorie.
    
    Exemples:
        "Arène : Assassin" -> "Assassin"
        "BTB : Capture du drapeau" -> "Capture du drapeau"
        "Super Husky Raid : CDD" -> "CDD"
    """
    if " : " in full_mode:
        return full_mode.split(" : ", 1)[1].strip()
    return full_mode


def render_hierarchical_checkbox_filter(
    *,
    label: str,
    options: list[str],
    session_key: str,
    default_unchecked: Optional[set[str]] = None,
    expanded: bool = False,
) -> set[str]:
    """Affiche un expander avec checkboxes groupées par catégorie.

    Les modes sont fusionnés par nom (ex: "Arène : Assassin" et "Communauté : Assassin"
    deviennent une seule checkbox "Assassin" dans la catégorie correspondante).

    Utilise un système de version pour forcer le reset des checkboxes.
    Les boutons Tout/Aucun incrémentent la version, ce qui crée de nouvelles clés
    et donc de nouvelles checkboxes avec les bonnes valeurs initiales.

    Args:
        label: Titre de l'expander principal (ex: "Modes").
        options: Liste des valeurs disponibles (modes traduits avec préfixe).
        session_key: Clé session_state pour persister la sélection.
        default_unchecked: Valeurs décochées par défaut.
        expanded: Si l'expander principal est ouvert par défaut.

    Returns:
        Ensemble des valeurs sélectionnées (cochées) - valeurs originales avec préfixe.
    """
    if not options:
        return set()

    # Grouper les options par catégorie, puis par nom de mode (sans préfixe)
    # Structure: {category: {mode_name: [full_mode1, full_mode2, ...]}}
    categories: dict[str, dict[str, list[str]]] = {}
    for opt in options:
        cat = _infer_category(opt)
        mode_name = _extract_mode_name(opt)
        if cat not in categories:
            categories[cat] = {}
        if mode_name not in categories[cat]:
            categories[cat][mode_name] = []
        categories[cat][mode_name].append(opt)
    
    # Trier les catégories selon l'ordre de priorité
    priority_order = ["Assassin", "Fiesta", "BTB", "Ranked", "Firefight", "Other"]
    sorted_cats = []
    for cat in priority_order:
        if cat in categories:
            sorted_cats.append(cat)
    for cat in sorted(categories.keys()):
        if cat not in sorted_cats:
            sorted_cats.append(cat)

    # Système de version pour les boutons Tout/Aucun
    version_key = f"{session_key}_version"
    version = st.session_state.get(version_key, 0)

    # Initialisation session_state
    if session_key not in st.session_state:
        if default_unchecked:
            st.session_state[session_key] = set(options) - default_unchecked
        else:
            st.session_state[session_key] = set(options)

    # Nettoyer les valeurs obsolètes
    current_selection: set[str] = st.session_state[session_key]
    current_selection = current_selection & set(options)
    st.session_state[session_key] = current_selection

    selected_count = len(current_selection)
    total_count = len(options)

    # Titre principal avec compteur
    if selected_count == total_count:
        title = f"{label} (tous)"
    elif selected_count == 0:
        title = f"{label} (aucun)"
    else:
        title = f"{label} ({selected_count}/{total_count})"

    with st.expander(title, expanded=expanded):
        # Boutons globaux Tout / Aucun
        cols = st.columns(2)
        if cols[0].button(
            "✓ Tout",
            key=f"{session_key}_all",
            use_container_width=True,
        ):
            st.session_state[session_key] = set(options)
            st.session_state[version_key] = version + 1
            st.rerun()
        if cols[1].button(
            "✗ Aucun",
            key=f"{session_key}_none",
            use_container_width=True,
        ):
            st.session_state[session_key] = set()
            st.session_state[version_key] = version + 1
            st.rerun()

        st.markdown("---")

        # Collecter les changements sans rerun immédiat
        changes_to_add: set[str] = set()
        changes_to_remove: set[str] = set()

        # Afficher chaque catégorie
        for cat in sorted_cats:
            cat_modes = categories[cat]  # dict {mode_name: [full_modes]}
            cat_fr = _translate_category(cat)
            
            # Récupérer tous les full_modes de cette catégorie
            all_cat_options = [fm for modes in cat_modes.values() for fm in modes]
            
            # Compter les sélections dans cette catégorie
            cat_selected = [m for m in all_cat_options if m in st.session_state[session_key]]
            all_selected = len(cat_selected) == len(all_cat_options)
            none_selected = len(cat_selected) == 0
            
            # Nombre de modes uniques (après fusion)
            unique_modes_count = len(cat_modes)
            
            if unique_modes_count == 1:
                # Une seule catégorie/mode : checkbox simple
                mode_name = list(cat_modes.keys())[0]
                full_modes = cat_modes[mode_name]
                
                # Le mode est coché si TOUS les full_modes sont cochés
                checked = all(fm in st.session_state[session_key] for fm in full_modes)
                
                new_val = st.checkbox(
                    f"{cat_fr}",
                    value=checked,
                    key=f"{session_key}_cat_{cat}_v{version}",
                )
                if new_val and not checked:
                    changes_to_add.update(full_modes)
                elif not new_val and checked:
                    changes_to_remove.update(full_modes)
            else:
                # Plusieurs modes dans la catégorie
                # Compter les modes uniques sélectionnés
                modes_selected_count = sum(
                    1 for mn, full_modes in cat_modes.items()
                    if all(fm in st.session_state[session_key] for fm in full_modes)
                )
                
                cat_label = f"{cat_fr} ({modes_selected_count}/{unique_modes_count})"
                
                # Checkbox pour toute la catégorie
                cat_checkbox_val = st.checkbox(
                    cat_label,
                    value=all_selected,
                    key=f"{session_key}_cat_{cat}_v{version}",
                )
                
                # Si l'utilisateur a cliqué sur la checkbox catégorie
                if cat_checkbox_val and not all_selected:
                    changes_to_add.update(all_cat_options)
                elif not cat_checkbox_val and not none_selected:
                    changes_to_remove.update(all_cat_options)
                
                # Sous-expander pour les modes individuels
                with st.expander("", expanded=False):
                    for mode_name in sorted(cat_modes.keys()):
                        full_modes = cat_modes[mode_name]
                        # Le mode est coché si TOUS les full_modes sont cochés
                        checked = all(fm in st.session_state[session_key] for fm in full_modes)
                        
                        new_val = st.checkbox(
                            mode_name,
                            value=checked,
                            key=f"{session_key}_mode_{cat}_{mode_name}_v{version}",
                        )
                        if new_val and not checked:
                            changes_to_add.update(full_modes)
                        elif not new_val and checked:
                            changes_to_remove.update(full_modes)

        # Appliquer tous les changements en une fois (sans rerun)
        if changes_to_add or changes_to_remove:
            updated = (st.session_state[session_key] | changes_to_add) - changes_to_remove
            st.session_state[session_key] = updated

    return st.session_state[session_key]


def get_firefight_playlists(playlist_values: list[str]) -> set[str]:
    """Identifie les playlists Firefight dans une liste.

    Args:
        playlist_values: Liste des noms de playlists.

    Returns:
        Ensemble des playlists contenant "Firefight".
    """
    return {p for p in playlist_values if "firefight" in p.lower()}
