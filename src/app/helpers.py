"""Fonctions utilitaires pour l'application Streamlit.

Ce module contient des helpers génériques utilisés dans l'application principale.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import streamlit as st

from src.config import HALO_COLORS
from src.ui import translate_playlist_name, translate_pair_name


# =============================================================================
# Regex & constantes
# =============================================================================

_LABEL_SUFFIX_RE = re.compile(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", re.IGNORECASE)


# =============================================================================
# Nettoyage de labels
# =============================================================================


def clean_asset_label(s: str | None) -> str | None:
    """Nettoie un label d'asset en retirant les suffixes techniques (IDs).
    
    Args:
        s: Label brut.
        
    Returns:
        Label nettoyé ou None.
    """
    if s is None:
        return None
    v = str(s).strip()
    if not v:
        return None
    m = _LABEL_SUFFIX_RE.match(v)
    if m:
        v = (m.group(1) or "").strip()
    return v or None


def is_uuid_like(s: str) -> bool:
    """Vérifie si une chaîne ressemble à un UUID (ex: a446725e-b281-414c-a21e)."""
    return bool(re.match(r'^[a-f0-9]{8}(-[a-f0-9]{4}){0,3}(-[a-f0-9]{1,12})?$', s.lower()))


def normalize_mode_label(pair_name: str | None) -> str | None:
    """Normalise le label d'un mode de jeu.
    
    - Traduit le mode
    - Retire le nom de carte si présent ("X on MapName")
    - Retire les suffixes Forge/Ranked
    
    Args:
        pair_name: Nom du pair (mode + carte).
        
    Returns:
        Label normalisé ou None.
    """
    if pair_name is None:
        return None
    base = clean_asset_label(pair_name)
    t = translate_pair_name(base)
    if t is None:
        return None
    s = str(t).strip()
    if " on " in s:
        s = s.split(" on ", 1)[0].strip()
    s = re.sub(r"\s*-\s*Forge\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*-\s*Ranked\b", "", s, flags=re.IGNORECASE).strip()
    return s or None


def normalize_map_label(map_name: str | None) -> str | None:
    """Normalise le nom d'une carte pour le filtre.
    
    - Supprime les suffixes après '-' (ex: 'Aquarius - Live Fire' → 'Aquarius')
    - Masque les UUIDs non résolus en 'Carte inconnue'
    
    Args:
        map_name: Nom brut de la carte.
        
    Returns:
        Nom normalisé ou None.
    """
    base = clean_asset_label(map_name)
    if base is None:
        return None
    if is_uuid_like(base):
        return "Carte inconnue"
    if " - " in base:
        base = base.split(" - ")[0].strip()
    return base or None


# =============================================================================
# Calculs temporels
# =============================================================================


def compute_session_span_seconds(df_: pd.DataFrame) -> float | None:
    """Calcule la durée totale d'une session (premier match → fin dernier match).
    
    Args:
        df_: DataFrame des matchs de la session.
        
    Returns:
        Durée en secondes ou None.
    """
    if df_ is None or df_.empty or "start_time" not in df_.columns:
        return None
    starts = pd.to_datetime(df_["start_time"], errors="coerce")
    if starts.dropna().empty:
        return None
    t0 = starts.min()
    if "time_played_seconds" in df_.columns:
        durations = pd.to_numeric(df_["time_played_seconds"], errors="coerce")
        ends = starts + pd.to_timedelta(durations.fillna(0), unit="s")
    else:
        ends = starts
    t1 = ends.max()
    if pd.isna(t0) or pd.isna(t1):
        return None
    return float((t1 - t0).total_seconds())


def compute_total_play_seconds(df_: pd.DataFrame) -> float | None:
    """Calcule le temps de jeu total (somme des durées de matchs).
    
    Args:
        df_: DataFrame des matchs.
        
    Returns:
        Durée totale en secondes ou None.
    """
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.sum())


def avg_match_duration_seconds(df_: pd.DataFrame) -> float | None:
    """Calcule la durée moyenne d'un match.
    
    Args:
        df_: DataFrame des matchs.
        
    Returns:
        Durée moyenne en secondes ou None.
    """
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


# =============================================================================
# Plage de dates
# =============================================================================


def date_range(df: pd.DataFrame) -> tuple:
    """Retourne la plage de dates du DataFrame.
    
    Args:
        df: DataFrame avec colonne 'date'.
        
    Returns:
        Tuple (date_min, date_max).
    """
    dmin = df["date"].min()
    dmax = df["date"].max()
    return dmin, dmax


# =============================================================================
# Couleurs joueurs
# =============================================================================


def assign_player_colors(names: list[str]) -> dict[str, str]:
    """Assigne des couleurs persistantes aux joueurs.
    
    Les couleurs sont stockées en session_state pour rester cohérentes
    tout au long de la session.
    
    Args:
        names: Liste des noms de joueurs.
        
    Returns:
        Mapping {nom: couleur_hex}.
    """
    palette = HALO_COLORS.as_dict()
    cycle = [
        palette["cyan"],
        palette["violet"],
        palette["amber"],
        palette["red"],
        palette["green"],
        palette["slate"],
    ]
    state_key = "_os_player_colors"
    persisted = st.session_state.get(state_key)
    if not isinstance(persisted, dict):
        persisted = {}

    used = {str(v) for v in persisted.values() if v is not None}

    for n in names:
        key = str(n)
        if not key or key in persisted:
            continue

        chosen = None
        for c in cycle:
            if c not in used:
                chosen = c
                break
        if chosen is None:
            chosen = cycle[len(persisted) % len(cycle)]

        persisted[key] = chosen
        used.add(chosen)

    st.session_state[state_key] = persisted
    return {str(n): persisted[str(n)] for n in names if str(n) in persisted}


# =============================================================================
# Pandas Styler compat
# =============================================================================


def styler_map(styler, func, subset):
    """Applique une fonction de style (compat pandas anciennes versions).
    
    - pandas récents: .map(func, subset=...)
    - pandas anciens: .applymap(func, subset=...)
    
    Args:
        styler: Pandas Styler.
        func: Fonction de style.
        subset: Colonnes à cibler.
        
    Returns:
        Styler modifié.
    """
    try:
        if hasattr(styler, "map"):
            return styler.map(func, subset=subset)
    except Exception:
        pass
    # Fallback
    try:
        return styler.applymap(func, subset=subset)
    except Exception:
        return styler


# =============================================================================
# Session state helpers
# =============================================================================


def clear_min_matches_maps_auto() -> None:
    """Réinitialise le flag auto pour min_matches_maps."""
    st.session_state["_min_matches_maps_auto"] = False


def clear_min_matches_maps_friends_auto() -> None:
    """Réinitialise le flag auto pour min_matches_maps_friends."""
    st.session_state["_min_matches_maps_friends_auto"] = False
