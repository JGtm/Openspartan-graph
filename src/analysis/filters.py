"""Fonctions de filtrage et helpers pour les options UI."""

import re
from typing import Dict, List, Callable

import pandas as pd


def mark_firefight(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne is_firefight pour identifier les matchs PvE.
    
    Heuristique basée sur les libellés Playlist / Pair contenant "Firefight".
    
    Args:
        df: DataFrame de matchs.
        
    Returns:
        DataFrame avec colonne is_firefight ajoutée.
    """
    d = df.copy()
    pl = d.get("playlist_name")
    pair = d.get("pair_name")
    gv = d.get("game_variant_name")
    pl_s = pl.fillna("").astype(str) if pl is not None else pd.Series([""] * len(d))
    pair_s = pair.fillna("").astype(str) if pair is not None else pd.Series([""] * len(d))
    gv_s = gv.fillna("").astype(str) if gv is not None else pd.Series([""] * len(d))

    pat = r"\bfirefight\b"
    d["is_firefight"] = (
        pl_s.str.contains(pat, case=False, regex=True) |
        pair_s.str.contains(pat, case=False, regex=True) |
        gv_s.str.contains(pat, case=False, regex=True)
    )
    return d


def is_allowed_playlist_name(name: str) -> bool:
    """Vérifie si une playlist est dans la liste autorisée.
    
    Playlists autorisées par défaut:
    - Quick Play
    - Ranked Slayer
    - Ranked Arena
    
    Args:
        name: Nom de la playlist.
        
    Returns:
        True si la playlist est autorisée.
    """
    s = (name or "").strip().casefold()
    if not s:
        return False
    # FR (UI)
    if re.search(r"\bpartie\s*rapide\b", s):
        return True
    if re.search(r"\bar(?:e|è)ne\b.*\bclass(?:e|é)e\b", s):
        return True
    # "classé" (masc) / "classée" (fém)
    if re.search(r"\bassassin\b.*\bclass(?:e|é)(?:e)?\b", s):
        return True
    # EN (API)
    if re.search(r"\bquick\s*play\b", s):
        return True
    if re.search(r"\branked\b.*\bslayer\b", s):
        return True
    if re.search(r"\branked\b.*\barena\b", s):
        return True
    return False


def build_option_map(series_name: pd.Series, series_id: pd.Series) -> Dict[str, str]:
    """Construit un dictionnaire label -> id pour les selectbox.
    
    Nettoie les libellés (supprime les suffixes UUID) et gère les collisions.
    
    Args:
        series_name: Série des noms.
        series_id: Série des IDs correspondants.
        
    Returns:
        Dictionnaire {label_propre: id} trié alphabétiquement.
    """
    def clean_label(s: str) -> str:
        s = (s or "").strip()
        # Supprime les suffixes type " - <hash/uuid>"
        m = re.match(r"^(.*?)(?:\s*[\-–—]\s*[0-9A-Za-z]{8,})$", s)
        if m:
            s = (m.group(1) or "").strip()
        return s

    out: Dict[str, str] = {}
    collisions: Dict[str, int] = {}

    for name, _id in zip(series_name.fillna(""), series_id.fillna("")):
        if not _id:
            continue
        if not (isinstance(name, str) and name.strip()):
            continue

        label = clean_label(name)
        if not label:
            continue

        # Gère les collisions de noms
        key = label
        if key in out and out[key] != _id:
            collisions[label] = collisions.get(label, 1) + 1
            key = f"{label} (v{collisions[label]})"

        out[key] = _id

    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def build_xuid_option_map(
    xuids: List[str],
    display_name_fn: Callable[[str], str] | None = None,
) -> Dict[str, str]:
    """Construit un dictionnaire label -> xuid pour les selectbox.
    
    Args:
        xuids: Liste de XUID.
        display_name_fn: Fonction pour obtenir le display name d'un XUID.
                        Si None, utilise le XUID tel quel.
        
    Returns:
        Dictionnaire {label: xuid} trié alphabétiquement.
    """
    out: Dict[str, str] = {}
    for x in xuids:
        if display_name_fn:
            label = display_name_fn(x)
        else:
            label = x
        out[f"{label} — {x}"] = x
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))
