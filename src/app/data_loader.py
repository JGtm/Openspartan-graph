"""Chargement et initialisation des données pour l'application Streamlit.

Ce module gère :
- L'initialisation de l'état source (DB, XUID, joueur Waypoint)
- La résolution des identités joueur
- Le chargement des données avec cache
- La génération automatique des références (citations H5G)
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    get_default_db_path,
    get_repo_root,
    DEFAULT_PLAYER_GAMERTAG,
    DEFAULT_PLAYER_XUID,
    DEFAULT_WAYPOINT_PLAYER,
    get_aliases_file_path,
)
from src.db import guess_xuid_from_db_path, infer_spnkr_player_from_db_path, resolve_xuid_from_db
from src.db.parsers import parse_xuid_input
from src.ui import AppSettings
from src.ui.sync import pick_latest_spnkr_db_if_any, is_spnkr_db_path
from src.ui.cache import load_df_optimized, db_cache_key


# =============================================================================
# Identité joueur depuis secrets/env
# =============================================================================


def default_identity_from_secrets() -> tuple[str, str, str]:
    """Retourne (xuid_or_gamertag, xuid_fallback, waypoint_player) depuis secrets/env/constants.
    
    Ordre de priorité :
    1. Secrets Streamlit (.streamlit/secrets.toml)
    2. Variables d'environnement
    3. Constantes du projet
    
    Returns:
        Tuple (xuid_or_gamertag, xuid_fallback, waypoint_player).
    """
    # Secrets Streamlit: .streamlit/secrets.toml
    try:
        player = st.secrets.get("player", {})
        if isinstance(player, Mapping):
            gt = str(player.get("gamertag") or "").strip()
            xu = str(player.get("xuid") or "").strip()
            wp = str(player.get("waypoint_player") or "").strip()
        else:
            gt = xu = wp = ""
    except Exception:
        gt = xu = wp = ""

    # Env vars (utile Docker/CLI)
    gt = gt or str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip()
    xu = xu or str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip()
    wp = wp or str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip() or gt

    # Fallback constants
    gt = gt or str(DEFAULT_PLAYER_GAMERTAG or "").strip()
    xu = xu or str(DEFAULT_PLAYER_XUID or "").strip()
    wp = wp or str(DEFAULT_WAYPOINT_PLAYER or "").strip() or gt

    # UI: on préfère afficher le gamertag, tout en conservant xuid en fallback.
    xuid_or_gt = gt or xu
    return xuid_or_gt, xu, wp


def propagate_identity_env(xuid_or_gt: str, xuid_fallback: str, wp: str) -> None:
    """Propage les defaults depuis secrets vers l'env.
    
    Utile notamment pour résoudre un XUID quand la DB SPNKr ne contient pas les gamertags.
    """
    try:
        if xuid_or_gt and not str(xuid_or_gt).strip().isdigit() and xuid_fallback:
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_GAMERTAG"] = str(xuid_or_gt).strip()
            if not str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip():
                os.environ["OPENSPARTAN_DEFAULT_XUID"] = str(xuid_fallback).strip()
        if wp and not str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip():
            os.environ["OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER"] = str(wp).strip()
    except Exception:
        pass


def apply_settings_path_overrides(settings: AppSettings) -> None:
    """Applique les overrides de chemins depuis les paramètres.
    
    Args:
        settings: Paramètres de l'application.
    """
    try:
        aliases_override = str(getattr(settings, "aliases_path", "") or "").strip()
        if aliases_override:
            os.environ["OPENSPARTAN_ALIASES_PATH"] = aliases_override
        else:
            os.environ.pop("OPENSPARTAN_ALIASES_PATH", None)
    except Exception:
        pass
    try:
        profiles_override = str(getattr(settings, "profiles_path", "") or "").strip()
        if profiles_override:
            os.environ["OPENSPARTAN_PROFILES_PATH"] = profiles_override
        else:
            os.environ.pop("OPENSPARTAN_PROFILES_PATH", None)
    except Exception:
        pass


# =============================================================================
# Initialisation source state
# =============================================================================


def init_source_state(default_db: str, settings: AppSettings) -> None:
    """Initialise l'état source (DB path, xuid, waypoint) en session_state.
    
    Args:
        default_db: Chemin par défaut de la DB.
        settings: Paramètres de l'application.
    """
    if "db_path" not in st.session_state:
        chosen = str(default_db or "")
        # Si l'utilisateur force une DB via env, ne pas l'écraser par auto-sélection SPNKr.
        forced_env_db = str(os.environ.get("OPENSPARTAN_DB") or os.environ.get("OPENSPARTAN_DB_PATH") or "").strip()
        if (not forced_env_db) and bool(getattr(settings, "prefer_spnkr_db_if_available", False)):
            spnkr = pick_latest_spnkr_db_if_any()
            if spnkr and os.path.exists(spnkr) and os.path.getsize(spnkr) > 0:
                chosen = spnkr
        st.session_state["db_path"] = chosen
        
    if "xuid_input" not in st.session_state:
        legacy = str(st.session_state.get("xuid", "") or "").strip()
        guessed = guess_xuid_from_db_path(st.session_state.get("db_path", "") or "") or ""
        xuid_or_gt, _xuid_fallback, _wp = default_identity_from_secrets()
        # Pour les DB SPNKr, pré-remplir avec le joueur déduit du nom de DB.
        inferred = infer_spnkr_player_from_db_path(str(st.session_state.get("db_path", "") or "")) or ""
        st.session_state["xuid_input"] = legacy or inferred or guessed or xuid_or_gt
        
    if "waypoint_player" not in st.session_state:
        _xuid_or_gt, _xuid_fallback, wp = default_identity_from_secrets()
        st.session_state["waypoint_player"] = wp


def resolve_xuid_input(xuid_input: str, db_path: str) -> str:
    """Résout un XUID à partir de l'entrée utilisateur.
    
    Args:
        xuid_input: Entrée utilisateur (XUID ou gamertag).
        db_path: Chemin vers la base de données.
        
    Returns:
        XUID résolu ou chaîne vide.
    """
    xraw = (xuid_input or "").strip()
    xuid_resolved = parse_xuid_input(xraw) or ""
    
    if not xuid_resolved and xraw and not xraw.isdigit() and db_path:
        xuid_resolved = resolve_xuid_from_db(db_path, xraw) or ""
        # Fallback: si la DB ne permet pas de résoudre,
        # utiliser les defaults quand l'entrée correspond au gamertag par défaut.
        if not xuid_resolved:
            try:
                xuid_or_gt, xuid_fallback, _wp = default_identity_from_secrets()
                if (
                    xuid_or_gt
                    and xuid_fallback
                    and (not str(xuid_or_gt).strip().isdigit())
                    and str(xuid_or_gt).strip().casefold() == str(xraw).strip().casefold()
                ):
                    xuid_resolved = str(xuid_fallback).strip()
            except Exception:
                pass
                
    if not xuid_resolved and not xraw and db_path:
        xuid_or_gt, xuid_fallback, _wp = default_identity_from_secrets()
        if xuid_or_gt and not xuid_or_gt.isdigit():
            xuid_resolved = resolve_xuid_from_db(db_path, xuid_or_gt) or xuid_fallback
        else:
            xuid_resolved = xuid_or_gt or xuid_fallback

    return xuid_resolved or ""


def validate_db_path(db_path: str, default_db: str) -> str:
    """Valide et corrige le chemin de la DB si nécessaire.
    
    - Si le fichier n'existe pas, retourne chaîne vide
    - Si le fichier est vide (0 octet), tente un fallback
    
    Args:
        db_path: Chemin actuel de la DB.
        default_db: Chemin par défaut en fallback.
        
    Returns:
        Chemin validé ou chaîne vide.
    """
    if db_path and not os.path.exists(db_path):
        return ""
    
    # Si la DB existe mais est vide (0 octet), tenter un fallback automatique.
    if db_path and os.path.exists(db_path):
        try:
            if os.path.getsize(db_path) <= 0:
                fallback = ""
                if is_spnkr_db_path(db_path):
                    fallback = pick_latest_spnkr_db_if_any()
                    if fallback and os.path.exists(fallback) and os.path.getsize(fallback) <= 0:
                        fallback = ""
                if not fallback:
                    fallback = str(default_db or "").strip()
                    if not (fallback and os.path.exists(fallback)):
                        fallback = ""
                if fallback and fallback != db_path:
                    return fallback
                return ""
        except Exception:
            pass
    
    return db_path


# =============================================================================
# Cache keys
# =============================================================================


def get_db_cache_key(db_path: str) -> tuple[int, int] | None:
    """Retourne une clé de cache basée sur (mtime, size) du fichier DB.
    
    Args:
        db_path: Chemin vers la base de données.
        
    Returns:
        Tuple (mtime_ns, size) ou None si le fichier n'existe pas.
    """
    return db_cache_key(db_path)


def get_aliases_cache_key() -> int | None:
    """Retourne une clé de cache pour le fichier d'alias.
    
    Returns:
        Timestamp de modification ou None.
    """
    try:
        p = get_aliases_file_path()
        st_ = os.stat(p)
        return int(getattr(st_, "st_mtime_ns", int(st_.st_mtime * 1e9)))
    except OSError:
        return None


# =============================================================================
# Chargement des données
# =============================================================================


def load_match_data(db_path: str, xuid: str, db_key: tuple[int, int] | None) -> pd.DataFrame:
    """Charge les données de matchs depuis la DB.
    
    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        db_key: Clé de cache.
        
    Returns:
        DataFrame des matchs ou DataFrame vide.
    """
    if not db_path or not os.path.exists(db_path) or not str(xuid or "").strip():
        return pd.DataFrame()
    
    return load_df_optimized(db_path, xuid.strip(), db_key=db_key)


# =============================================================================
# Génération automatique des références
# =============================================================================


def ensure_h5g_commendations_repo() -> None:
    """Génère automatiquement le référentiel Citations s'il est absent.
    
    Ne fait rien si :
    - Déjà généré dans cette session
    - Le fichier JSON existe déjà
    - Les fichiers source (HTML, script) n'existent pas
    """
    if st.session_state.get("_h5g_repo_ensured") is True:
        return
    st.session_state["_h5g_repo_ensured"] = True

    try:
        repo_root = get_repo_root(__file__)
    except Exception:
        repo_root = os.path.abspath(os.path.dirname(__file__))
        
    json_path = os.path.join(repo_root, "data", "wiki", "halo5_commendations_fr.json")
    html_path = os.path.join(repo_root, "data", "wiki", "halo5_citations_printable.html")
    script_path = os.path.join(repo_root, "scripts", "extract_h5g_commendations_fr.py")

    if os.path.exists(json_path):
        return
    if not (os.path.exists(html_path) and os.path.exists(script_path)):
        return

    with st.spinner("Génération du référentiel Citations (offline)…"):
        try:
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--input-html",
                    html_path,
                    "--clean-output",
                    "--exclude",
                    os.path.join(repo_root, "data", "wiki", "halo5_commendations_exclude.json"),
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            try:
                from src.ui.commendations import load_h5g_commendations_json
                getattr(load_h5g_commendations_json, "clear")()
            except Exception:
                pass
        except Exception:
            return
