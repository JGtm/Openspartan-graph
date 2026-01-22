"""Sections UI: configuration de la source (DB/XUID), profils et alias."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import DEFAULT_PLAYER_GAMERTAG, DEFAULT_WAYPOINT_PLAYER, get_aliases_file_path
from src.db import (
    guess_xuid_from_db_path,
    infer_spnkr_player_from_db_path,
    load_profiles,
    save_profiles,
    resolve_xuid_from_db,
)
from src.ui.aliases import load_aliases_file, save_aliases_file, display_name_from_xuid
from src.ui.path_picker import file_input


def _default_identity_from_secrets() -> tuple[str, str]:
    """Retourne (xuid_or_gamertag, waypoint_player) depuis secrets/env/constants."""
    try:
        player = st.secrets.get("player", {})
        if isinstance(player, dict):
            gt = str(player.get("gamertag") or "").strip()
            xu = str(player.get("xuid") or "").strip()
            wp = str(player.get("waypoint_player") or "").strip()
        else:
            gt = xu = wp = ""
    except Exception:
        gt = xu = wp = ""

    gt = gt or str(os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip()
    xu = xu or str(os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip()
    wp = wp or str(os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or "").strip()

    gt = gt or str(DEFAULT_PLAYER_GAMERTAG or "").strip()
    wp = wp or str(DEFAULT_WAYPOINT_PLAYER or "").strip() or gt

    # UI: on préfère afficher le gamertag, tout en conservant xuid en fallback.
    xuid_or_gt = gt or xu
    return xuid_or_gt, wp


def render_source_section(
    default_db: str,
    *,
    get_local_dbs: Callable[[], list[str]],
    on_clear_caches: Callable[[], None],
) -> tuple[str, str, str]:
    """Rend la section "Source" dans la sidebar.

    Returns:
        (db_path, xuid, waypoint_player)
    """

    # --- Multi-DB / Profils ---
    profiles = load_profiles()

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = default_db
    # IMPORTANT Streamlit: ne pas modifier une key de widget après instanciation.
    # On sépare donc l'entrée utilisateur (xuid_input) du XUID effectivement utilisé (résolu plus bas).
    if "xuid_input" not in st.session_state:
        # migration douce depuis l'ancien key "xuid" s'il existe encore
        legacy = str(st.session_state.get("xuid", "") or "").strip()
        guessed = guess_xuid_from_db_path(st.session_state.get("db_path", "")) or ""
        env_player = (os.environ.get("SPNKR_PLAYER") or "").strip()
        secret_player, _secret_wp = _default_identity_from_secrets()
        st.session_state["xuid_input"] = legacy or guessed or env_player or secret_player
    if "waypoint_player" not in st.session_state:
        _secret_player, secret_wp = _default_identity_from_secrets()
        st.session_state["waypoint_player"] = secret_wp

    # DB SPNKr locale (par défaut: data/spnkr.db à la racine du repo)
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data"
    spnkr_db_path = str(data_dir / "spnkr.db")
    try:
        spnkr_candidates = [p for p in data_dir.glob("spnkr*.db") if p.is_file()]
    except Exception:
        spnkr_candidates = []

    spnkr_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    latest_spnkr_db_path = str(spnkr_candidates[0]) if spnkr_candidates else spnkr_db_path

    current_db_path = str(st.session_state.get("db_path", "") or "")
    try:
        cur_parent = Path(current_db_path).resolve().parent
    except Exception:
        cur_parent = Path(current_db_path).parent
    is_spnkr = (
        os.path.normcase(str(cur_parent)) == os.path.normcase(str(data_dir))
        and Path(current_db_path).suffix.lower() == ".db"
        and Path(current_db_path).name.lower().startswith("spnkr")
    )

    c_top = st.columns(3)
    if c_top[0].button("Vider caches", width="stretch"):
        on_clear_caches()
        st.success("Caches vidés.")
        st.rerun()
    if c_top[1].button("Rafraîchir DB", width="stretch"):
        # Le get_local_dbs est censé être caché côté app (ttl) ; on le force via clear/closure.
        try:
            getattr(get_local_dbs, "clear")()  # st.cache_data wrapper
        except Exception:
            pass
        st.rerun()

    switch_label = "Basculer vers DB OpenSpartan" if is_spnkr else "Basculer vers DB SPNKr"
    if c_top[2].button(switch_label, width="stretch"):
        if is_spnkr:
            st.session_state["db_path"] = default_db
            guessed = guess_xuid_from_db_path(str(default_db)) or ""
            if guessed:
                st.session_state["xuid_input"] = guessed
            st.rerun()
        else:
            if not os.path.exists(latest_spnkr_db_path):
                st.warning("DB SPNKr introuvable (data/spnkr*.db). Lance d'abord le refresh SPNKr.")
            st.session_state["db_path"] = latest_spnkr_db_path
            # Si SPNKR_PLAYER est un gamertag, on résout le XUID via la DB.
            spnkr_player = (os.environ.get("SPNKR_PLAYER") or "").strip()
            if spnkr_player:
                if spnkr_player.isdigit():
                    st.session_state["xuid_input"] = spnkr_player
                else:
                    resolved = resolve_xuid_from_db(latest_spnkr_db_path, spnkr_player)
                    if resolved:
                        st.session_state["xuid_input"] = resolved
            else:
                # Par défaut: déduire le joueur depuis le nom de la DB SPNKr.
                inferred = infer_spnkr_player_from_db_path(latest_spnkr_db_path) or ""
                if inferred:
                    st.session_state["xuid_input"] = inferred
                else:
                    # Fallback local pour éviter un état "vide" qui bloque l'app.
                    secret_player, _secret_wp = _default_identity_from_secrets()
                    st.session_state["xuid_input"] = secret_player
            st.rerun()

    # UI simplifiée: sélection directe parmi les DB SPNKr détectées.
    if spnkr_candidates:
        with st.expander("DB détectées (SPNKr)", expanded=False):
            # Déterminer l'index actuel
            current_name = Path(current_db_path).name if current_db_path else ""
            db_names = [p.name for p in spnkr_candidates]
            try:
                current_idx = db_names.index(current_name)
            except ValueError:
                current_idx = 0

            def _on_db_change():
                pick = st.session_state.get("_spnkr_db_picker", "")
                sel_p = next((p for p in spnkr_candidates if p.name == pick), None)
                if sel_p:
                    st.session_state["db_path"] = str(sel_p)
                    inferred = infer_spnkr_player_from_db_path(str(sel_p)) or ""
                    if inferred:
                        st.session_state["xuid_input"] = inferred
                    on_clear_caches()

            st.selectbox(
                "DB",
                options=db_names,
                index=current_idx,
                key="_spnkr_db_picker",
                on_change=_on_db_change,
            )

    # Mémoriser l'ancienne DB pour détecter un changement
    previous_db = str(st.session_state.get("db_path", "") or "").strip()

    db_path = file_input(
        "Chemin du .db",
        key="db_path",
        exts=(".db",),
        help="Sélectionne un fichier SQLite (.db).",
        placeholder="Ex: C:\\Users\\Guillaume\\AppData\\Local\\OpenSpartan.Workshop\\data\\2533....db",
    )

    # Détection de changement de DB: mettre à jour le xuid_input automatiquement
    current_db = str(st.session_state.get("db_path", "") or "").strip()
    if current_db and current_db != previous_db and os.path.exists(current_db):
        # La DB a changé → déduire le nouveau joueur
        inferred = infer_spnkr_player_from_db_path(current_db)
        if not inferred:
            inferred = guess_xuid_from_db_path(current_db)
        if inferred:
            st.session_state["xuid_input"] = inferred
        # Vider les caches pour forcer le rechargement avec la nouvelle DB
        on_clear_caches()
        st.rerun()

    # Identité: UI masquée (on résout et affiche uniquement le XUID effectif)
    raw_identity = str(st.session_state.get("xuid_input", "") or "").strip()
    xuid = resolve_xuid_from_db(str(db_path), raw_identity) or raw_identity
    if xuid and (not str(xuid).strip().isdigit()) and raw_identity and (not raw_identity.isdigit()):
        # Même fallback que le bouton: secrets → xuid
        try:
            player = st.secrets.get("player", {})
            if isinstance(player, Mapping):
                gt = str(player.get("gamertag") or "").strip()
                xu = str(player.get("xuid") or "").strip()
                if gt and xu and gt.casefold() == raw_identity.casefold():
                    xuid = xu
        except Exception:
            pass
    # Affichage simplifié (copiable) + déduction du slug Waypoint via alias
    st.text_input("XUID", value=str(xuid or "").strip(), disabled=True)

    name_guess = display_name_from_xuid(str(xuid or "").strip())
    waypoint_player = str(st.session_state.get("waypoint_player", "") or "").strip()
    if name_guess and name_guess != "-":
        waypoint_player = str(name_guess).strip()
        st.session_state["waypoint_player"] = waypoint_player

    # On garde une valeur non vide (fallback secrets)
    if not waypoint_player:
        _secret_player, secret_wp = _default_identity_from_secrets()
        waypoint_player = secret_wp
        st.session_state["waypoint_player"] = waypoint_player

    # Alias (XUID → gamertag) : UI masquée.

    return str(db_path), str(xuid), str(waypoint_player)
