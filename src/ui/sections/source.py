"""Sections UI: configuration de la source (DB/XUID), profils et alias."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from src.config import DEFAULT_PLAYER_GAMERTAG, DEFAULT_WAYPOINT_PLAYER, get_aliases_file_path
from src.db import guess_xuid_from_db_path, load_profiles, save_profiles, resolve_xuid_from_db
from src.ui.aliases import load_aliases_file, save_aliases_file, display_name_from_xuid


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
                # Fallback local pour éviter un état "vide" qui bloque l'app.
                secret_player, _secret_wp = _default_identity_from_secrets()
                st.session_state["xuid_input"] = secret_player
            st.rerun()

    with st.expander("Multi-DB (profils)", expanded=False):
        prof_names = ["(aucun)"] + sorted(profiles.keys())
        st.selectbox("Profil", options=prof_names, key="db_profile_selected")

        cols_p = st.columns(2)
        if cols_p[0].button("Appliquer", width="stretch"):
            sel = st.session_state.get("db_profile_selected")
            if isinstance(sel, str) and sel in profiles:
                p = profiles[sel]
                if p.get("db_path"):
                    st.session_state["db_path"] = p["db_path"]
                if p.get("xuid"):
                    st.session_state["xuid_input"] = p["xuid"]
                if p.get("waypoint_player"):
                    st.session_state["waypoint_player"] = p["waypoint_player"]
                st.rerun()

        if cols_p[1].button("Enregistrer profil", width="stretch"):
            name = st.session_state.get("db_profile_selected")
            if not isinstance(name, str) or not name.strip() or name == "(aucun)":
                st.warning("Choisis d'abord un nom de profil (via la liste), ou crée-en un ci-dessous.")
            else:
                profiles[name] = {
                    "db_path": str(st.session_state.get("db_path", "")).strip(),
                    "xuid": str(st.session_state.get("xuid_input", "")).strip(),
                    "waypoint_player": str(st.session_state.get("waypoint_player", "")).strip(),
                }
                ok, err = save_profiles(profiles)
                if ok:
                    st.success("Profil enregistré.")
                else:
                    st.error(err)

        with st.expander("Créer / supprimer", expanded=False):
            new_name = st.text_input("Nom", value="")
            c = st.columns(2)
            if c[0].button("Créer/mettre à jour", width="stretch"):
                nn = (new_name or "").strip()
                if not nn:
                    st.error("Nom vide.")
                else:
                    profiles[nn] = {
                        "db_path": str(st.session_state.get("db_path", "")).strip(),
                        "xuid": str(st.session_state.get("xuid_input", "")).strip(),
                        "waypoint_player": str(st.session_state.get("waypoint_player", "")).strip(),
                    }
                    ok, err = save_profiles(profiles)
                    if ok:
                        st.success("Profil sauvegardé.")
                        st.rerun()
                    else:
                        st.error(err)
            if c[1].button("Supprimer", width="stretch"):
                nn = (new_name or "").strip()
                if not nn:
                    st.error("Renseigne un nom.")
                elif nn not in profiles:
                    st.warning("Profil introuvable.")
                else:
                    del profiles[nn]
                    ok, err = save_profiles(profiles)
                    if ok:
                        st.success("Profil supprimé.")
                        st.rerun()
                    else:
                        st.error(err)

        local_dbs = get_local_dbs()
        if local_dbs:
            opts = ["(garder actuelle)"] + [os.path.basename(p) for p in local_dbs]
            pick = st.selectbox("DB détectées (OpenSpartan)", options=opts, index=0)
            if st.button("Utiliser cette DB", width="stretch") and pick != "(garder actuelle)":
                sel_path = next((p for p in local_dbs if os.path.basename(p) == pick), None)
                if sel_path:
                    st.session_state["db_path"] = sel_path
                    guessed = guess_xuid_from_db_path(sel_path) or ""
                    if guessed:
                        st.session_state["xuid_input"] = guessed
                    st.rerun()

        if spnkr_candidates:
            opts_s = ["(garder actuelle)"] + [p.name for p in spnkr_candidates]
            pick_s = st.selectbox("DB détectées (SPNKr)", options=opts_s, index=0)
            if st.button("Utiliser cette DB SPNKr", width="stretch") and pick_s != "(garder actuelle)":
                sel_p = next((p for p in spnkr_candidates if p.name == pick_s), None)
                if sel_p:
                    st.session_state["db_path"] = str(sel_p)
                    st.rerun()

    db_path = st.text_input("Chemin du .db", key="db_path")
    cols_x = st.columns([2, 1])
    with cols_x[0]:
        xuid_input = st.text_input("XUID ou Gamertag", key="xuid_input")
    with cols_x[1]:
        def _on_resolve_xuid() -> None:
            raw = str(st.session_state.get("xuid_input", "") or "").strip()
            guessed = guess_xuid_from_db_path(str(st.session_state.get("db_path", "") or "")) or ""
            resolved = resolve_xuid_from_db(str(st.session_state.get("db_path", "") or ""), raw) or ""
            if resolved:
                st.session_state["xuid_input"] = resolved
            elif guessed:
                st.session_state["xuid_input"] = guessed

        st.button("Résoudre XUID", width="stretch", on_click=_on_resolve_xuid)

    # Résolution douce pour l'affichage (sans modifier la valeur du widget)
    xuid = resolve_xuid_from_db(str(db_path), str(xuid_input)) or str(xuid_input).strip()
    if xuid and xuid != str(xuid_input).strip():
        st.caption(f"XUID résolu: {xuid}")

    _ = display_name_from_xuid(xuid.strip())

    waypoint_player = st.text_input(
        "HaloWaypoint player (slug)",
        key="waypoint_player",
        help="Ex: JGtm (sert à construire l'URL de match).",
    )

    with st.expander("Alias (XUID → gamertag)", expanded=False):
        st.caption(
            "La DB OpenSpartan ne contient pas les gamertags. "
            "Ici tu peux définir des alias locaux (persistés dans un fichier JSON)."
        )
        aliases_path = get_aliases_file_path()
        current_aliases = load_aliases_file(aliases_path)

        ax = st.text_input("XUID à aliaser", value="")
        an = st.text_input("Gamertag", value="")
        cols = st.columns(2)
        if cols[0].button("Enregistrer l'alias", width="stretch"):
            axc = (ax or "").strip()
            anc = (an or "").strip()
            if not axc.isdigit():
                st.error("XUID invalide (doit être numérique).")
            elif not anc:
                st.error("Gamertag vide.")
            else:
                current_aliases[axc] = anc
                save_aliases_file(current_aliases, aliases_path)
                st.success("Alias enregistré.")
                st.rerun()

        if cols[1].button("Supprimer l'alias", width="stretch"):
            axc = (ax or "").strip()
            if not axc:
                st.error("Renseigne un XUID à supprimer.")
            elif axc not in current_aliases:
                st.warning("Cet alias n'existe pas.")
            else:
                del current_aliases[axc]
                save_aliases_file(current_aliases, aliases_path)
                st.success("Alias supprimé.")
                st.rerun()

        if current_aliases:
            st.dataframe(
                pd.DataFrame(sorted(current_aliases.items()), columns=["XUID", "Gamertag"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Aucun alias personnalisé pour l'instant.")

    return str(db_path), str(xuid), str(waypoint_player)
