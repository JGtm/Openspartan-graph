"""Gestion des filtres sidebar pour l'application Streamlit.

Ce module gère :
- Les filtres de période (dates) et de sessions
- Les filtres en cascade (Playlist → Mode → Carte)
- L'état des filtres en session_state
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.db.parsers import parse_xuid_input
from src.ui import display_name_from_xuid, translate_playlist_name
from src.analysis import build_xuid_option_map
from src.ui.components import (
    render_checkbox_filter,
    render_hierarchical_checkbox_filter,
    get_firefight_playlists,
)
from src.ui.cache import (
    cached_compute_sessions_db,
    cached_list_other_xuids,
    cached_list_top_teammates,
    cached_same_team_match_ids_with_friend,
)
from src.app.helpers import (
    clean_asset_label,
    normalize_mode_label,
    normalize_map_label,
    date_range,
)


# =============================================================================
# Friends helpers
# =============================================================================


def _load_local_friends_defaults() -> dict[str, list[str]]:
    """Charge un mapping local {self_xuid: [friend1, friend2, ...]}.

    Fichier local (ignoré git): .streamlit/friends_defaults.json
    Valeurs: gamertags OU XUIDs.
    """
    try:
        p = Path(__file__).resolve().parents[2] / ".streamlit" / "friends_defaults.json"
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f) or {}
    except Exception:
        return {}

    if not isinstance(obj, dict):
        return {}

    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, list):
            continue
        vals: list[str] = []
        for it in v:
            if isinstance(it, str) and it.strip():
                vals.append(it.strip())
        if vals:
            out[k.strip()] = vals
    return out


@st.cache_data(show_spinner=False)
def build_friends_opts_map(
    db_path: str,
    self_xuid: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
) -> tuple[dict[str, str], list[str]]:
    """Construit le mapping d'options pour la sélection d'amis.
    
    Args:
        db_path: Chemin vers la base de données.
        self_xuid: XUID du joueur principal.
        db_key: Clé de cache DB.
        aliases_key: Clé de cache des alias.
        
    Returns:
        Tuple (opts_map {label: xuid}, default_labels [labels par défaut]).
    """
    top = cached_list_top_teammates(db_path, self_xuid, db_key=db_key, limit=20)
    default_two = [t[0] for t in top[:2]]
    all_other = cached_list_other_xuids(db_path, self_xuid, db_key=db_key, limit=500)

    ordered: list[str] = []
    seen: set[str] = set()
    for xx, _cnt in top:
        if xx not in seen:
            ordered.append(xx)
            seen.add(xx)
    for xx in all_other:
        if xx not in seen:
            ordered.append(xx)
            seen.add(xx)

    opts_map = build_xuid_option_map(ordered, display_name_fn=display_name_from_xuid)

    # Defaults: top 2, ou override local.
    default_xuids = list(default_two)
    try:
        overrides = _load_local_friends_defaults().get(str(self_xuid).strip())
        if overrides:
            name_to_xuid = {
                str(display_name_from_xuid(xu) or "").strip().casefold(): str(xu).strip() for xu in ordered
            }
            ordered_xuids = [str(xu).strip() for xu in ordered]

            chosen: list[str] = []
            for ident in overrides:
                s = str(ident or "").strip()
                if not s:
                    continue
                if s.isdigit() and s in ordered_xuids:
                    chosen.append(s)
                    continue
                xu = name_to_xuid.get(s.casefold())
                if xu:
                    chosen.append(xu)
            chosen = [x for x in chosen if x]
            if len(chosen) >= 2:
                default_xuids = chosen[:2]
    except Exception:
        pass

    label_by_xuid = {v: k for k, v in opts_map.items()}
    default_labels = [label_by_xuid[x] for x in default_xuids if x in label_by_xuid]
    return opts_map, default_labels


# =============================================================================
# Filtres DataFrame
# =============================================================================


def add_ui_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes UI au DataFrame (playlist_ui, mode_ui, map_ui).
    
    Args:
        df: DataFrame source.
        
    Returns:
        DataFrame avec colonnes UI ajoutées.
    """
    if "playlist_ui" not in df.columns:
        df["playlist_ui"] = df["playlist_name"].apply(clean_asset_label).apply(translate_playlist_name)
    if "mode_ui" not in df.columns:
        df["mode_ui"] = df["pair_name"].apply(normalize_mode_label)
    if "map_ui" not in df.columns:
        df["map_ui"] = df["map_name"].apply(normalize_map_label)
    return df


def apply_date_filter(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """Applique un filtre de dates au DataFrame.
    
    Args:
        df: DataFrame source.
        start_d: Date de début.
        end_d: Date de fin.
        
    Returns:
        DataFrame filtré.
    """
    mask = (df["date"] >= start_d) & (df["date"] <= end_d)
    return df.loc[mask].copy()


def apply_checkbox_filters(
    df: pd.DataFrame,
    playlists_selected: list[str] | None,
    modes_selected: list[str] | None,
    maps_selected: list[str] | None,
) -> pd.DataFrame:
    """Applique les filtres checkbox (playlists, modes, cartes).
    
    Args:
        df: DataFrame source avec colonnes UI.
        playlists_selected: Playlists sélectionnées ou None pour tout.
        modes_selected: Modes sélectionnés ou None pour tout.
        maps_selected: Cartes sélectionnées ou None pour tout.
        
    Returns:
        DataFrame filtré.
    """
    if playlists_selected:
        df = df.loc[df["playlist_ui"].fillna("").isin(playlists_selected)]
    if modes_selected:
        df = df.loc[df["mode_ui"].fillna("").isin(modes_selected)]
    if maps_selected:
        df = df.loc[df["map_ui"].fillna("").isin(maps_selected)]
    return df


# =============================================================================
# Rendu des filtres sidebar
# =============================================================================


def render_date_filters(
    dmin: date,
    dmax: date,
) -> tuple[date, date]:
    """Rend les filtres de date et retourne la sélection.
    
    Args:
        dmin: Date minimum disponible.
        dmax: Date maximum disponible.
        
    Returns:
        Tuple (start_date, end_date) sélectionnées.
    """
    cols = st.columns(2)
    with cols[0]:
        start_default = pd.to_datetime(dmin, errors="coerce")
        if pd.isna(start_default):
            start_default = pd.Timestamp.today().normalize()
        start_default_date = start_default.date()
        end_limit = pd.to_datetime(dmax, errors="coerce")
        if pd.isna(end_limit):
            end_limit = start_default
        end_limit_date = end_limit.date()
        start_value = st.session_state.get("start_date_cal", start_default_date)
        if not isinstance(start_value, date) or start_value < start_default_date or start_value > end_limit_date:
            start_value = start_default_date
        start_date = st.date_input(
            "Début",
            value=start_value,
            min_value=start_default_date,
            max_value=end_limit_date,
            format="DD/MM/YYYY",
            key="start_date_cal",
        )
    with cols[1]:
        end_default = pd.to_datetime(dmax, errors="coerce")
        if pd.isna(end_default):
            end_default = pd.Timestamp.today().normalize()
        end_default_date = end_default.date()
        start_limit = pd.to_datetime(dmin, errors="coerce")
        if pd.isna(start_limit):
            start_limit = end_default
        start_limit_date = start_limit.date()
        end_value = st.session_state.get("end_date_cal", end_default_date)
        if not isinstance(end_value, date) or end_value < start_limit_date or end_value > end_default_date:
            end_value = end_default_date
        end_date = st.date_input(
            "Fin",
            value=end_value,
            min_value=start_limit_date,
            max_value=end_default_date,
            format="DD/MM/YYYY",
            key="end_date_cal",
        )
    return start_date, end_date


def render_session_filters(
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
    base_for_filters: pd.DataFrame,
) -> tuple[int, list[str] | None]:
    """Rend les filtres de session et retourne la sélection.
    
    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        db_key: Clé de cache DB.
        aliases_key: Clé de cache des alias.
        base_for_filters: DataFrame de base pour construire les sessions.
        
    Returns:
        Tuple (gap_minutes, picked_session_labels ou None pour toutes).
    """
    gap_minutes = st.slider(
        "Écart max entre parties (minutes)",
        min_value=15,
        max_value=240,
        value=int(st.session_state.get("gap_minutes", 120)),
        step=5,
        key="gap_minutes",
    )

    base_s_ui = cached_compute_sessions_db(
        db_path,
        xuid.strip(),
        db_key,
        True,  # Inclure Firefight (filtrage via checkboxes)
        gap_minutes,
    )
    session_labels_ui = (
        base_s_ui[["session_id", "session_label"]]
        .drop_duplicates()
        .sort_values("session_id", ascending=False)
    )
    options_ui = session_labels_ui["session_label"].tolist()
    st.session_state["_latest_session_label"] = options_ui[0] if options_ui else None

    def _set_session_selection(label: str) -> None:
        st.session_state.picked_session_label = label
        if label == "(toutes)":
            st.session_state.picked_sessions = []
        elif label in options_ui:
            st.session_state.picked_sessions = [label]

    if "picked_session_label" not in st.session_state:
        _set_session_selection(options_ui[0] if options_ui else "(toutes)")
    if "picked_sessions" not in st.session_state:
        st.session_state.picked_sessions = options_ui[:1] if options_ui else []

    cols = st.columns(2)
    if cols[0].button("Dernière session", width="stretch"):
        _set_session_selection(options_ui[0] if options_ui else "(toutes)")
        st.session_state["min_matches_maps"] = 1
        st.session_state["_min_matches_maps_auto"] = True
        st.session_state["min_matches_maps_friends"] = 1
        st.session_state["_min_matches_maps_friends_auto"] = True
    if cols[1].button("Session précédente", width="stretch"):
        current = st.session_state.get("picked_session_label", "(toutes)")
        if not options_ui:
            _set_session_selection("(toutes)")
        elif current == "(toutes)" or current not in options_ui:
            _set_session_selection(options_ui[0])
        else:
            idx = options_ui.index(current)
            next_idx = min(idx + 1, len(options_ui) - 1)
            _set_session_selection(options_ui[next_idx])

    # Bouton Trio
    trio_label = _compute_trio_label(
        db_path, xuid, db_key, aliases_key, base_for_filters, base_s_ui
    )
    st.session_state["_trio_latest_session_label"] = trio_label
    disabled_trio = not isinstance(trio_label, str) or not trio_label
    if st.button("Dernière session en trio", width="stretch", disabled=disabled_trio):
        st.session_state["_pending_filter_mode"] = "Sessions"
        st.session_state["_pending_picked_session_label"] = trio_label
        st.session_state["_pending_picked_sessions"] = [trio_label]
        st.session_state["min_matches_maps"] = 1
        st.session_state["_min_matches_maps_auto"] = True
        st.session_state["min_matches_maps_friends"] = 1
        st.session_state["_min_matches_maps_friends_auto"] = True
        st.rerun()
    if disabled_trio:
        st.caption('Trio : sélectionne 2 amis dans "Avec mes amis" pour activer.')
    else:
        st.caption(f"Trio : {trio_label}")

    picked_one = st.selectbox("Session", options=["(toutes)"] + options_ui, key="picked_session_label")
    picked_session_labels = None if picked_one == "(toutes)" else [picked_one]

    return gap_minutes, picked_session_labels


def _compute_trio_label(
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
    base_for_filters: pd.DataFrame,
    base_s_ui: pd.DataFrame,
) -> str | None:
    """Calcule le label de la dernière session en trio."""
    try:
        friends_opts_map, friends_default_labels = build_friends_opts_map(
            db_path, xuid.strip(), db_key, aliases_key
        )
        picked_friend_labels = st.session_state.get("friends_picked_labels")
        if not isinstance(picked_friend_labels, list) or not picked_friend_labels:
            picked_friend_labels = friends_default_labels
        picked_xuids = [friends_opts_map[lbl] for lbl in picked_friend_labels if lbl in friends_opts_map]
        if len(picked_xuids) >= 2:
            f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
            ids_m = set(
                cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f1_xuid, db_key=db_key)
            )
            ids_c = set(
                cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f2_xuid, db_key=db_key)
            )
            trio_ids = ids_m & ids_c
            trio_ids = trio_ids & set(base_for_filters["match_id"].astype(str))
            if trio_ids:
                trio_rows = base_s_ui.loc[base_s_ui["match_id"].astype(str).isin(trio_ids)].copy()
                if not trio_rows.empty:
                    latest_sid = int(trio_rows["session_id"].max())
                    latest_labels = (
                        trio_rows.loc[trio_rows["session_id"] == latest_sid, "session_label"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    return latest_labels[0] if latest_labels else None
    except Exception:
        pass
    return None


def render_cascade_filters(
    dropdown_base: pd.DataFrame,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    """Rend les filtres en cascade (Playlist → Mode → Carte).
    
    Args:
        dropdown_base: DataFrame de base pour les options.
        
    Returns:
        Tuple (playlists_selected, modes_selected, maps_selected).
    """
    # Ajouter colonnes UI si nécessaire
    dropdown_base = add_ui_columns(dropdown_base.copy())

    # --- Playlists (avec Firefight décoché par défaut) ---
    playlist_values = sorted({
        str(x).strip() for x in dropdown_base["playlist_ui"].dropna().tolist() if str(x).strip()
    })
    preferred_order = ["Partie rapide", "Arène classée", "Assassin classé"]
    playlist_values = [p for p in preferred_order if p in playlist_values] + [
        p for p in playlist_values if p not in preferred_order
    ]

    firefight_playlists = get_firefight_playlists(playlist_values)
    playlists_selected = render_checkbox_filter(
        label="Playlists",
        options=playlist_values,
        session_key="filter_playlists",
        default_unchecked=firefight_playlists,
        expanded=False,
    )

    # Scope après filtre playlist
    scope1 = dropdown_base
    if playlists_selected and len(playlists_selected) < len(playlist_values):
        scope1 = scope1.loc[scope1["playlist_ui"].fillna("").isin(playlists_selected)].copy()

    # --- Modes (hiérarchique par catégorie) ---
    mode_values = sorted({str(x).strip() for x in scope1["mode_ui"].dropna().tolist() if str(x).strip()})
    modes_selected = render_hierarchical_checkbox_filter(
        label="Modes",
        options=mode_values,
        session_key="filter_modes",
        expanded=False,
    )

    # Scope après filtre mode
    scope2 = scope1
    if modes_selected and len(modes_selected) < len(mode_values):
        scope2 = scope2.loc[scope2["mode_ui"].fillna("").isin(modes_selected)].copy()

    # --- Cartes ---
    map_values = sorted({str(x).strip() for x in scope2["map_ui"].dropna().tolist() if str(x).strip()})
    maps_selected = render_checkbox_filter(
        label="Cartes",
        options=map_values,
        session_key="filter_maps",
        expanded=False,
    )

    return playlists_selected, modes_selected, maps_selected


# =============================================================================
# État des filtres
# =============================================================================


def consume_pending_filter_state() -> None:
    """Consomme l'état en attente pour les filtres (changements demandés).
    
    Applique les changements de mode/session stockés en session_state
    par d'autres composants (ex: boutons trio).
    """
    pending_mode = st.session_state.pop("_pending_filter_mode", None)
    if pending_mode in ("Période", "Sessions"):
        st.session_state["filter_mode"] = pending_mode

    pending_label = st.session_state.pop("_pending_picked_session_label", None)
    if isinstance(pending_label, str) and pending_label:
        st.session_state["picked_session_label"] = pending_label
    pending_sessions = st.session_state.pop("_pending_picked_sessions", None)
    if isinstance(pending_sessions, list):
        st.session_state["picked_sessions"] = pending_sessions


def reset_auto_min_matches(filter_mode: str) -> None:
    """Réinitialise les valeurs auto de min_matches si on revient en mode Période.
    
    Args:
        filter_mode: Mode de filtre actuel ("Période" ou "Sessions").
    """
    if filter_mode == "Période" and bool(st.session_state.get("_min_matches_maps_auto")):
        st.session_state["min_matches_maps"] = 5
        st.session_state["_min_matches_maps_auto"] = False

    if filter_mode == "Période" and bool(st.session_state.get("_min_matches_maps_friends_auto")):
        st.session_state["min_matches_maps_friends"] = 5
        st.session_state["_min_matches_maps_friends_auto"] = False
