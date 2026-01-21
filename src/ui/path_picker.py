"""Widgets Streamlit pour choisir un chemin local.

Streamlit ne fournit pas (encore) de sélecteur natif de dossier multi-plateforme.
On implémente donc un petit navigateur de dossiers, utile quand l'app tourne en local.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st


def _list_windows_drives() -> list[str]:
    drives: list[str] = []
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{c}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives


def _safe_listdir_dirs(path: str) -> list[str]:
    try:
        entries = os.listdir(path)
    except Exception:
        return []

    out: list[str] = []
    for name in entries:
        try:
            p = os.path.join(path, name)
            if os.path.isdir(p):
                out.append(name)
        except Exception:
            continue
    out.sort(key=lambda s: s.casefold())
    return out


def _safe_listdir_files(path: str, *, exts: Optional[Iterable[str]] = None) -> list[str]:
    try:
        entries = os.listdir(path)
    except Exception:
        return []

    norm_exts: Optional[set[str]] = None
    if exts is not None:
        norm_exts = set()
        for e in exts:
            s = str(e or "").strip().lower()
            if not s:
                continue
            if not s.startswith("."):
                s = "." + s
            norm_exts.add(s)
        if not norm_exts:
            norm_exts = None

    out: list[str] = []
    for name in entries:
        try:
            p = os.path.join(path, name)
            if not os.path.isfile(p):
                continue
            if norm_exts is not None:
                if Path(name).suffix.lower() not in norm_exts:
                    continue
            out.append(name)
        except Exception:
            continue
    out.sort(key=lambda s: s.casefold())
    return out


def directory_input(
    label: str,
    *,
    value: str = "",
    key: str,
    help: str | None = None,
    placeholder: str = "",
    start_path: str | None = None,
) -> str:
    """Champ de saisie + bouton Parcourir pour choisir un dossier.

    Args:
        label: Libellé.
        value: Valeur initiale.
        key: Base key (stable) pour session_state.
        help: Texte d'aide.
        placeholder: Placeholder du champ.
        start_path: Dossier initial du navigateur si `value` est vide.

    Returns:
        Chemin choisi (string).
    """

    text_key = f"{key}__text"
    browse_open_key = f"{key}__browse_open"
    browse_path_key = f"{key}__browse_path"
    jump_key = f"{key}__jump"
    subdir_key = f"{key}__subdir"
    drive_key = f"{key}__drive"

    if text_key not in st.session_state:
        st.session_state[text_key] = str(value or "")

    cols = st.columns([6, 1])
    with cols[0]:
        st.text_input(
            label,
            key=text_key,
            help=help,
            placeholder=placeholder,
        )
    with cols[1]:
        if st.button("Parcourir", key=f"{key}__browse_btn", width="stretch"):
            st.session_state[browse_open_key] = not bool(st.session_state.get(browse_open_key, False))

    if not bool(st.session_state.get(browse_open_key, False)):
        return str(st.session_state.get(text_key, "") or "").strip()

    # Determine current browse path
    current_text = str(st.session_state.get(text_key, "") or "").strip()

    def _pick_initial_path() -> str:
        candidates: list[str] = []
        if current_text:
            candidates.append(current_text)
        if start_path:
            candidates.append(str(start_path))
        try:
            candidates.append(str(Path.cwd()))
        except Exception:
            pass
        try:
            candidates.append(str(Path.home()))
        except Exception:
            pass

        for c in candidates:
            p = str(c or "").strip()
            if p and os.path.isdir(p):
                return p

        # fallback Windows drive
        if os.name == "nt":
            drives = _list_windows_drives()
            if drives:
                return drives[0]

        return os.path.abspath(os.sep)

    if browse_path_key not in st.session_state:
        st.session_state[browse_path_key] = _pick_initial_path()

    browse_path = str(st.session_state.get(browse_path_key, "") or "").strip()
    if not browse_path or not os.path.isdir(browse_path):
        browse_path = _pick_initial_path()
        st.session_state[browse_path_key] = browse_path

    with st.container(border=True):
        st.caption("Sélecteur de dossier")

        if os.name == "nt":
            drives = _list_windows_drives()
            if drives:
                cur_drive = None
                try:
                    cur_drive = str(Path(browse_path).drive) + "\\"
                except Exception:
                    cur_drive = None
                if cur_drive not in drives:
                    cur_drive = drives[0]

                pick = st.selectbox(
                    "Lecteur",
                    options=drives,
                    index=drives.index(cur_drive) if cur_drive in drives else 0,
                    key=drive_key,
                )
                if isinstance(pick, str) and pick and pick != cur_drive:
                    st.session_state[browse_path_key] = pick
                    st.rerun()

        st.code(browse_path, language=None)

        nav = st.columns([1, 1, 2])
        if nav[0].button("Monter", key=f"{key}__up", width="stretch"):
            parent = str(Path(browse_path).parent)
            if parent and os.path.isdir(parent) and parent != browse_path:
                st.session_state[browse_path_key] = parent
                st.rerun()

        if nav[1].button("Choisir", key=f"{key}__choose", width="stretch"):
            st.session_state[text_key] = browse_path
            st.session_state[browse_open_key] = False
            st.rerun()

        with nav[2]:
            st.text_input("Aller à", key=jump_key, value="", placeholder="Colle un chemin, ex: C:\\Users\\...\\Videos")
            if st.button("Aller", key=f"{key}__go", width="stretch"):
                dest = str(st.session_state.get(jump_key, "") or "").strip().strip('"')
                if dest and os.path.isdir(dest):
                    st.session_state[browse_path_key] = dest
                    st.session_state[jump_key] = ""
                    st.rerun()
                else:
                    st.warning("Chemin invalide ou inaccessible.")

        subdirs = _safe_listdir_dirs(browse_path)
        if not subdirs:
            st.info("Aucun sous-dossier (ou accès refusé).")
        else:
            picked = st.selectbox("Sous-dossier", options=subdirs, index=0, key=subdir_key)
            if st.button("Ouvrir", key=f"{key}__open", width="stretch"):
                if isinstance(picked, str) and picked:
                    nxt = os.path.join(browse_path, picked)
                    if os.path.isdir(nxt):
                        st.session_state[browse_path_key] = nxt
                        st.rerun()

        if st.button("Fermer", key=f"{key}__close", width="stretch"):
            st.session_state[browse_open_key] = False
            st.rerun()

    return str(st.session_state.get(text_key, "") or "").strip()


def file_input(
    label: str,
    *,
    value: str = "",
    key: str,
    help: str | None = None,
    placeholder: str = "",
    start_path: str | None = None,
    exts: Optional[Iterable[str]] = None,
) -> str:
    """Champ de saisie + bouton Parcourir pour choisir un fichier.

    Note: ce sélecteur parcourt le système de fichiers *du serveur Streamlit*.
    En usage local, c'est ton PC. En déploiement distant, ce sera la machine distante.
    """

    browse_open_key = f"{key}__browse_open"
    browse_path_key = f"{key}__browse_path"
    jump_key = f"{key}__jump"
    subdir_key = f"{key}__subdir"
    file_key = f"{key}__file"
    drive_key = f"{key}__drive"

    if key not in st.session_state:
        st.session_state[key] = str(value or "")

    cols = st.columns([6, 1])
    with cols[0]:
        st.text_input(
            label,
            key=key,
            help=help,
            placeholder=placeholder,
        )
    with cols[1]:
        if st.button("Parcourir", key=f"{key}__browse_btn", width="stretch"):
            st.session_state[browse_open_key] = not bool(st.session_state.get(browse_open_key, False))

    if not bool(st.session_state.get(browse_open_key, False)):
        return str(st.session_state.get(key, "") or "").strip()

    current_text = str(st.session_state.get(key, "") or "").strip()

    def _pick_initial_dir() -> str:
        candidates: list[str] = []
        if current_text:
            try:
                p = Path(current_text)
                if p.is_file():
                    candidates.append(str(p.parent))
                else:
                    candidates.append(str(p))
            except Exception:
                pass
        if start_path:
            candidates.append(str(start_path))
        try:
            candidates.append(str(Path.cwd()))
        except Exception:
            pass
        try:
            candidates.append(str(Path.home()))
        except Exception:
            pass

        for c in candidates:
            d = str(c or "").strip()
            if d and os.path.isdir(d):
                return d

        if os.name == "nt":
            drives = _list_windows_drives()
            if drives:
                return drives[0]
        return os.path.abspath(os.sep)

    if browse_path_key not in st.session_state:
        st.session_state[browse_path_key] = _pick_initial_dir()

    browse_path = str(st.session_state.get(browse_path_key, "") or "").strip()
    if not browse_path or not os.path.isdir(browse_path):
        browse_path = _pick_initial_dir()
        st.session_state[browse_path_key] = browse_path

    with st.container(border=True):
        st.caption("Sélecteur de fichier")

        if os.name == "nt":
            drives = _list_windows_drives()
            if drives:
                cur_drive = None
                try:
                    cur_drive = str(Path(browse_path).drive) + "\\"
                except Exception:
                    cur_drive = None
                if cur_drive not in drives:
                    cur_drive = drives[0]

                pick = st.selectbox(
                    "Lecteur",
                    options=drives,
                    index=drives.index(cur_drive) if cur_drive in drives else 0,
                    key=drive_key,
                )
                if isinstance(pick, str) and pick and pick != cur_drive:
                    st.session_state[browse_path_key] = pick
                    st.rerun()

        st.code(browse_path, language=None)

        nav = st.columns([1, 2])
        if nav[0].button("Monter", key=f"{key}__up", width="stretch"):
            parent = str(Path(browse_path).parent)
            if parent and os.path.isdir(parent) and parent != browse_path:
                st.session_state[browse_path_key] = parent
                st.rerun()

        with nav[1]:
            st.text_input("Aller à", key=jump_key, value="", placeholder="Colle un dossier, ex: C:\\Users\\...\\Downloads")
            if st.button("Aller", key=f"{key}__go", width="stretch"):
                dest = str(st.session_state.get(jump_key, "") or "").strip().strip('"')
                if dest and os.path.isdir(dest):
                    st.session_state[browse_path_key] = dest
                    st.session_state[jump_key] = ""
                    st.rerun()
                else:
                    st.warning("Chemin invalide ou inaccessible.")

        subdirs = _safe_listdir_dirs(browse_path)
        if subdirs:
            picked = st.selectbox("Sous-dossier", options=subdirs, index=0, key=subdir_key)
            if st.button("Ouvrir", key=f"{key}__open", width="stretch"):
                if isinstance(picked, str) and picked:
                    nxt = os.path.join(browse_path, picked)
                    if os.path.isdir(nxt):
                        st.session_state[browse_path_key] = nxt
                        st.rerun()
        else:
            st.info("Aucun sous-dossier (ou accès refusé).")

        files = _safe_listdir_files(browse_path, exts=exts)
        if not files:
            st.info("Aucun fichier correspondant dans ce dossier.")
        else:
            picked_file = st.selectbox("Fichier", options=files, index=0, key=file_key)
            if st.button("Choisir", key=f"{key}__choose", width="stretch"):
                if isinstance(picked_file, str) and picked_file:
                    full = os.path.join(browse_path, picked_file)
                    st.session_state[key] = full
                    st.session_state[browse_open_key] = False
                    st.rerun()

        if st.button("Fermer", key=f"{key}__close", width="stretch"):
            st.session_state[browse_open_key] = False
            st.rerun()

    return str(st.session_state.get(key, "") or "").strip()
