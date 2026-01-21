"""UI: affichage des commendations (Halo 5 : Guardians).

Cette UI s'appuie sur le référentiel offline généré par:
- scripts/extract_h5g_commendations_fr.py

Fichiers attendus:
- data/wiki/halo5_commendations_fr.json
- static/commendations/h5g/*.png
"""

from __future__ import annotations

import json
import os
import html
import unicodedata
from typing import Any

import pandas as pd
import streamlit as st

from src.config import get_repo_root

DEFAULT_H5G_JSON_PATH = os.path.join("data", "wiki", "halo5_commendations_fr.json")
DEFAULT_H5G_EXCLUDE_PATH = os.path.join("data", "wiki", "halo5_commendations_exclude.json")


def _repo_root() -> str:
    # Robuste si le projet est lancé depuis un autre CWD.
    return get_repo_root(__file__)


def _abs_from_repo(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _normalize_name(s: str) -> str:
    base = " ".join(str(s or "").strip().lower().split())
    # Ignore les accents pour rendre les exclusions robustes (é/è/ê, etc.).
    return "".join(ch for ch in unicodedata.normalize("NFKD", base) if not unicodedata.combining(ch))


def _image_basename_from_item(item: dict[str, Any]) -> str | None:
    for k in ("image_path", "image_url", "image_file"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return os.path.basename(v.strip().replace("\\", "/"))
    return None


@st.cache_data(show_spinner=False)
def load_h5g_commendations_exclude(
    path: str = DEFAULT_H5G_EXCLUDE_PATH,
    mtime: float | None = None,
) -> tuple[set[str], set[str]]:
    abs_path = _abs_from_repo(path)
    if not os.path.exists(abs_path):
        return set(), set()

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return set(), set()

    excluded_images: set[str] = set()
    excluded_names: set[str] = set()

    def _consume(values: Any, *, as_image: bool) -> None:
        if not isinstance(values, list):
            return
        for v in values:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if as_image:
                excluded_images.add(os.path.basename(s.replace("\\", "/")))
            else:
                excluded_names.add(_normalize_name(s))

    if isinstance(raw, list):
        for v in raw:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            # Heuristique: si ça ressemble à un nom de fichier, c'est une image.
            if "." in os.path.basename(s.replace("\\", "/")):
                excluded_images.add(os.path.basename(s.replace("\\", "/")))
            else:
                excluded_names.add(_normalize_name(s))
        return excluded_images, excluded_names

    if isinstance(raw, dict):
        _consume(raw.get("image_basenames"), as_image=True)
        _consume(raw.get("names"), as_image=False)
        # Compat: certains préfèrent {items:[...]}.
        _consume(raw.get("items"), as_image=False)
        return excluded_images, excluded_names

    return set(), set()


@st.cache_data(show_spinner=False)
def load_h5g_commendations_json(path: str = DEFAULT_H5G_JSON_PATH, mtime: float | None = None) -> dict[str, Any]:
    abs_path = _abs_from_repo(path)
    if not os.path.exists(abs_path):
        return {"items": []}
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    if not isinstance(data, dict):
        return {"items": []}
    if not isinstance(data.get("items"), list):
        data["items"] = []
    return data


def _img_src(item: dict[str, Any]) -> str | None:
    # Priorité: chemin local.
    p = item.get("image_path")
    if isinstance(p, str) and p.strip():
        abs_p = _abs_from_repo(p.strip())
        if os.path.exists(abs_p):
            return abs_p
    return None


def render_h5g_commendations_section() -> None:
    abs_json = _abs_from_repo(DEFAULT_H5G_JSON_PATH)
    json_mtime = None
    try:
        json_mtime = os.path.getmtime(abs_json)
    except OSError:
        json_mtime = None

    abs_excl = _abs_from_repo(DEFAULT_H5G_EXCLUDE_PATH)
    excl_mtime = None
    try:
        excl_mtime = os.path.getmtime(abs_excl)
    except OSError:
        excl_mtime = None

    data = load_h5g_commendations_json(DEFAULT_H5G_JSON_PATH, json_mtime)
    items: list[dict[str, Any]] = list(data.get("items") or [])

    excluded_images, excluded_names = load_h5g_commendations_exclude(DEFAULT_H5G_EXCLUDE_PATH, excl_mtime)
    if items and (excluded_images or excluded_names):
        kept: list[dict[str, Any]] = []
        for it in items:
            key = _image_basename_from_item(it)
            if key and key in excluded_images:
                continue
            if _normalize_name(str(it.get("name") or "")) in excluded_names:
                continue
            kept.append(it)
        excluded_count = len(items) - len(kept)
        items = kept
    else:
        excluded_count = 0

    st.subheader("Citations")
    if not items:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(
                "Référentiel indisponible. "
                "Si le fichier JSON vient d'être créé/modifié, clique sur *Recharger* (cache Streamlit)."
            )
            st.caption(f"Chemin attendu: {abs_json}")
            if excluded_count:
                st.caption(
                    f"Note: {excluded_count} citation(s) sont exclues via {abs_excl} (blacklist)."
                )
        with c2:
            if st.button("Recharger", width="stretch"):
                load_h5g_commendations_json.clear()
                load_h5g_commendations_exclude.clear()
                st.rerun()
        return

    # Infos offline
    local_icons_dir = _abs_from_repo(os.path.join("static", "commendations", "h5g"))
    has_local_icons = os.path.isdir(local_icons_dir)
    extra = f" — {excluded_count} exclue(s)" if excluded_count else ""

    # Filtres
    cats = sorted({str(x.get("category") or "").strip() for x in items if str(x.get("category") or "").strip()})
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        picked_cat = st.selectbox("Catégorie", options=["(toutes)"] + cats, index=0)
    with c2:
        q = st.text_input("Recherche", value="", placeholder="ex: assassin, pilote, multifrag…")
    with c3:
        limit = int(st.slider("Nombre affiché", 20, 200, 60, 10))

    filtered = items
    if picked_cat != "(toutes)":
        filtered = [x for x in filtered if str(x.get("category") or "").strip() == picked_cat]
    if q.strip():
        qn = q.strip().lower()
        filtered = [
            x
            for x in filtered
            if (qn in str(x.get("name") or "").lower())
            or (qn in str(x.get("description") or "").lower())
            or (qn in str(x.get("category") or "").lower())
        ]

    filtered = filtered[:limit]

    # 8 colonnes au lieu de 6 ≈ -25% de largeur par vignette.
    cols_per_row = 8
    cols = st.columns(cols_per_row)
    for i, item in enumerate(filtered):
        col = cols[i % cols_per_row]
        name = str(item.get("name") or "").strip()
        desc = str(item.get("description") or "").strip()
        master_count = item.get("master_count")
        img = _img_src(item)
        key = _image_basename_from_item(item)

        with col:
            if img:
                st.image(img, width="stretch")
            else:
                st.markdown("<div class='os-medal-missing'>?</div>", unsafe_allow_html=True)

            tip = html.escape(desc) if desc else ""
            st.markdown(
                "<div class='os-citation-name' title='" + tip + "'>" + html.escape(name) + "</div>",
                unsafe_allow_html=True,
            )
            if master_count is not None:
                st.caption(f"Maître: {master_count}")

            with st.expander("Paliers", expanded=False):
                if key:
                    st.caption(f"ID (icône): {key}")
                tiers = item.get("tiers") or []
                if not tiers:
                    st.info("Paliers indisponibles")
                else:
                    df = pd.DataFrame(tiers)
                    # Normalise l'ordre si besoin
                    if "tier" in df.columns:
                        df = df.sort_values("tier")
                    st.dataframe(
                        df[[c for c in ("tier", "target_count", "reward") if c in df.columns]],
                        width="stretch",
                        hide_index=True,
                    )
