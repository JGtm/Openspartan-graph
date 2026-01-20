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
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_H5G_JSON_PATH = os.path.join("data", "wiki", "halo5_commendations_fr.json")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _abs_from_repo(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


@st.cache_data(show_spinner=False)
def load_h5g_commendations_json(path: str = DEFAULT_H5G_JSON_PATH) -> dict[str, Any]:
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
    data = load_h5g_commendations_json()
    items: list[dict[str, Any]] = list(data.get("items") or [])

    st.subheader("Citations")
    if not items:
        st.info("Référentiel indisponible (génération automatique impossible ou échouée).")
        return

    # Infos offline
    local_icons_dir = _abs_from_repo(os.path.join("static", "commendations", "h5g"))
    has_local_icons = os.path.isdir(local_icons_dir)
    st.caption(
        f"Référentiel local: {len(items)} citations — "
        + ("icônes locales OK" if has_local_icons else "icônes locales absentes")
    )

    # Filtres
    cats = sorted({str(x.get("category") or "").strip() for x in items if str(x.get("category") or "").strip()})
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        picked_cat = st.selectbox("Catégorie", options=["(toutes)"] + cats, index=0)
    with c2:
        q = st.text_input("Recherche", value="", placeholder="ex: chasseur, sniper, zone de combat…")
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

    st.markdown(
        "Rang **Maître**: atteint lorsque le compteur atteint le **dernier palier** "
        "(champ `master_count`)."
    )

    cols_per_row = 6
    cols = st.columns(cols_per_row)
    for i, item in enumerate(filtered):
        col = cols[i % cols_per_row]
        name = str(item.get("name") or "").strip()
        desc = str(item.get("description") or "").strip()
        master_count = item.get("master_count")
        img = _img_src(item)

        with col:
            if img:
                st.image(img, width="stretch")
            else:
                st.markdown("<div class='os-medal-missing'>?</div>", unsafe_allow_html=True)

            st.markdown(f"**{name}**", help=desc or None)
            if master_count is not None:
                st.caption(f"Maître: {master_count}")

            with st.expander("Paliers", expanded=False):
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
