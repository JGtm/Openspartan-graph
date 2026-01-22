"""Page Citations (Commendations & Médailles)."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from src.ui.commendations import render_h5g_commendations_section
from src.ui.medals import medal_label, render_medals_grid


def render_citations_page(
    *,
    dff: pd.DataFrame,
    xuid: str | None,
    db_path: str,
    db_key: tuple[int, int] | None,
    top_medals_fn: Callable[[str, str, list[str], int | None, tuple[int, int] | None], list[tuple[int, int]] | None],
) -> None:
    """Rend la page Citations (Commendations H5G + Médailles HI).

    Parameters
    ----------
    dff : pd.DataFrame
        DataFrame filtré des matchs.
    xuid : str | None
        XUID du joueur principal.
    db_path : str
        Chemin vers la base de données.
    db_key : tuple[int, int] | None
        Clé de cache pour la base de données.
    top_medals_fn : Callable
        Fonction pour récupérer les top médailles (signature: db_path, xuid, match_ids, top_n, db_key).
    """
    # Agrège les médailles une seule fois (utilisé pour l'UI Citations + la grille Médailles).
    counts_by_medal: dict[int, int] = {}
    stats_totals: dict[str, int] = {}

    if not dff.empty and str(xuid or "").strip():
        match_ids = [str(x) for x in dff["match_id"].dropna().astype(str).tolist()]
        with st.spinner("Agrégation des médailles…"):
            top_all = top_medals_fn(db_path, xuid.strip(), match_ids, None, db_key)
        try:
            counts_by_medal = {int(nid): int(cnt) for nid, cnt in (top_all or [])}
        except Exception:
            counts_by_medal = {}

        # Stats agrégées (utilisées par certaines citations suivies via candidates.type=stat).
        for col in ("kills", "deaths", "assists", "headshot_kills"):
            if col not in dff.columns:
                continue
            try:
                stats_totals[col] = int(pd.to_numeric(dff[col], errors="coerce").fillna(0).sum())
            except Exception:
                stats_totals[col] = 0

    # 1) Commendations Halo 5 (référentiel offline)
    render_h5g_commendations_section(counts_by_medal=counts_by_medal, stats_totals=stats_totals)
    st.divider()

    # 2) Médailles (Halo Infinite) sur la sélection/filtres actuels
    st.caption("Médailles sur la sélection/filtres actuels (non limitées).")
    if dff.empty:
        st.info("Aucun match disponible avec les filtres actuels.")
    else:
        show_all = st.toggle("Afficher toutes les médailles (peut être lent)", value=False)
        top_n = None if show_all else int(st.slider("Nombre de médailles", 25, 500, 100, 25))

        top = sorted(counts_by_medal.items(), key=lambda kv: kv[1], reverse=True)
        if top_n is not None:
            top = top[:top_n]

        if not top:
            st.info("Aucune médaille trouvée (ou payload médailles absent).")
        else:
            md = pd.DataFrame(top, columns=["name_id", "count"])
            md["label"] = md["name_id"].apply(lambda x: medal_label(int(x)))
            md_desc = md.sort_values("count", ascending=False)
            render_medals_grid(md_desc[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)
