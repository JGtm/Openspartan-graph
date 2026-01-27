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
    df_full: pd.DataFrame,
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
    df_full : pd.DataFrame
        DataFrame complet (non filtré) pour calculer les deltas.
    xuid : str | None
        XUID du joueur principal.
    db_path : str
        Chemin vers la base de données.
    db_key : tuple[int, int] | None
        Clé de cache pour la base de données.
    top_medals_fn : Callable
        Fonction pour récupérer les top médailles (signature: db_path, xuid, match_ids, top_n, db_key).
    """
    # Agrège les médailles pour les matchs filtrés.
    counts_by_medal: dict[int, int] = {}
    stats_totals: dict[str, int] = {}
    # Agrège les médailles pour TOUS les matchs (pour calculer le delta).
    counts_by_medal_full: dict[int, int] = {}
    stats_totals_full: dict[str, int] = {}

    xuid_clean = str(xuid or "").strip()

    # Charger les médailles pour le DataFrame filtré.
    if not dff.empty and xuid_clean:
        match_ids = [str(x) for x in dff["match_id"].dropna().astype(str).tolist()]
        with st.spinner("Agrégation des médailles…"):
            top_all = top_medals_fn(db_path, xuid_clean, match_ids, top_n=None, db_key=db_key)
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

    # Charger les médailles pour le DataFrame complet (pour le delta).
    if not df_full.empty and xuid_clean:
        match_ids_full = [str(x) for x in df_full["match_id"].dropna().astype(str).tolist()]
        # Utilise le cache existant si possible.
        top_all_full = top_medals_fn(db_path, xuid_clean, match_ids_full, top_n=None, db_key=db_key)
        try:
            counts_by_medal_full = {int(nid): int(cnt) for nid, cnt in (top_all_full or [])}
        except Exception:
            counts_by_medal_full = {}

        for col in ("kills", "deaths", "assists", "headshot_kills"):
            if col not in df_full.columns:
                continue
            try:
                stats_totals_full[col] = int(pd.to_numeric(df_full[col], errors="coerce").fillna(0).sum())
            except Exception:
                stats_totals_full[col] = 0

    # Calcul des deltas.
    total_medals_filtered = sum(counts_by_medal.values())
    total_medals_full = sum(counts_by_medal_full.values())
    delta_medals = total_medals_filtered  # Delta = ce qui est dans le filtre.

    total_citations_filtered = sum(stats_totals.values()) + total_medals_filtered
    total_citations_full = sum(stats_totals_full.values()) + total_medals_full
    delta_citations = total_citations_filtered

    # Détermine si on est en mode "filtré" (pas tous les matchs).
    is_filtered = len(dff) < len(df_full) if not df_full.empty else False

    # 1) Commendations Halo 5 (référentiel offline)
    # Passer les compteurs full pour afficher les deltas par citation
    render_h5g_commendations_section(
        counts_by_medal=counts_by_medal,
        stats_totals=stats_totals,
        counts_by_medal_full=counts_by_medal_full if is_filtered else None,
        stats_totals_full=stats_totals_full if is_filtered else None,
        df=dff,
        df_full=df_full if is_filtered else None,
    )
    st.divider()

    # 2) Médailles (Halo Infinite) - Affiche TOUJOURS toutes les médailles.
    st.caption("Médailles sur la sélection/filtres actuels.")
    if dff.empty:
        st.info("Aucun match disponible avec les filtres actuels.")
    else:
        top = sorted(counts_by_medal.items(), key=lambda kv: kv[1], reverse=True)

        if not top:
            st.info("Aucune médaille trouvée (ou payload médailles absent).")
        else:
            md = pd.DataFrame(top, columns=["name_id", "count"])
            md["label"] = md["name_id"].apply(lambda x: medal_label(int(x)))
            md_desc = md.sort_values("count", ascending=False)
            # Passer les deltas par médaille si filtré
            deltas = None
            if is_filtered:
                deltas = {int(nid): int(cnt) for nid, cnt in counts_by_medal.items()}
            render_medals_grid(
                md_desc[["name_id", "count"]].to_dict(orient="records"),
                cols_per_row=8,
                deltas=deltas,
            )
