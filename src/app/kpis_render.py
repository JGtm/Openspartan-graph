"""Rendu des KPIs extraits de main() pour simplification.

Ce module gère:
- Le calcul des métriques de base (win rate, KPIs)
- Le rendu du bandeau résumé
- Le rendu des cartes KPI carrière
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from src.analysis import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
)
from src.analysis.stats import format_mmss
from src.ui.formatting import (
    format_duration_hms,
    format_duration_dhm,
)
from src.ui.components import (
    render_kpi_cards,
    render_top_summary,
)
from src.app.helpers import (
    avg_match_duration_seconds,
    compute_total_play_seconds,
)
from src.analysis.performance_config import PERFORMANCE_SCORE_FULL_DESC

if TYPE_CHECKING:
    pass


def render_kpis_section(dff: pd.DataFrame) -> None:
    """Rend la section complète des KPIs.
    
    Args:
        dff: DataFrame filtré des matchs.
    """
    from src.ui.perf import perf_section
    
    with perf_section("kpis"):
        rates = compute_outcome_rates(dff)
        total_outcomes = max(1, rates.total)
        win_rate = rates.wins / total_outcomes
        loss_rate = rates.losses / total_outcomes

        avg_acc = dff["accuracy"].dropna().mean() if not dff.empty else None
        global_ratio = compute_global_ratio(dff)
        avg_life = dff["average_life_seconds"].dropna().mean() if not dff.empty else None

    # Durées
    avg_match_seconds = avg_match_duration_seconds(dff)
    total_play_seconds = compute_total_play_seconds(dff)
    avg_match_txt = format_duration_hms(avg_match_seconds)
    total_play_txt = format_duration_dhm(total_play_seconds)

    # Stats par minute / totaux
    stats = compute_aggregated_stats(dff)

    # Moyennes par partie
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    # Rendu des sections
    st.subheader("Parties")
    render_top_summary(len(dff), rates)
    render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Durée totale", total_play_txt),
        ]
    )

    st.subheader("Carrière")
    render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Frags par partie", f"{kpg:.2f}" if (kpg is not None and pd.notna(kpg)) else "-"),
            ("Morts par partie", f"{dpg:.2f}" if (dpg is not None and pd.notna(dpg)) else "-"),
            ("Assistances par partie", f"{apg:.2f}" if (apg is not None and pd.notna(apg)) else "-"),
        ],
        dense=False,
    )
    render_kpi_cards(
        [
            ("Frags / min", f"{stats.kills_per_minute:.2f}" if stats.kills_per_minute else "-"),
            ("Morts / min", f"{stats.deaths_per_minute:.2f}" if stats.deaths_per_minute else "-"),
            ("Assistances / min", f"{stats.assists_per_minute:.2f}" if stats.assists_per_minute else "-"),
            ("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-"),
            ("Durée de vie moyenne", format_mmss(avg_life)),
            ("Taux de victoire", f"{win_rate*100:.1f}%" if rates.total else "-"),
            ("Taux de défaite", f"{loss_rate*100:.1f}%" if rates.total else "-"),
            ("Ratio", f"{global_ratio:.2f}" if global_ratio is not None else "-"),
        ],
        dense=False,
    )


def render_performance_info() -> None:
    """Rend l'expander d'explication du score de performance."""
    with st.expander("ℹ️ À propos du score de performance", expanded=False):
        st.markdown(PERFORMANCE_SCORE_FULL_DESC)
