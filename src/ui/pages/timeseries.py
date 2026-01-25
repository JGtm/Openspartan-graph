"""Page Séries temporelles.

Graphes d'évolution des statistiques dans le temps.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.visualization.timeseries import (
    plot_assists_timeseries,
    plot_average_life,
    plot_per_minute_timeseries,
    plot_spree_headshots_accuracy,
    plot_timeseries,
    plot_performance_timeseries,
)
from src.visualization.distributions import plot_kda_distribution


def render_timeseries_page(dff: pd.DataFrame, df_full: pd.DataFrame | None = None) -> None:
    """Affiche la page Séries temporelles.

    Args:
        dff: DataFrame filtré des matchs.
        df_full: DataFrame complet pour le calcul du score relatif.
    """
    with st.spinner("Génération des graphes…"):
        fig = plot_timeseries(dff)
        st.plotly_chart(fig, width="stretch")

        st.subheader("FDA")
        valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
        if valid.empty:
            st.info("FDA indisponible sur ce filtre.")
        else:
            m = st.columns(1)
            m[0].metric("KDA moyen", f"{valid['kda'].mean():.2f}", label_visibility="collapsed")
            st.plotly_chart(plot_kda_distribution(dff), width="stretch")

        st.subheader("Performance")
        history = df_full if df_full is not None else dff
        st.plotly_chart(plot_performance_timeseries(dff, df_history=history), width="stretch")

        st.subheader("Assistances")
        st.plotly_chart(plot_assists_timeseries(dff), width="stretch")

        st.subheader("Stats par minute")
        st.plotly_chart(
            plot_per_minute_timeseries(dff),
            width="stretch",
        )

        st.subheader("Durée de vie moyenne")
        if dff.dropna(subset=["average_life_seconds"]).empty:
            st.info("Average Life indisponible sur ce filtre.")
        else:
            st.plotly_chart(plot_average_life(dff), width="stretch")

        st.subheader("Folie meurtrière / Tirs à la tête / Précision")
        st.plotly_chart(plot_spree_headshots_accuracy(dff), width="stretch")
