"""Page de comparaison de sessions.

Cette page permet de comparer les performances entre deux sessions de jeu,
avec des graphiques radar, des barres comparatives, et un tableau historique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.components.performance import (
    compute_session_performance_score,
    render_performance_score_card,
    render_metric_comparison_row,
)

if TYPE_CHECKING:
    pass


def render_session_history_table(df_sess: pd.DataFrame, session_name: str) -> None:
    """Affiche le tableau historique d'une session.
    
    Args:
        df_sess: DataFrame de la session.
        session_name: Nom de la session pour les messages.
    """
    if df_sess.empty:
        st.info(f"Aucune partie dans {session_name}.")
        return
    
    # Copie pour éviter les warnings pandas
    df_sess = df_sess.copy()
    
    # Préparer les colonnes à afficher
    display_cols = []
    col_map = {}
    
    if "start_time" in df_sess.columns:
        df_sess["Heure"] = df_sess["start_time"].apply(
            lambda x: x.strftime("%H:%M") if pd.notna(x) else "-"
        )
        display_cols.append("Heure")
    
    if "pair_fr" in df_sess.columns:
        col_map["pair_fr"] = "Mode"
        display_cols.append("pair_fr")
    elif "pair_name" in df_sess.columns:
        col_map["pair_name"] = "Mode"
        display_cols.append("pair_name")
    
    if "map_ui" in df_sess.columns:
        col_map["map_ui"] = "Carte"
        display_cols.append("map_ui")
    elif "map_name" in df_sess.columns:
        col_map["map_name"] = "Carte"
        display_cols.append("map_name")
    
    for c in ["kills", "deaths", "assists"]:
        if c in df_sess.columns:
            col_map[c] = c.capitalize()
            display_cols.append(c)
    
    if "outcome" in df_sess.columns:
        outcome_map = {2: "✅", 3: "❌", 1: "🟰", 4: "⏹️"}
        df_sess["Résultat"] = df_sess["outcome"].map(outcome_map).fillna("-")
        display_cols.append("Résultat")
    
    if "team_mmr" in df_sess.columns:
        df_sess["MMR Équipe"] = df_sess["team_mmr"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
        display_cols.append("MMR Équipe")
    
    if "enemy_mmr" in df_sess.columns:
        df_sess["MMR Adverse"] = df_sess["enemy_mmr"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
        display_cols.append("MMR Adverse")
    
    # Renommer les colonnes
    df_display = df_sess[display_cols].copy()
    df_display = df_display.rename(columns=col_map)
    
    # Trier par heure
    if "Heure" in df_display.columns:
        df_display = df_display.sort_values("Heure", ascending=True)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_comparison_radar_chart(perf_a: dict, perf_b: dict) -> None:
    """Affiche le radar chart comparatif.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
    """
    categories = ["K/D", "Win%", "Précision", "Score/match"]
    
    def _normalize_for_radar(kd, wr, acc, score):
        kd_norm = min(100, (kd or 0) * 50)  # K/D 2.0 = 100
        wr_norm = wr or 0  # Déjà en %
        acc_norm = acc or 50  # Déjà en %
        score_norm = min(100, (score or 10) * 5)  # Score 20 = 100
        return [kd_norm, wr_norm, acc_norm, score_norm]
    
    values_a = _normalize_for_radar(
        perf_a["kd_ratio"], perf_a["win_rate"], perf_a["accuracy"], perf_a["avg_score"]
    )
    values_b = _normalize_for_radar(
        perf_b["kd_ratio"], perf_b["win_rate"], perf_b["accuracy"], perf_b["avg_score"]
    )
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name='Session A',
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.3)',
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Session B',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)',
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        height=400,
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)


def render_comparison_bar_chart(perf_a: dict, perf_b: dict) -> None:
    """Affiche le graphique en barres comparatif.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
    """
    metrics_labels = ["K/D Ratio", "Win Rate (%)", "Précision (%)", "Score/match"]
    values_a_bar = [
        perf_a["kd_ratio"] or 0,
        perf_a["win_rate"] or 0,
        perf_a["accuracy"] or 0,
        perf_a["avg_score"] or 0,
    ]
    values_b_bar = [
        perf_b["kd_ratio"] or 0,
        perf_b["win_rate"] or 0,
        perf_b["accuracy"] or 0,
        perf_b["avg_score"] or 0,
    ]
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Session A',
        x=metrics_labels,
        y=values_a_bar,
        marker_color='#2196F3',
    ))
    fig_bar.add_trace(go.Bar(
        name='Session B',
        x=metrics_labels,
        y=values_b_bar,
        marker_color='#4CAF50',
    ))
    
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=350,
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)


def render_session_comparison_page(
    all_sessions_df: pd.DataFrame,
) -> None:
    """Rend la page de comparaison de sessions.
    
    Args:
        all_sessions_df: DataFrame avec toutes les sessions (colonnes session_id, session_label).
    """
    st.caption("Compare les performances entre deux sessions de jeu.")
    
    if all_sessions_df.empty:
        st.info("Aucune session disponible.")
        return
    
    # Liste des sessions triées (plus récente en premier)
    session_info = (
        all_sessions_df.groupby(["session_id", "session_label"])
        .size()
        .reset_index(name="count")
        .sort_values("session_id", ascending=False)
    )
    session_labels = session_info["session_label"].tolist()
    
    if len(session_labels) < 2:
        st.warning("Il faut au moins 2 sessions pour comparer.")
        return
    
    # Sélecteurs de sessions
    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        # Session A = avant-dernière par défaut
        default_a = session_labels[1] if len(session_labels) > 1 else session_labels[0]
        session_a_label = st.selectbox(
            "Session A (référence)",
            options=session_labels,
            index=session_labels.index(default_a) if default_a in session_labels else 1,
            key="compare_session_a",
        )
    with col_sel_b:
        # Session B = dernière par défaut
        session_b_label = st.selectbox(
            "Session B (à comparer)",
            options=session_labels,
            index=0,
            key="compare_session_b",
        )
    
    # Filtrer les DataFrames
    df_session_a = all_sessions_df[all_sessions_df["session_label"] == session_a_label].copy()
    df_session_b = all_sessions_df[all_sessions_df["session_label"] == session_b_label].copy()
    
    # Calculer les scores de performance
    perf_a = compute_session_performance_score(df_session_a)
    perf_b = compute_session_performance_score(df_session_b)
    
    # Déterminer qui est meilleur
    score_a = perf_a.get("score")
    score_b = perf_b.get("score")
    is_b_better = None
    if score_a is not None and score_b is not None:
        if score_b > score_a:
            is_b_better = True
        elif score_b < score_a:
            is_b_better = False
    
    # ══════════════════════════════════════════════════════════════════════════
    # Grandes cartes de score côte à côte
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏆 Score de performance")
    col_score_a, col_score_b = st.columns(2)
    with col_score_a:
        render_performance_score_card(
            "Session A",
            perf_a,
            is_better=not is_b_better if is_b_better is not None else None,
        )
    with col_score_b:
        render_performance_score_card("Session B", perf_b, is_better=is_b_better)
    
    st.markdown("---")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Colonnes Session A / Session B avec métriques détaillées
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Métriques détaillées")
    
    # En-têtes
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.markdown("**Métrique**")
    with col_h2:
        st.markdown("**Session A**")
    with col_h3:
        st.markdown("**Session B**")
    
    st.markdown("---")
    
    render_metric_comparison_row("Nombre de parties", perf_a["matches"], perf_b["matches"], "{}")
    render_metric_comparison_row("K/D Ratio", perf_a["kd_ratio"], perf_b["kd_ratio"], "{:.2f}")
    render_metric_comparison_row("Taux de victoire", perf_a["win_rate"], perf_b["win_rate"], "{:.1f}%")
    render_metric_comparison_row("Précision moyenne", perf_a["accuracy"], perf_b["accuracy"], "{:.1f}%")
    render_metric_comparison_row("Score moyen/partie", perf_a["avg_score"], perf_b["avg_score"], "{:.1f}")
    
    st.markdown("---")
    
    render_metric_comparison_row("Total Kills", perf_a["kills"], perf_b["kills"], "{}")
    render_metric_comparison_row(
        "Total Deaths", perf_a["deaths"], perf_b["deaths"], "{}", higher_is_better=False
    )
    render_metric_comparison_row("Total Assists", perf_a["assists"], perf_b["assists"], "{}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Comparaison MMR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🎯 Comparaison MMR")
    
    col_mmr1, col_mmr2, col_mmr3 = st.columns([2, 1, 1])
    with col_mmr1:
        st.markdown("**Métrique MMR**")
    with col_mmr2:
        st.markdown("**Session A**")
    with col_mmr3:
        st.markdown("**Session B**")
    
    render_metric_comparison_row(
        "MMR équipe (moy)", perf_a["team_mmr_avg"], perf_b["team_mmr_avg"], "{:.1f}"
    )
    render_metric_comparison_row(
        "MMR adverse (moy)",
        perf_a["enemy_mmr_avg"],
        perf_b["enemy_mmr_avg"],
        "{:.1f}",
        higher_is_better=False,
    )
    render_metric_comparison_row(
        "Écart MMR (moy)", perf_a["delta_mmr_avg"], perf_b["delta_mmr_avg"], "{:+.1f}"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # Graphiques comparatifs
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📈 Graphiques comparatifs")
    
    render_comparison_radar_chart(perf_a, perf_b)
    
    st.markdown("#### Comparaison par métrique")
    render_comparison_bar_chart(perf_a, perf_b)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Tableau historique des parties
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Historique des parties")
    
    tab_hist_a, tab_hist_b = st.tabs(["Session A", "Session B"])
    
    with tab_hist_a:
        render_session_history_table(df_session_a, "Session A")
    
    with tab_hist_b:
        render_session_history_table(df_session_b, "Session B")
