"""Page de comparaison de sessions.

Cette page permet de comparer les performances entre deux sessions de jeu,
avec des graphiques radar, des barres comparatives, et un tableau historique.
"""

from __future__ import annotations

import locale
from typing import TYPE_CHECKING, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import translate_pair_name
from src.ui.components.performance import (
    compute_session_performance_score,
    compute_session_performance_score_v2_ui,
    render_performance_score_card,
    render_metric_comparison_row,
    get_score_class,
)
from src.analysis.performance_score import compute_performance_series
from src.analysis.performance_config import PERFORMANCE_SCORE_SHORT_DESC

if TYPE_CHECKING:
    pass


def _format_seconds_to_mmss(seconds) -> str:
    """Formate des secondes en mm:ss ou retourne la valeur formatée."""
    if seconds is None:
        return "—"
    try:
        total = int(round(float(seconds)))
        if total < 0:
            return "—"
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"
    except Exception:
        return "—"


def _format_date_with_weekday(dt) -> str:
    """Formate une date avec jour de la semaine abrégé : lun. 12/01/26 14:30."""
    if dt is None or pd.isna(dt):
        return "-"
    try:
        # Jours en français
        weekdays_fr = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
        wd = weekdays_fr[dt.weekday()]
        return f"{wd} {dt.strftime('%d/%m/%y %H:%M')}"
    except Exception:
        return "-"


def _outcome_class(label: str) -> str:
    """Retourne la classe CSS pour un résultat."""
    v = str(label or "").strip().casefold()
    if v.startswith("victoire"):
        return "text-win"
    if v.startswith("défaite") or v.startswith("defaite"):
        return "text-loss"
    if v.startswith("égalité") or v.startswith("egalite"):
        return "text-tie"
    if v.startswith("non"):
        return "text-nf"
    return ""


def render_session_history_table(
    df_sess: pd.DataFrame,
    session_name: str,
    df_full: pd.DataFrame | None = None,
) -> None:
    """Affiche le tableau historique d'une session.
    
    Args:
        df_sess: DataFrame de la session.
        session_name: Nom de la session pour les messages.
        df_full: DataFrame complet pour le calcul du score relatif.
    """
    if df_sess.empty:
        st.info(f"Aucune partie dans {session_name}.")
        return
    
    # Copie pour éviter les warnings pandas
    df_sess = df_sess.copy()
    
    # Traduire le mode si non traduit
    if "pair_fr" not in df_sess.columns and "pair_name" in df_sess.columns:
        df_sess["pair_fr"] = df_sess["pair_name"].apply(translate_pair_name)
    
    # Préparer les colonnes à afficher
    display_cols = []
    col_map = {}
    
    if "start_time" in df_sess.columns:
        df_sess["Heure"] = df_sess["start_time"].apply(_format_date_with_weekday)
        display_cols.append("Heure")
    
    if "pair_fr" in df_sess.columns:
        col_map["pair_fr"] = "Mode"
        display_cols.append("pair_fr")
    elif "pair_name" in df_sess.columns:
        # Traduire à la volée si pair_fr n'existe pas
        df_sess["mode_traduit"] = df_sess["pair_name"].apply(translate_pair_name)
        col_map["mode_traduit"] = "Mode"
        display_cols.append("mode_traduit")
    
    if "map_ui" in df_sess.columns:
        col_map["map_ui"] = "Carte"
        display_cols.append("map_ui")
    elif "map_name" in df_sess.columns:
        col_map["map_name"] = "Carte"
        display_cols.append("map_name")
    
    for c in ["kills", "deaths", "assists"]:
        if c in df_sess.columns:
            col_map[c] = {"kills": "Frags", "deaths": "Morts", "assists": "Assists"}[c]
            display_cols.append(c)
    
    if "outcome" in df_sess.columns:
        outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
        df_sess["Résultat"] = df_sess["outcome"].map(outcome_map).fillna("-")
        display_cols.append("Résultat")
    
    # Colonne Performance RELATIVE (après Résultat)
    history_df = df_full if df_full is not None else df_sess
    df_sess["Performance"] = compute_performance_series(df_sess, history_df)
    df_sess["Perf_display"] = df_sess["Performance"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "-"
    )
    display_cols.append("Perf_display")
    col_map["Perf_display"] = "Performance"
    
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
    
    # Garder les scores de performance pour la coloration
    perf_scores = df_sess["Performance"].values if "Performance" in df_sess.columns else None
    
    # Trier par heure (conserver l'ordre chronologique via start_time)
    if "start_time" in df_sess.columns:
        sort_indices = df_sess["start_time"].argsort().values
        df_display = df_display.iloc[sort_indices]
        if perf_scores is not None:
            perf_scores = perf_scores[sort_indices]
    
    # Affichage HTML pour styliser les résultats
    import html as html_lib
    html_rows = []
    for idx, (_, row) in enumerate(df_display.iterrows()):
        cells = []
        for col in df_display.columns:
            val = row[col]
            if col == "Résultat":
                css_class = _outcome_class(str(val))
                cells.append(f"<td class='{css_class}'>{html_lib.escape(str(val))}</td>")
            elif col == "Performance":
                # Coloration selon le score
                perf_val = perf_scores[idx] if perf_scores is not None else None
                css_class = get_score_class(perf_val)
                cells.append(f"<td class='{css_class}'>{html_lib.escape(str(val) if val is not None else '-')}</td>")
            else:
                cells.append(f"<td>{html_lib.escape(str(val) if val is not None else '-')}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    
    header_cells = "".join(f"<th>{html_lib.escape(c)}</th>" for c in df_display.columns)
    html_table = f"""
    <table class="session-history-table">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)


def render_comparison_radar_chart(perf_a: dict, perf_b: dict) -> None:
    """Affiche le radar chart comparatif.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
    """
    categories = ["K/D", "Victoire %", "Précision"]
    
    def _normalize_for_radar(kd, wr, acc):
        kd_norm = min(100, (kd or 0) * 50)  # K/D 2.0 = 100
        wr_norm = wr or 0  # Déjà en %
        acc_norm = acc if acc is not None else 50  # Déjà en %
        return [kd_norm, wr_norm, acc_norm]
    
    values_a = _normalize_for_radar(
        perf_a["kd_ratio"], perf_a["win_rate"], perf_a["accuracy"]
    )
    values_b = _normalize_for_radar(
        perf_b["kd_ratio"], perf_b["win_rate"], perf_b["accuracy"]
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
    
    st.plotly_chart(fig_radar, width="stretch")


def render_comparison_bar_chart(perf_a: dict, perf_b: dict) -> None:
    """Affiche le graphique en barres comparatif.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
    """
    metrics_labels = ["K/D Ratio", "Victoire (%)"]
    values_a_bar = [
        perf_a["kd_ratio"] or 0,
        perf_a["win_rate"] or 0,
    ]
    values_b_bar = [
        perf_b["kd_ratio"] or 0,
        perf_b["win_rate"] or 0,
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
    
    st.plotly_chart(fig_bar, width="stretch")


def render_session_comparison_page(
    all_sessions_df: pd.DataFrame,
    df_full: pd.DataFrame | None = None,
) -> None:
    """Rend la page de comparaison de sessions.
    
    Args:
        all_sessions_df: DataFrame avec toutes les sessions (colonnes session_id, session_label).
        df_full: DataFrame complet pour le calcul du score relatif.
    """
    st.caption("Compare les performances entre deux sessions de jeu.")
    
    # Note explicative sur le score de performance
    with st.expander("ℹ️ À propos du score de performance", expanded=False):
        st.markdown(PERFORMANCE_SCORE_SHORT_DESC)
    
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
    
    # Calculer les scores de performance (v2)
    perf_a = compute_session_performance_score_v2_ui(df_session_a)
    perf_b = compute_session_performance_score_v2_ui(df_session_b)
    
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
    render_metric_comparison_row("FDA (Frags-Décès-Assists)", perf_a["kda"], perf_b["kda"], "{:.2f}")
    render_metric_comparison_row("Taux de victoire", perf_a["win_rate"], perf_b["win_rate"], "{:.1f}%")
    render_metric_comparison_row("Durée de vie moyenne", perf_a["avg_life_seconds"], perf_b["avg_life_seconds"], _format_seconds_to_mmss)
    
    st.markdown("---")
    
    render_metric_comparison_row("Total des frags", perf_a["kills"], perf_b["kills"], "{}")
    render_metric_comparison_row(
        "Total des morts", perf_a["deaths"], perf_b["deaths"], "{}", higher_is_better=False
    )
    render_metric_comparison_row("Total des assistances", perf_a["assists"], perf_b["assists"], "{}")
    
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
    # Graphiques comparatifs (côte à côte)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📈 Graphiques comparatifs")
    
    col_radar, col_bars = st.columns(2)
    
    with col_radar:
        st.markdown("#### Vue radar")
        render_comparison_radar_chart(perf_a, perf_b)
    
    with col_bars:
        st.markdown("#### Comparaison par métrique")
        render_comparison_bar_chart(perf_a, perf_b)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Tableau historique des parties
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Historique des parties")
    
    tab_hist_a, tab_hist_b = st.tabs(["Session A", "Session B"])
    
    with tab_hist_a:
        render_session_history_table(df_session_a, "Session A", df_full=df_full)
    
    with tab_hist_b:
        render_session_history_table(df_session_b, "Session B", df_full=df_full)
