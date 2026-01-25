"""Composants UI pour les scores de performance.

Ce module contient les fonctions de calcul et d'affichage
des scores de performance pour les sessions de jeu.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from src.analysis.performance_score import (
    compute_session_performance_score_v1,
    compute_session_performance_score_v2,
)
from src.analysis.performance_config import SCORE_THRESHOLDS, SCORE_LABELS


def compute_session_performance_score(df_session: pd.DataFrame) -> dict:
    """Calcule un score de performance (0-100) pour une session.
    
    Le score est une moyenne pondérée de :
    - K/D ratio normalisé (30%)
    - Win rate (25%)
    - Précision moyenne (25%)
    - Score moyen par partie normalisé (20%)
    
    Args:
        df_session: DataFrame contenant les matchs de la session.
        
    Returns:
        Dict avec score global et composantes détaillées.
    """
    return compute_session_performance_score_v1(df_session)


def compute_session_performance_score_v2_ui(
    df_session: pd.DataFrame,
    *,
    include_mmr_adjustment: bool = True,
) -> dict:
    """Calcule un score de performance v2 (0–100) pour une session.

    Version pensée pour être réutilisée ailleurs que dans la page de comparaison.
    """
    return compute_session_performance_score_v2(
        df_session,
        include_mmr_adjustment=include_mmr_adjustment,
    )


def get_score_color(score: float | None) -> str:
    """Retourne la couleur CSS selon le score de performance."""
    if score is None:
        return "var(--color-neutral)"
    if score >= SCORE_THRESHOLDS["excellent"]:
        return "var(--color-excellent)"
    if score >= SCORE_THRESHOLDS["good"]:
        return "var(--color-good)"
    if score >= SCORE_THRESHOLDS["average"]:
        return "var(--color-average)"
    if score >= SCORE_THRESHOLDS["below_average"]:
        return "var(--color-poor)"
    return "var(--color-bad)"


def get_score_class(score: float | None) -> str:
    """Retourne la classe CSS selon le score de performance."""
    if score is None:
        return "text-neutral"
    if score >= SCORE_THRESHOLDS["excellent"]:
        return "text-excellent"
    if score >= SCORE_THRESHOLDS["good"]:
        return "text-good"
    if score >= SCORE_THRESHOLDS["average"]:
        return "text-average"
    if score >= SCORE_THRESHOLDS["below_average"]:
        return "text-poor"
    return "text-bad"


def get_score_label(score: float | None) -> str:
    """Retourne le label textuel selon le score."""
    if score is None:
        return "N/A"
    if score >= SCORE_THRESHOLDS["excellent"]:
        return SCORE_LABELS["excellent"]
    if score >= SCORE_THRESHOLDS["good"]:
        return SCORE_LABELS["good"]
    if score >= SCORE_THRESHOLDS["average"]:
        return SCORE_LABELS["average"]
    if score >= SCORE_THRESHOLDS["below_average"]:
        return SCORE_LABELS["below_average"]
    return SCORE_LABELS["bad"]


def render_performance_score_card(
    label: str,
    perf: dict,
    is_better: bool | None = None,
) -> None:
    """Affiche une grande carte avec le score de performance.
    
    Args:
        label: Titre de la carte (ex: "Session A").
        perf: Dict retourné par compute_session_performance_score.
        is_better: True si cette session est meilleure, False si pire, None si pas de comparaison.
    """
    score = perf.get("score")
    score_class = get_score_class(score)
    score_label = get_score_label(score)
    score_display = f"{score:.0f}" if score is not None else "—"
    
    # Indicateur de comparaison
    badge = ""
    if is_better is True:
        badge = "<span class='text-positive text-lg' style='margin-left: 8px;'>▲</span>"
    elif is_better is False:
        badge = "<span class='text-negative text-lg' style='margin-left: 8px;'>▼</span>"
    
    st.markdown(
        f"""
        <div class="os-perf-card">
            <div class="os-perf-card__label">{label}</div>
            <div class="os-perf-card__score {score_class}">{score_display}{badge}</div>
            <div class="os-perf-card__status {score_class}">{score_label}</div>
            <div class="os-perf-card__meta">{perf.get('matches', 0)} parties</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_comparison_row(
    label: str,
    val_a,
    val_b,
    fmt: str | Callable = "{}",
    higher_is_better: bool = True,
) -> None:
    """Affiche une ligne de métrique avec comparaison colorée.
    
    Args:
        label: Nom de la métrique.
        val_a: Valeur pour la session A.
        val_b: Valeur pour la session B.
        fmt: Format string pour l'affichage OU une fonction callable.
        higher_is_better: Si True, la valeur la plus haute est verte.
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Déterminer les classes CSS
    class_a, class_b = "os-metric-value--neutral", "os-metric-value--neutral"
    if val_a is not None and val_b is not None:
        if higher_is_better:
            if val_a > val_b:
                class_a = "os-metric-value--better"
            elif val_b > val_a:
                class_b = "os-metric-value--better"
        else:
            if val_a < val_b:
                class_a = "os-metric-value--better"
            elif val_b < val_a:
                class_b = "os-metric-value--better"
    
    # Fonction de formatage
    def _format_value(val):
        if val is None:
            return "—"
        if callable(fmt):
            return fmt(val)
        return fmt.format(val)
    
    with col1:
        st.markdown(f"**{label}**")
    with col2:
        display_a = _format_value(val_a)
        st.markdown(
            f"<span class='os-metric-value {class_a}'>{display_a}</span>",
            unsafe_allow_html=True,
        )
    with col3:
        display_b = _format_value(val_b)
        st.markdown(
            f"<span class='os-metric-value {class_b}'>{display_b}</span>",
            unsafe_allow_html=True,
        )
