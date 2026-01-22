"""Composants UI pour les scores de performance.

Ce module contient les fonctions de calcul et d'affichage
des scores de performance pour les sessions de jeu.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


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
    if df_session is None or df_session.empty:
        return {
            "score": None,
            "kd_ratio": None,
            "win_rate": None,
            "accuracy": None,
            "avg_score": None,
            "matches": 0,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "team_mmr_avg": None,
            "enemy_mmr_avg": None,
            "delta_mmr_avg": None,
        }
    
    # Métriques de base
    total_kills = int(df_session["kills"].sum())
    total_deaths = int(df_session["deaths"].sum())
    total_assists = int(df_session["assists"].sum())
    n_matches = len(df_session)
    
    # K/D ratio
    kd_ratio = total_kills / total_deaths if total_deaths > 0 else float(total_kills)
    # Normalisation: K/D 0.5 = 25pts, K/D 1.0 = 50pts, K/D 2.0 = 100pts
    kd_score = min(100, max(0, kd_ratio * 50))
    
    # Win rate
    wins = len(df_session[df_session["outcome"] == 2]) if "outcome" in df_session.columns else 0
    win_rate = wins / n_matches if n_matches > 0 else 0
    win_score = win_rate * 100
    
    # Précision moyenne
    if "shots_accuracy" in df_session.columns:
        acc_values = pd.to_numeric(df_session["shots_accuracy"], errors="coerce").dropna()
        accuracy = float(acc_values.mean()) if not acc_values.empty else None
    else:
        accuracy = None
    acc_score = accuracy if accuracy is not None else 50  # Neutre si pas de données
    
    # Score moyen par partie
    if "match_score" in df_session.columns:
        score_values = pd.to_numeric(df_session["match_score"], errors="coerce").dropna()
        avg_score = float(score_values.mean()) if not score_values.empty else None
    else:
        avg_score = None
    # Normalisation: score 10 = 50pts, score 20 = 100pts
    score_pts = min(100, max(0, (avg_score or 10) * 5)) if avg_score is not None else 50
    
    # MMR moyens
    team_mmr_avg = None
    enemy_mmr_avg = None
    delta_mmr_avg = None
    if "team_mmr" in df_session.columns:
        team_vals = pd.to_numeric(df_session["team_mmr"], errors="coerce").dropna()
        team_mmr_avg = float(team_vals.mean()) if not team_vals.empty else None
    if "enemy_mmr" in df_session.columns:
        enemy_vals = pd.to_numeric(df_session["enemy_mmr"], errors="coerce").dropna()
        enemy_mmr_avg = float(enemy_vals.mean()) if not enemy_vals.empty else None
    if team_mmr_avg is not None and enemy_mmr_avg is not None:
        delta_mmr_avg = team_mmr_avg - enemy_mmr_avg
    
    # Score global pondéré
    final_score = (kd_score * 0.30) + (win_score * 0.25) + (acc_score * 0.25) + (score_pts * 0.20)
    
    return {
        "score": round(final_score, 1),
        "kd_ratio": round(kd_ratio, 2),
        "win_rate": round(win_rate * 100, 1),
        "accuracy": round(accuracy, 1) if accuracy is not None else None,
        "avg_score": round(avg_score, 1) if avg_score is not None else None,
        "matches": n_matches,
        "kills": total_kills,
        "deaths": total_deaths,
        "assists": total_assists,
        "team_mmr_avg": round(team_mmr_avg, 1) if team_mmr_avg is not None else None,
        "enemy_mmr_avg": round(enemy_mmr_avg, 1) if enemy_mmr_avg is not None else None,
        "delta_mmr_avg": round(delta_mmr_avg, 1) if delta_mmr_avg is not None else None,
    }


def get_score_color(score: float | None) -> str:
    """Retourne la couleur CSS selon le score de performance."""
    if score is None:
        return "#9E9E9E"  # Gris
    if score >= 75:
        return "#1B5E20"  # Vert foncé (excellent)
    if score >= 60:
        return "#4CAF50"  # Vert (bon)
    if score >= 45:
        return "#FF9800"  # Orange (moyen)
    if score >= 30:
        return "#F44336"  # Rouge (faible)
    return "#B71C1C"  # Rouge foncé (mauvais)


def get_score_label(score: float | None) -> str:
    """Retourne le label textuel selon le score."""
    if score is None:
        return "N/A"
    if score >= 75:
        return "Excellent"
    if score >= 60:
        return "Bon"
    if score >= 45:
        return "Moyen"
    if score >= 30:
        return "Faible"
    return "Difficile"


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
    color = get_score_color(score)
    score_label = get_score_label(score)
    score_display = f"{score:.0f}" if score is not None else "—"
    
    # Indicateur de comparaison
    badge = ""
    if is_better is True:
        badge = "<span style='color: #1B5E20; font-size: 1.2rem; margin-left: 8px;'>▲</span>"
    elif is_better is False:
        badge = "<span style='color: #B71C1C; font-size: 1.2rem; margin-left: 8px;'>▼</span>"
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid {color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            <div style="color: #9E9E9E; font-size: 0.9rem; margin-bottom: 8px;">{label}</div>
            <div style="
                color: {color};
                font-size: 4rem;
                font-weight: 800;
                line-height: 1;
            ">{score_display}{badge}</div>
            <div style="color: {color}; font-size: 1.1rem; margin-top: 8px; font-weight: 600;">{score_label}</div>
            <div style="color: #757575; font-size: 0.85rem; margin-top: 12px;">
                {perf.get('matches', 0)} parties
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_comparison_row(
    label: str,
    val_a,
    val_b,
    fmt: str = "{}",
    higher_is_better: bool = True,
) -> None:
    """Affiche une ligne de métrique avec comparaison colorée.
    
    Args:
        label: Nom de la métrique.
        val_a: Valeur pour la session A.
        val_b: Valeur pour la session B.
        fmt: Format string pour l'affichage.
        higher_is_better: Si True, la valeur la plus haute est verte.
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Déterminer les couleurs
    color_a, color_b = "#E0E0E0", "#E0E0E0"
    if val_a is not None and val_b is not None:
        if higher_is_better:
            if val_a > val_b:
                color_a = "#4CAF50"
            elif val_b > val_a:
                color_b = "#4CAF50"
        else:
            if val_a < val_b:
                color_a = "#4CAF50"
            elif val_b < val_a:
                color_b = "#4CAF50"
    
    with col1:
        st.markdown(f"**{label}**")
    with col2:
        display_a = fmt.format(val_a) if val_a is not None else "—"
        st.markdown(
            f"<span style='color: {color_a}; font-weight: 600;'>{display_a}</span>",
            unsafe_allow_html=True,
        )
    with col3:
        display_b = fmt.format(val_b) if val_b is not None else "—"
        st.markdown(
            f"<span style='color: {color_b}; font-weight: 600;'>{display_b}</span>",
            unsafe_allow_html=True,
        )
