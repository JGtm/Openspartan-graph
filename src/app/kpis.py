"""Calcul et affichage des KPIs pour l'application Streamlit.

Ce module gère :
- Le calcul des statistiques agrégées
- L'affichage des cartes KPI
- Le résumé du bandeau supérieur
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import streamlit as st

from src.analysis import compute_aggregated_stats, compute_outcome_rates, compute_global_ratio
from src.analysis.stats import format_mmss
from src.ui.formatting import format_duration_hms, format_duration_dhm
from src.ui.components import render_kpi_cards, render_top_summary


# =============================================================================
# Calculs temporels
# =============================================================================


def avg_match_duration_seconds(df_: pd.DataFrame) -> float | None:
    """Calcule la durée moyenne d'un match.
    
    Args:
        df_: DataFrame des matchs.
        
    Returns:
        Durée moyenne en secondes ou None.
    """
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def compute_total_play_seconds(df_: pd.DataFrame) -> float | None:
    """Calcule le temps de jeu total (somme des durées de matchs).
    
    Args:
        df_: DataFrame des matchs.
        
    Returns:
        Durée totale en secondes ou None.
    """
    if df_ is None or df_.empty or "time_played_seconds" not in df_.columns:
        return None
    s = pd.to_numeric(df_["time_played_seconds"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.sum())


# =============================================================================
# Statistiques KPI
# =============================================================================


class KPIStats(NamedTuple):
    """Statistiques KPI calculées."""
    # Outcomes
    win_rate: float
    loss_rate: float
    total_matches: int
    wins: int
    losses: int
    ties: int
    
    # Performance
    avg_accuracy: float | None
    global_ratio: float | None
    avg_life_seconds: float | None
    
    # Per-game averages
    kills_per_game: float | None
    deaths_per_game: float | None
    assists_per_game: float | None
    
    # Per-minute rates
    kills_per_minute: float | None
    deaths_per_minute: float | None
    assists_per_minute: float | None
    
    # Time
    avg_match_seconds: float | None
    total_play_seconds: float | None


def compute_kpi_stats(df: pd.DataFrame) -> KPIStats:
    """Calcule toutes les statistiques KPI.
    
    Args:
        df: DataFrame des matchs filtrés.
        
    Returns:
        KPIStats avec toutes les statistiques calculées.
    """
    # Outcomes
    rates = compute_outcome_rates(df)
    total_outcomes = max(1, rates.total)
    win_rate = rates.wins / total_outcomes
    loss_rate = rates.losses / total_outcomes
    
    # Performance
    avg_acc = df["accuracy"].dropna().mean() if not df.empty else None
    global_ratio = compute_global_ratio(df)
    avg_life = df["average_life_seconds"].dropna().mean() if not df.empty else None
    
    # Per-game averages
    kpg = df["kills"].mean() if not df.empty else None
    dpg = df["deaths"].mean() if not df.empty else None
    apg = df["assists"].mean() if not df.empty else None
    
    # Per-minute rates
    stats = compute_aggregated_stats(df)
    
    # Time
    avg_match_s = avg_match_duration_seconds(df)
    total_play_s = compute_total_play_seconds(df)
    
    return KPIStats(
        win_rate=win_rate,
        loss_rate=loss_rate,
        total_matches=rates.total,
        wins=rates.wins,
        losses=rates.losses,
        ties=rates.ties,
        avg_accuracy=avg_acc,
        global_ratio=global_ratio,
        avg_life_seconds=avg_life,
        kills_per_game=kpg,
        deaths_per_game=dpg,
        assists_per_game=apg,
        kills_per_minute=stats.kills_per_minute,
        deaths_per_minute=stats.deaths_per_minute,
        assists_per_minute=stats.assists_per_minute,
        avg_match_seconds=avg_match_s,
        total_play_seconds=total_play_s,
    )


# =============================================================================
# Rendu des KPIs
# =============================================================================


def render_matches_summary(df: pd.DataFrame, kpis: KPIStats) -> None:
    """Rend le résumé des parties (bandeau supérieur).
    
    Args:
        df: DataFrame des matchs filtrés.
        kpis: Statistiques KPI calculées.
    """
    rates = compute_outcome_rates(df)
    
    avg_match_txt = format_duration_hms(kpis.avg_match_seconds)
    total_play_txt = format_duration_dhm(kpis.total_play_seconds)
    
    st.subheader("Parties")
    render_top_summary(len(df), rates)
    render_kpi_cards([
        ("Durée moyenne / match", avg_match_txt),
        ("Durée totale", total_play_txt),
    ])


def render_career_kpis(kpis: KPIStats) -> None:
    """Rend les KPIs de carrière.
    
    Args:
        kpis: Statistiques KPI calculées.
    """
    avg_match_txt = format_duration_hms(kpis.avg_match_seconds)
    
    st.subheader("Carrière")
    render_kpi_cards(
        [
            ("Durée moyenne / match", avg_match_txt),
            ("Frags par partie", f"{kpis.kills_per_game:.2f}" if (kpis.kills_per_game is not None and pd.notna(kpis.kills_per_game)) else "-"),
            ("Morts par partie", f"{kpis.deaths_per_game:.2f}" if (kpis.deaths_per_game is not None and pd.notna(kpis.deaths_per_game)) else "-"),
            ("Assistances par partie", f"{kpis.assists_per_game:.2f}" if (kpis.assists_per_game is not None and pd.notna(kpis.assists_per_game)) else "-"),
        ],
        dense=False,
    )
    render_kpi_cards(
        [
            ("Frags / min", f"{kpis.kills_per_minute:.2f}" if kpis.kills_per_minute else "-"),
            ("Morts / min", f"{kpis.deaths_per_minute:.2f}" if kpis.deaths_per_minute else "-"),
            ("Assistances / min", f"{kpis.assists_per_minute:.2f}" if kpis.assists_per_minute else "-"),
            ("Précision moyenne", f"{kpis.avg_accuracy:.2f}%" if kpis.avg_accuracy is not None else "-"),
            ("Durée de vie moyenne", format_mmss(kpis.avg_life_seconds)),
            ("Taux de victoire", f"{kpis.win_rate*100:.1f}%" if kpis.total_matches else "-"),
            ("Taux de défaite", f"{kpis.loss_rate*100:.1f}%" if kpis.total_matches else "-"),
            ("Ratio", f"{kpis.global_ratio:.2f}" if kpis.global_ratio is not None else "-"),
        ],
        dense=False,
    )


def render_all_kpis(df: pd.DataFrame) -> KPIStats:
    """Rend tous les KPIs (parties + carrière) et retourne les stats.
    
    Args:
        df: DataFrame des matchs filtrés.
        
    Returns:
        KPIStats calculées.
    """
    kpis = compute_kpi_stats(df)
    render_matches_summary(df, kpis)
    render_career_kpis(kpis)
    return kpis
