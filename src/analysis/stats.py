"""Calcul des statistiques agrégées."""

from typing import Optional

import pandas as pd

from src.models import AggregatedStats, OutcomeRates
from src.ui.formatting import format_mmss


def compute_aggregated_stats(df: pd.DataFrame) -> AggregatedStats:
    """Agrège les statistiques d'un DataFrame de matchs.
    
    Args:
        df: DataFrame avec colonnes kills, deaths, assists, time_played_seconds.
        
    Returns:
        AggregatedStats contenant les totaux.
    """
    if df.empty:
        return AggregatedStats()
    
    total_time = (
        pd.to_numeric(df["time_played_seconds"], errors="coerce").dropna().sum()
        if "time_played_seconds" in df.columns
        else 0.0
    )
    
    return AggregatedStats(
        total_kills=int(df["kills"].sum()),
        total_deaths=int(df["deaths"].sum()),
        total_assists=int(df["assists"].sum()),
        total_matches=len(df),
        total_time_seconds=float(total_time),
    )


def compute_outcome_rates(df: pd.DataFrame) -> OutcomeRates:
    """Calcule les taux de victoire/défaite.
    
    Args:
        df: DataFrame avec colonne outcome.
        
    Returns:
        OutcomeRates avec les comptages.
        
    Note:
        Codes outcome: 2=Wins, 3=Losses, 1=Ties, 4=NoFinishes
    """
    d = df.dropna(subset=["outcome"]).copy()
    total = len(d)
    counts = d["outcome"].value_counts().to_dict() if total else {}
    
    return OutcomeRates(
        wins=int(counts.get(2, 0)),
        losses=int(counts.get(3, 0)),
        ties=int(counts.get(1, 0)),
        no_finish=int(counts.get(4, 0)),
        total=total,
    )


def compute_global_ratio(df: pd.DataFrame) -> Optional[float]:
    """Calcule le ratio global (K + A/2) / D sur un DataFrame.
    
    Args:
        df: DataFrame avec colonnes kills, deaths, assists.
        
    Returns:
        Le ratio global, ou None si pas de deaths.
    """
    if df.empty:
        return None
    deaths = float(df["deaths"].sum())
    if deaths <= 0:
        return None
    return (float(df["kills"].sum()) + (float(df["assists"].sum()) / 2.0)) / deaths


def format_selected_matches_summary(n: int, rates: OutcomeRates) -> str:
    """Formate un résumé des matchs sélectionnés pour l'UI.
    
    Args:
        n: Nombre de matchs.
        rates: OutcomeRates calculé.
        
    Returns:
        Chaîne formatée pour affichage.
    """
    if n <= 0:
        return "Aucun match sélectionné"

    def plural(n_: int, one: str, many: str) -> str:
        return one if int(n_) == 1 else many

    wins = rates.wins
    losses = rates.losses
    ties = rates.ties
    nofinish = rates.no_finish
    
    return (
        f"{plural(n, 'Partie', 'Parties')} sélectionnée{'' if n == 1 else 's'}: {n} | "
        f"{plural(wins, 'Victoire', 'Victoires')}: {wins} | "
        f"{plural(losses, 'Défaite', 'Défaites')}: {losses} | "
        f"{plural(ties, 'Égalité', 'Égalités')}: {ties} | "
        f"{plural(nofinish, 'Non terminé', 'Non terminés')}: {nofinish}"
    )


# NOTE: format_mmss est importé depuis src.ui.formatting pour éviter la duplication.
# Re-export pour rétrocompatibilité des imports existants.
__all__ = ["compute_aggregated_stats", "compute_outcome_rates", "compute_global_ratio", "format_selected_matches_summary", "format_mmss"]
