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


def extract_mode_category(pair_name: str | None) -> str:
    """Extrait la catégorie de mode depuis le pair_name.
    
    Exemples:
        "Arena:CTF on Aquarius" → "Arena"
        "BTB:CTF" → "BTB"
        "Firefight: Kilo Five" → "Firefight"
        "Slayer" → "Unknown"
    
    Args:
        pair_name: Nom du couple mode/carte (ex: "Arena:Slayer on Aquarius").
        
    Returns:
        Catégorie du mode (Arena, BTB, Firefight, Community, etc.) ou "Unknown".
    """
    if not pair_name:
        return "Unknown"
    
    s = str(pair_name).strip()
    
    # Format "Category:Mode" ou "Category:Mode on Map"
    if ":" in s:
        category = s.split(":", 1)[0].strip()
        # Normaliser les noms courants
        cat_lower = category.lower()
        if cat_lower in ("arena", "arène"):
            return "Arena"
        if cat_lower in ("btb", "big team battle", "btb:"):
            return "BTB"
        if "firefight" in cat_lower or "pve" in cat_lower:
            return "Firefight"
        if cat_lower in ("ranked", "classé", "classée"):
            return "Ranked"
        if cat_lower in ("community", "communauté"):
            return "Community"
        return category.title()
    
    # Pas de préfixe, essayer de deviner
    s_lower = s.lower()
    if "firefight" in s_lower or "pve" in s_lower:
        return "Firefight"
    if "big team" in s_lower or "btb" in s_lower:
        return "BTB"
    
    return "Unknown"


def compute_mode_category_averages(
    df: pd.DataFrame,
    category: str,
) -> dict[str, float | None]:
    """Calcule les moyennes historiques pour une catégorie de mode.
    
    Args:
        df: DataFrame des matchs avec colonnes pair_name, kills, deaths, assists.
        category: Catégorie de mode (Arena, BTB, Firefight, etc.).
        
    Returns:
        Dict avec les moyennes: avg_kills, avg_deaths, avg_assists, avg_ratio, match_count.
    """
    empty_result = {
        "avg_kills": None,
        "avg_deaths": None,
        "avg_assists": None,
        "avg_ratio": None,
        "match_count": 0,
    }
    
    if df.empty:
        return empty_result
    
    # Filtrer par catégorie - vectorisé pour performance
    mask = df["pair_name"].apply(extract_mode_category) == category
    filtered = df.loc[mask]
    
    if filtered.empty:
        return empty_result
    
    match_count = len(filtered)
    avg_kills = filtered["kills"].mean()
    avg_deaths = filtered["deaths"].mean()
    avg_assists = filtered["assists"].mean()
    
    # Ratio moyen (somme des frags / somme des morts)
    total_deaths = filtered["deaths"].sum()
    if total_deaths > 0:
        avg_ratio = (filtered["kills"].sum() + filtered["assists"].sum() / 2.0) / total_deaths
    else:
        avg_ratio = None
    
    return {
        "avg_kills": float(avg_kills) if pd.notna(avg_kills) else None,
        "avg_deaths": float(avg_deaths) if pd.notna(avg_deaths) else None,
        "avg_assists": float(avg_assists) if pd.notna(avg_assists) else None,
        "avg_ratio": float(avg_ratio) if avg_ratio is not None else None,
        "match_count": len(filtered),
    }


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
__all__ = [
    "compute_aggregated_stats",
    "compute_outcome_rates",
    "compute_global_ratio",
    "format_selected_matches_summary",
    "format_mmss",
    "extract_mode_category",
    "compute_mode_category_averages",
]
