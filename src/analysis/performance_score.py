from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.analysis.performance_config import (
    RELATIVE_WEIGHTS,
    MIN_MATCHES_FOR_RELATIVE,
    SCORE_THRESHOLDS,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Score de performance RELATIF par match (0-100)
# =============================================================================
# 
# Ce score compare la performance du match à l'historique personnel du joueur.
# - 50 = match dans ta moyenne
# - 100 = meilleur match de ton historique
# - 0 = pire match de ton historique
#
# Configuration centralisée dans : src/analysis/performance_config.py
# =============================================================================


def _compute_per_minute(value: float | None, duration_seconds: float | None) -> float | None:
    """Calcule une valeur par minute."""
    if value is None or duration_seconds is None:
        return None
    if duration_seconds <= 0:
        return None
    return float(value) / (float(duration_seconds) / 60.0)


def _percentile_rank(value: float, series: pd.Series) -> float:
    """Calcule le percentile d'une valeur dans une série (0-100).
    
    Args:
        value: Valeur à évaluer.
        series: Série de référence (historique).
        
    Returns:
        Percentile 0-100 où 50 = médiane.
    """
    if series.empty or len(series) < 2:
        return 50.0  # Pas assez de données, on retourne la moyenne
    
    # Nombre de valeurs inférieures ou égales
    below_or_equal = (series <= value).sum()
    # Pourcentage
    percentile = (below_or_equal / len(series)) * 100.0
    return _clamp(percentile, 0.0, 100.0)


def _percentile_rank_inverse(value: float, series: pd.Series) -> float:
    """Percentile inversé (pour les morts: moins = mieux)."""
    if series.empty or len(series) < 2:
        return 50.0
    # Plus la valeur est basse, meilleur est le percentile
    above_or_equal = (series >= value).sum()
    percentile = (above_or_equal / len(series)) * 100.0
    return _clamp(percentile, 0.0, 100.0)


def _prepare_history_metrics(df_history: pd.DataFrame) -> pd.DataFrame:
    """Prépare les métriques normalisées par minute pour l'historique.
    
    Args:
        df_history: DataFrame de l'historique des matchs.
        
    Returns:
        DataFrame avec colonnes kpm, dpm, apm, kda, accuracy.
    """
    if df_history.empty:
        return pd.DataFrame(columns=["kpm", "dpm", "apm", "kda", "accuracy"])
    
    df = df_history.copy()
    
    # Durée du match en secondes
    duration_col = None
    for col in ["time_played_seconds", "duration_seconds", "match_duration_seconds"]:
        if col in df.columns:
            duration_col = col
            break
    
    if duration_col is None:
        # Fallback: estimer 10 minutes par défaut
        df["_duration"] = 600.0
    else:
        df["_duration"] = pd.to_numeric(df[duration_col], errors="coerce").fillna(600.0)
        df.loc[df["_duration"] <= 0, "_duration"] = 600.0
    
    # Calcul des métriques par minute
    df["kpm"] = pd.to_numeric(df.get("kills", 0), errors="coerce").fillna(0) / (df["_duration"] / 60.0)
    df["dpm"] = pd.to_numeric(df.get("deaths", 0), errors="coerce").fillna(0) / (df["_duration"] / 60.0)
    df["apm"] = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0) / (df["_duration"] / 60.0)
    
    # FDA (KDA)
    if "kda" in df.columns:
        df["kda"] = pd.to_numeric(df["kda"], errors="coerce")
    else:
        # Calculer KDA : (K + A) / max(1, D)
        k = pd.to_numeric(df.get("kills", 0), errors="coerce").fillna(0)
        d = pd.to_numeric(df.get("deaths", 0), errors="coerce").fillna(0)
        a = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0)
        df["kda"] = (k + a) / d.clip(lower=1)
    
    # Accuracy
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    else:
        df["accuracy"] = None
    
    return df[["kpm", "dpm", "apm", "kda", "accuracy"]].copy()


def compute_relative_performance_score(
    row: pd.Series,
    df_history: pd.DataFrame,
) -> float | None:
    """Calcule le score de performance RELATIF d'un match.
    
    Compare le match à l'historique personnel du joueur.
    
    Args:
        row: Ligne du match avec kills, deaths, assists, kda, accuracy, time_played_seconds.
        df_history: DataFrame de l'historique complet du joueur.
        
    Returns:
        Score 0-100 où 50 = performance moyenne, 100 = meilleure perf, 0 = pire perf.
        None si pas assez de données.
    """
    if df_history is None or df_history.empty:
        return None
    
    if len(df_history) < MIN_MATCHES_FOR_RELATIVE:
        # Pas assez de matchs, on ne peut pas calculer un score relatif fiable
        return None
    
    # Préparer l'historique
    history_metrics = _prepare_history_metrics(df_history)
    
    # Extraire les valeurs du match actuel
    try:
        # Durée du match
        duration = None
        for col in ["time_played_seconds", "duration_seconds", "match_duration_seconds"]:
            if col in row.index and row.get(col) is not None:
                try:
                    duration = float(row.get(col))
                    if duration > 0:
                        break
                except (ValueError, TypeError):
                    pass
        if duration is None or duration <= 0:
            duration = 600.0  # 10 min par défaut
        
        kills = float(row.get("kills") or 0)
        deaths = float(row.get("deaths") or 0)
        assists = float(row.get("assists") or 0)
        
        # Métriques par minute
        kpm = kills / (duration / 60.0)
        dpm = deaths / (duration / 60.0)
        apm = assists / (duration / 60.0)
        
        # KDA
        kda = row.get("kda")
        if kda is not None:
            try:
                kda = float(kda)
            except (ValueError, TypeError):
                kda = (kills + assists) / max(1, deaths)
        else:
            kda = (kills + assists) / max(1, deaths)
        
        # Accuracy
        accuracy = row.get("accuracy")
        if accuracy is not None:
            try:
                accuracy = float(accuracy)
            except (ValueError, TypeError):
                accuracy = None
        
    except Exception:
        return None
    
    # Calculer les percentiles pour chaque métrique
    percentiles = {}
    weights_used = {}
    
    # KPM - plus c'est haut, mieux c'est
    kpm_series = history_metrics["kpm"].dropna()
    if not kpm_series.empty:
        percentiles["kpm"] = _percentile_rank(kpm, kpm_series)
        weights_used["kpm"] = RELATIVE_WEIGHTS["kpm"]
    
    # DPM - moins c'est haut, mieux c'est (inversé)
    dpm_series = history_metrics["dpm"].dropna()
    if not dpm_series.empty:
        percentiles["dpm"] = _percentile_rank_inverse(dpm, dpm_series)
        weights_used["dpm"] = RELATIVE_WEIGHTS["dpm"]
    
    # APM - plus c'est haut, mieux c'est
    apm_series = history_metrics["apm"].dropna()
    if not apm_series.empty:
        percentiles["apm"] = _percentile_rank(apm, apm_series)
        weights_used["apm"] = RELATIVE_WEIGHTS["apm"]
    
    # KDA - plus c'est haut, mieux c'est
    kda_series = history_metrics["kda"].dropna()
    if not kda_series.empty:
        percentiles["kda"] = _percentile_rank(kda, kda_series)
        weights_used["kda"] = RELATIVE_WEIGHTS["kda"]
    
    # Accuracy - plus c'est haut, mieux c'est
    if accuracy is not None:
        acc_series = history_metrics["accuracy"].dropna()
        if not acc_series.empty:
            percentiles["accuracy"] = _percentile_rank(accuracy, acc_series)
            weights_used["accuracy"] = RELATIVE_WEIGHTS["accuracy"]
    
    if not percentiles:
        return None
    
    # Moyenne pondérée des percentiles
    total_weight = sum(weights_used.values())
    if total_weight <= 0:
        return None
    
    score = sum(percentiles[k] * weights_used[k] for k in percentiles) / total_weight
    
    return round(score, 1)


def compute_match_performance_from_row(
    row: pd.Series,
    df_history: pd.DataFrame | None = None,
) -> float | None:
    """Calcule le score de performance à partir d'une ligne de DataFrame.
    
    Si df_history est fourni et suffisant, calcule un score RELATIF.
    Sinon, retourne None (pas de score absolu fallback).
    
    Args:
        row: Ligne avec colonnes kills, deaths, assists, accuracy, kda, time_played_seconds.
        df_history: DataFrame de l'historique du joueur (optionnel mais recommandé).
        
    Returns:
        Score entre 0 et 100, ou None si historique insuffisant.
    """
    if df_history is not None and len(df_history) >= MIN_MATCHES_FOR_RELATIVE:
        return compute_relative_performance_score(row, df_history)
    return None


def compute_performance_series(
    df: pd.DataFrame,
    df_history: pd.DataFrame | None = None,
) -> pd.Series:
    """Calcule le score de performance pour chaque match d'un DataFrame.
    
    Args:
        df: DataFrame des matchs à évaluer.
        df_history: Historique complet pour le calcul relatif.
                    Si None, utilise df comme historique.
        
    Returns:
        Series avec les scores de performance.
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    # Si pas d'historique fourni, on utilise le DataFrame lui-même
    history = df_history if df_history is not None else df
    
    if len(history) < MIN_MATCHES_FOR_RELATIVE:
        # Pas assez de matchs
        return pd.Series([None] * len(df), index=df.index)
    
    # Calculer le score pour chaque ligne
    scores = df.apply(
        lambda row: compute_relative_performance_score(row, history),
        axis=1,
    )
    
    return scores


# =============================================================================
# Helpers internes
# =============================================================================

def _clamp_internal(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _mean_numeric(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _sum_int(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return int(values.sum())


def _saturation_score(x: float, scale: float) -> float:
    """Score 0–100 avec saturation exponentielle.

    - x=0 -> 0
    - x ~= scale*ln(2) -> 50
    - x -> +inf -> 100

    Args:
        x: valeur positive.
        scale: échelle de la courbe (doit être > 0).

    Returns:
        Score entre 0 et 100.
    """
    if scale <= 0:
        return 0.0
    if x <= 0:
        return 0.0
    return _clamp(100.0 * (1.0 - math.exp(-x / scale)))


@dataclass(frozen=True)
class ScoreComponent:
    """Une composante de score (0–100) avec une pondération."""

    key: str
    label: str
    weight: float
    compute: Callable[[pd.DataFrame], tuple[float | None, dict[str, Any]]]


def _compute_kd_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    kills = _sum_int(df, "kills")
    deaths = _sum_int(df, "deaths")
    if kills == 0 and deaths == 0:
        return None, {"kd_ratio": None}

    kd_ratio = (kills / deaths) if deaths > 0 else float(kills)
    kd_score = _clamp(kd_ratio * 50.0)
    return kd_score, {"kd_ratio": round(kd_ratio, 2)}


def _compute_win_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    if "outcome" not in df.columns:
        return None, {"win_rate": None}

    n = len(df)
    if n <= 0:
        return None, {"win_rate": None}

    wins = int((pd.to_numeric(df["outcome"], errors="coerce") == 2).sum())
    win_rate = wins / n
    return _clamp(win_rate * 100.0), {"win_rate": round(win_rate * 100.0, 1)}


def _compute_accuracy_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    acc = None
    if "accuracy" in df.columns:
        acc = _mean_numeric(df, "accuracy")
    elif "shots_accuracy" in df.columns:
        acc = _mean_numeric(df, "shots_accuracy")

    if acc is None:
        return None, {"accuracy": None}

    return _clamp(acc), {"accuracy": round(acc, 1)}


def _compute_kpm_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    kpm = _mean_numeric(df, "kills_per_min")
    if kpm is None:
        return None, {"kills_per_min": None}

    # Calibration empirique : ~0.55 kpm ~50 pts.
    score = _saturation_score(kpm, scale=0.8)
    return score, {"kills_per_min": round(kpm, 2)}


def _compute_life_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    life = _mean_numeric(df, "average_life_seconds")
    if life is None:
        return None, {"avg_life_seconds": None}

    # Calibration : ~35s ~50 pts.
    score = _saturation_score(life, scale=50.0)
    return score, {"avg_life_seconds": round(life, 1)}


_OBJECTIVE_COLUMN_WEIGHTS: dict[str, float] = {
    # CTF
    "flag_captures": 3.0,
    "flag_returns": 1.0,
    # Strongholds
    "zones_captured": 2.0,
    "zones_defended": 1.0,
    # Oddball
    "ball_time_seconds": 1.0 / 30.0,
    "time_with_ball_seconds": 1.0 / 30.0,
    # King of the Hill
    "hill_time_seconds": 1.0 / 30.0,
    "time_in_hill_seconds": 1.0 / 30.0,
    # Assault / autres (noms possibles)
    "core_captures": 3.0,
    "objective_carries": 1.0,
}


def _compute_objective_component(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    used: dict[str, float] = {}
    total_points = 0.0

    for col, w in _OBJECTIVE_COLUMN_WEIGHTS.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        mean_val = float(values.mean())
        if mean_val <= 0:
            continue
        used[col] = w
        total_points += mean_val * w

    if not used:
        return None, {"objective_score": None, "objective_points_per_match": None, "objective_columns": []}

    # Calibration : ~2.1 points/match ~50 pts.
    score = _saturation_score(total_points, scale=3.0)
    return (
        score,
        {
            "objective_score": round(score, 1),
            "objective_points_per_match": round(total_points, 2),
            "objective_columns": sorted(used.keys()),
        },
    )


def _compute_mmr_aggregates(df: pd.DataFrame) -> dict[str, float | None]:
    team = _mean_numeric(df, "team_mmr")
    enemy = _mean_numeric(df, "enemy_mmr")
    delta = (team - enemy) if (team is not None and enemy is not None) else None

    return {
        "team_mmr_avg": round(team, 1) if team is not None else None,
        "enemy_mmr_avg": round(enemy, 1) if enemy is not None else None,
        "delta_mmr_avg": round(delta, 1) if delta is not None else None,
    }


def _mmr_difficulty_multiplier(delta_mmr_avg: float | None) -> float:
    """Applique un ajustement léger selon la difficulté.

    - Si ton équipe est "plus forte" (delta positif), on réduit légèrement le score.
    - Si ton équipe est "plus faible" (delta négatif), on augmente légèrement le score.

    Ajustement volontairement borné pour éviter de dominer le score.
    """
    if delta_mmr_avg is None:
        return 1.0

    # ~ +/- 300 MMR => +/- 5%
    adj = _clamp((-delta_mmr_avg / 300.0) * 5.0, lo=-5.0, hi=5.0) / 100.0
    return 1.0 + adj


def compute_session_performance_score_v1(df_session: pd.DataFrame) -> dict[str, Any]:
    """Version historique du score (0-100).

    Cette fonction est gardée pour rétrocompatibilité.
    """
    if df_session is None or df_session.empty:
        return {
            "score": None,
            "kd_ratio": None,
            "kda": None,
            "win_rate": None,
            "accuracy": None,
            "avg_score": None,
            "avg_life_seconds": None,
            "matches": 0,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "team_mmr_avg": None,
            "enemy_mmr_avg": None,
            "delta_mmr_avg": None,
        }

    total_kills = _sum_int(df_session, "kills")
    total_deaths = _sum_int(df_session, "deaths")
    total_assists = _sum_int(df_session, "assists")
    n_matches = len(df_session)

    kd_ratio = total_kills / total_deaths if total_deaths > 0 else float(total_kills)
    kd_score = _clamp(kd_ratio * 50.0)

    kda = (
        (total_kills + total_assists) / total_deaths
        if total_deaths > 0
        else float(total_kills + total_assists)
    )

    wins = (
        int((pd.to_numeric(df_session["outcome"], errors="coerce") == 2).sum())
        if "outcome" in df_session.columns
        else 0
    )
    win_rate = wins / n_matches if n_matches > 0 else 0.0
    win_score = win_rate * 100.0

    accuracy = None
    if "accuracy" in df_session.columns:
        accuracy = _mean_numeric(df_session, "accuracy")
    elif "shots_accuracy" in df_session.columns:
        accuracy = _mean_numeric(df_session, "shots_accuracy")
    acc_score = accuracy if accuracy is not None else 50.0

    avg_life_seconds = _mean_numeric(df_session, "average_life_seconds")

    avg_score = _mean_numeric(df_session, "match_score")
    score_pts = _clamp((avg_score or 10.0) * 5.0) if avg_score is not None else 50.0

    mmr = _compute_mmr_aggregates(df_session)

    final_score = (kd_score * 0.30) + (win_score * 0.25) + (acc_score * 0.25) + (score_pts * 0.20)

    return {
        "score": round(final_score, 1),
        "kd_ratio": round(kd_ratio, 2),
        "kda": round(kda, 2),
        "win_rate": round(win_rate * 100.0, 1),
        "accuracy": round(accuracy, 1) if accuracy is not None else None,
        "avg_score": round(avg_score, 1) if avg_score is not None else None,
        "avg_life_seconds": round(avg_life_seconds, 1) if avg_life_seconds is not None else None,
        "matches": n_matches,
        "kills": total_kills,
        "deaths": total_deaths,
        "assists": total_assists,
        **mmr,
    }


def compute_session_performance_score_v2(
    df_session: pd.DataFrame,
    *,
    include_mmr_adjustment: bool = True,
) -> dict[str, Any]:
    """Calcule un score de performance (0–100) plus robuste et modulaire.

    Principes :
    - On n’utilise que les composantes disponibles.
    - On renormalise les poids si une composante manque.
    - On peut ajouter une composante "objectif" quand des colonnes existent.

    Returns:
        Dict compatible avec la v1, + champs v2:
        - components: scores par composante (0-100)
        - weights_used: pondérations réellement utilisées
        - confidence: (0-1) indicateur simple basé sur le nombre de matchs
    """
    if df_session is None or df_session.empty:
        base = compute_session_performance_score_v1(df_session)
        base.update(
            {
                "components": {},
                "weights_used": {},
                "confidence": 0.0,
                "confidence_label": "faible",
                "objective_score": None,
                "objective_points_per_match": None,
                "objective_columns": [],
                "version": "v2",
            }
        )
        return base

    total_kills = _sum_int(df_session, "kills")
    total_deaths = _sum_int(df_session, "deaths")
    total_assists = _sum_int(df_session, "assists")
    n_matches = len(df_session)

    kd_ratio = (total_kills / total_deaths) if total_deaths > 0 else float(total_kills)
    kda = (
        (total_kills + total_assists) / total_deaths
        if total_deaths > 0
        else float(total_kills + total_assists)
    )

    avg_life_seconds = _mean_numeric(df_session, "average_life_seconds")
    accuracy = _mean_numeric(df_session, "accuracy")
    if accuracy is None:
        accuracy = _mean_numeric(df_session, "shots_accuracy")

    mmr = _compute_mmr_aggregates(df_session)

    components: list[ScoreComponent] = [
        ScoreComponent(key="kd", label="K/D", weight=0.25, compute=_compute_kd_component),
        ScoreComponent(key="win", label="Victoires", weight=0.20, compute=_compute_win_component),
        ScoreComponent(key="acc", label="Précision", weight=0.15, compute=_compute_accuracy_component),
        ScoreComponent(key="kpm", label="Kills/min", weight=0.15, compute=_compute_kpm_component),
        ScoreComponent(key="life", label="Survie", weight=0.10, compute=_compute_life_component),
        ScoreComponent(key="obj", label="Objectif", weight=0.15, compute=_compute_objective_component),
    ]

    computed_scores: dict[str, float] = {}
    component_meta: dict[str, Any] = {}
    weights_used: dict[str, float] = {}

    for comp in components:
        score, meta = comp.compute(df_session)
        component_meta[comp.key] = meta
        if score is None:
            continue
        computed_scores[comp.key] = float(score)
        weights_used[comp.key] = float(comp.weight)

    total_weight = sum(weights_used.values())
    if total_weight <= 0:
        final_score = None
    else:
        final_score = 0.0
        for key, w in weights_used.items():
            final_score += computed_scores[key] * (w / total_weight)

        if include_mmr_adjustment:
            final_score *= _mmr_difficulty_multiplier(mmr.get("delta_mmr_avg"))

        final_score = _clamp(final_score)

    # Confiance : simple, basé sur la taille d’échantillon.
    confidence = _clamp((n_matches / 10.0) * 100.0, lo=0.0, hi=100.0) / 100.0
    confidence_label = "faible" if n_matches < 4 else ("moyenne" if n_matches < 10 else "élevée")

    obj_meta = component_meta.get("obj", {})

    return {
        "score": round(final_score, 1) if final_score is not None else None,
        "kd_ratio": round(kd_ratio, 2),
        "kda": round(kda, 2),
        "win_rate": component_meta.get("win", {}).get("win_rate"),
        "accuracy": round(accuracy, 1) if accuracy is not None else None,
        "avg_score": None,
        "avg_life_seconds": round(avg_life_seconds, 1) if avg_life_seconds is not None else None,
        "matches": n_matches,
        "kills": total_kills,
        "deaths": total_deaths,
        "assists": total_assists,
        **mmr,
        "objective_score": obj_meta.get("objective_score"),
        "objective_points_per_match": obj_meta.get("objective_points_per_match"),
        "objective_columns": obj_meta.get("objective_columns", []),
        "components": {k: round(v, 1) for k, v in computed_scores.items()},
        "weights_used": weights_used,
        "confidence": round(confidence, 2),
        "confidence_label": confidence_label,
        "version": "v2",
    }
