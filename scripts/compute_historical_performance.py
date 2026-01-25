#!/usr/bin/env python
"""Script de migration pour calculer les scores de performance historiques.

Ce script calcule le score de performance pour tous les matchs existants,
en utilisant une approche "rolling" : chaque match est comparé aux matchs
**précédents** uniquement, pour refléter la progression réelle du joueur.

Usage:
    python scripts/compute_historical_performance.py <db_path> [--xuid XUID]
    
Options:
    --xuid XUID     Calculer uniquement pour ce joueur (sinon tous les joueurs)
    --batch-size N  Taille des batches pour le commit (défaut: 100)
    --dry-run       Afficher les scores sans modifier la DB
    --force         Recalculer même si un score existe déjà

Exemple:
    python scripts/compute_historical_performance.py data/halo_guillaume.db
    python scripts/compute_historical_performance.py data/halo_all.db --xuid xuid(1234567890)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.analysis.performance_config import (
    MIN_MATCHES_FOR_RELATIVE,
    PERFORMANCE_SCORE_VERSION,
    RELATIVE_WEIGHTS,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _percentile_rank(value: float, series: pd.Series) -> float:
    """Calcule le percentile d'une valeur dans une série (0-100)."""
    if series.empty or len(series) < 2:
        return 50.0
    below_or_equal = (series <= value).sum()
    percentile = (below_or_equal / len(series)) * 100.0
    return _clamp(percentile, 0.0, 100.0)


def _percentile_rank_inverse(value: float, series: pd.Series) -> float:
    """Percentile inversé (pour les morts: moins = mieux)."""
    if series.empty or len(series) < 2:
        return 50.0
    above_or_equal = (series >= value).sum()
    percentile = (above_or_equal / len(series)) * 100.0
    return _clamp(percentile, 0.0, 100.0)


def compute_score_for_match(
    match_row: pd.Series,
    history_df: pd.DataFrame,
) -> float | None:
    """Calcule le score de performance pour un match donné.
    
    Args:
        match_row: Données du match à évaluer.
        history_df: Historique des matchs PRÉCÉDENTS ce match.
        
    Returns:
        Score 0-100 ou None si pas assez d'historique.
    """
    if len(history_df) < MIN_MATCHES_FOR_RELATIVE:
        return None
    
    try:
        # Durée du match
        duration = match_row.get("time_played_seconds")
        if duration is None or duration <= 0:
            duration = 600.0
        
        kills = float(match_row.get("kills") or 0)
        deaths = float(match_row.get("deaths") or 0)
        assists = float(match_row.get("assists") or 0)
        
        # Métriques par minute
        kpm = kills / (duration / 60.0)
        dpm = deaths / (duration / 60.0)
        apm = assists / (duration / 60.0)
        
        # KDA
        kda = match_row.get("kda")
        if kda is None or pd.isna(kda):
            kda = (kills + assists) / max(1, deaths)
        else:
            kda = float(kda)
        
        # Accuracy
        accuracy = match_row.get("accuracy")
        if accuracy is not None and not pd.isna(accuracy):
            accuracy = float(accuracy)
        else:
            accuracy = None
        
        # Préparer l'historique
        h = history_df.copy()
        h_dur = pd.to_numeric(h.get("time_played_seconds", 600), errors="coerce").fillna(600)
        h_dur = h_dur.clip(lower=60)
        
        h_kpm = pd.to_numeric(h.get("kills", 0), errors="coerce").fillna(0) / (h_dur / 60.0)
        h_dpm = pd.to_numeric(h.get("deaths", 0), errors="coerce").fillna(0) / (h_dur / 60.0)
        h_apm = pd.to_numeric(h.get("assists", 0), errors="coerce").fillna(0) / (h_dur / 60.0)
        
        h_kda = pd.to_numeric(h.get("kda"), errors="coerce")
        if h_kda.isna().all():
            h_k = pd.to_numeric(h.get("kills", 0), errors="coerce").fillna(0)
            h_d = pd.to_numeric(h.get("deaths", 0), errors="coerce").fillna(0).clip(lower=1)
            h_a = pd.to_numeric(h.get("assists", 0), errors="coerce").fillna(0)
            h_kda = (h_k + h_a) / h_d
        
        h_acc = pd.to_numeric(h.get("accuracy"), errors="coerce")
        
        # Calculer les percentiles
        percentiles = {}
        weights_used = {}
        
        if not h_kpm.dropna().empty:
            percentiles["kpm"] = _percentile_rank(kpm, h_kpm.dropna())
            weights_used["kpm"] = RELATIVE_WEIGHTS["kpm"]
        
        if not h_dpm.dropna().empty:
            percentiles["dpm"] = _percentile_rank_inverse(dpm, h_dpm.dropna())
            weights_used["dpm"] = RELATIVE_WEIGHTS["dpm"]
        
        if not h_apm.dropna().empty:
            percentiles["apm"] = _percentile_rank(apm, h_apm.dropna())
            weights_used["apm"] = RELATIVE_WEIGHTS["apm"]
        
        if not h_kda.dropna().empty:
            percentiles["kda"] = _percentile_rank(kda, h_kda.dropna())
            weights_used["kda"] = RELATIVE_WEIGHTS["kda"]
        
        if accuracy is not None and not h_acc.dropna().empty:
            percentiles["accuracy"] = _percentile_rank(accuracy, h_acc.dropna())
            weights_used["accuracy"] = RELATIVE_WEIGHTS["accuracy"]
        
        if not percentiles:
            return None
        
        total_weight = sum(weights_used.values())
        if total_weight <= 0:
            return None
        
        score = sum(percentiles[k] * weights_used[k] for k in percentiles) / total_weight
        return round(score, 1)
        
    except Exception as e:
        print(f"Erreur calcul score: {e}")
        return None


def load_matches_for_player(con: sqlite3.Connection, xuid: str) -> pd.DataFrame:
    """Charge tous les matchs d'un joueur triés par date."""
    query = """
    SELECT 
        match_id,
        xuid,
        start_time,
        kills,
        deaths,
        assists,
        accuracy,
        kda,
        time_played_seconds,
        outcome,
        performance_score
    FROM MatchCache
    WHERE xuid = ?
    ORDER BY start_time ASC
    """
    return pd.read_sql_query(query, con, params=(xuid,))


def load_matches_from_match_stats(con: sqlite3.Connection, xuid: str) -> pd.DataFrame:
    """Charge les matchs depuis MatchStats si MatchCache n'existe pas."""
    query = """
    SELECT 
        m.MatchId as match_id,
        m.XUID as xuid,
        datetime(m.MatchInfo_StartTime) as start_time,
        m.PlayerTeamStats_CoreStats_Kills as kills,
        m.PlayerTeamStats_CoreStats_Deaths as deaths,
        m.PlayerTeamStats_CoreStats_Assists as assists,
        m.PlayerTeamStats_CoreStats_Accuracy as accuracy,
        (m.PlayerTeamStats_CoreStats_Kills + m.PlayerTeamStats_CoreStats_Assists) * 1.0 / 
            MAX(1, m.PlayerTeamStats_CoreStats_Deaths) as kda,
        m.MatchInfo_Duration as time_played_seconds,
        m.Outcome as outcome
    FROM MatchStats m
    WHERE m.XUID = ?
    ORDER BY m.MatchInfo_StartTime ASC
    """
    return pd.read_sql_query(query, con, params=(xuid,))


def get_all_xuids(con: sqlite3.Connection, table: str = "MatchCache") -> list[str]:
    """Récupère tous les XUIDs distincts."""
    try:
        cur = con.execute(f"SELECT DISTINCT xuid FROM {table}")
        return [row[0] for row in cur.fetchall() if row[0]]
    except Exception:
        return []


def has_table(con: sqlite3.Connection, table_name: str) -> bool:
    """Vérifie si une table existe."""
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cur.fetchone() is not None


def update_performance_score(
    con: sqlite3.Connection,
    match_id: str,
    xuid: str,
    score: float,
    table: str = "MatchCache",
) -> None:
    """Met à jour le score de performance d'un match."""
    con.execute(
        f"UPDATE {table} SET performance_score = ?, updated_at = ? WHERE match_id = ? AND xuid = ?",
        (score, datetime.now().isoformat(), match_id, xuid)
    )


def process_player(
    con: sqlite3.Connection,
    xuid: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = 100,
    use_match_stats: bool = False,
) -> dict:
    """Calcule les scores pour un joueur.
    
    Returns:
        Statistiques du traitement.
    """
    stats = {"total": 0, "computed": 0, "skipped": 0, "errors": 0}
    
    # Charger les matchs
    if use_match_stats:
        df = load_matches_from_match_stats(con, xuid)
    else:
        df = load_matches_for_player(con, xuid)
    
    if df.empty:
        return stats
    
    stats["total"] = len(df)
    
    # Convertir start_time en datetime pour le tri
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.sort_values("start_time").reset_index(drop=True)
    
    batch_updates = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {xuid[:20]}...", leave=False):
        match_id = row["match_id"]
        
        # Skip si score existe déjà et pas force
        if not force and "performance_score" in row and pd.notna(row.get("performance_score")):
            stats["skipped"] += 1
            continue
        
        # Historique = matchs AVANT ce match
        history = df.iloc[:idx]
        
        # Calculer le score
        score = compute_score_for_match(row, history)
        
        if score is not None:
            stats["computed"] += 1
            if not dry_run:
                batch_updates.append((score, datetime.now().isoformat(), match_id, xuid))
                
                # Commit par batch
                if len(batch_updates) >= batch_size:
                    con.executemany(
                        "UPDATE MatchCache SET performance_score = ?, updated_at = ? WHERE match_id = ? AND xuid = ?",
                        batch_updates
                    )
                    con.commit()
                    batch_updates = []
        else:
            stats["skipped"] += 1
    
    # Commit restant
    if batch_updates and not dry_run:
        con.executemany(
            "UPDATE MatchCache SET performance_score = ?, updated_at = ? WHERE match_id = ? AND xuid = ?",
            batch_updates
        )
        con.commit()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les scores de performance historiques.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("db_path", help="Chemin vers la base de données SQLite")
    parser.add_argument("--xuid", help="XUID spécifique (sinon tous les joueurs)")
    parser.add_argument("--batch-size", type=int, default=100, help="Taille des batches")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    parser.add_argument("--force", action="store_true", help="Recalculer tous les scores")
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Base de données introuvable: {db_path}")
        sys.exit(1)
    
    print(f"📊 Calcul des scores de performance historiques")
    print(f"   DB: {db_path}")
    print(f"   Version: {PERFORMANCE_SCORE_VERSION}")
    print(f"   Min matchs: {MIN_MATCHES_FOR_RELATIVE}")
    print(f"   Dry-run: {args.dry_run}")
    print(f"   Force: {args.force}")
    print()
    
    con = sqlite3.connect(str(db_path))
    
    # Vérifier quelle table utiliser
    use_cache = has_table(con, "MatchCache")
    if not use_cache:
        print("⚠️  Table MatchCache absente, utilisation de MatchStats")
        print("   Conseil: exécuter d'abord migrate_to_cache.py")
    
    # Récupérer les joueurs
    if args.xuid:
        xuids = [args.xuid]
    else:
        table = "MatchCache" if use_cache else "MatchStats"
        xuids = get_all_xuids(con, table if table == "MatchCache" else "MatchStats")
        if not xuids and not use_cache:
            cur = con.execute("SELECT DISTINCT XUID FROM MatchStats")
            xuids = [row[0] for row in cur.fetchall() if row[0]]
    
    if not xuids:
        print("❌ Aucun joueur trouvé")
        sys.exit(1)
    
    print(f"👥 {len(xuids)} joueur(s) à traiter")
    print()
    
    total_stats = {"total": 0, "computed": 0, "skipped": 0, "errors": 0}
    
    for xuid in tqdm(xuids, desc="Joueurs"):
        stats = process_player(
            con,
            xuid,
            dry_run=args.dry_run,
            force=args.force,
            batch_size=args.batch_size,
            use_match_stats=not use_cache,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
    
    con.close()
    
    print()
    print("=" * 50)
    print(f"✅ Terminé")
    print(f"   Total matchs: {total_stats['total']}")
    print(f"   Scores calculés: {total_stats['computed']}")
    print(f"   Ignorés (insuffisant/existant): {total_stats['skipped']}")
    if total_stats['errors']:
        print(f"   Erreurs: {total_stats['errors']}")


if __name__ == "__main__":
    main()
