#!/usr/bin/env python3
"""Script de synchronisation unifié.

Point d'entrée unique pour toutes les opérations de synchronisation :
- Import des matchs via SPNKr
- Reconstruction du cache (MatchCache)
- Téléchargement des assets (médailles, maps)
- Application des index

Usage:
    python scripts/sync.py --help
    python scripts/sync.py --delta                    # Sync incrémentale
    python scripts/sync.py --full                     # Sync complète
    python scripts/sync.py --rebuild-cache            # Reconstruit MatchCache
    python scripts/sync.py --apply-indexes            # Applique les index optimisés
    python scripts/sync.py --delta --with-assets      # Sync + assets
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ajouter le répertoire parent au path pour les imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_default_db_path
from src.db.connection import get_connection
from src.db.schema import get_all_cache_table_ddl, get_source_table_indexes, CACHE_SCHEMA_VERSION

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _get_iso_now() -> str:
    """Retourne le timestamp ISO 8601 actuel (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def _update_sync_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    """Met à jour une entrée dans SyncMeta."""
    cur = con.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO SyncMeta (Key, Value, UpdatedAt)
           VALUES (?, ?, ?)""",
        (key, value, _get_iso_now()),
    )
    con.commit()


def _get_sync_meta(con: sqlite3.Connection, key: str) -> str | None:
    """Récupère une valeur depuis SyncMeta."""
    try:
        cur = con.cursor()
        cur.execute("SELECT Value FROM SyncMeta WHERE Key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _has_table(con: sqlite3.Connection, table_name: str) -> bool:
    """Vérifie si une table existe."""
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    )
    return cur.fetchone() is not None


def _count_rows(con: sqlite3.Connection, table_name: str) -> int:
    """Compte le nombre de lignes dans une table."""
    try:
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        return cur.fetchone()[0]
    except Exception:
        return 0


# =============================================================================
# Opérations de synchronisation
# =============================================================================


def apply_indexes(db_path: str) -> tuple[bool, str]:
    """Applique les index optimisés sur la base de données.
    
    Args:
        db_path: Chemin vers la base de données.
        
    Returns:
        Tuple (success, message).
    """
    logger.info("Application des index optimisés...")
    
    try:
        with get_connection(db_path) as con:
            cur = con.cursor()
            
            # Index des tables sources
            source_indexes = get_source_table_indexes()
            applied = 0
            
            for ddl in source_indexes:
                try:
                    cur.execute(ddl)
                    applied += 1
                except sqlite3.OperationalError as e:
                    # Index déjà existant ou table manquante
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Index ignoré: {e}")
            
            con.commit()
            
            # Analyser les tables pour optimiser le query planner
            logger.info("Analyse des tables (ANALYZE)...")
            cur.execute("ANALYZE")
            con.commit()
            
            msg = f"Index appliqués: {applied}/{len(source_indexes)}"
            logger.info(msg)
            return True, msg
            
    except Exception as e:
        msg = f"Erreur lors de l'application des index: {e}"
        logger.error(msg)
        return False, msg


def ensure_cache_tables(db_path: str) -> tuple[bool, str]:
    """Crée les tables de cache si elles n'existent pas.
    
    Args:
        db_path: Chemin vers la base de données.
        
    Returns:
        Tuple (success, message).
    """
    logger.info("Vérification des tables de cache...")
    
    try:
        with get_connection(db_path) as con:
            cur = con.cursor()
            
            # Créer les tables de cache
            ddl_list = get_all_cache_table_ddl()
            created = 0
            
            for ddl in ddl_list:
                try:
                    cur.execute(ddl)
                    created += 1
                except sqlite3.OperationalError:
                    pass  # Table/index déjà existant
            
            con.commit()
            
            # Mettre à jour la version du schéma
            cur.execute(
                """INSERT OR REPLACE INTO CacheMeta (key, value, updated_at)
                   VALUES ('schema_version', ?, ?)""",
                (CACHE_SCHEMA_VERSION, _get_iso_now()),
            )
            con.commit()
            
            msg = f"Tables de cache vérifiées ({created} opérations)"
            logger.info(msg)
            return True, msg
            
    except Exception as e:
        msg = f"Erreur lors de la création des tables de cache: {e}"
        logger.error(msg)
        return False, msg


def rebuild_match_cache(db_path: str, xuid: str | None = None) -> tuple[bool, str]:
    """Reconstruit le cache MatchCache depuis MatchStats.
    
    Args:
        db_path: Chemin vers la base de données.
        xuid: XUID optionnel pour filtrer (None = tous les joueurs).
        
    Returns:
        Tuple (success, message).
    """
    logger.info(f"Reconstruction du cache MatchCache{f' pour {xuid}' if xuid else ''}...")
    
    try:
        # Import des modules nécessaires
        import pandas as pd
        from src.db.loaders import load_matches, get_players_from_db
        from src.analysis.sessions import compute_sessions
        from src.analysis.filters import mark_firefight
        
        with get_connection(db_path) as con:
            cur = con.cursor()
            
            # Vérifier que MatchStats existe
            if not _has_table(con, "MatchStats"):
                return False, "Table MatchStats introuvable"
            
            # Déterminer les joueurs à traiter
            if xuid:
                xuids_to_process = [xuid]
            else:
                # Charger tous les joueurs depuis la table Players
                players = get_players_from_db(db_path)
                if not players:
                    return False, "Aucun joueur trouvé dans la table Players"
                xuids_to_process = [p["xuid"] for p in players]
            
            total_inserted = 0
            
            for player_xuid in xuids_to_process:
                logger.info(f"Traitement du joueur {player_xuid}...")
                
                # Charger les matchs (retourne une liste de MatchRow)
                matches = load_matches(db_path, player_xuid)
                
                if not matches:
                    logger.info(f"  Aucun match trouvé pour {player_xuid}")
                    continue
                
                # Convertir en DataFrame
                df = pd.DataFrame([
                    {
                        "match_id": m.match_id,
                        "start_time": m.start_time,
                        "map_id": m.map_id,
                        "map_name": m.map_name,
                        "playlist_id": m.playlist_id,
                        "playlist_name": m.playlist_name,
                        "pair_id": m.map_mode_pair_id,
                        "pair_name": m.map_mode_pair_name,
                        "game_variant_id": m.game_variant_id,
                        "game_variant_name": m.game_variant_name,
                        "outcome": m.outcome,
                        "last_team_id": m.last_team_id,
                        "kda": m.kda,
                        "max_killing_spree": m.max_killing_spree,
                        "headshot_kills": m.headshot_kills,
                        "average_life_seconds": m.average_life_seconds,
                        "time_played_seconds": m.time_played_seconds,
                        "kills": m.kills,
                        "deaths": m.deaths,
                        "assists": m.assists,
                        "accuracy": m.accuracy,
                        "my_team_score": m.my_team_score,
                        "enemy_team_score": m.enemy_team_score,
                        "team_mmr": m.team_mmr,
                        "enemy_mmr": m.enemy_mmr,
                        "xuid": player_xuid,
                    }
                    for m in matches
                ])
                
                if df.empty:
                    continue
            
                # Marquer Firefight
                df = mark_firefight(df)
            
                # Calculer les sessions
                logger.info(f"  Calcul des sessions pour {len(df)} matchs...")
                df = compute_sessions(df, gap_minutes=45)
            
                # Préparer les données pour insertion
                logger.info(f"  Insertion de {len(df)} matchs dans MatchCache...")
            
                inserted = 0
                for _, row in df.iterrows():
                    try:
                        cur.execute(
                            """INSERT OR REPLACE INTO MatchCache (
                                match_id, xuid, start_time,
                                playlist_id, playlist_name,
                                map_id, map_name,
                                pair_id, pair_name,
                                game_variant_id, game_variant_name,
                                outcome, last_team_id,
                                kills, deaths, assists,
                                accuracy, kda, time_played_seconds,
                                average_life_seconds, max_killing_spree, headshot_kills,
                                my_team_score, enemy_team_score,
                                team_mmr, enemy_mmr,
                                session_id, session_label,
                                is_firefight,
                                updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                str(row.get("match_id") or ""),
                                str(player_xuid),
                                str(row.get("start_time") or ""),
                                str(row.get("playlist_id") or ""),
                                str(row.get("playlist_name") or ""),
                                str(row.get("map_id") or ""),
                                str(row.get("map_name") or ""),
                                str(row.get("pair_id") or ""),
                                str(row.get("pair_name") or ""),
                                str(row.get("game_variant_id") or ""),
                                str(row.get("game_variant_name") or ""),
                                int(row.get("outcome") or 0) if row.get("outcome") == row.get("outcome") else None,
                                int(row.get("last_team_id") or 0) if row.get("last_team_id") == row.get("last_team_id") else None,
                                int(row.get("kills") or 0),
                                int(row.get("deaths") or 0),
                                int(row.get("assists") or 0),
                                float(row.get("accuracy") or 0.0) if row.get("accuracy") == row.get("accuracy") else None,
                                float(row.get("kda") or 0.0) if row.get("kda") == row.get("kda") else None,
                                float(row.get("time_played_seconds") or 0.0) if row.get("time_played_seconds") == row.get("time_played_seconds") else None,
                                float(row.get("average_life_seconds") or 0.0) if row.get("average_life_seconds") == row.get("average_life_seconds") else None,
                                int(row.get("max_killing_spree") or 0) if row.get("max_killing_spree") == row.get("max_killing_spree") else None,
                                int(row.get("headshot_kills") or 0) if row.get("headshot_kills") == row.get("headshot_kills") else None,
                                int(row.get("my_team_score") or 0) if row.get("my_team_score") == row.get("my_team_score") else None,
                                int(row.get("enemy_team_score") or 0) if row.get("enemy_team_score") == row.get("enemy_team_score") else None,
                                float(row.get("team_mmr") or 0.0) if row.get("team_mmr") == row.get("team_mmr") else None,
                                float(row.get("enemy_mmr") or 0.0) if row.get("enemy_mmr") == row.get("enemy_mmr") else None,
                                int(row.get("session_id") or 0) if row.get("session_id") == row.get("session_id") else None,
                                str(row.get("session_label") or ""),
                                1 if row.get("is_firefight") else 0,
                                _get_iso_now(),
                            ),
                        )
                        inserted += 1
                    except Exception as e:
                        logger.warning(f"Erreur insertion match {row.get('match_id')}: {e}")
                
                total_inserted += inserted
                logger.info(f"  {inserted} matchs insérés pour {player_xuid}")
            
            con.commit()
            
            # Mettre à jour les métadonnées
            cur.execute(
                """INSERT OR REPLACE INTO CacheMeta (key, value, updated_at)
                   VALUES ('match_cache_count', ?, ?)""",
                (str(total_inserted), _get_iso_now()),
            )
            con.commit()
            
            msg = f"Cache reconstruit: {total_inserted} matchs total"
            logger.info(msg)
            
            # Reconstruire MedalsAggregate
            logger.info("Reconstruction de MedalsAggregate...")
            medals_ok, medals_msg = rebuild_medals_aggregate(db_path)
            if medals_ok:
                logger.info(medals_msg)
            else:
                logger.warning(medals_msg)
            
            return True, msg
            
    except Exception as e:
        msg = f"Erreur lors de la reconstruction du cache: {e}"
        logger.error(msg)
        return False, msg


def rebuild_medals_aggregate(db_path: str) -> tuple[bool, str]:
    """Reconstruit la table MedalsAggregate depuis MatchStats.
    
    Agrège les médailles par joueur (scope global).
    
    Args:
        db_path: Chemin vers la base de données.
        
    Returns:
        Tuple (success, message).
    """
    try:
        from src.db.loaders import load_top_medals, get_players_from_db
        
        with get_connection(db_path) as con:
            cur = con.cursor()
            
            # Vider la table
            cur.execute("DELETE FROM MedalsAggregate")
            
            # Récupérer tous les joueurs
            players = get_players_from_db(db_path)
            if not players:
                return True, "Aucun joueur trouvé"
            
            total_medals = 0
            
            for player in players:
                xuid = player["xuid"]
                
                # Récupérer tous les match_ids pour ce joueur depuis MatchCache
                cur.execute("SELECT match_id FROM MatchCache WHERE xuid = ?", (xuid,))
                match_ids = [row[0] for row in cur.fetchall()]
                
                if not match_ids:
                    continue
                
                # Charger toutes les médailles (pas de limite)
                medals = load_top_medals(db_path, xuid, match_ids, top_n=None)
                
                # Insérer dans MedalsAggregate
                for medal_id, count in medals:
                    cur.execute(
                        """INSERT OR REPLACE INTO MedalsAggregate 
                           (xuid, scope_type, scope_id, medal_id, total_count, computed_at)
                           VALUES (?, 'global', NULL, ?, ?, ?)""",
                        (xuid, medal_id, count, _get_iso_now()),
                    )
                    total_medals += 1
            
            con.commit()
            
            return True, f"MedalsAggregate reconstruit: {total_medals} entrées"
            
    except Exception as e:
        return False, f"Erreur MedalsAggregate: {e}"


def sync_delta(
    db_path: str,
    *,
    match_type: str = "matchmaking",
    max_matches: int = 200,
    with_highlight_events: bool = True,
    with_aliases: bool = True,
) -> tuple[bool, str]:
    """Effectue une synchronisation incrémentale via SPNKr.
    
    Args:
        db_path: Chemin vers la base de données.
        match_type: Type de matchs à récupérer.
        max_matches: Nombre maximum de matchs.
        with_highlight_events: Inclure les highlight events.
        with_aliases: Mettre à jour les alias XUID.
        
    Returns:
        Tuple (success, message).
    """
    logger.info("Synchronisation incrémentale (delta)...")
    
    try:
        from src.ui.sync import sync_all_players
        
        ok, msg = sync_all_players(
            db_path=db_path,
            match_type=match_type,
            max_matches=max_matches,
            with_highlight_events=with_highlight_events,
            with_aliases=with_aliases,
            delta=True,
            timeout_seconds=300,
        )
        
        if ok:
            logger.info(msg)
            # Rebuild cache avec les nouvelles données
            cache_ok, cache_msg = rebuild_match_cache(db_path)
            if cache_ok:
                logger.info(f"Cache mis à jour: {cache_msg}")
            else:
                logger.warning(f"Cache non mis à jour: {cache_msg}")
        else:
            logger.error(msg)
            
        return ok, msg
        
    except ImportError as e:
        msg = f"SPNKr non disponible: {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Erreur lors de la synchronisation: {e}"
        logger.error(msg)
        return False, msg


def sync_full(
    db_path: str,
    *,
    match_type: str = "matchmaking",
    max_matches: int = 1000,
    with_highlight_events: bool = True,
    with_aliases: bool = True,
) -> tuple[bool, str]:
    """Effectue une synchronisation complète via SPNKr.
    
    Args:
        db_path: Chemin vers la base de données.
        match_type: Type de matchs à récupérer.
        max_matches: Nombre maximum de matchs.
        with_highlight_events: Inclure les highlight events.
        with_aliases: Mettre à jour les alias XUID.
        
    Returns:
        Tuple (success, message).
    """
    logger.info("Synchronisation complète...")
    
    try:
        from src.ui.sync import sync_all_players
        
        ok, msg = sync_all_players(
            db_path=db_path,
            match_type=match_type,
            max_matches=max_matches,
            with_highlight_events=with_highlight_events,
            with_aliases=with_aliases,
            delta=False,
            timeout_seconds=600,
        )
        
        if ok:
            logger.info(msg)
            # Rebuild cache avec les nouvelles données
            cache_ok, cache_msg = rebuild_match_cache(db_path)
            if cache_ok:
                logger.info(f"Cache mis à jour: {cache_msg}")
            else:
                logger.warning(f"Cache non mis à jour: {cache_msg}")
        else:
            logger.error(msg)
            
        return ok, msg
        
    except ImportError as e:
        msg = f"SPNKr non disponible: {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Erreur lors de la synchronisation: {e}"
        logger.error(msg)
        return False, msg


def download_assets(db_path: str) -> tuple[bool, str]:
    """Télécharge les assets manquants (médailles, maps).
    
    Args:
        db_path: Chemin vers la base de données.
        
    Returns:
        Tuple (success, message).
    """
    logger.info("Téléchargement des assets...")
    
    try:
        from src.ui.medals import download_missing_medal_icons
        
        # Télécharger les icônes de médailles manquantes
        downloaded = download_missing_medal_icons(db_path)
        
        msg = f"Assets téléchargés: {downloaded} médailles"
        logger.info(msg)
        return True, msg
        
    except ImportError as e:
        msg = f"Module assets non disponible: {e}"
        logger.warning(msg)
        return True, msg  # Non bloquant
    except Exception as e:
        msg = f"Erreur lors du téléchargement des assets: {e}"
        logger.error(msg)
        return False, msg


def print_stats(db_path: str) -> None:
    """Affiche les statistiques de la base de données."""
    logger.info("=== Statistiques de la base de données ===")
    
    try:
        with get_connection(db_path) as con:
            tables = [
                ("MatchStats", "Matchs"),
                ("PlayerMatchStats", "Stats joueur"),
                ("HighlightEvents", "Highlight events"),
                ("XuidAliases", "Alias XUID"),
                ("MatchCache", "Cache matchs"),
            ]
            
            for table, label in tables:
                if _has_table(con, table):
                    count = _count_rows(con, table)
                    logger.info(f"  {label}: {count:,}")
                else:
                    logger.info(f"  {label}: (table absente)")
                    
            # Taille du fichier
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            logger.info(f"  Taille: {size_mb:.2f} MB")
            
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des stats: {e}")


# =============================================================================
# Point d'entrée
# =============================================================================


def main() -> int:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Script de synchronisation unifié pour OpenSpartan Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/sync.py --delta                    # Sync incrémentale
  python scripts/sync.py --full --max-matches 500   # Sync complète (500 matchs)
  python scripts/sync.py --rebuild-cache            # Reconstruit le cache
  python scripts/sync.py --apply-indexes            # Applique les index
  python scripts/sync.py --delta --with-assets      # Sync + téléchargement assets
  python scripts/sync.py --stats                    # Affiche les statistiques
        """,
    )
    
    # Base de données
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Chemin vers la base de données (défaut: auto-détection)",
    )
    
    # Modes de synchronisation
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--delta",
        action="store_true",
        help="Synchronisation incrémentale (nouveaux matchs uniquement)",
    )
    sync_group.add_argument(
        "--full",
        action="store_true",
        help="Synchronisation complète",
    )
    
    # Options de sync
    parser.add_argument(
        "--match-type",
        type=str,
        default="matchmaking",
        choices=["matchmaking", "custom", "all"],
        help="Type de matchs à synchroniser",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=200,
        help="Nombre maximum de matchs à récupérer",
    )
    parser.add_argument(
        "--no-highlight-events",
        action="store_true",
        help="Ne pas récupérer les highlight events",
    )
    parser.add_argument(
        "--no-aliases",
        action="store_true",
        help="Ne pas mettre à jour les alias XUID",
    )
    
    # Opérations de maintenance
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Reconstruit le cache MatchCache",
    )
    parser.add_argument(
        "--apply-indexes",
        action="store_true",
        help="Applique les index optimisés",
    )
    parser.add_argument(
        "--with-assets",
        action="store_true",
        help="Télécharge les assets manquants",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Affiche les statistiques de la DB",
    )
    
    # Verbosité
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbeux",
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Déterminer le chemin de la DB
    db_path = args.db
    if not db_path:
        db_path = get_default_db_path()
        
    if not db_path:
        logger.error("Aucune base de données trouvée. Utilisez --db pour spécifier le chemin.")
        return 1
        
    if not os.path.exists(db_path):
        logger.error(f"Base de données introuvable: {db_path}")
        return 1
        
    logger.info(f"Base de données: {db_path}")
    
    # Exécuter les opérations demandées
    success = True
    
    # Statistiques seules
    if args.stats:
        print_stats(db_path)
        return 0
    
    # Vérifier/créer les tables de cache
    ok, msg = ensure_cache_tables(db_path)
    if not ok:
        logger.error(msg)
        success = False
    
    # Appliquer les index
    if args.apply_indexes or args.delta or args.full:
        ok, msg = apply_indexes(db_path)
        if not ok:
            success = False
    
    # Synchronisation
    if args.delta:
        ok, msg = sync_delta(
            db_path,
            match_type=args.match_type,
            max_matches=args.max_matches,
            with_highlight_events=not args.no_highlight_events,
            with_aliases=not args.no_aliases,
        )
        if not ok:
            success = False
            
    elif args.full:
        ok, msg = sync_full(
            db_path,
            match_type=args.match_type,
            max_matches=args.max_matches,
            with_highlight_events=not args.no_highlight_events,
            with_aliases=not args.no_aliases,
        )
        if not ok:
            success = False
    
    # Reconstruction du cache
    if args.rebuild_cache:
        ok, msg = rebuild_match_cache(db_path)
        if not ok:
            success = False
    
    # Téléchargement des assets
    if args.with_assets:
        ok, msg = download_assets(db_path)
        if not ok:
            success = False
    
    # Afficher les stats finales
    if args.delta or args.full or args.rebuild_cache:
        print_stats(db_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
