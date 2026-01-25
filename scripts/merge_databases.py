#!/usr/bin/env python
"""Script de fusion de plusieurs bases de données Halo en une DB unifiée.

Ce script permet de regrouper les DBs de plusieurs joueurs en une seule base,
ce qui permet :
- Des comparaisons de stats entre joueurs
- Des analyses de coéquipiers plus précises
- Un seul fichier à gérer

Usage:
    python scripts/merge_databases.py <output_db> <input_db1> [input_db2] [...]
    python scripts/merge_databases.py --config merge_config.json
    
Options:
    --config FILE    Utiliser un fichier de configuration JSON
    --dry-run        Simuler sans créer de fichier
    --verbose        Afficher les détails

Exemple:
    python scripts/merge_databases.py data/halo_unified.db data/halo_guillaume.db data/halo_elisa.db data/halo_arnaud.db

Configuration JSON (optionnelle):
{
    "output": "data/halo_unified.db",
    "inputs": [
        {"path": "data/halo_guillaume.db", "label": "Guillaume"},
        {"path": "data/halo_elisa.db", "label": "Élisa"},
        {"path": "data/halo_arnaud.db", "label": "Arnaud"}
    ],
    "options": {
        "copy_highlight_events": true,
        "copy_player_match_stats": true,
        "rebuild_cache": true
    }
}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MergeInput:
    """Définition d'une DB source à fusionner."""
    path: str
    label: str | None = None
    xuid: str | None = None  # Auto-détecté si non fourni


@dataclass 
class MergeConfig:
    """Configuration complète de la fusion."""
    output: str
    inputs: list[MergeInput] = field(default_factory=list)
    copy_highlight_events: bool = True
    copy_player_match_stats: bool = True
    copy_medals: bool = True
    rebuild_cache: bool = True
    

def load_config_from_json(path: str) -> MergeConfig:
    """Charge la configuration depuis un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    inputs = []
    for inp in data.get("inputs", []):
        inputs.append(MergeInput(
            path=inp["path"],
            label=inp.get("label"),
            xuid=inp.get("xuid"),
        ))
    
    opts = data.get("options", {})
    return MergeConfig(
        output=data["output"],
        inputs=inputs,
        copy_highlight_events=opts.get("copy_highlight_events", True),
        copy_player_match_stats=opts.get("copy_player_match_stats", True),
        copy_medals=opts.get("copy_medals", True),
        rebuild_cache=opts.get("rebuild_cache", True),
    )


# =============================================================================
# Helpers DB
# =============================================================================

def get_table_names(con: sqlite3.Connection) -> list[str]:
    """Liste toutes les tables d'une DB."""
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]


def get_primary_xuid(con: sqlite3.Connection, db_path: str | None = None) -> str | None:
    """Détecte le XUID principal d'une DB.
    
    Stratégie:
    1. Déduire le gamertag depuis le nom de fichier (spnkr_gt_<gamertag>.db)
    2. Chercher ce gamertag dans XuidAliases
    3. Fallback: prendre le XUID le plus fréquent dans XuidAliases
    """
    # Stratégie 1: Déduire depuis le nom de fichier
    if db_path:
        import re
        match = re.search(r'spnkr_gt_([^/\\]+)\.db$', str(db_path), re.IGNORECASE)
        if match:
            gamertag = match.group(1)
            try:
                # Note: colonnes avec majuscules (Xuid, Gamertag)
                cur = con.execute(
                    "SELECT Xuid FROM XuidAliases WHERE LOWER(Gamertag) = LOWER(?) LIMIT 1",
                    (gamertag,)
                )
                row = cur.fetchone()
                if row:
                    return row[0]
            except Exception:
                pass
    
    # Stratégie 2: SyncMeta
    try:
        cur = con.execute("SELECT value FROM SyncMeta WHERE key = 'xuid'")
        row = cur.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    
    # Stratégie 3: XUID le plus fréquent dans XuidAliases (en comptant les occurrences)
    # Note: Ce n'est pas idéal mais c'est un fallback
    try:
        cur = con.execute("""
            SELECT Xuid, COUNT(*) as cnt 
            FROM XuidAliases 
            GROUP BY Xuid 
            ORDER BY cnt DESC 
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    
    return None


def get_gamertag_for_xuid(con: sqlite3.Connection, xuid: str) -> str | None:
    """Récupère le gamertag associé à un XUID."""
    try:
        # Note: les colonnes sont Xuid, Gamertag, LastSeen (avec majuscules)
        cur = con.execute(
            "SELECT Gamertag FROM XuidAliases WHERE Xuid = ? ORDER BY LastSeen DESC LIMIT 1",
            (xuid,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def copy_table_data(
    src_con: sqlite3.Connection,
    dst_con: sqlite3.Connection,
    table_name: str,
    *,
    on_conflict: str = "IGNORE",
    batch_size: int = 1000,
) -> int:
    """Copie les données d'une table source vers destination.
    
    Returns:
        Nombre de lignes copiées.
    """
    # Récupérer la structure de la table
    cur = src_con.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]
    
    if not columns:
        return 0
    
    # Construire la requête d'insertion
    placeholders = ", ".join(["?"] * len(columns))
    columns_str = ", ".join(columns)
    
    insert_sql = f"""
        INSERT OR {on_conflict} INTO {table_name} ({columns_str})
        VALUES ({placeholders})
    """
    
    # Copier par batch
    cur = src_con.execute(f"SELECT {columns_str} FROM {table_name}")
    total = 0
    batch = []
    
    for row in cur:
        batch.append(row)
        if len(batch) >= batch_size:
            dst_con.executemany(insert_sql, batch)
            total += len(batch)
            batch = []
    
    if batch:
        dst_con.executemany(insert_sql, batch)
        total += len(batch)
    
    return total


def create_players_table(con: sqlite3.Connection) -> None:
    """Crée la table Players pour la DB unifiée."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS Players (
            xuid TEXT PRIMARY KEY,
            gamertag TEXT,
            label TEXT,
            source_db TEXT,
            first_match_date TEXT,
            last_match_date TEXT,
            total_matches INTEGER DEFAULT 0,
            added_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_Players_gamertag ON Players(gamertag)")


def register_player(
    con: sqlite3.Connection,
    xuid: str,
    gamertag: str | None,
    label: str | None,
    source_db: str,
) -> None:
    """Enregistre un joueur dans la table Players."""
    con.execute("""
        INSERT INTO Players (xuid, gamertag, label, source_db, added_at, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(xuid) DO UPDATE SET
            gamertag = COALESCE(excluded.gamertag, gamertag),
            label = COALESCE(excluded.label, label),
            updated_at = datetime('now')
    """, (xuid, gamertag, label, source_db))


def update_player_stats(con: sqlite3.Connection, xuid: str, match_count: int) -> None:
    """Met à jour les statistiques d'un joueur.
    
    Note: MatchStats n'a pas de colonne XUID directe, donc on passe le count
    depuis la source où on l'a calculé.
    """
    con.execute("""
        UPDATE Players SET
            total_matches = ?,
            updated_at = datetime('now')
        WHERE xuid = ?
    """, (match_count, xuid))


# =============================================================================
# Fusion principale
# =============================================================================

def merge_databases(config: MergeConfig, *, dry_run: bool = False, verbose: bool = False) -> dict:
    """Fusionne plusieurs DBs en une seule.
    
    Returns:
        Statistiques de la fusion.
    """
    stats = {
        "players": 0,
        "match_stats": 0,
        "player_match_stats": 0,
        "highlight_events": 0,
        "xuid_aliases": 0,
        "medals": 0,
    }
    
    output_path = Path(config.output)
    
    if dry_run:
        print(f"🔍 Mode simulation - pas de modification")
        print(f"   Output serait: {output_path}")
        return stats
    
    # Créer le dossier de sortie si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Supprimer la DB de sortie si elle existe
    if output_path.exists():
        backup = output_path.with_suffix(".db.bak")
        print(f"📦 Backup de l'ancienne DB: {backup}")
        shutil.copy(output_path, backup)
        output_path.unlink()
    
    # Utiliser la première DB comme base
    first_input = config.inputs[0]
    print(f"📋 Copie de la structure depuis: {first_input.path}")
    shutil.copy(first_input.path, output_path)
    
    dst_con = sqlite3.connect(str(output_path))
    dst_con.execute("PRAGMA journal_mode=WAL")
    
    # Créer la table Players
    create_players_table(dst_con)
    
    # Traiter chaque DB source
    for inp in tqdm(config.inputs, desc="DBs"):
        src_path = Path(inp.path)
        if not src_path.exists():
            print(f"⚠️  DB introuvable: {src_path}")
            continue
        
        src_con = sqlite3.connect(str(src_path))
        
        # Détecter le XUID si non fourni (passer le chemin pour la déduction depuis le nom)
        xuid = inp.xuid or get_primary_xuid(src_con, str(src_path))
        if not xuid:
            print(f"⚠️  XUID non détectable pour: {src_path}")
            src_con.close()
            continue
        
        gamertag = get_gamertag_for_xuid(src_con, xuid)
        label = inp.label or gamertag or xuid[:15]
        
        if verbose:
            print(f"\n  → {label} ({xuid})")
        
        # Enregistrer le joueur
        register_player(dst_con, xuid, gamertag, label, str(src_path.name))
        stats["players"] += 1
        
        # Tables principales à copier
        tables_to_copy = [
            ("MatchStats", "match_stats"),
            ("XuidAliases", "xuid_aliases"),
        ]
        
        if config.copy_player_match_stats:
            tables_to_copy.append(("PlayerMatchStats", "player_match_stats"))
        
        if config.copy_highlight_events:
            tables_to_copy.append(("HighlightEvents", "highlight_events"))
        
        if config.copy_medals:
            tables_to_copy.append(("MatchMedals", "medals"))
        
        # Copier les tables et compter les matchs
        src_tables = get_table_names(src_con)
        match_count = 0
        for table, stat_key in tables_to_copy:
            if table in src_tables:
                # Ne pas recopier depuis la première DB (déjà copiée)
                if inp == config.inputs[0]:
                    cur = src_con.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                else:
                    count = copy_table_data(src_con, dst_con, table)
                stats[stat_key] += count
                if verbose:
                    print(f"    {table}: {count} lignes")
                
                # Garder le count de MatchStats pour les stats du joueur
                if table == "MatchStats":
                    match_count = count
        
        # Mettre à jour les stats du joueur
        update_player_stats(dst_con, xuid, match_count)
        
        src_con.close()
        dst_con.commit()
    
    # Créer les index manquants
    print("\n📇 Création des index...")
    _create_indexes(dst_con)
    
    # Rebuild cache si demandé
    if config.rebuild_cache:
        print("🔄 Reconstruction du cache...")
        # Note: on pourrait appeler migrate_to_cache ici
        # Pour l'instant, on laisse l'utilisateur le faire manuellement
        print("   → Exécuter: python scripts/migrate_to_cache.py " + str(output_path))
    
    dst_con.commit()
    dst_con.close()
    
    return stats


def _create_indexes(con: sqlite3.Connection) -> None:
    """Crée les index pour la DB unifiée."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_MatchStats_XUID ON MatchStats(XUID)",
        "CREATE INDEX IF NOT EXISTS idx_MatchStats_StartTime ON MatchStats(MatchInfo_StartTime)",
        "CREATE INDEX IF NOT EXISTS idx_MatchStats_MatchId ON MatchStats(MatchId)",
        "CREATE INDEX IF NOT EXISTS idx_XuidAliases_xuid ON XuidAliases(xuid)",
        "CREATE INDEX IF NOT EXISTS idx_PlayerMatchStats_MatchId ON PlayerMatchStats(match_id)",
        "CREATE INDEX IF NOT EXISTS idx_HighlightEvents_MatchId ON HighlightEvents(match_id)",
    ]
    for idx_sql in indexes:
        try:
            con.execute(idx_sql)
        except Exception:
            pass


# =============================================================================
# Point d'entrée
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fusionne plusieurs bases de données Halo en une DB unifiée.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_db", nargs="?", help="Chemin de la DB de sortie")
    parser.add_argument("input_dbs", nargs="*", help="Chemins des DBs à fusionner")
    parser.add_argument("--config", help="Fichier de configuration JSON")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    # Charger la configuration
    if args.config:
        config = load_config_from_json(args.config)
    elif args.output_db and args.input_dbs:
        config = MergeConfig(
            output=args.output_db,
            inputs=[MergeInput(path=p) for p in args.input_dbs],
        )
    else:
        parser.print_help()
        print("\n❌ Erreur: spécifier --config ou output_db + input_dbs")
        sys.exit(1)
    
    if not config.inputs:
        print("❌ Aucune DB source spécifiée")
        sys.exit(1)
    
    print("=" * 60)
    print("🔗 FUSION DE BASES DE DONNÉES HALO")
    print("=" * 60)
    print(f"   Output: {config.output}")
    print(f"   Sources: {len(config.inputs)} DB(s)")
    for inp in config.inputs:
        print(f"     - {inp.path}" + (f" ({inp.label})" if inp.label else ""))
    print()
    
    stats = merge_databases(config, dry_run=args.dry_run, verbose=args.verbose)
    
    print()
    print("=" * 60)
    print("✅ FUSION TERMINÉE")
    print("=" * 60)
    print(f"   Joueurs: {stats['players']}")
    print(f"   MatchStats: {stats['match_stats']}")
    print(f"   PlayerMatchStats: {stats['player_match_stats']}")
    print(f"   HighlightEvents: {stats['highlight_events']}")
    print(f"   XuidAliases: {stats['xuid_aliases']}")
    print()
    
    if not args.dry_run:
        print("📝 Prochaines étapes:")
        print(f"   1. python scripts/migrate_to_cache.py {config.output}")
        print(f"   2. python scripts/compute_historical_performance.py {config.output}")
        print(f"   3. Mettre à jour db_path dans l'app: {config.output}")


if __name__ == "__main__":
    main()
