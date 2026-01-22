"""Migration des aliases du fichier JSON vers la table XuidAliases en DB.

Usage:
    python scripts/migrate_aliases_to_db.py --db data/spnkr_gt_JGtm.db
    python scripts/migrate_aliases_to_db.py --db data/spnkr_gt_JGtm.db --json xuid_aliases.json

Ce script:
1. Lit le fichier xuid_aliases.json (ou celui spécifié)
2. Insère les aliases dans la table XuidAliases de la DB
3. Ne remplace pas les aliases existants (INSERT OR IGNORE)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _get_iso_now() -> str:
    """Retourne le timestamp ISO 8601 actuel (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def _ensure_xuid_aliases_table(con: sqlite3.Connection) -> None:
    """Crée la table XuidAliases si elle n'existe pas."""
    cur = con.cursor()
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS XuidAliases (
   Xuid TEXT PRIMARY KEY,
   Gamertag TEXT NOT NULL,
   LastSeen TEXT,
   Source TEXT,
   UpdatedAt TEXT
)
"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_XuidAliases_Gamertag ON XuidAliases(Gamertag COLLATE NOCASE)"
    )
    con.commit()


def migrate_aliases_json_to_db(
    json_path: str,
    db_path: str,
    *,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Migre les aliases du fichier JSON vers la table XuidAliases.
    
    Args:
        json_path: Chemin vers le fichier JSON d'aliases
        db_path: Chemin vers la DB SQLite
        overwrite: Si True, remplace les aliases existants
    
    Returns:
        Tuple (inserted, skipped)
    """
    # Lire le JSON
    if not os.path.exists(json_path):
        print(f"[WARN] Fichier JSON non trouvé: {json_path}")
        return 0, 0
    
    with open(json_path, "r", encoding="utf-8") as f:
        aliases = json.load(f)
    
    if not isinstance(aliases, dict):
        print(f"[ERROR] Le fichier JSON doit contenir un objet (dict), pas {type(aliases)}")
        return 0, 0
    
    # Ouvrir la DB
    con = sqlite3.connect(db_path)
    try:
        _ensure_xuid_aliases_table(con)
        
        cur = con.cursor()
        now = _get_iso_now()
        inserted = 0
        skipped = 0
        
        for xuid, gamertag in aliases.items():
            xuid_str = str(xuid).strip()
            gt_str = str(gamertag).strip()
            
            if not xuid_str or not gt_str:
                skipped += 1
                continue
            
            if overwrite:
                cur.execute(
                    """INSERT OR REPLACE INTO XuidAliases 
                       (Xuid, Gamertag, Source, UpdatedAt)
                       VALUES (?, ?, 'migrated', ?)""",
                    (xuid_str, gt_str, now),
                )
                inserted += 1
            else:
                cur.execute(
                    """INSERT OR IGNORE INTO XuidAliases 
                       (Xuid, Gamertag, Source, UpdatedAt)
                       VALUES (?, ?, 'migrated', ?)""",
                    (xuid_str, gt_str, now),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
        
        con.commit()
        return inserted, skipped
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migre les aliases du fichier JSON vers la table XuidAliases en DB"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Chemin vers la DB SQLite",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Chemin vers le fichier JSON d'aliases (défaut: xuid_aliases.json à la racine)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace les aliases existants (par défaut: ignore)",
    )
    
    args = parser.parse_args()
    
    # Déterminer le chemin du JSON
    if args.json:
        json_path = args.json
    else:
        repo_root = Path(__file__).resolve().parent.parent
        json_path = str(repo_root / "xuid_aliases.json")
    
    print(f"[INFO] Migration des aliases:")
    print(f"  - JSON: {json_path}")
    print(f"  - DB: {args.db}")
    print(f"  - Mode: {'OVERWRITE' if args.overwrite else 'IGNORE si existe'}")
    
    inserted, skipped = migrate_aliases_json_to_db(
        json_path,
        args.db,
        overwrite=args.overwrite,
    )
    
    print(f"[OK] {inserted} aliases insérés, {skipped} ignorés")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
