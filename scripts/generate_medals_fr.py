"""Génère un fichier de traductions FR pour les médailles rencontrées dans la DB.

But:
- créer un squelette `static/medals/medals_fr.json` (mapping NameId -> "")
- à remplir ensuite manuellement

Usage:
  python scripts/generate_medals_fr.py --db path/to/OpenSpartan.db

Optionnel:
  --out static/medals/medals_fr.json

Notes:
- On utilise `json_each` (SQLite JSON1) pour parcourir MatchStats.
- On extrait tous les NameId trouvés (tous joueurs confondus).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Iterable


SQL_DISTINCT_MEDAL_IDS = """
WITH base AS (
    SELECT ResponseBody AS Body
    FROM MatchStats
),
players AS (
    SELECT j.value AS PlayerObj
    FROM base
    JOIN json_each(json_extract(base.Body, '$.Players')) AS j
),
pts AS (
    SELECT t.value AS TeamStats
    FROM players
    JOIN json_each(json_extract(players.PlayerObj, '$.PlayerTeamStats')) AS t
),
medals AS (
    SELECT CAST(json_extract(m.value, '$.NameId') AS INTEGER) AS NameId
    FROM pts
    JOIN json_each(json_extract(pts.TeamStats, '$.Stats.CoreStats.Medals')) AS m
)
SELECT DISTINCT NameId
FROM medals
WHERE NameId IS NOT NULL
ORDER BY NameId ASC;
"""


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _fetch_ids(db_path: str) -> list[int]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(SQL_DISTINCT_MEDAL_IDS)
        out: list[int] = []
        for (nid,) in cur.fetchall():
            try:
                out.append(int(nid))
            except Exception:
                continue
        return out
    finally:
        con.close()


def _merge_existing(existing: dict, ids: Iterable[int]) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(existing, dict):
        for k, v in existing.items():
            if isinstance(k, (str, int)):
                ks = str(k)
                if isinstance(v, str):
                    out[ks] = v
                elif isinstance(v, dict):
                    # on conserve une éventuelle structure, mais on normalise vers string si possible
                    val = v.get("fr") or v.get("name_fr") or v.get("label") or v.get("name")
                    if isinstance(val, str):
                        out[ks] = val

    for nid in ids:
        key = str(int(nid))
        out.setdefault(key, "")

    # tri
    return {k: out[k] for k in sorted(out.keys(), key=lambda x: int(x) if x.isdigit() else x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chemin vers le fichier .db OpenSpartan")
    ap.add_argument("--out", default=os.path.join("static", "medals", "medals_fr.json"))
    args = ap.parse_args()

    db_path = args.db
    out_path = args.out

    if not os.path.exists(db_path):
        raise SystemExit(f"DB introuvable: {db_path}")

    ids = _fetch_ids(db_path)

    existing: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}

    merged = _merge_existing(existing, ids)
    _ensure_dir(out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(ids)} NameId écrits dans {out_path}")


if __name__ == "__main__":
    main()
