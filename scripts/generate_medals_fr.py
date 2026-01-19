"""Génère un mapping de libellés de médailles (NameId -> label).

But:
- créer/maintenir un JSON (ex: static/medals/medals_fr.json) au format `{NameId: "label"}`
- optionnellement remplir automatiquement les libellés via un JSON "metadata"
    (structure: `medals[].nameId` + `medals[].name.translations[<lang>]`).

Usage:
    python scripts/generate_medals_fr.py --db path/to/OpenSpartan.db --out static/medals/medals_fr.json

Optionnel:
    --metadata path/to/medals_metadata.json
    --lang fr-FR|en-US|...
    --ids db|metadata|union

Notes:
- On utilise `json_each` (SQLite JSON1) pour parcourir MatchStats.
- `--ids db` génère uniquement les NameId rencontrés en DB.
- `--ids metadata` / `--ids union` permettent un fichier plus complet.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import urllib.request
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


def _load_json_from_source(source: str) -> dict:
    """Charge un JSON depuis un fichier local ou une URL."""
    s = (source or "").strip()
    if not s:
        return {}

    if os.path.exists(s):
        try:
            with open(s, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    if s.lower().startswith(("http://", "https://")):
        # urllib suffit dans la majorité des environnements Python.
        # Si ton environnement a un souci de CA/SSL, préfère un fichier local.
        try:
            with urllib.request.urlopen(s, timeout=30) as resp:
                data = resp.read().decode("utf-8", "ignore")
            return json.loads(data) or {}
        except Exception:
            return {}

    return {}


def _build_nameid_to_label(metadata: dict, *, lang: str) -> dict[str, str]:
    """Construit une map NameId(str) -> libellé depuis un payload metadata."""
    if not isinstance(metadata, dict):
        return {}
    medals = metadata.get("medals")
    if not isinstance(medals, list):
        return {}

    out: dict[str, str] = {}
    for m in medals:
        if not isinstance(m, dict):
            continue
        nid = m.get("nameId")
        try:
            key = str(int(nid))
        except Exception:
            continue

        name = m.get("name")
        label: str | None = None
        if isinstance(name, dict):
            tr = name.get("translations")
            if isinstance(tr, dict):
                val = tr.get(lang)
                if isinstance(val, str) and val.strip():
                    label = val.strip()
            if label is None:
                val = name.get("value")
                if isinstance(val, str) and val.strip():
                    label = val.strip()

        if label:
            out[key] = label

    return out


def _extract_nameids_from_metadata(metadata: dict) -> list[int]:
    if not isinstance(metadata, dict):
        return []
    medals = metadata.get("medals")
    if not isinstance(medals, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for m in medals:
        if not isinstance(m, dict):
            continue
        nid = m.get("nameId")
        try:
            i = int(nid)
        except Exception:
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    out.sort()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chemin vers le fichier .db OpenSpartan")
    ap.add_argument("--out", default=os.path.join("static", "medals", "medals_fr.json"))
    ap.add_argument(
        "--metadata",
        default="",
        help="Chemin/URL d'un JSON metadata pour remplir les libellés automatiquement",
    )
    ap.add_argument(
        "--lang",
        default="fr-FR",
        help="Code de langue à utiliser dans metadata (ex: fr-FR, en-US)",
    )
    ap.add_argument(
        "--ids",
        choices=("db", "metadata", "union"),
        default="db",
        help=(
            "Source des NameId à inclure: db (seulement ceux présents dans la DB), "
            "metadata (tous ceux du JSON metadata), union (les deux)."
        ),
    )
    args = ap.parse_args()

    db_path = args.db
    out_path = args.out

    if not os.path.exists(db_path):
        raise SystemExit(f"DB introuvable: {db_path}")

    db_ids: list[int] = _fetch_ids(db_path)

    meta_map: dict[str, str] = {}
    meta_ids: list[int] = []
    if isinstance(args.metadata, str) and args.metadata.strip():
        meta = _load_json_from_source(args.metadata)
        meta_map = _build_nameid_to_label(meta, lang=str(args.lang or "fr-FR"))
        meta_ids = _extract_nameids_from_metadata(meta)

    if args.ids == "db":
        ids = db_ids
    elif args.ids == "metadata":
        ids = meta_ids
    else:
        ids = sorted(set(db_ids).union(meta_ids))

    existing: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}

    merged = _merge_existing(existing, ids)
    if meta_map:
        # Remplit uniquement les valeurs vides, sans écraser les traductions déjà renseignées.
        for k, v in list(merged.items()):
            if isinstance(v, str) and not v.strip():
                fill = meta_map.get(str(k))
                if isinstance(fill, str) and fill.strip():
                    merged[k] = fill.strip()
    _ensure_dir(out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    filled = 0
    if meta_map:
        filled = sum(1 for v in merged.values() if isinstance(v, str) and v.strip())
    print(f"OK: {len(ids)} NameId écrits dans {out_path}" + (f" | {filled} libellés non-vides" if meta_map else ""))


if __name__ == "__main__":
    main()
