"""Génère un mapping "au mieux" entre citations Halo 5 (WikiHalo) et données dispo dans l'app.

Objectif
- Proposer des correspondances exploitables (ou au moins candidates) :
  - vers des champs "stats" disponibles dans les DataFrames (kills, headshot_kills, etc.)
  - vers des médailles Halo Infinite (NameId) via matching de libellés

Sorties
- out/commendations_mapping_assumed.json : correspondances trouvées/supposées
- out/commendations_mapping_unmatched.json : introuvables (aucune correspondance)
- out/commendations_mapping_assumed.csv : vue tabulaire rapide

Notes
- Halo 5 "commendations" != Halo Infinite "medals" : ce mapping est heuristique.
- À valider manuellement (c'est le but de ces fichiers).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    # Ignore les annotations éditoriales ajoutées à la main, ex: "[sic]".
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set[str]:
    stop = {
        "de",
        "d",
        "des",
        "du",
        "la",
        "le",
        "les",
        "un",
        "une",
        "a",
        "à",
        "au",
        "aux",
        "et",
        "en",
        "pour",
        "avec",
        "sur",
        "dans",
        "par",
        "match",
        "matchs",
        "partie",
        "parties",
    }
    return {t for t in _norm(s).split(" ") if t and t not in stop}


@dataclass(frozen=True)
class Medal:
    name_id: int
    label: str
    label_norm: str
    tokens: set[str]


def _load_medals(paths: list[Path]) -> list[Medal]:
    medals: dict[int, str] = {}
    for p in paths:
        if not p.exists():
            continue
        data = _load_json(p)
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            try:
                nid = int(str(k))
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                medals.setdefault(nid, v.strip())
            elif isinstance(v, dict):
                for kk in ("fr", "name_fr", "label_fr", "label", "name", "en"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        medals.setdefault(nid, vv.strip())
                        break

    out: list[Medal] = []
    for nid, label in medals.items():
        out.append(Medal(name_id=nid, label=label, label_norm=_norm(label), tokens=_token_set(label)))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _best_medal_match(name: str, medals: list[Medal]) -> dict[str, Any] | None:
    name_norm = _norm(name)
    name_tokens = _token_set(name)

    # 1) Exact match sur normalisation
    for m in medals:
        if m.label_norm == name_norm and name_norm:
            return {
                "type": "medal",
                "name_id": m.name_id,
                "medal_label": m.label,
                "match": "exact_norm",
                "confidence": 0.98,
            }

    # 2) Match token Jaccard
    best = None
    best_score = 0.0
    for m in medals:
        score = _jaccard(name_tokens, m.tokens)
        if score > best_score:
            best_score = score
            best = m

    if best and best_score >= 0.70:
        return {
            "type": "medal",
            "name_id": best.name_id,
            "medal_label": best.label,
            "match": "token_jaccard",
            "confidence": round(min(0.95, 0.60 + best_score * 0.5), 3),
        }

    return None


_STAT_RULES: list[tuple[str, list[str], str, float]] = [
    # (key, keywords, expression, base_confidence)
    (
        "wins",
        ["gagne", "gagner", "win", "victoire", "remporter"],
        "wins = count(outcome == 2)",
        0.75,
    ),
    (
        "matches_played",
        ["jouez", "jouer", "terminez", "terminer", "complete", "compl\u00e9tez"],
        "matches = count(matches)",
        0.45,
    ),
    (
        "kills",
        ["tuez", "tuer", "kills", "\u00e9liminez", "eliminez", "\u00e9liminations"],
        "kills = sum(kills)",
        0.55,
    ),
    (
        "headshot_kills",
        ["tirs a la tete", "tirs a la tete", "headshot", "tete", "\u00e0 la t\u00eate"],
        "headshots = sum(headshot_kills)",
        0.80,
    ),
    (
        "assists",
        ["assistance", "assistances", "assist"],
        "assists = sum(assists)",
        0.75,
    ),
    (
        "deaths",
        ["morts", "mort", "deaths"],
        "deaths = sum(deaths)",
        0.40,
    ),
]


def _match_stat_rule(name: str, desc: str, category: str) -> dict[str, Any] | None:
    blob = " ".join([name or "", desc or "", category or ""]).strip()
    bnorm = _norm(blob)

    cat_norm = _norm(category or "")

    # Si la citation est clairement "spécifique" (arme/véhicule/ennemi),
    # on évite de la mapper à des kills génériques (trop trompeur sans event breakdown).
    if cat_norm in {"arme", "vehicule", "ennemi"}:
        # Cas fréquents: "à l'aide de ..." / "occupées" / nom d'ennemi (élite, chasseur...)
        if any(k in bnorm for k in ["a l aide", "a l'aide", "a l'aide d", "occup", "vehicule", "elite", "chasseur", "grognard", "sentinelle", "chevalier", "rampant", "soldat", "prometheen", "covenant", "forerunner"]):
            return None

    # Règles spécifiques: si headshot mentionné => headshot, même si "tuez" aussi.
    if any(k in bnorm for k in ["headshot", "tirs a la tete", "a la tete", "tete"]):
        return {
            "type": "stat",
            "stat": "headshot_kills",
            "expression": "headshots = sum(headshot_kills)",
            "match": "keyword",
            "confidence": 0.85,
        }

    # Victoires (ex: "Win matches")
    if any(k in bnorm for k in ["win", "gagne", "gagner", "victoire", "remporter"]):
        return {
            "type": "stat",
            "stat": "wins",
            "expression": "wins = count(outcome == 2)",
            "match": "keyword",
            "confidence": 0.78,
        }

    # Matches joués (terminer des parties)
    if any(k in bnorm for k in ["terminez", "terminer", "jouez", "jouer", "complete", "completer", "completez", "compl\u00e9tez"]):
        return {
            "type": "stat",
            "stat": "matches_played",
            "expression": "matches = count(matches)",
            "match": "keyword",
            "confidence": 0.70,
        }

    for stat, keywords, expr, conf in _STAT_RULES:
        if any(_norm(k) in bnorm for k in keywords):
            return {
                "type": "stat",
                "stat": stat,
                "expression": expr,
                "match": "keyword",
                "confidence": conf,
            }

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Génère un mapping heuristique citations Halo 5 -> stats/médailles")
    ap.add_argument(
        "--commendations-json",
        default=str(Path("data") / "wiki" / "halo5_commendations_fr.json"),
        help="Chemin du JSON de citations Halo 5",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path("out")),
        help="Dossier de sortie",
    )
    args = ap.parse_args()

    comm_path = Path(args.commendations_json)
    data = _load_json(comm_path)
    items: list[dict[str, Any]] = list(data.get("items") or [])

    medals = _load_medals(
        [
            Path("static") / "medals" / "medals_fr.json",
            Path("static") / "medals" / "medals_en.json",
        ]
    )

    assumed: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for it in items:
        name = str(it.get("name") or "").strip()
        desc = str(it.get("description") or "").strip()
        category = str(it.get("category") or "").strip()
        tiers = it.get("tiers") or []
        master_count = it.get("master_count")

        # 1) Tentative medal (forte si exact)
        medal_match = _best_medal_match(name, medals)

        # 2) Tentative stat
        stat_match = _match_stat_rule(name, desc, category)

        # 3) Candidate "kills génériques" (faible confiance) pour aider le tri
        generic_kills = None
        if not stat_match and any(k in _norm(desc) for k in ["tuez", "tuer", "eliminez", "\u00e9liminez"]):
            generic_kills = {
                "type": "stat",
                "stat": "kills",
                "expression": "kills = sum(kills)",
                "match": "fallback_generic",
                "confidence": 0.25,
            }

        # Choix: si medal_match très fort, on prend medal. Sinon, si stat_match fort, on prend stat.
        chosen = None
        candidates: list[dict[str, Any]] = []
        if medal_match:
            candidates.append(medal_match)
        if stat_match:
            candidates.append(stat_match)
        if generic_kills:
            candidates.append(generic_kills)

        # Choix plus conservateur: on ne "choisit" que si confiance suffisante.
        if medal_match and medal_match.get("confidence", 0) >= 0.90:
            chosen = medal_match
        elif stat_match and stat_match.get("confidence", 0) >= 0.70:
            chosen = stat_match
        elif medal_match and medal_match.get("confidence", 0) >= 0.80:
            chosen = medal_match

        record = {
            "name": name,
            "category": category,
            "description": desc,
            "master_count": master_count,
            "tiers": tiers,
            "chosen": chosen,
            "candidates": candidates,
            "notes": "Heuristique: à valider manuellement",
        }

        if chosen is None:
            unmatched.append(record)
        else:
            assumed.append(record)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assumed_path = out_dir / "commendations_mapping_assumed.json"
    unmatched_path = out_dir / "commendations_mapping_unmatched.json"
    assumed_path.write_text(json.dumps({"count": len(assumed), "items": assumed}, ensure_ascii=False, indent=2), encoding="utf-8")
    unmatched_path.write_text(
        json.dumps({"count": len(unmatched), "items": unmatched}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV pour tri rapide
    csv_path = out_dir / "commendations_mapping_assumed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "name",
                "master_count",
                "chosen_type",
                "chosen_key",
                "confidence",
                "expression_or_medal",
            ],
        )
        w.writeheader()
        for r in assumed:
            ch = r.get("chosen") or {}
            ctype = ch.get("type")
            if ctype == "stat":
                ckey = ch.get("stat")
                expr = ch.get("expression")
            elif ctype == "medal":
                ckey = ch.get("name_id")
                expr = ch.get("medal_label")
            else:
                ckey = ""
                expr = ""

            w.writerow(
                {
                    "category": r.get("category") or "",
                    "name": r.get("name") or "",
                    "master_count": r.get("master_count"),
                    "chosen_type": ctype,
                    "chosen_key": ckey,
                    "confidence": ch.get("confidence"),
                    "expression_or_medal": expr,
                }
            )

    print(f"OK: assumed={len(assumed)} unmatched={len(unmatched)}")
    print(f"- {assumed_path}")
    print(f"- {unmatched_path}")
    print(f"- {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
