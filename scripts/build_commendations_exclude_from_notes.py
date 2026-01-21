"""Construit/Met à jour la blacklist des citations Halo 5 à exclure.

Source: fichiers de mapping dans out/ (ex: commendations_mapping_*_old.json)
Règle: si le champ "notes" (ou "note") contient "a supprimer" / "à supprimer" (case-insensitive),
alors le nom de la citation est ajouté à data/wiki/halo5_commendations_exclude.json.

Aucun accès réseau.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
    )


def _norm_space_lower(s: str) -> str:
    base = " ".join(str(s or "").strip().lower().split())
    return _strip_accents(base)


def _is_marked_for_delete(note: str) -> bool:
    s = _strip_accents(_norm_space_lower(note))
    # couvre: "a supprimer", "à supprimer", "Ha supprimer", etc.
    return bool(re.search(r"\bsupprim", s))


def _load_json(path: Path) -> Any:
    # Certains fichiers out/ peuvent être en cp1252/latin-1 selon l'outil qui les a produits.
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            # Si ça passe en unicode mais pas en JSON, inutile de tester d'autres encodages.
            raise
    # Fallback ultime: remplace les caractères illisibles.
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _load_exclude(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"image_basenames": [], "names": []}
    raw = _load_json(path)
    if isinstance(raw, dict):
        raw.setdefault("image_basenames", [])
        raw.setdefault("names", [])
        return raw
    if isinstance(raw, list):
        # compat: ancienne forme list
        return {"image_basenames": [], "names": list(raw)}
    return {"image_basenames": [], "names": []}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        k = _norm_space_lower(v)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(v.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Construit halo5_commendations_exclude.json depuis les notes de mapping")
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(Path("out") / "commendations_mapping_unmatched_old.json"),
            str(Path("out") / "commendations_mapping_assumed_old.json"),
        ],
        help="Fichiers JSON de mapping à analyser",
    )
    ap.add_argument(
        "--exclude",
        default=str(Path("data") / "wiki" / "halo5_commendations_exclude.json"),
        help="Blacklist JSON à mettre à jour",
    )
    ap.add_argument("--dry-run", action="store_true", help="N'écrit rien, affiche juste un résumé")
    args = ap.parse_args()

    input_paths = [Path(p) for p in (args.inputs or [])]
    input_paths = [p for p in input_paths if p.exists()]
    if not input_paths:
        print("Aucun fichier input trouvé.")
        return 2

    to_exclude_names: list[str] = []
    total_items = 0

    for p in input_paths:
        raw = _load_json(p)
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            total_items += 1
            name = str(it.get("name") or "").strip()
            note = it.get("notes")
            if note is None:
                note = it.get("note")
            note_s = str(note or "").strip()
            if not name or not note_s:
                continue
            if _is_marked_for_delete(note_s):
                to_exclude_names.append(name)

    excl_path = Path(args.exclude)
    payload = _load_exclude(excl_path)
    names = payload.get("names")
    if not isinstance(names, list):
        names = []
    names = [str(x) for x in names if isinstance(x, str)]

    before = len(names)
    names.extend(to_exclude_names)
    names = _dedupe_keep_order(names)
    after = len(names)

    payload["names"] = names
    payload.setdefault("image_basenames", [])

    added = after - before
    print(f"Inputs: {len(input_paths)} fichier(s), {total_items} item(s) inspecté(s)")
    print(f"Marqués à supprimer: {len(_dedupe_keep_order(to_exclude_names))}")
    print(f"Blacklist: +{added} noms (total {after})")

    if args.dry_run:
        return 0

    excl_path.parent.mkdir(parents=True, exist_ok=True)
    excl_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: écrit {excl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
