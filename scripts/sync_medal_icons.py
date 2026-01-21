"""Synchronise les icônes de médailles Halo Infinite en local.

But:
- Copier les PNG du cache OpenSpartan.Workshop (ou un dossier source fourni)
  vers ce projet, dans static/medals/icons.

Usage:
  python scripts/sync_medal_icons.py
  python scripts/sync_medal_icons.py --dry-run
  python scripts/sync_medal_icons.py --source "C:/path/to/medals" --overwrite

Notes:
- Le script ne télécharge rien : il copie uniquement depuis un cache déjà présent.
- Les fichiers attendus sont des PNG nommés "<NameId>.png".
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


_DIGITS_PNG = re.compile(r"^\d+\.png$", re.IGNORECASE)


def _default_source_dir() -> Path:
    override = os.environ.get("OPENSPARTAN_MEDALS_CACHE")
    if override:
        return Path(override)

    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        return Path(localappdata) / "OpenSpartan.Workshop" / "imagecache" / "medals"

    return Path.home() / "AppData" / "Local" / "OpenSpartan.Workshop" / "imagecache" / "medals"


def _repo_root() -> Path:
    # scripts/sync_medal_icons.py -> repo root
    return Path(__file__).resolve().parents[1]


def _default_dest_dir() -> Path:
    return _repo_root() / "static" / "medals" / "icons"


@dataclass
class SyncResult:
    copied: int = 0
    skipped_existing: int = 0
    skipped_non_png: int = 0
    missing_source: bool = False


def sync_medal_icons(
    source_dir: Path,
    dest_dir: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> SyncResult:
    res = SyncResult()

    if not source_dir.exists() or not source_dir.is_dir():
        res.missing_source = True
        return res

    dest_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(source_dir.iterdir()):
        if not p.is_file():
            continue
        if not _DIGITS_PNG.match(p.name):
            res.skipped_non_png += 1
            continue

        out = dest_dir / p.name
        if out.exists() and not overwrite:
            res.skipped_existing += 1
            continue

        if dry_run:
            res.copied += 1
            continue

        shutil.copy2(str(p), str(out))
        res.copied += 1

    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Copie les icônes de médailles OpenSpartan en local.")
    parser.add_argument("--source", type=str, default=None, help="Dossier source (cache medals).")
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Dossier destination (défaut: static/medals/icons).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Écrase les fichiers existants.")
    parser.add_argument("--dry-run", action="store_true", help="N'écrit rien, affiche seulement le bilan.")

    args = parser.parse_args()

    source_dir = Path(args.source).expanduser() if args.source else _default_source_dir()
    dest_dir = Path(args.dest).expanduser() if args.dest else _default_dest_dir()

    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_dir}")

    result = sync_medal_icons(source_dir, dest_dir, overwrite=bool(args.overwrite), dry_run=bool(args.dry_run))

    if result.missing_source:
        print("\nERREUR: dossier source introuvable.")
        print("- Installe/synchronise OpenSpartan.Workshop, ou passe --source.")
        return 2

    print("\nBilan:")
    print(f"- Copiés:             {result.copied}{' (dry-run)' if args.dry_run else ''}")
    print(f"- Déjà présents:      {result.skipped_existing}")
    print(f"- Ignorés (non PNG):  {result.skipped_non_png}")

    if not args.dry_run:
        # Petite indication utile si le dossier est vide
        if result.copied == 0 and result.skipped_existing == 0:
            print("\nAucun fichier PNG de type '<NameId>.png' trouvé.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
