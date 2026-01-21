"""Lanceur unique (Python) pour OpenSpartan Graph.

Objectif
- Remplacer les .bat par un script Python unique.
- Proposer des modes de lancement/refresh clairs via CLI (avec --help),
  et un mode interactif simple (max 2 questions) si lancé sans arguments.

Exemples
- Mode interactif:
  python openspartan_launcher.py

- Lancer le dashboard:
  python openspartan_launcher.py run

- Lancer le dashboard en forçant une DB:
  python openspartan_launcher.py run --db data/spnkr_gt_JGtm.db

- Refresh SPNKr (safe tmp + replace) pour un joueur:
  python openspartan_launcher.py refresh --player JGtm --out-db data/spnkr_gt_JGtm.db

- Refresh toutes les DB data/spnkr*.db:
  python openspartan_launcher.py refresh-all --max-matches 200 --match-type matchmaking --rps 5 --with-highlight-events
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_STREAMLIT_APP = REPO_ROOT / "streamlit_app.py"
DEFAULT_IMPORTER = REPO_ROOT / "scripts" / "spnkr_import_db.py"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _safe_filename_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_ .")
    return s[:80]


def _default_spnkr_db_path_for_player(player: str) -> Path:
    tag = f"xuid_{player}" if str(player).strip().isdigit() else f"gt_{player}"
    safe = _safe_filename_component(tag)
    name = f"spnkr_{safe}.db" if safe else "spnkr.db"
    return DEFAULT_DATA_DIR / name


def _ensure_importer_exists() -> None:
    if not DEFAULT_IMPORTER.exists():
        raise SystemExit(f"Importer introuvable: {DEFAULT_IMPORTER}")


@dataclass(frozen=True)
class RefreshOptions:
    player: str
    out_db: Path
    match_type: str
    max_matches: int
    rps: int
    no_assets: bool
    no_skill: bool
    with_highlight_events: bool


def _run_spnkr_import(opts: RefreshOptions) -> int:
    """Exécute l'import SPNKr en mode safe: tmp DB puis replace si OK."""

    _ensure_importer_exists()
    opts.out_db.parent.mkdir(parents=True, exist_ok=True)

    tmp_db = opts.out_db.with_suffix(f"{opts.out_db.suffix}.tmp.{int(time.time())}.{os.getpid()}")

    # Copie la DB existante vers TMP pour éviter toute corruption si import interrompu.
    try:
        if opts.out_db.exists():
            shutil.copy2(opts.out_db, tmp_db)
    except Exception:
        pass

    cmd = [
        sys.executable,
        str(DEFAULT_IMPORTER),
        "--out-db",
        str(tmp_db),
        "--player",
        str(opts.player),
        "--match-type",
        str(opts.match_type),
        "--max-matches",
        str(int(opts.max_matches)),
        "--requests-per-second",
        str(int(opts.rps)),
        "--resume",
    ]
    if opts.no_assets:
        cmd.append("--no-assets")
    if opts.no_skill:
        cmd.append("--no-skill")
    if opts.with_highlight_events:
        cmd.append("--with-highlight-events")

    print("[SPNKr] Import…")
    print("- player:", opts.player)
    print("- out_db:", opts.out_db)
    print("- match_type:", opts.match_type)
    print("- max_matches:", opts.max_matches)
    print("- rps:", opts.rps)
    if opts.with_highlight_events:
        print("- highlight_events: ON")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"[SPNKr] Échec import (code={proc.returncode}). DB originale conservée.")
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass
        return int(proc.returncode)

    # Validation minimale: fichier tmp non vide.
    try:
        if not tmp_db.exists() or tmp_db.stat().st_size <= 0:
            print("[SPNKr] Import OK mais DB tmp vide. DB originale conservée.")
            return 0
        os.replace(tmp_db, opts.out_db)
    except Exception as e:
        print("[SPNKr] Import OK mais remplacement DB a échoué:", e)
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass
        return 2

    print("[SPNKr] OK")
    return 0


def _launch_streamlit(*, db_path: Path | None, port: int | None, no_browser: bool) -> int:
    if not DEFAULT_STREAMLIT_APP.exists():
        raise SystemExit(f"Introuvable: {DEFAULT_STREAMLIT_APP}")

    if db_path is not None:
        os.environ["OPENSPARTAN_DB_PATH"] = str(db_path)

    chosen_port = int(port) if port else _pick_free_port()
    url = f"http://localhost:{chosen_port}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DEFAULT_STREAMLIT_APP),
        "--server.address",
        "localhost",
        "--server.port",
        str(chosen_port),
        "--server.headless",
        "true",
    ]

    print("Lancement du dashboard…")
    print("URL:", url)
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    if not no_browser:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        return 130


def _iter_spnkr_dbs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("spnkr*.db"))


def _infer_player_from_db_filename(db_path: Path) -> str | None:
    base = db_path.stem
    if base.startswith("spnkr_gt_"):
        return base[len("spnkr_gt_") :]
    if base.startswith("spnkr_xuid_"):
        return base[len("spnkr_xuid_") :]
    if base.startswith("spnkr_"):
        return base[len("spnkr_") :]
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    db = Path(args.db).expanduser().resolve() if args.db else None
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _cmd_refresh(args: argparse.Namespace) -> int:
    player = args.player or os.environ.get("SPNKR_PLAYER")
    if not player:
        raise SystemExit("Fournis --player (ou SPNKR_PLAYER)")

    out_db = Path(args.out_db).expanduser().resolve() if args.out_db else _default_spnkr_db_path_for_player(player)
    opts = RefreshOptions(
        player=str(player),
        out_db=out_db,
        match_type=str(args.match_type),
        max_matches=int(args.max_matches),
        rps=int(args.rps),
        no_assets=bool(args.no_assets),
        no_skill=bool(args.no_skill),
        with_highlight_events=bool(args.with_highlight_events),
    )
    return _run_spnkr_import(opts)


def _cmd_refresh_all(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else DEFAULT_DATA_DIR
    dbs = _iter_spnkr_dbs(data_dir)
    if not dbs:
        print(f"[INFO] Aucune DB trouvée: {data_dir / 'spnkr*.db'}")
        return 0

    failures = 0
    for db in dbs:
        player = _infer_player_from_db_filename(db)
        if not player:
            print(f"[SKIP] {db.name} (impossible de déduire --player depuis le nom)")
            continue

        opts = RefreshOptions(
            player=str(player),
            out_db=db,
            match_type=str(args.match_type),
            max_matches=int(args.max_matches),
            rps=int(args.rps),
            no_assets=bool(args.no_assets),
            no_skill=bool(args.no_skill),
            with_highlight_events=bool(args.with_highlight_events),
        )
        rc = _run_spnkr_import(opts)
        if rc != 0:
            failures += 1

    if failures:
        print(f"Terminé avec {failures} échec(s).")
        return 2
    print("Terminé.")
    return 0


def _cmd_run_with_refresh(args: argparse.Namespace) -> int:
    rc = _cmd_refresh(args)
    if rc not in (0, 2):
        # 2 = erreur de paramétrage, on ne lance pas.
        print("[WARN] Refresh en échec, lancement du dashboard quand même…")
    db = Path(args.out_db).expanduser().resolve() if args.out_db else None
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _interactive(argv0: str) -> int:
    print("OpenSpartan Graphs — Lanceur")
    print("(Astuce: ajoute --help pour les options CLI)")
    print("\nChoisis un mode:")
    print("  1) Lancer le dashboard")
    print("  2) Lancer le dashboard + refresh SPNKr")
    print("  3) Refresh SPNKr (une DB)")
    print("  4) Refresh SPNKr (toutes les DB data/spnkr*.db)")
    print("  Q) Quitter")

    choice = input("\nTon choix (1/2/3/4/Q): ").strip().lower()
    if choice in {"q", "quit", "exit"}:
        return 0

    # Question 2 max: demander le joueur uniquement si nécessaire.
    player: str | None = None
    if choice in {"2", "3"}:
        player = (input("Joueur SPNKr (gamertag ou XUID) [SPNKR_PLAYER]: ").strip() or os.environ.get("SPNKR_PLAYER"))
        if not player:
            print("Aucun joueur fourni.")
            return 2

    if choice == "1":
        return _launch_streamlit(db_path=None, port=None, no_browser=False)

    if choice == "2":
        args = argparse.Namespace(
            player=player,
            out_db=None,
            match_type="matchmaking",
            max_matches=500,
            rps=2,
            no_assets=False,
            no_skill=False,
            with_highlight_events=True,
            port=None,
            no_browser=False,
            db=None,
        )
        return _cmd_run_with_refresh(args)

    if choice == "3":
        args = argparse.Namespace(
            player=player,
            out_db=None,
            match_type="matchmaking",
            max_matches=500,
            rps=2,
            no_assets=False,
            no_skill=False,
            with_highlight_events=True,
        )
        return _cmd_refresh(args)

    if choice == "4":
        args = argparse.Namespace(
            data_dir=str(DEFAULT_DATA_DIR),
            match_type="matchmaking",
            max_matches=200,
            rps=5,
            no_assets=False,
            no_skill=False,
            with_highlight_events=True,
        )
        return _cmd_refresh_all(args)

    print("Choix invalide.")
    print(f"Usage CLI: {argv0} --help")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="openspartan",
        description=(
            "Lanceur OpenSpartan Graphs (dashboard Streamlit + refresh SPNKr).\n"
            "- Mode interactif si lancé sans sous-commande (max 2 questions).\n"
            "- Mode CLI avec sous-commandes et --help."
        ),
    )

    sub = ap.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Lance le dashboard Streamlit")
    p_run.add_argument("--db", default=None, help="Chemin DB à utiliser (OPENSPARTAN_DB_PATH)")
    p_run.add_argument("--port", type=int, default=None, help="Port (sinon auto)")
    p_run.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur")
    p_run.set_defaults(func=_cmd_run)

    p_ref = sub.add_parser("refresh", help="Refresh SPNKr (safe tmp + replace) pour un joueur")
    p_ref.add_argument("--player", default=None, help="Gamertag ou XUID (sinon: SPNKR_PLAYER)")
    p_ref.add_argument("--out-db", default=None, help="DB cible (sinon: data/spnkr_<player>.db)")
    p_ref.add_argument(
        "--match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs (défaut: matchmaking)",
    )
    p_ref.add_argument("--max-matches", type=int, default=500, help="Max matchs (défaut: 500)")
    p_ref.add_argument("--rps", type=int, default=2, help="Requests/sec (défaut: 2)")
    p_ref.add_argument("--no-assets", action="store_true", help="Désactive l'import des assets")
    p_ref.add_argument("--no-skill", action="store_true", help="Désactive l'import du skill")
    p_ref.add_argument(
        "--with-highlight-events",
        action="store_true",
        help="Inclut highlight events (film) (plus lent)",
    )
    p_ref.set_defaults(func=_cmd_refresh)

    p_ra = sub.add_parser("refresh-all", help="Refresh toutes les DB data/spnkr*.db")
    p_ra.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dossier contenant les DB (défaut: data/)")
    p_ra.add_argument(
        "--match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs (défaut: matchmaking)",
    )
    p_ra.add_argument("--max-matches", type=int, default=200, help="Max matchs par DB (défaut: 200)")
    p_ra.add_argument("--rps", type=int, default=5, help="Requests/sec (défaut: 5)")
    p_ra.add_argument("--no-assets", action="store_true", help="Désactive l'import des assets")
    p_ra.add_argument("--no-skill", action="store_true", help="Désactive l'import du skill")
    p_ra.add_argument(
        "--with-highlight-events",
        action="store_true",
        help="Inclut highlight events (film) (plus lent)",
    )
    p_ra.set_defaults(func=_cmd_refresh_all)

    p_runref = sub.add_parser("run+refresh", help="Refresh SPNKr puis lance le dashboard")
    p_runref.add_argument("--player", default=None, help="Gamertag ou XUID (sinon: SPNKR_PLAYER)")
    p_runref.add_argument("--out-db", default=None, help="DB cible (sinon: data/spnkr_<player>.db)")
    p_runref.add_argument(
        "--match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs (défaut: matchmaking)",
    )
    p_runref.add_argument("--max-matches", type=int, default=500, help="Max matchs (défaut: 500)")
    p_runref.add_argument("--rps", type=int, default=2, help="Requests/sec (défaut: 2)")
    p_runref.add_argument("--no-assets", action="store_true", help="Désactive l'import des assets")
    p_runref.add_argument("--no-skill", action="store_true", help="Désactive l'import du skill")
    p_runref.add_argument(
        "--with-highlight-events",
        action="store_true",
        help="Inclut highlight events (film) (plus lent)",
    )
    p_runref.add_argument("--port", type=int, default=None, help="Port (sinon auto)")
    p_runref.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur")
    p_runref.set_defaults(func=_cmd_run_with_refresh)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return _interactive(argv0=f"{Path(sys.argv[0]).name}")

    ap = _build_parser()
    args = ap.parse_args(argv)

    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
