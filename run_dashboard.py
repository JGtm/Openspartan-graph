import os
import socket
import subprocess
import sys
import time
import webbrowser
import argparse
import sqlite3


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _is_spnkr_db_empty(db_path: str) -> bool:
    try:
        if not db_path or not os.path.exists(db_path):
            return True
        con = sqlite3.connect(db_path)
        try:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MatchStats'")
            if cur.fetchone() is None:
                return True
            cur.execute("SELECT COUNT(*) FROM MatchStats")
            n = cur.fetchone()[0]
            return int(n) <= 0
        finally:
            con.close()
    except Exception:
        # En cas de DB corrompue/inaccessible, on considère vide et on retente un import.
        return True


def _maybe_refresh_spnkr_db(*, repo_root: str, args: argparse.Namespace) -> int:
    if not args.refresh_spnkr:
        return 0

    player = args.refresh_player or os.environ.get("SPNKR_PLAYER")
    if not player:
        print("[SPNKr] Skip: aucun joueur. Fournis --refresh-player ou SPNKR_PLAYER.")
        return 2

    out_db = args.refresh_out_db
    if not out_db:
        out_db = os.path.join(repo_root, "data", "spnkr.db")

    is_bootstrap = _is_spnkr_db_empty(out_db)
    max_matches = int(args.refresh_bootstrap_max_matches if is_bootstrap else args.refresh_max_matches)
    match_type = str(args.refresh_bootstrap_match_type if is_bootstrap else args.refresh_match_type)

    importer = os.path.join(repo_root, "scripts", "spnkr_import_db.py")
    if not os.path.exists(importer):
        print(f"[SPNKr] Skip: script introuvable: {importer}")
        return 2

    cmd = [
        sys.executable,
        importer,
        "--out-db",
        out_db,
        "--player",
        player,
        "--match-type",
        match_type,
        "--max-matches",
        str(int(max_matches)),
        "--resume",
        "--requests-per-second",
        str(int(args.refresh_rps)),
    ]
    if args.refresh_no_assets:
        cmd.append("--no-assets")
    if args.refresh_no_skill:
        cmd.append("--no-skill")

    if is_bootstrap:
        print("[SPNKr] Bootstrap DB (première construction)…")
    else:
        print("[SPNKr] Refresh DB…")
    print("[SPNKr] out_db:", out_db)
    print("[SPNKr] player:", player)
    print("[SPNKr] match_type:", match_type)
    print("[SPNKr] max_matches:", max_matches)
    try:
        proc = subprocess.run(cmd, cwd=repo_root)
    except Exception as e:
        print("[SPNKr] Erreur au lancement de l'import:", e)
        return 2

    if proc.returncode != 0:
        print(f"[SPNKr] Import en échec (code={proc.returncode}). Le dashboard va quand même se lancer.")
    else:
        print("[SPNKr] OK")
    return int(proc.returncode)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--refresh-spnkr",
        action="store_true",
        help="Rafraîchit la DB SPNKr (data/spnkr.db) avant de lancer Streamlit.",
    )
    ap.add_argument(
        "--refresh-player",
        "--player",
        dest="refresh_player",
        default=None,
        help="Gamertag ou XUID pour SPNKr (sinon: SPNKR_PLAYER env).",
    )
    ap.add_argument("--refresh-out-db", default=None, help="Chemin DB SPNKr (défaut: data/spnkr.db)")
    ap.add_argument(
        "--refresh-match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs pour le refresh normal (défaut: matchmaking)",
    )
    ap.add_argument("--refresh-max-matches", type=int, default=500, help="Nombre max de matchs à importer (défaut: 500)")
    ap.add_argument(
        "--refresh-bootstrap-match-type",
        default="all",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs pour la première construction de DB (défaut: all)",
    )
    ap.add_argument(
        "--refresh-bootstrap-max-matches",
        type=int,
        default=2000,
        help="Nombre max de matchs à importer lors de la première construction (défaut: 2000)",
    )
    ap.add_argument("--refresh-rps", type=int, default=2, help="Rate limit (requests/sec) (défaut: 2)")
    ap.add_argument("--refresh-no-assets", action="store_true", help="Désactive l'import des assets UGC")
    ap.add_argument("--refresh-no-skill", action="store_true", help="Désactive l'import du skill (PlayerMatchStats)")
    args, _unknown = ap.parse_known_args()
    app_path = os.path.join(here, "streamlit_app.py")
    if not os.path.exists(app_path):
        print(f"Erreur: introuvable: {app_path}")
        return 2

    _maybe_refresh_spnkr_db(repo_root=here, args=args)

    port = _pick_free_port()
    url = f"http://localhost:{port}"

    # Lance Streamlit dans ce même environnement Python.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "localhost",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]

    print("Lancement du dashboard…")
    print("URL:", url)

    # Démarre Streamlit et ouvre le navigateur après un court délai.
    proc = subprocess.Popen(cmd, cwd=here)
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
