"""Lanceur LevelUp pour OpenSpartan Graph.

Architecture simplifiée centrée sur la DB fusionnée multi-joueurs.

Usage
-----
Mode interactif (recommandé):
  python openspartan_launcher.py

Commandes CLI:
  python openspartan_launcher.py run              # Dashboard seul
  python openspartan_launcher.py sync             # Sync tous les joueurs + rebuild DB fusionnée
  python openspartan_launcher.py sync --run       # Sync + lance le dashboard
  python openspartan_launcher.py merge            # Fusionne les DBs sources

Configuration:
  - La DB fusionnée par défaut: data/halo_unified.db
  - Les DBs sources: data/spnkr_gt_*.db
  - Variable d'environnement: OPENSPARTAN_DB_PATH
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sqlite3
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_UNIFIED_DB = DEFAULT_DATA_DIR / "halo_unified.db"
DEFAULT_STREAMLIT_APP = REPO_ROOT / "streamlit_app.py"
DEFAULT_IMPORTER = REPO_ROOT / "scripts" / "spnkr_import_db.py"
DEFAULT_MERGER = REPO_ROOT / "scripts" / "merge_databases.py"
DEFAULT_FILM_ROSTER_REFETCH = REPO_ROOT / "scripts" / "refetch_film_roster.py"


# =============================================================================
# Gestion propre du Ctrl+C
# =============================================================================

_shutdown_event = threading.Event()
_active_process: subprocess.Popen | None = None
_shutdown_lock = threading.Lock()
_ctrl_c_count = 0


def _subprocess_creation_flags() -> int:
    """Retourne les flags pour le sous-processus.
    
    Note: On n'utilise PAS CREATE_NEW_PROCESS_GROUP pour que Ctrl+C
    soit propagé au processus enfant.
    """
    return 0


def _kill_active_process() -> None:
    """Termine le processus enfant actif."""
    proc = _active_process
    if proc is None:
        return
    
    # Sur Windows, utiliser taskkill pour tuer l'arbre de processus
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
    
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _signal_handler(signum: int, frame) -> None:
    """Handler pour Ctrl+C."""
    global _ctrl_c_count
    
    with _shutdown_lock:
        _ctrl_c_count += 1
        count = _ctrl_c_count
        
        if count == 1:
            _shutdown_event.set()
            print("\n⏹ Arrêt en cours (Ctrl+C à nouveau pour forcer)...", flush=True)
            _kill_active_process()
        elif count >= 2:
            print("\n⚠ Arrêt forcé.", flush=True)
            _kill_active_process()
            os._exit(1)


def _install_signal_handler() -> None:
    """Installe le handler de signal."""
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)


def _check_shutdown() -> bool:
    """Vérifie si un arrêt a été demandé."""
    return _shutdown_event.is_set()


# =============================================================================
# Helpers Python / venv
# =============================================================================

def _preferred_python_executable() -> Path | None:
    """Trouve le python du venv local."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
        REPO_ROOT / ".venv" / "bin" / "python",  # Linux/macOS
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _maybe_reexec_into_venv(argv: list[str]) -> None:
    """Re-exécute dans le venv si nécessaire."""
    if os.environ.get("OPENSPARTAN_LAUNCHER_NO_REEXEC"):
        return

    preferred = _preferred_python_executable()
    if preferred is None:
        return

    try:
        current = Path(sys.executable).resolve()
        preferred_r = preferred.resolve()
    except Exception:
        return

    if current == preferred_r:
        return

    os.environ["OPENSPARTAN_LAUNCHER_NO_REEXEC"] = "1"
    os.execv(str(preferred_r), [str(preferred_r), str(Path(__file__).resolve()), *argv])


def _require_module(name: str, *, install_hint: str) -> None:
    """Vérifie qu'un module est disponible."""
    try:
        __import__(name)
    except Exception as e:
        print(f"Dépendance manquante: {name}")
        print("Détail:", e)
        print("Installe-la puis relance:")
        print(f"  {install_hint}")
        raise SystemExit(2)


def _pick_free_port() -> int:
    """Trouve un port libre."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# =============================================================================
# Helpers DB
# =============================================================================

def _safe_filename_component(s: str) -> str:
    """Nettoie une chaîne pour un nom de fichier."""
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_ .")
    return s[:80]


def _is_merged_db(db_path: Path | str) -> bool:
    """Vérifie si la DB est fusionnée (a une table Players)."""
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Players'"
        )
        result = cur.fetchone() is not None
        con.close()
        return result
    except Exception:
        return False


@dataclass
class PlayerInfo:
    """Informations sur un joueur dans la DB fusionnée."""
    xuid: str
    gamertag: str | None
    label: str | None
    source_db: str | None
    total_matches: int


def _list_players_from_merged_db(db_path: Path | str) -> list[PlayerInfo]:
    """Liste les joueurs d'une DB fusionnée."""
    players = []
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute("""
            SELECT xuid, gamertag, label, source_db, total_matches
            FROM Players
            ORDER BY total_matches DESC
        """)
        for row in cur.fetchall():
            players.append(PlayerInfo(
                xuid=row[0],
                gamertag=row[1],
                label=row[2],
                source_db=row[3],
                total_matches=row[4] or 0,
            ))
        con.close()
    except Exception:
        pass
    return players


def _iter_source_dbs(data_dir: Path) -> list[Path]:
    """Liste les DBs sources (spnkr_gt_*.db)."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("spnkr_gt_*.db"))


def _infer_player_from_db_filename(db_path: Path) -> str | None:
    """Déduit le gamertag depuis le nom de fichier."""
    base = db_path.stem
    if base.startswith("spnkr_gt_"):
        return base[len("spnkr_gt_"):]
    if base.startswith("spnkr_xuid_"):
        return base[len("spnkr_xuid_"):]
    return None


def _default_spnkr_db_path_for_player(player: str) -> Path:
    """Construit le chemin DB pour un joueur."""
    tag = f"xuid_{player}" if str(player).strip().isdigit() else f"gt_{player}"
    safe = _safe_filename_component(tag)
    name = f"spnkr_{safe}.db" if safe else "spnkr.db"
    return DEFAULT_DATA_DIR / name


def _count_matches_in_db(db_path: Path) -> int:
    """Compte les matchs dans une DB."""
    if not db_path.exists():
        return 0
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM MatchStats")
        row = cur.fetchone()
        con.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _get_db_path_from_env_or_default() -> Path:
    """Retourne la DB à utiliser (env ou défaut)."""
    env = os.environ.get("OPENSPARTAN_DB_PATH") or os.environ.get("OPENSPARTAN_DB")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()
    if DEFAULT_UNIFIED_DB.exists():
        return DEFAULT_UNIFIED_DB
    return DEFAULT_UNIFIED_DB


def _display_path(p: Path) -> str:
    """Affiche un chemin relatif au repo."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


# =============================================================================
# Import SPNKr
# =============================================================================

@dataclass(frozen=True)
class RefreshOptions:
    """Options pour un refresh SPNKr."""
    player: str
    out_db: Path
    match_type: str = "matchmaking"
    max_matches: int = 100
    rps: int = 2
    no_assets: bool = False
    no_skill: bool = False
    with_highlight_events: bool = True
    with_aliases: bool = True
    delta: bool = True


def _run_spnkr_import(opts: RefreshOptions) -> int:
    """Exécute l'import SPNKr (safe tmp + replace)."""
    
    if not DEFAULT_IMPORTER.exists():
        raise SystemExit(f"Importer introuvable: {DEFAULT_IMPORTER}")
    
    opts.out_db.parent.mkdir(parents=True, exist_ok=True)

    tmp_db = opts.out_db.with_suffix(f"{opts.out_db.suffix}.tmp.{int(time.time())}.{os.getpid()}")

    # Copie la DB existante vers TMP
    try:
        if opts.out_db.exists():
            shutil.copy2(opts.out_db, tmp_db)
    except Exception:
        pass

    cmd = [
        sys.executable,
        str(DEFAULT_IMPORTER),
        "--out-db", str(tmp_db),
        "--player", str(opts.player),
        "--match-type", str(opts.match_type),
        "--max-matches", str(int(opts.max_matches)),
        "--requests-per-second", str(int(opts.rps)),
        "--resume",
    ]
    if opts.no_assets:
        cmd.append("--no-assets")
    if opts.no_skill:
        cmd.append("--no-skill")
    if not opts.with_highlight_events:
        cmd.append("--no-highlight-events")
    if not opts.with_aliases:
        cmd.append("--no-aliases")
    if opts.delta:
        cmd.append("--delta")

    print(f"  → Import {opts.player}...")

    global _active_process
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), creationflags=_subprocess_creation_flags())
    _active_process = proc
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass
        return 0
    finally:
        _active_process = None
    
    if proc.returncode != 0:
        if _shutdown_event.is_set():
            try:
                if tmp_db.exists():
                    tmp_db.unlink()
            except Exception:
                pass
            return 0
        print(f"  ⚠ Échec import (code={proc.returncode})")
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass
        return int(proc.returncode)

    # Validation et remplacement
    try:
        if not tmp_db.exists() or tmp_db.stat().st_size <= 0:
            return 0
        os.replace(tmp_db, opts.out_db)
    except Exception as e:
        print(f"  ⚠ Remplacement échoué: {e}")
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass
        return 2

    return 0


def _repair_aliases_for_recent_matches(db_path: Path, player: str, count: int = 20) -> None:
    """Répare les aliases sur les matchs récents."""
    if not DEFAULT_FILM_ROSTER_REFETCH.exists():
        return
    
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute("""
            SELECT json_extract(ResponseBody,'$.MatchId') as MatchId
            FROM MatchStats
            WHERE json_extract(ResponseBody,'$.MatchId') IS NOT NULL
            ORDER BY json_extract(ResponseBody,'$.MatchInfo.StartTime') DESC
            LIMIT ?
        """, (count,))
        match_ids = [row[0] for row in cur.fetchall() if row[0]]
        con.close()
    except Exception:
        return
    
    if not match_ids:
        return
    
    print(f"  → Réparation aliases ({len(match_ids)} matchs)...")
    
    for mid in match_ids:
        if _check_shutdown():
            return
        cmd = [
            sys.executable, str(DEFAULT_FILM_ROSTER_REFETCH),
            "--db", str(db_path),
            "--write-aliases",
            "--patch-highlight-events",
            "--match-id", str(mid),
        ]
        try:
            subprocess.run(cmd, cwd=str(REPO_ROOT), creationflags=_subprocess_creation_flags(),
                          capture_output=True)
        except Exception:
            pass


def _fetch_profile_assets(player: str) -> None:
    """Récupère les assets profil du joueur."""
    try:
        from src.ui.profile_api import (
            fetch_appearance_via_spnkr,
            fetch_xuid_via_spnkr,
            save_cached_appearance,
            save_cached_xuid,
        )
    except ImportError:
        return
    
    print(f"  → Fetch assets profil...")
    
    player_str = str(player).strip()
    xuid = None
    
    if player_str.isdigit():
        xuid = player_str
    else:
        try:
            xuid, _ = fetch_xuid_via_spnkr(gamertag=player_str)
            if xuid:
                save_cached_xuid(player_str, xuid)
        except Exception:
            pass
    
    if not xuid:
        return
    
    try:
        appearance = fetch_appearance_via_spnkr(xuid=xuid)
        if appearance:
            save_cached_appearance(xuid, appearance)
    except Exception:
        pass


# =============================================================================
# Commandes principales
# =============================================================================

def _launch_streamlit(*, db_path: Path | None, port: int | None, no_browser: bool) -> int:
    """Lance le dashboard Streamlit."""
    if not DEFAULT_STREAMLIT_APP.exists():
        raise SystemExit(f"Introuvable: {DEFAULT_STREAMLIT_APP}")

    _require_module("streamlit", install_hint="./.venv/Scripts/python -m pip install -r requirements.txt")

    if db_path is not None:
        os.environ["OPENSPARTAN_DB_PATH"] = str(db_path)

    chosen_port = int(port) if port else _pick_free_port()
    url = f"http://localhost:{chosen_port}"

    cmd = [
        sys.executable,
        "-m", "streamlit", "run", str(DEFAULT_STREAMLIT_APP),
        "--server.address", "localhost",
        "--server.port", str(chosen_port),
        "--server.headless", "true",
    ]

    print("\n🚀 Lancement du dashboard…")
    print(f"   URL: {url}")
    if db_path:
        print(f"   DB: {_display_path(db_path)}")
    
    global _active_process
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), creationflags=_subprocess_creation_flags())
    _active_process = proc

    if not no_browser:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        return 0
    finally:
        _active_process = None
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _cmd_run(args: argparse.Namespace) -> int:
    """Commande: lance le dashboard."""
    db = Path(args.db).expanduser().resolve() if args.db else _get_db_path_from_env_or_default()
    
    if not db.exists():
        print(f"❌ DB introuvable: {db}")
        print("\n   Tu dois d'abord synchroniser les données:")
        print("   python openspartan_launcher.py sync")
        return 2
    
    # Afficher les infos si DB fusionnée
    if _is_merged_db(db):
        players = _list_players_from_merged_db(db)
        print(f"\n📊 DB fusionnée: {len(players)} joueur(s)")
        for p in players:
            name = p.label or p.gamertag or p.xuid[:15]
            print(f"   - {name}: {p.total_matches} matchs")
    
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _cmd_sync(args: argparse.Namespace) -> int:
    """Commande: sync tous les joueurs + rebuild DB fusionnée."""
    
    # Trouver les DBs sources
    source_dbs = _iter_source_dbs(DEFAULT_DATA_DIR)
    
    if not source_dbs:
        print("❌ Aucune DB source trouvée dans data/spnkr_gt_*.db")
        print("\n   Tu dois d'abord créer une DB avec le script d'import:")
        print("   python scripts/spnkr_import_db.py --player <gamertag> --out-db data/spnkr_gt_<gamertag>.db")
        return 2
    
    print("=" * 60)
    print("🔄 SYNCHRONISATION")
    print("=" * 60)
    print(f"\n   {len(source_dbs)} DB(s) source(s) détectée(s):")
    for db in source_dbs:
        player = _infer_player_from_db_filename(db)
        count = _count_matches_in_db(db)
        print(f"   - {db.name} ({player}): {count} matchs")
    
    print("\n📥 Étape 1/3: Refresh des DBs sources...")
    
    failures = 0
    for db in source_dbs:
        if _check_shutdown():
            return 0
        
        player = _infer_player_from_db_filename(db)
        if not player:
            continue
        
        print(f"\n[{player}]")
        
        opts = RefreshOptions(
            player=player,
            out_db=db,
            match_type="matchmaking",
            max_matches=int(getattr(args, "max_matches", 100)),
            rps=int(getattr(args, "rps", 2)),
            delta=not getattr(args, "full", False),
        )
        
        matches_before = _count_matches_in_db(db)
        rc = _run_spnkr_import(opts)
        
        if _check_shutdown():
            return 0
        
        if rc != 0:
            failures += 1
            continue
        
        matches_after = _count_matches_in_db(db)
        new_matches = matches_after - matches_before
        
        if new_matches > 0:
            print(f"  ✓ {new_matches} nouveau(x) match(s)")
        else:
            print(f"  ✓ À jour")
        
        # Réparer les aliases sur les matchs récents
        _repair_aliases_for_recent_matches(db, player, count=min(new_matches + 1, 30))
        
        # Fetch assets profil
        _fetch_profile_assets(player)
    
    if _check_shutdown():
        return 0
    
    print("\n📦 Étape 2/3: Fusion des DBs...")
    
    # Appeler le script de merge
    rc = _run_merge(source_dbs)
    if rc != 0:
        print("⚠ La fusion a échoué")
        return rc
    
    print("\n📊 Étape 3/3: Calcul des scores de performance...")
    
    # Calculer les scores historiques
    _run_compute_performance()
    
    print("\n" + "=" * 60)
    print("✅ SYNCHRONISATION TERMINÉE")
    print("=" * 60)
    
    unified_db = DEFAULT_UNIFIED_DB
    if unified_db.exists():
        players = _list_players_from_merged_db(unified_db)
        total_matches = sum(p.total_matches for p in players)
        print(f"\n   DB: {_display_path(unified_db)}")
        print(f"   Joueurs: {len(players)}")
        print(f"   Matchs: {total_matches}")
    
    # Lancer le dashboard si demandé
    if getattr(args, "run", False):
        return _launch_streamlit(db_path=unified_db, port=None, no_browser=False)
    
    return 0


def _run_merge(source_dbs: list[Path]) -> int:
    """Exécute la fusion des DBs."""
    if not DEFAULT_MERGER.exists():
        print(f"❌ Script de fusion introuvable: {DEFAULT_MERGER}")
        return 2
    
    cmd = [
        sys.executable,
        str(DEFAULT_MERGER),
        str(DEFAULT_UNIFIED_DB),
    ] + [str(db) for db in source_dbs]
    
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), creationflags=_subprocess_creation_flags())
        return proc.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"⚠ Erreur fusion: {e}")
        return 2


def _run_compute_performance() -> None:
    """Calcule les scores de performance historiques."""
    script = REPO_ROOT / "scripts" / "compute_historical_performance.py"
    if not script.exists():
        return
    
    cmd = [sys.executable, str(script), str(DEFAULT_UNIFIED_DB)]
    
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), creationflags=_subprocess_creation_flags(),
                      capture_output=True)
    except Exception:
        pass


def _cmd_merge(args: argparse.Namespace) -> int:
    """Commande: fusionne les DBs sources."""
    source_dbs = _iter_source_dbs(DEFAULT_DATA_DIR)
    
    if not source_dbs:
        print("❌ Aucune DB source trouvée dans data/spnkr_gt_*.db")
        return 2
    
    print("📦 Fusion des DBs...")
    for db in source_dbs:
        print(f"   - {db.name}")
    print(f"   → {_display_path(DEFAULT_UNIFIED_DB)}")
    
    rc = _run_merge(source_dbs)
    
    if rc == 0:
        print("\n✅ Fusion terminée")
        _run_compute_performance()
    
    return rc


# =============================================================================
# Mode interactif
# =============================================================================

def _interactive() -> int:
    """Menu interactif simplifié."""
    print("=" * 60)
    print("        LevelUp - Dashboard Halo Infinite")
    print("=" * 60)
    
    unified_db = DEFAULT_UNIFIED_DB
    source_dbs = _iter_source_dbs(DEFAULT_DATA_DIR)
    
    # Afficher l'état actuel
    print("\n📊 État actuel:")
    
    if unified_db.exists() and _is_merged_db(unified_db):
        players = _list_players_from_merged_db(unified_db)
        total_matches = sum(p.total_matches for p in players)
        print(f"   DB: {_display_path(unified_db)}")
        print(f"   Joueurs: {len(players)}")
        for p in players:
            name = p.label or p.gamertag or p.xuid[:15]
            print(f"      - {name}: {p.total_matches} matchs")
        print(f"   Total: {total_matches} matchs")
    elif source_dbs:
        print(f"   {len(source_dbs)} DB(s) source(s) détectée(s)")
        for db in source_dbs:
            player = _infer_player_from_db_filename(db)
            count = _count_matches_in_db(db)
            print(f"      - {player}: {count} matchs")
        print("   ⚠ Pas de DB fusionnée (lance 'sync' pour créer)")
    else:
        print("   ❌ Aucune DB trouvée")
        print("   → Tu dois d'abord importer des données")
    
    print("\n" + "-" * 60)
    print("Choisis une action:\n")
    print("  1) 🚀 Dashboard                       [recommandé]")
    print("     Lance le dashboard directement")
    print()
    print("  2) 🔄 Sync + Dashboard")
    print("     Synchronise les nouveaux matchs puis lance le dashboard")
    print()
    print("  3) 🔄 Sync seul")
    print("     Synchronise les données sans lancer le dashboard")
    print()
    print("  4) 📦 Fusion seule")
    print("     Refait la fusion des DBs sans synchroniser")
    print()
    print("  Q) Quitter")
    print()
    
    choice = input("Ton choix (1/2/3/4/Q): ").strip().lower()
    
    if choice in {"q", "quit", "exit"}:
        return 0
    
    if choice == "1":
        if not unified_db.exists():
            print("\n⚠ La DB fusionnée n'existe pas encore.")
            print("  Lance d'abord une synchronisation (choix 2 ou 3)")
            return 2
        return _launch_streamlit(db_path=unified_db, port=None, no_browser=False)
    
    if choice == "2":
        args = argparse.Namespace(max_matches=100, rps=2, full=False, run=True)
        return _cmd_sync(args)
    
    if choice == "3":
        args = argparse.Namespace(max_matches=100, rps=2, full=False, run=False)
        return _cmd_sync(args)
    
    if choice == "4":
        return _cmd_merge(argparse.Namespace())
    
    print("Choix invalide.")
    return 2


# =============================================================================
# Parser CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI."""
    ap = argparse.ArgumentParser(
        prog="openspartan",
        description="Lanceur LevelUp - Dashboard Halo Infinite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python openspartan_launcher.py           # Mode interactif
  python openspartan_launcher.py run       # Dashboard seul
  python openspartan_launcher.py sync      # Sync tous les joueurs
  python openspartan_launcher.py sync --run  # Sync + dashboard
""",
    )

    sub = ap.add_subparsers(dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Lance le dashboard")
    p_run.add_argument("--db", default=None, help="Chemin DB (défaut: data/halo_unified.db)")
    p_run.add_argument("--port", type=int, default=None, help="Port (sinon auto)")
    p_run.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur")
    p_run.set_defaults(func=_cmd_run)

    # sync
    p_sync = sub.add_parser("sync", help="Synchronise les données de tous les joueurs")
    p_sync.add_argument("--run", action="store_true", help="Lance le dashboard après la sync")
    p_sync.add_argument("--full", action="store_true", help="Sync complète (pas de delta)")
    p_sync.add_argument("--max-matches", type=int, default=100, help="Max matchs par joueur (défaut: 100)")
    p_sync.add_argument("--rps", type=int, default=2, help="Requêtes/sec (défaut: 2)")
    p_sync.set_defaults(func=_cmd_sync)

    # merge
    p_merge = sub.add_parser("merge", help="Fusionne les DBs sources")
    p_merge.set_defaults(func=_cmd_merge)

    return ap


# =============================================================================
# Point d'entrée
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal."""
    _install_signal_handler()
    
    argv = list(sys.argv[1:] if argv is None else argv)
    _maybe_reexec_into_venv(argv)

    try:
        if not argv:
            return _interactive()

        ap = _build_parser()
        args = ap.parse_args(argv)

        if not getattr(args, "cmd", None):
            ap.print_help()
            return 2

        return int(args.func(args))
    
    except KeyboardInterrupt:
        if not _shutdown_event.is_set():
            print("\n⏹ Arrêt en cours...", flush=True)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
