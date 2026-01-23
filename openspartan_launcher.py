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

- Refresh toutes les DB data/spnkr*.db (highlight events + aliases activés par défaut):
  python openspartan_launcher.py refresh-all --max-matches 200 --match-type matchmaking --rps 5

- Refresh rapide (sans highlight events ni aliases):
  python openspartan_launcher.py refresh --player JGtm --no-highlight-events --no-aliases
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sqlite3
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_STREAMLIT_APP = REPO_ROOT / "streamlit_app.py"
DEFAULT_IMPORTER = REPO_ROOT / "scripts" / "spnkr_import_db.py"
DEFAULT_FILM_ROSTER_REFETCH = REPO_ROOT / "scripts" / "refetch_film_roster.py"


def _preferred_python_executable() -> Path | None:
    # Préfère un venv local si présent (évite d'utiliser un python système
    # qui n'a pas les dépendances: streamlit, aiohttp, etc.).
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
        REPO_ROOT / ".venv" / "bin" / "python",  # Linux/macOS
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _maybe_reexec_into_venv(argv: list[str]) -> None:
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
    try:
        __import__(name)
    except Exception as e:
        print(f"Dépendance manquante: {name}")
        print("Détail:", e)
        print("Installe-la dans ton environnement Python actif puis relance.")
        print("Si tu utilises le venv du repo, tu peux faire:")
        print(f"  {install_hint}")
        raise SystemExit(2)


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
    with_highlight_events: bool  # True = activer (défaut), False = désactiver
    with_aliases: bool = True    # True = activer (défaut), False = désactiver
    delta: bool = False          # True = arrêt dès match connu (sync rapide)


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
    # Highlight events: activés par défaut, désactivables via --no-highlight-events
    if not opts.with_highlight_events:
        cmd.append("--no-highlight-events")
    # Aliases: activés par défaut, désactivables via --no-aliases
    if not opts.with_aliases:
        cmd.append("--no-aliases")
    # Mode delta: s'arrête dès qu'un match connu est rencontré
    if opts.delta:
        cmd.append("--delta")

    print("[SPNKr] Import…")
    print("- player:", opts.player)
    print("- out_db:", opts.out_db)
    print("- match_type:", opts.match_type)
    print("- max_matches:", opts.max_matches)
    print("- rps:", opts.rps)
    print("- highlight_events:", "ON" if opts.with_highlight_events else "OFF")
    print("- aliases:", "ON" if opts.with_aliases else "OFF")
    print("- delta:", "ON" if opts.delta else "OFF")

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


def _fetch_profile_assets(*, player: str, xuid: str | None = None) -> int:
    """Récupère les assets profil (emblem, backdrop, nameplate) via SPNKr.
    
    Cette fonction est appelée après un refresh pour mettre à jour le cache
    des assets visuels du joueur.
    """
    print("[Profile] Fetch assets profil...")
    print("- player:", player)
    
    try:
        # Import dynamique pour éviter les erreurs si spnkr n'est pas installé
        from src.ui.profile_api import (
            fetch_appearance_via_spnkr,
            fetch_xuid_via_spnkr,
            save_cached_appearance,
            save_cached_xuid,
        )
    except ImportError as e:
        print(f"[Profile] Module profile_api non disponible: {e}")
        return 0  # Non bloquant
    
    # Résoudre le XUID si on a un gamertag
    resolved_xuid = xuid
    if not resolved_xuid or not str(resolved_xuid).strip().isdigit():
        player_str = str(player).strip()
        if player_str.isdigit():
            resolved_xuid = player_str
        else:
            print(f"[Profile] Résolution XUID pour {player}...")
            try:
                resolved_xuid, canonical_gt = fetch_xuid_via_spnkr(gamertag=player_str)
                if resolved_xuid:
                    save_cached_xuid(player_str, resolved_xuid)
                    print(f"[Profile] XUID résolu: {resolved_xuid} ({canonical_gt})")
            except Exception as e:
                print(f"[Profile] Échec résolution XUID: {e}")
                return 0  # Non bloquant
    
    if not resolved_xuid:
        print("[Profile] Impossible de résoudre le XUID, skip fetch assets.")
        return 0
    
    # Fetch les assets
    try:
        print(f"[Profile] Fetch appearance pour XUID {resolved_xuid}...")
        appearance = fetch_appearance_via_spnkr(xuid=resolved_xuid)
        
        if appearance:
            save_cached_appearance(resolved_xuid, appearance)
            print("[Profile] Assets mis en cache:")
            if appearance.service_tag:
                print(f"  - Service tag: {appearance.service_tag}")
            if appearance.emblem_image_url:
                print(f"  - Emblem: OK")
            if appearance.backdrop_image_url:
                print(f"  - Backdrop: OK")
            if appearance.nameplate_image_url:
                print(f"  - Nameplate: OK")
            if appearance.rank_label:
                print(f"  - Rank: {appearance.rank_label}")
        else:
            print("[Profile] Aucune donnée retournée")
            
        return 0
    except Exception as e:
        print(f"[Profile] Échec fetch assets: {e}")
        return 0  # Non bloquant


def _launch_streamlit(*, db_path: Path | None, port: int | None, no_browser: bool) -> int:
    if not DEFAULT_STREAMLIT_APP.exists():
        raise SystemExit(f"Introuvable: {DEFAULT_STREAMLIT_APP}")

    _require_module("streamlit", install_hint="./.venv/Scripts/python -m pip install -r requirements.txt")

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


def _guess_default_spnkr_db() -> Path | None:
    env = os.environ.get("OPENSPARTAN_DB_PATH") or os.environ.get("OPENSPARTAN_DB")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    dbs = _iter_spnkr_dbs(DEFAULT_DATA_DIR)
    if not dbs:
        return None
    # Choisit la plus récemment modifiée.
    return max(dbs, key=lambda p: p.stat().st_mtime)


def _display_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _prompt_db_choice(*, purpose: str, default_db: Path | None, allow_none: bool = False) -> Path | None:
    candidates: list[Path] = []

    if default_db is not None and default_db.exists():
        candidates.append(default_db.resolve())

    for p in _iter_spnkr_dbs(DEFAULT_DATA_DIR):
        rp = p.resolve()
        if rp not in candidates:
            candidates.append(rp)

    if candidates:
        print(f"\nDBs disponibles ({purpose}):")
        if allow_none:
            print("  0) (auto / aucune DB forcée)")
        for i, p in enumerate(candidates, start=1):
            marker = " (défaut)" if default_db is not None and p == default_db.resolve() else ""
            print(f"  {i}) {_display_path(p)}{marker}")
        print("(Tu peux aussi coller un chemin complet vers une DB .db)")
    else:
        print(f"\nAucune DB détectée pour {purpose}.")
        print("Colle un chemin complet vers une DB .db")

    default_hint = _display_path(default_db) if default_db is not None else ""
    raw = input(f"DB ({purpose}) [défaut: {default_hint}]: ").strip()
    if not raw:
        if allow_none:
            return default_db.resolve() if default_db is not None and default_db.exists() else None
        if default_db is not None and default_db.exists():
            return default_db.resolve()
        return candidates[0] if candidates else None

    if allow_none and raw == "0":
        return None

    if raw.isdigit() and candidates:
        idx = int(raw)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print("Numéro invalide.")
        return None

    # Sinon, l'entrée est interprétée comme un chemin.
    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    if not p.exists():
        print("DB introuvable:", p)
        return None
    return p


def _prompt_player_choice(*, default_player: str | None) -> str | None:
    candidates: list[tuple[str, Path | None]] = []

    dp = (default_player or "").strip()
    if dp:
        candidates.append((dp, None))

    for db in _iter_spnkr_dbs(DEFAULT_DATA_DIR):
        p = _infer_player_from_db_filename(db)
        if not p:
            continue
        if any(p == existing for existing, _ in candidates):
            continue
        candidates.append((p, db))

    if candidates:
        print("\nJoueurs détectés (depuis les DB existantes):")
        for i, (p, db) in enumerate(candidates, start=1):
            origin = f" ({db.name})" if db is not None else ""
            marker = " (défaut)" if dp and p == dp else ""
            print(f"  {i}) {p}{origin}{marker}")

    raw = input("Joueur SPNKr (gamertag ou XUID) [SPNKR_PLAYER]: ").strip()
    if not raw:
        return dp or (candidates[0][0] if candidates else None)

    if raw.isdigit() and candidates:
        idx = int(raw)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1][0]
        print("Numéro invalide.")
        return None

    return raw


def _latest_match_id_from_db(db_path: Path) -> str | None:
    """Retourne le MatchId le plus récent selon MatchStats.MatchInfo.StartTime."""
    try:
        con = sqlite3.connect(str(db_path))
    except Exception:
        return None
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
              json_extract(ResponseBody,'$.MatchId') as MatchId,
              json_extract(ResponseBody,'$.MatchInfo.StartTime') as Start
            FROM MatchStats
            WHERE json_extract(ResponseBody,'$.MatchId') IS NOT NULL
              AND json_extract(ResponseBody,'$.MatchInfo.StartTime') IS NOT NULL
            ORDER BY Start DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        mid = row[0]
        return str(mid).strip() if isinstance(mid, str) and mid.strip() else None
    except Exception:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass


def _latest_match_ids_from_db(db_path: Path, *, limit: int) -> list[str]:
    """Retourne les N MatchId les plus récents selon MatchStats.MatchInfo.StartTime."""
    n = max(1, int(limit))
    try:
        con = sqlite3.connect(str(db_path))
    except Exception:
        return []
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
              json_extract(ResponseBody,'$.MatchId') as MatchId,
              json_extract(ResponseBody,'$.MatchInfo.StartTime') as Start
            FROM MatchStats
            WHERE json_extract(ResponseBody,'$.MatchId') IS NOT NULL
              AND json_extract(ResponseBody,'$.MatchInfo.StartTime') IS NOT NULL
            ORDER BY Start DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall() or []
        out: list[str] = []
        for r in rows:
            mid = r[0] if r else None
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
        return out
    except Exception:
        return []
    finally:
        try:
            con.close()
        except Exception:
            pass


def _resolve_out_db(*, player: str, out_db_arg: str | None) -> Path:
    return Path(out_db_arg).expanduser().resolve() if out_db_arg else _default_spnkr_db_path_for_player(player)


def _cmd_run(args: argparse.Namespace) -> int:
    db = Path(args.db).expanduser().resolve() if args.db else None
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _cmd_refresh(args: argparse.Namespace) -> int:
    player = args.player or os.environ.get("SPNKR_PLAYER")
    if not player:
        raise SystemExit("Fournis --player (ou SPNKR_PLAYER)")

    out_db = _resolve_out_db(player=str(player), out_db_arg=getattr(args, "out_db", None))
    # Rend le chemin visible aux commandes suivantes (run+refresh, run+refresh+aliases, etc.)
    try:
        args.out_db = str(out_db)
    except Exception:
        pass
    opts = RefreshOptions(
        player=str(player),
        out_db=out_db,
        match_type=str(args.match_type),
        max_matches=int(args.max_matches),
        rps=int(args.rps),
        no_assets=bool(args.no_assets),
        no_skill=bool(args.no_skill),
        with_highlight_events=not bool(getattr(args, "no_highlight_events", False)),
        with_aliases=not bool(getattr(args, "no_aliases", False)),
        delta=bool(getattr(args, "delta", False)),
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
            with_highlight_events=not bool(getattr(args, "no_highlight_events", False)),
            with_aliases=not bool(getattr(args, "no_aliases", False)),
            delta=bool(getattr(args, "delta", False)),
        )
        rc = _run_spnkr_import(opts)
        if rc != 0:
            failures += 1

    if failures:
        print(f"Terminé avec {failures} échec(s).")
        return 2
    print("Terminé.")
    return 0


def _cmd_refresh_all_with_aliases(args: argparse.Namespace) -> int:
    """Refresh toutes les DB SPNKr avec highlight events, aliases film et profil."""
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

        print(f"\n{'='*60}")
        print(f"[DB] {db.name} (player={player})")
        print(f"{'='*60}")

        # Créer un namespace avec les bons paramètres pour ce joueur
        sub_args = argparse.Namespace(
            player=str(player),
            out_db=str(db),
            match_type=str(args.match_type),
            max_matches=int(args.max_matches),
            rps=int(args.rps),
            no_assets=bool(getattr(args, "no_assets", False)),
            no_skill=bool(getattr(args, "no_skill", False)),
            no_highlight_events=bool(getattr(args, "no_highlight_events", False)),
            no_aliases=bool(getattr(args, "no_aliases", False)),
            delta=bool(getattr(args, "delta", False)),
            with_highlight_events=not bool(getattr(args, "no_highlight_events", False)),
            aliases_last=int(getattr(args, "aliases_last", 50)),
            patch_highlight_events=bool(getattr(args, "patch_highlight_events", False)),
            include_type2=bool(getattr(args, "include_type2", False)),
            max_type2_chunks=int(getattr(args, "max_type2_chunks", 0)),
            max_total_chunks=getattr(args, "max_total_chunks", None),
            print_limit=int(getattr(args, "print_limit", 10)),
            no_fetch_profile=bool(getattr(args, "no_fetch_profile", False)),
        )

        rc = _cmd_refresh_with_aliases(sub_args)
        if rc != 0:
            failures += 1

    if failures:
        print(f"\nTerminé avec {failures} échec(s).")
        return 2
    print("\nTerminé.")
    return 0


def _cmd_run_with_refresh(args: argparse.Namespace) -> int:
    # Par défaut, run+refresh est utilisé quand on veut un dashboard complet.
    # Highlight events et aliases sont activés par défaut.
    try:
        args.no_highlight_events = False
        args.no_aliases = False
    except Exception:
        pass
    rc = _cmd_refresh(args)
    if rc not in (0, 2):
        # 2 = erreur de paramétrage, on ne lance pas.
        print("[WARN] Refresh en échec, lancement du dashboard quand même…")
    db = Path(args.out_db).expanduser().resolve() if getattr(args, "out_db", None) else None
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _cmd_refresh_with_aliases(args: argparse.Namespace) -> int:
    """Refresh SPNKr + highlight events, puis répare les aliases sur les N derniers matchs."""

    # Force highlight events: indispensable pour certaines données.
    args.with_highlight_events = True

    player = args.player or os.environ.get("SPNKR_PLAYER")
    if not player:
        raise SystemExit("Fournis --player (ou SPNKR_PLAYER)")

    out_db = _resolve_out_db(player=str(player), out_db_arg=getattr(args, "out_db", None))
    try:
        args.out_db = str(out_db)
    except Exception:
        pass

    rc = _cmd_refresh(args)
    if rc == 2:
        return 2

    # Réparation aliases (sur les N derniers matchs)
    aliases_last = int(getattr(args, "aliases_last", 0) or 0)
    if aliases_last <= 0:
        aliases_last = int(getattr(args, "max_matches", 10) or 10)
    aliases_last = max(1, min(aliases_last, 200))

    match_ids = _latest_match_ids_from_db(out_db, limit=aliases_last)
    if not match_ids:
        print("[Aliases] Aucun match trouvé pour réparer les aliases.")
        return 0

    print("[Aliases] Film roster -> xuid_aliases.json")
    print("- db:", out_db)
    print("- matches:", len(match_ids))

    for i, mid in enumerate(match_ids, start=1):
        cmd = [sys.executable, str(DEFAULT_FILM_ROSTER_REFETCH), "--db", str(out_db), "--write-aliases", "--match-id", str(mid)]
        if getattr(args, "patch_highlight_events", False):
            cmd.append("--patch-highlight-events")
        if getattr(args, "include_type2", False):
            cmd.append("--include-type2")
        if getattr(args, "max_type2_chunks", None) is not None:
            cmd += ["--max-type2-chunks", str(int(args.max_type2_chunks))]
        if getattr(args, "max_total_chunks", None) is not None:
            cmd += ["--max-total-chunks", str(int(args.max_total_chunks))]
        if getattr(args, "print_limit", None) is not None:
            cmd += ["--print-limit", str(int(args.print_limit))]

        print(f"[Aliases] {i}/{len(match_ids)} match_id={mid}")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            print(f"[WARN] Repair aliases échoué sur {mid} (code={proc.returncode})")

    # Fetch des assets profil (emblem, backdrop, nameplate)
    if not getattr(args, "no_fetch_profile", False):
        _fetch_profile_assets(player=str(player))

    return 0


def _cmd_run_with_refresh_and_aliases(args: argparse.Namespace) -> int:
    rc = _cmd_refresh_with_aliases(args)
    if rc == 2:
        return 2
    db = Path(args.out_db).expanduser().resolve() if getattr(args, "out_db", None) else None
    return _launch_streamlit(db_path=db, port=args.port, no_browser=args.no_browser)


def _cmd_repair_aliases(args: argparse.Namespace) -> int:
    if not DEFAULT_FILM_ROSTER_REFETCH.exists():
        raise SystemExit(f"Introuvable: {DEFAULT_FILM_ROSTER_REFETCH}")

    _require_module("aiohttp", install_hint="./.venv/Scripts/python -m pip install -r requirements.txt")

    db_path = Path(args.db).expanduser().resolve() if args.db else None
    if db_path is None:
        guessed = _guess_default_spnkr_db()
        if guessed is None:
            raise SystemExit("Fournis --db (aucune DB par défaut détectée).")
        db_path = guessed

    # Par défaut (sans options), on fait comme l'interactif: --latest.
    if not getattr(args, "all_matches", False) and not getattr(args, "latest", False) and not getattr(args, "match_id", None):
        args.latest = True

    match_id = str(getattr(args, "match_id", "") or "").strip() or None
    if getattr(args, "latest", False) and not match_id:
        match_id = _latest_match_id_from_db(db_path)
        if not match_id:
            raise SystemExit("Impossible de déterminer le match le plus récent (MatchStats).")

    cmd = [sys.executable, str(DEFAULT_FILM_ROSTER_REFETCH), "--db", str(db_path), "--write-aliases"]

    if getattr(args, "aliases", None):
        cmd += ["--aliases", str(args.aliases)]

    if getattr(args, "patch_highlight_events", False):
        cmd.append("--patch-highlight-events")

    # Ciblage: match unique (par défaut) ou tous.
    if getattr(args, "all_matches", False):
        cmd.append("--all-matches")
        if getattr(args, "db_source_table", None):
            cmd += ["--db-source-table", str(args.db_source_table)]
    else:
        if not match_id:
            raise SystemExit("Choisis --latest, --match-id, ou --all-matches")
        cmd += ["--match-id", match_id]

    # Contrôle download
    if getattr(args, "include_type2", False):
        cmd.append("--include-type2")
    if getattr(args, "max_type2_chunks", None) is not None:
        cmd += ["--max-type2-chunks", str(int(args.max_type2_chunks))]
    if getattr(args, "max_total_chunks", None) is not None:
        cmd += ["--max-total-chunks", str(int(args.max_total_chunks))]
    if getattr(args, "print_limit", None) is not None:
        cmd += ["--print-limit", str(int(args.print_limit))]

    print("[Repair] Film roster -> xuid_aliases.json")
    print("- db:", db_path)
    if getattr(args, "all_matches", False):
        print("- mode: all-matches")
    else:
        print("- match_id:", match_id)

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def _interactive(argv0: str) -> int:
    print("=" * 66)
    print("           LevelUp - Lanceur interactif")
    print("=" * 66)
    print("\n(Astuce: utilise --help pour les options CLI avancees)")
    print("\nChoisis un mode:\n")
    print("  1) Dashboard seul (sans refresh)")
    print()
    print("  -- Sync delta (rapide, nouveaux matchs uniquement) -------------")
    print("  2) Sync complete + dashboard                          [recommande]")
    print("     -> Delta + highlights + aliases film + profil, puis dashboard")
    print()
    print("  3) Sync complete (sans dashboard)")
    print("     -> Delta + highlights + aliases film + profil")
    print()
    print("  4) Sync complete (toutes les DB data/spnkr*.db)")
    print("     -> Delta + highlights + aliases film + profil pour chaque DB")
    print()
    print("  -- Refresh total (depuis zero) ---------------------------------")
    print("  5) Refresh total (recharge tous les matchs, sans delta)")
    print("     -> 1000 matchs + highlights + aliases film + profil")
    print()
    print("  Q) Quitter")
    print()

    choice = input("Ton choix (1/2/3/4/5/Q): ").strip().lower()
    if choice in {"q", "quit", "exit"}:
        return 0

    # Mode 1: Dashboard seul
    if choice == "1":
        db_p = _prompt_db_choice(purpose="à utiliser", default_db=_guess_default_spnkr_db(), allow_none=True)
        return _launch_streamlit(db_path=db_p, port=None, no_browser=False)

    # Modes 2-5: nécessitent un joueur (sauf mode 4 qui déduit depuis les DB)
    player: str | None = None
    if choice in {"2", "3", "5"}:
        player = _prompt_player_choice(default_player=os.environ.get("SPNKR_PLAYER"))
        if not player:
            print("Aucun joueur fourni.")
            return 2

    # Mode 2: Sync delta + aliases + profil + dashboard
    if choice == "2":
        args = argparse.Namespace(
            player=player,
            out_db=None,
            match_type="matchmaking",
            max_matches=100,  # Delta s'arrêtera avant si matchs connus
            rps=2,
            no_assets=False,
            no_skill=False,
            no_highlight_events=False,
            no_aliases=False,
            delta=True,  # Mode delta activé
            with_highlight_events=True,
            aliases_last=50,
            patch_highlight_events=True,
            include_type2=False,
            max_type2_chunks=0,
            max_total_chunks=None,
            print_limit=20,
            no_fetch_profile=False,
            port=None,
            no_browser=False,
            db=None,
        )
        return _cmd_run_with_refresh_and_aliases(args)

    # Mode 3: Sync delta + aliases + profil (sans dashboard)
    if choice == "3":
        args = argparse.Namespace(
            player=player,
            out_db=None,
            match_type="matchmaking",
            max_matches=100,
            rps=2,
            no_assets=False,
            no_skill=False,
            no_highlight_events=False,
            no_aliases=False,
            delta=True,  # Mode delta activé
            with_highlight_events=True,
            aliases_last=50,
            patch_highlight_events=True,
            include_type2=False,
            max_type2_chunks=0,
            max_total_chunks=None,
            print_limit=20,
            no_fetch_profile=False,
        )
        return _cmd_refresh_with_aliases(args)

    # Mode 4: Sync delta toutes les DB
    if choice == "4":
        return _cmd_refresh_all_with_aliases(argparse.Namespace(
            data_dir=str(DEFAULT_DATA_DIR),
            match_type="matchmaking",
            max_matches=100,
            rps=3,
            no_assets=False,
            no_skill=False,
            no_highlight_events=False,
            no_aliases=False,
            delta=True,  # Mode delta activé
            with_highlight_events=True,
            aliases_last=50,
            patch_highlight_events=True,
            include_type2=False,
            max_type2_chunks=0,
            max_total_chunks=None,
            print_limit=10,
            no_fetch_profile=False,
        ))

    # Mode 5: Refresh total (sans delta, depuis zéro)
    if choice == "5":
        args = argparse.Namespace(
            player=player,
            out_db=None,
            match_type="matchmaking",
            max_matches=1000,  # Beaucoup plus de matchs
            rps=2,
            no_assets=False,
            no_skill=False,
            no_highlight_events=False,
            no_aliases=False,
            delta=False,  # PAS de delta = refresh complet
            with_highlight_events=True,
            aliases_last=100,
            patch_highlight_events=True,
            include_type2=False,
            max_type2_chunks=0,
            max_total_chunks=None,
            print_limit=20,
            no_fetch_profile=False,
        )
        return _cmd_refresh_with_aliases(args)

    print("Choix invalide.")
    print(f"Usage CLI: {argv0} --help")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="openspartan",
        description=(
            "Lanceur LevelUp (dashboard Streamlit + refresh SPNKr).\n"
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
        "--no-highlight-events",
        action="store_true",
        help="Désactive l'import des highlight events (accélère l'import)",
    )
    p_ref.add_argument(
        "--no-aliases",
        action="store_true",
        help="Désactive le refresh des aliases (XUID → Gamertag)",
    )
    p_ref.add_argument(
        "--delta",
        action="store_true",
        help="Mode delta: s'arrête dès qu'un match déjà connu est rencontré (sync rapide)",
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
        "--no-highlight-events",
        action="store_true",
        help="Désactive l'import des highlight events (accélère l'import)",
    )
    p_ra.add_argument(
        "--no-aliases",
        action="store_true",
        help="Désactive le refresh des aliases (XUID → Gamertag)",
    )
    p_ra.add_argument(
        "--delta",
        action="store_true",
        help="Mode delta: s'arrête dès qu'un match déjà connu est rencontré (sync rapide)",
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
        "--no-highlight-events",
        action="store_true",
        help="Désactive l'import des highlight events (accélère l'import)",
    )
    p_runref.add_argument(
        "--no-aliases",
        action="store_true",
        help="Désactive le refresh des aliases (XUID → Gamertag)",
    )
    p_runref.add_argument("--port", type=int, default=None, help="Port (sinon auto)")
    p_runref.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur")
    p_runref.set_defaults(func=_cmd_run_with_refresh)

    p_refa = sub.add_parser(
        "refresh+aliases",
        help="Refresh SPNKr (avec highlight events) puis répare les aliases sur les N derniers matchs",
    )
    p_refa.add_argument("--player", default=None, help="Gamertag ou XUID (sinon: SPNKR_PLAYER)")
    p_refa.add_argument("--out-db", default=None, help="DB cible (sinon: data/spnkr_<player>.db)")
    p_refa.add_argument(
        "--match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs (défaut: matchmaking)",
    )
    p_refa.add_argument("--max-matches", type=int, default=50, help="Max matchs importés (défaut: 50)")
    p_refa.add_argument("--rps", type=int, default=2, help="Requests/sec (défaut: 2)")
    p_refa.add_argument("--no-assets", action="store_true", help="Désactive l'import des assets")
    p_refa.add_argument("--no-skill", action="store_true", help="Désactive l'import du skill")
    p_refa.add_argument(
        "--delta",
        action="store_true",
        help="Mode delta: s'arrête dès qu'un match déjà connu est rencontré (sync rapide)",
    )
    p_refa.add_argument(
        "--aliases-last",
        type=int,
        default=0,
        help="Répare les aliases sur les N derniers matchs (défaut: = --max-matches).",
    )
    p_refa.add_argument(
        "--patch-highlight-events",
        action="store_true",
        help="Optionnel: patch HighlightEvents.ResponseBody.gamertag en plus des aliases",
    )
    p_refa.add_argument("--include-type2", action="store_true", help="Inclut aussi des chunks type2")
    p_refa.add_argument("--max-type2-chunks", type=int, default=0, help="Limite chunks type2 (défaut: 0)")
    p_refa.add_argument("--max-total-chunks", type=int, default=None, help="Limite totale chunks")
    p_refa.add_argument("--print-limit", type=int, default=20, help="Limite logs (défaut: 20)")
    p_refa.add_argument(
        "--no-fetch-profile",
        action="store_true",
        help="Désactive le fetch des assets profil (emblem, backdrop, nameplate)",
    )
    p_refa.set_defaults(func=_cmd_refresh_with_aliases)

    p_runrefa = sub.add_parser(
        "run+refresh+aliases",
        help="Refresh SPNKr (avec highlight events) + répare aliases, puis lance le dashboard",
    )
    p_runrefa.add_argument("--player", default=None, help="Gamertag ou XUID (sinon: SPNKR_PLAYER)")
    p_runrefa.add_argument("--out-db", default=None, help="DB cible (sinon: data/spnkr_<player>.db)")
    p_runrefa.add_argument(
        "--match-type",
        default="matchmaking",
        choices=["all", "matchmaking", "custom", "local"],
        help="Type de matchs (défaut: matchmaking)",
    )
    p_runrefa.add_argument("--max-matches", type=int, default=50, help="Max matchs importés (défaut: 50)")
    p_runrefa.add_argument("--rps", type=int, default=2, help="Requests/sec (défaut: 2)")
    p_runrefa.add_argument("--no-assets", action="store_true", help="Désactive l'import des assets")
    p_runrefa.add_argument("--no-skill", action="store_true", help="Désactive l'import du skill")
    p_runrefa.add_argument(
        "--delta",
        action="store_true",
        help="Mode delta: s'arrête dès qu'un match déjà connu est rencontré (sync rapide)",
    )
    p_runrefa.add_argument(
        "--aliases-last",
        type=int,
        default=0,
        help="Répare les aliases sur les N derniers matchs (défaut: = --max-matches).",
    )
    p_runrefa.add_argument(
        "--patch-highlight-events",
        action="store_true",
        help="Optionnel: patch HighlightEvents.ResponseBody.gamertag en plus des aliases",
    )
    p_runrefa.add_argument("--include-type2", action="store_true", help="Inclut aussi des chunks type2")
    p_runrefa.add_argument("--max-type2-chunks", type=int, default=0, help="Limite chunks type2 (défaut: 0)")
    p_runrefa.add_argument("--max-total-chunks", type=int, default=None, help="Limite totale chunks")
    p_runrefa.add_argument("--print-limit", type=int, default=20, help="Limite logs (défaut: 20)")
    p_runrefa.add_argument(
        "--no-fetch-profile",
        action="store_true",
        help="Désactive le fetch des assets profil (emblem, backdrop, nameplate)",
    )
    p_runrefa.add_argument("--port", type=int, default=None, help="Port (sinon auto)")
    p_runrefa.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur")
    p_runrefa.set_defaults(func=_cmd_run_with_refresh_and_aliases)

    p_rep = sub.add_parser(
        "repair-aliases",
        help="Répare/complète xuid_aliases.json depuis les film chunks (SPNKr auth requise)",
    )
    p_rep.add_argument("--db", default=None, help="DB cible (défaut: OPENSPARTAN_DB_PATH ou data/spnkr*.db récent)")
    p_rep.add_argument("--match-id", default=None, help="Match GUID à réparer")
    p_rep.add_argument("--latest", action="store_true", help="Répare le match le plus récent (MatchStats)")
    p_rep.add_argument("--all-matches", action="store_true", help="Répare tous les matchs de la DB (long)")
    p_rep.add_argument(
        "--db-source-table",
        choices=["HighlightEvents", "MatchStats"],
        default="HighlightEvents",
        help="Table utilisée pour lister les MatchId quand --all-matches est activé",
    )
    p_rep.add_argument("--aliases", default=None, help="Chemin vers xuid_aliases.json (optionnel)")
    p_rep.add_argument(
        "--patch-highlight-events",
        action="store_true",
        help="Optionnel: patch HighlightEvents.ResponseBody.gamertag en plus des aliases",
    )
    p_rep.add_argument(
        "--include-type2",
        action="store_true",
        help="Inclut aussi des chunks type2 (par défaut: type1 seulement)",
    )
    p_rep.add_argument(
        "--max-type2-chunks",
        type=int,
        default=0,
        help="Limite le nombre de chunks type2 téléchargés (défaut: 0)",
    )
    p_rep.add_argument(
        "--max-total-chunks",
        type=int,
        default=None,
        help="Limite le nombre total de chunks téléchargés (type1+type2)",
    )
    p_rep.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="Limite le nombre de lignes affichées",
    )
    p_rep.set_defaults(func=_cmd_repair_aliases)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    _maybe_reexec_into_venv(argv)

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
