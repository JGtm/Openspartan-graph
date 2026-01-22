"""Précharge l'apparence (service tag + URLs) et les images profil pour un gamertag.

But:
- Résoudre gamertag -> XUID via l'API Halo Waypoint (SPNKr)
- Récupérer l'apparence (emblem/backdrop/nameplate)
- Télécharger les images et les mettre en cache dans data/cache/player_assets/

Pré-requis (si pas déjà installé):
- pip install "spnkr @ git+https://github.com/acurtis166/SPNKr.git" aiohttp pydantic

Authentification:
- Soit tokens manuels: SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN
- Soit Azure refresh: SPNKR_AZURE_CLIENT_ID + SPNKR_AZURE_CLIENT_SECRET + SPNKR_OAUTH_REFRESH_TOKEN

Notes:
- Ce script déclenche du réseau explicitement (sauf --offline).
- Cache API: data/cache/profile_api/
- Cache images: data/cache/player_assets/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import importlib.util
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_dotenv_if_present() -> None:
    """Charge un fichier `.env.local` puis `.env` à la racine du repo (si présents).

    Objectif: faciliter l'usage en local sans installer python-dotenv.
    Règles:
    - lignes `KEY=VALUE`
    - ignore lignes vides / commentaires (#)
    - ne remplace pas une variable déjà définie dans l'environnement
    """

    repo_root = _repo_root()

    for name in (".env.local", ".env"):
        dotenv_path = repo_root / name
        if not dotenv_path.exists():
            continue
        try:
            content = dotenv_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if os.environ.get(key) is None:
                os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_present()

    ap = argparse.ArgumentParser(description="Précharge le profil (appearance + images) pour un gamertag")
    ap.add_argument("gamertag", nargs="+", help="Un ou plusieurs gamertags")

    ap.add_argument("--offline", action="store_true", help="N'utilise pas le réseau (cache uniquement)")
    ap.add_argument("--no-download", action="store_true", help="Ne télécharge pas les images (API only)")

    ap.add_argument(
        "--strict",
        action="store_true",
        help="Échoue (exit code 1) si un asset ne peut pas être téléchargé.",
    )

    ap.add_argument("--force", action="store_true", help="Ignore les caches (API + re-download images)")

    ap.add_argument(
        "--api-refresh-hours",
        type=int,
        default=24,
        help="TTL du cache API (heures). 0 = ne jamais re-fetch si cache présent.",
    )
    ap.add_argument(
        "--assets-refresh-hours",
        type=int,
        default=0,
        help="TTL du cache images (heures). 0 = ne jamais re-download si cache présent.",
    )

    ap.add_argument("--requests-per-second", type=int, default=3, help="Rate limit SPNKr (par service)")
    ap.add_argument("--timeout-seconds", type=int, default=12, help="Timeout HTTP")

    args = ap.parse_args(argv)

    # Permet d'importer `src.*` depuis scripts/
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    def _load_module(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Impossible de charger le module: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    try:
        profile_api_mod = _load_module(
            "openspartan_profile_api",
            repo_root / "src" / "ui" / "profile_api.py",
        )
        player_assets_mod = _load_module(
            "openspartan_player_assets",
            repo_root / "src" / "ui" / "player_assets.py",
        )
    except Exception as e:
        print(f"Import impossible: {e}")
        return 2

    get_profile_appearance = profile_api_mod.get_profile_appearance
    get_profile_api_cache_dir = profile_api_mod.get_profile_api_cache_dir
    get_xuid_for_gamertag = profile_api_mod.get_xuid_for_gamertag

    download_image_to_cache = player_assets_mod.download_image_to_cache
    ensure_local_image_path = player_assets_mod.ensure_local_image_path
    get_player_assets_cache_dir = player_assets_mod.get_player_assets_cache_dir

    enabled = not bool(args.offline)
    download_enabled = (not bool(args.no_download)) and enabled

    had_error = False
    had_warning = False

    print(f"Cache API: {get_profile_api_cache_dir()}")
    print(f"Cache images: {get_player_assets_cache_dir()}")

    for raw_gt in args.gamertag:
        gt = str(raw_gt or "").strip()
        if not gt:
            continue

        print("\n" + "=" * 60)
        print(f"Gamertag: {gt}")

        xuid, xuid_err = get_xuid_for_gamertag(
            gamertag=gt,
            enabled=enabled,
            refresh_hours=int(args.api_refresh_hours),
            force_refresh=bool(args.force),
            requests_per_second=int(args.requests_per_second),
            timeout_seconds=int(args.timeout_seconds),
        )
        if xuid_err:
            had_error = True
            print(f"[ERREUR] {xuid_err}")
        if not xuid:
            had_error = True
            print("[ERREUR] XUID introuvable.")
            continue

        print(f"XUID: {xuid}")

        appearance, app_err = get_profile_appearance(
            xuid=xuid,
            enabled=enabled,
            refresh_hours=int(args.api_refresh_hours),
            force_refresh=bool(args.force),
            requests_per_second=int(args.requests_per_second),
            timeout_seconds=int(args.timeout_seconds),
        )

        if app_err:
            had_error = True
            print(f"[ERREUR] {app_err}")

        if appearance is None:
            had_error = True
            print("[ERREUR] Appearance introuvable.")
            continue

        print(f"Service tag: {appearance.service_tag or ''}")
        print(f"Emblem URL: {appearance.emblem_image_url or ''}")
        print(f"Backdrop URL: {appearance.backdrop_image_url or ''}")
        print(f"Nameplate URL: {appearance.nameplate_image_url or ''}")
        print(f"Rank label: {appearance.rank_label or ''}")
        print(f"Rank subtitle: {appearance.rank_subtitle or ''}")
        print(f"Rank icon URL: {getattr(appearance, 'rank_image_url', '') or ''}")

        if not download_enabled:
            continue

        for prefix, url in (
            ("emblem", appearance.emblem_image_url),
            ("backdrop", appearance.backdrop_image_url),
            ("nameplate", appearance.nameplate_image_url),
            ("rank", getattr(appearance, "rank_image_url", None)),
        ):
            if not url:
                continue

            if bool(args.force):
                ok, err, out_path = download_image_to_cache(
                    url,
                    prefix=prefix,
                    timeout_seconds=int(args.timeout_seconds),
                )
                if not ok:
                    had_warning = True
                    print(f"[WARN] Download {prefix}: {err}")
                    if bool(args.strict):
                        had_error = True
                else:
                    print(f"Downloaded {prefix}: {out_path}")
                continue

            local_path = ensure_local_image_path(
                url,
                prefix=prefix,
                download_enabled=True,
                auto_refresh_hours=int(args.assets_refresh_hours),
                timeout_seconds=int(args.timeout_seconds),
            )
            if not local_path:
                had_warning = True
                print(f"[WARN] Cache {prefix}: impossible de récupérer {url}")
                if bool(args.strict):
                    had_error = True
            else:
                print(f"Cached {prefix}: {local_path}")

    if had_error:
        return 1
    # Warnings non bloquants: exit 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
