#!/usr/bin/env python3
"""Télécharge toutes les icônes de Career Rank Halo Infinite en local.

Usage:
    python scripts/download_career_rank_icons.py
    python scripts/download_career_rank_icons.py --metadata-only
    python scripts/download_career_rank_icons.py --force

Prérequis:
    - Tokens SPNKr configurés (SPNKR_AZURE_CLIENT_ID, etc.) ou cache career_ranks_metadata.json existant
    - Accès réseau pour télécharger les images

Output:
    - data/cache/career_ranks/ contenant 272 icônes PNG (rank_001_large.png à rank_272_large.png)
    - data/cache/career_ranks_metadata.json avec les métadonnées complètes des rangs
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

# Ajouter le dossier parent au path pour importer src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.profile_api import _load_dotenv_if_present, ensure_spnkr_tokens


def get_cache_dir() -> Path:
    return PROJECT_ROOT / "data" / "cache" / "career_ranks"


def get_metadata_path() -> Path:
    return PROJECT_ROOT / "data" / "cache" / "career_ranks_metadata.json"


async def fetch_career_ranks_metadata() -> dict:
    """Récupère les métadonnées des Career Ranks depuis l'API CMS."""
    st = os.environ.get("SPNKR_SPARTAN_TOKEN", "")
    ct = os.environ.get("SPNKR_CLEARANCE_TOKEN", "")
    
    if not (st and ct):
        raise RuntimeError("Tokens SPNKr non disponibles")
    
    headers = {
        "Accept": "application/json",
        "X-343-Authorization-Spartan": st,
        "343-Clearance": ct,
    }
    
    url = "https://gamecms-hacs.svc.halowaypoint.com/hi/Progression/file/RewardTracks/CareerRanks/careerRank1.json"
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API error: {resp.status}")
            return await resp.json()


def load_or_fetch_metadata() -> dict:
    """Charge les métadonnées depuis le cache ou les récupère depuis l'API."""
    metadata_path = get_metadata_path()
    
    if metadata_path.exists():
        print(f"Chargement des metadonnees depuis {metadata_path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    
    print("Recuperation des metadonnees depuis l'API...")
    _load_dotenv_if_present()
    ensure_spnkr_tokens(timeout_seconds=20)
    
    data = asyncio.run(fetch_career_ranks_metadata())
    
    # Sauvegarder
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadonnees sauvegardees dans {metadata_path}")
    
    return data


async def download_icon(
    session: aiohttp.ClientSession,
    rank_num: int,
    icon_path: str,
    out_path: Path,
    headers: dict,
) -> str:
    """Télécharge une icône de rang.
    
    Returns: "ok", "skip", ou code d'erreur
    """
    if out_path.exists():
        return "skip"
    
    url = f"https://gamecms-hacs.svc.halowaypoint.com/hi/images/file/{icon_path}"
    
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.read()
                if data[:8] == b'\x89PNG\r\n\x1a\n':
                    out_path.write_bytes(data)
                    return "ok"
                return "not_png"
            return f"http_{resp.status}"
    except Exception as e:
        return f"error_{type(e).__name__}"


async def download_all_icons(metadata: dict, *, force: bool = False) -> tuple[int, int, int]:
    """Télécharge toutes les icônes de rang en parallèle.
    
    Returns: (downloaded, skipped, failed)
    """
    _load_dotenv_if_present()
    ensure_spnkr_tokens(timeout_seconds=20)
    
    st = os.environ.get("SPNKR_SPARTAN_TOKEN", "")
    ct = os.environ.get("SPNKR_CLEARANCE_TOKEN", "")
    
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    ranks = metadata.get("Ranks", [])
    
    headers = {
        "Accept": "image/png",
        "X-343-Authorization-Spartan": st,
        "343-Clearance": ct,
    }
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=5)  # Max 5 connexions parallèles
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        
        for rank_data in ranks:
            rank_num = rank_data.get("Rank", 0)
            icon_large = rank_data.get("RankLargeIcon", "")
            
            if not icon_large:
                continue
            
            out_path = cache_dir / f"rank_{rank_num:03d}_large.png"
            
            if out_path.exists() and not force:
                tasks.append((rank_num, asyncio.coroutine(lambda: "skip")()))
            else:
                tasks.append((rank_num, download_icon(session, rank_num, icon_large, out_path, headers)))
        
        # Traiter par lots de 10
        total = len(tasks)
        for i in range(0, total, 10):
            batch = tasks[i:i+10]
            results = await asyncio.gather(*[t[1] for t in batch])
            
            for (rank_num, _), result in zip(batch, results):
                if result == "ok":
                    downloaded += 1
                elif result == "skip":
                    skipped += 1
                else:
                    failed += 1
            
            progress = i + len(batch)
            if progress % 50 == 0 or progress == total:
                print(f"  Progression: {progress}/{total}")
    
    return downloaded, skipped, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Telecharge les icones de Career Rank Halo Infinite")
    parser.add_argument("--force", "-f", action="store_true", help="Re-telecharger meme si le fichier existe")
    parser.add_argument("--metadata-only", "-m", action="store_true", help="Ne telecharger que les metadonnees")
    args = parser.parse_args()
    
    metadata = load_or_fetch_metadata()
    
    ranks = metadata.get("Ranks", [])
    print(f"Nombre de rangs: {len(ranks)}")
    
    if args.metadata_only:
        print("Mode metadata-only: pas de telechargement d'icones.")
        return
    
    print("Telechargement des icones...")
    downloaded, skipped, failed = asyncio.run(download_all_icons(metadata, force=args.force))
    
    print(f"\nTermine!")
    print(f"  Telecharges: {downloaded}")
    print(f"  Ignores (existants): {skipped}")
    print(f"  Echecs: {failed}")
    
    cache_dir = get_cache_dir()
    files = list(cache_dir.glob("*.png"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nTotal: {len(files)} fichiers, {total_size / 1024 / 1024:.1f} Mo")


if __name__ == "__main__":
    main()
