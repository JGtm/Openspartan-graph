"""Fonctions de parsing et utilitaires pour la DB."""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import DEFAULT_PLAYER_GAMERTAG, DEFAULT_PLAYER_XUID, XUID_ALIASES_DEFAULT, get_aliases_file_path
from src.db.connection import get_connection


def infer_spnkr_player_from_db_path(db_path: str) -> Optional[str]:
    """Déduit le paramètre --player à utiliser pour une DB SPNKr.

    Conventions supportées (aligné avec refresh_all_dbs.bat):
    - spnkr_gt_<Gamertag>.db  -> <Gamertag>
    - spnkr_xuid_<XUID>.db   -> <XUID>
    - spnkr_<something>.db   -> <something> (souvent un gamertag)
    """
    base = os.path.basename(db_path or "")
    stem, _ = os.path.splitext(base)
    s = (stem or "").strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("spnkr_gt_"):
        out = s[len("spnkr_gt_") :]
    elif low.startswith("spnkr_xuid_"):
        out = s[len("spnkr_xuid_") :]
    elif low.startswith("spnkr_"):
        out = s[len("spnkr_") :]
    else:
        return None
    out = (out or "").strip().strip("_- ")
    return out or None


def guess_xuid_from_db_path(db_path: str) -> Optional[str]:
    """Devine le XUID à partir du nom du fichier .db.
    
    La convention OpenSpartan Workshop nomme les fichiers <XUID>.db.
    Mais on supporte aussi des noms "conviviaux" (ex: spnkr_gt_<Gamertag>.db)
    ou des fichiers renommés à la main (ex: <Gamertag>.db).
    
    Args:
        db_path: Chemin vers le fichier .db.
        
    Returns:
        Le XUID si le nom de fichier est numérique, None sinon.
    """
    base = os.path.basename(db_path or "")
    stem, _ = os.path.splitext(base)
    if stem.isdigit():
        return stem

    # Ex: spnkr_xuid_2533....db / name_2533....db
    m = _XUID_DIGITS_RE.search(stem)
    if m:
        return m.group(1)

    # Ex: spnkr_gt_Chocoboflor.db -> Chocoboflor
    gt_guess = stem
    if gt_guess.lower().startswith("spnkr_gt_"):
        gt_guess = gt_guess[len("spnkr_gt_") :]
    elif gt_guess.lower().startswith("spnkr_gt-"):
        gt_guess = gt_guess[len("spnkr_gt-") :]
    elif gt_guess.lower().startswith("spnkr_"):
        # Au cas où: spnkr_<something>
        gt_guess = gt_guess[len("spnkr_") :]

    gt_guess = (gt_guess or "").strip().strip("_-")
    if not gt_guess:
        return None

    # Si le nom de fichier a été "sanitisé" (espaces -> _) on tente aussi la variante.
    gt_candidates = {gt_guess, gt_guess.replace("_", " "), gt_guess.replace("_", "")}
    gt_candidates = {c.strip() for c in gt_candidates if c and c.strip()}
    if not gt_candidates:
        return None

    # Tente de résoudre via alias hardcodés + fichier d'alias.
    try:
        aliases: dict[str, str] = dict(XUID_ALIASES_DEFAULT)
        aliases_path = get_aliases_file_path()
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for x, gt in obj.items():
                    if isinstance(x, str) and isinstance(gt, str):
                        aliases[x.strip()] = gt.strip()

        # Inversion (gamertag -> xuid)
        inv: dict[str, str] = {}
        for x, gt in aliases.items():
            if isinstance(x, str) and x.strip().isdigit() and isinstance(gt, str) and gt.strip():
                inv[gt.strip().casefold()] = x.strip()

        for c in gt_candidates:
            hit = inv.get(c.casefold())
            if hit:
                return hit
    except Exception:
        pass

    return None


def parse_iso_utc(s: str) -> datetime:
    """Parse une date ISO 8601 en datetime UTC.
    
    Gère le format utilisé par l'API Halo: 2026-01-02T20:18:01.293Z
    
    Args:
        s: Chaîne de date au format ISO 8601.
        
    Returns:
        datetime en timezone UTC.
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def coerce_number(v: Any) -> Optional[float]:
    """Convertit une valeur en float de manière robuste.
    
    Gère différents formats vus dans l'API Halo:
    - Nombres directs (int, float)
    - Chaînes numériques
    - Dictionnaires avec clés Count, Value, Seconds, etc.
    
    Args:
        v: Valeur à convertir.
        
    Returns:
        La valeur en float, ou None si la conversion échoue.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    if isinstance(v, dict):
        # Formats vus dans certaines APIs: {"Count": 19} ou {"Value": 19}
        for k in ("Count", "Value", "value", "Seconds", "Milliseconds", "Ms"):
            if k in v:
                return coerce_number(v.get(k))
    return None


# Regex pour les durées ISO 8601 (ex: PT31.5S)
_ISO8601_DURATION_RE = re.compile(
    r"^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$"
)


def coerce_duration_seconds(v: Any) -> Optional[float]:
    """Convertit une durée en secondes.
    
    Gère différents formats:
    - Nombres directs (déjà en secondes)
    - Chaînes ISO 8601 (ex: "PT31.5S")
    - Dictionnaires avec Seconds ou Milliseconds
    
    Args:
        v: Valeur de durée à convertir.
        
    Returns:
        La durée en secondes, ou None si la conversion échoue.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "Milliseconds" in v or "Ms" in v:
            ms = coerce_number(v.get("Milliseconds") if "Milliseconds" in v else v.get("Ms"))
            return (ms / 1000.0) if ms is not None else None
        if "Seconds" in v:
            return coerce_number(v.get("Seconds"))
        return coerce_number(v)
    if isinstance(v, str):
        s = v.strip()
        m = _ISO8601_DURATION_RE.match(s)
        if not m:
            return None
        hours = float(m.group("h") or 0)
        minutes = float(m.group("m") or 0)
        seconds = float(m.group("s") or 0)
        return (hours * 3600.0) + (minutes * 60.0) + seconds
    return None


def parse_xuid_input(s: str) -> Optional[str]:
    """Parse une entrée utilisateur de XUID.
    
    Accepte:
    - Un nombre direct: "2533274823110022"
    - Le format xuid(): "xuid(2533274823110022)"
    
    Args:
        s: Entrée utilisateur.
        
    Returns:
        Le XUID extrait, ou None si invalide.
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return s
    m = re.fullmatch(r"xuid\((\d+)\)", s)
    if m:
        return m.group(1)
    return None


_XUID_DIGITS_RE = re.compile(r"(\d{12,20})")


def _extract_xuid_from_player_id(player_id: Any) -> Optional[str]:
    if player_id is None:
        return None
    if isinstance(player_id, dict):
        for k in ("Xuid", "xuid", "XUID"):
            if k in player_id:
                v = player_id.get(k)
                if isinstance(v, (int, str)):
                    parsed = parse_xuid_input(str(v))
                    if parsed:
                        return parsed
                    m = _XUID_DIGITS_RE.search(str(v))
                    if m:
                        return m.group(1)
        return None
    if isinstance(player_id, (int, str)):
        s = str(player_id)
        parsed = parse_xuid_input(s)
        if parsed:
            return parsed
        m = _XUID_DIGITS_RE.search(s)
        if m:
            return m.group(1)
    return None


def _extract_gamertag_from_player_id(player_id: Any) -> Optional[str]:
    if player_id is None:
        return None
    if isinstance(player_id, dict):
        for k in ("Gamertag", "gamertag", "GT"):
            v = player_id.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    return None


def resolve_xuid_from_db(db_path: str, player: str, *, limit_rows: int = 400) -> Optional[str]:
    """Résout un XUID à partir d'une entrée utilisateur.

    - Si `player` est déjà un XUID (ou xuid(...)), renvoie le XUID.
    - Sinon, tente de retrouver le XUID à partir du gamertag en scannant les JSON
      de `MatchStats.ResponseBody` (utile pour la DB SPNKr comme pour la DB Workshop).
    """
    p = (player or "").strip()
    if not p:
        return None

    parsed = parse_xuid_input(p)
    if parsed:
        return parsed

    # Fallback 1: valeurs par défaut (local)
    default_gt = (os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or DEFAULT_PLAYER_GAMERTAG or "").strip()
    default_xuid = (os.environ.get("OPENSPARTAN_DEFAULT_XUID") or DEFAULT_PLAYER_XUID or "").strip()
    if default_gt and p.casefold() == default_gt.casefold():
        return default_xuid or None

    # Fallback 2: aliases locaux (et hardcodés)
    try:
        aliases: dict[str, str] = dict(XUID_ALIASES_DEFAULT)
        aliases_path = get_aliases_file_path()
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for x, gt in obj.items():
                    if isinstance(x, str) and isinstance(gt, str):
                        aliases[x.strip()] = gt.strip()
        for x, gt in aliases.items():
            if gt and gt.casefold() == p.casefold() and x.strip().isdigit():
                return x.strip()
    except Exception:
        # Non bloquant
        pass

    if not db_path or not os.path.exists(db_path):
        return None

    gt = p.casefold()

    try:
        with get_connection(db_path) as con:
            cur = con.cursor()
            cur.execute(
                "SELECT ResponseBody FROM MatchStats WHERE ResponseBody IS NOT NULL ORDER BY rowid DESC LIMIT ?",
                (int(limit_rows),),
            )
            rows = cur.fetchall()
    except (sqlite3.Error, OSError):
        return None

    for (raw,) in rows:
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        players = obj.get("Players")
        if not isinstance(players, list):
            continue

        for pl in players:
            if not isinstance(pl, dict):
                continue
            pid = pl.get("PlayerId")
            gamertag = _extract_gamertag_from_player_id(pid)
            if not gamertag:
                continue
            if gamertag.casefold() != gt:
                continue
            xuid = _extract_xuid_from_player_id(pid)
            if xuid:
                return xuid

    return None
