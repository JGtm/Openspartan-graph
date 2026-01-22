"""Chargement des données depuis la base SQLite."""

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

from src.config import BOT_MAP, TEAM_MAP
from src.db.connection import get_connection
from src.db.parsers import (
    coerce_duration_seconds,
    coerce_number,
    parse_iso_utc,
    parse_xuid_input,
)
from src.db import queries
from src.models import MatchRow, FriendMatch


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€˜",
    "â€œ",
    "â€�",
    "â€“",
    "â€”",
    "ðŸ",
)


def _mojibake_score(s: str) -> int:
    return sum(s.count(m) for m in _MOJIBAKE_MARKERS)


def _fix_mojibake(s: str) -> str:
    """Tente de corriger un texte UTF-8 mal décodé (mojibake).

    Exemples typiques:
    - "FranÃ§ois" -> "François"
    - "Dâ€™Artagnan" -> "D’Artagnan"
    """
    if not s:
        return s

    base_score = _mojibake_score(s)
    if base_score == 0:
        return s

    best = s
    best_score = base_score

    for enc in ("latin1", "cp1252"):
        try:
            candidate = s.encode(enc).decode("utf-8")
        except Exception:
            continue
        cand_score = _mojibake_score(candidate)
        if cand_score < best_score:
            best = candidate
            best_score = cand_score
            if best_score == 0:
                break

    return best


def _sanitize_gamertag(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        for enc in ("utf-8", "utf-16-le", "utf-16"):
            try:
                value = raw.decode(enc)
                break
            except Exception:
                continue
        else:
            value = raw.decode("utf-8", errors="replace")

    if not isinstance(value, str):
        return value

    # Certains dumps SPNKr mettent dans "gamertag" un champ binaire/structuré
    # (souvent avec des NUL). Dans ce cas, ce n'est pas un vrai gamertag,
    # et le garder dégrade l'UI (ex: "ipP\u0100", "aba73\u0103").
    # On préfère alors considérer le nom comme absent et retomber sur XUID/alias.
    if _CTRL_RE.search(value):
        return ""

    s = _fix_mojibake(value)
    s = s.replace("\ufffd", "")
    s = _CTRL_RE.sub("", s)
    s = " ".join(s.split()).strip()
    return s or value


def has_table(db_path: str, table_name: str) -> bool:
    if not db_path or not table_name:
        return False
    try:
        with get_connection(db_path) as con:
            cur = con.cursor()
            cur.execute(queries.HAS_TABLE, (table_name,))
            return cur.fetchone() is not None
    except Exception:
        return False


def load_highlight_events_for_match(db_path: str, match_id: str) -> list[dict[str, Any]]:
    """Charge les highlight events (film) pour un match.

    Source: table HighlightEvents (produite par scripts/spnkr_import_db.py avec --with-highlight-events)

    Returns:
        Liste de dicts (JSON brut) — typiquement avec: event_type, time_ms, xuid, gamertag, type_hint, ...
    """
    if not match_id:
        return []
    if not has_table(db_path, "HighlightEvents"):
        return []

    out: list[dict[str, Any]] = []
    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LOAD_HIGHLIGHT_EVENTS_BY_MATCH_ID, (match_id,))
        for (body,) in cur.fetchall():
            if body is None:
                continue

            body_str: str | None = None
            if isinstance(body, str):
                body_str = body
            elif isinstance(body, (bytes, bytearray, memoryview)):
                raw = bytes(body)
                for enc in ("utf-8", "utf-16-le", "utf-16"):
                    try:
                        body_str = raw.decode(enc)
                        break
                    except Exception:
                        continue
                if body_str is None:
                    body_str = raw.decode("utf-8", errors="replace")
            else:
                continue

            if not body_str:
                continue
            try:
                obj = json.loads(body_str)
            except Exception:
                continue
            if isinstance(obj, dict):
                if "gamertag" in obj:
                    obj["gamertag"] = _sanitize_gamertag(obj.get("gamertag"))
                out.append(obj)
    return out


def load_match_player_gamertags(db_path: str, match_id: str) -> Dict[str, str]:
    """Retourne un mapping XUID -> Gamertag pour un match.

    Les gamertags des HighlightEvents peuvent être absents ou mal encodés.
    MatchStats contient généralement les identités des joueurs de manière plus fiable.

    Returns:
        Dict {xuid_str: gamertag_str}
    """
    if not match_id:
        return {}

    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LOAD_MATCH_STATS_BY_MATCH_ID, (match_id,))
        row = cur.fetchone()
        if not row or not isinstance(row[0], str) or not row[0]:
            return {}

        try:
            payload = json.loads(row[0])
        except Exception:
            return {}

    players = payload.get("Players")
    if not isinstance(players, list) or not players:
        return {}

    out: Dict[str, str] = {}

    def _extract_from_player_obj(p: Any) -> tuple[str | None, str | None]:
        if not isinstance(p, dict):
            return None, None

        pid = p.get("PlayerId")
        xuid_val: Any = None
        gt_val: Any = None

        if isinstance(pid, dict):
            xuid_val = pid.get("Xuid") or pid.get("xuid")
            gt_val = pid.get("Gamertag") or pid.get("gamertag")
        elif isinstance(pid, str):
            # Parfois, PlayerId peut être une chaîne (gamertag ou xuid)
            xuid_val = pid
            gt_val = p.get("Gamertag") or p.get("gamertag")

        if xuid_val is None:
            xuid_val = p.get("Xuid") or p.get("xuid")
        if gt_val is None:
            gt_val = p.get("Gamertag") or p.get("gamertag")

        xuid_s = str(xuid_val or "").strip()
        # Normalise xuid("...") -> "..." si besoin
        xuid_norm = parse_xuid_input(xuid_s) or xuid_s
        gt_s = _sanitize_gamertag(gt_val)
        if not isinstance(gt_s, str):
            gt_s = str(gt_s or "").strip()
        gt_s = gt_s.strip()
        return (xuid_norm or None), (gt_s or None)

    for p in players:
        xuid_norm, gt = _extract_from_player_obj(p)
        if xuid_norm and gt:
            out[str(xuid_norm)] = gt

    return out


def load_match_rosters(
    db_path: str,
    match_id: str,
    xuid: str,
) -> Optional[Dict[str, Any]]:
    """Retourne les rosters du match (mon équipe vs équipe adverse).

    Source: table MatchStats (Players[] / LastTeamId).

    Returns:
        None si indisponible ou si l'équipe du joueur ne peut pas être déterminée.
        Sinon un dict:
            {
              "my_team_id": int,
              "my_team": [{"xuid": str, "gamertag": str|None, "team_id": int|None, "is_me": bool}],
              "enemy_team": [...],
            }
    """
    if not match_id or not xuid:
        return None

    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LOAD_MATCH_STATS_BY_MATCH_ID, (match_id,))
        row = cur.fetchone()
        if not row or not isinstance(row[0], str) or not row[0]:
            return None

        try:
            payload = json.loads(row[0])
        except Exception:
            return None

    players = payload.get("Players")
    if not isinstance(players, list) or not players:
        return None

    _BOT_ID_RE = re.compile(r"^\s*bid\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*$", re.IGNORECASE)

    def _normalize_bot_key(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            m = _BOT_ID_RE.match(s)
            if not m:
                return None
            try:
                n = float(m.group(1))
            except Exception:
                return None
            return f"bid({n:.1f})"
        if isinstance(value, (int, float)):
            try:
                return f"bid({float(value):.1f})"
            except Exception:
                return None
        return None

    def _extract_bot_name(p: Any) -> tuple[str | None, str | None]:
        if not isinstance(p, dict):
            return None, None
        pid = p.get("PlayerId")

        bot_key = _normalize_bot_key(pid)
        if bot_key is None and isinstance(pid, dict):
            bot_key = _normalize_bot_key(pid.get("BotId") or pid.get("botId") or pid.get("Bot"))
        if bot_key is None:
            bot_key = _normalize_bot_key(p.get("BotId") or p.get("botId"))

        if not bot_key:
            return None, None
        return bot_key, BOT_MAP.get(bot_key)

    def _extract_identity(p: Any) -> tuple[str | None, str | None]:
        if not isinstance(p, dict):
            return None, None

        pid = p.get("PlayerId")
        xuid_val: Any = None
        gt_val: Any = None

        if isinstance(pid, dict):
            xuid_val = pid.get("Xuid") or pid.get("xuid")
            gt_val = pid.get("Gamertag") or pid.get("gamertag")
        elif isinstance(pid, str):
            xuid_val = pid
            gt_val = p.get("Gamertag") or p.get("gamertag")

        if xuid_val is None:
            xuid_val = p.get("Xuid") or p.get("xuid")
        if gt_val is None:
            gt_val = p.get("Gamertag") or p.get("gamertag")

        xuid_s = str(xuid_val or "").strip()
        xuid_norm = parse_xuid_input(xuid_s) or xuid_s
        gt_s = _sanitize_gamertag(gt_val)
        if not isinstance(gt_s, str):
            gt_s = str(gt_s or "").strip()
        gt_s = gt_s.strip()
        return (xuid_norm or None), (gt_s or None)

    out_rows: list[dict[str, Any]] = []
    my_team_id: int | None = None

    for p in players:
        if not isinstance(p, dict):
            continue

        team_raw = p.get("LastTeamId")
        if team_raw is None:
            team_raw = p.get("TeamId")

        team_id: int | None = None
        try:
            if team_raw is not None and team_raw == team_raw:
                team_id = int(team_raw)
        except Exception:
            team_id = None

        xuid_norm, gt = _extract_identity(p)
        bot_key, bot_name = _extract_bot_name(p)

        is_me = False
        try:
            is_me = _xuid_id_matches(p.get("PlayerId"), xuid)
        except Exception:
            is_me = False

        if is_me and team_id is not None:
            my_team_id = team_id

        # Nom d'affichage: bot > gamertag > vide
        display_name = bot_name or gt

        out_rows.append(
            {
                "xuid": str(xuid_norm) if xuid_norm is not None else "",
                "gamertag": gt,
                "bot_id": bot_key,
                "is_bot": bool(bot_key),
                "display_name": display_name,
                "team_id": team_id,
                "team_name": TEAM_MAP.get(team_id) if team_id is not None else None,
                "is_me": bool(is_me),
            }
        )

    if my_team_id is None:
        return None

    my_team = [r for r in out_rows if r.get("team_id") == my_team_id]
    enemy_team = [r for r in out_rows if r.get("team_id") != my_team_id]

    def _sort_key(r: dict[str, Any]) -> tuple[int, str]:
        # Moi en premier, puis ordre alphabétique stable (gamertag puis xuid)
        me_rank = 0 if r.get("is_me") else 1
        name = str(r.get("gamertag") or "").strip().lower()
        if not name:
            name = str(r.get("xuid") or "").strip().lower()
        return (me_rank, name)

    my_team.sort(key=_sort_key)
    enemy_team.sort(key=_sort_key)

    enemy_team_ids = sorted({int(r["team_id"]) for r in enemy_team if r.get("team_id") is not None})
    enemy_team_names = [TEAM_MAP.get(tid) for tid in enemy_team_ids]
    enemy_team_names = [n for n in enemy_team_names if isinstance(n, str) and n]

    return {
        "my_team_id": int(my_team_id),
        "my_team_name": TEAM_MAP.get(int(my_team_id)),
        "my_team": my_team,
        "enemy_team": enemy_team,
        "enemy_team_ids": enemy_team_ids,
        "enemy_team_names": enemy_team_names,
    }


def load_top_medals(
    db_path: str,
    xuid: str,
    match_ids: List[str],
    *,
    top_n: int | None = 25,
) -> List[tuple[int, int]]:
    """Retourne les médailles les plus fréquentes (NameId -> total).

    Agrège uniquement sur la liste de MatchIds fournie (utile pour respecter
    exactement les filtres UI déjà appliqués).

    Notes:
        - SQLite a une limite sur le nombre de paramètres, on exécute par chunks.
        - Le mapping NameId -> libellé/icône est optionnel côté UI.
    """
    if not xuid or not match_ids:
        return []

    if top_n is not None and int(top_n) <= 0:
        return []

    # Normalise et déduplique en gardant l'ordre
    norm: List[str] = []
    seen: set[str] = set()
    for mid in match_ids:
        if not isinstance(mid, str):
            continue
        s = mid.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        norm.append(s)

    if not norm:
        return []

    me_id = f"xuid({xuid})"
    totals: Dict[int, int] = {}
    chunk_size = 800

    with get_connection(db_path) as con:
        cur = con.cursor()
        for i in range(0, len(norm), chunk_size):
            chunk = norm[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = queries.LOAD_TOP_MEDALS_FOR_MATCH_IDS.format(match_ids=placeholders)
            cur.execute(sql, (*chunk, me_id))
            for name_id, total in cur.fetchall():
                if name_id is None or total is None:
                    continue
                try:
                    nid = int(name_id)
                    cnt = int(total)
                except Exception:
                    continue
                totals[nid] = totals.get(nid, 0) + cnt

    out = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is None:
        return out
    return out[: int(top_n)]


def _xuid_id_matches(pid: Any, xuid: str) -> bool:
    """Retourne True si un identifiant joueur correspond au XUID.

    Dans les payloads OpenSpartan, l'Id est souvent de la forme "xuid(123...)".
    On reste tolérant en comparant via json.dumps comme ailleurs dans ce fichier.
    """
    if pid is None:
        return False
    try:
        return xuid in json.dumps(pid)
    except Exception:
        return False


def load_player_match_result(
    db_path: str,
    match_id: str,
    xuid: str,
) -> Optional[Dict[str, Any]]:
    """Charge le résultat PlayerMatchStats pour un match et un joueur.

    On s'appuie sur la table PlayerMatchStats (join via colonne MatchId).

    Returns:
        Un dict avec des champs normalisés utiles à l'UI, ou None si indisponible.
    """
    if not match_id or not xuid:
        return None

    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LOAD_PLAYER_MATCH_STATS_BY_MATCH_ID, (match_id,))
        row = cur.fetchone()
        if not row or not isinstance(row[0], str):
            return None

        try:
            payload = json.loads(row[0])
        except Exception:
            return None

        values = payload.get("Value")
        if not isinstance(values, list) or not values:
            return None

        entry: Optional[Dict[str, Any]] = None
        for v in values:
            if not isinstance(v, dict):
                continue
            if _xuid_id_matches(v.get("Id"), xuid):
                entry = v
                break

        if entry is None:
            return None

        result = entry.get("Result")
        if not isinstance(result, dict):
            return None

        team_id = result.get("TeamId")
        team_id_i = int(team_id) if isinstance(team_id, int) else None
        team_mmr = coerce_number(result.get("TeamMmr"))

        # MMRs par équipe (dict {"0": float, "1": float})
        team_mmrs_raw = result.get("TeamMmrs")
        team_mmrs: Dict[str, float] = {}
        if isinstance(team_mmrs_raw, dict):
            for k, v in team_mmrs_raw.items():
                fv = coerce_number(v)
                if fv is not None and isinstance(k, str):
                    team_mmrs[k] = float(fv)

        enemy_mmr: Optional[float] = None
        if team_id_i is not None and team_mmrs:
            my_key = str(team_id_i)
            for k, v in team_mmrs.items():
                if k != my_key:
                    enemy_mmr = float(v)
                    break

        statp = result.get("StatPerformances")
        kills = deaths = assists = None
        if isinstance(statp, dict):
            kills = statp.get("Kills") if isinstance(statp.get("Kills"), dict) else None
            deaths = statp.get("Deaths") if isinstance(statp.get("Deaths"), dict) else None
            assists = statp.get("Assists") if isinstance(statp.get("Assists"), dict) else None

        def _perf(d: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
            if not isinstance(d, dict):
                return {"count": None, "expected": None, "stddev": None}
            return {
                "count": coerce_number(d.get("Count")),
                "expected": coerce_number(d.get("Expected")),
                "stddev": coerce_number(d.get("StdDev")),
            }

        kills_p = _perf(kills)
        deaths_p = _perf(deaths)
        assists_p = _perf(assists)

        return {
            "team_id": team_id_i,
            "team_mmr": float(team_mmr) if team_mmr is not None else None,
            "enemy_mmr": enemy_mmr,
            "team_mmrs": team_mmrs if team_mmrs else None,
            "kills": kills_p,
            "deaths": deaths_p,
            "assists": assists_p,
        }


def load_match_medals_for_player(
    db_path: str,
    match_id: str,
    xuid: str,
) -> list[dict[str, int]]:
    """Retourne la liste des médailles (NameId/Count) du joueur sur un match.

    Source: table MatchStats (Players[].PlayerTeamStats[].Stats.CoreStats.Medals[])

    Returns:
        Liste de dicts: {"name_id": int, "count": int}
    """
    if not match_id or not xuid:
        return []

    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LOAD_MATCH_STATS_BY_MATCH_ID, (match_id,))
        row = cur.fetchone()
        if not row or not isinstance(row[0], str):
            return []

        try:
            payload = json.loads(row[0])
        except Exception:
            return []

    players = payload.get("Players")
    if not isinstance(players, list) or not players:
        return []

    me: Optional[Dict[str, Any]] = None
    for p in players:
        if not isinstance(p, dict):
            continue
        if _xuid_id_matches(p.get("PlayerId"), xuid):
            me = p
            break

    if me is None:
        return []

    pts = me.get("PlayerTeamStats")
    if not isinstance(pts, list) or not pts:
        return []

    totals: dict[int, int] = {}
    for ts in pts:
        if not isinstance(ts, dict):
            continue
        medals = (
            ts.get("Stats", {})
            .get("CoreStats", {})
            .get("Medals")
        )
        if not isinstance(medals, list):
            continue
        for m in medals:
            if not isinstance(m, dict):
                continue
            try:
                nid = int(m.get("NameId"))
                cnt = int(m.get("Count"))
            except Exception:
                continue
            totals[nid] = totals.get(nid, 0) + cnt

    out = [{"name_id": nid, "count": cnt} for nid, cnt in totals.items() if cnt]
    out.sort(key=lambda d: d["count"], reverse=True)
    return out


def load_asset_name_map(con: sqlite3.Connection, table: str) -> Dict[str, str]:
    """Charge la table de correspondance AssetId -> Nom.
    
    Args:
        con: Connexion SQLite ouverte.
        table: Nom de la table (Maps, Playlists, PlaylistMapModePairs).
        
    Returns:
        Dictionnaire {asset_id: nom}.
    """
    cur = con.cursor()
    try:
        cur.execute(f"SELECT ResponseBody FROM {table}")
    except sqlite3.OperationalError:
        return {}
    out: Dict[str, str] = {}
    for (body,) in cur.fetchall():
        try:
            obj = json.loads(body)
        except Exception:
            continue
        asset_id = obj.get("AssetId")
        name = obj.get("PublicName") or obj.get("Title")
        if isinstance(asset_id, str) and isinstance(name, str) and name.strip():
            out[asset_id] = name.strip()
    return out


def _find_player(players: List[Dict[str, Any]], xuid: str) -> Optional[Dict[str, Any]]:
    """Trouve un joueur dans la liste par son XUID."""
    for pl in players:
        pid = pl.get("PlayerId")
        if pid is None:
            continue
        if xuid in json.dumps(pid):
            return pl
    return None


def _find_player_core_stats_dict(player_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Trouve le dictionnaire contenant les stats Kills/Deaths/Assists."""
    targets = {"Kills", "Deaths", "Assists", "ShotsFired", "ShotsHit", "Accuracy"}

    def find_stats_dict(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict):
            if "Kills" in x and "Deaths" in x and any(k in x for k in targets):
                if coerce_number(x.get("Kills")) is not None or coerce_number(x.get("Deaths")) is not None:
                    return x
            for v in x.values():
                r = find_stats_dict(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for v in x:
                r = find_stats_dict(v)
                if r is not None:
                    return r
        return None

    return find_stats_dict(player_obj.get("PlayerTeamStats"))


def _extract_player_stats(player_obj: Dict[str, Any]) -> tuple[int, int, int, Optional[float]]:
    """Extrait kills, deaths, assists, accuracy d'un joueur."""
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is None:
        return 0, 0, 0, None

    kills = int(coerce_number(stats_dict.get("Kills")) or 0)
    deaths = int(coerce_number(stats_dict.get("Deaths")) or 0)
    assists = int(coerce_number(stats_dict.get("Assists")) or 0)
    accuracy = coerce_number(stats_dict.get("Accuracy"))
    return kills, deaths, assists, accuracy


def _extract_player_outcome_team(player_obj: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Extrait outcome et team_id d'un joueur."""
    outcome = player_obj.get("Outcome")
    last_team_id = player_obj.get("LastTeamId")
    outcome_i = int(outcome) if isinstance(outcome, int) else None
    team_i = int(last_team_id) if isinstance(last_team_id, int) else None
    return outcome_i, team_i


def _extract_player_kda(player_obj: Dict[str, Any]) -> Optional[float]:
    """Extrait le KDA d'un joueur."""
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is not None:
        v = coerce_number(stats_dict.get("KDA"))
        if v is not None:
            return v

    def find_kda(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            if "KDA" in x:
                v = coerce_number(x.get("KDA"))
                if v is not None:
                    return v
            for v in x.values():
                r = find_kda(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for v in x:
                r = find_kda(v)
                if r is not None:
                    return r
        return None

    return find_kda(player_obj.get("PlayerTeamStats"))


def _extract_player_spree_headshots(player_obj: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Extrait max_killing_spree et headshot_kills."""
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is None:
        return None, None
    spree = coerce_number(stats_dict.get("MaxKillingSpree"))
    headshots = coerce_number(stats_dict.get("HeadshotKills"))
    return (
        int(spree) if spree is not None else None,
        int(headshots) if headshots is not None else None,
    )


def _extract_player_average_life_seconds(player_obj: Dict[str, Any]) -> Optional[float]:
    """Extrait la durée de vie moyenne."""
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is not None:
        v = coerce_duration_seconds(stats_dict.get("AverageLifeDuration"))
        if v is not None:
            return v

    def find_avg_life(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            if "AverageLifeDuration" in x:
                v = coerce_duration_seconds(x.get("AverageLifeDuration"))
                if v is not None:
                    return v
            for v in x.values():
                r = find_avg_life(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for v in x:
                r = find_avg_life(v)
                if r is not None:
                    return r
        return None

    return find_avg_life(player_obj.get("PlayerTeamStats"))


def _extract_player_time_played_seconds(player_obj: Dict[str, Any]) -> Optional[float]:
    """Extrait le temps de jeu."""
    pi = player_obj.get("ParticipationInfo")
    if not isinstance(pi, dict):
        return None
    return coerce_duration_seconds(pi.get("TimePlayed"))


def _extract_team_scores(match_obj: Dict[str, Any], my_team_id: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    """Extrait les scores d'équipe depuis MatchStats.

    Dans les payloads OpenSpartan/Halo, le score de chaque équipe est typiquement
    présent dans: Teams[].Stats.CoreStats.Score.

    Returns:
        (my_team_score, enemy_team_score)
    """
    teams = match_obj.get("Teams")
    if not isinstance(teams, list) or not teams:
        return None, None

    team_scores: dict[int, int] = {}
    for t in teams:
        if not isinstance(t, dict):
            continue
        tid = t.get("TeamId")
        if not isinstance(tid, int):
            continue
        score_raw = (
            t.get("Stats", {})
            .get("CoreStats", {})
            .get("Score")
        )
        score = coerce_number(score_raw)
        if score is None:
            continue
        try:
            team_scores[int(tid)] = int(score)
        except Exception:
            continue

    if not team_scores:
        return None, None

    my_score: Optional[int] = None
    enemy_score: Optional[int] = None

    if my_team_id is not None:
        my_score = team_scores.get(int(my_team_id))
        for tid, sc in team_scores.items():
            if int(tid) != int(my_team_id):
                enemy_score = sc
                break

    return my_score, enemy_score


def _extract_player_mmrs(player: Dict[str, Any], my_team_id: Optional[int]) -> tuple[Optional[float], Optional[float]]:
    """Extrait les MMRs depuis les stats du joueur.

    Le payload contient typiquement:
    - PlayerTeamStats[].Stats.TeamMmr : MMR de l'équipe du joueur
    - PlayerTeamStats[].Stats.TeamMmrs : dict {"0": float, "1": float} avec MMR par équipe

    Returns:
        (team_mmr, enemy_mmr)
    """
    team_stats = player.get("PlayerTeamStats")
    if not isinstance(team_stats, list):
        return None, None

    team_mmr: Optional[float] = None
    enemy_mmr: Optional[float] = None

    for ts in team_stats:
        if not isinstance(ts, dict):
            continue
        stats = ts.get("Stats")
        if not isinstance(stats, dict):
            continue

        # MMR de l'équipe du joueur
        raw_team_mmr = stats.get("TeamMmr")
        if raw_team_mmr is not None:
            try:
                team_mmr = float(raw_team_mmr)
            except (TypeError, ValueError):
                pass

        # MMRs par équipe (pour calculer l'ennemi)
        team_mmrs_raw = stats.get("TeamMmrs")
        if isinstance(team_mmrs_raw, dict) and my_team_id is not None:
            my_key = str(my_team_id)
            for k, v in team_mmrs_raw.items():
                if k != my_key:
                    try:
                        enemy_mmr = float(v)
                        break
                    except (TypeError, ValueError):
                        continue

        # On a trouvé les données, on peut sortir
        if team_mmr is not None or enemy_mmr is not None:
            break

    return team_mmr, enemy_mmr


def load_matches(
    db_path: str,
    xuid: str,
    *,
    playlist_filter: Optional[str] = None,
    map_mode_pair_filter: Optional[str] = None,
    map_filter: Optional[str] = None,
    game_variant_filter: Optional[str] = None,
) -> List[MatchRow]:
    """Charge tous les matchs d'un joueur depuis la DB.
    
    Args:
        db_path: Chemin vers le fichier .db.
        xuid: XUID du joueur.
        playlist_filter: Filtre optionnel sur playlist_id.
        map_mode_pair_filter: Filtre optionnel sur map_mode_pair_id.
        map_filter: Filtre optionnel sur map_id.
        
    Returns:
        Liste de MatchRow triée par date croissante.
    """
    with get_connection(db_path) as con:
        map_names = load_asset_name_map(con, "Maps")
        playlist_names = load_asset_name_map(con, "Playlists")
        map_mode_pair_names = load_asset_name_map(con, "PlaylistMapModePairs")
        game_variant_names = load_asset_name_map(con, "GameVariants")

        cur = con.cursor()
        cur.execute(queries.LOAD_MATCH_STATS)

        rows: List[MatchRow] = []
        for (body,) in cur.fetchall():
            try:
                obj = json.loads(body)
            except Exception:
                continue

            match_id = obj.get("MatchId")
            if not isinstance(match_id, str):
                continue

            match_info = obj.get("MatchInfo")
            if not isinstance(match_info, dict):
                continue
            start_time_raw = match_info.get("StartTime")
            if not isinstance(start_time_raw, str):
                continue
            start_time = parse_iso_utc(start_time_raw)

            playlist_id = None
            playlist_obj = match_info.get("Playlist")
            if isinstance(playlist_obj, dict):
                playlist_id = playlist_obj.get("AssetId")
            if not isinstance(playlist_id, str):
                playlist_id = None

            map_id = None
            map_variant = match_info.get("MapVariant")
            if isinstance(map_variant, dict):
                map_id = map_variant.get("AssetId")
            if not isinstance(map_id, str):
                map_id = None

            map_mode_pair_id = None
            pair_obj = match_info.get("PlaylistMapModePair")
            if isinstance(pair_obj, dict):
                map_mode_pair_id = pair_obj.get("AssetId")
            if not isinstance(map_mode_pair_id, str):
                map_mode_pair_id = None

            game_variant_id = None
            ugc_variant = match_info.get("UgcGameVariant")
            if isinstance(ugc_variant, dict):
                game_variant_id = ugc_variant.get("AssetId")
            if not isinstance(game_variant_id, str):
                game_variant_id = None

            # Applique les filtres
            if playlist_filter is not None and (playlist_id or "") != playlist_filter:
                continue
            if map_mode_pair_filter is not None and (map_mode_pair_id or "") != map_mode_pair_filter:
                continue
            if map_filter is not None and (map_id or "") != map_filter:
                continue
            if game_variant_filter is not None and (game_variant_id or "") != game_variant_filter:
                continue

            players = obj.get("Players")
            if not isinstance(players, list):
                continue

            me = _find_player(players, xuid)
            if me is None:
                continue

            kills, deaths, assists, accuracy = _extract_player_stats(me)
            outcome, last_team_id = _extract_player_outcome_team(me)
            kda = _extract_player_kda(me)
            max_spree, headshots = _extract_player_spree_headshots(me)
            avg_life = _extract_player_average_life_seconds(me)
            time_played = _extract_player_time_played_seconds(me)

            my_team_score, enemy_team_score = _extract_team_scores(obj, last_team_id)

            # Extraire les MMRs
            team_mmr, enemy_mmr = _extract_player_mmrs(me, last_team_id)

            # Fallback important pour les DB générées sans import d'assets (SPNKr --no-assets).
            # Sans ça, playlist/pair/map sont None => filtres UI vides.
            playlist_name = playlist_names.get(playlist_id) if playlist_id else None
            if playlist_name is None and playlist_id:
                playlist_name = playlist_id

            pair_name = map_mode_pair_names.get(map_mode_pair_id) if map_mode_pair_id else None
            if pair_name is None and map_mode_pair_id:
                pair_name = map_mode_pair_id

            map_name = map_names.get(map_id) if map_id else None
            if map_name is None and map_id:
                map_name = map_id

            game_variant_name = game_variant_names.get(game_variant_id) if game_variant_id else None
            if game_variant_name is None and game_variant_id:
                game_variant_name = game_variant_id

            rows.append(
                MatchRow(
                    match_id=match_id,
                    start_time=start_time,
                    map_id=map_id,
                    map_name=map_name,
                    playlist_id=playlist_id,
                    playlist_name=playlist_name,
                    map_mode_pair_id=map_mode_pair_id,
                    map_mode_pair_name=pair_name,
                    game_variant_id=game_variant_id,
                    game_variant_name=game_variant_name,
                    outcome=outcome,
                    last_team_id=last_team_id,
                    kda=kda,
                    max_killing_spree=max_spree,
                    headshot_kills=headshots,
                    average_life_seconds=avg_life,
                    time_played_seconds=time_played,
                    kills=kills,
                    deaths=deaths,
                    assists=assists,
                    accuracy=accuracy,

                    my_team_score=my_team_score,
                    enemy_team_score=enemy_team_score,
                    team_mmr=team_mmr,
                    enemy_mmr=enemy_mmr,
                )
            )

        rows.sort(key=lambda r: r.start_time)
        return rows


def query_matches_with_friend(
    db_path: str,
    self_xuid: str,
    friend_xuid: str,
) -> List[FriendMatch]:
    """Retourne les matchs partagés avec un autre joueur.
    
    Args:
        db_path: Chemin vers le fichier .db.
        self_xuid: XUID du joueur principal.
        friend_xuid: XUID de l'ami.
        
    Returns:
        Liste de FriendMatch triée par date décroissante.
    """
    with get_connection(db_path) as con:
        playlist_names = load_asset_name_map(con, "Playlists")
        map_mode_pair_names = load_asset_name_map(con, "PlaylistMapModePairs")

        cur = con.cursor()
        me_id = f"xuid({self_xuid})"
        fr_id = f"xuid({friend_xuid})"
        cur.execute(queries.QUERY_MATCHES_WITH_FRIEND, (me_id, fr_id))
        
        out: List[FriendMatch] = []
        for row in cur.fetchall():
            match_id, start_time_raw, playlist_id, pair_id, my_team, my_out, fr_team, fr_out, same_team = row
            if not isinstance(match_id, str) or not isinstance(start_time_raw, str):
                continue
            start_time = parse_iso_utc(start_time_raw)
            out.append(
                FriendMatch(
                    match_id=match_id,
                    start_time=start_time,
                    playlist_id=playlist_id,
                    playlist_name=playlist_names.get(playlist_id),
                    pair_id=pair_id,
                    pair_name=map_mode_pair_names.get(pair_id),
                    my_team_id=my_team,
                    my_outcome=my_out,
                    friend_team_id=fr_team,
                    friend_outcome=fr_out,
                    same_team=bool(same_team),
                )
            )
        return out


def list_other_player_xuids(db_path: str, self_xuid: str, limit: int = 500) -> List[str]:
    """Liste les XUID des autres joueurs rencontrés.
    
    Args:
        db_path: Chemin vers le fichier .db.
        self_xuid: XUID du joueur principal (à exclure).
        limit: Nombre maximum de résultats.
        
    Returns:
        Liste de XUID (chaînes numériques).
    """
    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LIST_OTHER_PLAYER_XUIDS, (limit,))
        xuids: set[str] = set()
        for (pid,) in cur.fetchall():
            if not isinstance(pid, str):
                continue
            if pid == f"xuid({self_xuid})":
                continue
            m = re.fullmatch(r"xuid\((\d+)\)", pid)
            if m:
                xuids.add(m.group(1))
        return sorted(xuids)


def list_top_teammates(db_path: str, self_xuid: str, limit: int = 20) -> List[tuple[str, int]]:
    """Liste les coéquipiers les plus fréquents.
    
    Args:
        db_path: Chemin vers le fichier .db.
        self_xuid: XUID du joueur principal.
        limit: Nombre maximum de résultats.
        
    Returns:
        Liste de tuples (xuid, nombre_de_matchs) triée par fréquence.
    """
    me_id = f"xuid({self_xuid})"
    with get_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(queries.LIST_TOP_TEAMMATES, (me_id, me_id, int(limit)))
        out: List[tuple[str, int]] = []
        for pid, matches in cur.fetchall():
            if not isinstance(pid, str):
                continue
            m = re.fullmatch(r"xuid\((\d+)\)", pid)
            if not m:
                continue
            out.append((m.group(1), int(matches)))
        return out


def get_sync_metadata(db_path: str) -> Dict[str, Any]:
    """Récupère les métadonnées de synchronisation depuis la table SyncMeta.
    
    Args:
        db_path: Chemin vers le fichier .db.
        
    Returns:
        Dictionnaire contenant:
        - last_sync_at: datetime du dernier sync (ou None)
        - last_match_time: datetime du dernier match importé (ou None)
        - total_matches: nombre total de matchs (ou 0)
        - player_xuid: XUID du joueur principal (ou None)
    """
    result: Dict[str, Any] = {
        "last_sync_at": None,
        "last_match_time": None,
        "total_matches": 0,
        "player_xuid": None,
    }
    
    try:
        with get_connection(db_path) as con:
            cur = con.cursor()
            
            # Vérifier si la table existe
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='SyncMeta'"
            )
            if not cur.fetchone():
                # Fallback: compter les matchs depuis MatchStats
                cur.execute("SELECT COUNT(*) FROM MatchStats")
                row = cur.fetchone()
                result["total_matches"] = row[0] if row else 0
                return result
            
            # Récupérer toutes les métadonnées
            cur.execute("SELECT Key, Value, UpdatedAt FROM SyncMeta")
            for key, value, updated_at in cur.fetchall():
                if key == "last_sync_at" and value:
                    result["last_sync_at"] = parse_iso_utc(value)
                elif key == "last_match_time" and value:
                    result["last_match_time"] = parse_iso_utc(value)
                elif key == "total_matches" and value:
                    try:
                        result["total_matches"] = int(value)
                    except (ValueError, TypeError):
                        pass
                elif key == "player_xuid" and value:
                    result["player_xuid"] = str(value).strip()
            
            # Si total_matches n'est pas dans SyncMeta, compter depuis MatchStats
            if result["total_matches"] == 0:
                cur.execute("SELECT COUNT(*) FROM MatchStats")
                row = cur.fetchone()
                result["total_matches"] = row[0] if row else 0
                
    except Exception:
        pass
    
    return result
