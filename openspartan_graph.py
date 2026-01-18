import argparse
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class MatchRow:
    match_id: str
    start_time: datetime
    map_id: Optional[str]
    map_name: Optional[str]
    playlist_id: Optional[str]
    playlist_name: Optional[str]
    map_mode_pair_id: Optional[str]
    map_mode_pair_name: Optional[str]
    outcome: Optional[int]
    last_team_id: Optional[int]
    kda: Optional[float]
    max_killing_spree: Optional[int]
    headshot_kills: Optional[int]
    average_life_seconds: Optional[float]
    time_played_seconds: Optional[float]
    kills: int
    deaths: int
    assists: int
    accuracy: Optional[float]

    @property
    def ratio(self) -> float:
        # Ratio demandé: (Frags + assists/2) / morts
        # Si deaths==0 on retourne NaN pour éviter un inf non-plot-friendly.
        denom = self.deaths
        if denom <= 0:
            return float("nan")
        return (self.kills + (self.assists / 2.0)) / denom


def _parse_iso_utc(s: str) -> datetime:
    # Exemple dans la DB: 2026-01-02T20:18:01.293Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _guess_xuid_from_db_path(db_path: str) -> Optional[str]:
    base = os.path.basename(db_path)
    stem, _ = os.path.splitext(base)
    return stem if stem.isdigit() else None


def _load_asset_name_map(con: sqlite3.Connection, table: str) -> Dict[str, str]:
    cur = con.cursor()
    cur.execute(f"SELECT ResponseBody FROM {table}")
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
    for pl in players:
        pid = pl.get("PlayerId")
        if pid is None:
            continue
        # Le PlayerId est souvent une structure, on fait un match robuste via JSON.
        if xuid in json.dumps(pid):
            return pl
    return None


def _get_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    v = d.get(key, default)
    n = _coerce_number(v)
    if n is None:
        return default
    return int(n)


def _get_float_opt(d: Dict[str, Any], key: str) -> Optional[float]:
    return _coerce_number(d.get(key))


def _coerce_number(v: Any) -> Optional[float]:
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
                return _coerce_number(v.get(k))
    return None


_ISO8601_DURATION_RE = re.compile(
    r"^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$"
)


def _coerce_duration_seconds(v: Any) -> Optional[float]:
    # Durées vues dans la DB: 'PT31.5S'
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        # Parfois structuré, ex: {"Seconds": 31.5} ou {"Milliseconds": 31500}
        if "Milliseconds" in v or "Ms" in v:
            ms = _coerce_number(v.get("Milliseconds") if "Milliseconds" in v else v.get("Ms"))
            return (ms / 1000.0) if ms is not None else None
        if "Seconds" in v:
            return _coerce_number(v.get("Seconds"))
        # fallback
        return _coerce_number(v)
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


def _extract_player_average_life_seconds(player_obj: Dict[str, Any]) -> Optional[float]:
    # AverageLifeDuration est généralement dans CoreStats, mais la structure varie.
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is not None:
        v = _coerce_duration_seconds(stats_dict.get("AverageLifeDuration"))
        if v is not None:
            return v

    def find_avg_life(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            if "AverageLifeDuration" in x:
                v = _coerce_duration_seconds(x.get("AverageLifeDuration"))
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
    pi = player_obj.get("ParticipationInfo")
    if not isinstance(pi, dict):
        return None
    return _coerce_duration_seconds(pi.get("TimePlayed"))


def _extract_player_stats(player_obj: Dict[str, Any]) -> Tuple[int, int, int, Optional[float]]:
    # Le format varie selon les modes/saisons:
    # - ancien: PlayerTeamStats est un dict avec Kills/Deaths/Assists/Accuracy
    # - récent: PlayerTeamStats est une liste [{TeamId, Stats:{CoreStats:{...}}}, ...]

    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is None:
        return 0, 0, 0, None

    kills = int(_coerce_number(stats_dict.get("Kills")) or 0)
    deaths = int(_coerce_number(stats_dict.get("Deaths")) or 0)
    assists = int(_coerce_number(stats_dict.get("Assists")) or 0)
    accuracy = _coerce_number(stats_dict.get("Accuracy"))
    return kills, deaths, assists, accuracy


def _find_player_core_stats_dict(player_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Retourne le dict qui contient réellement Kills/Deaths/Assists/Accuracy.
    targets = {"Kills", "Deaths", "Assists", "ShotsFired", "ShotsHit", "Accuracy"}

    def find_stats_dict(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict):
            if "Kills" in x and "Deaths" in x and any(k in x for k in targets):
                # Vérifie que c'est bien numérique (pas un placeholder)
                if _coerce_number(x.get("Kills")) is not None or _coerce_number(x.get("Deaths")) is not None:
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


def _extract_player_outcome_team(player_obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    outcome = player_obj.get("Outcome")
    last_team_id = player_obj.get("LastTeamId")
    outcome_i = int(outcome) if isinstance(outcome, int) else None
    team_i = int(last_team_id) if isinstance(last_team_id, int) else None
    return outcome_i, team_i


def _extract_player_kda(player_obj: Dict[str, Any]) -> Optional[float]:
    # KDA se trouve soit directement à côté de Kills/Deaths dans le dict final,
    # soit dans un sous-objet (CoreStats).
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is not None:
        v = _coerce_number(stats_dict.get("KDA"))
        if v is not None:
            return v

    def find_kda(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            if "KDA" in x:
                v = _coerce_number(x.get("KDA"))
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


def _extract_player_spree_headshots(player_obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    stats_dict = _find_player_core_stats_dict(player_obj)
    if stats_dict is None:
        return None, None
    spree = _coerce_number(stats_dict.get("MaxKillingSpree"))
    headshots = _coerce_number(stats_dict.get("HeadshotKills"))
    return (int(spree) if spree is not None else None, int(headshots) if headshots is not None else None)


def load_matches(
    db_path: str,
    xuid: str,
    *,
    playlist_filter: Optional[str] = None,
    map_mode_pair_filter: Optional[str] = None,
    map_filter: Optional[str] = None,
) -> List[MatchRow]:
    con = sqlite3.connect(db_path)
    try:
        map_names = _load_asset_name_map(con, "Maps")
        playlist_names = _load_asset_name_map(con, "Playlists")
        map_mode_pair_names = _load_asset_name_map(con, "PlaylistMapModePairs")

        cur = con.cursor()
        cur.execute("SELECT ResponseBody FROM MatchStats")

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
            start_time = _parse_iso_utc(start_time_raw)

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

            if playlist_filter is not None and (playlist_id or "") != playlist_filter:
                continue
            if map_mode_pair_filter is not None and (map_mode_pair_id or "") != map_mode_pair_filter:
                continue
            if map_filter is not None and (map_id or "") != map_filter:
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

            playlist_name = playlist_names.get(playlist_id) if playlist_id else None
            pair_name = map_mode_pair_names.get(map_mode_pair_id) if map_mode_pair_id else None
            map_name = map_names.get(map_id) if map_id else None

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
                )
            )

        rows.sort(key=lambda r: r.start_time)
        return rows
    finally:
        con.close()


def plot_kills_deaths_ratio(
    matches: List[MatchRow],
    out_path: str,
    *,
    title: str,
    last_n: Optional[int] = None,
) -> None:
    if last_n is not None:
        matches = matches[-last_n:]

    if not matches:
        raise SystemExit("Aucun match trouvé (filtre trop strict ?) ")

    try:
        import matplotlib

        # Important sur Windows: éviter Tk/Tcl (souvent absent ou en conflit).
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib est requis. Installe-le avec: pip install matplotlib\n"
            f"Détail: {e}"
        )

    x = [m.start_time for m in matches]
    kills = [m.kills for m in matches]
    deaths = [m.deaths for m in matches]
    ratio = [m.ratio for m in matches]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title)

    axes[0].plot(x, kills, label="Kills", color="#2E86AB")
    axes[0].set_ylabel("Kills")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, deaths, label="Deaths", color="#D1495B")
    axes[1].set_ylabel("Deaths")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, ratio, label="Ratio (K + A/2) / D", color="#3A7D44")
    axes[2].set_ylabel("Ratio")
    axes[2].grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Génère des graphes à partir de la DB OpenSpartan (SQLite).\n"
            "V1: kills/morts/ratio dans le temps."
        )
    )
    ap.add_argument(
        "--db",
        required=True,
        help="Chemin vers le fichier .db (SQLite)",
    )
    ap.add_argument(
        "--xuid",
        default=None,
        help="Ton XUID (par défaut: déduit du nom du .db si possible)",
    )
    ap.add_argument(
        "--last",
        type=int,
        default=None,
        help="Ne garder que les N derniers matchs (après tri par date)",
    )
    ap.add_argument(
        "--playlist-id",
        default=None,
        help="Filtre sur MatchInfo.Playlist.AssetId (UUID)",
    )
    ap.add_argument(
        "--pair-id",
        default=None,
        help="Filtre sur MatchInfo.PlaylistMapModePair.AssetId (UUID)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join("out", "kills_deaths_ratio.png"),
        help="Chemin de sortie PNG",
    )

    args = ap.parse_args()

    db_path = args.db
    if args.xuid is None:
        guessed = _guess_xuid_from_db_path(db_path)
        if guessed is None:
            raise SystemExit("Impossible de deviner le XUID. Passe --xuid.")
        xuid = guessed
    else:
        xuid = str(args.xuid)

    matches = load_matches(
        db_path,
        xuid,
        playlist_filter=args.playlist_id,
        map_mode_pair_filter=args.pair_id,
    )

    title_parts = ["OpenSpartan", f"XUID {xuid}"]
    if args.playlist_id:
        title_parts.append(f"Playlist {args.playlist_id}")
    if args.pair_id:
        title_parts.append(f"Pair {args.pair_id}")
    title = " — ".join(title_parts)

    plot_kills_deaths_ratio(matches, args.out, title=title, last_n=args.last)
    print(f"OK: {args.out} ({len(matches)} matchs, last={args.last})")
    return 0


def query_matches_with_friend(
        db_path: str,
        self_xuid: str,
        friend_xuid: str,
) -> List[Dict[str, Any]]:
        """Retourne les matchs partagés avec un autre joueur.

        Note: la DB locale ne contient pas de gamertags, uniquement des PlayerId de type "xuid(123)".
        """
        con = sqlite3.connect(db_path)
        try:
                playlist_names = _load_asset_name_map(con, "Playlists")
                map_mode_pair_names = _load_asset_name_map(con, "PlaylistMapModePairs")

                cur = con.cursor()
                sql = """
                WITH base AS (
                    SELECT
                        json_extract(ResponseBody, '$.MatchId') AS MatchId,
                        json_extract(ResponseBody, '$.MatchInfo.StartTime') AS StartTime,
                        json_extract(ResponseBody, '$.MatchInfo.Playlist.AssetId') AS PlaylistId,
                        json_extract(ResponseBody, '$.MatchInfo.PlaylistMapModePair.AssetId') AS PairId,
                        json_extract(ResponseBody, '$.Players') AS Players
                    FROM MatchStats
                ),
                p AS (
                    SELECT
                        b.MatchId AS MatchId,
                        b.StartTime AS StartTime,
                        b.PlaylistId AS PlaylistId,
                        b.PairId AS PairId,
                        json_extract(j.value, '$.PlayerId') AS PlayerId,
                        CAST(json_extract(j.value, '$.LastTeamId') AS INTEGER) AS LastTeamId,
                        CAST(json_extract(j.value, '$.Outcome') AS INTEGER) AS Outcome
                    FROM base b
                    JOIN json_each(b.Players) AS j
                ),
                me AS (
                    SELECT * FROM p WHERE PlayerId = ?
                ),
                fr AS (
                    SELECT * FROM p WHERE PlayerId = ?
                )
                SELECT
                    me.MatchId,
                    me.StartTime,
                    me.PlaylistId,
                    me.PairId,
                    me.LastTeamId AS MyTeamId,
                    me.Outcome AS MyOutcome,
                    fr.LastTeamId AS FriendTeamId,
                    fr.Outcome AS FriendOutcome,
                    CASE WHEN me.LastTeamId = fr.LastTeamId THEN 1 ELSE 0 END AS SameTeam
                FROM me
                JOIN fr ON me.MatchId = fr.MatchId
                ORDER BY me.StartTime DESC;
                """
                me_id = f"xuid({self_xuid})"
                fr_id = f"xuid({friend_xuid})"
                cur.execute(sql, (me_id, fr_id))
                out: List[Dict[str, Any]] = []
                for match_id, start_time_raw, playlist_id, pair_id, my_team, my_out, fr_team, fr_out, same_team in cur.fetchall():
                        if not isinstance(match_id, str) or not isinstance(start_time_raw, str):
                                continue
                        start_time = _parse_iso_utc(start_time_raw)
                        out.append(
                                {
                                        "match_id": match_id,
                                        "start_time": start_time,
                                        "playlist_id": playlist_id,
                                        "playlist_name": playlist_names.get(playlist_id),
                                        "pair_id": pair_id,
                                        "pair_name": map_mode_pair_names.get(pair_id),
                                        "my_team_id": my_team,
                                        "my_outcome": my_out,
                                        "friend_team_id": fr_team,
                                        "friend_outcome": fr_out,
                                        "same_team": bool(same_team),
                                }
                        )
                return out
        finally:
                con.close()


if __name__ == "__main__":
    raise SystemExit(main())
