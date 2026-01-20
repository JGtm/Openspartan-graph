"""Analyse killer → victime à partir des highlight events (film).

Les highlight events (SPNKr) fournissent typiquement des events 'kill' et 'death'
avec un timestamp en ms depuis le début du match, mais sans lien direct
killer→victim. L'approche consiste à joindre:
- chaque kill event (t)
- avec un death event (t')
avec |t - t'| <= tolérance.

Référence: discussions den.dev / SPNKr (jointure kill/death ~ 5ms).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class KVPair:
    killer_xuid: str
    killer_gamertag: str
    victim_xuid: str
    victim_gamertag: str
    time_ms: int


def _coerce_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        return int(v)
    except Exception:
        return None


def _coerce_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _infer_event_type(event: dict[str, Any]) -> str | None:
    et = _coerce_str(event.get("event_type"))
    if et:
        return et.lower()

    # Fallback: type_hint (blog den.dev)
    th = _coerce_int(event.get("type_hint"))
    if th == 50:
        return "kill"
    if th == 20:
        return "death"
    if th == 10:
        return "mode"
    return None


def compute_killer_victim_pairs(
    events: Iterable[dict[str, Any]],
    *,
    tolerance_ms: int = 5,
) -> list[KVPair]:
    """Construit les paires killer→victim à partir des highlight events.

    Stratégie:
    - sépare les kills et deaths
    - trie les deaths par time_ms
    - pour chaque kill, cherche les deaths dans [t-tol, t+tol]
      et choisit le death le plus proche (en évitant de réutiliser le même death)

    Args:
        events: liste de dicts (un event par entrée)
        tolerance_ms: fenêtre de jointure en millisecondes

    Returns:
        Liste de KVPair (killer, victim, time_ms).
    """

    if tolerance_ms < 0:
        tolerance_ms = 0

    kills: list[tuple[int, dict[str, Any]]] = []
    deaths: list[tuple[int, dict[str, Any]]] = []

    for e in events:
        if not isinstance(e, dict):
            continue
        et = _infer_event_type(e)
        t = _coerce_int(e.get("time_ms"))
        if t is None:
            continue
        if et == "kill":
            kills.append((t, e))
        elif et == "death":
            deaths.append((t, e))

    if not kills or not deaths:
        return []

    kills.sort(key=lambda x: x[0])
    deaths.sort(key=lambda x: x[0])

    death_times = [t for t, _ in deaths]
    used_death_idx: set[int] = set()

    out: list[KVPair] = []

    for t_kill, kill_event in kills:
        lo = bisect_left(death_times, t_kill - tolerance_ms)
        hi = bisect_right(death_times, t_kill + tolerance_ms)
        if lo >= hi:
            continue

        best_idx: int | None = None
        best_delta: int | None = None
        for idx in range(lo, hi):
            if idx in used_death_idx:
                continue
            delta = abs(death_times[idx] - t_kill)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx is None:
            continue

        used_death_idx.add(best_idx)
        victim_event = deaths[best_idx][1]

        killer_xuid = _coerce_str(kill_event.get("xuid")) or ""
        victim_xuid = _coerce_str(victim_event.get("xuid")) or ""
        killer_gt = _coerce_str(kill_event.get("gamertag")) or killer_xuid or "?"
        victim_gt = _coerce_str(victim_event.get("gamertag")) or victim_xuid or "?"

        if not killer_xuid or not victim_xuid:
            # On garde quand même la paire si les gamertags existent.
            pass

        out.append(
            KVPair(
                killer_xuid=killer_xuid,
                killer_gamertag=killer_gt,
                victim_xuid=victim_xuid,
                victim_gamertag=victim_gt,
                time_ms=int(t_kill),
            )
        )

    return out


def killer_victim_counts_long(pairs: Iterable[KVPair]) -> pd.DataFrame:
    """Retourne un DF long: killer, victim, count (agrégé)."""

    counter = Counter((p.killer_xuid, p.killer_gamertag, p.victim_xuid, p.victim_gamertag) for p in pairs)
    rows = [
        {
            "killer_xuid": kx,
            "killer_gamertag": kgt,
            "victim_xuid": vx,
            "victim_gamertag": vgt,
            "count": int(cnt),
        }
        for (kx, kgt, vx, vgt), cnt in counter.items()
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["count", "killer_gamertag", "victim_gamertag"], ascending=[False, True, True])


def killer_victim_matrix(pairs: Iterable[KVPair]) -> pd.DataFrame:
    """Retourne un DF matrice: index=killer, colonnes=victim, valeurs=count."""

    df = killer_victim_counts_long(pairs)
    if df.empty:
        return df

    pivot = df.pivot_table(
        index="killer_gamertag",
        columns="victim_gamertag",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    # Tri stable: killers/victims les plus "actifs" d'abord
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
    return pivot
