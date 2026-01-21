"""Debug: trouve un match via une liste de gamertags, puis inspecte les HighlightEvents.

Usage:
  python scripts/debug_highlight_gamertags.py --db data/spnkr_gt_JGtm.db \
    --players "SouLsRipP,Chrisbaba73,got a bot9883,EROK KRUEL,GoingScissors25,JGtm,Alexa Nuggets,Fizzle3412" \
    --limit 500

Si le match est trouvé, affiche:
- MatchId + liste Players (MatchStats)
- pour chaque XUID vu dans HighlightEvents: valeurs brutes de gamertag rencontrées
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import re
from collections import defaultdict


def _sanitize_preview(value: object) -> str:
    """Aperçu de sanitation (debug) proche de l'app.

    - si présence de caractères de contrôle: invalide -> ""
    - sinon: trim + collapse spaces
    """
    if value is None:
        return ""
    s = str(value)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in s):
        return ""
    return " ".join(s.split()).strip()


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _ascii_token_from_weird_gamertag(value: object) -> str:
    """Extrait un token ASCII plausible depuis un champ 'gamertag' corrompu.

    Observé sur certaines DB SPNKr: la valeur contient des NUL + un bout de nom.
    Exemple: 'aba73\x00...\u0103\x01' -> 'aba73'
    """
    if value is None:
        return ""
    s = str(value)
    parts = _ASCII_TOKEN_RE.findall(s)
    if not parts:
        return ""
    parts.sort(key=len, reverse=True)
    return parts[0]


def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().split()).casefold()


_XUID_RE = re.compile(r"\b(\d{10,20})\b")


def _norm_xuid(value: object) -> str:
    s = str(value or "").strip()
    m = _XUID_RE.search(s)
    return m.group(1) if m else s


def _iter_recent_matchstats(con: sqlite3.Connection, limit: int):
    # rowid est pratique pour une notion de "récent" sur une DB append-only.
    cur = con.cursor()
    cur.execute("SELECT ResponseBody FROM MatchStats ORDER BY rowid DESC LIMIT ?", (int(limit),))
    for (body,) in cur.fetchall():
        if not isinstance(body, str) or not body:
            continue
        try:
            payload = json.loads(body)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def _extract_players(payload: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    players = payload.get("Players")
    if not isinstance(players, list):
        return out
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get("PlayerId")
        gt = None
        xu = None
        if isinstance(pid, dict):
            gt = pid.get("Gamertag") or pid.get("gamertag")
            xu = pid.get("Xuid") or pid.get("xuid")
        elif isinstance(pid, str):
            xu = pid
        # fallback
        if gt is None:
            gt = p.get("Gamertag") or p.get("gamertag")
        if xu is None:
            xu = p.get("Xuid") or p.get("xuid")
        gt_s = str(gt or "").strip()
        xu_s = str(xu or "").strip()
        if gt_s or xu_s:
            out.append((xu_s, gt_s))
    return out


def find_match_id(db_path: str, expected_players: list[str], limit: int) -> tuple[str, dict, list[tuple[str, str]]] | None:
    expected_norm = {_norm_name(p) for p in expected_players if _norm_name(p)}
    if not expected_norm:
        return None

    con = sqlite3.connect(db_path)
    try:
        for payload in _iter_recent_matchstats(con, limit=limit):
            mid = payload.get("MatchId")
            if not isinstance(mid, str) or not mid.strip():
                continue
            plist = _extract_players(payload)
            names = {_norm_name(gt) for _, gt in plist if _norm_name(gt)}

            # on veut que tous les players fournis soient présents (au moins par gamertag)
            if expected_norm.issubset(names):
                return mid.strip(), payload, plist
        return None
    finally:
        con.close()


def load_highlight_events(db_path: str, match_id: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT ResponseBody FROM HighlightEvents WHERE MatchId = ?", (match_id,))
        rows = cur.fetchall()
    finally:
        con.close()

    out: list[dict] = []
    for (body,) in rows:
        if not isinstance(body, str) or not body:
            continue
        try:
            obj = json.loads(body)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def main() -> int:
    # Évite les crashs UnicodeEncodeError sur consoles Windows (cp1252).
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--players", default=None, help="CSV gamertags")
    ap.add_argument("--roster", default=None, help="Alias de --players (CSV gamertags)")
    ap.add_argument("--write-aliases", action="store_true", help="Écrit/merge les alias dans xuid_aliases.json")
    ap.add_argument("--aliases-path", default="xuid_aliases.json")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    if args.match_id:
        match_id = str(args.match_id).strip()
        if not match_id:
            print("MatchId vide.")
            return 2
        # Charge MatchStats pour lister les PlayerId
        con = sqlite3.connect(args.db)
        try:
            cur = con.cursor()
            cur.execute("SELECT ResponseBody FROM MatchStats WHERE json_extract(ResponseBody,'$.MatchId')=? LIMIT 1", (match_id,))
            row = cur.fetchone()
            if not row or not isinstance(row[0], str):
                payload = {}
                plist = []
            else:
                payload = json.loads(row[0])
                plist = _extract_players(payload)
        finally:
            con.close()
    else:
        roster_csv = args.players or args.roster
        if not roster_csv:
            print("Fournis --match-id ou --players.")
            return 2
        expected = [p.strip() for p in roster_csv.split(",") if p.strip()]
        found = find_match_id(args.db, expected, limit=args.limit)
        if not found:
            print("Match non trouvé dans les derniers MatchStats. Augmente --limit ou vérifie les noms.")
            return 2

        match_id, payload, plist = found
    print("MATCH_ID:", match_id)
    print("Players (MatchStats):")
    for xuid, gt in plist:
        print(f"  - {gt}    [{xuid}]")

    # HighlightEvents
    events = load_highlight_events(args.db, match_id)
    if not events:
        print("Aucun HighlightEvent pour ce match (table vide ou pas importée avec --with-highlight-events).")
        return 3

    by_xuid: dict[str, set[str]] = defaultdict(set)
    by_gt: dict[str, set[str]] = defaultdict(set)
    tokens_by_xuid: dict[str, set[str]] = defaultdict(set)

    for e in events:
        x = str(e.get("xuid") or "").strip()
        gt = str(e.get("gamertag") or "").strip()
        _ = _sanitize_preview(gt)
        tok = _ascii_token_from_weird_gamertag(e.get("gamertag"))
        if x:
            by_xuid[x].add(gt or "<empty>")
            if tok:
                tokens_by_xuid[x].add(tok)
        if gt:
            by_gt[gt].add(x or "<empty>")

    by_xuid_clean: dict[str, set[str]] = defaultdict(set)
    for e in events:
        x = str(e.get("xuid") or "").strip()
        gt = e.get("gamertag")
        s = _sanitize_preview(gt)
        if x:
            by_xuid_clean[x].add(s or "<empty>")

    print("\nHighlightEvents: gamertags uniques par XUID")
    for xuid, gts in sorted(by_xuid.items(), key=lambda kv: (kv[0])):
        if not xuid:
            continue
        shown = ", ".join(sorted(gts))
        print(f"  - {xuid}: {shown}")

    print("\nHighlightEvents: gamertags après sanitation par XUID")
    for xuid, gts in sorted(by_xuid_clean.items(), key=lambda kv: (kv[0])):
        if not xuid:
            continue
        shown = ", ".join(sorted(gts))
        print(f"  - {xuid}: {shown}")

    print("\nHighlightEvents: XUIDs uniques par gamertag brut")
    for gt, xs in sorted(by_gt.items(), key=lambda kv: (kv[0])):
        shown = ", ".join(sorted(xs))
        print(f"  - {gt}: {shown}")

    roster_csv = args.players or args.roster
    if roster_csv:
        roster = [p.strip() for p in roster_csv.split(",") if p.strip()]
        roster_norm = {p: _norm_name(p) for p in roster}

        print("\nDéduction XUID -> Gamertag (via tokens ASCII des HighlightEvents)")
        mapping: dict[str, str] = {}
        for xuid, toks in sorted(tokens_by_xuid.items(), key=lambda kv: kv[0]):
            xuid_n = _norm_xuid(xuid)
            tok_norms = {_norm_name(t) for t in toks if _norm_name(t)}
            candidates = []
            for p, pn in roster_norm.items():
                if any(pn.endswith(tn) or tn in pn for tn in tok_norms):
                    candidates.append(p)

            tok_s = ",".join(sorted(toks)) if toks else ""
            if len(candidates) == 1:
                mapping[xuid_n] = candidates[0]
                print(f"  - {xuid} <= {candidates[0]}   (tokens={tok_s})")
            elif len(candidates) == 0:
                print(f"  - {xuid} <= ?   (tokens={tok_s})")
            else:
                print(f"  - {xuid} <= AMBIGU   {candidates}   (tokens={tok_s})")

        # Complète si un seul joueur reste non assigné (cas fréquent: ton propre XUID)
        roster_remaining = [p for p in roster if p not in set(mapping.values())]
        xuids_in_matchstats = [_norm_xuid(xu) for (xu, _gt) in plist if _norm_xuid(xu)]
        xuid_remaining = [xu for xu in xuids_in_matchstats if xu not in mapping]
        if len(roster_remaining) == 1 and len(xuid_remaining) == 1:
            mapping[xuid_remaining[0]] = roster_remaining[0]
            print(f"  - {xuid_remaining[0]} <= {roster_remaining[0]}   (complété par élimination)")

        if args.write_aliases:
            try:
                with open(args.aliases_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}

            merged = {str(k).strip(): str(v).strip() for k, v in existing.items() if str(k).strip() and str(v).strip()}
            merged.update({k: v for k, v in mapping.items() if k and v})
            with open(args.aliases_path, "w", encoding="utf-8") as f:
                json.dump(dict(sorted(merged.items())), f, ensure_ascii=False, indent=2)
            print(f"\nOK: alias écrits dans {args.aliases_path} (ajouts={len(mapping)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
