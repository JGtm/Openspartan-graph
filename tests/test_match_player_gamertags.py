"""Tests pour le mapping XUID -> Gamertag basé sur MatchStats."""

import json
import sqlite3

from src.db.loaders import load_match_player_gamertags


def test_load_match_player_gamertags_from_matchstats(tmp_path):
    db_path = str(tmp_path / "test.db")
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE MatchStats (ResponseBody TEXT)")
        payload = {
            "MatchId": "m1",
            "Players": [
                {"PlayerId": {"Gamertag": "FranÃ§ois", "Xuid": "2533270000000001"}},
                {"PlayerId": {"Gamertag": "Other", "Xuid": "2533270000000002"}},
            ],
        }
        con.execute("INSERT INTO MatchStats(ResponseBody) VALUES (?)", (json.dumps(payload, ensure_ascii=False),))
        con.commit()
    finally:
        con.close()

    m = load_match_player_gamertags(db_path, "m1")
    assert m["2533270000000001"] == "François"
    assert m["2533270000000002"] == "Other"
