"""Tests pour les fonctions de parsing."""

from datetime import datetime, timezone

import pytest

from src.db.parsers import (
    guess_xuid_from_db_path,
    parse_iso_utc,
    coerce_number,
    coerce_duration_seconds,
    parse_xuid_input,
    resolve_xuid_from_db,
)


class TestResolveXuidFromDb:
    def test_returns_xuid_if_already_xuid(self, tmp_path):
        db_path = str(tmp_path / "dummy.db")
        assert resolve_xuid_from_db(db_path, "2533274823110022") == "2533274823110022"
        assert resolve_xuid_from_db(db_path, "xuid(2533274823110022)") == "2533274823110022"

    def test_resolves_from_matchstats_players(self, tmp_path):
        import json
        import sqlite3

        db_path = str(tmp_path / "test.db")
        con = sqlite3.connect(db_path)
        try:
            con.execute("CREATE TABLE MatchStats (ResponseBody TEXT)")
            payload = {
                "MatchId": "m1",
                "Players": [
                    {"PlayerId": {"Gamertag": "JGtm", "Xuid": "2533274823110022"}},
                    {"PlayerId": {"Gamertag": "Other", "Xuid": "2533270000000000"}},
                ],
            }
            con.execute("INSERT INTO MatchStats(ResponseBody) VALUES (?)", (json.dumps(payload),))
            con.commit()
        finally:
            con.close()

        assert resolve_xuid_from_db(db_path, "JGtm") == "2533274823110022"
        assert resolve_xuid_from_db(db_path, "jgtm") == "2533274823110022"

    def test_fallback_default_player(self, tmp_path, monkeypatch):
        # Même si la DB n'aide pas, on doit pouvoir résoudre via des defaults locaux.
        monkeypatch.setenv("OPENSPARTAN_DEFAULT_GAMERTAG", "JGtm")
        monkeypatch.setenv("OPENSPARTAN_DEFAULT_XUID", "2533274823110022")
        db_path = str(tmp_path / "empty.db")
        assert resolve_xuid_from_db(db_path, "JGtm") == "2533274823110022"


class TestGuessXuidFromDbPath:
    """Tests pour guess_xuid_from_db_path."""

    def test_valid_xuid(self):
        """Test avec un chemin valide."""
        assert guess_xuid_from_db_path("/path/to/2533274823110022.db") == "2533274823110022"
        assert guess_xuid_from_db_path("C:\\Users\\data\\2533274823110022.db") == "2533274823110022"

    def test_invalid_xuid(self):
        """Test avec un chemin invalide."""
        assert guess_xuid_from_db_path("/path/to/mydata.db") is None
        assert guess_xuid_from_db_path("/path/to/abc123.db") is None


class TestParseIsoUtc:
    """Tests pour parse_iso_utc."""

    def test_z_suffix(self):
        """Test avec suffixe Z."""
        result = parse_iso_utc("2026-01-02T20:18:01.293Z")
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 2
        assert result.hour == 20
        assert result.minute == 18
        assert result.tzinfo == timezone.utc

    def test_with_offset(self):
        """Test avec offset timezone."""
        result = parse_iso_utc("2026-01-02T20:18:01+00:00")
        assert result.tzinfo == timezone.utc


class TestCoerceNumber:
    """Tests pour coerce_number."""

    def test_int(self):
        """Test avec un entier."""
        assert coerce_number(42) == 42.0

    def test_float(self):
        """Test avec un float."""
        assert coerce_number(3.14) == 3.14

    def test_string(self):
        """Test avec une chaîne."""
        assert coerce_number("42.5") == 42.5

    def test_dict_count(self):
        """Test avec un dict contenant Count."""
        assert coerce_number({"Count": 19}) == 19.0

    def test_dict_value(self):
        """Test avec un dict contenant Value."""
        assert coerce_number({"Value": 3.14}) == 3.14

    def test_none(self):
        """Test avec None."""
        assert coerce_number(None) is None

    def test_bool(self):
        """Test avec un booléen."""
        assert coerce_number(True) is None
        assert coerce_number(False) is None

    def test_invalid_string(self):
        """Test avec une chaîne invalide."""
        assert coerce_number("not a number") is None


class TestCoerceDurationSeconds:
    """Tests pour coerce_duration_seconds."""

    def test_number(self):
        """Test avec un nombre direct."""
        assert coerce_duration_seconds(31.5) == 31.5

    def test_iso_duration(self):
        """Test avec durée ISO 8601."""
        assert coerce_duration_seconds("PT31.5S") == 31.5
        assert coerce_duration_seconds("PT1M30S") == 90.0
        assert coerce_duration_seconds("PT1H30M15S") == 5415.0

    def test_dict_milliseconds(self):
        """Test avec dict contenant Milliseconds."""
        assert coerce_duration_seconds({"Milliseconds": 31500}) == 31.5

    def test_dict_seconds(self):
        """Test avec dict contenant Seconds."""
        assert coerce_duration_seconds({"Seconds": 31.5}) == 31.5

    def test_none(self):
        """Test avec None."""
        assert coerce_duration_seconds(None) is None

    def test_invalid_string(self):
        """Test avec chaîne invalide."""
        assert coerce_duration_seconds("invalid") is None


class TestParseXuidInput:
    """Tests pour parse_xuid_input."""

    def test_numeric_string(self):
        """Test avec chaîne numérique."""
        assert parse_xuid_input("2533274823110022") == "2533274823110022"

    def test_xuid_format(self):
        """Test avec format xuid()."""
        assert parse_xuid_input("xuid(2533274823110022)") == "2533274823110022"

    def test_with_whitespace(self):
        """Test avec espaces."""
        assert parse_xuid_input("  2533274823110022  ") == "2533274823110022"

    def test_empty(self):
        """Test avec chaîne vide."""
        assert parse_xuid_input("") is None
        assert parse_xuid_input("   ") is None

    def test_invalid(self):
        """Test avec valeur invalide."""
        assert parse_xuid_input("abc123") is None
        assert parse_xuid_input("xuid(abc)") is None
