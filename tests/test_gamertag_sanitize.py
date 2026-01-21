"""Tests de normalisation des gamertags (encodage/chaînes corrompues)."""

import pytest

from src.db.loaders import _sanitize_gamertag


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("FranÃ§ois", "François"),
        ("Dâ€™Artagnan", "D’Artagnan"),
        ("Hello", "Hello"),
    ],
)
def test_sanitize_gamertag_fixes_mojibake(raw, expected):
    assert _sanitize_gamertag(raw) == expected


def test_sanitize_gamertag_strips_ctrl_chars():
    assert _sanitize_gamertag("A\x00B\x1fC") == ""


def test_sanitize_gamertag_rejects_embedded_nul_like_spnkr():
    assert _sanitize_gamertag("aba73\x00\x00\x00\u0103\x01") == ""
