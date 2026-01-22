# -*- coding: utf-8 -*-
"""Fonctions de formatage pour l'interface utilisateur.

Ce module centralise les utilitaires de formatage :
- Dates en français
- Durées (mm:ss, hh:mm:ss, jours)
- Scores et styles
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import HALO_COLORS

__all__ = [
    "format_date_fr",
    "format_mmss",
    "format_duration_hms",
    "format_duration_dhm",
    "format_datetime_fr_hm",
    "format_score_label",
    "score_css_color",
    "style_outcome_text",
    "style_signed_number",
    "style_score_label",
    "parse_date_fr_input",
    "coerce_int",
    "to_paris_naive",
    "paris_epoch_seconds",
    "PARIS_TZ",
    "PARIS_TZ_NAME",
]

# Timezone Paris pour les conversions
PARIS_TZ_NAME = "Europe/Paris"
PARIS_TZ = ZoneInfo(PARIS_TZ_NAME)

_SCORE_LABEL_RE = re.compile(r"^\s*(-?\d+)\s*[-–—]\s*(-?\d+)\s*$")
_DATE_FR_RE = re.compile(r"^\s*(\d{1,2})\s*[\-/]\s*(\d{1,2})\s*[\-/]\s*(\d{4})\s*$")


def to_paris_naive(dt_value) -> datetime | None:
    """Convertit une date en datetime naïf (sans tzinfo) en heure de Paris.

    - tz-aware -> convertit en Europe/Paris puis enlève tzinfo
    - naïf -> supposé déjà en heure de Paris
    """
    if dt_value is None:
        return None
    try:
        ts = pd.to_datetime(dt_value, errors="coerce")
        if pd.isna(ts):
            return None

        try:
            if getattr(ts, "tz", None) is not None:
                ts = ts.tz_convert(PARIS_TZ_NAME).tz_localize(None)
        except Exception:
            pass

        d = ts.to_pydatetime()
        if getattr(d, "tzinfo", None) is not None:
            d = d.astimezone(PARIS_TZ).replace(tzinfo=None)
        return d
    except Exception:
        return None


def paris_epoch_seconds(dt_value) -> float | None:
    """Retourne un timestamp Unix (UTC) pour une date exprimée en heure de Paris."""
    d = to_paris_naive(dt_value)
    if d is None:
        return None
    try:
        aware = PARIS_TZ.localize(d) if d.tzinfo is None else d
        return aware.timestamp()
    except Exception:
        return None


def format_date_fr(dt_value) -> str:
    """Formate une date en français, ex: 'Lun. 4 décembre 2025'.

    Args:
        dt_value: Valeur de date (datetime, Timestamp, str, etc.).

    Returns:
        Date formatée en français ou "-" si invalide.
    """
    if dt_value is None:
        return "-"
    try:
        ts = pd.to_datetime(dt_value)
        if pd.isna(ts):
            return "-"
        d = ts.to_pydatetime()
    except Exception:
        return str(dt_value)

    jours = ["Lun.", "Mar.", "Mer.", "Jeu.", "Ven.", "Sam.", "Dim."]
    mois = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]

    return f"{jours[d.weekday()]} {d.day} {mois[d.month - 1]} {d.year}"


def format_mmss(seconds: float | int | None) -> str:
    """Formate une durée en mm:ss.

    Args:
        seconds: Durée en secondes.

    Returns:
        Chaîne formatée "mm:ss" ou "-" si invalide.
    """
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return "-"
    try:
        secs = int(seconds)
        if secs < 0:
            return "-"
        td = timedelta(seconds=secs)
        total_minutes = td.seconds // 60
        remaining_seconds = td.seconds % 60
        return f"{total_minutes:02d}:{remaining_seconds:02d}"
    except (ValueError, TypeError):
        return "-"


def format_duration_hms(seconds: float | int | None) -> str:
    """Formate une durée en hh:mm:ss ou mm:ss.

    Args:
        seconds: Durée en secondes.

    Returns:
        Chaîne formatée ou "-" si invalide.
    """
    if seconds is None or seconds != seconds:
        return "-"
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "-"
    if total < 0:
        return "-"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, hh = divmod(h, 24)
        return f"{d}j {hh:02d}:{m:02d}:{s:02d}"
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def format_duration_dhm(seconds: float | int | None) -> str:
    """Formate une durée en jours/heures/minutes.

    Args:
        seconds: Durée en secondes.

    Returns:
        Chaîne formatée (ex: "2j 5h 30min") ou "-" si invalide.
    """
    if seconds is None or seconds != seconds:
        return "-"
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "-"
    if total < 0:
        return "-"

    minutes, _s = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts: list[str] = []
    if days:
        parts.append(f"{days}j")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}min")
    return " ".join(parts)


def format_datetime_fr_hm(dt_value) -> str:
    """Formate une date avec heure en français (ex: 'Lun. 4 décembre 2025 14:30').

    Args:
        dt_value: Valeur de date.

    Returns:
        Date formatée avec heure ou "-" si invalide.
    """
    if dt_value is None:
        return "-"
    d = to_paris_naive(dt_value)
    if d is None:
        return "-"
    return f"{format_date_fr(d)} {d:%H:%M}"


def coerce_int(v) -> int | None:
    """Convertit une valeur en int si possible.

    Args:
        v: Valeur à convertir.

    Returns:
        Entier ou None si invalide.
    """
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = float(v)
        if x != x:
            return None
        return int(round(x))
    except Exception:
        return None


def format_score_label(my_team_score, enemy_team_score) -> str:
    """Formate le score d'un match (ex: '50 - 48').

    Args:
        my_team_score: Score de mon équipe.
        enemy_team_score: Score de l'équipe adverse.

    Returns:
        Score formaté ou "-" si invalide.
    """
    my_s = coerce_int(my_team_score)
    en_s = coerce_int(enemy_team_score)
    if my_s is None or en_s is None:
        return "-"
    return f"{my_s} - {en_s}"


def score_css_color(my_team_score, enemy_team_score) -> str:
    """Retourne la couleur CSS appropriée pour un score.

    Args:
        my_team_score: Score de mon équipe.
        enemy_team_score: Score de l'équipe adverse.

    Returns:
        Code couleur hex.
    """
    colors = HALO_COLORS.as_dict()
    my_s = coerce_int(my_team_score)
    en_s = coerce_int(enemy_team_score)
    if my_s is None or en_s is None:
        return colors["slate"]
    if my_s > en_s:
        return colors["green"]
    if my_s < en_s:
        return colors["red"]
    return colors["violet"]


def style_outcome_text(v: str) -> str:
    """Retourne le style CSS pour un résultat de match.

    Args:
        v: Texte du résultat (Victoire, Défaite, etc.).

    Returns:
        Chaîne de style CSS.
    """
    s = (v or "").strip().lower()
    if s == "victoire":
        return "color: #1B5E20; font-weight: 700;"
    if s in ("défaite", "defaite"):
        return "color: #B71C1C; font-weight: 700;"
    if s in ("égalité", "egalite"):
        return "color: #8E6CFF; font-weight: 700;"
    if s in ("non terminé", "non termine"):
        return "color: #8E6CFF; font-weight: 700;"
    return ""


def style_signed_number(v) -> str:
    """Retourne le style CSS pour un nombre signé.

    Args:
        v: Nombre.

    Returns:
        Chaîne de style CSS (vert si positif, rouge si négatif).
    """
    try:
        x = float(v)
    except Exception:
        return ""
    if x > 0:
        return "color: #1B5E20; font-weight: 700;"
    if x < 0:
        return "color: #B71C1C; font-weight: 700;"
    return "color: #424242;"


def style_score_label(v: str) -> str:
    """Retourne le style CSS pour un label de score.

    Args:
        v: Label de score (ex: "50 - 48").

    Returns:
        Chaîne de style CSS.
    """
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s == "-":
        return "color: #616161;"
    m = _SCORE_LABEL_RE.match(s)
    if not m:
        return ""
    try:
        my_s = int(m.group(1))
        en_s = int(m.group(2))
    except Exception:
        return ""
    if my_s > en_s:
        return "color: #1B5E20; font-weight: 800;"
    if my_s < en_s:
        return "color: #B71C1C; font-weight: 800;"
    return "color: #8E6CFF; font-weight: 800;"


def parse_date_fr_input(value: str | None, *, default_value: date) -> date:
    """Parse une date au format dd/mm/yyyy (ou dd-mm-yyyy).

    Args:
        value: Chaîne de date à parser.
        default_value: Valeur par défaut si le parsing échoue.

    Returns:
        Date parsée ou default_value si invalide.
    """
    s = (value or "").strip()
    if not s:
        return default_value
    m = _DATE_FR_RE.match(s)
    if not m:
        return default_value
    try:
        dd = int(m.group(1))
        mm = int(m.group(2))
        yyyy = int(m.group(3))
        return date(yyyy, mm, dd)
    except Exception:
        return default_value
