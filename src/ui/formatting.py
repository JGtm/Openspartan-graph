# -*- coding: utf-8 -*-
"""Fonctions de formatage pour l'interface utilisateur.

Ce module centralise les utilitaires de formatage :
- Dates en français
- Durées (mm:ss)
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

__all__ = [
    "format_date_fr",
    "format_mmss",
]


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
