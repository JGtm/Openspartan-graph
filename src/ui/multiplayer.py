"""Gestion du sélecteur multi-joueurs pour les DBs fusionnées.

Ce module fournit les fonctions pour :
- Détecter si une DB est multi-joueurs (table Players)
- Lister les joueurs disponibles
- Afficher un sélecteur dans la sidebar
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    pass


@dataclass
class PlayerInfo:
    """Informations sur un joueur dans une DB multi-joueurs."""
    xuid: str
    gamertag: str | None
    label: str | None
    total_matches: int
    first_match_date: str | None
    last_match_date: str | None
    
    @property
    def display_name(self) -> str:
        """Nom d'affichage pour le sélecteur."""
        if self.label:
            return self.label
        if self.gamertag:
            return self.gamertag
        return self.xuid[:15] + "…"
    
    @property
    def display_with_stats(self) -> str:
        """Nom d'affichage avec statistiques."""
        name = self.display_name
        if self.total_matches:
            return f"{name} ({self.total_matches} matchs)"
        return name


def is_multi_player_db(db_path: str) -> bool:
    """Vérifie si la DB contient une table Players (DB fusionnée)."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Players'"
        )
        result = cur.fetchone() is not None
        con.close()
        return result
    except Exception:
        return False


def list_players_in_db(db_path: str) -> list[PlayerInfo]:
    """Liste les joueurs disponibles dans une DB multi-joueurs.
    
    Returns:
        Liste triée par nombre de matchs (décroissant).
    """
    players: list[PlayerInfo] = []
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute("""
            SELECT xuid, gamertag, label, total_matches, 
                   first_match_date, last_match_date
            FROM Players
            ORDER BY total_matches DESC
        """)
        for row in cur.fetchall():
            players.append(PlayerInfo(
                xuid=row[0],
                gamertag=row[1],
                label=row[2],
                total_matches=row[3] or 0,
                first_match_date=row[4],
                last_match_date=row[5],
            ))
        con.close()
    except Exception:
        pass
    return players


def get_unique_xuids_from_matchstats(db_path: str) -> list[tuple[str, int]]:
    """Fallback : liste les XUIDs distincts depuis MatchStats.
    
    Utilisé si la table Players n'existe pas mais que la DB contient
    des matchs de plusieurs joueurs.
    
    Returns:
        Liste de (xuid, count) triée par count décroissant.
    """
    xuids: list[tuple[str, int]] = []
    try:
        con = sqlite3.connect(db_path)
        # Essayer d'abord MatchCache (plus rapide)
        try:
            cur = con.execute("""
                SELECT xuid, COUNT(*) as cnt
                FROM MatchCache
                GROUP BY xuid
                ORDER BY cnt DESC
            """)
            xuids = [(row[0], row[1]) for row in cur.fetchall()]
        except Exception:
            # Fallback sur MatchStats avec extraction JSON
            cur = con.execute("""
                SELECT DISTINCT XUID
                FROM MatchStats
            """)
            xuids = [(row[0], 0) for row in cur.fetchall() if row[0]]
        con.close()
    except Exception:
        pass
    return xuids


def render_player_selector(
    db_path: str,
    current_xuid: str,
    key: str = "player_selector",
) -> str | None:
    """Affiche un sélecteur de joueur si la DB est multi-joueurs.
    
    Args:
        db_path: Chemin vers la DB.
        current_xuid: XUID actuellement sélectionné.
        key: Clé Streamlit pour le widget.
        
    Returns:
        XUID sélectionné, ou None si pas de changement / pas multi-joueurs.
    """
    if not db_path or not is_multi_player_db(db_path):
        return None
    
    players = list_players_in_db(db_path)
    if len(players) <= 1:
        return None
    
    # Construire les options
    options = {p.xuid: p.display_with_stats for p in players}
    xuids = list(options.keys())
    labels = list(options.values())
    
    # Index actuel
    try:
        current_idx = xuids.index(current_xuid)
    except ValueError:
        current_idx = 0
    
    # Afficher le sélecteur
    st.markdown("#### 👥 Joueur")
    selected_label = st.selectbox(
        "Joueur",
        options=labels,
        index=current_idx,
        key=key,
        label_visibility="collapsed",
    )
    
    # Retrouver le XUID sélectionné
    try:
        selected_idx = labels.index(selected_label)
        selected_xuid = xuids[selected_idx]
    except (ValueError, IndexError):
        selected_xuid = current_xuid
    
    if selected_xuid != current_xuid:
        return selected_xuid
    
    return None


def get_player_display_name(db_path: str, xuid: str) -> str | None:
    """Récupère le nom d'affichage d'un joueur depuis la table Players."""
    if not db_path or not xuid:
        return None
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT label, gamertag FROM Players WHERE xuid = ?",
            (xuid,)
        )
        row = cur.fetchone()
        con.close()
        if row:
            return row[0] or row[1] or None
    except Exception:
        pass
    return None
