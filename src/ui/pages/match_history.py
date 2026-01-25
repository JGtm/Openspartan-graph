"""Page Historique des parties.

Tableau complet de l'historique des matchs avec liens et MMR.
"""

from __future__ import annotations

import html as html_lib
from typing import Optional

import pandas as pd
import streamlit as st

from src.analysis.stats import format_mmss
from src.analysis.performance_score import compute_performance_series
from src.ui.cache import cached_load_player_match_result
from src.ui.translations import translate_playlist_name
from src.ui.components.performance import get_score_class


def _normalize_mode_label(pair_name: str | None) -> str | None:
    """Normalise un pair_name en label UI."""
    from src.ui.translations import translate_pair_name
    return translate_pair_name(pair_name) if pair_name else None


def _format_datetime_fr_hm(dt: pd.Timestamp | None) -> str:
    """Formate une date FR avec heures/minutes."""
    if pd.isna(dt):
        return "-"
    try:
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(dt)


def _app_url(page: str, **params: str) -> str:
    """Génère une URL interne vers une page de l'app."""
    import urllib.parse
    base = "/"
    qp = {"page": page, **params}
    return base + "?" + urllib.parse.urlencode(qp)


def _format_score_label(my_score: object, enemy_score: object) -> str:
    """Formate le score du match."""
    def _safe(v: object) -> str:
        if v is None:
            return "-"
        try:
            if v != v:  # NaN
                return "-"
        except Exception:
            pass
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)

    return f"{_safe(my_score)} - {_safe(enemy_score)}"


def _fmt(v) -> str:
    """Formate une valeur pour affichage."""
    if v is None:
        return "-"
    try:
        if v != v:  # NaN
            return "-"
    except Exception:
        pass
    s = str(v)
    return s if s.strip() else "-"


def _fmt_mmr_int(v) -> str:
    """Formate une valeur MMR en entier."""
    if v is None:
        return "-"
    try:
        if v != v:  # NaN
            return "-"
    except Exception:
        pass
    try:
        return str(int(round(float(v))))
    except Exception:
        return _fmt(v)


def render_match_history_page(
    dff: pd.DataFrame,
    waypoint_player: str,
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    df_full: pd.DataFrame | None = None,
) -> None:
    """Affiche la page Historique des parties.

    Args:
        dff: DataFrame filtré des matchs.
        waypoint_player: Nom Waypoint du joueur.
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        db_key: Clé de cache de la DB.
        df_full: DataFrame complet (non filtré) pour le calcul du score relatif.
    """
    st.subheader("Historique des parties")

    dff_table = dff.copy()
    if "playlist_fr" not in dff_table.columns:
        dff_table["playlist_fr"] = dff_table["playlist_name"].apply(translate_playlist_name)
    if "mode_ui" not in dff_table.columns:
        dff_table["mode_ui"] = dff_table["pair_name"].apply(_normalize_mode_label)
    dff_table["match_url"] = (
        "https://www.halowaypoint.com/halo-infinite/players/"
        + waypoint_player.strip()
        + "/matches/"
        + dff_table["match_id"].astype(str)
    )

    outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
    dff_table["outcome_label"] = dff_table["outcome"].map(outcome_map).fillna("-")

    dff_table["score"] = dff_table.apply(
        lambda r: _format_score_label(r.get("my_team_score"), r.get("enemy_team_score")), axis=1
    )

    # MMR équipe/adverse pour chaque match (source PlayerMatchStats).
    with st.spinner("Chargement des MMR (équipe/adverse)…"):
        def _mmr_tuple(match_id: str):
            pm = cached_load_player_match_result(db_path, str(match_id), xuid.strip(), db_key=db_key)
            if not isinstance(pm, dict):
                return (None, None)
            return (pm.get("team_mmr"), pm.get("enemy_mmr"))

        mmr_pairs = dff_table["match_id"].astype(str).apply(_mmr_tuple)
        dff_table["team_mmr"] = mmr_pairs.apply(lambda t: t[0])
        dff_table["enemy_mmr"] = mmr_pairs.apply(lambda t: t[1])
        dff_table["delta_mmr"] = dff_table.apply(
            lambda r: (float(r.get("team_mmr")) - float(r.get("enemy_mmr")))
            if (r.get("team_mmr") is not None and r.get("enemy_mmr") is not None)
            else None,
            axis=1,
        )

    dff_table["start_time_fr"] = dff_table["start_time"].apply(_format_datetime_fr_hm)
    dff_table["average_life_mmss"] = dff_table["average_life_seconds"].apply(lambda x: format_mmss(x))
    
    # Calcul de la note de performance RELATIVE (basée sur l'historique complet)
    history_df = df_full if df_full is not None else dff_table
    dff_table["performance"] = compute_performance_series(dff_table, history_df)
    dff_table["performance_display"] = dff_table["performance"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "-"
    )

    # Table HTML
    _render_history_table(dff_table)

    # Export CSV
    _render_csv_download(dff_table)


def _render_history_table(dff_table: pd.DataFrame) -> None:
    """Génère et affiche le tableau HTML de l'historique."""

    def _outcome_class(label: str) -> str:
        """Retourne la classe CSS pour un résultat."""
        v = str(label or "").strip().casefold()
        if v.startswith("victoire"):
            return "text-win"
        if v.startswith("défaite") or v.startswith("defaite"):
            return "text-loss"
        if v.startswith("égalité") or v.startswith("egalite"):
            return "text-tie"
        if v.startswith("non"):
            return "text-nf"
        return ""

    cols = [
        ("Match", "_app"),
        ("HaloWaypoint", "match_url"),
        ("Date de début", "start_time_fr"),
        ("Carte", "map_name"),
        ("Playlist", "playlist_fr"),
        ("Mode", "mode_ui"),
        ("Résultat", "outcome_label"),
        ("Score", "score"),
        ("Performance", "performance_display"),
        ("MMR équipe", "team_mmr"),
        ("MMR adverse", "enemy_mmr"),
        ("Écart MMR", "delta_mmr"),
        ("FDA", "kda"),
        ("Frags", "kills"),
        ("Morts", "deaths"),
        ("Spree (max)", "max_killing_spree"),
        ("Têtes", "headshot_kills"),
        ("Durée vie", "average_life_mmss"),
        ("Assists", "assists"),
        ("Précision", "accuracy"),
        ("Ratio", "ratio"),
    ]

    view = dff_table.sort_values("start_time", ascending=False).head(250).reset_index(drop=True)

    head = "".join(f"<th>{html_lib.escape(h)}</th>" for h, _ in cols)
    body_rows: list[str] = []
    for _, r in view.iterrows():
        mid = str(r.get("match_id") or "").strip()
        app = _app_url("Match", match_id=mid)
        match_link = f"<a href='{html_lib.escape(app)}' target='_self'>Ouvrir</a>" if mid else "-"
        hw = str(r.get("match_url") or "").strip()
        hw_link = f"<a href='{html_lib.escape(hw)}' target='_blank' rel='noopener'>Ouvrir</a>" if hw else "-"

        tds: list[str] = []
        for _h, key in cols:
            if key == "_app":
                tds.append(f"<td>{match_link}</td>")
            elif key == "match_url":
                tds.append(f"<td>{hw_link}</td>")
            elif key == "outcome_label":
                val = _fmt(r.get(key))
                css_class = _outcome_class(val)
                tds.append(f"<td class='{css_class}'>{html_lib.escape(val)}</td>")
            elif key == "performance_display":
                val = _fmt(r.get(key))
                perf_val = r.get("performance")
                css_class = get_score_class(perf_val)
                tds.append(f"<td class='{css_class}'>{html_lib.escape(val)}</td>")
            elif key in ("team_mmr", "enemy_mmr", "delta_mmr"):
                val = _fmt_mmr_int(r.get(key))
                tds.append(f"<td>{html_lib.escape(val)}</td>")
            else:
                val = _fmt(r.get(key))
                tds.append(f"<td>{html_lib.escape(val)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    st.markdown(
        "<div class='os-table-wrap'><table class='os-table'><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_csv_download(dff_table: pd.DataFrame) -> None:
    """Affiche le bouton de téléchargement CSV."""
    show_cols = [
        "match_url", "start_time_fr", "map_name", "playlist_fr", "mode_ui", "outcome_label", "score",
        "team_mmr", "enemy_mmr", "delta_mmr",
        "kda", "kills", "deaths", "max_killing_spree", "headshot_kills",
        "average_life_mmss", "assists", "accuracy", "ratio",
    ]
    table = dff_table[show_cols + ["start_time"]].sort_values("start_time", ascending=False).reset_index(drop=True)
    table = table[show_cols]

    csv_table = table.rename(columns={"start_time_fr": "Date de début"})
    csv = csv_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger CSV",
        data=csv,
        file_name="openspartan_matches.csv",
        mime="text/csv",
    )
