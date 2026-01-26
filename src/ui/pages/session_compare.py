"""Page de comparaison de sessions.

Cette page permet de comparer les performances entre deux sessions de jeu,
avec des graphiques radar, des barres comparatives, et un tableau historique.
"""

from __future__ import annotations

import locale
from typing import TYPE_CHECKING, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import translate_pair_name
from src.ui.components.performance import (
    compute_session_performance_score,
    compute_session_performance_score_v2_ui,
    render_performance_score_card,
    render_metric_comparison_row,
    get_score_class,
)
from src.analysis.performance_score import compute_performance_series
from src.analysis.performance_config import PERFORMANCE_SCORE_SHORT_DESC

if TYPE_CHECKING:
    pass


def _get_friends_names(df_session: pd.DataFrame) -> set[str]:
    """Récupère les noms/nicknames des amis présents dans une session.
    
    Utilise les aliases chargés en session_state si disponibles,
    sinon retourne les XUIDs tronqués.
    
    Args:
        df_session: DataFrame des matchs avec colonne friends_xuids.
        
    Returns:
        Set des noms d'amis (gamertags ou nicknames).
    """
    if df_session.empty or "friends_xuids" not in df_session.columns:
        return set()
    
    # Collecter tous les XUIDs des amis
    friends_xuids: set[str] = set()
    for friends_str in df_session["friends_xuids"].dropna():
        if friends_str:
            friends_xuids.update(friends_str.split(","))
    friends_xuids.discard("")
    
    if not friends_xuids:
        return set()
    
    # Charger les noms des amis depuis la DB si possible
    friends_mapping: dict[str, str] = {}
    
    # 1. Essayer depuis session_state (aliases chargés)
    aliases = st.session_state.get("xuid_aliases", {})
    
    # 2. Essayer de charger depuis la table Friends
    db_path = st.session_state.get("db_path")
    xuid = st.session_state.get("player_xuid")
    if db_path and xuid:
        try:
            from src.db.connection import get_connection
            with get_connection(db_path) as con:
                cur = con.cursor()
                cur.execute(
                    "SELECT friend_xuid, friend_gamertag, nickname FROM Friends WHERE owner_xuid = ?",
                    (xuid,)
                )
                for row in cur.fetchall():
                    fxuid, gamertag, nickname = row
                    # Priorité au nickname
                    friends_mapping[fxuid] = nickname or gamertag or fxuid
        except Exception:
            pass
    
    # Construire les noms
    names: set[str] = set()
    for xuid in friends_xuids:
        if xuid in friends_mapping:
            names.add(friends_mapping[xuid])
        elif xuid in aliases:
            names.add(aliases[xuid])
        else:
            # Garder le XUID tronqué comme fallback
            names.add(xuid[-6:] if len(xuid) > 6 else xuid)
    
    return names

def is_session_with_friends(df_session: pd.DataFrame) -> bool:
    """Détermine si une session est considérée comme 'avec amis'.
    
    Une session est avec amis si la majorité des matchs ont is_with_friends=True.
    
    Args:
        df_session: DataFrame des matchs d'une session.
        
    Returns:
        True si la majorité des matchs sont avec amis.
    """
    if df_session.empty:
        return False
    if "is_with_friends" not in df_session.columns:
        return False
    return df_session["is_with_friends"].sum() > len(df_session) / 2


def get_session_friends_signature(df_session: pd.DataFrame) -> set[str]:
    """Extrait l'ensemble des amis présents dans une session.
    
    Args:
        df_session: DataFrame des matchs d'une session.
        
    Returns:
        Set des XUIDs des amis présents (union de tous les matchs).
    """
    if df_session.empty or "friends_xuids" not in df_session.columns:
        return set()
    
    all_friends: set[str] = set()
    for friends_str in df_session["friends_xuids"].dropna():
        if friends_str:
            all_friends.update(friends_str.split(","))
    
    # Retirer les chaînes vides
    all_friends.discard("")
    return all_friends


def compute_similar_sessions_average(
    all_sessions_df: pd.DataFrame,
    is_with_friends: bool,
    exclude_session_ids: list[int] | None = None,
    same_friends_xuids: set[str] | None = None,
) -> dict:
    """Calcule la moyenne des sessions similaires.
    
    Modes de comparaison :
    - Si same_friends_xuids est fourni : compare avec sessions ayant les MÊMES amis
    - Sinon : compare solo vs avec amis (n'importe lesquels)
    
    Args:
        all_sessions_df: DataFrame avec toutes les sessions.
        is_with_friends: True pour sessions avec amis, False pour solo.
        exclude_session_ids: Session IDs à exclure du calcul.
        same_friends_xuids: Si fourni, ne matcher que les sessions avec ces amis exacts.
        
    Returns:
        Dict avec kd_ratio, win_rate, avg_life, session_count, friends_label.
    """
    if all_sessions_df.empty or "session_id" not in all_sessions_df.columns:
        return {}
    
    if "is_with_friends" not in all_sessions_df.columns:
        return {}
    
    exclude_ids = set(exclude_session_ids or [])
    
    # Filtrer les sessions exclues
    df = all_sessions_df[~all_sessions_df["session_id"].isin(exclude_ids)].copy()
    if df.empty:
        return {}
    
    # Mode "mêmes amis" : matcher les sessions avec exactement les mêmes amis
    if same_friends_xuids and len(same_friends_xuids) > 0:
        matching_session_ids = []
        for session_id, group in df.groupby("session_id"):
            session_friends = get_session_friends_signature(group)
            # Match si au moins les mêmes amis sont présents (peut avoir plus)
            if same_friends_xuids <= session_friends:
                matching_session_ids.append(session_id)
    else:
        # Mode "solo vs avec amis" classique
        session_friend_ratio = df.groupby("session_id")["is_with_friends"].mean()
        matching_session_ids = session_friend_ratio[
            (session_friend_ratio > 0.5) == is_with_friends
        ].index.tolist()
    
    if len(matching_session_ids) == 0:
        return {}
    
    # Filtrer les matchs des sessions correspondantes
    df_matching = df[df["session_id"].isin(matching_session_ids)]
    
    if df_matching.empty:
        return {}
    
    session_count = len(matching_session_ids)
    
    # Calculs agrégés directs sur le DataFrame (beaucoup plus rapide)
    total_kills = df_matching["kills"].sum()
    total_deaths = df_matching["deaths"].sum()
    total_assists = df_matching["assists"].sum()
    total_matches = len(df_matching)
    
    # K/D ratio moyen par session
    session_stats = df_matching.groupby("session_id").agg({
        "kills": "sum",
        "deaths": "sum",
        "assists": "sum",
        "outcome": lambda x: (x == 2).sum() / len(x) * 100 if len(x) > 0 else 0,
        "average_life_seconds": "mean",
    }).rename(columns={"outcome": "win_rate"})
    
    # Calculer K/D par session puis moyenne
    session_stats["kd_ratio"] = session_stats.apply(
        lambda r: r["kills"] / r["deaths"] if r["deaths"] > 0 else r["kills"],
        axis=1
    )
    
    avg_kd = session_stats["kd_ratio"].mean()
    avg_win_rate = session_stats["win_rate"].mean()
    avg_life = session_stats["average_life_seconds"].mean()
    
    return {
        "kd_ratio": avg_kd,
        "win_rate": avg_win_rate,
        "avg_life_seconds": avg_life,
        "kills_per_match": total_kills / total_matches if total_matches > 0 else 0,
        "deaths_per_match": total_deaths / total_matches if total_matches > 0 else 0,
        "assists_per_match": total_assists / total_matches if total_matches > 0 else 0,
        "session_count": session_count,
    }


def _format_seconds_to_mmss(seconds) -> str:
    """Formate des secondes en mm:ss ou retourne la valeur formatée."""
    if seconds is None:
        return "—"
    try:
        total = int(round(float(seconds)))
        if total < 0:
            return "—"
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"
    except Exception:
        return "—"


def _format_date_with_weekday(dt) -> str:
    """Formate une date avec jour de la semaine abrégé : lun. 12/01/26 14:30."""
    if dt is None or pd.isna(dt):
        return "-"
    try:
        # Jours en français
        weekdays_fr = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
        wd = weekdays_fr[dt.weekday()]
        return f"{wd} {dt.strftime('%d/%m/%y %H:%M')}"
    except Exception:
        return "-"


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


def render_session_history_table(
    df_sess: pd.DataFrame,
    session_name: str,
    df_full: pd.DataFrame | None = None,
) -> None:
    """Affiche le tableau historique d'une session.
    
    Args:
        df_sess: DataFrame de la session.
        session_name: Nom de la session pour les messages.
        df_full: DataFrame complet pour le calcul du score relatif.
    """
    if df_sess.empty:
        st.info(f"Aucune partie dans {session_name}.")
        return
    
    # Copie pour éviter les warnings pandas
    df_sess = df_sess.copy()
    
    # Traduire le mode si non traduit
    if "pair_fr" not in df_sess.columns and "pair_name" in df_sess.columns:
        df_sess["pair_fr"] = df_sess["pair_name"].apply(translate_pair_name)
    
    # Préparer les colonnes à afficher
    display_cols = []
    col_map = {}
    
    if "start_time" in df_sess.columns:
        df_sess["Heure"] = df_sess["start_time"].apply(_format_date_with_weekday)
        display_cols.append("Heure")
    
    if "pair_fr" in df_sess.columns:
        col_map["pair_fr"] = "Mode"
        display_cols.append("pair_fr")
    elif "pair_name" in df_sess.columns:
        # Traduire à la volée si pair_fr n'existe pas
        df_sess["mode_traduit"] = df_sess["pair_name"].apply(translate_pair_name)
        col_map["mode_traduit"] = "Mode"
        display_cols.append("mode_traduit")
    
    if "map_ui" in df_sess.columns:
        col_map["map_ui"] = "Carte"
        display_cols.append("map_ui")
    elif "map_name" in df_sess.columns:
        col_map["map_name"] = "Carte"
        display_cols.append("map_name")
    
    for c in ["kills", "deaths", "assists"]:
        if c in df_sess.columns:
            col_map[c] = {"kills": "Frags", "deaths": "Morts", "assists": "Assists"}[c]
            display_cols.append(c)
    
    if "outcome" in df_sess.columns:
        outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
        df_sess["Résultat"] = df_sess["outcome"].map(outcome_map).fillna("-")
        display_cols.append("Résultat")
    
    # Colonne Performance RELATIVE (après Résultat)
    history_df = df_full if df_full is not None else df_sess
    df_sess["Performance"] = compute_performance_series(df_sess, history_df)
    df_sess["Perf_display"] = df_sess["Performance"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "-"
    )
    display_cols.append("Perf_display")
    col_map["Perf_display"] = "Performance"
    
    if "team_mmr" in df_sess.columns:
        df_sess["MMR Équipe"] = df_sess["team_mmr"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
        display_cols.append("MMR Équipe")
    
    if "enemy_mmr" in df_sess.columns:
        df_sess["MMR Adverse"] = df_sess["enemy_mmr"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
        display_cols.append("MMR Adverse")
    
    # Renommer les colonnes
    df_display = df_sess[display_cols].copy()
    df_display = df_display.rename(columns=col_map)
    
    # Garder les scores de performance pour la coloration
    perf_scores = df_sess["Performance"].values if "Performance" in df_sess.columns else None
    
    # Trier par heure (conserver l'ordre chronologique via start_time)
    if "start_time" in df_sess.columns:
        sort_indices = df_sess["start_time"].argsort().values
        df_display = df_display.iloc[sort_indices]
        if perf_scores is not None:
            perf_scores = perf_scores[sort_indices]
    
    # Affichage HTML pour styliser les résultats
    import html as html_lib
    html_rows = []
    for idx, (_, row) in enumerate(df_display.iterrows()):
        cells = []
        for col in df_display.columns:
            val = row[col]
            if col == "Résultat":
                css_class = _outcome_class(str(val))
                cells.append(f"<td class='{css_class}'>{html_lib.escape(str(val))}</td>")
            elif col == "Performance":
                # Coloration selon le score
                perf_val = perf_scores[idx] if perf_scores is not None else None
                css_class = get_score_class(perf_val)
                cells.append(f"<td class='{css_class}'>{html_lib.escape(str(val) if val is not None else '-')}</td>")
            else:
                cells.append(f"<td>{html_lib.escape(str(val) if val is not None else '-')}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    
    header_cells = "".join(f"<th>{html_lib.escape(c)}</th>" for c in df_display.columns)
    html_table = f"""
    <table class="session-history-table">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)


# Couleurs distinctes pour les sessions (contraste élevé, accessible daltoniens)
SESSION_COLORS = {
    "session_a": "#E74C3C",  # Rouge corail
    "session_a_fill": "rgba(231, 76, 60, 0.3)",
    "session_b": "#3498DB",  # Bleu vif
    "session_b_fill": "rgba(52, 152, 219, 0.3)",
    "historical": "#9B59B6",  # Violet
    "historical_fill": "rgba(155, 89, 182, 0.2)",
}


def render_comparison_radar_chart(
    perf_a: dict,
    perf_b: dict,
    hist_avg: dict | None = None,
) -> None:
    """Affiche le radar chart comparatif avec moyenne historique optionnelle.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
        hist_avg: Moyenne historique des sessions similaires (optionnel).
    """
    categories = ["K/D", "Victoire %", "Précision"]
    
    def _normalize_for_radar(kd, wr, acc):
        kd_norm = min(100, (kd or 0) * 50)  # K/D 2.0 = 100
        wr_norm = wr or 0  # Déjà en %
        acc_norm = acc if acc is not None else 50  # Déjà en %
        return [kd_norm, wr_norm, acc_norm]
    
    values_a = _normalize_for_radar(
        perf_a["kd_ratio"], perf_a["win_rate"], perf_a["accuracy"]
    )
    values_b = _normalize_for_radar(
        perf_b["kd_ratio"], perf_b["win_rate"], perf_b["accuracy"]
    )
    
    fig_radar = go.Figure()
    
    # Moyenne historique en fond (si disponible)
    if hist_avg and hist_avg.get("session_count", 0) >= 3:
        values_hist = _normalize_for_radar(
            hist_avg.get("kd_ratio"), hist_avg.get("win_rate"), hist_avg.get("accuracy")
        )
        fig_radar.add_trace(go.Scatterpolar(
            r=values_hist + [values_hist[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Moy. historique ({hist_avg["session_count"]} sessions)',
            line_color=SESSION_COLORS["historical"],
            fillcolor=SESSION_COLORS["historical_fill"],
            line=dict(dash="dot"),
        ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name='Session A',
        line_color=SESSION_COLORS["session_a"],
        fillcolor=SESSION_COLORS["session_a_fill"],
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Session B',
        line_color=SESSION_COLORS["session_b"],
        fillcolor=SESSION_COLORS["session_b_fill"],
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        height=400,
    )
    
    st.plotly_chart(fig_radar, width="stretch")


def render_comparison_bar_chart(
    perf_a: dict, 
    perf_b: dict, 
    hist_avg: dict | None = None,
) -> None:
    """Affiche le graphique en barres comparatif.
    
    Args:
        perf_a: Métriques de la session A.
        perf_b: Métriques de la session B.
        hist_avg: Moyenne historique des sessions similaires (optionnel).
    """
    metrics_labels = ["K/D Ratio", "Victoire (%)"]
    values_a_bar = [
        perf_a["kd_ratio"] or 0,
        perf_a["win_rate"] or 0,
    ]
    values_b_bar = [
        perf_b["kd_ratio"] or 0,
        perf_b["win_rate"] or 0,
    ]
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Session A',
        x=metrics_labels,
        y=values_a_bar,
        marker_color=SESSION_COLORS["session_a"],
    ))
    fig_bar.add_trace(go.Bar(
        name='Session B',
        x=metrics_labels,
        y=values_b_bar,
        marker_color=SESSION_COLORS["session_b"],
    ))
    
    # Ajouter la moyenne historique si disponible
    if hist_avg and hist_avg.get("session_count", 0) >= 3:
        values_hist = [
            hist_avg.get("kd_ratio", 0) or 0,
            hist_avg.get("win_rate", 0) or 0,
        ]
        fig_bar.add_trace(go.Bar(
            name=f'Moy. historique ({hist_avg["session_count"]} sessions)',
            x=metrics_labels,
            y=values_hist,
            marker_color=SESSION_COLORS["historical"],
        ))
    
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=350,
    )
    
    st.plotly_chart(fig_bar, width="stretch")


def render_session_comparison_page(
    all_sessions_df: pd.DataFrame,
    df_full: pd.DataFrame | None = None,
) -> None:
    """Rend la page de comparaison de sessions.
    
    Args:
        all_sessions_df: DataFrame avec toutes les sessions (colonnes session_id, session_label).
        df_full: DataFrame complet pour le calcul du score relatif.
    """
    st.caption("Compare les performances entre deux sessions de jeu.")
    
    # Note explicative sur le score de performance
    with st.expander("ℹ️ À propos du score de performance", expanded=False):
        st.markdown(PERFORMANCE_SCORE_SHORT_DESC)
    
    if all_sessions_df.empty:
        st.info("Aucune session disponible.")
        return
    
    # Liste des sessions triées (plus récente en premier)
    session_info = (
        all_sessions_df.groupby(["session_id", "session_label"])
        .size()
        .reset_index(name="count")
        .sort_values("session_id", ascending=False)
    )
    session_labels = session_info["session_label"].tolist()
    
    if len(session_labels) < 2:
        st.warning("Il faut au moins 2 sessions pour comparer.")
        return
    
    # Sélecteurs de sessions
    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        # Session A = avant-dernière par défaut
        default_a = session_labels[1] if len(session_labels) > 1 else session_labels[0]
        session_a_label = st.selectbox(
            "Session A (référence)",
            options=session_labels,
            index=session_labels.index(default_a) if default_a in session_labels else 1,
            key="compare_session_a",
        )
    with col_sel_b:
        # Session B = dernière par défaut
        session_b_label = st.selectbox(
            "Session B (à comparer)",
            options=session_labels,
            index=0,
            key="compare_session_b",
        )
    
    # Filtrer les DataFrames
    df_session_a = all_sessions_df[all_sessions_df["session_label"] == session_a_label].copy()
    df_session_b = all_sessions_df[all_sessions_df["session_label"] == session_b_label].copy()
    
    # Calculer les scores de performance (v2)
    perf_a = compute_session_performance_score_v2_ui(df_session_a)
    perf_b = compute_session_performance_score_v2_ui(df_session_b)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Calcul de la moyenne historique des sessions similaires
    # ══════════════════════════════════════════════════════════════════════════
    # Récupérer les session_ids à exclure
    session_a_id = df_session_a["session_id"].iloc[0] if not df_session_a.empty and "session_id" in df_session_a.columns else None
    session_b_id = df_session_b["session_id"].iloc[0] if not df_session_b.empty and "session_id" in df_session_b.columns else None
    exclude_ids = [sid for sid in [session_a_id, session_b_id] if sid is not None]
    
    # Déterminer le type de session (solo vs avec amis) - basé sur la session B (celle qu'on compare)
    session_b_with_friends = is_session_with_friends(df_session_b)
    session_b_friends = get_session_friends_signature(df_session_b)
    
    # Option de comparaison : mêmes amis vs solo/avec amis
    compare_mode = "same_friends"  # Par défaut : mêmes amis si possible
    if session_b_with_friends and len(session_b_friends) > 0:
        # Essayer d'abord avec les mêmes amis
        hist_avg = compute_similar_sessions_average(
            all_sessions_df,
            is_with_friends=True,
            exclude_session_ids=exclude_ids,
            same_friends_xuids=session_b_friends,
        )
        
        # Si pas assez de sessions avec les mêmes amis, fallback sur "avec amis"
        if hist_avg.get("session_count", 0) < 3:
            hist_avg = compute_similar_sessions_average(
                all_sessions_df,
                is_with_friends=True,
                exclude_session_ids=exclude_ids,
                same_friends_xuids=None,  # N'importe quels amis
            )
            compare_mode = "any_friends"
    else:
        # Solo : comparer avec autres sessions solo
        hist_avg = compute_similar_sessions_average(
            all_sessions_df,
            is_with_friends=False,
            exclude_session_ids=exclude_ids,
        )
        compare_mode = "solo"
    
    # Construire le label du type de session
    if session_b_with_friends and len(session_b_friends) > 0:
        # Récupérer les gamertags des amis (depuis la table Friends si possible)
        friends_names = _get_friends_names(df_session_b)
        if friends_names:
            friends_display = ", ".join(sorted(friends_names))
            session_type_label = f"avec {friends_display} 👥"
        else:
            session_type_label = f"avec {len(session_b_friends)} ami(s) 👥"
        
        if compare_mode == "same_friends":
            compare_label = f"mêmes amis ({hist_avg.get('session_count', 0)} sessions)"
        else:
            compare_label = f"sessions avec amis ({hist_avg.get('session_count', 0)} sessions)"
    else:
        session_type_label = "solo 🎮"
        compare_label = f"sessions solo ({hist_avg.get('session_count', 0)} sessions)"
    
    # Déterminer qui est meilleur
    score_a = perf_a.get("score")
    score_b = perf_b.get("score")
    is_b_better = None
    if score_a is not None and score_b is not None:
        if score_b > score_a:
            is_b_better = True
        elif score_b < score_a:
            is_b_better = False
    
    # ══════════════════════════════════════════════════════════════════════════
    # Grandes cartes de score côte à côte
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏆 Score de performance")
    col_score_a, col_score_b = st.columns(2)
    with col_score_a:
        render_performance_score_card(
            "Session A",
            perf_a,
            is_better=not is_b_better if is_b_better is not None else None,
        )
    with col_score_b:
        render_performance_score_card("Session B", perf_b, is_better=is_b_better)
    
    st.markdown("---")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Colonnes Session A / Session B avec métriques détaillées
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Métriques détaillées")
    
    # En-têtes
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.markdown("**Métrique**")
    with col_h2:
        st.markdown("**Session A**")
    with col_h3:
        st.markdown("**Session B**")
    
    st.markdown("---")
    
    render_metric_comparison_row("Nombre de parties", perf_a["matches"], perf_b["matches"], "{}")
    render_metric_comparison_row("FDA (Frags-Décès-Assists)", perf_a["kda"], perf_b["kda"], "{:.2f}")
    render_metric_comparison_row("Taux de victoire", perf_a["win_rate"], perf_b["win_rate"], "{:.1f}%")
    render_metric_comparison_row("Durée de vie moyenne", perf_a["avg_life_seconds"], perf_b["avg_life_seconds"], _format_seconds_to_mmss)
    
    st.markdown("---")
    
    render_metric_comparison_row("Total des frags", perf_a["kills"], perf_b["kills"], "{}")
    render_metric_comparison_row(
        "Total des morts", perf_a["deaths"], perf_b["deaths"], "{}", higher_is_better=False
    )
    render_metric_comparison_row("Total des assistances", perf_a["assists"], perf_b["assists"], "{}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Comparaison MMR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🎯 Comparaison MMR")
    
    col_mmr1, col_mmr2, col_mmr3 = st.columns([2, 1, 1])
    with col_mmr1:
        st.markdown("**Métrique MMR**")
    with col_mmr2:
        st.markdown("**Session A**")
    with col_mmr3:
        st.markdown("**Session B**")
    
    render_metric_comparison_row(
        "MMR équipe (moy)", perf_a["team_mmr_avg"], perf_b["team_mmr_avg"], "{:.1f}"
    )
    render_metric_comparison_row(
        "MMR adverse (moy)",
        perf_a["enemy_mmr_avg"],
        perf_b["enemy_mmr_avg"],
        "{:.1f}",
        higher_is_better=False,
    )
    render_metric_comparison_row(
        "Écart MMR (moy)", perf_a["delta_mmr_avg"], perf_b["delta_mmr_avg"], "{:+.1f}"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # Graphiques comparatifs (côte à côte)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    
    # Afficher l'indicateur du type de session et la moyenne historique
    if hist_avg and hist_avg.get("session_count", 0) >= 3:
        st.markdown(
            f"### 📈 Graphiques comparatifs\n"
            f"*Session {session_type_label} — Moyenne historique : {compare_label}*"
        )
    else:
        st.markdown(f"### 📈 Graphiques comparatifs\n*Session {session_type_label}*")
    
    col_radar, col_bars = st.columns(2)
    
    with col_radar:
        st.markdown("#### Vue radar")
        render_comparison_radar_chart(perf_a, perf_b, hist_avg=hist_avg)
    
    with col_bars:
        st.markdown("#### Comparaison par métrique")
        render_comparison_bar_chart(perf_a, perf_b, hist_avg=hist_avg)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Tableau historique des parties
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Historique des parties")
    
    tab_hist_a, tab_hist_b = st.tabs(["Session A", "Session B"])
    
    with tab_hist_a:
        render_session_history_table(df_session_a, "Session A", df_full=df_full)
    
    with tab_hist_b:
        render_session_history_table(df_session_b, "Session B", df_full=df_full)
