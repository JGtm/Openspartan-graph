"""OpenSpartan Graphs - Dashboard Streamlit.

Application de visualisation des statistiques Halo Infinite
depuis la base de données OpenSpartan Workshop.
"""

import os
from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Imports depuis la nouvelle architecture
from src.config import (
    get_default_db_path,
    get_default_workshop_exe_path,
    DEFAULT_WAYPOINT_PLAYER,
    HALO_COLORS,
    OUTCOME_CODES,
)
from src.db import (
    load_matches,
    query_matches_with_friend,
    list_other_player_xuids,
    list_top_teammates,
    guess_xuid_from_db_path,
    load_player_match_result,
    load_match_medals_for_player,
    load_top_medals,
)
from src.db.parsers import parse_xuid_input
from src.analysis import (
    compute_aggregated_stats,
    compute_outcome_rates,
    compute_global_ratio,
    compute_sessions,
    compute_map_breakdown,
    mark_firefight,
    is_allowed_playlist_name,
    build_option_map,
    build_xuid_option_map,
)
from src.analysis.stats import format_selected_matches_summary, format_mmss
from src.visualization import (
    plot_timeseries,
    plot_assists_timeseries,
    plot_per_minute_timeseries,
    plot_average_life,
    plot_spree_headshots_accuracy,
    plot_kda_distribution,
    plot_outcomes_over_time,
    plot_map_comparison,
    plot_map_ratio_with_winloss,
    plot_trio_metric,
)
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom
from src.ui import (
    load_aliases_file,
    save_aliases_file,
    get_xuid_aliases,
    display_name_from_xuid,
    load_css,
    get_hero_html,
    translate_playlist_name,
    translate_pair_name,
)
from src.ui.medals import (
    load_medal_name_maps,
    medal_has_known_label,
    get_medals_cache_dir,
    medal_label,
    medal_icon_path,
    render_medals_grid,
)
from src.ui.formatting import format_date_fr
from src.db.profiles import (
    PROFILES_PATH,
    load_profiles,
    save_profiles,
    list_local_dbs,
)
from src.config import get_aliases_file_path


# =============================================================================
# Chargement des données (avec cache)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_df(db_path: str, xuid: str) -> pd.DataFrame:
    """Charge les matchs et les convertit en DataFrame."""
    matches = load_matches(db_path, xuid)
    df = pd.DataFrame(
        {
            "match_id": [m.match_id for m in matches],
            "start_time": [m.start_time for m in matches],
            "map_id": [m.map_id for m in matches],
            "map_name": [m.map_name for m in matches],
            "playlist_id": [m.playlist_id for m in matches],
            "playlist_name": [m.playlist_name for m in matches],
            "pair_id": [m.map_mode_pair_id for m in matches],
            "pair_name": [m.map_mode_pair_name for m in matches],
            "game_variant_id": [m.game_variant_id for m in matches],
            "game_variant_name": [m.game_variant_name for m in matches],
            "outcome": [m.outcome for m in matches],
            "kda": [m.kda for m in matches],
            "max_killing_spree": [m.max_killing_spree for m in matches],
            "headshot_kills": [m.headshot_kills for m in matches],
            "average_life_seconds": [m.average_life_seconds for m in matches],
            "time_played_seconds": [m.time_played_seconds for m in matches],
            "kills": [m.kills for m in matches],
            "deaths": [m.deaths for m in matches],
            "assists": [m.assists for m in matches],
            "accuracy": [m.accuracy for m in matches],
            "ratio": [m.ratio for m in matches],
        }
    )
    # Facilite les filtres date
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(None)
    df["date"] = df["start_time"].dt.date

    # Stats par minute
    minutes = (pd.to_numeric(df["time_played_seconds"], errors="coerce") / 60.0).astype(float)
    minutes = minutes.where(minutes > 0)
    df["kills_per_min"] = pd.to_numeric(df["kills"], errors="coerce") / minutes
    df["deaths_per_min"] = pd.to_numeric(df["deaths"], errors="coerce") / minutes
    df["assists_per_min"] = pd.to_numeric(df["assists"], errors="coerce") / minutes
    return df


@st.cache_data(show_spinner=False)
def cached_list_other_xuids(db_path: str, self_xuid: str, limit: int = 500) -> list[str]:
    """Version cachée de list_other_player_xuids."""
    return list_other_player_xuids(db_path, self_xuid, limit)


@st.cache_data(show_spinner=False)
def cached_list_top_teammates(db_path: str, self_xuid: str, limit: int = 20) -> list[tuple[str, int]]:
    """Version cachée de list_top_teammates."""
    return list_top_teammates(db_path, self_xuid, limit)


# =============================================================================
# Helpers UI
# =============================================================================

def _date_range(df: pd.DataFrame) -> tuple[date, date]:
    """Retourne la plage de dates du DataFrame."""
    dmin = df["date"].min()
    dmax = df["date"].max()
    return dmin, dmax


# =============================================================================
# Application principale
# =============================================================================

def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    st.set_page_config(page_title="OpenSpartan Graphs", layout="wide")

    # Charge et applique le CSS
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown(get_hero_html(), unsafe_allow_html=True)

    # ==========================================================================
    # Sidebar - Configuration
    # ==========================================================================
    
    DEFAULT_DB = get_default_db_path()
    
    with st.sidebar:
        with st.expander("Source", expanded=False):
            # --- Multi-DB / Profils ---
            profiles = load_profiles()

            if "db_path" not in st.session_state:
                st.session_state["db_path"] = DEFAULT_DB
            if "xuid" not in st.session_state:
                st.session_state["xuid"] = guess_xuid_from_db_path(st.session_state.get("db_path", "")) or ""
            if "waypoint_player" not in st.session_state:
                st.session_state["waypoint_player"] = DEFAULT_WAYPOINT_PLAYER

            with st.expander("Multi-DB (profils)", expanded=False):
                prof_names = ["(aucun)"] + sorted(profiles.keys())
                st.selectbox("Profil", options=prof_names, key="db_profile_selected")

                cols_p = st.columns(2)
                if cols_p[0].button("Appliquer", width="stretch"):
                    sel = st.session_state.get("db_profile_selected")
                    if isinstance(sel, str) and sel in profiles:
                        p = profiles[sel]
                        if p.get("db_path"):
                            st.session_state["db_path"] = p["db_path"]
                        if p.get("xuid"):
                            st.session_state["xuid"] = p["xuid"]
                        if p.get("waypoint_player"):
                            st.session_state["waypoint_player"] = p["waypoint_player"]
                        st.rerun()

                if cols_p[1].button("Enregistrer profil", width="stretch"):
                    name = st.session_state.get("db_profile_selected")
                    if not isinstance(name, str) or not name.strip() or name == "(aucun)":
                        st.warning("Choisis d'abord un nom de profil (via la liste), ou crée-en un ci-dessous.")
                    else:
                        profiles[name] = {
                            "db_path": str(st.session_state.get("db_path", "")).strip(),
                            "xuid": str(st.session_state.get("xuid", "")).strip(),
                            "waypoint_player": str(st.session_state.get("waypoint_player", "")).strip(),
                        }
                        ok, err = save_profiles(profiles)
                        if ok:
                            st.success("Profil enregistré.")
                        else:
                            st.error(err)

                with st.expander("Créer / supprimer", expanded=False):
                    new_name = st.text_input("Nom", value="")
                    c = st.columns(2)
                    if c[0].button("Créer/mettre à jour", width="stretch"):
                        nn = (new_name or "").strip()
                        if not nn:
                            st.error("Nom vide.")
                        else:
                            profiles[nn] = {
                                "db_path": str(st.session_state.get("db_path", "")).strip(),
                                "xuid": str(st.session_state.get("xuid", "")).strip(),
                                "waypoint_player": str(st.session_state.get("waypoint_player", "")).strip(),
                            }
                            ok, err = save_profiles(profiles)
                            if ok:
                                st.success("Profil sauvegardé.")
                                st.rerun()
                            else:
                                st.error(err)
                    if c[1].button("Supprimer", width="stretch"):
                        nn = (new_name or "").strip()
                        if not nn:
                            st.error("Renseigne un nom.")
                        elif nn not in profiles:
                            st.warning("Profil introuvable.")
                        else:
                            del profiles[nn]
                            ok, err = save_profiles(profiles)
                            if ok:
                                st.success("Profil supprimé.")
                                st.rerun()
                            else:
                                st.error(err)

                local_dbs = list_local_dbs()
                if local_dbs:
                    opts = ["(garder actuelle)"] + [os.path.basename(p) for p in local_dbs]
                    pick = st.selectbox("DB détectées (OpenSpartan)", options=opts, index=0)
                    if st.button("Utiliser cette DB", width="stretch") and pick != "(garder actuelle)":
                        sel_path = next((p for p in local_dbs if os.path.basename(p) == pick), None)
                        if sel_path:
                            st.session_state["db_path"] = sel_path
                            guessed = guess_xuid_from_db_path(sel_path) or ""
                            if guessed:
                                st.session_state["xuid"] = guessed
                            st.rerun()

            db_path = st.text_input("Chemin du .db", key="db_path")
            cols_x = st.columns([2, 1])
            with cols_x[0]:
                xuid = st.text_input("XUID", key="xuid")
            with cols_x[1]:
                if st.button("Deviner XUID", width="stretch"):
                    guessed = guess_xuid_from_db_path(str(db_path)) or ""
                    if guessed:
                        st.session_state["xuid"] = guessed
                        st.rerun()
                    else:
                        st.warning("Impossible de deviner le XUID depuis ce chemin.")

            me_name = display_name_from_xuid(xuid.strip())
            waypoint_player = st.text_input(
                "HaloWaypoint player (slug)",
                key="waypoint_player",
                help="Ex: JGtm (sert à construire l'URL de match).",
            )

            # Section alias
            with st.expander("Alias (XUID → gamertag)", expanded=False):
                st.caption(
                    "La DB OpenSpartan ne contient pas les gamertags. "
                    "Ici tu peux définir des alias locaux (persistés dans un fichier JSON)."
                )
                aliases_path = get_aliases_file_path()
                current_aliases = load_aliases_file(aliases_path)

                ax = st.text_input("XUID à aliaser", value="")
                an = st.text_input("Gamertag", value="")
                cols = st.columns(2)
                if cols[0].button("Enregistrer l'alias", width="stretch"):
                    axc = (ax or "").strip()
                    anc = (an or "").strip()
                    if not axc.isdigit():
                        st.error("XUID invalide (doit être numérique).")
                    elif not anc:
                        st.error("Gamertag vide.")
                    else:
                        current_aliases[axc] = anc
                        save_aliases_file(current_aliases, aliases_path)
                        st.success("Alias enregistré.")
                        st.rerun()

                if cols[1].button("Supprimer l'alias", width="stretch"):
                    axc = (ax or "").strip()
                    if not axc:
                        st.error("Renseigne un XUID à supprimer.")
                    elif axc not in current_aliases:
                        st.warning("Cet alias n'existe pas.")
                    else:
                        del current_aliases[axc]
                        save_aliases_file(current_aliases, aliases_path)
                        st.success("Alias supprimé.")
                        st.rerun()

                if current_aliases:
                    st.dataframe(
                        pd.DataFrame(
                            sorted(current_aliases.items()),
                            columns=["XUID", "Gamertag"],
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Aucun alias personnalisé pour l'instant.")

        st.divider()
        st.header("OpenSpartan")
        workshop_exe = st.text_input(
            "Chemin de OpenSpartan.Workshop.exe",
            value=get_default_workshop_exe_path(),
            help="Bouton pratique pour lancer l'app OpenSpartan Workshop.",
        )
        if st.button("Lancer OpenSpartan Workshop", width="stretch"):
            if not os.path.exists(workshop_exe):
                st.error("Executable introuvable à ce chemin.")
            else:
                try:
                    if hasattr(os, "startfile"):
                        os.startfile(workshop_exe)  # type: ignore[attr-defined]
                    else:
                        import subprocess
                        subprocess.Popen([workshop_exe], close_fds=True)
                    st.success("OpenSpartan Workshop lancé.")
                except Exception as e:
                    st.error(f"Impossible de lancer OpenSpartan Workshop: {e}")

        # Validation des entrées
        if not db_path.strip():
            st.error(
                "Aucun .db détecté automatiquement. "
                "Vérifie que OpenSpartan Workshop est installé."
            )
            st.stop()
        if not os.path.exists(db_path):
            st.error("Le fichier .db n'existe pas à ce chemin.")
            st.stop()
        if not xuid.strip().isdigit():
            st.error("XUID invalide (doit être numérique).")
            st.stop()

    # ==========================================================================
    # Chargement des données
    # ==========================================================================
    
    df = load_df(db_path, xuid.strip())
    if df.empty:
        st.warning("Aucun match trouvé.")
        st.stop()

    df = mark_firefight(df)

    # ==========================================================================
    # Sidebar - Filtres
    # ==========================================================================
    
    with st.sidebar:
        st.header("Filtres")
        if "include_firefight" not in st.session_state:
            st.session_state["include_firefight"] = False
        if "restrict_playlists" not in st.session_state:
            st.session_state["restrict_playlists"] = True

    include_firefight = bool(st.session_state.get("include_firefight", False))

    # Firefight exclu par défaut
    base_for_filters = df.copy()
    if (not include_firefight) and ("is_firefight" in base_for_filters.columns):
        base_for_filters = base_for_filters.loc[~base_for_filters["is_firefight"]].copy()

    with st.sidebar:
        dmin, dmax = _date_range(base_for_filters)
        if "filter_mode" not in st.session_state:
            st.session_state["filter_mode"] = "Période"
        filter_mode = st.radio(
            "Sélection",
            options=["Période", "Sessions"],
            horizontal=True,
            key="filter_mode",
        )

        start_d, end_d = dmin, dmax
        gap_minutes = 35
        picked_session_labels: Optional[list[str]] = None
        
        if filter_mode == "Période":
            cols = st.columns(2)
            with cols[0]:
                start_d = st.date_input("Début", value=dmin, min_value=dmin, max_value=dmax)
            with cols[1]:
                end_d = st.date_input("Fin", value=dmax, min_value=dmin, max_value=dmax)
            if start_d > end_d:
                st.warning("La date de début est après la date de fin.")
        else:
            gap_minutes = st.slider(
                "Écart max entre parties (minutes)",
                min_value=15,
                max_value=240,
                value=int(st.session_state.get("gap_minutes", 120)),
                step=5,
                key="gap_minutes",
            )

            base_s_ui = compute_sessions(base_for_filters, gap_minutes=gap_minutes)
            session_labels_ui = (
                base_s_ui[["session_id", "session_label"]]
                .drop_duplicates()
                .sort_values("session_id", ascending=False)
            )
            options_ui = session_labels_ui["session_label"].tolist()

            compare_multi = st.toggle("Comparer plusieurs sessions", value=False)

            def _set_session_selection(label: str) -> None:
                st.session_state.picked_session_label = label
                if label == "(toutes)":
                    st.session_state.picked_sessions = []
                elif label in options_ui:
                    st.session_state.picked_sessions = [label]

            if "picked_session_label" not in st.session_state:
                _set_session_selection(options_ui[0] if options_ui else "(toutes)")
            if "picked_sessions" not in st.session_state:
                st.session_state.picked_sessions = options_ui[:1] if options_ui else []

            cols = st.columns(2)
            if cols[0].button("Dernière session", width="stretch"):
                _set_session_selection(options_ui[0] if options_ui else "(toutes)")
            if cols[1].button("Session précédente", width="stretch"):
                current = st.session_state.get("picked_session_label", "(toutes)")
                if not options_ui:
                    _set_session_selection("(toutes)")
                elif current == "(toutes)" or current not in options_ui:
                    _set_session_selection(options_ui[0])
                else:
                    idx = options_ui.index(current)
                    next_idx = min(idx + 1, len(options_ui) - 1)
                    _set_session_selection(options_ui[next_idx])

            if compare_multi:
                picked = st.multiselect("Sessions", options=options_ui, key="picked_sessions")
                picked_session_labels = picked if picked else None
            else:
                picked_one = st.selectbox("Session", options=["(toutes)"] + options_ui, key="picked_session_label")
                picked_session_labels = None if picked_one == "(toutes)" else [picked_one]

        playlist_opts = build_option_map(base_for_filters["playlist_name"], base_for_filters["playlist_id"])
        base_for_filters = base_for_filters.copy()
        base_for_filters["playlist_fr"] = base_for_filters["playlist_name"].apply(translate_playlist_name)
        playlist_fr_values = sorted(
            {str(x).strip() for x in base_for_filters["playlist_fr"].dropna().tolist() if str(x).strip()}
        )
        playlist_fr = st.selectbox("Playlist", options=["(toutes)"] + playlist_fr_values, index=0)

        mode_opts = build_option_map(base_for_filters["game_variant_name"], base_for_filters["game_variant_id"])
        mode_label = st.selectbox("Mode", options=["(tous)"] + list(mode_opts.keys()), index=0)
        mode_id: Optional[str] = None
        if mode_label != "(tous)":
            mode_id = mode_opts[mode_label]

        map_opts = build_option_map(base_for_filters["map_name"], base_for_filters["map_id"])
        map_label = st.selectbox("Carte", options=["(toutes)"] + list(map_opts.keys()), index=0)
        map_id: Optional[str] = None
        if map_label != "(toutes)":
            map_id = map_opts[map_label]

        last_n_acc = st.slider("Précision: derniers matchs", 5, 50, 20, step=1)

        with st.expander("Paramètres avancés", expanded=False):
            st.toggle("Inclure Firefight (PvE)", key="include_firefight", value=False)
            st.toggle(
                "Limiter aux playlists (Quick Play / Ranked Slayer / Ranked Arena)",
                key="restrict_playlists",
                value=True,
            )

    # ==========================================================================
    # Application des filtres
    # ==========================================================================
    
    base = base_for_filters.copy()
    if "playlist_fr" not in base.columns:
        base["playlist_fr"] = base["playlist_name"].apply(translate_playlist_name)
    if "pair_fr" not in base.columns:
        base["pair_fr"] = base["pair_name"].apply(translate_pair_name)

    if filter_mode == "Sessions":
        base_s = compute_sessions(base, gap_minutes=gap_minutes)
        dff = (
            base_s.loc[base_s["session_label"].isin(picked_session_labels)].copy()
            if picked_session_labels
            else base_s.copy()
        )
    else:
        dff = base.copy()

    restrict_playlists = bool(st.session_state.get("restrict_playlists", True))
    if restrict_playlists:
        pl = dff["playlist_name"].fillna("").astype(str)
        allowed_mask = pl.apply(is_allowed_playlist_name)
        if allowed_mask.any():
            dff = dff.loc[allowed_mask].copy()
        else:
            st.sidebar.warning(
                "Aucune playlist n'a matché Quick Play / Ranked Slayer / Ranked Arena. "
                "Désactive ce filtre si tes libellés sont différents."
            )
            
    if playlist_fr != "(toutes)":
        if "playlist_fr" not in dff.columns:
            dff["playlist_fr"] = dff["playlist_name"].apply(translate_playlist_name)
        dff = dff.loc[dff["playlist_fr"].fillna("") == playlist_fr]
    if mode_id is not None:
        dff = dff.loc[dff["game_variant_id"].fillna("") == mode_id]
    if map_id is not None:
        dff = dff.loc[dff["map_id"].fillna("") == map_id]

    if filter_mode == "Période":
        mask = (dff["date"] >= start_d) & (dff["date"] <= end_d)
        dff = dff.loc[mask].copy()


    # ==========================================================================
    # KPIs
    # ==========================================================================
    
    rates = compute_outcome_rates(dff)
    total_outcomes = max(1, rates.total)
    win_rate = rates.wins / total_outcomes
    loss_rate = rates.losses / total_outcomes

    avg_acc = dff["accuracy"].dropna().mean() if not dff.empty else None
    global_ratio = compute_global_ratio(dff)
    avg_life = dff["average_life_seconds"].dropna().mean() if not dff.empty else None

    # Moyennes par partie
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    avg_row = st.columns(3)
    avg_row[0].metric("Frags par partie", f"{kpg:.2f}" if kpg == kpg else "-")
    avg_row[1].metric("Morts par partie", f"{dpg:.2f}" if dpg == dpg else "-")
    avg_row[2].metric("Assistances par partie", f"{apg:.2f}" if apg == apg else "-")

    # Stats par minute
    stats = compute_aggregated_stats(dff)
    per_min_row = st.columns(3)
    per_min_row[0].metric("Frags / min", f"{stats.kills_per_minute:.2f}" if stats.kills_per_minute else "-")
    per_min_row[1].metric("Morts / min", f"{stats.deaths_per_minute:.2f}" if stats.deaths_per_minute else "-")
    per_min_row[2].metric("Assistances / min", f"{stats.assists_per_minute:.2f}" if stats.assists_per_minute else "-")

    kpi = st.columns(5)
    kpi[0].metric("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-")
    kpi[1].metric("Taux de victoire", f"{win_rate*100:.1f}%" if rates.total else "-")
    kpi[2].metric("Taux de défaite", f"{loss_rate*100:.1f}%" if rates.total else "-")
    kpi[3].metric("Ratio global", f"{global_ratio:.2f}" if global_ratio is not None else "-")
    kpi[4].metric("Durée de vie moyenne", format_mmss(avg_life))

    st.info(format_selected_matches_summary(len(dff), rates))

    # ==========================================================================
    # Onglets
    # ==========================================================================
    
    tab_series, tab_last, tab_medals, tab_mom, tab_friend, tab_friends, tab_maps, tab_table = st.tabs(
        [
            "Séries temporelles",
            "Dernier match",
            "Médailles (Top 25)",
            "Victoires/Défaites (par période)",
            "Avec un joueur",
            "Avec mes amis",
            "Ratio par cartes",
            "Historique des parties",
        ]
    )

    # --------------------------------------------------------------------------
    # Tab: Dernier match
    # --------------------------------------------------------------------------
    with tab_last:
        st.caption("Dernière partie selon la sélection/filtres actuels.")

        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            last_row = dff.sort_values("start_time").iloc[-1]
            last_match_id = str(last_row.get("match_id", ""))
            last_time = last_row.get("start_time")
            last_map = last_row.get("map_name")
            last_playlist = last_row.get("playlist_name")
            last_pair = last_row.get("pair_name")
            last_mode = last_row.get("game_variant_name")
            last_outcome = last_row.get("outcome")

            last_playlist_fr = translate_playlist_name(str(last_playlist)) if last_playlist else None
            last_pair_fr = translate_pair_name(str(last_pair)) if last_pair else None

            meta_cols = st.columns(4)
            meta_cols[0].metric("Date", format_date_fr(last_time))
            meta_cols[1].metric("Carte", str(last_map) if last_map else "-")
            meta_cols[2].metric("Playlist", str(last_playlist_fr or last_playlist) if (last_playlist_fr or last_playlist) else "-")
            meta_cols[3].metric("Mode", str(last_mode or last_pair_fr or last_pair) if (last_mode or last_pair_fr or last_pair) else "-")

            st.caption(f"MatchId: {last_match_id}")

            with st.spinner("Lecture des stats détaillées (attendu vs réel, médailles)…"):
                pm = load_player_match_result(db_path, last_match_id, xuid.strip())
                medals_last = load_match_medals_for_player(db_path, last_match_id, xuid.strip())

            # Rappel du résultat (même si PlayerMatchStats est indispo)
            outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
            st.metric(
                "Résultat",
                outcome_map.get(int(last_outcome), "?") if last_outcome == last_outcome else "-",
            )

            if not pm:
                st.info(
                    "Stats détaillées indisponibles pour ce match (PlayerMatchStats manquant ou format inattendu)."
                )
            else:
                team_mmr = pm.get("team_mmr")
                enemy_mmr = pm.get("enemy_mmr")
                delta_mmr = (team_mmr - enemy_mmr) if (team_mmr is not None and enemy_mmr is not None) else None

                mmr_cols = st.columns(3)
                mmr_cols[0].metric("MMR d'équipe", f"{team_mmr:.1f}" if team_mmr is not None else "-")
                mmr_cols[1].metric("MMR adverse", f"{enemy_mmr:.1f}" if enemy_mmr is not None else "-")
                mmr_cols[2].metric(
                    "Écart MMR (équipe - adverse)",
                    f"{delta_mmr:+.1f}" if delta_mmr is not None else "-",
                )

                # Attendu vs réel (K / D / A) + ratios (match uniquement)
                def _metric_expected_vs_actual(title: str, perf: dict, delta_color: str) -> None:
                    count = perf.get("count")
                    expected = perf.get("expected")
                    if count is None or expected is None:
                        st.metric(title, "-")
                        return
                    delta = float(count) - float(expected)
                    st.metric(title, f"{count:.0f} vs {expected:.1f}", f"{delta:+.1f}", delta_color=delta_color)

                perf_k = pm.get("kills") or {}
                perf_d = pm.get("deaths") or {}
                perf_a = pm.get("assists") or {}

                st.subheader("Attendu vs réel")
                av_cols = st.columns(4)
                with av_cols[0]:
                    _metric_expected_vs_actual("Kills", perf_k, delta_color="normal")
                with av_cols[1]:
                    _metric_expected_vs_actual("Morts", perf_d, delta_color="inverse")
                with av_cols[2]:
                    _metric_expected_vs_actual("Assistances", perf_a, delta_color="normal")
                with av_cols[3]:
                    avg_life_last = last_row.get("average_life_seconds")
                    st.metric("Durée de vie moyenne", format_mmss(avg_life_last))

                labels = ["F", "D", "A"]
                actual_vals = [
                    float(last_row.get("kills") or 0.0),
                    float(last_row.get("deaths") or 0.0),
                    float(last_row.get("assists") or 0.0),
                ]
                exp_vals = [
                    perf_k.get("expected"),
                    perf_d.get("expected"),
                    perf_a.get("expected"),
                ]

                show_expected_ratio = all(v is not None for v in exp_vals)
                real_ratio = last_row.get("ratio")
                try:
                    real_ratio_f = float(real_ratio) if real_ratio == real_ratio else None
                except Exception:
                    real_ratio_f = None
                if real_ratio_f is None:
                    denom = max(1.0, float(last_row.get("deaths") or 0.0))
                    real_ratio_f = (float(last_row.get("kills") or 0.0) + float(last_row.get("assists") or 0.0)) / denom

                exp_ratio_f = None
                if show_expected_ratio:
                    denom_e = max(1e-9, float(exp_vals[1] or 0.0))
                    exp_ratio_f = (float(exp_vals[0] or 0.0) + float(exp_vals[2] or 0.0)) / denom_e

                exp_fig = make_subplots(specs=[[{"secondary_y": True}]])

                bar_colors = [HALO_COLORS.green, HALO_COLORS.red, HALO_COLORS.cyan]
                exp_fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=exp_vals,
                        name="Attendu",
                        marker=dict(
                            color=bar_colors,
                            pattern=dict(shape="/", fgcolor="rgba(255,255,255,0.75)", solidity=0.22),
                        ),
                        opacity=0.50,
                        hovertemplate="%{x} (attendu): %{y:.1f}<extra></extra>",
                    ),
                    secondary_y=False,
                )
                exp_fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=actual_vals,
                        name="Réel",
                        marker_color=bar_colors,
                        opacity=0.90,
                        hovertemplate="%{x} (réel): %{y:.0f}<extra></extra>",
                    ),
                    secondary_y=False,
                )
                exp_fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=[real_ratio_f] * len(labels),
                        mode="lines+markers",
                        name="Ratio réel",
                        line=dict(color=HALO_COLORS.amber, width=4),
                        marker=dict(size=7),
                        hovertemplate="ratio (réel): %{y:.2f}<extra></extra>",
                    ),
                    secondary_y=True,
                )
                if exp_ratio_f is not None:
                    exp_fig.add_trace(
                        go.Scatter(
                            x=labels,
                            y=[exp_ratio_f] * len(labels),
                            mode="lines+markers",
                            name="Ratio attendu",
                            line=dict(color=HALO_COLORS.violet, width=3, dash="dot"),
                            marker=dict(size=6),
                            hovertemplate="ratio (attendu): %{y:.2f}<extra></extra>",
                        ),
                        secondary_y=True,
                    )
                else:
                    st.caption("Ratio attendu indisponible (Assists attendues manquantes dans PlayerMatchStats).")

                exp_fig.update_layout(
                    barmode="group",
                    height=360,
                    margin=dict(l=40, r=20, t=30, b=90),
                    legend=get_legend_horizontal_bottom(),
                )
                exp_fig.update_yaxes(title_text="F / D / A", rangemode="tozero", secondary_y=False)
                exp_fig.update_yaxes(title_text="Ratio", secondary_y=True)
                st.plotly_chart(exp_fig, width="stretch")

            # Médailles (match)
            st.subheader("Médailles (match)")
            if not medals_last:
                st.info("Médailles indisponibles pour ce match (ou aucune médaille).")
            else:
                md_df = pd.DataFrame(medals_last)
                md_df["label"] = md_df["name_id"].apply(lambda x: medal_label(int(x)))
                md_df = md_df.sort_values(["count", "label"], ascending=[False, True])
                render_medals_grid(md_df[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)



    # --------------------------------------------------------------------------
    # Tab: Médailles (Top 25)
    # --------------------------------------------------------------------------
    with tab_medals:
        st.caption("Top 25 des médailles sur la sélection/filtres actuels.")

        if dff.empty:
            st.info("Aucun match disponible avec les filtres actuels.")
        else:
            match_ids = [str(x) for x in dff["match_id"].dropna().astype(str).tolist()]

            with st.spinner("Agrégation des médailles…"):
                top = load_top_medals(db_path, xuid.strip(), match_ids, top_n=25)

            if not top:
                st.info("Aucune médaille trouvée (ou payload médailles absent).")
            else:
                md = pd.DataFrame(top, columns=["name_id", "count"])

                md["label"] = md["name_id"].apply(lambda x: medal_label(int(x)))
                md_desc = md.sort_values("count", ascending=False)
                render_medals_grid(md_desc[["name_id", "count"]].to_dict(orient="records"), cols_per_row=8)

    # --------------------------------------------------------------------------
    # Tab: Séries temporelles
    # --------------------------------------------------------------------------
    with tab_series:
        with st.spinner("Génération des graphes…"):
            fig = plot_timeseries(dff, title=f"{me_name}")
            st.plotly_chart(fig, width="stretch")

            st.subheader("FDA")
            valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
            if valid.empty:
                st.info("FDA indisponible sur ce filtre.")
            else:
                m = st.columns(1)
                m[0].metric("Moyenne FDA", f"{valid['kda'].mean():.2f}")
                st.caption("Densité (KDE) + rug : forme de la distribution + position des matchs.")
                st.plotly_chart(plot_kda_distribution(dff), width="stretch")

            st.subheader("Assistances")
            st.plotly_chart(plot_assists_timeseries(dff, title=f"{me_name} — Assistances"), width="stretch")

            st.subheader("Stats par minute")
            st.plotly_chart(
                plot_per_minute_timeseries(dff, title=f"{me_name} — Frags/Morts/Assistances par minute"),
                width="stretch",
            )

            st.subheader("Durée de vie moyenne")
            if dff.dropna(subset=["average_life_seconds"]).empty:
                st.info("Average Life indisponible sur ce filtre.")
            else:
                st.plotly_chart(plot_average_life(dff), width="stretch")

            st.subheader("Folie meurtrière / Tirs à la tête / Précision")
            st.plotly_chart(plot_spree_headshots_accuracy(dff), width="stretch")

    # --------------------------------------------------------------------------
    # Tab: Victoires/Défaites
    # --------------------------------------------------------------------------
    with tab_mom:
        with st.spinner("Calcul des victoires/défaites…"):
            fig_out, bucket_label = plot_outcomes_over_time(dff)
            st.markdown(
                f"Par **{bucket_label}** : on regroupe les parties par {bucket_label} et on compte le nombre de "
                "victoires/défaites (et autres statuts) pour suivre l'évolution."
            )
            st.caption("Basé sur Players[].Outcome (2=victoire, 3=défaite, 1=égalité, 4=non terminé).")
            st.plotly_chart(fig_out, width="stretch")

    # --------------------------------------------------------------------------
    # Tab: Avec un joueur
    # --------------------------------------------------------------------------
    with tab_friend:
        st.caption(
            "La DB locale ne contient pas les gamertags, uniquement des PlayerId de type xuid(...). "
            "Tu peux soit coller un XUID, soit sélectionner un XUID rencontré dans tes matchs."
        )
        cols = st.columns([2, 2, 1])
        with cols[0]:
            friend_raw = st.text_input("Ami: XUID ou xuid(123)", value="")
        with cols[1]:
            opts_map = build_xuid_option_map(
                cached_list_other_xuids(db_path, xuid.strip(), limit=500),
                display_name_fn=display_name_from_xuid,
            )
            friend_pick_label = st.selectbox("Ou choisir un XUID vu", options=["(aucun)"] + list(opts_map.keys()), index=0)
        with cols[2]:
            same_team_only = st.checkbox("Même équipe", value=True)

        friend_xuid = parse_xuid_input(friend_raw) or (
            opts_map.get(friend_pick_label) if friend_pick_label != "(aucun)" else None
        )

        with st.spinner("Chargement des matchs avec ce joueur…"):
            if friend_xuid is None:
                st.info("Renseigne un XUID (numérique) ou choisis-en un.")
            else:
                rows = query_matches_with_friend(db_path, xuid.strip(), friend_xuid)
                if same_team_only:
                    rows = [r for r in rows if r.same_team]

                if not rows:
                    st.warning("Aucun match trouvé avec ce joueur (selon le filtre).")
                else:
                    dfr = pd.DataFrame([
                        {
                            "match_id": r.match_id,
                            "start_time": r.start_time,
                            "playlist_name": translate_playlist_name(r.playlist_name),
                            "pair_name": translate_pair_name(r.pair_name),
                            "same_team": r.same_team,
                            "my_team_id": r.my_team_id,
                            "my_outcome": r.my_outcome,
                            "friend_team_id": r.friend_team_id,
                            "friend_outcome": r.friend_outcome,
                        }
                        for r in rows
                    ])
                    dfr["start_time"] = pd.to_datetime(dfr["start_time"], utc=True).dt.tz_convert(None)
                    dfr = dfr.sort_values("start_time", ascending=False)

                    outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
                    dfr["my_outcome_label"] = dfr["my_outcome"].map(outcome_map).fillna("?")
                    counts = dfr["my_outcome_label"].value_counts().reindex(
                        ["Victoire", "Défaite", "Égalité", "Non terminé", "?"], fill_value=0
                    )
                    colors = HALO_COLORS.as_dict()
                    fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color=colors["cyan"])])
                    fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
                    st.plotly_chart(fig, width="stretch")

                    st.dataframe(
                        dfr[
                            ["start_time", "playlist_name", "pair_name", "same_team",
                             "my_team_id", "my_outcome", "friend_team_id", "friend_outcome", "match_id"]
                        ].reset_index(drop=True),
                        width="stretch",
                        hide_index=True,
                    )

    # --------------------------------------------------------------------------
    # Tab: Avec mes amis
    # --------------------------------------------------------------------------
    with tab_friends:
        st.caption("Vue dédiée aux matchs joués avec tes amis (définis via alias XUID).")
        apply_current_filters = st.toggle(
            "Appliquer les filtres actuels (période/sessions + map/playlist)",
            value=True,
        )

        top = cached_list_top_teammates(db_path, xuid.strip(), limit=20)
        default_two = [t[0] for t in top[:2]]
        all_other = cached_list_other_xuids(db_path, xuid.strip(), limit=500)
        
        ordered = []
        seen: set[str] = set()
        for xx, _cnt in top:
            if xx not in seen:
                ordered.append(xx)
                seen.add(xx)
        for xx in all_other:
            if xx not in seen:
                ordered.append(xx)
                seen.add(xx)

        opts_map = build_xuid_option_map(ordered, display_name_fn=display_name_from_xuid)
        picked_labels = st.multiselect(
            "Coéquipiers",
            options=list(opts_map.keys()),
            default=[k for k, v in opts_map.items() if v in default_two],
        )
        picked_xuids = [opts_map[lbl] for lbl in picked_labels if lbl in opts_map]
        # Forcé: les stats "avec mes amis" n'ont de sens que si vous êtes dans la même équipe.
        same_team_only_friends = True

        if len(picked_xuids) < 1:
            st.info("Sélectionne au moins un coéquipier.")
        else:
            # Agrégation sur tous les coéquipiers
            st.subheader("Par carte — avec mes amis (agrégé)")
            with st.spinner("Calcul du ratio par carte (amis)…"):
                min_matches_maps_friends = st.slider("Minimum de matchs par carte (amis)", 1, 30, 5, step=1)

                base_for_friends_all = dff if apply_current_filters else df
                all_match_ids: set[str] = set()
                for fx in picked_xuids:
                    rows = query_matches_with_friend(db_path, xuid.strip(), fx)
                    if same_team_only_friends:
                        rows = [r for r in rows if r.same_team]
                    for r in rows:
                        all_match_ids.add(str(r.match_id))

                sub_all = base_for_friends_all.loc[
                    base_for_friends_all["match_id"].astype(str).isin(all_match_ids)
                ].copy()
                breakdown_all = compute_map_breakdown(sub_all)
                breakdown_all = breakdown_all.loc[breakdown_all["matches"] >= int(min_matches_maps_friends)].copy()
                
                if breakdown_all.empty:
                    st.info("Pas assez de matchs avec tes amis (selon le filtre actuel).")
                else:
                    view_all = breakdown_all.head(20).iloc[::-1]
                    title = f"Ratio global par carte — avec mes amis (min {min_matches_maps_friends} matchs)"
                    st.plotly_chart(plot_map_ratio_with_winloss(view_all, title=title), width="stretch")

            # Vue trio (moi + 2 coéquipiers) : uniquement si on a au moins deux personnes.
            if len(picked_xuids) >= 2:
                f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
                f1_name = display_name_from_xuid(f1_xuid)
                f2_name = display_name_from_xuid(f2_xuid)
                st.subheader(f"Tous les trois (même équipe) — {f1_name} + {f2_name}")

                rows_m = query_matches_with_friend(db_path, xuid.strip(), f1_xuid)
                rows_c = query_matches_with_friend(db_path, xuid.strip(), f2_xuid)
                rows_m = [r for r in rows_m if r.same_team]
                rows_c = [r for r in rows_c if r.same_team]
                ids_m = {r.match_id for r in rows_m}
                ids_c = {r.match_id for r in rows_c}
                trio_ids = ids_m & ids_c

                base_for_trio = dff if apply_current_filters else df
                trio_ids = trio_ids & set(base_for_trio["match_id"].astype(str))

                if not trio_ids:
                    st.warning("Aucun match trouvé où vous êtes tous les trois dans la même équipe (avec les filtres actuels).")
                else:
                    # UX: bouton pour basculer les filtres globaux sur la dernière session où vous êtes tous les trois.
                    # On utilise la logique de sessions globale (mêmes paramètres gap) pour que la session sélectionnée
                    # corresponde bien aux options de la sidebar.
                    trio_ids_set = {str(x) for x in trio_ids}
                    base_for_session_pick = base  # même base que les filtres globaux (Firefight déjà exclu si besoin)
                    try:
                        gm = int(st.session_state.get("gap_minutes", 120))
                    except Exception:
                        gm = 120
                    base_s_trio = compute_sessions(base_for_session_pick, gap_minutes=gm)
                    trio_rows = base_s_trio.loc[base_s_trio["match_id"].astype(str).isin(trio_ids_set)].copy()
                    latest_label = None
                    if not trio_rows.empty:
                        latest_sid = int(trio_rows["session_id"].max())
                        latest_labels = trio_rows.loc[trio_rows["session_id"] == latest_sid, "session_label"].dropna().unique().tolist()
                        latest_label = latest_labels[0] if latest_labels else None

                    bcols = st.columns([2, 3])
                    with bcols[0]:
                        if st.button("Focus: dernière session (trio)", width="stretch", disabled=(latest_label is None)):
                            st.session_state["filter_mode"] = "Sessions"
                            if latest_label:
                                st.session_state["picked_session_label"] = latest_label
                                st.session_state["picked_sessions"] = [latest_label]
                            st.rerun()
                    with bcols[1]:
                        if latest_label:
                            st.caption(f"Cible: {latest_label} (gap {gm} min)")
                        else:
                            st.caption("Impossible de déterminer une session trio (données insuffisantes).")

                    # Charge les stats de chacun et aligne par match_id.
                    me_df = base_for_trio.loc[base_for_trio["match_id"].isin(trio_ids)].copy()
                    f1_df = load_df(db_path, f1_xuid)
                    f2_df = load_df(db_path, f2_xuid)
                    f1_df = f1_df.loc[f1_df["match_id"].isin(trio_ids)].copy()
                    f2_df = f2_df.loc[f2_df["match_id"].isin(trio_ids)].copy()

                    # Aligne sur les mêmes match_id et utilise le start_time de moi comme référence d'axe.
                    me_df = me_df.sort_values("start_time")
                    f1_df = f1_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
                    f2_df = f2_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
                    merged = me_df[["match_id", "start_time", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].merge(
                        f1_df.add_prefix("f1_"), left_on="match_id", right_on="f1_match_id", how="inner"
                    ).merge(
                        f2_df.add_prefix("f2_"), left_on="match_id", right_on="f2_match_id", how="inner"
                    )
                    if merged.empty:
                        st.warning("Impossible d'aligner les stats des 3 joueurs sur ces matchs.")
                    else:
                        merged = merged.sort_values("start_time")
                        # Reconstitue 3 DF alignées pour le plot.
                        d_self = merged[["start_time", "kills", "deaths", "assists", "ratio", "accuracy", "average_life_seconds"]].copy()
                        d_f1 = merged[["start_time", "f1_kills", "f1_deaths", "f1_assists", "f1_ratio", "f1_accuracy", "f1_average_life_seconds"]].rename(
                            columns={
                                "f1_kills": "kills",
                                "f1_deaths": "deaths",
                                "f1_assists": "assists",
                                "f1_ratio": "ratio",
                                "f1_accuracy": "accuracy",
                                "f1_average_life_seconds": "average_life_seconds",
                            }
                        )
                        d_f2 = merged[["start_time", "f2_kills", "f2_deaths", "f2_assists", "f2_ratio", "f2_accuracy", "f2_average_life_seconds"]].rename(
                            columns={
                                "f2_kills": "kills",
                                "f2_deaths": "deaths",
                                "f2_assists": "assists",
                                "f2_ratio": "ratio",
                                "f2_accuracy": "accuracy",
                                "f2_average_life_seconds": "average_life_seconds",
                            }
                        )

                        names = (me_name, f1_name, f2_name)
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="kills", names=names, title="Frags (tous les trois)", y_title="Frags"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="deaths", names=names, title="Morts (tous les trois)", y_title="Morts"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="assists", names=names, title="Assistances (tous les trois)", y_title="Assists"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="ratio", names=names, title="Ratio (tous les trois)", y_title="Ratio", y_format=".3f"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="accuracy", names=names, title="Précision (tous les trois)", y_title="%", y_suffix="%", y_format=".2f"),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(d_self, d_f1, d_f2, metric="average_life_seconds", names=names, title="Durée de vie moyenne (tous les trois)", y_title="Secondes", y_format=".1f"),
                            width="stretch",
                        )

                        st.subheader("Médailles (tous les trois)")
                        trio_match_ids = [str(x) for x in merged["match_id"].dropna().astype(str).tolist()]
                        if not trio_match_ids:
                            st.info("Impossible de déterminer la liste des matchs pour l'agrégation des médailles.")
                        else:
                            with st.spinner("Agrégation des médailles (tous les trois)…"):
                                top_self = load_top_medals(db_path, xuid.strip(), trio_match_ids, top_n=12)
                                top_f1 = load_top_medals(db_path, f1_xuid, trio_match_ids, top_n=12)
                                top_f2 = load_top_medals(db_path, f2_xuid, trio_match_ids, top_n=12)

                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.caption(f"{me_name}")
                                render_medals_grid(
                                    [{"name_id": int(n), "count": int(c)} for n, c in (top_self or [])],
                                    cols_per_row=6,
                                )
                            with c2:
                                st.caption(f"{f1_name}")
                                render_medals_grid(
                                    [{"name_id": int(n), "count": int(c)} for n, c in (top_f1 or [])],
                                    cols_per_row=6,
                                )
                            with c3:
                                st.caption(f"{f2_name}")
                                render_medals_grid(
                                    [{"name_id": int(n), "count": int(c)} for n, c in (top_f2 or [])],
                                    cols_per_row=6,
                                )

            # Stats individuelles par ami
            for fx in picked_xuids:
                name = display_name_from_xuid(fx)
                rows = query_matches_with_friend(db_path, xuid.strip(), fx)
                if same_team_only_friends:
                    rows = [r for r in rows if r.same_team]
                match_ids = {r.match_id for r in rows}

                base_for_friends = dff if apply_current_filters else df
                sub = base_for_friends.loc[base_for_friends["match_id"].isin(match_ids)].copy()
                st.subheader(f"Avec {name}")
                
                if sub.empty:
                    st.warning("Aucun match trouvé (avec les filtres actuels).")
                    continue

                rates_sub = compute_outcome_rates(sub)
                total_out = max(1, rates_sub.total)
                win_rate_sub = rates_sub.wins / total_out
                loss_rate_sub = rates_sub.losses / total_out
                global_ratio_sub = compute_global_ratio(sub)

                k = st.columns(3)
                k[0].metric("Matchs", f"{len(sub)}")
                k[1].metric("Win/Loss", f"{win_rate_sub*100:.1f}% / {loss_rate_sub*100:.1f}%")
                k[2].metric("Ratio global", f"{global_ratio_sub:.2f}" if global_ratio_sub is not None else "-")

                stats_sub = compute_aggregated_stats(sub)
                per_min = st.columns(3)
                per_min[0].metric("Frags / min", f"{stats_sub.kills_per_minute:.2f}" if stats_sub.kills_per_minute else "-")
                per_min[1].metric("Morts / min", f"{stats_sub.deaths_per_minute:.2f}" if stats_sub.deaths_per_minute else "-")
                per_min[2].metric("Assistances / min", f"{stats_sub.assists_per_minute:.2f}" if stats_sub.assists_per_minute else "-")

                # Chargement des stats de l'ami
                shared_ids = set(sub["match_id"].astype(str))
                friend_df = load_df(db_path, fx)
                friend_sub = friend_df.loc[friend_df["match_id"].astype(str).isin(shared_ids)].copy()

                # Graphiques Kills/Deaths/Ratio
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_timeseries(sub, title=f"{me_name} — matchs avec {name}"), width="stretch")
                with c2:
                    if friend_sub.empty:
                        st.warning("Impossible de charger les stats de l'ami sur les matchs partagés.")
                    else:
                        st.plotly_chart(plot_timeseries(friend_sub, title=f"{name} — matchs avec {me_name}"), width="stretch")

                # Graphiques par minute
                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(
                        plot_per_minute_timeseries(sub, title=f"{me_name} — stats/min (avec {name})"),
                        width="stretch",
                    )
                with c4:
                    if not friend_sub.empty:
                        st.plotly_chart(
                            plot_per_minute_timeseries(friend_sub, title=f"{name} — stats/min (avec {me_name})"),
                            width="stretch",
                        )

                # Graphiques Average Life
                c5, c6 = st.columns(2)
                with c5:
                    if not sub.dropna(subset=["average_life_seconds"]).empty:
                        st.plotly_chart(
                            plot_average_life(sub, title=f"{me_name} — Durée de vie (avec {name})"),
                            width="stretch",
                        )
                with c6:
                    if not friend_sub.empty and not friend_sub.dropna(subset=["average_life_seconds"]).empty:
                        st.plotly_chart(
                            plot_average_life(friend_sub, title=f"{name} — Durée de vie (avec {me_name})"),
                            width="stretch",
                        )

                # Médailles (simple) : top médailles sur les matchs partagés
                st.subheader("Médailles (matchs partagés)")
                shared_list = sorted({str(x) for x in shared_ids if str(x).strip()})
                if not shared_list:
                    st.info("Aucun match partagé pour calculer les médailles.")
                else:
                    with st.spinner("Agrégation des médailles (moi + ami)…"):
                        my_top = load_top_medals(db_path, xuid.strip(), shared_list, top_n=12)
                        fr_top = load_top_medals(db_path, fx, shared_list, top_n=12)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.caption(f"{me_name}")
                        render_medals_grid(
                            [{"name_id": int(n), "count": int(c)} for n, c in (my_top or [])],
                            cols_per_row=6,
                        )
                    with m2:
                        st.caption(f"{name}")
                        render_medals_grid(
                            [{"name_id": int(n), "count": int(c)} for n, c in (fr_top or [])],
                            cols_per_row=6,
                        )

    # --------------------------------------------------------------------------
    # Tab: Ratio par cartes
    # --------------------------------------------------------------------------
    with tab_maps:
        st.caption("Compare tes performances par map.")

        scope = st.radio(
            "Scope",
            options=["Moi (filtres actuels)", "Moi (toutes les parties)", "Avec Madina972", "Avec Chocoboflor"],
            horizontal=True,
        )
        min_matches = st.slider("Minimum de matchs par map", 1, 30, 5, step=1)

        if scope == "Moi (toutes les parties)":
            base_scope = base
        elif scope == "Avec Madina972":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2533274858283686")
            rows = [r for r in rows if r.same_team]
            match_ids = {r.match_id for r in rows}
            base_scope = base.loc[base["match_id"].isin(match_ids)].copy()
        elif scope == "Avec Chocoboflor":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2535469190789936")
            rows = [r for r in rows if r.same_team]
            match_ids = {r.match_id for r in rows}
            base_scope = base.loc[base["match_id"].isin(match_ids)].copy()
        else:
            base_scope = dff

        with st.spinner("Calcul des stats par carte…"):
            breakdown = compute_map_breakdown(base_scope)
            breakdown = breakdown.loc[breakdown["matches"] >= int(min_matches)].copy()

        if breakdown.empty:
            st.warning("Pas assez de matchs par map avec ces filtres.")
        else:
            metric = st.selectbox(
                "Métrique",
                options=[
                    ("ratio_global", "Ratio global"),
                    ("win_rate", "Win rate"),
                    ("accuracy_avg", "Précision moyenne"),
                ],
                format_func=lambda x: x[1],
            )
            key, label = metric

            view = breakdown.head(20).iloc[::-1]
            title = f"{label} par carte — {scope} (min {min_matches} matchs)"
            if key == "ratio_global":
                fig = plot_map_ratio_with_winloss(view, title=title)
            else:
                fig = plot_map_comparison(view, key, title=title)

            if key in ("win_rate",):
                fig.update_xaxes(tickformat=".0%")
            if key in ("accuracy_avg",):
                fig.update_xaxes(ticksuffix="%")

            st.plotly_chart(fig, width="stretch")

            tbl = breakdown.copy()
            tbl["win_rate"] = (tbl["win_rate"] * 100).round(1)
            tbl["loss_rate"] = (tbl["loss_rate"] * 100).round(1)
            tbl["accuracy_avg"] = tbl["accuracy_avg"].round(2)
            tbl["ratio_global"] = tbl["ratio_global"].round(2)

            st.dataframe(
                tbl.rename(
                    columns={
                        "map_name": "Carte",
                        "matches": "Parties",
                        "accuracy_avg": "Précision moy. (%)",
                        "win_rate": "Taux victoire (%)",
                        "loss_rate": "Taux défaite (%)",
                        "ratio_global": "Ratio global",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------------------------
    # Tab: Historique des parties
    # --------------------------------------------------------------------------
    with tab_table:
        st.subheader("Historique des parties")
        dff_table = dff.copy()
        if "playlist_fr" not in dff_table.columns:
            dff_table["playlist_fr"] = dff_table["playlist_name"].apply(translate_playlist_name)
        dff_table["match_url"] = (
            "https://www.halowaypoint.com/halo-infinite/players/"
            + waypoint_player.strip()
            + "/matches/"
            + dff_table["match_id"].astype(str)
        )

        outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
        dff_table["outcome_label"] = dff_table["outcome"].map(outcome_map).fillna("-")

        show_cols = [
            "match_url", "start_time", "map_name", "playlist_fr", "outcome_label",
            "kda", "kills", "deaths", "max_killing_spree", "headshot_kills",
            "average_life_seconds", "assists", "accuracy", "ratio",
        ]
        table = dff_table[show_cols].sort_values("start_time", ascending=False).reset_index(drop=True)

        def _style_outcome(v: str) -> str:
            s = (v or "").strip().lower()
            if s == "victoire":
                return "color: #1B5E20; font-weight: 700;"
            if s in ("défaite", "defaite"):
                return "color: #B71C1C; font-weight: 700;"
            if s in ("égalité", "egalite", "non terminé", "non termine"):
                return "color: #8E6CFF; font-weight: 700;"
            return ""

        def _style_kda(v) -> str:
            try:
                x = float(v)
            except Exception:
                return ""
            if x > 0:
                return "color: #1B5E20; font-weight: 700;"
            if x < 0:
                return "color: #B71C1C; font-weight: 700;"
            return "color: #424242;"

        styled = table.style.map(_style_outcome, subset=["outcome_label"]).map(_style_kda, subset=["kda"])

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "match_url": st.column_config.LinkColumn(
                    "Consulter sur HaloWaypoint",
                    display_text="Ouvrir",
                ),
                "map_name": st.column_config.TextColumn("Carte"),
                "playlist_fr": st.column_config.TextColumn("Playlist"),
                "outcome_label": st.column_config.TextColumn("Résultat"),
                "kda": st.column_config.NumberColumn("FDA", format="%.2f"),
                "kills": st.column_config.NumberColumn("Frags"),
                "deaths": st.column_config.NumberColumn("Morts"),
                "assists": st.column_config.NumberColumn("Assistances"),
                "accuracy": st.column_config.NumberColumn("Précision (%)", format="%.2f"),
            },
        )

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger CSV",
            data=csv,
            file_name="openspartan_matches.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
