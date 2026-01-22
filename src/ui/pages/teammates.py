"""Page Mes coéquipiers.

Analyse des statistiques avec les coéquipiers fréquents.
"""

from __future__ import annotations

import html as html_lib
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import HALO_COLORS
from src.analysis import compute_aggregated_stats, compute_map_breakdown, compute_outcome_rates, compute_global_ratio
from src.ui import display_name_from_xuid, translate_playlist_name, get_hero_html, get_profile_appearance
from src.ui.cache import (
    cached_compute_sessions_db,
    cached_friend_matches_df,
    cached_load_player_match_result,
    cached_query_matches_with_friend,
    cached_same_team_match_ids_with_friend,
    load_df,
)
from src.ui.medals import render_medals_grid
from src.ui.player_assets import ensure_local_image_path
from src.visualization import (
    plot_average_life,
    plot_map_ratio_with_winloss,
    plot_per_minute_timeseries,
    plot_timeseries,
    plot_trio_metric,
)


def _format_datetime_fr_hm(dt: pd.Timestamp | None) -> str:
    """Formate une date FR avec heures/minutes."""
    if pd.isna(dt):
        return "-"
    try:
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(dt)


def _normalize_mode_label(pair_name: str | None) -> str | None:
    """Normalise un pair_name en label UI."""
    from src.ui.translations import translate_pair_name
    return translate_pair_name(pair_name) if pair_name else None


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


def _clear_min_matches_maps_friends_auto() -> None:
    """Callback pour désactiver le mode auto du slider coéquipiers."""
    st.session_state["_min_matches_maps_friends_auto"] = False


def render_teammates_page(
    df: pd.DataFrame,
    dff: pd.DataFrame,
    base: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: tuple[int, int] | None,
    aliases_key: int | None,
    settings: object,
    picked_session_labels: list[str] | None,
    include_firefight: bool,
    waypoint_player: str,
    build_friends_opts_map_fn,
    assign_player_colors_fn,
    plot_multi_metric_bars_fn,
    top_medals_fn,
) -> None:
    """Affiche la page Mes coéquipiers.

    Args:
        df: DataFrame complet des matchs.
        dff: DataFrame filtré des matchs.
        base: DataFrame de base (après filtres Firefight).
        me_name: Nom affiché du joueur.
        xuid: XUID du joueur.
        db_path: Chemin vers la base de données.
        db_key: Clé de cache de la DB.
        aliases_key: Clé de cache des alias.
        settings: Paramètres de l'application.
        picked_session_labels: Labels des sessions sélectionnées.
        include_firefight: Inclure Firefight dans les stats.
        waypoint_player: Nom Waypoint du joueur.
        build_friends_opts_map_fn: Fonction pour construire la map des coéquipiers.
        assign_player_colors_fn: Fonction pour assigner les couleurs aux joueurs.
        plot_multi_metric_bars_fn: Fonction pour tracer les barres multi-métriques.
        top_medals_fn: Fonction pour récupérer les top médailles.
    """
    apply_current_filters_teammates = st.toggle(
        "Appliquer les filtres actuels (période/sessions + map/playlist)",
        value=True,
        key="apply_current_filters_teammates",
    )
    same_team_only_teammates = st.checkbox("Même équipe", value=True, key="teammates_same_team_only")

    show_smooth_teammates = st.toggle(
        "Afficher les courbes lissées",
        value=bool(st.session_state.get("teammates_show_smooth", True)),
        key="teammates_show_smooth",
        help="Active/désactive les courbes de moyenne lissée sur les graphes de cette section.",
    )

    opts_map, default_labels = build_friends_opts_map_fn(db_path, xuid.strip(), db_key, aliases_key)
    picked_labels = st.multiselect(
        "Coéquipiers",
        options=list(opts_map.keys()),
        default=default_labels,
        key="teammates_picked_labels",
    )
    picked_xuids = [opts_map[lbl] for lbl in picked_labels if lbl in opts_map]

    # Afficher les Spartan ID cards des coéquipiers sélectionnés en grille 2 colonnes
    _render_teammate_cards(picked_xuids, settings)

    if len(picked_xuids) < 1:
        st.info("Sélectionne au moins un coéquipier.")
    elif len(picked_xuids) == 1:
        _render_single_teammate_view(
            df=df,
            dff=dff,
            me_name=me_name,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            picked_xuids=picked_xuids,
            apply_current_filters=apply_current_filters_teammates,
            same_team_only=same_team_only_teammates,
            show_smooth=show_smooth_teammates,
            assign_player_colors_fn=assign_player_colors_fn,
            plot_multi_metric_bars_fn=plot_multi_metric_bars_fn,
            top_medals_fn=top_medals_fn,
        )
    else:
        _render_multi_teammate_view(
            df=df,
            dff=dff,
            base=base,
            me_name=me_name,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            picked_xuids=picked_xuids,
            picked_session_labels=picked_session_labels,
            apply_current_filters=apply_current_filters_teammates,
            same_team_only=same_team_only_teammates,
            show_smooth=show_smooth_teammates,
            include_firefight=include_firefight,
            waypoint_player=waypoint_player,
            assign_player_colors_fn=assign_player_colors_fn,
            plot_multi_metric_bars_fn=plot_multi_metric_bars_fn,
            top_medals_fn=top_medals_fn,
        )


def _render_teammate_cards(picked_xuids: list[str], settings: object) -> None:
    """Affiche les cartes Spartan des coéquipiers sélectionnés."""
    if not picked_xuids:
        return

    teammate_cards_html = []
    dl_enabled = bool(getattr(settings, "profile_assets_download_enabled", False)) or bool(
        getattr(settings, "profile_api_enabled", False)
    )
    refresh_h = int(getattr(settings, "profile_assets_auto_refresh_hours", 0) or 0)
    api_refresh_h = int(getattr(settings, "profile_api_auto_refresh_hours", 0) or 0)

    for t_xuid in picked_xuids:
        t_name = display_name_from_xuid(t_xuid)
        t_app, _ = get_profile_appearance(
            xuid=t_xuid,
            enabled=True,
            refresh_hours=api_refresh_h,
        )
        t_emblem = getattr(t_app, "emblem_image_url", None) if t_app else None
        t_backdrop = getattr(t_app, "backdrop_image_url", None) if t_app else None
        t_nameplate = getattr(t_app, "nameplate_image_url", None) if t_app else None
        t_service_tag = getattr(t_app, "service_tag", None) if t_app else None

        t_emblem_path = ensure_local_image_path(t_emblem, prefix="emblem", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
        t_backdrop_path = ensure_local_image_path(t_backdrop, prefix="backdrop", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)
        t_nameplate_path = ensure_local_image_path(t_nameplate, prefix="nameplate", download_enabled=dl_enabled, auto_refresh_hours=refresh_h)

        card_html = get_hero_html(
            player_name=t_name,
            service_tag=t_service_tag,
            emblem_path=t_emblem_path,
            backdrop_path=t_backdrop_path,
            nameplate_path=t_nameplate_path,
            grid_mode=True,
        )
        teammate_cards_html.append(card_html)

    if teammate_cards_html:
        cols = st.columns(2)
        for i, card_html in enumerate(teammate_cards_html):
            with cols[i % 2]:
                st.markdown(card_html, unsafe_allow_html=True)


def _render_single_teammate_view(
    df: pd.DataFrame,
    dff: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: tuple[int, int] | None,
    picked_xuids: list[str],
    apply_current_filters: bool,
    same_team_only: bool,
    show_smooth: bool,
    assign_player_colors_fn,
    plot_multi_metric_bars_fn,
    top_medals_fn,
) -> None:
    """Vue pour un seul coéquipier sélectionné."""
    friend_xuid = picked_xuids[0]
    with st.spinner("Chargement des matchs avec ce coéquipier…"):
        dfr = cached_friend_matches_df(
            db_path,
            xuid.strip(),
            friend_xuid,
            same_team_only=bool(same_team_only),
            db_key=db_key,
        )
        if dfr.empty:
            st.warning("Aucun match trouvé avec ce coéquipier (selon le filtre).")
            return

        outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
        dfr["my_outcome_label"] = dfr["my_outcome"].map(outcome_map).fillna("?")
        counts = dfr["my_outcome_label"].value_counts().reindex(
            ["Victoire", "Défaite", "Égalité", "Non terminé", "?"], fill_value=0
        )
        colors = HALO_COLORS.as_dict()
        fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color=colors["cyan"])])
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, width="stretch")

        with st.expander("Détails des matchs (joueur vs joueur)", expanded=False):
            st.dataframe(
                dfr[
                    [
                        "start_time",
                        "playlist_name",
                        "pair_name",
                        "same_team",
                        "my_team_id",
                        "my_outcome",
                        "friend_team_id",
                        "friend_outcome",
                        "match_id",
                    ]
                ].reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )

        base_for_friend = dff if apply_current_filters else df
        shared_ids = set(dfr["match_id"].astype(str))
        sub = base_for_friend.loc[base_for_friend["match_id"].astype(str).isin(shared_ids)].copy()

        if sub.empty:
            st.info("Aucun match à afficher avec les filtres actuels (période/sessions + map/playlist).")
            return

        name = display_name_from_xuid(friend_xuid)

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
        per_min[0].metric(
            "Frags / min",
            f"{stats_sub.kills_per_minute:.2f}" if stats_sub.kills_per_minute else "-",
        )
        per_min[1].metric(
            "Morts / min",
            f"{stats_sub.deaths_per_minute:.2f}" if stats_sub.deaths_per_minute else "-",
        )
        per_min[2].metric(
            "Assistances / min",
            f"{stats_sub.assists_per_minute:.2f}" if stats_sub.assists_per_minute else "-",
        )

        friend_df = load_df(db_path, friend_xuid, db_key=db_key)
        friend_sub = friend_df.loc[friend_df["match_id"].astype(str).isin(shared_ids)].copy()

        # Graphes côte à côte
        _render_comparison_charts(
            sub=sub,
            friend_sub=friend_sub,
            me_name=me_name,
            friend_name=name,
            friend_xuid=friend_xuid,
        )

        # Graphes de barres (folie meurtrière, headshots)
        series = [(me_name, sub)]
        if not friend_sub.empty:
            series.append((name, friend_sub))
        colors_by_name = assign_player_colors_fn([n for n, _ in series])

        _render_metric_bar_charts(
            series=series,
            colors_by_name=colors_by_name,
            show_smooth=show_smooth,
            key_suffix=friend_xuid,
            plot_fn=plot_multi_metric_bars_fn,
        )

        # Médailles
        st.subheader("Médailles (matchs partagés)")
        shared_list = sorted({str(x) for x in shared_ids if str(x).strip()})
        if not shared_list:
            st.info("Aucun match partagé pour calculer les médailles.")
        else:
            with st.spinner("Agrégation des médailles (moi + coéquipier)…"):
                my_top = top_medals_fn(db_path, xuid.strip(), shared_list, top_n=12, db_key=db_key)
                fr_top = top_medals_fn(db_path, friend_xuid, shared_list, top_n=12, db_key=db_key)

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


def _render_comparison_charts(
    sub: pd.DataFrame,
    friend_sub: pd.DataFrame,
    me_name: str,
    friend_name: str,
    friend_xuid: str,
) -> None:
    """Affiche les graphes de comparaison côte à côte."""
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_timeseries(sub, title=f"{me_name} — matchs avec {friend_name}"),
            width="stretch",
            key=f"friend_ts_me_{friend_xuid}",
        )
    with c2:
        if friend_sub.empty:
            st.warning("Impossible de charger les stats du coéquipier sur les matchs partagés.")
        else:
            st.plotly_chart(
                plot_timeseries(friend_sub, title=f"{friend_name} — matchs avec {me_name}"),
                width="stretch",
                key=f"friend_ts_fr_{friend_xuid}",
            )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_per_minute_timeseries(sub, title=f"{me_name} — stats/min (avec {friend_name})"),
            width="stretch",
            key=f"friend_pm_me_{friend_xuid}",
        )
    with c4:
        if not friend_sub.empty:
            st.plotly_chart(
                plot_per_minute_timeseries(
                    friend_sub,
                    title=f"{friend_name} — stats/min (avec {me_name})",
                ),
                width="stretch",
                key=f"friend_pm_fr_{friend_xuid}",
            )

    c5, c6 = st.columns(2)
    with c5:
        if not sub.dropna(subset=["average_life_seconds"]).empty:
            st.plotly_chart(
                plot_average_life(sub, title=f"{me_name} — Durée de vie (avec {friend_name})"),
                width="stretch",
                key=f"friend_life_me_{friend_xuid}",
            )
    with c6:
        if not friend_sub.empty and not friend_sub.dropna(subset=["average_life_seconds"]).empty:
            st.plotly_chart(
                plot_average_life(friend_sub, title=f"{friend_name} — Durée de vie (avec {me_name})"),
                width="stretch",
                key=f"friend_life_fr_{friend_xuid}",
            )


def _render_metric_bar_charts(
    series: list[tuple[str, pd.DataFrame]],
    colors_by_name: dict[str, str],
    show_smooth: bool,
    key_suffix: str,
    plot_fn,
) -> None:
    """Affiche les graphes de barres pour les métriques."""
    fig_spree = plot_fn(
        series,
        metric_col="max_killing_spree",
        title="Folie meurtrière (max)",
        y_axis_title="Folie meurtrière (max)",
        hover_label="folie meurtrière",
        colors=colors_by_name,
        smooth_window=10,
        show_smooth_lines=show_smooth,
    )
    if fig_spree is None:
        st.info("Aucune donnée de folie meurtrière (max) sur ces matchs.")
    else:
        st.plotly_chart(fig_spree, width="stretch", key=f"friend_spree_multi_{key_suffix}")

    fig_hs = plot_fn(
        series,
        metric_col="headshot_kills",
        title="Tirs à la tête",
        y_axis_title="Tirs à la tête",
        hover_label="tirs à la tête",
        colors=colors_by_name,
        smooth_window=10,
        show_smooth_lines=show_smooth,
    )
    if fig_hs is None:
        st.info("Aucune donnée de tirs à la tête sur ces matchs.")
    else:
        st.plotly_chart(fig_hs, width="stretch", key=f"friend_hs_multi_{key_suffix}")


def _render_multi_teammate_view(
    df: pd.DataFrame,
    dff: pd.DataFrame,
    base: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: tuple[int, int] | None,
    picked_xuids: list[str],
    picked_session_labels: list[str] | None,
    apply_current_filters: bool,
    same_team_only: bool,
    show_smooth: bool,
    include_firefight: bool,
    waypoint_player: str,
    assign_player_colors_fn,
    plot_multi_metric_bars_fn,
    top_medals_fn,
) -> None:
    """Vue pour plusieurs coéquipiers sélectionnés."""
    st.subheader("Par carte — avec mes coéquipiers")
    with st.spinner("Calcul du ratio par carte (coéquipiers)…"):
        current_mode = st.session_state.get("filter_mode")
        latest_session_label = st.session_state.get("_latest_session_label")
        trio_latest_label = st.session_state.get("_trio_latest_session_label")

        selected_session = None
        if current_mode == "Sessions" and isinstance(picked_session_labels, list) and len(picked_session_labels) == 1:
            selected_session = picked_session_labels[0]

        is_last_session = bool(selected_session and selected_session == latest_session_label)
        is_last_trio_session = bool(selected_session and isinstance(trio_latest_label, str) and selected_session == trio_latest_label)

        if is_last_session or is_last_trio_session:
            last_applied = st.session_state.get("_friends_min_matches_last_session_label")
            if last_applied != selected_session:
                st.session_state["min_matches_maps_friends"] = 1
                st.session_state["_min_matches_maps_friends_auto"] = True
                st.session_state["_friends_min_matches_last_session_label"] = selected_session

        min_matches_maps_friends = st.slider(
            "Minimum de matchs par carte",
            1,
            30,
            1,
            step=1,
            key="min_matches_maps_friends",
            on_change=_clear_min_matches_maps_friends_auto,
        )

        base_for_friends_all = dff if apply_current_filters else df
        all_match_ids: set[str] = set()
        per_friend_ids: dict[str, set[str]] = {}
        for fx in picked_xuids:
            ids: set[str] = set()
            if bool(same_team_only):
                ids = {str(x) for x in cached_same_team_match_ids_with_friend(db_path, xuid.strip(), fx, db_key=db_key)}
            else:
                rows = cached_query_matches_with_friend(db_path, xuid.strip(), fx, db_key=db_key)
                ids = {str(r.match_id) for r in rows}
            per_friend_ids[str(fx)] = ids
            all_match_ids.update(ids)

        sub_all = base_for_friends_all.loc[
            base_for_friends_all["match_id"].astype(str).isin(all_match_ids)
        ].copy()

        use_xuids = picked_xuids

        series: list[tuple[str, pd.DataFrame]] = [(me_name, sub_all)]
        with st.spinner("Chargement des stats des coéquipiers…"):
            for fx in use_xuids:
                ids = per_friend_ids.get(str(fx), set())
                if not ids:
                    continue
                try:
                    fr_df = load_df(db_path, str(fx), db_key=db_key)
                except Exception:
                    continue
                fr_sub = fr_df.loc[fr_df["match_id"].astype(str).isin(ids)].copy()
                if fr_sub.empty:
                    continue
                series.append((display_name_from_xuid(str(fx)), fr_sub))
        colors_by_name = assign_player_colors_fn([n for n, _ in series])

        breakdown_all = compute_map_breakdown(sub_all)
        breakdown_all = breakdown_all.loc[breakdown_all["matches"] >= int(min_matches_maps_friends)].copy()

        if breakdown_all.empty:
            st.info("Pas assez de matchs avec tes coéquipiers (selon le filtre actuel).")
        else:
            view_all = breakdown_all.head(20).iloc[::-1]
            title = f"Ratio global par carte — avec mes coéquipiers (min {min_matches_maps_friends} matchs)"
            st.plotly_chart(plot_map_ratio_with_winloss(view_all, title=title), width="stretch")

            st.subheader("Historique — matchs avec mes coéquipiers")

        if sub_all.empty:
            st.info("Aucun match trouvé avec tes coéquipiers (selon le filtre actuel).")
        else:
            _render_friends_history_table(sub_all, db_path, xuid, db_key, waypoint_player)

        rendered_bottom_charts = False

    # Vue trio (moi + 2 coéquipiers)
    if len(picked_xuids) >= 2:
        rendered_bottom_charts = _render_trio_view(
            df=df,
            dff=dff,
            base=base,
            me_name=me_name,
            xuid=xuid,
            db_path=db_path,
            db_key=db_key,
            picked_xuids=picked_xuids,
            apply_current_filters=apply_current_filters,
            include_firefight=include_firefight,
            series=series,
            colors_by_name=colors_by_name,
            show_smooth=show_smooth,
            assign_player_colors_fn=assign_player_colors_fn,
            plot_multi_metric_bars_fn=plot_multi_metric_bars_fn,
            top_medals_fn=top_medals_fn,
        )

    if not rendered_bottom_charts:
        _render_metric_bar_charts(
            series=series,
            colors_by_name=colors_by_name,
            show_smooth=show_smooth,
            key_suffix=f"{len(series)}",
            plot_fn=plot_multi_metric_bars_fn,
        )


def _render_friends_history_table(
    sub_all: pd.DataFrame,
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
    waypoint_player: str,
) -> None:
    """Affiche le tableau d'historique des matchs avec coéquipiers."""
    friends_table = sub_all.copy()
    friends_table["start_time_fr"] = friends_table["start_time"].apply(_format_datetime_fr_hm)
    if "playlist_fr" not in friends_table.columns:
        friends_table["playlist_fr"] = friends_table["playlist_name"].apply(translate_playlist_name)
    if "mode_ui" in friends_table.columns:
        friends_table["mode"] = friends_table["mode_ui"].apply(
            lambda x: x if (x is not None and str(x).strip()) else None
        )
    else:
        friends_table["mode"] = None
    if friends_table["mode"].isna().any() and "pair_name" in friends_table.columns:
        mask = friends_table["mode"].isna()
        friends_table.loc[mask, "mode"] = friends_table.loc[mask, "pair_name"].apply(
            lambda p: _normalize_mode_label(str(p) if p is not None else None)
        )
    friends_table["mode"] = friends_table["mode"].fillna("-")
    friends_table["outcome_label"] = friends_table["outcome"].map({2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}).fillna("-")
    friends_table["score"] = friends_table.apply(
        lambda r: _format_score_label(r.get("my_team_score"), r.get("enemy_team_score")), axis=1
    )

    def _mmr_tuple(match_id: str):
        pm = cached_load_player_match_result(db_path, str(match_id), xuid.strip(), db_key=db_key)
        if not isinstance(pm, dict):
            return (None, None)
        return (pm.get("team_mmr"), pm.get("enemy_mmr"))

    mmr_pairs = friends_table["match_id"].astype(str).apply(_mmr_tuple)
    friends_table["team_mmr"] = mmr_pairs.apply(lambda t: t[0])
    friends_table["enemy_mmr"] = mmr_pairs.apply(lambda t: t[1])
    friends_table["delta_mmr"] = friends_table.apply(
        lambda r: (float(r.get("team_mmr")) - float(r.get("enemy_mmr")))
        if (r.get("team_mmr") is not None and r.get("enemy_mmr") is not None)
        else None,
        axis=1,
    )
    wp = str(waypoint_player or "").strip()
    if wp:
        friends_table["match_url"] = (
            "https://www.halowaypoint.com/halo-infinite/players/"
            + wp
            + "/matches/"
            + friends_table["match_id"].astype(str)
        )
    else:
        friends_table["match_url"] = ""

    view = friends_table.sort_values("start_time", ascending=False).head(250).reset_index(drop=True)

    def _fmt(v) -> str:
        if v is None:
            return "-"
        try:
            if v != v:
                return "-"
        except Exception:
            pass
        s = str(v)
        return s if s.strip() else "-"

    def _fmt_mmr_int(v) -> str:
        if v is None:
            return "-"
        try:
            if v != v:
                return "-"
        except Exception:
            pass
        try:
            return str(int(round(float(v))))
        except Exception:
            return _fmt(v)

    colors = HALO_COLORS.as_dict()

    def _outcome_style(label: str) -> str:
        v = str(label or "").strip().casefold()
        if v.startswith("victoire"):
            return f"color:{colors['green']}; font-weight:800"
        if v.startswith("défaite") or v.startswith("defaite"):
            return f"color:{colors['red']}; font-weight:800"
        if v.startswith("égalité") or v.startswith("egalite"):
            return f"color:{colors['violet']}; font-weight:800"
        if v.startswith("non"):
            return f"color:{colors['violet']}; font-weight:800"
        return "opacity:0.92"

    cols = [
        ("Match", "_app"),
        ("HaloWaypoint", "match_url"),
        ("Date", "start_time_fr"),
        ("Carte", "map_name"),
        ("Playlist", "playlist_fr"),
        ("Mode", "mode"),
        ("Résultat", "outcome_label"),
        ("Score", "score"),
        ("MMR équipe", "team_mmr"),
        ("MMR adverse", "enemy_mmr"),
        ("Écart MMR", "delta_mmr"),
    ]

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
                tds.append(f"<td style='{_outcome_style(val)}'>{html_lib.escape(val)}</td>")
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


def _render_trio_view(
    df: pd.DataFrame,
    dff: pd.DataFrame,
    base: pd.DataFrame,
    me_name: str,
    xuid: str,
    db_path: str,
    db_key: tuple[int, int] | None,
    picked_xuids: list[str],
    apply_current_filters: bool,
    include_firefight: bool,
    series: list[tuple[str, pd.DataFrame]],
    colors_by_name: dict[str, str],
    show_smooth: bool,
    assign_player_colors_fn,
    plot_multi_metric_bars_fn,
    top_medals_fn,
) -> bool:
    """Affiche la vue trio (moi + 2 coéquipiers). Retourne True si les graphes du bas ont été rendus."""
    f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
    f1_name = display_name_from_xuid(f1_xuid)
    f2_name = display_name_from_xuid(f2_xuid)
    st.subheader(f"Tous les trois — {f1_name} + {f2_name}")

    ids_m = set(
        cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f1_xuid, db_key=db_key)
    )
    ids_c = set(
        cached_same_team_match_ids_with_friend(db_path, xuid.strip(), f2_xuid, db_key=db_key)
    )
    trio_ids = ids_m & ids_c

    base_for_trio = dff if apply_current_filters else df
    trio_ids = trio_ids & set(base_for_trio["match_id"].astype(str))

    if not trio_ids:
        st.warning("Aucun match trouvé où vous êtes tous les trois dans la même équipe (avec les filtres actuels).")
        return False

    trio_ids_set = {str(x) for x in trio_ids}
    try:
        gm = int(st.session_state.get("gap_minutes", 120))
    except Exception:
        gm = 120
    base_s_trio = cached_compute_sessions_db(
        db_path,
        xuid.strip(),
        db_key,
        include_firefight,
        gm,
    )
    trio_rows = base_s_trio.loc[base_s_trio["match_id"].astype(str).isin(trio_ids_set)].copy()
    latest_label = None
    if not trio_rows.empty:
        latest_sid = int(trio_rows["session_id"].max())
        latest_labels = trio_rows.loc[trio_rows["session_id"] == latest_sid, "session_label"].dropna().unique().tolist()
        latest_label = latest_labels[0] if latest_labels else None

    st.session_state["_trio_latest_session_label"] = latest_label
    if latest_label:
        st.caption(f"Dernière session trio détectée : {latest_label} (gap {gm} min).")
    else:
        st.caption("Impossible de déterminer une session trio (données insuffisantes).")

    me_df = base_for_trio.loc[base_for_trio["match_id"].isin(trio_ids)].copy()
    f1_df = load_df(db_path, f1_xuid, db_key=db_key)
    f2_df = load_df(db_path, f2_xuid, db_key=db_key)
    f1_df = f1_df.loc[f1_df["match_id"].isin(trio_ids)].copy()
    f2_df = f2_df.loc[f2_df["match_id"].isin(trio_ids)].copy()

    me_df = me_df.sort_values("start_time")

    me_stats = compute_aggregated_stats(me_df)
    f1_stats = compute_aggregated_stats(f1_df)
    f2_stats = compute_aggregated_stats(f2_df)
    trio_per_min = pd.DataFrame(
        [
            {
                "Joueur": me_name,
                "Frags/min": round(float(me_stats.kills_per_minute), 2) if me_stats.kills_per_minute else None,
                "Morts/min": round(float(me_stats.deaths_per_minute), 2) if me_stats.deaths_per_minute else None,
                "Assists/min": round(float(me_stats.assists_per_minute), 2) if me_stats.assists_per_minute else None,
            },
            {
                "Joueur": f1_name,
                "Frags/min": round(float(f1_stats.kills_per_minute), 2) if f1_stats.kills_per_minute else None,
                "Morts/min": round(float(f1_stats.deaths_per_minute), 2) if f1_stats.deaths_per_minute else None,
                "Assists/min": round(float(f1_stats.assists_per_minute), 2) if f1_stats.assists_per_minute else None,
            },
            {
                "Joueur": f2_name,
                "Frags/min": round(float(f2_stats.kills_per_minute), 2) if f2_stats.kills_per_minute else None,
                "Morts/min": round(float(f2_stats.deaths_per_minute), 2) if f2_stats.deaths_per_minute else None,
                "Assists/min": round(float(f2_stats.assists_per_minute), 2) if f2_stats.assists_per_minute else None,
            },
        ]
    )
    st.subheader("Stats par minute")
    st.dataframe(trio_per_min, width="stretch", hide_index=True)

    f1_df = f1_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
    f2_df = f2_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].copy()
    merged = me_df[["match_id", "start_time", "kills", "deaths", "assists", "accuracy", "ratio", "average_life_seconds"]].merge(
        f1_df.add_prefix("f1_"), left_on="match_id", right_on="f1_match_id", how="inner"
    ).merge(
        f2_df.add_prefix("f2_"), left_on="match_id", right_on="f2_match_id", how="inner"
    )
    if merged.empty:
        st.warning("Impossible d'aligner les stats des 3 joueurs sur ces matchs.")
        return False

    merged = merged.sort_values("start_time")
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
        plot_trio_metric(d_self, d_f1, d_f2, metric="kills", names=names, title="Frags", y_title="Frags"),
        width="stretch",
        key=f"trio_kills_{f1_xuid}_{f2_xuid}",
    )
    st.plotly_chart(
        plot_trio_metric(d_self, d_f1, d_f2, metric="deaths", names=names, title="Morts", y_title="Morts"),
        width="stretch",
        key=f"trio_deaths_{f1_xuid}_{f2_xuid}",
    )
    st.plotly_chart(
        plot_trio_metric(d_self, d_f1, d_f2, metric="assists", names=names, title="Assistances", y_title="Assists"),
        width="stretch",
        key=f"trio_assists_{f1_xuid}_{f2_xuid}",
    )
    st.plotly_chart(
        plot_trio_metric(d_self, d_f1, d_f2, metric="ratio", names=names, title="FDA", y_title="FDA", y_format=".3f"),
        width="stretch",
        key=f"trio_ratio_{f1_xuid}_{f2_xuid}",
    )
    st.plotly_chart(
        plot_trio_metric(d_self, d_f1, d_f2, metric="accuracy", names=names, title="Précision", y_title="%", y_suffix="%", y_format=".2f"),
        width="stretch",
        key=f"trio_accuracy_{f1_xuid}_{f2_xuid}",
    )
    st.plotly_chart(
        plot_trio_metric(d_self, d_f1, d_f2, metric="average_life_seconds", names=names, title="Durée de vie moyenne", y_title="Secondes", y_format=".1f"),
        width="stretch",
        key=f"trio_life_{f1_xuid}_{f2_xuid}",
    )

    # Graphes de barres
    _render_metric_bar_charts(
        series=series,
        colors_by_name=colors_by_name,
        show_smooth=show_smooth,
        key_suffix=f"{len(series)}",
        plot_fn=plot_multi_metric_bars_fn,
    )

    # Médailles
    st.subheader("Médailles")
    trio_match_ids = [str(x) for x in merged["match_id"].dropna().astype(str).tolist()]
    if not trio_match_ids:
        st.info("Impossible de déterminer la liste des matchs pour l'agrégation des médailles.")
    else:
        with st.spinner("Agrégation des médailles…"):
            top_self = top_medals_fn(db_path, xuid.strip(), trio_match_ids, top_n=12, db_key=db_key)
            top_f1 = top_medals_fn(db_path, f1_xuid, trio_match_ids, top_n=12, db_key=db_key)
            top_f2 = top_medals_fn(db_path, f2_xuid, trio_match_ids, top_n=12, db_key=db_key)

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.expander(f"{me_name}", expanded=True):
                render_medals_grid(
                    [{"name_id": int(n), "count": int(c)} for n, c in (top_self or [])],
                    cols_per_row=6,
                )
        with c2:
            with st.expander(f"{f1_name}", expanded=True):
                render_medals_grid(
                    [{"name_id": int(n), "count": int(c)} for n, c in (top_f1 or [])],
                    cols_per_row=6,
                )
        with c3:
            with st.expander(f"{f2_name}", expanded=True):
                render_medals_grid(
                    [{"name_id": int(n), "count": int(c)} for n, c in (top_f2 or [])],
                    cols_per_row=6,
                )

    return True
