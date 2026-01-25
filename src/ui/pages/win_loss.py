"""Page Victoires/Défaites.

Analyse des victoires et défaites par période et par carte.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.config import HALO_COLORS, SESSION_CONFIG
from src.analysis import compute_map_breakdown
from src.visualization import plot_outcomes_over_time, plot_map_comparison, plot_map_ratio_with_winloss
from src.ui.cache import cached_same_team_match_ids_with_friend


def _clear_min_matches_maps_auto() -> None:
    """Callback pour désactiver le mode auto du slider."""
    st.session_state["_min_matches_maps_auto"] = False


def _styler_map(styler, func, subset):
    """Applique un style en mode compatible pandas 1.x et 2.x."""
    try:
        return styler.map(func, subset=subset)
    except AttributeError:
        return styler.applymap(func, subset=subset)


def _to_float(v: object) -> Optional[float]:
    """Convertit une valeur en float, ou None si impossible."""
    try:
        if v is None:
            return None
        x = float(v)
        return x if x == x else None
    except Exception:
        return None


def _style_map_table_row(row: pd.Series) -> pd.Series:
    """Style les lignes du tableau par carte."""
    green = str(getattr(HALO_COLORS, "green", "#2ECC71"))
    red = str(getattr(HALO_COLORS, "red", "#E74C3C"))
    violet = "#8E6CFF"

    w = _to_float(row.get("Taux victoire (%)"))
    l = _to_float(row.get("Taux défaite (%)"))
    r = _to_float(row.get("Ratio global"))

    styles: dict[str, str] = {str(c): "" for c in row.index}

    if w is not None and l is not None:
        if w > l:
            styles["Taux victoire (%)"] = f"color: {green}; font-weight: 800;"
            styles["Taux défaite (%)"] = f"color: {red}; font-weight: 800;"
        elif w < l:
            styles["Taux victoire (%)"] = f"color: {red}; font-weight: 800;"
            styles["Taux défaite (%)"] = f"color: {green}; font-weight: 800;"
        else:
            styles["Taux victoire (%)"] = f"color: {violet}; font-weight: 800;"
            styles["Taux défaite (%)"] = f"color: {violet}; font-weight: 800;"

    if r is not None:
        if r > 1.0:
            styles["Ratio global"] = f"color: {green}; font-weight: 800;"
        elif r < 1.0:
            styles["Ratio global"] = f"color: {red}; font-weight: 800;"
        else:
            styles["Ratio global"] = f"color: {violet}; font-weight: 800;"

    return pd.Series(styles)


def render_win_loss_page(
    dff: pd.DataFrame,
    base: pd.DataFrame,
    picked_session_labels: list[str] | None,
    db_path: str,
    xuid: str,
    db_key: tuple[int, int] | None,
) -> None:
    """Affiche la page Victoires/Défaites.

    Args:
        dff: DataFrame filtré des matchs.
        base: DataFrame de base (toutes les parties après filtres Firefight).
        picked_session_labels: Labels des sessions sélectionnées.
        db_path: Chemin vers la base de données.
        xuid: XUID du joueur.
        db_key: Clé de cache de la DB.
    """
    with st.spinner("Calcul des victoires/défaites…"):
        current_mode = st.session_state.get("filter_mode")
        is_session_scope = bool(current_mode == "Sessions" and picked_session_labels)
        fig_out, bucket_label = plot_outcomes_over_time(dff, session_style=is_session_scope)
        st.markdown(
            f"Par **{bucket_label}** : on regroupe les parties par {bucket_label} et on compte le nombre de "
            "victoires/défaites (et autres statuts) pour suivre l'évolution."
        )
        st.plotly_chart(fig_out, width="stretch")

        st.subheader("Par période")
        d = dff.dropna(subset=["outcome"]).copy()
        if d.empty:
            st.info("Aucune donnée pour construire le tableau.")
        else:
            if is_session_scope:
                d = d.sort_values("start_time").reset_index(drop=True)
                if len(d.index) <= 20:
                    d["bucket"] = d.index + 1
                else:
                    t = pd.to_datetime(d["start_time"], errors="coerce")
                    d["bucket"] = t.dt.floor("h")
            else:
                tmin = pd.to_datetime(d["start_time"], errors="coerce").min()
                tmax = pd.to_datetime(d["start_time"], errors="coerce").max()
                dt_range = (tmax - tmin) if (tmin == tmin and tmax == tmax) else pd.Timedelta(days=999)
                days = float(dt_range.total_seconds() / 86400.0) if dt_range is not None else 999.0
                cfg = SESSION_CONFIG
                if days < cfg.bucket_threshold_hourly:
                    d = d.sort_values("start_time").reset_index(drop=True)
                    d["bucket"] = d.index + 1
                elif days <= cfg.bucket_threshold_daily:
                    d["bucket"] = d["start_time"].dt.floor("h")
                elif days <= cfg.bucket_threshold_weekly:
                    d["bucket"] = d["start_time"].dt.to_period("D").astype(str)
                elif days <= cfg.bucket_threshold_monthly:
                    d["bucket"] = d["start_time"].dt.to_period("W-MON").astype(str)
                else:
                    d["bucket"] = d["start_time"].dt.to_period("M").astype(str)

            pivot = (
                d.pivot_table(index="bucket", columns="outcome", values="match_id", aggfunc="count")
                .fillna(0)
                .astype(int)
                .sort_index()
            )
            out_tbl = pd.DataFrame(index=pivot.index)
            out_tbl["Victoires"] = pivot[2] if 2 in pivot.columns else 0
            out_tbl["Défaites"] = pivot[3] if 3 in pivot.columns else 0
            out_tbl["Égalités"] = pivot[1] if 1 in pivot.columns else 0
            out_tbl["Non terminés"] = pivot[4] if 4 in pivot.columns else 0
            out_tbl["Total"] = out_tbl[["Victoires", "Défaites", "Égalités", "Non terminés"]].sum(axis=1)
            out_tbl["Taux de victoires"] = (
                100.0 * (out_tbl["Victoires"] / out_tbl["Total"].where(out_tbl["Total"] > 0))
            ).fillna(0.0)

            out_tbl = out_tbl.reset_index().rename(columns={"bucket": bucket_label.capitalize()})

            def _style_pct(v) -> str:
                try:
                    x = float(v)  # noqa: F841
                except Exception:
                    return ""
                return "color: #E0E0E0; font-weight: 700;"

            win_rate_col = next(
                (c for c in ("Taux de victoires", "Win rate", "Taux de victoire") if c in out_tbl.columns),
                None,
            )
            if win_rate_col:
                out_styled = _styler_map(out_tbl.style, _style_pct, subset=[win_rate_col])
                col_cfg = {win_rate_col: st.column_config.NumberColumn(win_rate_col, format="%.1f%%")}
            else:
                out_styled = out_tbl.style
                col_cfg = {}

            st.dataframe(out_styled, width="stretch", hide_index=True, column_config=col_cfg)

        st.divider()
        st.subheader("Ratio par cartes")
        st.caption("Compare tes performances par map.")

        scope = st.radio(
            "Scope",
            options=["Moi (filtres actuels)", "Moi (toutes les parties)", "Avec Madina972", "Avec Chocoboflor"],
            horizontal=True,
        )
        min_matches = st.slider(
            "Minimum de matchs par carte",
            1,
            30,
            1,
            step=1,
            key="min_matches_maps",
            on_change=_clear_min_matches_maps_auto,
        )

        if scope == "Moi (toutes les parties)":
            base_scope = base
        elif scope == "Avec Madina972":
            match_ids = set(
                cached_same_team_match_ids_with_friend(
                    db_path,
                    xuid.strip(),
                    "2533274858283686",
                    db_key=db_key,
                )
            )
            base_scope = base.loc[base["match_id"].astype(str).isin(match_ids)].copy()
        elif scope == "Avec Chocoboflor":
            match_ids = set(
                cached_same_team_match_ids_with_friend(
                    db_path,
                    xuid.strip(),
                    "2535469190789936",
                    db_key=db_key,
                )
            )
            base_scope = base.loc[base["match_id"].astype(str).isin(match_ids)].copy()
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
                    ("ratio_global", "Ratio Victoire/défaite"),
                    ("win_rate", "Taux de victoires"),
                    ("accuracy_avg", "Précision moyenne"),
                ],
                format_func=lambda x: x[1],
            )
            key, label = metric

            view = breakdown.head(20).iloc[::-1]
            if key == "ratio_global":
                fig = plot_map_ratio_with_winloss(view, title=title)
            else:
                fig = plot_map_comparison(view, key, title=title)

            if key in ("win_rate",):
                fig.update_xaxes(tickformat=".0%")
            if key in ("accuracy_avg",):
                fig.update_xaxes(ticksuffix="%")

            st.plotly_chart(fig, width="stretch")

            # Tableau détaillé
            _render_map_table(breakdown, base_scope)


def _render_map_table(breakdown: pd.DataFrame, base_scope: pd.DataFrame) -> None:
    """Affiche le tableau détaillé par carte."""
    from src.ui.translations import translate_playlist_name

    tbl = breakdown.copy()
    tbl["win_rate"] = (tbl["win_rate"] * 100).round(1)
    tbl["loss_rate"] = (tbl["loss_rate"] * 100).round(1)
    tbl["accuracy_avg"] = tbl["accuracy_avg"].round(2)
    tbl["ratio_global"] = tbl["ratio_global"].round(2)

    def _single_or_multi_label(series: pd.Series) -> str:
        try:
            vals = sorted({str(x).strip() for x in series.dropna().tolist() if str(x).strip()})
        except Exception:
            return "-"
        if len(vals) == 0:
            return "-"
        if len(vals) == 1:
            return vals[0]
        return "Plusieurs"

    def _clean_asset_label(s: str | None) -> str:
        if not s:
            return ""
        return str(s).split("/")[-1].replace("-", " ").strip().title()

    def _normalize_mode_label(p: str | None) -> str | None:
        from src.ui.translations import translate_pair_name
        return translate_pair_name(p) if p else None

    if "playlist_ui" in base_scope.columns:
        playlist_ctx = _single_or_multi_label(base_scope["playlist_ui"])
    else:
        playlist_ctx = _single_or_multi_label(
            base_scope["playlist_name"].apply(_clean_asset_label).apply(translate_playlist_name)
        )

    if "mode_ui" in base_scope.columns:
        mode_ctx = _single_or_multi_label(base_scope["mode_ui"])
    else:
        mode_ctx = _single_or_multi_label(base_scope["pair_name"].apply(_normalize_mode_label))

    tbl_disp = tbl.copy()
    tbl_disp["playlist_ctx"] = playlist_ctx
    tbl_disp["mode_ctx"] = mode_ctx
    tbl_disp = tbl_disp.rename(
        columns={
            "map_name": "Carte",
            "matches": "Parties",
            "accuracy_avg": "Précision moy. (%)",
            "win_rate": "Taux victoire (%)",
            "loss_rate": "Taux défaite (%)",
            "ratio_global": "Ratio global",
            "playlist_ctx": "Playlist",
            "mode_ctx": "Mode",
        }
    )
    ordered_cols = [
        "Carte",
        "Playlist",
        "Mode",
        "Parties",
        "Précision moy. (%)",
        "Taux victoire (%)",
        "Taux défaite (%)",
        "Ratio global",
    ]
    tbl_disp = tbl_disp[[c for c in ordered_cols if c in tbl_disp.columns]]

    tbl_styled = tbl_disp.style.apply(_style_map_table_row, axis=1)
    st.dataframe(
        tbl_styled,
        width="stretch",
        hide_index=True,
        column_config={
            "Parties": st.column_config.NumberColumn("Parties", format="%d"),
            "Précision moy. (%)": st.column_config.NumberColumn("Précision moy. (%)", format="%.2f"),
            "Taux victoire (%)": st.column_config.NumberColumn("Taux victoire (%)", format="%.1f"),
            "Taux défaite (%)": st.column_config.NumberColumn("Taux défaite (%)", format="%.1f"),
            "Ratio global": st.column_config.NumberColumn("Ratio global", format="%.2f"),
        },
    )
