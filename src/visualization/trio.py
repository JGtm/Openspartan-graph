"""Graphiques pour la comparaison trio (3 joueurs)."""

import pandas as pd
import plotly.graph_objects as go

from src.config import HALO_COLORS, PLOT_CONFIG
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom


def plot_trio_metric(
    d_self: pd.DataFrame,
    d_f1: pd.DataFrame,
    d_f2: pd.DataFrame,
    *,
    metric: str,
    names: tuple[str, str, str],
    title: str,
    y_title: str,
    y_suffix: str = "",
    y_format: str = "",
    smooth_window: int = 7,
) -> go.Figure:
    """Graphique comparant une métrique entre 3 joueurs.
    
    Les 3 DataFrames doivent être alignés sur les mêmes matchs.
    
    Args:
        d_self: DataFrame du joueur principal.
        d_f1: DataFrame du premier ami.
        d_f2: DataFrame du deuxième ami.
        metric: Nom de la colonne à comparer.
        names: Tuple des 3 noms (self, ami1, ami2).
        title: Titre du graphique.
        y_title: Titre de l'axe Y.
        y_suffix: Suffixe pour les valeurs Y (ex: "%").
        y_format: Format pour le hover (ex: ".2f").
        
    Returns:
        Figure Plotly.
    """
    colors = HALO_COLORS.as_dict()
    color_list = [colors["cyan"], colors["red"], colors["green"]]
    
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["start_time", metric])
        out = df[["start_time", metric]].copy()
        out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
        out = out.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
        return out

    a0 = _prep(d_self).rename(columns={metric: "v0"})
    a1 = _prep(d_f1).rename(columns={metric: "v1"})
    a2 = _prep(d_f2).rename(columns={metric: "v2"})

    # Aligne sur l'intersection des timestamps (les DFs sont censés être alignés, mais on reste robuste).
    aligned = a0.merge(a1, on="start_time", how="inner").merge(a2, on="start_time", how="inner")

    fig = go.Figure()
    if aligned.empty:
        fig.update_layout(title=title)
        return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)

    def _roll(s: pd.Series) -> pd.Series:
        w = int(smooth_window) if smooth_window else 0
        if w <= 1:
            return s
        return s.rolling(window=w, min_periods=1).mean()

    xs = aligned["start_time"]
    series = [aligned["v0"], aligned["v1"], aligned["v2"]]
    avg_all = pd.concat(series, axis=1).mean(axis=1)

    for idx, (s, name, color) in enumerate(zip(series, names, color_list)):
        hover_format = f"%{{y{':' + y_format if y_format else ''}}}{y_suffix}<extra></extra>"
        fig.add_trace(
            go.Bar(
                x=xs,
                y=s,
                name=f"{name} (match)",
                marker_color=color,
                opacity=0.28,
                hovertemplate=hover_format,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=_roll(s),
                mode="lines",
                name=f"{name} (moy. lissée)",
                line=dict(width=3, color=color),
                hovertemplate=hover_format,
            )
        )

    # Moyenne lissée des 3 (pointillés)
    avg_color = colors.get("amber", "rgba(255, 255, 255, 0.85)")
    hover_format_avg = f"%{{y{':' + y_format if y_format else ''}}}{y_suffix}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=_roll(avg_all),
            mode="lines",
            name="Moyenne (3) lissée",
            line=dict(width=3, color=avg_color, dash="dot"),
            hovertemplate=hover_format_avg,
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
        barmode="group",
    )
    fig.update_yaxes(title_text=y_title)
    
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    
    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)
