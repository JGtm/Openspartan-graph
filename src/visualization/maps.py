"""Graphiques par carte (map)."""

import pandas as pd
import plotly.graph_objects as go

from src.config import HALO_COLORS, PLOT_CONFIG
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom


def plot_map_comparison(df_breakdown: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Graphique de comparaison d'une métrique par carte.
    
    Args:
        df_breakdown: DataFrame issu de compute_map_breakdown.
        metric: Nom de la colonne à afficher (ratio_global, win_rate, accuracy_avg).
        title: Titre du graphique.
        
    Returns:
        Figure Plotly (barres horizontales).
    """
    colors = HALO_COLORS.as_dict()
    d = df_breakdown.dropna(subset=[metric]).copy()
    
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=PLOT_CONFIG.default_height, margin=dict(l=40, r=20, t=30, b=40))
        return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height)

    fig = go.Figure(
        data=[
            go.Bar(
                x=d[metric],
                y=d["map_name"],
                orientation="h",
                marker_color=colors["cyan"],
                customdata=list(zip(d["matches"], d.get("accuracy_avg"))),
                hovertemplate=(
                    "%{y}<br>value=%{x}<br>matches=%{customdata[0]}"
                    "<br>accuracy=%{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        height=PLOT_CONFIG.tall_height,
        title=title,
        margin=dict(l=40, r=20, t=60, b=90),
        legend=get_legend_horizontal_bottom(),
    )
    
    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.tall_height)


def plot_map_ratio_with_winloss(df_breakdown: pd.DataFrame, title: str) -> go.Figure:
    """Graphique de ratio par carte avec taux de victoire/défaite.
    
    Args:
        df_breakdown: DataFrame issu de compute_map_breakdown.
        title: Titre du graphique.
        
    Returns:
        Figure Plotly avec barres groupées Win/Loss.
    """
    colors = HALO_COLORS.as_dict()
    d = df_breakdown.dropna(subset=["win_rate", "loss_rate"]).copy()
    
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=PLOT_CONFIG.default_height, margin=dict(l=40, r=20, t=30, b=40))
        return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=d["win_rate"],
            y=d["map_name"],
            orientation="h",
            name="Taux de victoire",
            marker_color=colors["green"],
            opacity=0.70,
            hovertemplate="win=%{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=d["loss_rate"],
            y=d["map_name"],
            orientation="h",
            name="Taux de défaite",
            marker_color=colors["red"],
            opacity=0.55,
            hovertemplate="loss=%{x:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        height=PLOT_CONFIG.tall_height,
        title=title,
        margin=dict(l=40, r=20, t=60, b=90),
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_xaxes(title_text="Win / Loss", tickformat=".0%", range=[0, 1])
    
    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.tall_height)
