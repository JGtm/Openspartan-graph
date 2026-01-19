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
    
    fig = go.Figure()
    
    for d, name, color in zip((d_self, d_f1, d_f2), names, color_list):
        hover_format = f"{metric}=%{{y{':' + y_format if y_format else ''}}}{y_suffix}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=d["start_time"],
                y=d[metric],
                mode="lines+markers",
                name=name,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                hovertemplate=hover_format,
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_yaxes(title_text=y_title)
    
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    
    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)
