"""Thème et style des graphiques Plotly."""

import plotly.graph_objects as go

from src.config import HALO_COLORS, PLOT_CONFIG


def apply_halo_plot_style(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Applique le thème Halo aux graphiques Plotly.
    
    Style sombre avec fonds transparents pour s'intégrer au background CSS.
    
    Args:
        fig: Figure Plotly à styliser.
        title: Titre optionnel à ajouter.
        height: Hauteur optionnelle en pixels.
        
    Returns:
        La figure stylisée (modifiée in-place).
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)", size=13),
        hoverlabel=dict(
            bgcolor="rgba(10,16,32,0.92)",
            bordercolor="rgba(255,255,255,0.18)",
        ),
    )
    
    if title is not None:
        fig.update_layout(title=title)
    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.10)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.10)",
    )
    
    return fig


def get_default_layout_kwargs(height: int | None = None) -> dict:
    """Retourne les kwargs de layout par défaut.
    
    Args:
        height: Hauteur en pixels (default: PLOT_CONFIG.default_height).
        
    Returns:
        Dictionnaire de kwargs pour fig.update_layout().
    """
    cfg = PLOT_CONFIG
    return {
        "height": height or cfg.default_height,
        "margin": dict(
            l=cfg.margin_left,
            r=cfg.margin_right,
            t=cfg.margin_top,
            b=cfg.margin_bottom,
        ),
    }


def get_legend_horizontal_top() -> dict:
    """Retourne la configuration pour une légende horizontale en haut."""
    return dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    )


def get_legend_horizontal_bottom() -> dict:
    """Retourne la configuration pour une légende horizontale en bas."""
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="left",
        x=0,
    )
