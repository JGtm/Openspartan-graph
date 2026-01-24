"""Thème et style des graphiques Plotly."""

from __future__ import annotations

import plotly.graph_objects as go

from src.config import HALO_COLORS, PLOT_CONFIG, THEME_COLORS


def apply_halo_plot_style(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Applique le thème Halo aux graphiques Plotly.
    
    Style sombre avec fond Waypoint (rgb 29,35,40) pour cohérence visuelle.
    Utilise THEME_COLORS de config.py pour centraliser les couleurs.
    
    Args:
        fig: Figure Plotly à styliser.
        title: Titre optionnel à ajouter.
        height: Hauteur optionnelle en pixels.
        
    Returns:
        La figure stylisée (modifiée in-place).
    """
    # Couleurs centralisées depuis config.py
    bg_color = THEME_COLORS.bg_plot_rgba(1.0)
    
    fig.update_layout(
        template="plotly_dark",
        # Fond Waypoint (section news) pour cohérence avec l'UI
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=THEME_COLORS.text_primary, size=13),
        hoverlabel=dict(
            bgcolor=THEME_COLORS.bg_plot_rgba(0.96),
            bordercolor=THEME_COLORS.border,
        ),
    )
    
    if title is not None:
        fig.update_layout(title=title)
    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        showline=True,
        linecolor=THEME_COLORS.border,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        showline=True,
        linecolor=THEME_COLORS.border,
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
