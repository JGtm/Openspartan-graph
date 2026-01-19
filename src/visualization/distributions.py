"""Graphiques de distributions et répartitions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import HALO_COLORS, PLOT_CONFIG, SESSION_CONFIG
from src.visualization.theme import apply_halo_plot_style


def plot_kda_distribution(df: pd.DataFrame) -> go.Figure:
    """Graphique de distribution du KDA (FDA) avec KDE.
    
    Args:
        df: DataFrame avec colonne kda.
        
    Returns:
        Figure Plotly avec densité KDE et rug plot.
    """
    colors = HALO_COLORS.as_dict()
    d = df.dropna(subset=["kda"]).copy()
    x = pd.to_numeric(d["kda"], errors="coerce").dropna().astype(float).to_numpy()
    
    if x.size == 0:
        fig = go.Figure()
        fig.update_layout(height=PLOT_CONFIG.default_height, margin=dict(l=40, r=20, t=30, b=40))
        fig.update_xaxes(title_text="FDA")
        fig.update_yaxes(title_text="Densité")
        return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height)

    # KDE gaussien (règle de Silverman)
    n = int(x.size)
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q25, q75 = (np.percentile(x, [25, 75]).tolist() if n > 1 else [0.0, 0.0])
    iqr = float(q75 - q25)
    sigma = min(std, iqr / 1.34) if (std > 0 and iqr > 0) else max(std, iqr / 1.34)
    bw = (1.06 * sigma * (n ** (-1.0 / 5.0))) if sigma and sigma > 0 else 0.3
    bw = float(max(bw, 0.05))

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = max(0.25, xmax - xmin)
    pad = 0.15 * span
    grid = np.linspace(xmin - pad, xmax + pad, 256)
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * (z ** 2)).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=dens,
            mode="lines",
            name="Densité (KDE)",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["cyan"]),
            fill="tozeroy",
            fillcolor="rgba(53,208,255,0.18)",
            hovertemplate="FDA=%{x:.2f}<br>densité=%{y:.3f}<extra></extra>",
        )
    )

    # Rug plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.zeros_like(x),
            mode="markers",
            name="Matchs",
            marker=dict(symbol="line-ns-open", size=10, color="rgba(255,255,255,0.45)"),
            hovertemplate="FDA=%{x:.2f}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)")
    fig.update_layout(height=PLOT_CONFIG.default_height, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_xaxes(title_text="FDA", zeroline=True)
    fig.update_yaxes(title_text="Densité", rangemode="tozero")
    
    return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height)


def plot_outcomes_over_time(df: pd.DataFrame, *, session_style: bool = False) -> tuple[go.Figure, str]:
    """Graphique d'évolution des victoires/défaites dans le temps.
    
    Args:
        df: DataFrame avec colonnes outcome et start_time.
        
    Args:
        session_style: Si True, force une logique de bucket orientée "session" :
            - <= 20 matchs : bucket par partie (1..n)
            - > 20 matchs : bucket par heure

    Returns:
        Tuple (figure, bucket_label) où bucket_label décrit la granularité.
    """
    colors = HALO_COLORS.as_dict()
    d = df.dropna(subset=["outcome"]).copy()
    
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=PLOT_CONFIG.default_height, margin=dict(l=40, r=20, t=30, b=40))
        fig.update_yaxes(title_text="Nombre")
        return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height), "période"

    if session_style:
        d = d.sort_values("start_time").reset_index(drop=True)
        if len(d.index) <= 20:
            bucket = (d.index + 1)
            bucket_label = "partie"
        else:
            t = pd.to_datetime(d["start_time"], errors="coerce")
            bucket = t.dt.floor("h")
            bucket_label = "heure"
    else:
        tmin = pd.to_datetime(d["start_time"], errors="coerce").min()
        tmax = pd.to_datetime(d["start_time"], errors="coerce").max()

        dt_range = (tmax - tmin) if (tmin == tmin and tmax == tmax) else pd.Timedelta(days=999)
        days = float(dt_range.total_seconds() / 86400.0) if dt_range is not None else 999.0

        cfg = SESSION_CONFIG
        
        # Détermine le bucket selon la plage de dates
        if days < cfg.bucket_threshold_hourly:
            d = d.sort_values("start_time").reset_index(drop=True)
            bucket = (d.index + 1)
            bucket_label = "partie"
        elif days <= cfg.bucket_threshold_daily:
            bucket = d["start_time"].dt.floor("h")
            bucket_label = "heure"
        elif days <= cfg.bucket_threshold_weekly:
            bucket = d["start_time"].dt.to_period("D").astype(str)
            bucket_label = "jour"
        elif days <= cfg.bucket_threshold_monthly:
            bucket = d["start_time"].dt.to_period("W-MON").astype(str)
            bucket_label = "semaine"
        else:
            bucket = d["start_time"].dt.to_period("M").astype(str)
            bucket_label = "mois"

    d["bucket"] = bucket
    pivot = (
        d.pivot_table(index="bucket", columns="outcome", values="match_id", aggfunc="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    def col(code: int) -> pd.Series:
        return pivot[code] if code in pivot.columns else pd.Series([0] * len(pivot), index=pivot.index)

    wins = col(2)
    losses = col(3)
    ties = col(1)
    nofin = col(4)

    # Objectif UI: victoires au-dessus (positif) et défaites en dessous (négatif).
    # Plotly empile séparément positifs/négatifs en mode "relative".
    losses_neg = -losses

    fig = go.Figure()
    fig.add_bar(
        x=pivot.index,
        y=wins,
        name="Victoires",
        marker_color=colors["green"],
        hovertemplate="%{x}<br>Victoires: %{y}<extra></extra>",
    )
    fig.add_bar(
        x=pivot.index,
        y=losses_neg,
        name="Défaites",
        marker_color=colors["red"],
        customdata=losses.to_numpy(),
        hovertemplate="%{x}<br>Défaites: %{customdata}<extra></extra>",
    )

    # Ces statuts ne sont pas des "défaites" : on les garde au-dessus.
    if ties.sum() > 0:
        fig.add_bar(
            x=pivot.index,
            y=ties,
            name="Égalités",
            marker_color=colors["violet"],
            hovertemplate="%{x}<br>Égalités: %{y}<extra></extra>",
        )
    if nofin.sum() > 0:
        fig.add_bar(
            x=pivot.index,
            y=nofin,
            name="Non terminés",
            marker_color=colors["slate"],
            hovertemplate="%{x}<br>Non terminés: %{y}<extra></extra>",
        )

    fig.update_layout(
        barmode="relative",
        height=PLOT_CONFIG.default_height,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig.update_yaxes(title_text="Nombre", zeroline=True)
    
    if bucket_label == "partie" and len(pivot.index) > 30:
        fig.update_xaxes(showticklabels=False, title_text="")
    
    return apply_halo_plot_style(fig, height=PLOT_CONFIG.default_height), bucket_label
