"""Graphiques de séries temporelles."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import HALO_COLORS, PLOT_CONFIG
from src.visualization.theme import apply_halo_plot_style, get_legend_horizontal_bottom
from src.analysis.performance_config import SCORE_THRESHOLDS


def _rolling_mean(series: pd.Series, window: int = 10) -> pd.Series:
    w = int(window) if window and window > 0 else 1
    return series.rolling(window=w, min_periods=1).mean()


def plot_timeseries(df: pd.DataFrame, title: str = "Frags / Morts / Ratio") -> go.Figure:
    """Graphique principal: Kills/Deaths/Ratio dans le temps.
    
    Args:
        df: DataFrame avec colonnes kills, deaths, assists, accuracy, ratio, start_time.
        title: Titre du graphique.
        
    Returns:
        Figure Plotly.
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    colors = HALO_COLORS.as_dict()

    d = df.sort_values("start_time").reset_index(drop=True)
    x_idx = list(range(len(d)))

    common_hover = (
        "frags=%{customdata[0]} morts=%{customdata[1]} assistances=%{customdata[2]}<br>"
        "précision=%{customdata[3]}% ratio=%{customdata[4]:.3f}<extra></extra>"
    )

    customdata = list(
        zip(
            d["kills"],
            d["deaths"],
            d["assists"],
            pd.to_numeric(d["accuracy"], errors="coerce").fillna(0).round(2),
            d["ratio"],
        )
    )

    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=d["kills"],
            name="Frags",
            marker_color=colors["cyan"],
            opacity=PLOT_CONFIG.bar_opacity,
            alignmentgroup="kda_main",
            offsetgroup="kills",
            width=0.42,
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=d["deaths"],
            name="Morts",
            marker_color=colors["red"],
            opacity=PLOT_CONFIG.bar_opacity_secondary,
            alignmentgroup="kda_main",
            offsetgroup="deaths",
            width=0.42,
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=d["ratio"],
            mode="lines",
            name="Ratio",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["green"]),
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        legend=get_legend_horizontal_bottom(),
        margin=dict(l=40, r=20, t=80, b=90),
        hovermode="x unified",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.06,
    )

    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Frags / Morts", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Ratio", secondary_y=True)

    # Labels de date/heure espacés
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if len(labels) > 1 else 1
    tickvals = x_idx[::step]
    ticktext = labels[::step]
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    
    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.tall_height)


def plot_assists_timeseries(df: pd.DataFrame, title: str = "Assistances") -> go.Figure:
    """Graphique des assistances dans le temps.
    
    Args:
        df: DataFrame avec colonnes assists, start_time, etc.
        title: Titre du graphique.
        
    Returns:
        Figure Plotly.
    """
    colors = HALO_COLORS.as_dict()
    d = df.sort_values("start_time").reset_index(drop=True)
    x_idx = list(range(len(d)))
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if len(labels) > 1 else 1

    customdata = list(
        zip(
            d["kills"],
            d["deaths"],
            d["assists"],
            pd.to_numeric(d["accuracy"], errors="coerce").fillna(0).round(2),
            d["ratio"],
            d["map_name"].fillna(""),
            d["playlist_name"].fillna(""),
            d["match_id"],
        )
    )
    hover = (
        "assistances=%{y}<br>"
        "frags=%{customdata[0]} morts=%{customdata[1]}<br>"
        "précision=%{customdata[3]}% ratio=%{customdata[4]:.3f}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=d["assists"],
            name="Assistances",
            marker_color=colors["violet"],
            opacity=PLOT_CONFIG.bar_opacity,
            customdata=customdata,
            hovertemplate=hover,
        )
    )

    smooth = _rolling_mean(pd.to_numeric(d["assists"], errors="coerce"), window=10)
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=smooth,
            mode="lines",
            name="Moyenne (lissée)",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["green"]),
            hovertemplate="moyenne=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_yaxes(title_text="Assistances", rangemode="tozero")
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)


def plot_per_minute_timeseries(df: pd.DataFrame, title: str = "Frags / Morts / Assistances par minute") -> go.Figure:
    """Graphique des stats par minute.
    
    Args:
        df: DataFrame avec colonnes kills_per_min, deaths_per_min, assists_per_min.
        title: Titre du graphique.
        
    Returns:
        Figure Plotly.
    """
    colors = HALO_COLORS.as_dict()
    d = df.sort_values("start_time").reset_index(drop=True)
    x_idx = list(range(len(d)))
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if len(labels) > 1 else 1

    customdata = list(
        zip(
            d["time_played_seconds"].fillna(float("nan")).astype(float),
            d["kills"],
            d["deaths"],
            d["assists"],
            d["match_id"],
        )
    )

    kpm = pd.to_numeric(d["kills_per_min"], errors="coerce")
    dpm = pd.to_numeric(d["deaths_per_min"], errors="coerce")
    apm = pd.to_numeric(d["assists_per_min"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=kpm,
            name="Frags/min",
            marker_color=colors["cyan"],
            opacity=PLOT_CONFIG.bar_opacity,
            customdata=customdata,
            hovertemplate=(
                "frags/min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (frags=%{customdata[1]:.0f})<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=dpm,
            name="Morts/min",
            marker_color=colors["red"],
            opacity=PLOT_CONFIG.bar_opacity_secondary,
            customdata=customdata,
            hovertemplate=(
                "morts/min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (morts=%{customdata[2]:.0f})<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=apm,
            name="Assist./min",
            marker_color=colors["violet"],
            opacity=PLOT_CONFIG.bar_opacity_secondary,
            customdata=customdata,
            hovertemplate=(
                "assist./min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (assistances=%{customdata[3]:.0f})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=_rolling_mean(kpm, window=10),
            mode="lines",
            name="Moy. frags/min",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["cyan"]),
            hovertemplate="moy=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=_rolling_mean(dpm, window=10),
            mode="lines",
            name="Moy. morts/min",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["red"], dash="dot"),
            hovertemplate="moy=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=_rolling_mean(apm, window=10),
            mode="lines",
            name="Moy. assist./min",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["violet"], dash="dot"),
            hovertemplate="moy=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.06,
    )
    fig.update_yaxes(title_text="Par minute", rangemode="tozero")
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)


def plot_accuracy_last_n(df: pd.DataFrame, n: int) -> go.Figure:
    """Graphique de précision sur les N derniers matchs.
    
    Args:
        df: DataFrame avec colonne accuracy.
        n: Nombre de matchs à afficher.
        
    Returns:
        Figure Plotly.
    """
    colors = HALO_COLORS.as_dict()
    d = df.dropna(subset=["accuracy"]).tail(n)
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=d["start_time"],
                y=d["accuracy"],
                mode="lines",
                name="Accuracy",
                line=dict(width=PLOT_CONFIG.line_width, color=colors["violet"]),
                hovertemplate="précision=%{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(height=PLOT_CONFIG.short_height, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_yaxes(title_text="%", rangemode="tozero")
    
    return apply_halo_plot_style(fig, height=PLOT_CONFIG.short_height)


def plot_average_life(df: pd.DataFrame, title: str = "Durée de vie moyenne") -> go.Figure:
    """Graphique de la durée de vie moyenne.
    
    Args:
        df: DataFrame avec colonne average_life_seconds.
        title: Titre du graphique.
        
    Returns:
        Figure Plotly.
    """
    colors = HALO_COLORS.as_dict()
    d = df.dropna(subset=["average_life_seconds"]).sort_values("start_time").reset_index(drop=True).copy()
    x_idx = list(range(len(d)))
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if len(labels) > 1 else 1

    y = pd.to_numeric(d["average_life_seconds"], errors="coerce")
    custom = list(
        zip(
            d["deaths"].fillna(0).astype(int),
            d["time_played_seconds"].fillna(float("nan")).astype(float),
            d["match_id"].astype(str),
        )
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=y,
            name="Durée de vie (s)",
            marker_color=colors["green"],
            opacity=PLOT_CONFIG.bar_opacity,
            customdata=custom,
            hovertemplate=(
                "durée de vie moy.=%{y:.1f}s<br>"
                "morts=%{customdata[0]}<br>"
                "temps joué=%{customdata[1]:.0f}s<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=_rolling_mean(y, window=10),
            mode="lines",
            name="Moyenne (lissée)",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["cyan"]),
            hovertemplate="moyenne=%{y:.2f}s<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=50, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_yaxes(title_text="Secondes", rangemode="tozero")
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, height=PLOT_CONFIG.short_height)


def plot_spree_headshots_accuracy(df: pd.DataFrame) -> go.Figure:
    """Graphique combiné: Spree, Tirs à la tête et Précision.
    
    Args:
        df: DataFrame avec colonnes max_killing_spree, headshot_kills, accuracy.
        
    Returns:
        Figure Plotly avec axe Y secondaire pour la précision.
    """
    colors = HALO_COLORS.as_dict()
    d = df.sort_values("start_time").reset_index(drop=True).copy()
    x_idx = list(range(len(d)))

    spree = (
        pd.to_numeric(d.get("max_killing_spree"), errors="coerce")
        if "max_killing_spree" in d.columns
        else pd.Series([float("nan")] * len(d))
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=spree,
            name="Folie meurtrière (max)",
            marker_color=colors["amber"],
            opacity=PLOT_CONFIG.bar_opacity,
            alignmentgroup="spree_hs",
            offsetgroup="spree",
            width=0.42,
            hovertemplate="folie meurtrière=%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=d["headshot_kills"],
            name="Tirs à la tête",
            marker_color=colors["red"],
            opacity=0.70,
            alignmentgroup="spree_hs",
            offsetgroup="headshots",
            width=0.42,
            hovertemplate="tirs à la tête=%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=d["accuracy"],
            mode="lines",
            name="Précision (%)",
            line=dict(width=PLOT_CONFIG.line_width, color=colors["violet"]),
            hovertemplate="précision=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if labels else 1
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=50, t=30, b=90),
        legend=get_legend_horizontal_bottom(),
        hovermode="x unified",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.06,
    )

    fig.update_yaxes(title_text="Spree / Tirs à la tête", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Précision (%)", ticksuffix="%", rangemode="tozero", secondary_y=True)
    
    return apply_halo_plot_style(fig, height=420)


def plot_performance_timeseries(
    df: pd.DataFrame,
    df_history: pd.DataFrame | None = None,
    title: str = "Score de performance",
    show_smooth: bool = True,
) -> go.Figure:
    """Graphique du score de performance dans le temps.
    
    Args:
        df: DataFrame avec colonnes performance ou kills/deaths/assists/accuracy/outcome.
        df_history: DataFrame complet pour le calcul du score relatif.
        title: Titre du graphique.
        show_smooth: Afficher la courbe de moyenne lissée.
        
    Returns:
        Figure Plotly.
    """
    from src.analysis.performance_score import compute_performance_series

    colors = HALO_COLORS.as_dict()
    d = df.sort_values("start_time").reset_index(drop=True)
    x_idx = list(range(len(d)))
    labels = d["start_time"].dt.strftime("%m-%d %H:%M").tolist()
    step = max(1, len(labels) // 10) if len(labels) > 1 else 1

    # Calculer le score de performance RELATIF
    history = df_history if df_history is not None else df
    if "performance" not in d.columns or d["performance"].isna().all():
        d["performance"] = compute_performance_series(d, history)

    performance = pd.to_numeric(d["performance"], errors="coerce")

    # Déterminer la couleur en fonction du score
    def _get_perf_color(val: float) -> str:
        if val >= SCORE_THRESHOLDS["excellent"]:
            return colors.get("green", "#50C878")
        elif val >= SCORE_THRESHOLDS["good"]:
            return colors.get("cyan", "#00B7EB")
        elif val >= SCORE_THRESHOLDS["average"]:
            return colors.get("amber", "#FFBF00")
        elif val >= SCORE_THRESHOLDS["below_average"]:
            return colors.get("orange", "#FF8C00")
        else:
            return colors.get("red", "#FF4444")

    bar_colors = [_get_perf_color(v) if not pd.isna(v) else colors.get("gray", "#888888") for v in performance]

    hover = (
        "performance=%{y:.1f}<br>"
        "date=%{customdata[0]}<extra></extra>"
    )
    customdata = list(zip(d["start_time"].dt.strftime("%d/%m/%Y %H:%M")))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=performance,
            name="Performance",
            marker_color=bar_colors,
            opacity=PLOT_CONFIG.bar_opacity,
            customdata=customdata,
            hovertemplate=hover,
        )
    )

    if show_smooth:
        smooth = _rolling_mean(performance, window=10)
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=smooth,
                mode="lines",
                name="Moyenne (lissée)",
                line=dict(width=PLOT_CONFIG.line_width, color=colors.get("violet", "#8B5CF6")),
                hovertemplate="moyenne=%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=90),
        hovermode="x unified",
        legend=get_legend_horizontal_bottom(),
    )
    fig.update_yaxes(title_text="Score de performance", rangemode="tozero", range=[0, 100])
    fig.update_xaxes(
        title_text="Match (chronologique)",
        tickmode="array",
        tickvals=x_idx[::step],
        ticktext=labels[::step],
        type="category",
    )

    return apply_halo_plot_style(fig, title=title, height=PLOT_CONFIG.default_height)

