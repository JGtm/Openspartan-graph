import os
import re
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from openspartan_graph import _guess_xuid_from_db_path, load_matches, query_matches_with_friend


HALO_COLORS = {
    "cyan": "#35D0FF",
    "violet": "#8E6CFF",
    "green": "#3DFFB5",
    "red": "#FF4D6D",
    "amber": "#FFB703",
    "slate": "#A8B2D1",
}


def _apply_halo_plot_style(fig: go.Figure, *, title: str | None = None, height: int | None = None) -> go.Figure:
    # Thème sombre + fonds transparents pour s'intégrer au background CSS.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)", size=13),
        hoverlabel=dict(bgcolor="rgba(10,16,32,0.92)", bordercolor="rgba(255,255,255,0.18)"),
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


def _default_db_path() -> str:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return ""
    base = os.path.join(local, "OpenSpartan.Workshop", "data")
    if not os.path.isdir(base):
        return ""
    try:
        dbs = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(".db")]
    except Exception:
        return ""
    if not dbs:
        return ""
    # Si plusieurs DB, prend la plus récente (modification).
    dbs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dbs[0]


DEFAULT_DB = _default_db_path()
DEFAULT_WAYPOINT_PLAYER = "JGtm"


def _default_workshop_exe_path() -> str:
    # Chemin par défaut fourni par l'utilisateur, avec adaptation via variables d'env.
    pf86 = os.environ.get("ProgramFiles(x86)")
    if not pf86:
        pf86 = r"C:\Program Files (x86)"
    return os.path.join(pf86, "Den.Dev", "OpenSpartan Workshop", "OpenSpartan.Workshop.exe")

# Alias (local) pour un affichage plus lisible dans la GUI.
XUID_ALIASES: dict[str, str] = {
    "2533274823110022": "JGtm",
    "2533274858283686": "Madina972",
    "2535469190789936": "Chocoboflor",
}


def display_name_from_xuid(xuid: str) -> str:
    xuid = (xuid or "").strip()
    return XUID_ALIASES.get(xuid, xuid)


@st.cache_data(show_spinner=False)
def load_df(db_path: str, xuid: str) -> pd.DataFrame:
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
    # Facilite les filtres date (Streamlit n'aime pas toujours les tz-aware)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(None)
    df["date"] = df["start_time"].dt.date

    # Stats "par minute" basées sur le temps joué de chaque match.
    # Note: certaines entrées peuvent ne pas avoir de TimePlayed (NaN/0).
    minutes = (pd.to_numeric(df["time_played_seconds"], errors="coerce") / 60.0).astype(float)
    minutes = minutes.where(minutes > 0)
    df["kills_per_min"] = pd.to_numeric(df["kills"], errors="coerce") / minutes
    df["deaths_per_min"] = pd.to_numeric(df["deaths"], errors="coerce") / minutes
    df["assists_per_min"] = pd.to_numeric(df["assists"], errors="coerce") / minutes
    return df


def compute_global_ratio(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    deaths = float(df["deaths"].sum())
    if deaths <= 0:
        return None
    return (float(df["kills"].sum()) + (float(df["assists"].sum()) / 2.0)) / deaths


def compute_outcome_rates(df: pd.DataFrame) -> dict:
    # 2=Wins, 3=Losses, 1=Ties, 4=NoFinishes
    d = df.dropna(subset=["outcome"]).copy()
    total = int(len(d))
    counts = d["outcome"].value_counts().to_dict() if total else {}
    return {
        "wins": int(counts.get(2, 0)),
        "losses": int(counts.get(3, 0)),
        "ties": int(counts.get(1, 0)),
        "nofinish": int(counts.get(4, 0)),
        "total": total,
    }

def format_selected_matches_summary(n: int, rates: dict) -> str:
    # Mise en forme plus lisible pour l'UI.
    if n <= 0:
        return "Aucun match sélectionné"

    def plural(n_: int, one: str, many: str) -> str:
        return one if int(n_) == 1 else many

    wins = int(rates.get("wins", 0))
    losses = int(rates.get("losses", 0))
    ties = int(rates.get("ties", 0))
    nofinish = int(rates.get("nofinish", 0))
    return (
        f"{plural(n, 'Partie', 'Parties')} sélectionnée{'' if n == 1 else 's'}: {n} | "
        f"{plural(wins, 'Victoire', 'Victoires')}: {wins} | "
        f"{plural(losses, 'Défaite', 'Défaites')}: {losses} | "
        f"{plural(ties, 'Égalité', 'Égalités')}: {ties} | "
        f"{plural(nofinish, 'Non terminé', 'Non terminés')}: {nofinish}"
    )


def compute_sessions(df: pd.DataFrame, gap_minutes: int = 35) -> pd.DataFrame:
    # Regroupe les parties consécutives en sessions (gap > N minutes => nouvelle session).
    if df.empty:
        d = df.copy()
        d["session_id"] = pd.Series(dtype=int)
        d["session_label"] = pd.Series(dtype=str)
        return d

    d = df.sort_values("start_time").copy()
    gaps = d["start_time"].diff().dt.total_seconds().fillna(0)
    new_session = (gaps > (gap_minutes * 60)).astype(int)
    d["session_id"] = new_session.cumsum().astype(int)

    g = d.groupby("session_id")["start_time"].agg(["min", "max", "count"])
    labels = {}
    for sid, row in g.iterrows():
        start = row["min"]
        end = row["max"]
        cnt = int(row["count"])
        labels[sid] = f"{start:%Y-%m-%d} {start:%H:%M}–{end:%H:%M} ({cnt})"
    d["session_label"] = d["session_id"].map(labels)
    return d


def compute_map_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "map_name",
                "matches",
                "accuracy_avg",
                "win_rate",
                "loss_rate",
                "ratio_global",
            ]
        )

    d = df.copy()
    d["map_name"] = d["map_name"].fillna("")
    d = d.loc[d["map_name"].astype(str).str.strip() != ""]
    if d.empty:
        return pd.DataFrame(
            columns=[
                "map_name",
                "matches",
                "accuracy_avg",
                "win_rate",
                "loss_rate",
                "ratio_global",
            ]
        )

    rows = []
    for map_name, g in d.groupby("map_name", dropna=True):
        rates = compute_outcome_rates(g)
        total_out = max(1, rates["total"])
        acc = g["accuracy"].dropna().mean()
        rows.append(
            {
                "map_name": map_name,
                "matches": int(len(g)),
                "accuracy_avg": float(acc) if acc == acc else None,
                "win_rate": rates["wins"] / total_out if rates["total"] else None,
                "loss_rate": rates["losses"] / total_out if rates["total"] else None,
                "ratio_global": compute_global_ratio(g),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["matches", "ratio_global"], ascending=[False, False])
    return out


def plot_map_comparison(df_breakdown: pd.DataFrame, metric: str, title: str) -> go.Figure:
    d = df_breakdown.dropna(subset=[metric]).copy()
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=360, margin=dict(l=40, r=20, t=30, b=40))
        return _apply_halo_plot_style(fig, height=360)

    fig = go.Figure(
        data=[
            go.Bar(
                x=d[metric],
                y=d["map_name"],
                orientation="h",
                marker_color=HALO_COLORS["cyan"],
                customdata=list(zip(d["matches"], d.get("accuracy_avg"))),
                hovertemplate=(
                    "%{y}<br>value=%{x}<br>matches=%{customdata[0]}"
                    "<br>accuracy=%{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        height=520,
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return _apply_halo_plot_style(fig, title=title, height=520)


def plot_spree_headshots_accuracy(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["max_killing_spree"],
            mode="lines+markers",
            name="Max killing spree",
            line=dict(width=2.4, color=HALO_COLORS["amber"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["amber"]),
            hovertemplate="spree=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["headshot_kills"],
            mode="lines+markers",
            name="Headshot kills",
            line=dict(width=2.4, color=HALO_COLORS["red"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["red"]),
            hovertemplate="headshots=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
            go.Scatter(
            x=d["start_time"],
            y=d["accuracy"],
            mode="lines+markers",
            name="Précision (%)",
            yaxis="y2",
            line=dict(width=2.4, color=HALO_COLORS["violet"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["violet"]),
            hovertemplate="précision=%{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=50, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        yaxis=dict(title="Counts (spree / headshots)", rangemode="tozero"),
        yaxis2=dict(
            title="Précision (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            ticksuffix="%",
        ),
    )
    return _apply_halo_plot_style(fig, height=420)


def _date_range(df: pd.DataFrame) -> Tuple[date, date]:
    dmin = df["date"].min()
    dmax = df["date"].max()
    return dmin, dmax


def _build_option_map(series_name: pd.Series, series_id: pd.Series) -> dict:
    out = {}
    for name, _id in zip(series_name.fillna(""), series_id.fillna("")):
        if not _id:
            continue
        # N'affiche pas les entrées sans libellé.
        if not (isinstance(name, str) and name.strip()):
            continue
        label = name.strip()
        out[f"{label} — {_id}"] = _id
    # stable order
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _build_xuid_option_map(xuids: list[str]) -> dict[str, str]:
    # label -> xuid
    out: dict[str, str] = {}
    for x in xuids:
        label = display_name_from_xuid(x)
        out[f"{label} — {x}"] = x
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _mark_firefight(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristique: on se base sur les libellés Playlist / Pair.
    # Objectif: exclure Firefight par défaut (PvE) des stats "compétitives".
    d = df.copy()
    pl = d.get("playlist_name")
    pair = d.get("pair_name")
    pl_s = pl.fillna("").astype(str) if pl is not None else pd.Series([""] * len(d))
    pair_s = pair.fillna("").astype(str) if pair is not None else pd.Series([""] * len(d))

    pat = r"\bfirefight\b"
    d["is_firefight"] = pl_s.str.contains(pat, case=False, regex=True) | pair_s.str.contains(
        pat, case=False, regex=True
    )
    return d


def _is_allowed_playlist_name(name: str) -> bool:
    s = (name or "").strip().lower()
    if not s:
        return False
    if re.search(r"\bquick\s*play\b", s):
        return True
    # Variantes possibles: "Ranked Slayer", "Ranked: Slayer", etc.
    if re.search(r"\branked\b.*\bslayer\b", s):
        return True
    if re.search(r"\branked\b.*\barena\b", s):
        return True
    return False


def plot_timeseries(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    common_hover = (
        "frags=%{customdata[0]} morts=%{customdata[1]} assistances=%{customdata[2]}<br>"
        "précision=%{customdata[3]}% ratio=%{customdata[4]:.3f}<extra></extra>"
    )

    customdata = list(
        zip(
            df["kills"],
            df["deaths"],
            df["assists"],
            df["accuracy"].round(2).astype(object),
            df["ratio"],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["kills"],
            mode="lines+markers",
            name="Frags",
            line=dict(width=2.6, color=HALO_COLORS["cyan"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["cyan"]),
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["deaths"],
            mode="lines+markers",
            name="Morts",
            line=dict(width=2.6, color=HALO_COLORS["red"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["red"]),
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["ratio"],
            mode="lines+markers",
            name="Ratio",
            line=dict(width=2.6, color=HALO_COLORS["green"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["green"]),
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=40),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Frags / Morts", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Ratio", secondary_y=True)
    return _apply_halo_plot_style(fig, title=title, height=520)


def plot_assists_timeseries(df: pd.DataFrame, title: str) -> go.Figure:
    customdata = list(
        zip(
            df["kills"],
            df["deaths"],
            df["assists"],
            df["accuracy"].round(2).astype(object),
            df["ratio"],
            df["map_name"].fillna(""),
            df["playlist_name"].fillna(""),
            df["match_id"],
        )
    )
    hover = (
        "assistances=%{y}<br>"
        "frags=%{customdata[0]} morts=%{customdata[1]}<br>"
        "précision=%{customdata[3]}% ratio=%{customdata[4]:.3f}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["assists"],
            mode="lines+markers",
            name="Assistances",
            line=dict(width=2.6, color=HALO_COLORS["violet"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["violet"]),
            customdata=customdata,
            hovertemplate=hover,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Assistances", rangemode="tozero")
    return _apply_halo_plot_style(fig, title=title, height=360)


def plot_per_minute_timeseries(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()

    customdata = list(
        zip(
            df["time_played_seconds"].fillna(float("nan")).astype(float),
            df["kills"],
            df["deaths"],
            df["assists"],
            df["match_id"],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["kills_per_min"],
            mode="lines+markers",
            name="Frags/min",
            line=dict(width=2.4, color=HALO_COLORS["cyan"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["cyan"]),
            customdata=customdata,
            hovertemplate=(
                "frags/min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (frags=%{customdata[1]:.0f})<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["deaths_per_min"],
            mode="lines+markers",
            name="Morts/min",
            line=dict(width=2.4, color=HALO_COLORS["red"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["red"]),
            customdata=customdata,
            hovertemplate=(
                "morts/min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (morts=%{customdata[2]:.0f})<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["assists_per_min"],
            mode="lines+markers",
            name="Assist./min",
            line=dict(width=2.4, color=HALO_COLORS["violet"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["violet"]),
            customdata=customdata,
            hovertemplate=(
                "assist./min=%{y:.2f}<br>"
                "temps joué=%{customdata[0]:.0f}s (assistances=%{customdata[3]:.0f})<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Par minute", rangemode="tozero")
    return _apply_halo_plot_style(fig, title=title, height=360)


def plot_accuracy_last_n(df: pd.DataFrame, n: int) -> go.Figure:
    d = df.dropna(subset=["accuracy"]).tail(n)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=d["start_time"],
                y=d["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(width=2.6, color=HALO_COLORS["violet"], shape="spline", smoothing=0.85),
                marker=dict(size=6, color=HALO_COLORS["violet"]),
                hovertemplate="précision=%{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_yaxes(title_text="%", rangemode="tozero")
    return _apply_halo_plot_style(fig, height=320)


def plot_average_life(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["average_life_seconds"]).copy()
    fig = go.Figure()

    def fmt_mmss(s: float) -> str:
        if s != s:
            return "-"
        s_i = int(round(float(s)))
        m, sec = divmod(max(0, s_i), 60)
        return f"{m:d}:{sec:02d}"

    custom = list(
        zip(
            d["deaths"].astype(int),
            d["time_played_seconds"].fillna(float("nan")).astype(float),
            d["match_id"].astype(str),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["average_life_seconds"],
            mode="lines+markers",
            name="Average life (s)",
            line=dict(width=2.6, color=HALO_COLORS["green"], shape="spline", smoothing=0.85),
            marker=dict(size=6, color=HALO_COLORS["green"]),
            customdata=custom,
            hovertemplate=(
                "durée de vie moy.=%{y:.1f}s<br>"
                "morts=%{customdata[0]}<br>"
                "temps joué=%{customdata[1]:.0f}s<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Secondes (voir hover pour mm:ss)", rangemode="tozero")
    return _apply_halo_plot_style(fig, height=320)


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
    # Suppose que les 3 DF sont déjà alignées sur les mêmes match_id (inner join côté appelant).
    fig = go.Figure()
    colors = [HALO_COLORS["cyan"], HALO_COLORS["red"], HALO_COLORS["green"]]
    for d, name, color in zip((d_self, d_f1, d_f2), names, colors):
        fig.add_trace(
            go.Scatter(
                x=d["start_time"],
                y=d[metric],
                mode="lines+markers",
                name=name,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                hovertemplate=(
                    f"{metric}=%{{y{':' + y_format if y_format else ''}}}{y_suffix}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text=y_title)
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    return _apply_halo_plot_style(fig, title=title, height=360)


def plot_outcomes_mom(df: pd.DataFrame) -> go.Figure:
    # Outcome codes per OpenSpartan docs:
    # 2=Wins, 3=Losses, 1=Ties, 4=NoFinishes
    d = df.dropna(subset=["outcome"]).copy()
    d["month"] = d["start_time"].dt.to_period("M").astype(str)
    pivot = (
        d.pivot_table(index="month", columns="outcome", values="match_id", aggfunc="count")
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

    fig = go.Figure()
    fig.add_bar(x=pivot.index, y=wins, name="Victoires", marker_color=HALO_COLORS["green"])
    fig.add_bar(x=pivot.index, y=losses, name="Défaites", marker_color=HALO_COLORS["red"])
    fig.add_bar(x=pivot.index, y=ties, name="Égalités", marker_color=HALO_COLORS["violet"])
    fig.add_bar(x=pivot.index, y=nofin, name="Non terminés", marker_color=HALO_COLORS["violet"])
    fig.update_layout(barmode="stack", height=360, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_yaxes(title_text="Nombre")
    return _apply_halo_plot_style(fig, height=360)


def plot_kda_distribution(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["kda"]).copy()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=d["kda"],
                nbinsx=40,
                marker_color=HALO_COLORS["cyan"],
                hovertemplate="FDA=%{x:.2f}<br>nombre=%{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_xaxes(title_text="FDA")
    fig.update_yaxes(title_text="Nombre")
    return _apply_halo_plot_style(fig, height=360)


@st.cache_data(show_spinner=False)
def list_other_player_xuids(db_path: str, self_xuid: str, limit: int = 500) -> list[str]:
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        sql = """
        WITH base AS (
          SELECT json_extract(ResponseBody, '$.Players') AS Players
          FROM MatchStats
        )
        SELECT DISTINCT json_extract(j.value, '$.PlayerId') AS PlayerId
        FROM base
        JOIN json_each(base.Players) AS j
        WHERE PlayerId IS NOT NULL
        LIMIT ?;
        """
        cur.execute(sql, (limit,))
        xuids: set[str] = set()
        for (pid,) in cur.fetchall():
            if not isinstance(pid, str):
                continue
            if pid == f"xuid({self_xuid})":
                continue
            m = re.fullmatch(r"xuid\((\d+)\)", pid)
            if m:
                xuids.add(m.group(1))
        return sorted(xuids)
    finally:
        con.close()


def _parse_xuid_input(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return s
    m = re.fullmatch(r"xuid\((\d+)\)", s)
    if m:
        return m.group(1)
    return None

@st.cache_data(show_spinner=False)
def list_top_teammates(db_path: str, self_xuid: str, limit: int = 20) -> list[tuple[str, int]]:
    # Retourne les XUID les plus fréquents dans la même équipe que moi.
    import sqlite3

    me_id = f"xuid({self_xuid})"
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        sql = """
        WITH p AS (
            SELECT
                json_extract(ResponseBody, '$.MatchId') AS MatchId,
                json_extract(j.value, '$.PlayerId') AS PlayerId,
                CAST(json_extract(j.value, '$.LastTeamId') AS INTEGER) AS TeamId
            FROM MatchStats
            JOIN json_each(json_extract(ResponseBody, '$.Players')) AS j
        ),
        me AS (
            SELECT MatchId, TeamId
            FROM p
            WHERE PlayerId = ? AND TeamId IS NOT NULL
        )
        SELECT
            p.PlayerId,
            COUNT(DISTINCT p.MatchId) AS Matches
        FROM p
        JOIN me ON me.MatchId = p.MatchId AND me.TeamId = p.TeamId
        WHERE p.PlayerId IS NOT NULL AND p.PlayerId <> ?
        GROUP BY p.PlayerId
        ORDER BY Matches DESC
        LIMIT ?;
        """
        cur.execute(sql, (me_id, me_id, int(limit)))
        out: list[tuple[str, int]] = []
        for pid, matches in cur.fetchall():
            if not isinstance(pid, str):
                continue
            m = re.fullmatch(r"xuid\((\d+)\)", pid)
            if not m:
                continue
            out.append((m.group(1), int(matches)))
        return out
    finally:
        con.close()


def main() -> None:
    st.set_page_config(page_title="OpenSpartan Graphs", layout="wide")

    st.markdown(
        """
        <style>
                    :root {
                        --bg: #0b1220;
                        --border: rgba(255,255,255,0.10);
                        --text: rgba(255,255,255,0.92);
                        --muted: rgba(255,255,255,0.70);
                    }

                    .block-container {padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1400px;}
                    section.main {
                        background:
                            radial-gradient(1200px 600px at 20% 0%, rgba(53,208,255,0.26), transparent 60%),
                            radial-gradient(900px 500px at 80% 15%, rgba(142,108,255,0.22), transparent 60%),
                            linear-gradient(180deg, #0b1220 0%, #070b16 100%);
                        color: var(--text);
                    }

                    /* Overlay style "Halo" : grille + scanlines légères */
                    section.main:before {
                        content: "";
                        position: fixed;
                        inset: 0;
                        pointer-events: none;
                        background:
                            repeating-linear-gradient(
                                0deg,
                                rgba(255,255,255,0.035) 0px,
                                rgba(255,255,255,0.035) 1px,
                                transparent 1px,
                                transparent 28px
                            ),
                            repeating-linear-gradient(
                                90deg,
                                rgba(255,255,255,0.020) 0px,
                                rgba(255,255,255,0.020) 1px,
                                transparent 1px,
                                transparent 28px
                            );
                        opacity: 0.35;
                        mix-blend-mode: overlay;
                    }

                    [data-testid="stSidebar"] {
                        border-right: 1px solid rgba(255,255,255,0.08);
                        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                    }

                    .hero {
                        border: 1px solid var(--border);
                        background: linear-gradient(135deg, rgba(53,208,255,0.20), rgba(142,108,255,0.14));
                        box-shadow: 0 12px 30px rgba(0,0,0,0.28);
                        border-radius: 18px;
                        padding: 18px 18px;
                        margin-bottom: 14px;
                    }
                    .hero .title {font-size: 28px; font-weight: 750; letter-spacing: -0.02em; margin: 0;}
                    .hero .subtitle {margin-top: 4px; color: var(--muted); font-size: 14px;}
                    .chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;}
                    .chip {
                        border: 1px solid rgba(255,255,255,0.14);
                        background: rgba(0,0,0,0.14);
                        color: rgba(255,255,255,0.86);
                        padding: 6px 10px;
                        border-radius: 999px;
                        font-size: 12px;
                    }

                    .stButton>button {border-radius: 12px; border: 1px solid rgba(255,255,255,0.14);}
                    .stButton>button[kind="primary"] {
                        background: linear-gradient(90deg, rgba(53,208,255,0.95), rgba(142,108,255,0.95));
                        border: 0;
                    }

                    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
                    .stTabs [data-baseweb="tab"] {
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.10);
                        border-radius: 12px;
                        padding: 10px 12px;
                        color: rgba(255,255,255,0.86);
                    }
                    .stTabs [aria-selected="true"] {background: rgba(53,208,255,0.14); border-color: rgba(53,208,255,0.32);}

                    div[data-testid="stMetric"] {
                        background: rgba(255,255,255,0.05);
                        border: 1px solid rgba(255,255,255,0.10);
                        border-radius: 14px;
                        padding: 14px 14px;
                        box-shadow: 0 10px 22px rgba(0,0,0,0.20);
                    }
                    div[data-testid="stMetric"] label {color: rgba(255,255,255,0.74) !important;}
                    div[data-testid="stMetric"] div {color: rgba(255,255,255,0.92) !important;}

                    div[data-testid="stAlert"] {border-radius: 14px; border: 1px solid rgba(255,255,255,0.12);}
                    .stDataFrame {border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="title">OpenSpartan Graphs (local)</div>
            <div class="subtitle">Analyse tes parties Halo Infinite depuis la DB OpenSpartan Workshop — filtres, séries temporelles, amis, maps.</div>
            <div class="chips">
                <span class="chip">DB locale SQLite</span>
                <span class="chip">Streamlit + Plotly</span>
                <span class="chip">Survol (hover) riche</span>
                <span class="chip">Filtres playlists / période</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Source")
        db_path = st.text_input("Chemin du .db", value=DEFAULT_DB)
        xuid_default = _guess_xuid_from_db_path(db_path) or ""
        xuid = st.text_input("XUID", value=xuid_default)
        me_name = display_name_from_xuid(xuid.strip())
        waypoint_player = st.text_input(
            "HaloWaypoint player (slug)",
            value=DEFAULT_WAYPOINT_PLAYER,
            help="Ex: JGtm (sert à construire l'URL de match).",
        )

        st.divider()
        st.header("OpenSpartan")
        workshop_exe = st.text_input(
            "Chemin de OpenSpartan.Workshop.exe",
            value=_default_workshop_exe_path(),
            help="Bouton pratique pour lancer l'app OpenSpartan Workshop.",
        )
        if st.button("Lancer OpenSpartan Workshop", use_container_width=True):
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

        if not db_path.strip():
            st.error(
                "Aucun .db détecté automatiquement. "
                "Vérifie que OpenSpartan Workshop est installé et que la DB existe dans LOCALAPPDATA."
            )
            st.stop()
        if not os.path.exists(db_path):
            st.error("Le fichier .db n'existe pas à ce chemin.")
            st.stop()
        if not xuid.strip().isdigit():
            st.error("XUID invalide (doit être numérique).")
            st.stop()

    df = load_df(db_path, xuid.strip())
    if df.empty:
        st.warning("Aucun match trouvé.")
        st.stop()

    df = _mark_firefight(df)

    with st.sidebar:
        st.header("Filtres")
        dmin, dmax = _date_range(df)
        filter_mode = st.radio("Sélection", options=["Période", "Sessions"], horizontal=True)

        # Période en haut, sous forme de deux calendriers.
        start_d, end_d = dmin, dmax
        gap_minutes = 35
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
                max_value=90,
                value=35,
                step=5,
                help="Au-delà de cet écart, on considère que c'est une nouvelle session.",
            )

        playlist_opts = _build_option_map(df["playlist_name"], df["playlist_id"])
        playlist_label = st.selectbox(
            "Playlist",
            options=["(toutes)"] + list(playlist_opts.keys()),
            index=0,
        )
        playlist_id: Optional[str] = None
        if playlist_label != "(toutes)":
            playlist_id = playlist_opts[playlist_label]

        map_opts = _build_option_map(df["map_name"], df["map_id"])
        map_label = st.selectbox(
            "Carte",
            options=["(toutes)"] + list(map_opts.keys()),
            index=0,
        )
        map_id: Optional[str] = None
        if map_label != "(toutes)":
            map_id = map_opts[map_label]

        last_n_acc = st.slider("Précision: derniers matchs", 5, 50, 20, step=1)

        restrict_playlists = st.toggle(
            "Limiter aux playlists (Quick Play / Ranked Slayer / Ranked Arena)",
            value=True,
            help="Exclut par défaut tous les autres modes.",
        )

        include_firefight = st.toggle(
            "Inclure Firefight (PvE)",
            value=False,
            help="Par défaut, les parties Firefight sont exclues des stats.",
        )

        # (la sélection période/sessions est gérée plus haut)

    # Apply filters (map/playlist puis période ou sessions)
    base = df.copy()
    if not include_firefight and "is_firefight" in base.columns:
        base = base.loc[~base["is_firefight"]].copy()

    if restrict_playlists:
        pl = base["playlist_name"].fillna("").astype(str)
        allowed_mask = pl.apply(_is_allowed_playlist_name)
        if allowed_mask.any():
            base = base.loc[allowed_mask].copy()
        else:
            st.sidebar.warning(
                "Aucune playlist n'a matché Quick Play / Ranked Slayer / Ranked Arena. "
                "Désactive ce filtre si tes libellés sont différents."
            )
    if playlist_id is not None:
        base = base.loc[base["playlist_id"].fillna("") == playlist_id]
    if map_id is not None:
        base = base.loc[base["map_id"].fillna("") == map_id]

    if filter_mode == "Sessions":
        base_s = compute_sessions(base, gap_minutes=gap_minutes)
        session_labels = (
            base_s[["session_id", "session_label"]]
            .drop_duplicates()
            .sort_values("session_id", ascending=False)
        )
        options = session_labels["session_label"].tolist()

        # Raccourcis UX: dernière session / session précédente.
        if "picked_sessions" not in st.session_state:
            st.session_state.picked_sessions = options[:1] if options else []

        cols = st.sidebar.columns(2)
        if cols[0].button("Dernière session", use_container_width=True):
            st.session_state.picked_sessions = options[:1] if options else []
        if cols[1].button("Session précédente", use_container_width=True):
            st.session_state.picked_sessions = options[1:2] if len(options) >= 2 else options[:1]

        picked = st.sidebar.multiselect(
            "Sessions",
            options=options,
            key="picked_sessions",
            help="Tu peux en sélectionner plusieurs pour comparer.",
        )
        dff = base_s.loc[base_s["session_label"].isin(picked)].copy() if picked else base_s.copy()
    else:
        mask = (base["date"] >= start_d) & (base["date"] <= end_d)
        dff = base.loc[mask].copy()

    # KPIs demandés: accuracy moyenne, ratio wins/losses, ratio global
    rates = compute_outcome_rates(dff)
    total_outcomes = max(1, rates["total"])
    win_rate = rates["wins"] / total_outcomes
    loss_rate = rates["losses"] / total_outcomes

    avg_acc = dff["accuracy"].dropna().mean() if not dff.empty else None
    global_ratio = compute_global_ratio(dff)
    avg_life = dff["average_life_seconds"].dropna().mean() if not dff.empty else None

    def _mmss(seconds: Optional[float]) -> str:
        if seconds is None or seconds != seconds:
            return "-"
        s_i = int(round(float(seconds)))
        m, sec = divmod(max(0, s_i), 60)
        return f"{m:d}:{sec:02d}"

    # Moyennes par partie (en haut)
    kpg = dff["kills"].mean() if not dff.empty else None
    dpg = dff["deaths"].mean() if not dff.empty else None
    apg = dff["assists"].mean() if not dff.empty else None

    avg_row = st.columns(3)
    avg_row[0].metric("Frags par partie", f"{kpg:.2f}" if kpg == kpg else "-")
    avg_row[1].metric("Morts par partie", f"{dpg:.2f}" if dpg == dpg else "-")
    avg_row[2].metric("Assistances par partie", f"{apg:.2f}" if apg == apg else "-")

    # Stats par minute (basées sur le temps joué total)
    total_minutes = (
        pd.to_numeric(dff["time_played_seconds"], errors="coerce").dropna().sum() / 60.0
        if not dff.empty and "time_played_seconds" in dff.columns
        else 0.0
    )
    kpm = (float(dff["kills"].sum()) / total_minutes) if total_minutes > 0 else None
    dpm = (float(dff["deaths"].sum()) / total_minutes) if total_minutes > 0 else None
    apm = (float(dff["assists"].sum()) / total_minutes) if total_minutes > 0 else None
    per_min_row = st.columns(3)
    per_min_row[0].metric("Frags / min", f"{kpm:.2f}" if kpm is not None else "-")
    per_min_row[1].metric("Morts / min", f"{dpm:.2f}" if dpm is not None else "-")
    per_min_row[2].metric("Assistances / min", f"{apm:.2f}" if apm is not None else "-")

    kpi = st.columns(5)
    kpi[0].metric("Précision moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-")
    kpi[1].metric("Taux de victoire", f"{win_rate*100:.1f}%" if rates["total"] else "-")
    kpi[2].metric("Taux de défaite", f"{loss_rate*100:.1f}%" if rates["total"] else "-")
    kpi[3].metric("Ratio global", f"{global_ratio:.2f}" if global_ratio is not None else "-")
    kpi[4].metric("Durée de vie moyenne", _mmss(avg_life))

    st.info(format_selected_matches_summary(len(dff), rates))

    tab_series, tab_mom, tab_kda, tab_friend, tab_friends, tab_maps, tab_table = st.tabs(
        [
            "Séries temporelles",
            "Victoires/Défaites (par mois)",
            "FDA (distribution)",
            "Avec un joueur",
            "Avec mes amis",
            "Comparaison maps",
            "Historique des parties",
        ]
    )

    with tab_series:
        fig = plot_timeseries(dff, title=f"{me_name}")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Assistances")
        st.plotly_chart(plot_assists_timeseries(dff, title=f"{me_name} — Assistances"), width="stretch")

        st.subheader("Stats par minute")
        st.plotly_chart(
            plot_per_minute_timeseries(dff, title=f"{me_name} — Frags/Morts/Assistances par minute"),
            width="stretch",
        )

        st.subheader("Durée de vie moyenne (Average Life)")
        if dff.dropna(subset=["average_life_seconds"]).empty:
            st.info("Average Life indisponible sur ce filtre.")
        else:
            st.plotly_chart(plot_average_life(dff), width="stretch")

        st.subheader("Spree / Headshots / Précision")
        st.plotly_chart(plot_spree_headshots_accuracy(dff), width="stretch")

        st.subheader("Précision — derniers matchs")
        st.plotly_chart(plot_accuracy_last_n(dff, last_n_acc), width="stretch")

    with tab_mom:
        st.markdown(
            "Par mois : on regroupe les parties **par mois** et on compte le nombre de "
            "victoires/défaites (et autres statuts) pour suivre l'évolution."
        )
        st.caption("Basé sur Players[].Outcome (2=win, 3=loss, 1=tie, 4=no finish).")
        st.plotly_chart(plot_outcomes_mom(dff), width="stretch")

    with tab_kda:
        valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
        if valid.empty:
            st.warning("FDA indisponible sur ce filtre.")
        else:
            st.metric("FDA médiane", f"{valid['kda'].median():.2f}")
            st.metric("FDA moyenne", f"{valid['kda'].mean():.2f}")
            st.plotly_chart(plot_kda_distribution(dff), width="stretch")

    with tab_friend:
        st.caption(
            "La DB locale ne contient pas les gamertags, uniquement des PlayerId de type xuid(...). "
            "Tu peux soit coller un XUID, soit sélectionner un XUID rencontré dans tes matchs."
        )
        cols = st.columns([2, 2, 1])
        with cols[0]:
            friend_raw = st.text_input("Ami: XUID ou xuid(123)", value="")
        with cols[1]:
            opts_map = _build_xuid_option_map(list_other_player_xuids(db_path, xuid.strip(), limit=500))
            friend_pick_label = st.selectbox(
                "Ou choisir un XUID vu",
                options=["(aucun)"] + list(opts_map.keys()),
                index=0,
            )
        with cols[2]:
            same_team_only = st.checkbox("Même équipe", value=True)

        friend_xuid = _parse_xuid_input(friend_raw) or (
            opts_map.get(friend_pick_label) if friend_pick_label != "(aucun)" else None
        )

        if friend_xuid is None:
            st.info("Renseigne un XUID (numérique) ou choisis-en un.")
        else:
            rows = query_matches_with_friend(db_path, xuid.strip(), friend_xuid)
            if same_team_only:
                rows = [r for r in rows if r.get("same_team")]

            if not rows:
                st.warning("Aucun match trouvé avec ce joueur (selon le filtre).")
            else:
                dfr = pd.DataFrame(rows)
                dfr["start_time"] = pd.to_datetime(dfr["start_time"], utc=True).dt.tz_convert(None)
                dfr = dfr.sort_values("start_time", ascending=False)

                # outcome counts
                outcome_map = {2: "Victoire", 3: "Défaite", 1: "Égalité", 4: "Non terminé"}
                dfr["my_outcome_label"] = dfr["my_outcome"].map(outcome_map).fillna("?")
                counts = dfr["my_outcome_label"].value_counts().reindex(
                    ["Victoire", "Défaite", "Égalité", "Non terminé", "?"], fill_value=0
                )
                fig = go.Figure(
                    data=[go.Bar(x=counts.index, y=counts.values, marker_color="#2E86AB")]
                )
                fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
                st.plotly_chart(fig, width="stretch")

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

    with tab_friends:
        st.caption("Vue dédiée aux matchs joués avec tes amis (définis via alias XUID).")
        apply_current_filters = st.toggle(
            "Appliquer les filtres actuels (période/sessions + map/playlist)",
            value=True,
            help="Quand activé, la vue amis se limite aux matchs visibles avec les filtres.",
        )


        top = list_top_teammates(db_path, xuid.strip(), limit=20)
        default_two = [t[0] for t in top[:2]]
        all_other = list_other_player_xuids(db_path, xuid.strip(), limit=500)
        # Options: top teammates puis tous les autres xuids rencontrés.
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

        opts_map = _build_xuid_option_map(ordered)
        picked_labels = st.multiselect(
            "Coéquipiers",
            options=list(opts_map.keys()),
            default=[k for k, v in opts_map.items() if v in default_two],
            help="Sélectionne 1 ou plusieurs coéquipiers. Par défaut: les 2 plus fréquents.",
        )
        picked_xuids = [opts_map[lbl] for lbl in picked_labels if lbl in opts_map]
        same_team_only = st.checkbox("Même équipe (recommandé)", value=True, key="friends_same_team")


        if len(picked_xuids) < 1:
            st.info("Sélectionne au moins un coéquipier.")
        else:
            # Vue trio (moi + 2 coéquipiers) : uniquement si on a au moins deux personnes.
            if len(picked_xuids) >= 2:
                f1_xuid, f2_xuid = picked_xuids[0], picked_xuids[1]
                f1_name = display_name_from_xuid(f1_xuid)
                f2_name = display_name_from_xuid(f2_xuid)
                st.subheader(f"Tous les trois (même équipe) — {f1_name} + {f2_name}")

                rows_m = query_matches_with_friend(db_path, xuid.strip(), f1_xuid)
                rows_c = query_matches_with_friend(db_path, xuid.strip(), f2_xuid)
                rows_m = [r for r in rows_m if r.get("same_team")]
                rows_c = [r for r in rows_c if r.get("same_team")]
                ids_m = {r["match_id"] for r in rows_m}
                ids_c = {r["match_id"] for r in rows_c}
                trio_ids = ids_m & ids_c

                base_for_trio = dff if apply_current_filters else df
                trio_ids = trio_ids & set(base_for_trio["match_id"].astype(str))

                if not trio_ids:
                    st.warning("Aucun match trouvé où vous êtes tous les trois dans la même équipe (avec les filtres actuels).")
                else:
                    # Charge les stats de chacun et aligne par match_id.
                    me_df = base_for_trio.loc[base_for_trio["match_id"].isin(trio_ids)].copy()
                    f1_df = load_df(db_path, f1_xuid)
                    f2_df = load_df(db_path, f2_xuid)
                    f1_df = f1_df.loc[f1_df["match_id"].isin(trio_ids)].copy()
                    f2_df = f2_df.loc[f2_df["match_id"].isin(trio_ids)].copy()

                    # Aligne sur les mêmes match_id et utilise le start_time de moi comme référence d'axe.
                    me_df = me_df.sort_values("start_time")
                    f1_df = f1_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio"]].copy()
                    f2_df = f2_df[["match_id", "kills", "deaths", "assists", "accuracy", "ratio"]].copy()
                    merged = me_df[["match_id", "start_time", "kills", "deaths", "assists", "accuracy", "ratio"]].merge(
                        f1_df.add_prefix("f1_"), left_on="match_id", right_on="f1_match_id", how="inner"
                    ).merge(
                        f2_df.add_prefix("f2_"), left_on="match_id", right_on="f2_match_id", how="inner"
                    )
                    if merged.empty:
                        st.warning("Impossible d'aligner les stats des 3 joueurs sur ces matchs.")
                    else:
                        merged = merged.sort_values("start_time")
                        # Reconstitue 3 DF alignées pour le plot.
                        d_self = merged[["start_time", "kills", "deaths", "ratio", "accuracy"]].rename(
                            columns={"accuracy": "accuracy"}
                        )
                        d_f1 = merged[["start_time", "f1_kills", "f1_deaths", "f1_ratio", "f1_accuracy"]].rename(
                            columns={
                                "f1_kills": "kills",
                                "f1_deaths": "deaths",
                                "f1_ratio": "ratio",
                                "f1_accuracy": "accuracy",
                            }
                        )
                        d_f2 = merged[["start_time", "f2_kills", "f2_deaths", "f2_ratio", "f2_accuracy"]].rename(
                            columns={
                                "f2_kills": "kills",
                                "f2_deaths": "deaths",
                                "f2_ratio": "ratio",
                                "f2_accuracy": "accuracy",
                            }
                        )

                        names = (me_name, f1_name, f2_name)
                        st.plotly_chart(
                            plot_trio_metric(
                                d_self,
                                d_f1,
                                d_f2,
                                metric="kills",
                                names=names,
                                title="Kills (tous les trois)",
                                y_title="Kills",
                            ),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(
                                d_self,
                                d_f1,
                                d_f2,
                                metric="deaths",
                                names=names,
                                title="Morts / Deaths (tous les trois)",
                                y_title="Deaths",
                            ),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(
                                d_self,
                                d_f1,
                                d_f2,
                                metric="ratio",
                                names=names,
                                title="Ratio (tous les trois)",
                                y_title="Ratio",
                                y_format=".3f",
                            ),
                            width="stretch",
                        )
                        st.plotly_chart(
                            plot_trio_metric(
                                d_self,
                                d_f1,
                                d_f2,
                                metric="accuracy",
                                names=names,
                                title="Précision (tous les trois)",
                                y_title="%",
                                y_suffix="%",
                                y_format=".2f",
                            ),
                            width="stretch",
                        )


            for fx in picked_xuids:
                name = display_name_from_xuid(fx)
                rows = query_matches_with_friend(db_path, xuid.strip(), fx)
                if same_team_only:
                    rows = [r for r in rows if r.get("same_team")]
                match_ids = {r["match_id"] for r in rows}

                base_for_friends = dff if apply_current_filters else df
                sub = base_for_friends.loc[base_for_friends["match_id"].isin(match_ids)].copy()
                st.subheader(f"Avec {name}")
                if sub.empty:
                    st.warning("Aucun match trouvé (avec les filtres actuels).")
                    continue

                rates_sub = compute_outcome_rates(sub)
                total_out = max(1, rates_sub["total"])
                win_rate_sub = rates_sub["wins"] / total_out
                loss_rate_sub = rates_sub["losses"] / total_out
                global_ratio_sub = compute_global_ratio(sub)

                k = st.columns(3)
                k[0].metric("Matchs", f"{len(sub)}")
                k[1].metric("Win/Loss", f"{win_rate_sub*100:.1f}% / {loss_rate_sub*100:.1f}%")
                k[2].metric("Ratio global", f"{global_ratio_sub:.2f}" if global_ratio_sub is not None else "-")

                total_minutes_sub = (
                    pd.to_numeric(sub["time_played_seconds"], errors="coerce").dropna().sum() / 60.0
                    if "time_played_seconds" in sub.columns
                    else 0.0
                )
                kpm_sub = (float(sub["kills"].sum()) / total_minutes_sub) if total_minutes_sub > 0 else None
                dpm_sub = (float(sub["deaths"].sum()) / total_minutes_sub) if total_minutes_sub > 0 else None
                apm_sub = (float(sub["assists"].sum()) / total_minutes_sub) if total_minutes_sub > 0 else None
                per_min = st.columns(3)
                per_min[0].metric("Frags / min", f"{kpm_sub:.2f}" if kpm_sub is not None else "-")
                per_min[1].metric("Morts / min", f"{dpm_sub:.2f}" if dpm_sub is not None else "-")
                per_min[2].metric("Assistances / min", f"{apm_sub:.2f}" if apm_sub is not None else "-")

                st.plotly_chart(plot_timeseries(sub, title=f"{me_name} avec {name}"), width="stretch")

                st.plotly_chart(
                    plot_per_minute_timeseries(sub, title=f"{me_name} avec {name} — stats par minute"),
                    width="stretch",
                )

    with tab_maps:
        st.caption(
            "Compare tes performances par map. Astuce: utilise d'abord les filtres (playlist / période / sessions)."
        )

        scope = st.radio(
            "Scope",
            options=["Moi (filtres actuels)", "Moi (toutes les parties)", "Avec Madina972", "Avec Chocoboflor"],
            horizontal=True,
        )
        min_matches = st.slider("Minimum de matchs par map", 1, 30, 5, step=1)

        show_delta = st.toggle(
            "Afficher en diff vs baseline",
            value=False,
            help="Affiche la différence (scope - baseline) par map, pour voir où tu es meilleur/moins bon.",
        )
        baseline_scope_label = None
        if show_delta:
            baseline_scope_label = st.selectbox(
                "Baseline",
                options=["Moi (toutes les parties)", "Moi (filtres actuels)"],
                index=0,
            )

        if scope == "Moi (toutes les parties)":
            base_scope = base
        elif scope == "Avec Madina972":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2533274858283686")
            rows = [r for r in rows if r.get("same_team")]
            match_ids = {r["match_id"] for r in rows}
            base_scope = base.loc[base["match_id"].isin(match_ids)].copy()
        elif scope == "Avec Chocoboflor":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2535469190789936")
            rows = [r for r in rows if r.get("same_team")]
            match_ids = {r["match_id"] for r in rows}
            base_scope = base.loc[base["match_id"].isin(match_ids)].copy()
        else:
            base_scope = dff

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

            if show_delta and baseline_scope_label is not None:
                baseline_df = df if baseline_scope_label == "Moi (toutes les parties)" else dff
                baseline_breakdown = compute_map_breakdown(baseline_df)
                baseline_breakdown = baseline_breakdown.loc[
                    baseline_breakdown["matches"] >= int(min_matches)
                ].copy()

                merged = breakdown.merge(
                    baseline_breakdown[["map_name", key]].rename(columns={key: f"{key}_baseline"}),
                    on="map_name",
                    how="left",
                )
                merged[f"{key}_delta"] = merged[key] - merged[f"{key}_baseline"]

                view = merged.dropna(subset=[f"{key}_delta"]).sort_values(
                    f"{key}_delta", ascending=False
                ).head(20)
                view = view.iloc[::-1]

                title = f"Δ {label} par map — {scope} vs {baseline_scope_label} (min {min_matches} matchs)"
                fig = plot_map_comparison(view.rename(columns={f"{key}_delta": key}), key, title=title)
            else:
                view = breakdown.head(20).iloc[::-1]  # top 20, affichage vertical lisible
                title = f"{label} par map — {scope} (min {min_matches} matchs)"
                fig = plot_map_comparison(view, key, title=title)

            # Ajuste l'affichage des pourcentages.
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

            if show_delta and baseline_scope_label is not None:
                baseline_df = df if baseline_scope_label == "Moi (toutes les parties)" else dff
                baseline_tbl = compute_map_breakdown(baseline_df)
                baseline_tbl = baseline_tbl.loc[baseline_tbl["matches"] >= int(min_matches)].copy()
                baseline_tbl["win_rate"] = (baseline_tbl["win_rate"] * 100).round(1)
                baseline_tbl["loss_rate"] = (baseline_tbl["loss_rate"] * 100).round(1)
                baseline_tbl["accuracy_avg"] = baseline_tbl["accuracy_avg"].round(2)
                baseline_tbl["ratio_global"] = baseline_tbl["ratio_global"].round(2)

                tbl = tbl.merge(
                    baseline_tbl[["map_name", "accuracy_avg", "win_rate", "ratio_global"]].rename(
                        columns={
                            "accuracy_avg": "Accuracy baseline (%)",
                            "win_rate": "Win baseline (%)",
                            "ratio_global": "Ratio baseline",
                        }
                    ),
                    on="map_name",
                    how="left",
                )
                tbl["Δ Accuracy (%)"] = (tbl["accuracy_avg"] - tbl["Accuracy baseline (%)"]).round(2)
                tbl["Δ Win (%)"] = (tbl["win_rate"] - tbl["Win baseline (%)"]).round(1)
                tbl["Δ Ratio"] = (tbl["ratio_global"] - tbl["Ratio baseline"]).round(2)
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

    with tab_table:
        st.subheader("Historique des parties")
        dff = dff.copy()
        dff["match_url"] = (
            "https://www.halowaypoint.com/halo-infinite/players/"
            + waypoint_player.strip()
            + "/matches/"
            + dff["match_id"].astype(str)
        )

        outcome_map = {
            2: "Victoire",
            3: "Défaite",
            1: "Égalité",
            4: "Non terminé",
        }
        dff["outcome_label"] = dff["outcome"].map(outcome_map).fillna("-")

        show_cols = [
            "match_url",
            "start_time",
            "map_name",
            "playlist_name",
            "outcome_label",
            "kda",
            "kills",
            "deaths",
            "max_killing_spree",
            "headshot_kills",
            "average_life_seconds",
            "assists",
            "accuracy",
            "ratio",
        ]
        table = dff[show_cols].sort_values("start_time", ascending=False).reset_index(drop=True)

        def _style_outcome(v: str) -> str:
            s = (v or "").strip().lower()
            if s == "victoire":
                return "color: #1B5E20; font-weight: 700;"
            if s == "défaite" or s == "defaite":
                return "color: #B71C1C; font-weight: 700;"
            if s == "égalité" or s == "egalite":
                return "color: #8E6CFF; font-weight: 700;"
            if s == "non terminé" or s == "non termine":
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

        styled = table.style.applymap(_style_outcome, subset=["outcome_label"]).applymap(
            _style_kda, subset=["kda"]
        )

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "match_url": st.column_config.LinkColumn(
                    "Consulter sur HaloWaypoint",
                    display_text="Ouvrir",
                    help="Ouvre la page HaloWaypoint du match",
                ),
                "map_name": st.column_config.TextColumn("Carte"),
                "playlist_name": st.column_config.TextColumn("Playlist"),
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
