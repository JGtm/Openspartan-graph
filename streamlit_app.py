import os
import re
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from openspartan_graph import _guess_xuid_from_db_path, load_matches, query_matches_with_friend


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
        fig.update_layout(template="plotly_white", height=360, margin=dict(l=40, r=20, t=30, b=40))
        return fig

    fig = go.Figure(
        data=[
            go.Bar(
                x=d[metric],
                y=d["map_name"],
                orientation="h",
                marker_color="#2E86AB",
                customdata=list(zip(d["matches"], d.get("accuracy_avg"))),
                hovertemplate=(
                    "%{y}<br>value=%{x}<br>matches=%{customdata[0]}"
                    "<br>accuracy=%{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_spree_headshots_accuracy(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["max_killing_spree"],
            mode="lines+markers",
            name="Max killing spree",
            line=dict(width=2, color="#FFB020"),
            marker=dict(size=6, color="#FFB020"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>spree=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["headshot_kills"],
            mode="lines+markers",
            name="Headshot kills",
            yaxis="y2",
            line=dict(width=2, color="#E84A5F"),
            marker=dict(size=6, color="#E84A5F"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>headshots=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["accuracy"],
            mode="lines+markers",
            name="Accuracy (%)",
            yaxis="y3",
            line=dict(width=2, color="#7B61FF"),
            marker=dict(size=6, color="#7B61FF"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>accuracy=%{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=50, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        yaxis=dict(title="Spree", rangemode="tozero"),
        yaxis2=dict(
            title="Headshots",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        yaxis3=dict(
            title="Accuracy (%)",
            overlaying="y",
            side="right",
            position=1.0,
            showgrid=False,
            rangemode="tozero",
        ),
    )
    return fig


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


def plot_timeseries(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    common_hover = (
        "%{x|%Y-%m-%d %H:%M}<br>"
        "kills=%{customdata[0]} deaths=%{customdata[1]} assists=%{customdata[2]}<br>"
        "acc=%{customdata[3]} ratio=%{customdata[4]:.3f}<br>"
        "map=%{customdata[5]}<br>"
        "playlist=%{customdata[6]}<br>"
        "match=%{customdata[7]}<extra></extra>"
    )

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

    fig.add_trace(
        go.Scatter(
            x=df["start_time"],
            y=df["kills"],
            mode="lines+markers",
            name="Kills",
            line=dict(width=2, color="#2E86AB"),
            marker=dict(size=6, color="#2E86AB"),
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
            name="Deaths",
            line=dict(width=2, color="#D1495B"),
            marker=dict(size=6, color="#D1495B"),
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
            line=dict(width=2, color="#3A7D44"),
            marker=dict(size=6, color="#3A7D44"),
            customdata=customdata,
            hovertemplate=common_hover,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=40),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Kills / Deaths", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Ratio", secondary_y=True)
    return fig


def plot_accuracy_last_n(df: pd.DataFrame, n: int) -> go.Figure:
    d = df.dropna(subset=["accuracy"]).tail(n)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=d["start_time"],
                y=d["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(width=2, color="#7B61FF"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>accuracy=%{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_yaxes(title_text="%", rangemode="tozero")
    return fig


def plot_average_life(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["average_life_seconds"]).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["start_time"],
            y=d["average_life_seconds"],
            mode="lines+markers",
            name="Average life (s)",
            line=dict(width=2, color="#00A389"),
            marker=dict(size=6, color="#00A389"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>avg life=%{y:.1f}s<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Secondes", rangemode="tozero")
    return fig


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
    colors = ["#2E86AB", "#D1495B", "#3A7D44"]
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
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    + f"{metric}=%{{y{':' + y_format if y_format else ''}}}{y_suffix}"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=360,
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text=y_title)
    if y_suffix and not y_format:
        fig.update_yaxes(ticksuffix=y_suffix)
    return fig


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
    fig.add_bar(x=pivot.index, y=wins, name="Wins", marker_color="#3A7D44")
    fig.add_bar(x=pivot.index, y=losses, name="Losses", marker_color="#D1495B")
    fig.add_bar(x=pivot.index, y=ties, name="Ties", marker_color="#2E86AB")
    fig.add_bar(x=pivot.index, y=nofin, name="NoFinishes", marker_color="#8E8E8E")
    fig.update_layout(template="plotly_white", barmode="stack", height=360, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_yaxes(title_text="Matches")
    return fig


def plot_kda_distribution(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["kda"]).copy()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=d["kda"],
                nbinsx=40,
                marker_color="#2E86AB",
                hovertemplate="KDA=%{x:.2f}<br>count=%{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_xaxes(title_text="KDA")
    fig.update_yaxes(title_text="Count")
    return fig


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


def main() -> None:
    st.set_page_config(page_title="OpenSpartan Graphs", layout="wide")

    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
          h1 {letter-spacing: -0.02em;}
          [data-testid="stSidebar"] {border-right: 1px solid rgba(49,51,63,0.12);}
          .stMetric {background: rgba(46,134,171,0.06); padding: 12px 14px; border-radius: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("OpenSpartan Graphs (local)")

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

    with st.sidebar:
        st.header("Filtres")
        dmin, dmax = _date_range(df)
        filter_mode = st.radio("Sélection", options=["Période", "Sessions"], horizontal=True)

        start_d, end_d = dmin, dmax
        gap_minutes = 35

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
            "Map",
            options=["(toutes)"] + list(map_opts.keys()),
            index=0,
        )
        map_id: Optional[str] = None
        if map_label != "(toutes)":
            map_id = map_opts[map_label]

        last_n_acc = st.slider("Précision: derniers matchs", 5, 50, 20, step=1)

        if filter_mode == "Période":
            start_d, end_d = st.date_input(
                "Période",
                value=(dmin, dmax),
                min_value=dmin,
                max_value=dmax,
            )
        else:
            gap_minutes = st.slider(
                "Écart max entre parties (minutes)",
                min_value=15,
                max_value=90,
                value=35,
                step=5,
                help="Au-delà de cet écart, on considère que c'est une nouvelle session.",
            )

    # Apply filters (map/playlist puis période ou sessions)
    base = df.copy()
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

    kpi = st.columns(5)
    kpi[0].metric("Accuracy moyenne", f"{avg_acc:.2f}%" if avg_acc is not None else "-")
    kpi[1].metric("Win rate", f"{win_rate*100:.1f}%" if rates["total"] else "-")
    kpi[2].metric("Loss rate", f"{loss_rate*100:.1f}%" if rates["total"] else "-")
    kpi[3].metric("Ratio global", f"{global_ratio:.2f}" if global_ratio is not None else "-")
    kpi[4].metric("Average life", f"{avg_life:.1f}s" if avg_life == avg_life else "-")

    st.caption(
        f"Matchs sélectionnés: {len(dff)} — wins={rates['wins']} losses={rates['losses']} ties={rates['ties']} noFinish={rates['nofinish']}"
    )

    tab_series, tab_mom, tab_kda, tab_friend, tab_friends, tab_maps, tab_table = st.tabs(
        [
            "Séries temporelles",
            "Wins/Losses MoM",
            "KDA (distribution)",
            "Avec un joueur",
            "Avec mes amis",
            "Comparaison maps",
            "Table",
        ]
    )

    with tab_series:
        fig = plot_timeseries(dff, title=f"{me_name}")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Durée de vie moyenne (Average Life)")
        if dff.dropna(subset=["average_life_seconds"]).empty:
            st.info("Average Life indisponible sur ce filtre.")
        else:
            st.plotly_chart(plot_average_life(dff), width="stretch")

        st.subheader("Spree / Headshots / Accuracy")
        st.plotly_chart(plot_spree_headshots_accuracy(dff), width="stretch")

        st.subheader("Précision (accuracy) — derniers matchs")
        st.plotly_chart(plot_accuracy_last_n(dff, last_n_acc), width="stretch")

    with tab_mom:
        st.caption("Basé sur Players[].Outcome (2=win, 3=loss, 1=tie, 4=no finish).")
        st.plotly_chart(plot_outcomes_mom(dff), width="stretch")

    with tab_kda:
        valid = dff.dropna(subset=["kda"]) if "kda" in dff.columns else pd.DataFrame()
        if valid.empty:
            st.warning("KDA indisponible sur ce filtre.")
        else:
            st.metric("KDA médiane", f"{valid['kda'].median():.2f}")
            st.metric("KDA moyenne", f"{valid['kda'].mean():.2f}")
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
                outcome_map = {2: "Win", 3: "Loss", 1: "Tie", 4: "NoFinish"}
                dfr["my_outcome_label"] = dfr["my_outcome"].map(outcome_map).fillna("?")
                counts = dfr["my_outcome_label"].value_counts().reindex(["Win", "Loss", "Tie", "NoFinish", "?"], fill_value=0)
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

        friends = {
            "Madina972": "2533274858283686",
            "Chocoboflor": "2535469190789936",
        }

        picked_names = st.multiselect(
            "Amis",
            options=list(friends.keys()),
            default=list(friends.keys()),
        )
        same_team_only = st.checkbox("Même équipe (recommandé)", value=True, key="friends_same_team")

        if not picked_names:
            st.info("Sélectionne au moins un ami.")
        else:
            # Vue trio (moi + les 2 amis) : uniquement si les 2 sont sélectionnés.
            if set(picked_names) >= {"Madina972", "Chocoboflor"}:
                st.subheader("Tous les trois (même équipe)")
                rows_m = query_matches_with_friend(db_path, xuid.strip(), friends["Madina972"])
                rows_c = query_matches_with_friend(db_path, xuid.strip(), friends["Chocoboflor"])
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
                    f1_name, f2_name = "Madina972", "Chocoboflor"
                    f1_df = load_df(db_path, friends[f1_name])
                    f2_df = load_df(db_path, friends[f2_name])
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
                                title="Accuracy (tous les trois)",
                                y_title="%",
                                y_suffix="%",
                                y_format=".2f",
                            ),
                            width="stretch",
                        )

            for name in picked_names:
                fx = friends[name]
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
                avg_acc_sub = sub["accuracy"].dropna().mean()
                global_ratio_sub = compute_global_ratio(sub)

                k = st.columns(4)
                k[0].metric("Matchs", f"{len(sub)}")
                k[1].metric("Accuracy moy.", f"{avg_acc_sub:.2f}%" if avg_acc_sub == avg_acc_sub else "-")
                k[2].metric("Win/Loss", f"{win_rate_sub*100:.1f}% / {loss_rate_sub*100:.1f}%")
                k[3].metric("Ratio global", f"{global_ratio_sub:.2f}" if global_ratio_sub is not None else "-")

                st.plotly_chart(plot_timeseries(sub, title=f"{me_name} avec {name}"), width="stretch")

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
            base_scope = df
        elif scope == "Avec Madina972":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2533274858283686")
            rows = [r for r in rows if r.get("same_team")]
            match_ids = {r["match_id"] for r in rows}
            base_scope = df.loc[df["match_id"].isin(match_ids)].copy()
        elif scope == "Avec Chocoboflor":
            rows = query_matches_with_friend(db_path, xuid.strip(), "2535469190789936")
            rows = [r for r in rows if r.get("same_team")]
            match_ids = {r["match_id"] for r in rows}
            base_scope = df.loc[df["match_id"].isin(match_ids)].copy()
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
                    ("accuracy_avg", "Accuracy moyenne"),
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
                        "map_name": "Map",
                        "matches": "Matchs",
                        "accuracy_avg": "Accuracy avg (%)",
                        "win_rate": "Win rate (%)",
                        "loss_rate": "Loss rate (%)",
                        "ratio_global": "Ratio global",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_table:
        st.subheader("Table")
        dff = dff.copy()
        dff["match_url"] = (
            "https://www.halowaypoint.com/halo-infinite/players/"
            + waypoint_player.strip()
            + "/matches/"
            + dff["match_id"].astype(str)
        )
        show_cols = [
            "start_time",
            "map_name",
            "playlist_name",
            "outcome",
            "kda",
            "max_killing_spree",
            "headshot_kills",
            "average_life_seconds",
            "kills",
            "deaths",
            "assists",
            "accuracy",
            "ratio",
            "match_url",
        ]
        table = dff[show_cols].sort_values("start_time", ascending=False).reset_index(drop=True)
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "match_url": st.column_config.LinkColumn(
                    "Match",
                    display_text="Ouvrir",
                    help="Ouvre la page HaloWaypoint du match",
                )
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
