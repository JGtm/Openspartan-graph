"""Outils de mesure de performance (Streamlit).

Streamlit rerun le script à chaque interaction. Ce module fournit un mode
"perf" simple pour mesurer les sections clés (sidebar, chargement DB, filtres,
charts) sans dépendance externe.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

import pandas as pd
import streamlit as st


_PERF_ENABLED_KEY = "perf_enabled"
_PERF_TIMINGS_KEY = "_perf_timings_ms"


def perf_enabled() -> bool:
    return bool(st.session_state.get(_PERF_ENABLED_KEY, False))


def perf_reset_run() -> None:
    if perf_enabled():
        st.session_state[_PERF_TIMINGS_KEY] = []


@contextmanager
def perf_section(name: str) -> Iterator[None]:
    if not perf_enabled():
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rows = st.session_state.setdefault(_PERF_TIMINGS_KEY, [])
        rows.append({"section": str(name), "ms": float(dt_ms)})


def perf_dataframe() -> pd.DataFrame:
    rows = st.session_state.get(_PERF_TIMINGS_KEY, [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["section", "ms"])
    return pd.DataFrame(rows)


def render_perf_panel(*, location: str = "sidebar") -> None:
    if location == "sidebar":
        container = st.sidebar
    else:
        container = st

    container.checkbox("Mode perf", key=_PERF_ENABLED_KEY)

    if not perf_enabled():
        return

    c = container.columns(2)
    if c[0].button("Reset timings", width="stretch"):
        st.session_state[_PERF_TIMINGS_KEY] = []
        st.rerun()

    df = perf_dataframe()
    if df.empty:
        c[1].caption("En attente…")
        return

    total = float(df["ms"].sum())
    c[1].caption(f"Total: {total:.0f} ms")

    container.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "section": st.column_config.TextColumn("Section"),
            "ms": st.column_config.NumberColumn("ms", format="%.0f"),
        },
    )
