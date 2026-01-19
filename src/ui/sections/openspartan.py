"""Section UI: outils OpenSpartan (lancer Workshop)."""

from __future__ import annotations

import os
import streamlit as st

from src.config import get_default_workshop_exe_path


def render_openspartan_tools() -> None:
    st.divider()
    st.header("OpenSpartan")

    workshop_exe = st.text_input(
        "Chemin de OpenSpartan.Workshop.exe",
        value=get_default_workshop_exe_path(),
        help="Bouton pratique pour lancer l'app OpenSpartan Workshop.",
    )

    if st.button("Lancer OpenSpartan Workshop", width="stretch"):
        if not os.path.exists(workshop_exe):
            st.error("Executable introuvable à ce chemin.")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(workshop_exe)  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen([workshop_exe], close_fds=True)
            st.success("OpenSpartan Workshop lancé.")
        except Exception as e:
            st.error(f"Impossible de lancer OpenSpartan Workshop: {e}")
