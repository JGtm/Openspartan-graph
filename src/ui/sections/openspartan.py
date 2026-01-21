"""Section UI: outils OpenSpartan (lancer Workshop)."""

from __future__ import annotations

import os
import streamlit as st

from src.config import get_default_workshop_exe_path


def render_openspartan_tools() -> None:
    st.divider()
    st.header("OpenSpartan")

    if st.button("Lancer OpenSpartan Workshop", width="stretch", help="Lance l'app OpenSpartan Workshop"):
        workshop_exe = (os.environ.get("OPENSPARTAN_WORKSHOP_EXE") or get_default_workshop_exe_path()).strip()
        if not workshop_exe or not os.path.exists(workshop_exe):
            st.error("Executable introuvable. Tu peux définir OPENSPARTAN_WORKSHOP_EXE si besoin.")
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
