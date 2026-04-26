"""OASIS-LLM Streamlit control plane.

Pages:
  Home          — at-a-glance summary, recent activity
  Datasets      — generate, review, approve curated image sets
  Image Explorer— browse the full OASIS pool with filters
  Experiments   — multi-config rating campaigns
  Runs          — read-only run inspector
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="OASIS-LLM",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

from oasis_llm.dashboard_pages._ui import apply_theme

apply_theme()

PAGES = {
    "Home":           "🏠",
    "Datasets":       "📁",
    "Image Explorer": "🖼️",
    "Experiments":    "🧪",
    "Queue":          "⏭️",
    "Runs":           "📊",
    "Compare Runs":   "🔀",
    "Analyses":       "🔬",
    "Leaderboard":    "🏆",
    "Settings":       "⚙️",
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0.25rem 1rem;">
          <div style="font-weight: 700; font-size: 1.15rem; letter-spacing: 0.02em;">OASIS-LLM</div>
          <div style="color:#8a8aa0; font-size: 0.78rem;">Control plane</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda k: f"{PAGES[k]}  {k}",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("⚡ Local DuckDB · single writer")
    st.caption("📚 [Docs](http://localhost:3333)")

if page == "Home":
    from oasis_llm.dashboard_pages import home
    home.render()
elif page == "Datasets":
    from oasis_llm.dashboard_pages import datasets as datasets_page
    datasets_page.render()
elif page == "Image Explorer":
    from oasis_llm.dashboard_pages import explorer
    explorer.render()
elif page == "Experiments":
    from oasis_llm.dashboard_pages import experiments as experiments_page
    experiments_page.render()
elif page == "Queue":
    from oasis_llm.dashboard_pages import queue as queue_page
    queue_page.render()
elif page == "Runs":
    from oasis_llm.dashboard_pages import runs
    runs.render()
elif page == "Compare Runs":
    from oasis_llm.dashboard_pages import compare_runs
    compare_runs.render()
elif page == "Analyses":
    from oasis_llm.dashboard_pages import analyses
    analyses.render()
elif page == "Leaderboard":
    from oasis_llm.dashboard_pages import leaderboard
    leaderboard.render()
elif page == "Settings":
    from oasis_llm.dashboard_pages import settings
    settings.render()
