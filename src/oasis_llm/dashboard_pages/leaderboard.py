"""Leaderboard page: rank every run by alignment with OASIS human norms."""
from __future__ import annotations

import streamlit as st

from oasis_llm import analyses as an
from oasis_llm.dashboard_pages._ui import (
    connect_ro, db_locked_warning, kpi, page_header,
)


def render():
    page_header(
        "Leaderboard",
        "Every run scored against OASIS human norms (Pearson r, Spearman ρ, MAE, RMSE).",
        icon="🏆",
    )
    con = connect_ro()
    if con is None:
        db_locked_warning(); return

    cs = st.columns([1, 1, 2])
    min_images = cs[0].number_input(
        "Min images", min_value=1, max_value=900, value=10, step=1,
        help="Drop runs with fewer than this many images per dimension.",
    )
    sort_by = cs[1].selectbox(
        "Sort by", ["pearson_r", "spearman_rho", "mae", "rmse"], index=0,
    )
    dim_filter = cs[2].multiselect(
        "Dimensions", ["valence", "arousal", "dominance"],
        default=["valence", "arousal"],
    )

    with st.spinner("Scoring runs against OASIS norms…"):
        df = an.leaderboard(con, min_images=int(min_images))
    if df.empty:
        st.info("No runs have enough completed trials matched to OASIS norms yet.")
        return

    if dim_filter:
        df = df[df["dimension"].isin(dim_filter)]
    ascending = sort_by in ("mae", "rmse")
    df = df.sort_values(["dimension", sort_by], ascending=[True, ascending])

    # KPIs
    cs = st.columns(3)
    cs[0].markdown(kpi("Runs scored", df["run_id"].nunique()), unsafe_allow_html=True)
    cs[1].markdown(
        kpi("Best Pearson r",
            f"{df['pearson_r'].max():.3f}" if not df.empty else "—"),
        unsafe_allow_html=True,
    )
    cs[2].markdown(
        kpi("Median MAE",
            f"{df['mae'].median():.3f}" if not df.empty else "—"),
        unsafe_allow_html=True,
    )

    # Per-dimension tables
    for dim, sub in df.groupby("dimension"):
        st.markdown(f"### {dim.title()}")
        view = sub[[
            "run_id", "model", "provider", "n_images",
            "pearson_r", "spearman_rho", "mae", "rmse",
            "mean_pred", "mean_human", "temperature",
        ]].reset_index(drop=True)
        st.dataframe(
            view,
            width='stretch',
            hide_index=True,
            column_config={
                "pearson_r": st.column_config.NumberColumn(
                    "r (Pearson)", format="%.3f"),
                "spearman_rho": st.column_config.NumberColumn(
                    "ρ (Spearman)", format="%.3f"),
                "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "mean_pred": st.column_config.NumberColumn(
                    "mean pred", format="%.2f"),
                "mean_human": st.column_config.NumberColumn(
                    "mean human", format="%.2f"),
                "temperature": st.column_config.NumberColumn(
                    "T", format="%.1f"),
            },
        )

    # CSV export
    st.download_button(
        "Download leaderboard.csv",
        data=df.to_csv(index=False).encode(),
        file_name="leaderboard.csv",
        mime="text/csv",
    )
