"""Image Explorer: browse the OASIS pool with category and norm filters."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import duckdb
import streamlit as st

from oasis_llm.dashboard_pages._ui import kpi, page_header
from oasis_llm.images import all_image_ids, image_categories, image_path

NORMS_CSV = Path("OASIS/OASIS.csv")


@lru_cache(maxsize=1)
def _norms_df():
    if not NORMS_CSV.exists():
        return None
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT "Theme" AS image_id,
               "Category" AS category,
               "Source" AS source,
               "Valence_mean" AS valence_mean,
               "Valence_SD"   AS valence_sd,
               "Arousal_mean" AS arousal_mean,
               "Arousal_SD"   AS arousal_sd
        FROM read_csv_auto('{NORMS_CSV}', header=true)
        """
    ).fetchdf()
    return df


def render():
    page_header(
        "Image Explorer",
        "Browse the full OASIS pool. Filter by category and human-norm valence/arousal means.",
        icon="🖼️",
    )
    df = _norms_df()
    if df is None or df.empty:
        st.error(f"Couldn't load norms CSV at `{NORMS_CSV}`.")
        return

    cats = image_categories()
    all_ids = set(all_image_ids())
    df = df[df["image_id"].isin(all_ids)].copy()
    # Use the long-CSV category mapping where available (it's the canonical theme/category)
    df["category_resolved"] = df["image_id"].map(lambda i: cats.get(i) or "Unknown")

    # KPI
    cols = st.columns(4)
    cols[0].markdown(kpi("Total images", len(df)), unsafe_allow_html=True)
    cols[1].markdown(kpi("Categories", df["category_resolved"].nunique()), unsafe_allow_html=True)
    cols[2].markdown(
        kpi("Valence range", f"{df['valence_mean'].min():.1f}–{df['valence_mean'].max():.1f}"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        kpi("Arousal range", f"{df['arousal_mean'].min():.1f}–{df['arousal_mean'].max():.1f}"),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Filter bar
    f1, f2, f3, f4 = st.columns([2, 3, 3, 2])
    with f1:
        cat_options = ["All"] + sorted(df["category_resolved"].unique().tolist())
        selected_cat = st.selectbox("Category", cat_options)
    with f2:
        v_min, v_max = float(df["valence_mean"].min()), float(df["valence_mean"].max())
        v_range = st.slider(
            "Valence mean", min_value=round(v_min, 1), max_value=round(v_max, 1),
            value=(round(v_min, 1), round(v_max, 1)), step=0.1,
        )
    with f3:
        a_min, a_max = float(df["arousal_mean"].min()), float(df["arousal_mean"].max())
        a_range = st.slider(
            "Arousal mean", min_value=round(a_min, 1), max_value=round(a_max, 1),
            value=(round(a_min, 1), round(a_max, 1)), step=0.1,
        )
    with f4:
        sort_by = st.selectbox(
            "Sort by",
            ["image_id", "valence_mean ↑", "valence_mean ↓", "arousal_mean ↑", "arousal_mean ↓"],
        )

    # Apply filters
    filtered = df[
        (df["valence_mean"] >= v_range[0]) & (df["valence_mean"] <= v_range[1]) &
        (df["arousal_mean"] >= a_range[0]) & (df["arousal_mean"] <= a_range[1])
    ]
    if selected_cat != "All":
        filtered = filtered[filtered["category_resolved"] == selected_cat]

    if sort_by == "image_id":
        filtered = filtered.sort_values("image_id")
    else:
        col, direction = sort_by.split(" ")
        filtered = filtered.sort_values(col, ascending=(direction == "↑"))

    st.caption(f"**{len(filtered):,}** images match.")
    st.markdown("---")

    # Pagination
    PAGE_SIZE = 30
    page_idx = st.session_state.get("explorer_page", 0)
    n_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page_idx = min(page_idx, n_pages - 1)
    pcols = st.columns([1, 4, 1])
    with pcols[0]:
        if st.button("‹ Prev", disabled=(page_idx == 0), width='stretch'):
            st.session_state["explorer_page"] = max(0, page_idx - 1)
            st.rerun()
    with pcols[1]:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.5rem; color:#8a8aa0;'>"
            f"Page {page_idx + 1} of {n_pages} · {PAGE_SIZE} per page"
            f"</div>",
            unsafe_allow_html=True,
        )
    with pcols[2]:
        if st.button("Next ›", disabled=(page_idx >= n_pages - 1), width='stretch'):
            st.session_state["explorer_page"] = min(n_pages - 1, page_idx + 1)
            st.rerun()

    page_df = filtered.iloc[page_idx * PAGE_SIZE:(page_idx + 1) * PAGE_SIZE]
    n_cols = 5
    rows = [page_df.iloc[i:i + n_cols] for i in range(0, len(page_df), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, (_, img) in zip(cols, row.iterrows()):
            with col:
                p = image_path(img["image_id"])
                if p.exists():
                    st.image(str(p), width='stretch')
                st.markdown(
                    f"<div style='font-weight:600; font-size:0.88rem;'>{img['image_id']}</div>"
                    f"<div style='color:#8a8aa0; font-size:0.75rem;'>{img['category_resolved']}</div>"
                    f"<div style='font-size:0.78rem;'>"
                    f"V <b>{img['valence_mean']:.2f}</b> &nbsp; "
                    f"A <b>{img['arousal_mean']:.2f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
