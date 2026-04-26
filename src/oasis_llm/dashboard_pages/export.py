"""Global Export page — bundle multiple entities into a single .zip.

Lets the user pick any combination of datasets / experiments / analyses
(filtered by name + favourites) and produces a single nested zip via
``bundles.export_bundle``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from oasis_llm import analyses as an
from oasis_llm import datasets as ds
from oasis_llm import experiments as ex
from oasis_llm import favorites as fav
from oasis_llm.bundles import export_bundle
from oasis_llm.dashboard_pages._ui import (
    connect_ro, db_locked_warning, kpi, page_header,
)


def render():
    page_header(
        "Export",
        "Bundle datasets, experiments, and analyses into a single zip for backup or sharing.",
        icon="📦",
    )

    con = connect_ro()
    if con is None:
        db_locked_warning(); return

    datasets = ds.list_all(con)
    experiments = ex.list_all(con)
    analyses = an.list_all(con)

    starred_ds = fav.starred_set(con, "dataset")
    starred_ex = fav.starred_set(con, "experiment")
    starred_an = fav.starred_set(con, "analysis")

    cols = st.columns(4)
    cols[0].markdown(kpi("Datasets", len(datasets)), unsafe_allow_html=True)
    cols[1].markdown(kpi("Experiments", len(experiments)), unsafe_allow_html=True)
    cols[2].markdown(kpi("Analyses", len(analyses)), unsafe_allow_html=True)
    cols[3].markdown(
        kpi("⭐ Starred", len(starred_ds) + len(starred_ex) + len(starred_an)),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    ctrl = st.columns([2, 2, 1])
    with ctrl[0]:
        search = st.text_input(
            "🔎 Filter by name / id",
            value="",
            key="export_search",
            placeholder="substring match (case-insensitive)",
        )
    with ctrl[1]:
        starred_only = st.toggle(
            "⭐ Starred only",
            value=False,
            key="export_starred_only",
            help="Restrict the lists below to items you've starred.",
        )
    with ctrl[2]:
        select_all_starred = st.button(
            "✅ Select starred",
            help="Tick every starred item in all three lists.",
            width='stretch',
        )

    if select_all_starred:
        for did in starred_ds:
            st.session_state[f"exp_pick_dataset_{did}"] = True
        for eid in starred_ex:
            st.session_state[f"exp_pick_experiment_{eid}"] = True
        for aid in starred_an:
            st.session_state[f"exp_pick_analysis_{aid}"] = True
        st.rerun()

    needle = search.strip().lower()

    def _match(*fields: str) -> bool:
        if not needle:
            return True
        return any(needle in (f or "").lower() for f in fields)

    # ─── three side-by-side picker columns ─────────────────────────────────
    pick_cols = st.columns(3)

    def _bulk_buttons(kind: str, ids: list[str]) -> None:
        b1, b2 = st.columns(2)
        if b1.button("✅ All visible", key=f"exp_all_{kind}", width='stretch'):
            for x in ids:
                st.session_state[f"exp_pick_{kind}_{x}"] = True
            st.rerun()
        if b2.button("⬜ Clear", key=f"exp_none_{kind}", width='stretch'):
            for x in ids:
                st.session_state[f"exp_pick_{kind}_{x}"] = False
            st.rerun()

    with pick_cols[0]:
        st.markdown("### 📁 Datasets")
        rows = [
            d for d in datasets
            if _match(d.dataset_id, d.name, d.description)
            and (not starred_only or d.dataset_id in starred_ds)
        ]
        if rows:
            _bulk_buttons("dataset", [d.dataset_id for d in rows])
        else:
            st.caption("No datasets match the current filter.")
        for d in rows:
            star = "⭐ " if d.dataset_id in starred_ds else ""
            st.checkbox(
                f"{star}{d.dataset_id} — {d.name or ''}",
                key=f"exp_pick_dataset_{d.dataset_id}",
            )

    with pick_cols[1]:
        st.markdown("### 🧪 Experiments")
        rows = [
            e for e in experiments
            if _match(e.experiment_id, e.name, e.description)
            and (not starred_only or e.experiment_id in starred_ex)
        ]
        if rows:
            _bulk_buttons("experiment", [e.experiment_id for e in rows])
        else:
            st.caption("No experiments match the current filter.")
        for e in rows:
            star = "⭐ " if e.experiment_id in starred_ex else ""
            st.checkbox(
                f"{star}{e.name} ({e.experiment_id[:8]}…)",
                key=f"exp_pick_experiment_{e.experiment_id}",
            )

    with pick_cols[2]:
        st.markdown("### 🔬 Analyses")
        rows = [
            a for a in analyses
            if _match(a.analysis_id, a.name, a.description)
            and (not starred_only or a.analysis_id in starred_an)
        ]
        if rows:
            _bulk_buttons("analysis", [a.analysis_id for a in rows])
        else:
            st.caption("No analyses match the current filter.")
        for a in rows:
            star = "⭐ " if a.analysis_id in starred_an else ""
            st.checkbox(
                f"{star}{a.name} ({a.analysis_id[:8]}…)",
                key=f"exp_pick_analysis_{a.analysis_id}",
            )

    # ─── action row ────────────────────────────────────────────────────────
    selected_ds = [
        d.dataset_id for d in datasets
        if st.session_state.get(f"exp_pick_dataset_{d.dataset_id}")
    ]
    selected_ex = [
        e.experiment_id for e in experiments
        if st.session_state.get(f"exp_pick_experiment_{e.experiment_id}")
    ]
    selected_an = [
        a.analysis_id for a in analyses
        if st.session_state.get(f"exp_pick_analysis_{a.analysis_id}")
    ]
    total_selected = len(selected_ds) + len(selected_ex) + len(selected_an)

    st.markdown("---")
    st.caption(
        f"**Selected:** {len(selected_ds)} datasets · "
        f"{len(selected_ex)} experiments · "
        f"{len(selected_an)} analyses"
    )

    action = st.columns([1, 1, 2])
    with action[0]:
        include_images = st.checkbox(
            "Bundle dataset image files",
            value=False,
            key="export_include_images",
            help=(
                "Include the OASIS .jpg files inside the zip so the receiving "
                "workspace doesn't need a local images checkout. Adds roughly "
                "100 KB per image — large datasets can produce ≥100 MB zips."
            ),
        )
    with action[1]:
        build = st.button(
            "📦 Build bundle",
            disabled=total_selected == 0,
            type="primary",
            width='stretch',
        )

    if build:
        with st.spinner("Packaging…"):
            try:
                blob = export_bundle(
                    con,
                    dataset_ids=selected_ds,
                    experiment_ids=selected_ex,
                    analysis_ids=selected_an,
                    include_images=include_images,
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
                return
        st.session_state["export_blob"] = blob
        st.session_state["export_blob_name"] = (
            f"oasis_export_{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}.zip"
        )
        st.success(f"Bundle ready · {len(blob) // 1024} KB · {total_selected} entities.")

    blob = st.session_state.get("export_blob")
    name = st.session_state.get("export_blob_name")
    if blob and name:
        with action[2]:
            st.download_button(
                f"⬇️ Download {name} ({len(blob) // 1024} KB)",
                data=blob,
                file_name=name,
                mime="application/zip",
                width='stretch',
            )

    # ─── housekeeping ──────────────────────────────────────────────────────
    with st.expander("🧹 Prune stale favourites"):
        st.caption(
            "Stars whose target dataset / experiment / analysis no longer "
            "exists are kept in the table by default (no FK cascade). Run "
            "this to clean them out."
        )
        if st.button("Prune now"):
            from oasis_llm.db import connect as _connect
            try:
                rw = _connect()
                removed = fav.prune(rw)
                st.success(f"Removed {removed} stale star(s).")
            except Exception as e:
                st.error(f"Prune failed: {e}")
