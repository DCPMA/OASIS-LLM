"""Home page: KPIs, recent activity, quick actions."""
from __future__ import annotations

import streamlit as st

from oasis_llm.dashboard_pages._ui import (
    connect_ro, kpi, page_header, status_pill,
)


def render():
    page_header(
        "OASIS-LLM",
        "Rating OASIS images with LLMs — a research control plane.",
        icon="🧠",
    )
    con = connect_ro()
    if con is None:
        st.warning("No database yet. Generate a dataset to get started.")
        return

    n_datasets = con.execute("SELECT count(*) FROM datasets").fetchone()[0]
    n_approved = con.execute(
        "SELECT count(*) FROM datasets WHERE status='approved'"
    ).fetchone()[0]
    n_experiments = con.execute("SELECT count(*) FROM experiments").fetchone()[0]
    n_runs = con.execute("SELECT count(*) FROM runs").fetchone()[0]
    n_done = con.execute("SELECT count(*) FROM trials WHERE status='done'").fetchone()[0]
    n_failed = con.execute("SELECT count(*) FROM trials WHERE status='failed'").fetchone()[0]
    n_pending = con.execute(
        "SELECT count(*) FROM trials WHERE status IN ('pending','running')"
    ).fetchone()[0]
    total_cost = con.execute(
        "SELECT COALESCE(round(sum(cost_usd), 4), 0) FROM trials"
    ).fetchone()[0]

    # KPI row
    cols = st.columns(4)
    cols[0].markdown(kpi("Datasets", n_datasets, f"{n_approved} approved"), unsafe_allow_html=True)
    cols[1].markdown(kpi("Experiments", n_experiments), unsafe_allow_html=True)
    cols[2].markdown(
        kpi("Trials done", f"{n_done:,}", f"{n_pending} pending · {n_failed} failed"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(kpi("Total spend", f"${total_cost:.4f}"), unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Recent experiments")
        recent_exp = con.execute(
            """
            SELECT e.experiment_id, e.name, e.status, e.dataset_id,
                   count(ec.config_name) AS n_configs,
                   e.created_at
            FROM experiments e
            LEFT JOIN experiment_configs ec USING (experiment_id)
            GROUP BY e.experiment_id, e.name, e.status, e.dataset_id, e.created_at
            ORDER BY e.created_at DESC NULLS LAST
            LIMIT 5
            """
        ).fetchall()
        if not recent_exp:
            st.caption("No experiments yet. Create one in the Experiments tab.")
        else:
            for r in recent_exp:
                exp_id, name, status, dataset_id, n_cfg, created = r
                st.markdown(
                    f"<div style='padding: 0.5rem 0; border-bottom: 1px solid #1f1f2e;'>"
                    f"<b>{name}</b> {status_pill(status)}<br>"
                    f"<span style='color:#8a8aa0; font-size:0.85rem;'>"
                    f"📁 {dataset_id} · {n_cfg} configs · "
                    f"{str(created)[:19] if created else '-'}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("### Approved datasets")
        approved = con.execute(
            """
            SELECT d.dataset_id,
                   count(di.image_id) AS total,
                   sum(CASE WHEN NOT di.excluded THEN 1 ELSE 0 END) AS active,
                   d.source
            FROM datasets d LEFT JOIN dataset_images di USING (dataset_id)
            WHERE d.status='approved'
            GROUP BY d.dataset_id, d.source
            ORDER BY d.dataset_id
            """
        ).fetchall()
        if not approved:
            st.caption("No approved datasets yet.")
        else:
            for r in approved:
                ds_id, total, active, source = r
                st.markdown(
                    f"<div style='padding: 0.5rem 0; border-bottom: 1px solid #1f1f2e;'>"
                    f"<b>{ds_id}</b> {status_pill(source)}<br>"
                    f"<span style='color:#8a8aa0; font-size:0.85rem;'>"
                    f"{active}/{total} images"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
