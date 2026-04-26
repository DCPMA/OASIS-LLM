"""Analyses page: aggregate ratings across runs that share a dataset."""
from __future__ import annotations

import streamlit as st

from oasis_llm import analyses as an
from oasis_llm import datasets as ds
from oasis_llm.dashboard_pages._ui import (
    connect_rw, db_locked_warning, kpi, page_header,
)


def render():
    page_header(
        "Analyses",
        "Aggregate and compare runs that rated the same dataset.",
        icon="🔬",
    )
    detail_id = st.query_params.get("analysis")
    if detail_id:
        _render_detail(detail_id)
        return
    _render_list()


def _render_list():
    con = connect_rw()
    if con is None:
        db_locked_warning(); return
    rows = an.list_all(con)

    cols = st.columns(3)
    cols[0].markdown(kpi("Analyses", len(rows)), unsafe_allow_html=True)
    cols[1].markdown(
        kpi("Total runs aggregated", sum(len(a.run_ids) for a in rows)),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        kpi("Datasets covered", len({a.dataset_id for a in rows})),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    tab_list, tab_new = st.tabs(["📋  All analyses", "✨  Create new"])

    with tab_list:
        if not rows:
            st.info("No analyses yet.")
        else:
            for a in rows:
                _row_card(a)

    with tab_new:
        _create_form(con)


def _row_card(a):
    cols = st.columns([4, 2, 2, 2])
    cols[0].markdown(
        f"<div style='font-weight:600; font-size:1.05rem'>{a.name}</div>"
        f"<div style='color:#8a8aa0; font-size:0.78rem'>{a.analysis_id}</div>",
        unsafe_allow_html=True,
    )
    cols[1].markdown(f"📁 **{a.dataset_id}**")
    cols[2].markdown(f"🏃 {len(a.run_ids)} runs")
    if cols[3].button("Open ›", key=f"openana_{a.analysis_id}", width='stretch'):
        st.query_params["analysis"] = a.analysis_id
        st.rerun()
    st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)


def _create_form(con):
    datasets = [d for d in ds.list_all(con)]
    if not datasets:
        st.warning("Create a dataset first."); return
    with st.form("new_analysis"):
        name = st.text_input("Analysis name", value="my-analysis")
        dataset_choice = st.selectbox(
            "Dataset (analyses are scoped to ONE dataset)",
            [d.dataset_id for d in datasets],
        )
        description = st.text_area("Description (optional)")
        if st.form_submit_button("Create analysis", type="primary"):
            try:
                aid = an.create(con, name, dataset_choice, description=description or None)
            except Exception as e:
                st.error(str(e)); return
            st.success(f"Created analysis `{aid}`.")
            st.query_params["analysis"] = aid
            st.rerun()


def _render_detail(analysis_id: str):
    con = connect_rw()
    if con is None:
        db_locked_warning(); return
    a = an.get(con, analysis_id)
    if a is None:
        st.error(f"Unknown analysis `{analysis_id}`.")
        if st.button("← Back"):
            del st.query_params["analysis"]; st.rerun()
        return

    top = st.columns([6, 2])
    top[0].markdown(f"# 🔬 {a.name}")
    top[0].caption(
        f"`{a.analysis_id}` · dataset: **{a.dataset_id}** · "
        f"{len(a.run_ids)} runs · created {str(a.created_at)[:19] if a.created_at else '—'}"
    )
    if a.description:
        top[0].write(a.description)
    if top[1].button("← All analyses", width='stretch'):
        del st.query_params["analysis"]; st.rerun()

    cols = st.columns(4)
    cols[0].markdown(kpi("Dataset", a.dataset_id), unsafe_allow_html=True)
    cols[1].markdown(kpi("Runs", len(a.run_ids)), unsafe_allow_html=True)
    if a.run_ids:
        df = an.per_image_aggregate(con, analysis_id)
        cols[2].markdown(
            kpi("Images covered", df["image_id"].nunique() if not df.empty else 0),
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            kpi("Trials", int(df["n"].sum()) if not df.empty else 0),
            unsafe_allow_html=True,
        )

    # ── Run management ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Runs in this analysis")
    if not a.run_ids:
        st.caption("No runs added yet.")
    else:
        for rid in a.run_ids:
            rcols = st.columns([6, 1])
            rcols[0].markdown(f"• **{rid}**")
            if rcols[1].button("Remove", key=f"rm_{analysis_id}_{rid}"):
                an.remove_run(con, analysis_id, rid)
                st.rerun()

    with st.expander("➕ Add a run"):
        eligible = an.eligible_runs(con, a.dataset_id)
        already = set(a.run_ids)
        eligible = [r for r in eligible if r["run_id"] not in already]
        if not eligible:
            st.caption(f"No additional runs match dataset `{a.dataset_id}`.")
        else:
            label_for = lambda r: f"{r['run_id']} ({r['status']}, {r['model']})"
            choice = st.selectbox(
                "Pick a run",
                eligible, format_func=label_for, key=f"add_{analysis_id}",
            )
            if st.button("Add to analysis", key=f"addbtn_{analysis_id}"):
                try:
                    an.add_run(con, analysis_id, choice["run_id"])
                except Exception as e:
                    st.error(str(e)); return
                st.rerun()

    if not a.run_ids:
        return

    # ── Aggregations ───────────────────────────────────────────────────────
    st.markdown("---")
    tab_per_img, tab_corr, tab_vs_human, tab_pairs, tab_icc, tab_export = st.tabs([
        "📊  Per-image means", "🔗  Cross-run correlation",
        "👥  vs Human norms", "🥊  Pair deltas", "🎯  ICC", "📤  Export",
    ])
    df = an.per_image_aggregate(con, analysis_id)

    with tab_per_img:
        if df.empty:
            st.caption("No completed trials yet.")
        else:
            wide = df.pivot_table(
                index=["image_id", "dimension"], columns="run_id",
                values="mean_rating",
            ).round(2).reset_index()
            st.dataframe(wide, width='stretch', hide_index=True)

    with tab_corr:
        corrs = an.cross_run_correlations(con, analysis_id)
        if not corrs:
            st.caption("Need at least 2 runs with shared images.")
        else:
            for dim, mat in corrs.items():
                st.markdown(f"#### {dim}")
                st.dataframe(mat.round(3), width='stretch')

    with tab_vs_human:
        merged = an.vs_human_norms(con, analysis_id)
        if merged.empty:
            st.caption("No data.")
        else:
            for dim, sub in merged.groupby("dimension"):
                st.markdown(f"#### {dim}")
                summary = (
                    sub.groupby("run_id")
                    .apply(lambda g: g[["mean_rating", "human_value"]].corr().iloc[0, 1])
                    .round(3)
                    .rename("ρ vs human")
                    .reset_index()
                )
                summary["mean Δ"] = (
                    sub.groupby("run_id")["delta"].mean().round(3).values
                )
                st.dataframe(summary, width='stretch', hide_index=True)

    with tab_pairs:
        pairs = an.model_pair_deltas(con, analysis_id)
        if pairs.empty:
            st.caption("Need ≥2 runs with overlapping images.")
        else:
            for dim, sub in pairs.groupby("dimension"):
                st.markdown(f"#### {dim}")
                st.dataframe(
                    sub.drop(columns=["dimension"]).reset_index(drop=True),
                    width='stretch', hide_index=True,
                    column_config={
                        "mean_delta":     st.column_config.NumberColumn("mean Δ", format="%.3f"),
                        "abs_mean_delta": st.column_config.NumberColumn("|mean Δ|", format="%.3f"),
                        "max_abs_delta":  st.column_config.NumberColumn("max |Δ|", format="%.3f"),
                        "sd_delta":       st.column_config.NumberColumn("sd Δ", format="%.3f"),
                    },
                )

    with tab_icc:
        icc = an.icc_across_runs(con, analysis_id)
        if icc.empty:
            st.caption("No data.")
        else:
            st.markdown(
                "**Inter-rater reliability across runs.** "
                "ICC(2,1) = absolute agreement (single rater, runs treated as a "
                "random sample). ICC(3,1) = consistency (the specific runs are of "
                "interest). Rule of thumb: <0.5 poor · 0.5–0.75 moderate · "
                "0.75–0.9 good · >0.9 excellent (Koo & Li 2016)."
            )
            st.dataframe(
                icc, width='stretch', hide_index=True,
                column_config={
                    "icc2_1": st.column_config.NumberColumn("ICC(2,1) absolute", format="%.3f"),
                    "icc3_1": st.column_config.NumberColumn("ICC(3,1) consistency", format="%.3f"),
                },
            )

    with tab_export:
        if df.empty:
            st.caption("Nothing to export.")
        else:
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Download per_image_means.csv",
                data=buf.getvalue().encode(),
                file_name=f"{analysis_id}_per_image_means.csv",
                mime="text/csv", width='stretch',
            )
            merged = an.vs_human_norms(con, analysis_id)
            if not merged.empty:
                buf2 = io.StringIO()
                merged.to_csv(buf2, index=False)
                st.download_button(
                    "⬇️ Download with_human_norms.csv",
                    data=buf2.getvalue().encode(),
                    file_name=f"{analysis_id}_with_human_norms.csv",
                    mime="text/csv", width='stretch',
                )

    # ── Danger zone ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚠️ Danger zone"):
        if st.button("🗑️ Delete analysis", type="secondary"):
            an.delete(con, analysis_id)
            del st.query_params["analysis"]
            st.rerun()
