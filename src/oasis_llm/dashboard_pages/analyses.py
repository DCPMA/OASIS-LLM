"""Analyses page: aggregate ratings across runs that share a dataset."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from oasis_llm import analyses as an
from oasis_llm import datasets as ds
from oasis_llm.dashboard_pages._ui import (
    connect_ro, connect_rw, db_locked_warning, kpi, page_header, star_button,
    starred_filter_toggle,
)


def _render_bland_altman(con, analysis_id: str, run_a: str, run_b: str) -> None:
    """Render Bland–Altman scatter for a chosen run pair, faceted by dimension."""
    import altair as alt

    df = an.per_image_aggregate(con, analysis_id)
    if df.empty:
        st.caption("No data.")
        return
    df = df[df["run_id"].isin([run_a, run_b])]
    wide = (
        df.pivot_table(
            index=["image_id", "dimension"], columns="run_id",
            values="mean_rating", aggfunc="first",
        )
        .reset_index()
    )
    if run_a not in wide.columns or run_b not in wide.columns:
        st.caption("No overlapping images for this pair.")
        return
    wide = wide.dropna(subset=[run_a, run_b])
    wide["mean"] = (wide[run_a] + wide[run_b]) / 2.0
    wide["diff"] = wide[run_a] - wide[run_b]
    if wide.empty:
        st.caption("No overlapping images for this pair.")
        return

    # Per-dimension chart with bias + LoA reference lines.
    charts = []
    for dim, sub in wide.groupby("dimension"):
        bias = float(sub["diff"].mean())
        sd = float(sub["diff"].std(ddof=1)) if len(sub) > 1 else 0.0
        loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd
        scatter = (
            alt.Chart(sub)
            .mark_circle(size=55, opacity=0.65, color="#4C78A8")
            .encode(
                x=alt.X("mean:Q", title=f"({run_a} + {run_b}) / 2"),
                y=alt.Y("diff:Q", title=f"{run_a} − {run_b}"),
                tooltip=["image_id", "mean", "diff"],
            )
        )
        rules = pd.DataFrame({
            "y": [bias, loa_hi, loa_lo],
            "lab": [f"bias {bias:.2f}", f"+1.96·SD {loa_hi:.2f}", f"−1.96·SD {loa_lo:.2f}"],
            "kind": ["bias", "loa", "loa"],
        })
        rule_layer = (
            alt.Chart(rules)
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y="y:Q",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["bias", "loa"], range=["#333", "#c44"]),
                    legend=None,
                ),
            )
        )
        chart = (scatter + rule_layer).properties(
            title=f"{dim} (bias {bias:+.2f}, LoA [{loa_lo:+.2f}, {loa_hi:+.2f}], n={len(sub)})",
            height=300,
        )
        charts.append(chart)
    if charts:
        st.altair_chart(alt.vconcat(*charts), use_container_width=True)


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
            starred_only = starred_filter_toggle("analysis")
            if starred_only:
                from oasis_llm import favorites as _fav
                star_set = _fav.starred_set(con, "analysis")
                rows = [a for a in rows if a.analysis_id in star_set]
                if not rows:
                    st.caption("No starred analyses — click ☆ on a row to add one.")
            for a in rows:
                _row_card(a)

    with tab_new:
        _create_form(con)


def _row_card(a):
    cols = st.columns([0.4, 4, 2, 2, 2])
    with cols[0]:
        star_button("analysis", a.analysis_id, key_suffix="list")
    cols[1].markdown(
        f"<div style='font-weight:600; font-size:1.05rem'>{a.name}</div>"
        f"<div style='color:#8a8aa0; font-size:0.78rem'>{a.analysis_id}</div>",
        unsafe_allow_html=True,
    )
    cols[2].markdown(f"📁 **{a.dataset_id}**")
    cols[3].markdown(f"🏃 {len(a.run_ids)} runs")
    if cols[4].button("Open ›", key=f"openana_{a.analysis_id}", width='stretch'):
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

    top = st.columns([5, 1, 1, 1])
    top[0].markdown(f"# 🔬 {a.name}")
    top[0].caption(
        f"`{a.analysis_id}` · dataset: **{a.dataset_id}** · "
        f"{len(a.run_ids)} runs · created {str(a.created_at)[:19] if a.created_at else '—'}"
    )
    if a.description:
        top[0].write(a.description)
    with top[1]:
        star_button("analysis", analysis_id, key_suffix="detail")
    with top[2]:
        with st.popover("📤 Export", use_container_width=True):
            if st.button(
                "Build zip", key=f"an_export_btn_{analysis_id}",
                width='stretch', type="primary",
            ):
                from oasis_llm.bundles import export_analysis
                blob = export_analysis(con, analysis_id)
                st.session_state[f"an_export_blob_{analysis_id}"] = blob
            blob = st.session_state.get(f"an_export_blob_{analysis_id}")
            if blob:
                st.download_button(
                    f"⬇️ Download ({len(blob) // 1024} KB)",
                    data=blob,
                    file_name=f"analysis_{analysis_id}.zip",
                    mime="application/zip",
                    key=f"an_export_dl_{analysis_id}",
                    width='stretch',
                )
    if top[3].button("← All", width='stretch'):
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
                # Heatmap via pandas Styler.background_gradient (RdYlGn diverging,
                # centred at 0).
                st.dataframe(
                    mat.round(3).style.background_gradient(
                        cmap="RdYlGn", vmin=-1, vmax=1, axis=None,
                    ),
                    width='stretch',
                )

    with tab_vs_human:
        merged = an.vs_human_norms(con, analysis_id)
        if merged.empty:
            st.caption("No data.")
        else:
            import altair as alt
            for dim, sub in merged.groupby("dimension"):
                st.markdown(f"#### {dim}")

                # Scatter: human (x) vs LLM (y), one colour per run, with
                # identity line y=x for "perfect calibration" reference.
                domain_cols = ["human_value", "mean_rating", "image_id", "run_id"]
                plot_df = sub.dropna(subset=["human_value", "mean_rating"])[domain_cols]
                if not plot_df.empty:
                    lo = float(min(plot_df["human_value"].min(), plot_df["mean_rating"].min())) - 0.2
                    hi = float(max(plot_df["human_value"].max(), plot_df["mean_rating"].max())) + 0.2
                    scatter = (
                        alt.Chart(plot_df)
                        .mark_circle(size=60, opacity=0.65)
                        .encode(
                            x=alt.X("human_value:Q", title="Human mean (1–7)",
                                    scale=alt.Scale(domain=[lo, hi])),
                            y=alt.Y("mean_rating:Q", title="LLM mean (1–7)",
                                    scale=alt.Scale(domain=[lo, hi])),
                            color=alt.Color("run_id:N", title="Run"),
                            tooltip=["run_id", "image_id", "human_value", "mean_rating"],
                        )
                    )
                    identity = (
                        alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]}))
                        .mark_line(strokeDash=[4, 4], color="grey")
                        .encode(x="x:Q", y="y:Q")
                    )
                    st.altair_chart(
                        (identity + scatter).properties(height=380),
                        use_container_width=True,
                    )

                # Numerical summary
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

            # Bland–Altman per pair × dim. Lets the user pick one pair to
            # inspect at a time (full grid would be O(n²) charts).
            st.markdown("---")
            st.markdown("#### 📐 Bland–Altman")
            st.caption(
                "X = mean of two runs' per-image ratings; Y = run A − run B. "
                "Solid line = bias (mean diff); dashed lines = limits of "
                "agreement (bias ± 1.96·SD)."
            )
            run_ids = sorted({r for col in ("run_a", "run_b") for r in pairs[col]})
            ba_cols = st.columns(2)
            run_a = ba_cols[0].selectbox("Run A", run_ids, key="ba_run_a")
            run_b_options = [r for r in run_ids if r != run_a]
            if run_b_options:
                run_b = ba_cols[1].selectbox("Run B", run_b_options, key="ba_run_b")
                _render_bland_altman(con, analysis_id, run_a, run_b)
            else:
                st.caption("Need ≥2 distinct runs.")

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
