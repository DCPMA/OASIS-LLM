"""Analysis page: three views over completed runs.

Three top-level tabs:

- **Curated** — Named, persisted analyses (CRUD). Each analysis pins a set of
  runs against one dataset and surfaces aggregations, cross-run correlation,
  vs-human-norms scatter, pair deltas + Bland–Altman, and ICC.
- **Ad-hoc** — Pick any runs (no save) and view scatter vs human + pairwise
  Pearson correlation, plus a long-format raw table.
- **Leaderboard** — Score every run against OASIS human norms, rank by
  Pearson r / Spearman ρ / MAE / RMSE, filter by min image count + dimension.

Replaces the previous separate Compare Runs and Leaderboard pages.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd
import streamlit as st

from oasis_llm import analyses as an
from oasis_llm import datasets as ds
from oasis_llm.dashboard_pages._ui import (
    connect_ro, connect_rw, db_locked_warning, kpi, page_header, star_button,
    starred_filter_toggle,
)


HUMAN_NORMS_CSV = "OASIS/OASIS.csv"


# ── Metric documentation ────────────────────────────────────────────────────
# Centralised inline "ℹ️ About this metric" copy, shown next to each chart so
# readers can see what the number means and where it comes from. Citations
# point to the canonical paper for each statistic.
_METRIC_DOCS: dict[str, tuple[str, str]] = {
    "pearson_r": (
        "Pearson r — linear correlation",
        "Measures the **linear** association between two ratings on a "
        "continuous scale (LLM mean vs. human mean per image). "
        "Range: −1 (perfect inverse) … 0 (none) … +1 (perfect). "
        "Sensitive to outliers; assumes roughly bivariate-normal data.\n\n"
        "*Reference:* Pearson, K. (1895). *Notes on regression and "
        "inheritance in the case of two parents.* Proc. R. Soc. London, "
        "**58**, 240–242."
    ),
    "spearman_rho": (
        "Spearman ρ — rank correlation",
        "Pearson r computed on **ranks** rather than raw values. Robust to "
        "outliers and monotonic-but-nonlinear relationships. Useful when "
        "the LLM's scale calibration differs from human raters but the "
        "*ordering* of images is preserved.\n\n"
        "*Reference:* Spearman, C. (1904). *The proof and measurement of "
        "association between two things.* Am. J. Psychol., **15**, 72–101."
    ),
    "mae_rmse": (
        "MAE / RMSE — absolute error",
        "**MAE** = mean(|LLM − human|), in rating-scale units. **RMSE** = "
        "√mean((LLM − human)²); penalises large deviations more heavily. "
        "Both measure **calibration** (does the LLM hit the right level?), "
        "complementing correlation which only measures the *shape* of the "
        "relationship. A run can have r ≈ 0.9 yet MAE = 1.5 if it's "
        "consistently shifted."
    ),
    "bland_altman": (
        "Bland–Altman — agreement",
        "Plots the **difference** between two raters (y) against their "
        "**mean** (x). The horizontal *bias* line shows systematic "
        "shift; the dashed *limits of agreement* (bias ± 1.96·SD) show "
        "the range within which 95% of differences fall. Designed for "
        "comparing measurement methods — here, treating LLMs as raters.\n\n"
        "*Reference:* Bland, J.M., & Altman, D.G. (1986). *Statistical "
        "methods for assessing agreement between two methods of clinical "
        "measurement.* The Lancet, **327**(8476), 307–310."
    ),
    "icc": (
        "ICC — inter-rater reliability",
        "Intraclass correlation. **ICC(2,1)** = absolute agreement, single "
        "rater (treats runs as a random sample of possible raters). "
        "**ICC(3,1)** = consistency only (runs are the specific raters of "
        "interest). Rule of thumb: <0.5 poor · 0.5–0.75 moderate · "
        "0.75–0.9 good · >0.9 excellent.\n\n"
        "*References:* Shrout, P.E. & Fleiss, J.L. (1979). *Intraclass "
        "correlations: uses in assessing rater reliability.* Psychol. Bull., "
        "**86**(2), 420–428. — McGraw, K.O. & Wong, S.P. (1996). *Forming "
        "inferences about some intraclass correlation coefficients.* "
        "Psychol. Methods, **1**(1), 30–46. — Koo, T.K. & Li, M.Y. (2016). "
        "*A guideline of selecting and reporting ICC for reliability "
        "research.* J. Chiropr. Med., **15**(2), 155–163."
    ),
    "correlation_heatmap": (
        "Cross-run correlation matrix",
        "Pairwise Pearson r between every pair of runs, computed on the "
        "per-image mean ratings they share. High off-diagonal values mean "
        "the LLMs agree with **each other** even if they disagree with "
        "humans — useful for spotting systematic LLM biases distinct from "
        "noise."
    ),
}


def _metric_doc(key: str, *, expanded: bool = False) -> None:
    """Render an inline ``ℹ️ About this metric`` expander."""
    title, body = _METRIC_DOCS[key]
    with st.expander(f"ℹ️ About this metric — {title}", expanded=expanded):
        st.markdown(body)


# ── Shared helpers (lifted from the retired compare_runs page) ─────────────
def _all_runs_meta(con) -> list[dict]:
    """Run metadata + completed-trial count, sorted newest first."""
    rows = con.execute(
        """
        SELECT r.run_id, r.status, r.config_json, r.created_at,
               count(t.image_id) FILTER (WHERE t.status='done') AS done
        FROM runs r LEFT JOIN trials t USING (run_id)
        GROUP BY r.run_id, r.status, r.config_json, r.created_at
        ORDER BY r.created_at DESC NULLS LAST
        """
    ).fetchall()
    out = []
    for run_id, status, cfg_json, created, done in rows:
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        out.append({
            "run_id": run_id,
            "status": status,
            "model": cfg.get("model"),
            "provider": cfg.get("provider"),
            "dataset_id": cfg.get("image_set"),
            "done": int(done or 0),
            "created_at": created,
        })
    return out


def _per_image_means(con, run_ids: list[str]) -> pd.DataFrame:
    """Long-format means per (run_id, image_id, dimension)."""
    if not run_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(run_ids))
    return con.execute(
        f"""
        SELECT run_id, image_id, dimension, avg(rating) AS mean_rating, count(*) AS n
        FROM trials
        WHERE run_id IN ({placeholders}) AND status='done'
        GROUP BY run_id, image_id, dimension
        """,
        run_ids,
    ).fetchdf()


def _human_norms() -> pd.DataFrame:
    return duckdb.connect(":memory:").execute(
        f"""
        SELECT "Theme" AS image_id,
               "Valence_mean" AS human_valence,
               "Arousal_mean" AS human_arousal
        FROM read_csv_auto('{HUMAN_NORMS_CSV}', header=true)
        """
    ).fetchdf()


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
        "Analysis",
        "Curated bundles · ad-hoc comparison · global leaderboard. "
        "All three views work on the same completed-runs corpus.",
        icon="🔬",
    )
    detail_id = st.query_params.get("analysis")
    if detail_id:
        _render_detail(detail_id)
        return

    tab_curated, tab_adhoc, tab_lb = st.tabs([
        "📋  Curated", "🔀  Ad-hoc", "🏆  Leaderboard",
    ])
    with tab_curated:
        _render_curated_list()
    with tab_adhoc:
        _render_adhoc_compare()
    with tab_lb:
        _render_leaderboard()


def _render_curated_list():
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
    sub_list, sub_new = st.tabs(["📚  All analyses", "✨  Create new"])

    with sub_list:
        if not rows:
            st.info(
                "No saved analyses yet. Use **Create new** to bundle runs "
                "into a named, persisted analysis."
            )
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

    with sub_new:
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


def _render_adhoc_compare():
    """Ad-hoc multi-run comparison (formerly the Compare Runs page).

    Pick any runs across providers/datasets, view per-image scatter vs
    human norms, pairwise inter-run correlation, and a long-format raw
    table. State is widget-local; nothing is persisted.
    """
    con = connect_ro()
    if con is None:
        db_locked_warning(); return

    runs = _all_runs_meta(con)
    if not runs:
        st.info("No runs yet."); return

    # Filters
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    models_avail = sorted({r["model"] for r in runs if r["model"]})
    providers_avail = sorted({r["provider"] for r in runs if r["provider"]})
    datasets_avail = sorted({r["dataset_id"] for r in runs if r["dataset_id"]})

    f_models = fc1.multiselect("Filter model", models_avail, default=[])
    f_providers = fc2.multiselect("Filter provider", providers_avail, default=[])
    f_datasets = fc3.multiselect("Filter dataset", datasets_avail, default=[])

    def _passes(r):
        if f_models and r["model"] not in f_models: return False
        if f_providers and r["provider"] not in f_providers: return False
        if f_datasets and r["dataset_id"] not in f_datasets: return False
        if r["done"] == 0: return False
        return True

    eligible = [r for r in runs if _passes(r)]
    if not eligible:
        st.info(
            "No runs match these filters (note: runs with 0 completed trials "
            "are hidden)."
        )
        return

    label_to_id = {
        f"{r['run_id']}  ·  {r['model']}  ·  {r['done']} done": r["run_id"]
        for r in eligible
    }
    selected_labels = st.multiselect(
        "Select runs to compare (2+ recommended)",
        list(label_to_id.keys()),
        default=list(label_to_id.keys())[:min(3, len(label_to_id))],
        key="adhoc_run_picker",
    )
    selected_ids = [label_to_id[l] for l in selected_labels]
    if not selected_ids:
        st.info("Select at least one run."); return

    with st.spinner("Aggregating per-image means…"):
        df = _per_image_means(con, selected_ids)
        norms = _human_norms()
    if df.empty:
        st.info("Selected runs have no completed trials."); return

    run_to_label = {r["run_id"]: r["model"] or r["run_id"] for r in eligible}
    df["run_label"] = df["run_id"].map(
        lambda rid: f"{run_to_label[rid]} · {rid[-6:]}"
    )

    merged = df.merge(norms, on="image_id", how="inner")
    merged["human_value"] = merged.apply(
        lambda r: r["human_valence"] if r["dimension"] == "valence"
        else r["human_arousal"] if r["dimension"] == "arousal"
        else None, axis=1,
    )
    merged = merged.dropna(subset=["human_value"])

    cs = st.columns(3)
    cs[0].markdown(kpi("Runs", len(selected_ids)), unsafe_allow_html=True)
    cs[1].markdown(
        kpi("Image overlap", merged["image_id"].nunique()),
        unsafe_allow_html=True,
    )
    cs[2].markdown(
        kpi("Trials aggregated", int(df["n"].sum())),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    sub_scatter, sub_corr, sub_table = st.tabs([
        "📈 Scatter vs human", "🔗 Pairwise correlations", "📋 Long table",
    ])

    with sub_scatter:
        for dim in sorted(merged["dimension"].unique()):
            sub = merged[merged["dimension"] == dim].copy()
            if sub.empty: continue
            st.markdown(f"#### {dim.title()}")
            chart_df = sub[
                ["image_id", "run_label", "human_value", "mean_rating"]
            ].rename(columns={"human_value": "human", "mean_rating": "llm"})
            st.scatter_chart(
                chart_df, x="human", y="llm", color="run_label", height=360,
            )
            summary = (
                sub.groupby("run_label")
                .apply(
                    lambda g: pd.Series({
                        "n": len(g),
                        "pearson_r": (
                            float(g["mean_rating"].corr(g["human_value"]))
                            if g["mean_rating"].std() > 0 else float("nan")
                        ),
                        "mae": float((g["mean_rating"] - g["human_value"]).abs().mean()),
                        "rmse": float(((g["mean_rating"] - g["human_value"]) ** 2).mean() ** 0.5),
                    }),
                    include_groups=False,
                )
                .reset_index()
            )
            st.dataframe(
                summary, width='stretch', hide_index=True,
                column_config={
                    "pearson_r": st.column_config.NumberColumn("r", format="%.3f"),
                    "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                    "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                },
            )
        _metric_doc("pearson_r")
        _metric_doc("mae_rmse")

    with sub_corr:
        if len(selected_ids) < 2:
            st.info("Pick 2+ runs for pairwise correlations.")
        else:
            for dim in sorted(df["dimension"].unique()):
                sub = df[df["dimension"] == dim]
                wide = sub.pivot(
                    index="image_id", columns="run_label", values="mean_rating"
                )
                if wide.shape[1] < 2 or wide.dropna().shape[0] < 3:
                    continue
                st.markdown(f"#### {dim.title()} — Pearson r")
                corr = wide.corr()
                st.dataframe(
                    corr.style.format("{:.3f}").background_gradient(
                        cmap="RdYlGn", vmin=-1, vmax=1, axis=None,
                    ),
                    width='stretch',
                )
                st.caption(
                    f"n images with overlap (all runs): "
                    f"{int(wide.dropna().shape[0])}"
                )
            _metric_doc("correlation_heatmap")

    with sub_table:
        st.dataframe(
            merged[[
                "run_id", "run_label", "image_id", "dimension",
                "mean_rating", "human_value", "n",
            ]],
            width='stretch', hide_index=True,
        )
        st.download_button(
            "Download compare.csv",
            data=merged.to_csv(index=False).encode(),
            file_name="compare.csv",
            mime="text/csv",
        )


def _render_leaderboard():
    """Global leaderboard (formerly the Leaderboard page).

    Scores every run against OASIS human norms via
    :func:`oasis_llm.analyses.leaderboard` and surfaces per-dimension
    rankings.
    """
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

    for dim, sub in df.groupby("dimension"):
        st.markdown(f"### {dim.title()}")
        view = sub[[
            "run_id", "model", "provider", "n_images",
            "pearson_r", "spearman_rho", "mae", "rmse",
            "mean_pred", "mean_human", "temperature",
        ]].reset_index(drop=True)
        st.dataframe(
            view, width='stretch', hide_index=True,
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

    _metric_doc("pearson_r")
    _metric_doc("spearman_rho")
    _metric_doc("mae_rmse")

    st.download_button(
        "Download leaderboard.csv",
        data=df.to_csv(index=False).encode(),
        file_name="leaderboard.csv",
        mime="text/csv",
    )


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
            _metric_doc("correlation_heatmap")

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
            _metric_doc("pearson_r")
            _metric_doc("mae_rmse")

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
            _metric_doc("bland_altman")

    with tab_icc:
        icc = an.icc_across_runs(con, analysis_id)
        if icc.empty:
            st.caption("No data.")
        else:
            st.dataframe(
                icc, width='stretch', hide_index=True,
                column_config={
                    "icc2_1": st.column_config.NumberColumn("ICC(2,1) absolute", format="%.3f"),
                    "icc3_1": st.column_config.NumberColumn("ICC(3,1) consistency", format="%.3f"),
                },
            )
            _metric_doc("icc", expanded=True)

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
