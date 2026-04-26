"""Analysis page: unified vs-Human analytics over completed runs.

Replaces both the old multi-tab Analysis page and the standalone
"🆚 LLM vs Human" Explorer page. Two operating modes share the same
analytics body:

- **Ad-hoc** — pick runs via sidebar filters (image_set + models).
- **Curated** — select a saved Analysis bundle; its run_ids are pinned
  and remaining filters (categories, dimensions, image search, scope)
  still apply.

Below the analytics tabs we keep the curated-bundle CRUD (browse / create
/ open), a per-image means view, and CSV / zip export.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd
import streamlit as st

from oasis_llm import analyses as an
from oasis_llm import datasets as ds
from oasis_llm.dashboard_pages import _vs_human_analytics as _vsh
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




# ── Public render ──────────────────────────────────────────────────────────
def render() -> None:
    page_header(
        "Analysis",
        "LLM vs Human analytics — ad-hoc filter or curated bundle. "
        "Sidebar filters drive every tab.",
        icon="🔬",
    )

    # Detail view (?analysis=<id>) is deprecated; redirect to Curated mode
    detail_id = st.query_params.get("analysis")
    if detail_id:
        st.session_state["analysis_curated_id"] = detail_id
        st.session_state["analysis_mode"] = "Curated"
        del st.query_params["analysis"]

    # ── Mode switch (top of page, sticky-feeling) ──
    mode_default = st.session_state.get("analysis_mode", "Ad-hoc")
    mode = st.radio(
        "Mode",
        ["Ad-hoc", "Curated"],
        index=["Ad-hoc", "Curated"].index(mode_default) if mode_default in ("Ad-hoc", "Curated") else 0,
        horizontal=True,
        help=(
            "**Ad-hoc** — sidebar filters pick runs directly. "
            "**Curated** — select a saved Analysis bundle to pin its run_ids."
        ),
        key="analysis_mode",
    )

    pinned_run_ids: list[str] | None = None
    pinned_analysis = None

    if mode == "Curated":
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        pinned_analysis = _curated_picker(con)
        if pinned_analysis is None:
            return
        pinned_run_ids = pinned_analysis.run_ids
        if not pinned_run_ids:
            st.info("This analysis has no runs yet — add some below.")
            _render_run_management(con, pinned_analysis)
            return

    # ── Analytics body (shared) ──
    _vsh.render_analytics(pinned_run_ids=pinned_run_ids)

    # ── Extra tabs: Per-image means + Export (+ run management when curated) ──
    st.markdown("---")
    if pinned_analysis is not None:
        extra_tabs = st.tabs([
            "🔢 Per-image means", "📤 Export", "🛠 Manage runs",
        ])
    else:
        extra_tabs = st.tabs(["🔢 Per-image means", "📤 Export"])

    with extra_tabs[0]:
        _render_per_image_means(pinned_run_ids=pinned_run_ids)

    with extra_tabs[1]:
        _render_export(pinned_analysis=pinned_analysis,
                       pinned_run_ids=pinned_run_ids)

    if pinned_analysis is not None:
        with extra_tabs[2]:
            con2 = connect_rw()
            if con2 is None:
                db_locked_warning()
            else:
                _render_run_management(con2, pinned_analysis)


# ── Curated picker ─────────────────────────────────────────────────────────
def _curated_picker(con):
    rows = an.list_all(con)
    if not rows:
        st.info(
            "No saved analyses yet. Switch to **Ad-hoc** mode to explore now, "
            "or create one below."
        )
        with st.expander("✨ Create new analysis", expanded=True):
            _create_form(con)
        return None

    starred_only = starred_filter_toggle("analysis", key="analysis_curated_starred")
    if starred_only:
        from oasis_llm import favorites as _fav
        star_set = _fav.starred_set(con, "analysis")
        rows = [a for a in rows if a.analysis_id in star_set]
        if not rows:
            st.caption("No starred analyses — click ☆ in the picker below.")
            return None

    label_for = lambda a: f"{a.name}  ·  {a.dataset_id}  ·  {len(a.run_ids)} runs"
    ids = [a.analysis_id for a in rows]
    default_id = st.session_state.get("analysis_curated_id")
    default_idx = ids.index(default_id) if default_id in ids else 0
    chosen_id = st.selectbox(
        "Curated analysis",
        ids,
        index=default_idx,
        format_func=lambda aid: label_for(next(a for a in rows if a.analysis_id == aid)),
        key="analysis_curated_id",
    )
    chosen = next(a for a in rows if a.analysis_id == chosen_id)
    cols = st.columns([0.4, 5, 1])
    with cols[0]:
        star_button("analysis", chosen.analysis_id, key_suffix="picker")
    cols[1].caption(
        f"`{chosen.analysis_id}` · dataset **{chosen.dataset_id}** · "
        f"{len(chosen.run_ids)} runs"
    )
    if cols[2].button("✨ New", key="curated_new_btn"):
        st.session_state["_show_create_form"] = not st.session_state.get("_show_create_form", False)

    if st.session_state.get("_show_create_form"):
        with st.expander("Create new analysis", expanded=True):
            _create_form(con)

    if chosen.description:
        st.write(chosen.description)

    return chosen


def _create_form(con):
    datasets = list(ds.list_all(con))
    if not datasets:
        st.warning("Create a dataset first.")
        return
    with st.form("new_analysis"):
        name = st.text_input("Analysis name", value="my-analysis")
        dataset_choice = st.selectbox(
            "Dataset (analyses are scoped to ONE dataset)",
            [d.dataset_id for d in datasets],
        )
        description = st.text_area("Description (optional)")
        if st.form_submit_button("Create analysis", type="primary"):
            try:
                aid = an.create(
                    con, name, dataset_choice,
                    description=description or None,
                )
            except Exception as e:
                st.error(str(e)); return
            st.success(f"Created analysis `{aid}`.")
            st.session_state["analysis_curated_id"] = aid
            st.session_state["_show_create_form"] = False
            st.rerun()


def _render_run_management(con, a) -> None:
    st.markdown("#### Runs in this analysis")
    if not a.run_ids:
        st.caption("No runs added yet.")
    else:
        for rid in a.run_ids:
            rcols = st.columns([6, 1])
            rcols[0].markdown(f"• **{rid}**")
            if rcols[1].button("Remove", key=f"rm_{a.analysis_id}_{rid}"):
                an.remove_run(con, a.analysis_id, rid)
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
                "Pick a run", eligible,
                format_func=label_for, key=f"add_{a.analysis_id}",
            )
            if st.button("Add to analysis", key=f"addbtn_{a.analysis_id}"):
                try:
                    an.add_run(con, a.analysis_id, choice["run_id"])
                except Exception as e:
                    st.error(str(e)); return
                st.rerun()


# ── Per-image means tab ────────────────────────────────────────────────────
def _render_per_image_means(*, pinned_run_ids: list[str] | None) -> None:
    con = connect_ro()
    if con is None:
        db_locked_warning(); return

    if pinned_run_ids:
        run_ids = pinned_run_ids
    else:
        # Use the last sidebar selection (best effort) — fall back to "all done runs"
        sel_models = st.session_state.get("vsh_models_adhoc", [])
        sel_set = st.session_state.get("vsh_imgset")
        runs = _all_runs_meta(con)
        run_ids = [
            r["run_id"] for r in runs
            if r["done"] > 0
            and (not sel_set or r.get("dataset_id") == sel_set)
            and (not sel_models or r.get("model") in sel_models)
        ]

    if not run_ids:
        st.caption("No runs in scope. Select runs in the sidebar (Ad-hoc) or pick an analysis (Curated).")
        return

    df = _per_image_means(con, run_ids)
    if df.empty:
        st.caption("Selected runs have no completed trials.")
        return
    wide = df.pivot_table(
        index=["image_id", "dimension"], columns="run_id",
        values="mean_rating",
    ).round(2).reset_index()
    st.dataframe(wide, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download per_image_means.csv",
        data=wide.to_csv(index=False).encode(),
        file_name="per_image_means.csv",
        mime="text/csv",
        key="dl_pim",
    )


# ── Export tab ─────────────────────────────────────────────────────────────
def _render_export(*, pinned_analysis, pinned_run_ids) -> None:
    if pinned_analysis is None:
        st.caption(
            "Curated bundles can be exported as zip. Switch to **Curated** "
            "mode to bind a set of runs to a dataset, then export here."
        )
        return
    aid = pinned_analysis.analysis_id
    con = connect_rw()
    if con is None:
        db_locked_warning(); return
    if st.button("Build zip", key=f"an_export_btn_{aid}", type="primary"):
        from oasis_llm.bundles import export_analysis
        blob = export_analysis(con, aid)
        st.session_state[f"an_export_blob_{aid}"] = blob
    blob = st.session_state.get(f"an_export_blob_{aid}")
    if blob:
        st.download_button(
            f"⬇️ Download analysis_{aid}.zip ({len(blob) // 1024} KB)",
            data=blob,
            file_name=f"analysis_{aid}.zip",
            mime="application/zip",
            key=f"an_export_dl_{aid}",
        )
