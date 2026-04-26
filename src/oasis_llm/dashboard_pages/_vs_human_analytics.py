"""Reusable analytics body comparing LLM ratings to OASIS human norms.

Used by the Analysis page (``analyses.py``) in two modes:

- **Ad-hoc** (``run_ids=None``): the sidebar exposes image-set + model
  pickers, letting the user scope analytics by directly choosing runs.
- **Curated** (``run_ids=[...]``): the caller pins a saved Analysis bundle
  and only the secondary filters (dimensions, categories, image search,
  scope, n_boot) remain in the sidebar.

The page used to live as a standalone "🆚 LLM vs Human" entry; it is now
imported by the Analysis page and that nav entry has been removed.
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from oasis_llm import analyses as an
from oasis_llm.images import image_path
from oasis_llm.dashboard_pages._ui import (
    apply_blinding,
    blind_models_toggle,
    connect_ro,
    reveal_blinding_expander,
)

HUMAN_NORMS_CSV = "OASIS/OASIS.csv"
CATEGORY_PALETTE = {
    "Animal": "#D62728",
    "Scene":  "#2CA02C",
    "Person": "#1F77B4",
    "Object": "#E6A700",
}


def _csv_download(df: pd.DataFrame, *, filename: str, key: str) -> None:
    """Render a 'Download CSV' button below a tab table."""
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=False,
    )


# ─── data helpers ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def _load_run_meta() -> pd.DataFrame:
    con = connect_ro()
    rows = con.execute(
        "SELECT run_id, status, config_json, created_at FROM runs ORDER BY created_at DESC NULLS LAST"
    ).fetchall()
    out = []
    for rid, status, cfg_json, created_at in rows:
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        out.append({
            "run_id": rid,
            "status": status,
            "model": cfg.get("model"),
            "provider": cfg.get("provider"),
            "image_set": cfg.get("image_set"),
            "samples_per_image": cfg.get("samples_per_image"),
            "created_at": created_at,
        })
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False, ttl=30)
def _load_trials(run_ids: tuple[str, ...]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()
    con = connect_ro()
    placeholders = ",".join("?" * len(run_ids))
    df = con.execute(
        f"""
        SELECT run_id, image_id, dimension, sample_idx, rating
        FROM trials
        WHERE run_id IN ({placeholders})
          AND status='done' AND rating IS NOT NULL
        """,
        list(run_ids),
    ).fetchdf()
    return df


@st.cache_data(show_spinner=False, ttl=300)
def _load_norms() -> pd.DataFrame:
    return an._load_norms_with_category(HUMAN_NORMS_CSV)


# ─── main render ───────────────────────────────────────────────────────────
def render_analytics(
    *,
    pinned_run_ids: list[str] | None = None,
    sidebar_prefix: str = "vsh",
) -> None:
    """Render the LLM-vs-Human analytics body.

    Parameters
    ----------
    pinned_run_ids
        If provided, ad-hoc model/image-set pickers are skipped and these
        runs (filtered to ``status=='done'``) drive the analytics. Used by
        the Analysis page when the user has selected a curated bundle.
    sidebar_prefix
        Used to namespace widget keys so the same helpers can be hosted
        twice on a page without colliding.
    """
    runs_meta = _load_run_meta()
    if runs_meta.empty:
        st.info("No runs found in the database yet.")
        return

    runs_done = runs_meta[runs_meta["status"] == "done"].copy()
    if runs_done.empty:
        st.info("No completed runs yet.")
        return

    if pinned_run_ids:
        runs_in_scope = runs_done[runs_done["run_id"].isin(pinned_run_ids)].copy()
        if runs_in_scope.empty:
            st.info("None of the analysis's runs are completed yet.")
            return
    else:
        runs_in_scope = runs_done

    # ── Sidebar filters ──
    norms = _load_norms()
    with st.sidebar:
        st.markdown("### Analysis filters")

        if pinned_run_ids:
            sel_set = None
            st.caption(f"Pinned: **{len(runs_in_scope)} run(s)** from the selected analysis.")
            models_avail = sorted(runs_in_scope["model"].dropna().unique().tolist())
            sel_models = st.multiselect(
                "Models", models_avail,
                default=models_avail,
                key=f"{sidebar_prefix}_models_curated",
                help="Subset the curated analysis's runs.",
            )
            runs_in_set = runs_in_scope
        else:
            image_sets = sorted(runs_done["image_set"].dropna().unique().tolist())
            sel_set = st.selectbox(
                "Image set",
                image_sets,
                index=0 if image_sets else None,
                key=f"{sidebar_prefix}_imgset",
            )
            runs_in_set = runs_done[runs_done["image_set"] == sel_set]
            models_avail = sorted(runs_in_set["model"].dropna().unique().tolist())
            sel_models = st.multiselect(
                "Models", models_avail,
                default=models_avail,
                key=f"{sidebar_prefix}_models_adhoc",
                help="Select one or more models.",
            )

        sel_dims = st.multiselect(
            "Dimensions", ["valence", "arousal"],
            default=["valence", "arousal"],
            key=f"{sidebar_prefix}_dims",
        )

        cat_avail = sorted(norms["category"].dropna().unique().tolist())
        sel_cats = st.multiselect(
            "Categories", cat_avail, default=cat_avail,
            key=f"{sidebar_prefix}_cats",
        )

        scope = st.radio(
            "Aggregation scope",
            ["Pooled all-LLMs", "By model", "By category", "Model × Category"],
            index=0,
            key=f"{sidebar_prefix}_scope",
        )

        img_query = st.text_input(
            "Image search", "",
            help="Substring filter on image_id (e.g. 'Wolf', 'Flowers').",
            key=f"{sidebar_prefix}_imgq",
        ).strip()

        n_boot = st.number_input(
            "Bootstrap replicates (0 to disable)",
            min_value=0, max_value=5000, value=0, step=500,
            help="Adds 95% CI columns to t-test outputs. >0 is slow.",
            key=f"{sidebar_prefix}_nboot",
        )

        st.markdown("---")
        is_blind = blind_models_toggle(key=f"{sidebar_prefix}_blind")

    if not sel_models or not sel_dims or not sel_cats:
        st.warning("Select at least one model, dimension, and category.")
        return

    # ── Load + filter trials ──
    selected_runs = runs_in_set[runs_in_set["model"].isin(sel_models)]
    run_to_model = dict(zip(selected_runs["run_id"], selected_runs["model"]))
    trials = _load_trials(tuple(selected_runs["run_id"].tolist()))
    if trials.empty:
        st.info("No completed trials for this filter combination.")
        return

    trials = trials[trials["dimension"].isin(sel_dims)].copy()
    trials["model"] = trials["run_id"].map(run_to_model)
    trials = trials.merge(
        norms.rename(columns={"category": "category"}),
        on="image_id", how="left",
    )
    trials = trials[trials["category"].isin(sel_cats)]
    if img_query:
        trials = trials[trials["image_id"].str.contains(img_query, case=False, regex=False)]
    if trials.empty:
        st.info("Filter matched 0 trials.")
        return

    trials["human_value"] = np.where(
        trials["dimension"] == "valence",
        trials["human_valence"],
        trials["human_arousal"],
    )

    # ── Apply blinding ──
    blind_map = apply_blinding(
        sorted(trials["model"].dropna().unique().tolist()),
        on=is_blind,
    )
    trials["model"] = trials["model"].map(lambda m: blind_map.get(m, m))
    sel_models_disp = [blind_map.get(m, m) for m in sel_models]

    # Per-image (run × image × dim) means
    per_img_run = (
        trials.groupby(["run_id", "model", "image_id", "category", "dimension"], as_index=False)
        .agg(llm_mean=("rating", "mean"),
             llm_sd=("rating", "std"),
             n=("rating", "count"),
             human_value=("human_value", "first"))
    )

    # ── Top KPI strip ──
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Models", per_img_run["model"].nunique())
    kpi_cols[1].metric("Images", per_img_run["image_id"].nunique())
    kpi_cols[2].metric("Categories", per_img_run["category"].nunique())
    kpi_cols[3].metric("Dimensions", per_img_run["dimension"].nunique())
    kpi_cols[4].metric("Trials", int(trials.shape[0]))

    # ── Active filter banner (sidebar controls apply to every tab) ──
    _models_str = (
        ", ".join(sel_models_disp) if len(sel_models_disp) <= 4
        else f"{len(sel_models_disp)} models"
    )
    _cats_str = ", ".join(sel_cats) if len(sel_cats) <= 4 else f"{len(sel_cats)} categories"
    _dims_str = ", ".join(sel_dims)
    _img_q = f" · image~'{img_query}'" if img_query else ""
    _scope_src = f"image set `{sel_set}`" if sel_set else f"pinned analysis ({len(runs_in_scope)} runs)"
    _blind_str = " · 🙈 blinded" if is_blind else ""
    st.info(
        f"**Sidebar filters apply to every tab below.** "
        f"{_scope_src} · models: {_models_str} · "
        f"dimensions: {_dims_str} · categories: {_cats_str} · "
        f"scope: **{scope}**{_img_q}{_blind_str}",
        icon="🎛️",
    )
    if is_blind:
        reveal_blinding_expander(blind_map)

    tabs = st.tabs([
        "📊 Descriptives",
        "🧪 t-tests",
        "📈 Regression",
        "🟦 Scatter",
        "📦 Distribution",
        "🚨 Outliers",
        "🔁 Inter-LLM agreement",
        "🧮 Cat × Model ANOVA",
        "🔬 Sample size & power",
    ])

    # ── Tab 1: descriptives ──
    with tabs[0]:
        rows = []
        for dim, sub_dim in trials.groupby("dimension"):
            human_vals = norms[
                "human_valence" if dim == "valence" else "human_arousal"
            ].dropna()
            if sel_cats:
                hcat = norms[norms["category"].isin(sel_cats)]
                human_vals = hcat[
                    "human_valence" if dim == "valence" else "human_arousal"
                ].dropna()
            rows.append({
                "scope": "Human (OASIS)", "dimension": dim,
                "n": int(human_vals.size),
                "mean": float(human_vals.mean()), "sd": float(human_vals.std(ddof=1)),
                "median": float(human_vals.median()),
                "min": float(human_vals.min()), "max": float(human_vals.max()),
            })
            for m, sub_m in sub_dim.groupby("model"):
                r = sub_m["rating"].astype(float)
                rows.append({
                    "scope": m, "dimension": dim, "n": int(r.size),
                    "mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                    "median": float(r.median()),
                    "min": float(r.min()), "max": float(r.max()),
                })
            r_all = sub_dim["rating"].astype(float)
            rows.append({
                "scope": "All LLMs (pooled trials)", "dimension": dim,
                "n": int(r_all.size),
                "mean": float(r_all.mean()), "sd": float(r_all.std(ddof=1)),
                "median": float(r_all.median()),
                "min": float(r_all.min()), "max": float(r_all.max()),
            })
        st.caption("Means / SDs respect the current filters (models, categories, image search).")
        df_desc = pd.DataFrame(rows).round(3)
        st.dataframe(df_desc, use_container_width=True, hide_index=True)
        _csv_download(df_desc, filename="descriptives.csv", key="dl_descriptives")

    # ── Tab 2: t-tests (in-page implementation respects current filter) ──
    with tabs[1]:
        st.caption("Per-image paired t-test (LLM image-mean vs OASIS human image-mean).")
        per_img_tt = _exclude_models_widget(
            per_img_run, sel_models_disp, key=f"{sidebar_prefix}_excl_tt",
        )
        rows = _ttest_rows(per_img_tt, scope=scope, n_boot=int(n_boot))
        if not rows:
            st.info("Not enough data for a paired t-test under the current filter.")
        else:
            df_tt = pd.DataFrame(rows).round(4)
            n_sig = int((df_tt["p"] < 0.05).sum()) if "p" in df_tt else 0
            max_d = float(df_tt["cohens_d"].abs().max()) if "cohens_d" in df_tt else float("nan")
            max_bias = float(df_tt["mean_diff"].abs().max()) if "mean_diff" in df_tt else float("nan")
            kc = st.columns(4)
            kc[0].metric("Tests", len(df_tt))
            kc[1].metric("Significant (p<0.05)", n_sig)
            kc[2].metric("Max |Cohen's d|", f"{max_d:.2f}")
            kc[3].metric("Max |bias|", f"{max_bias:.2f}")
            st.dataframe(df_tt, use_container_width=True, hide_index=True)
            _csv_download(df_tt, filename="ttests.csv", key="dl_ttests")

        st.markdown("---")
        _render_pairwise_compare(
            per_img_tt, sel_models_disp,
            key_prefix=f"{sidebar_prefix}_pair",
        )

    # ── Tab 3: regression LLM = a + b·Human ──
    with tabs[2]:
        st.caption("OLS regression LLM = a + b · Human (perfect calibration: a=0, b=1).")
        per_img_reg = _exclude_models_widget(
            per_img_run, sel_models_disp, key=f"{sidebar_prefix}_excl_reg",
        )
        rows = _regression_rows(per_img_reg, scope=scope)
        if not rows:
            st.info("Not enough data for a regression fit.")
        else:
            df_reg = pd.DataFrame(rows).round(4)
            best_r2 = float(df_reg["r2"].max()) if "r2" in df_reg else float("nan")
            worst_r2 = float(df_reg["r2"].min()) if "r2" in df_reg else float("nan")
            mean_slope = float(df_reg["slope"].mean()) if "slope" in df_reg else float("nan")
            kc = st.columns(4)
            kc[0].metric("Fits", len(df_reg))
            kc[1].metric("Best R²", f"{best_r2:.3f}")
            kc[2].metric("Worst R²", f"{worst_r2:.3f}")
            kc[3].metric("Mean slope", f"{mean_slope:.2f}")
            st.dataframe(df_reg, use_container_width=True, hide_index=True)
            _csv_download(df_reg, filename="regression.csv", key="dl_regression")

    # ── Tab 4: scatter ──
    with tabs[3]:
        try:
            _render_scatter(per_img_run, scope=scope)
        except Exception as e:
            st.error(f"Plot failed: {e}")
        _csv_download(
            per_img_run[["run_id", "model", "image_id", "category", "dimension",
                          "llm_mean", "human_value"]],
            filename="per_image_scatter.csv",
            key="dl_scatter",
        )

    # ── Tab 5: distribution ──
    with tabs[4]:
        _render_distribution(trials, norms, sel_cats=sel_cats, scope=scope)

    # ── Tab 6: outliers ──
    with tabs[5]:
        st.caption("Top images where the LLM mean disagrees most with the human mean.")
        per_img_out = _exclude_models_widget(
            per_img_run, sel_models_disp, key=f"{sidebar_prefix}_excl_out",
        )
        oc1, oc2 = st.columns([3, 2])
        top_k = oc1.slider("Top K", 5, 100, 20, 5, key=f"{sidebar_prefix}_topk")
        show_thumbs = oc2.checkbox("🖼️ Show thumbnails", value=True, key=f"{sidebar_prefix}_thumbs")
        if scope.startswith("Pooled"):
            agg = (
                per_img_out.groupby(["image_id", "category", "dimension"], as_index=False)
                .agg(llm_mean=("llm_mean", "mean"),
                     human_value=("human_value", "first"),
                     k_runs=("run_id", "nunique"))
            )
            agg["delta"] = agg["llm_mean"] - agg["human_value"]
        else:
            agg = per_img_out.copy()
            agg["delta"] = agg["llm_mean"] - agg["human_value"]
        agg["abs_delta"] = agg["delta"].abs()
        top = (
            agg.sort_values(["dimension", "abs_delta"], ascending=[True, False])
               .groupby("dimension", group_keys=False)
               .head(top_k)
               .reset_index(drop=True)
        )
        df_top = top.round(3)
        max_d = float(df_top["abs_delta"].max()) if not df_top.empty else float("nan")
        n_severe = int((df_top["abs_delta"] > 1.0).sum()) if not df_top.empty else 0
        kc = st.columns(3)
        kc[0].metric("Outliers shown", len(df_top))
        kc[1].metric("Max |Δ|", f"{max_d:.2f}")
        kc[2].metric("|Δ| > 1.0", n_severe)
        st.dataframe(df_top, use_container_width=True, hide_index=True)
        _csv_download(df_top, filename="outliers.csv", key="dl_outliers")
        if show_thumbs and not df_top.empty:
            _render_outlier_thumbnails(df_top)

    # ── Tab 7: inter-LLM agreement ──
    with tabs[6]:
        st.caption("Across the currently selected models, per dimension.")
        rows = _inter_llm_rows(per_img_run)
        if not rows:
            st.info("Need ≥ 2 models with overlapping images.")
        else:
            df_inter = pd.DataFrame(rows).round(3)
            best_r = float(df_inter["mean_pairwise_r"].max()) if "mean_pairwise_r" in df_inter else float("nan")
            best_icc = float(df_inter["icc3_1"].max()) if "icc3_1" in df_inter else float("nan")
            kc = st.columns(3)
            kc[0].metric("Dimensions", len(df_inter))
            kc[1].metric("Best mean pairwise r", f"{best_r:.3f}")
            kc[2].metric("Best ICC(3,1)", f"{best_icc:.3f}")
            st.dataframe(df_inter, use_container_width=True, hide_index=True)
            _csv_download(df_inter, filename="inter_llm_agreement.csv", key="dl_inter")

    # ── Tab 8: ANOVA Category × Model ──
    with tabs[7]:
        _render_anova_tab(per_img_run)

    # ── Tab 9: Sample size & power ──
    with tabs[8]:
        _render_power_tab(
            per_img_run, sel_models_disp,
            key_prefix=f"{sidebar_prefix}_power",
        )


# ─── tab helpers ───────────────────────────────────────────────────────────
def _cohens_d_paired(diff: np.ndarray) -> float:
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def _ttest_one(pred: np.ndarray, human: np.ndarray) -> dict:
    try:
        from scipy import stats as sps
        t_stat, p_val = sps.ttest_rel(pred, human)
        r_pearson = float(sps.pearsonr(pred, human)[0]) if pred.std() and human.std() else float("nan")
    except Exception:
        diff = pred - human
        sd_diff = float(diff.std(ddof=1))
        n = len(diff)
        se = sd_diff / np.sqrt(n) if sd_diff > 0 else float("nan")
        t_stat = float(diff.mean() / se) if se and se > 0 else float("nan")
        p_val = float("nan")
        r_pearson = float(np.corrcoef(pred, human)[0, 1]) if pred.std() and human.std() else float("nan")
    diff = pred - human
    return {
        "n_images": int(len(pred)),
        "llm_mean": float(pred.mean()),
        "human_mean": float(human.mean()),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "t": float(t_stat), "df": int(len(pred) - 1),
        "p": float(p_val),
        "cohens_d": _cohens_d_paired(diff),
        "pearson_r": float(r_pearson),
    }


def _ttest_rows(per_img_run: pd.DataFrame, *, scope: str, n_boot: int) -> list[dict]:
    rows = []
    if scope == "Pooled all-LLMs":
        # Average across runs first → one LLM per image per dim
        agg = (
            per_img_run.groupby(["image_id", "dimension"], as_index=False)
            .agg(llm_mean=("llm_mean", "mean"),
                 human_value=("human_value", "first"),
                 k_runs=("run_id", "nunique"))
        )
        for dim, sub in agg.groupby("dimension"):
            j = sub.dropna(subset=["llm_mean", "human_value"])
            if len(j) < 2:
                continue
            row = {"scope": "Pooled all-LLMs", "dimension": dim,
                   "k_runs": int(j["k_runs"].max())}
            row.update(_ttest_one(j["llm_mean"].to_numpy(), j["human_value"].to_numpy()))
            if n_boot > 0:
                p = j["llm_mean"].to_numpy(); h = j["human_value"].to_numpy()
                mlo, mhi = an._bootstrap_ci(p, h, lambda a, b: float(np.mean(a - b)), n_boot=n_boot)
                row["mean_diff_ci_lo"] = mlo; row["mean_diff_ci_hi"] = mhi
            rows.append(row)
    elif scope == "By model":
        for (m, dim), sub in per_img_run.groupby(["model", "dimension"]):
            j = sub.dropna(subset=["llm_mean", "human_value"])
            if len(j) < 2:
                continue
            row = {"scope": m, "dimension": dim}
            row.update(_ttest_one(j["llm_mean"].to_numpy(), j["human_value"].to_numpy()))
            rows.append(row)
    elif scope == "By category":
        agg = (
            per_img_run.groupby(["image_id", "category", "dimension"], as_index=False)
            .agg(llm_mean=("llm_mean", "mean"), human_value=("human_value", "first"),
                 k_runs=("run_id", "nunique"))
        )
        for (cat, dim), sub in agg.groupby(["category", "dimension"]):
            j = sub.dropna(subset=["llm_mean", "human_value"])
            if len(j) < 2:
                continue
            row = {"scope": cat, "dimension": dim}
            row.update(_ttest_one(j["llm_mean"].to_numpy(), j["human_value"].to_numpy()))
            rows.append(row)
    else:  # Model × Category
        for (m, cat, dim), sub in per_img_run.groupby(["model", "category", "dimension"]):
            j = sub.dropna(subset=["llm_mean", "human_value"])
            if len(j) < 2:
                continue
            row = {"scope": f"{m} | {cat}", "dimension": dim}
            row.update(_ttest_one(j["llm_mean"].to_numpy(), j["human_value"].to_numpy()))
            rows.append(row)
    return rows


def _regression_rows(per_img_run: pd.DataFrame, *, scope: str) -> list[dict]:
    rows = []

    def _fit(label: str, j: pd.DataFrame, dim: str):
        if len(j) < 3:
            return
        x = j["human_value"].to_numpy(dtype=float)
        y = j["llm_mean"].to_numpy(dtype=float)
        if x.std() == 0:
            return
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({
            "scope": label, "dimension": dim, "n": int(len(j)),
            "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2),
            "residual_sd": float(np.sqrt(ss_res / max(len(j) - 2, 1))),
        })

    if scope == "Pooled all-LLMs":
        agg = per_img_run.groupby(["image_id", "dimension"], as_index=False).agg(
            llm_mean=("llm_mean", "mean"), human_value=("human_value", "first"),
        )
        for dim, sub in agg.groupby("dimension"):
            _fit("Pooled all-LLMs", sub.dropna(), dim)
    elif scope == "By model":
        for (m, dim), sub in per_img_run.groupby(["model", "dimension"]):
            _fit(m, sub.dropna(subset=["llm_mean", "human_value"]), dim)
    elif scope == "By category":
        agg = per_img_run.groupby(["image_id", "category", "dimension"], as_index=False).agg(
            llm_mean=("llm_mean", "mean"), human_value=("human_value", "first"),
        )
        for (cat, dim), sub in agg.groupby(["category", "dimension"]):
            _fit(cat, sub.dropna(), dim)
    else:
        for (m, cat, dim), sub in per_img_run.groupby(["model", "category", "dimension"]):
            _fit(f"{m} | {cat}", sub.dropna(subset=["llm_mean", "human_value"]), dim)
    return rows


def _render_scatter(per_img_run: pd.DataFrame, *, scope: str) -> None:
    import plotly.express as px

    if scope == "Pooled all-LLMs":
        agg = per_img_run.groupby(["image_id", "category", "dimension"], as_index=False).agg(
            llm_mean=("llm_mean", "mean"), human_value=("human_value", "first"),
        )
        color_col = "category"
        palette = CATEGORY_PALETTE
    elif scope == "By model":
        agg = per_img_run.copy()
        color_col = "model"
        palette = None
    elif scope == "By category":
        agg = per_img_run.groupby(["image_id", "category", "dimension"], as_index=False).agg(
            llm_mean=("llm_mean", "mean"), human_value=("human_value", "first"),
        )
        color_col = "category"
        palette = CATEGORY_PALETTE
    else:
        agg = per_img_run.copy()
        agg["model_x_cat"] = agg["model"].astype(str) + " | " + agg["category"].astype(str)
        color_col = "model_x_cat"
        palette = None

    for dim in sorted(agg["dimension"].unique()):
        sub = agg[agg["dimension"] == dim].dropna(subset=["llm_mean", "human_value"])
        if sub.empty:
            continue
        fig = px.scatter(
            sub, x="llm_mean", y="human_value",
            color=color_col,
            color_discrete_map=palette,
            hover_data=[c for c in ("image_id", "model", "category", "k_runs") if c in sub.columns],
            title=f"{dim.capitalize()} — LLM vs Human (n={len(sub)})",
        )
        fig.add_shape(
            type="line", x0=1, y0=1, x1=7, y1=7,
            line=dict(dash="dash", color="rgba(255,255,255,0.5)"),
        )
        fig.update_layout(
            xaxis=dict(range=[1, 7], title="LLM rating"),
            yaxis=dict(range=[1, 7], title="Human rating"),
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_distribution(trials: pd.DataFrame, norms: pd.DataFrame, *, sel_cats: list[str], scope: str) -> None:
    import plotly.express as px

    sub_norms = norms[norms["category"].isin(sel_cats)] if sel_cats else norms
    for dim in sorted(trials["dimension"].unique()):
        col = "human_valence" if dim == "valence" else "human_arousal"
        h = sub_norms[col].dropna().to_numpy()
        ll = trials[trials["dimension"] == dim]
        if ll.empty:
            continue
        fig = px.histogram(
            ll, x="rating",
            color="model" if scope in ("By model", "Model × Category") else None,
            nbins=7, barmode="overlay", opacity=0.55, histnorm="probability density",
            title=f"{dim.capitalize()} — LLM trial ratings (filtered) vs human image means",
        )
        fig.add_histogram(
            x=h, name="Human (image means)",
            histnorm="probability density",
            opacity=0.55, marker_color="#E45756", nbinsx=20,
        )
        # KS + Wasserstein in caption
        try:
            from scipy import stats as sps
            ks = sps.ks_2samp(ll["rating"].to_numpy(dtype=float), h)
            w = float(sps.wasserstein_distance(ll["rating"].to_numpy(dtype=float), h))
            st.caption(f"**{dim}** — KS = {ks.statistic:.3f}, p = {ks.pvalue:.3g}; Wasserstein = {w:.3f}")
        except Exception:
            pass
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)


def _inter_llm_rows(per_img_run: pd.DataFrame) -> list[dict]:
    rows = []
    for dim, sub in per_img_run.groupby("dimension"):
        wide = sub.pivot_table(index="image_id", columns="model", values="llm_mean", aggfunc="mean").dropna()
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue
        corr = wide.corr().values
        iu = np.triu_indices_from(corr, k=1)
        rows.append({
            "dimension": dim,
            "n_images": int(wide.shape[0]),
            "k_models": int(wide.shape[1]),
            "mean_pairwise_r": float(np.nanmean(corr[iu])),
            "median_pairwise_r": float(np.nanmedian(corr[iu])),
            "mean_per_image_sd": float(wide.std(axis=1, ddof=1).mean()),
        })
    return rows


# ─── helpers added by analytics merge (per-tab exclude / thumbs / power / anova) ───
def _exclude_models_widget(
    per_img_run: pd.DataFrame,
    sel_models_disp: list[str],
    *,
    key: str,
) -> pd.DataFrame:
    """Render an 'exclude models from this tab' multiselect. Returns a filtered
    copy of ``per_img_run`` with the chosen models dropped.

    Honours the (possibly blinded) display labels used in ``per_img_run.model``.
    """
    if not sel_models_disp or per_img_run.empty:
        return per_img_run
    excl = st.multiselect(
        "Exclude models from this tab",
        options=sel_models_disp,
        default=[],
        key=key,
        help="Per-tab override — does not affect the sidebar selection.",
    )
    if not excl:
        return per_img_run
    return per_img_run[~per_img_run["model"].isin(excl)].copy()


def _render_outlier_thumbnails(df_top: pd.DataFrame) -> None:
    """Render outlier rows as a grid of thumbnails (4 per row)."""
    st.markdown("##### Thumbnails")
    rows = df_top.to_dict("records")
    for i in range(0, len(rows), 4):
        chunk = rows[i:i + 4]
        cols = st.columns(4)
        for col, r in zip(cols, chunk):
            with col:
                try:
                    p = image_path(r["image_id"])
                    col.image(str(p), use_container_width=True)
                except FileNotFoundError:
                    col.caption("(image missing)")
                col.caption(
                    f"**{r['image_id']}** · {r['category']} · {r['dimension']}\n"
                    f"LLM={r['llm_mean']:.2f} · Human={r['human_value']:.2f} · Δ={r['delta']:+.2f}"
                )


def _render_anova_tab(per_img_run: pd.DataFrame) -> None:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        st.info("Install `statsmodels` to enable the Category × Model ANOVA.")
        return
    df_an = per_img_run.copy()
    df_an["abs_err"] = (df_an["llm_mean"] - df_an["human_value"]).abs()
    rows = []
    for dim, sub in df_an.groupby("dimension"):
        sub = sub.dropna(subset=["abs_err", "category", "model"])
        if sub["category"].nunique() < 2 or sub["model"].nunique() < 2 or len(sub) < 8:
            continue
        try:
            model = smf.ols("abs_err ~ C(category) + C(model) + C(category):C(model)", data=sub).fit()
            aov = sm.stats.anova_lm(model, typ=2)
        except Exception as e:
            st.warning(f"ANOVA failed for {dim}: {e}")
            continue
        ss_total = float(aov["sum_sq"].sum())
        for fac, r in aov.iterrows():
            rows.append({
                "dimension": dim,
                "factor": fac,
                "df": float(r["df"]),
                "ss": float(r["sum_sq"]),
                "F": float(r["F"]) if "F" in r and not pd.isna(r["F"]) else float("nan"),
                "p": float(r["PR(>F)"]) if "PR(>F)" in r and not pd.isna(r["PR(>F)"]) else float("nan"),
                "eta_sq": float(r["sum_sq"]) / ss_total if ss_total > 0 else float("nan"),
            })
    if not rows:
        st.info("Need at least 2 models × 2 categories for the ANOVA.")
        return
    df_anova = pd.DataFrame(rows).round(4)
    st.dataframe(df_anova, use_container_width=True, hide_index=True)
    _csv_download(df_anova, filename="anova_category_model.csv", key="dl_anova")


def _render_power_tab(
    per_img_run: pd.DataFrame,
    sel_models_disp: list[str],
    *,
    key_prefix: str,
) -> None:
    """Sample size & power for the per-image paired t-test against human norms.

    Two sub-views:
      * **Observed** — for each (model × dimension) combination in scope,
        report n, observed Cohen's d (paired), achieved power at α=0.05
        (two-sided), and the n required to hit a target power for that
        observed effect size.
      * **A priori** — sliders for d and target power → required n.
    """
    st.caption(
        "Power analysis for the **per-image paired t-test** "
        "(LLM image-mean vs OASIS human image-mean), two-sided, α=0.05."
    )
    try:
        from statsmodels.stats.power import TTestPower
    except Exception:
        st.info("Install `statsmodels` to enable power analysis.")
        return

    tp = TTestPower()
    alpha = st.number_input(
        "α (two-sided)", min_value=0.001, max_value=0.20,
        value=0.05, step=0.005, key=f"{key_prefix}_alpha",
    )

    sub_a, sub_b = st.tabs(["Observed", "A priori"])

    # ── Observed ──
    with sub_a:
        per_img_p = _exclude_models_widget(
            per_img_run, sel_models_disp, key=f"{key_prefix}_excl",
        )
        target_power = st.slider(
            "Target power (for required-n column)", 0.5, 0.99, 0.80, 0.01,
            key=f"{key_prefix}_target",
        )
        rows = []
        for (m, dim), sub in per_img_p.groupby(["model", "dimension"]):
            d_pairs = (sub.groupby("image_id")
                          .agg(llm=("llm_mean", "mean"),
                               human=("human_value", "first"))
                          .dropna())
            n = int(d_pairs.shape[0])
            if n < 3:
                continue
            diff = (d_pairs["llm"] - d_pairs["human"]).to_numpy()
            sd = float(diff.std(ddof=1))
            d_obs = float(diff.mean() / sd) if sd > 0 else float("nan")
            try:
                achieved = float(tp.power(
                    effect_size=abs(d_obs), nobs=n, alpha=float(alpha)
                )) if not np.isnan(d_obs) and abs(d_obs) > 0 else float("nan")
            except Exception:
                achieved = float("nan")
            try:
                req_n = float(tp.solve_power(
                    effect_size=abs(d_obs), alpha=float(alpha),
                    power=float(target_power), alternative="two-sided",
                )) if not np.isnan(d_obs) and abs(d_obs) > 0 else float("nan")
            except Exception:
                req_n = float("nan")
            rows.append({
                "model": m, "dimension": dim, "n_images": n,
                "cohens_d": round(d_obs, 3),
                "achieved_power": round(achieved, 3) if not np.isnan(achieved) else None,
                f"n_for_power_{target_power:.2f}": (
                    int(np.ceil(req_n)) if not np.isnan(req_n) else None
                ),
            })
        if not rows:
            st.info("Not enough data to compute power.")
        else:
            df_pow = pd.DataFrame(rows)
            kc = st.columns(3)
            kc[0].metric("Rows", len(df_pow))
            ach_col = "achieved_power"
            if ach_col in df_pow:
                med = df_pow[ach_col].dropna().median()
                kc[1].metric("Median achieved power", f"{med:.2f}" if pd.notna(med) else "n/a")
                under = int((df_pow[ach_col] < 0.80).sum())
                kc[2].metric("Underpowered (<0.80)", under)
            st.dataframe(df_pow, use_container_width=True, hide_index=True)
            _csv_download(df_pow, filename="power_observed.csv",
                          key=f"{key_prefix}_dl_obs")

    # ── A priori ──
    with sub_b:
        c1, c2 = st.columns(2)
        d_in = c1.slider("Effect size d (paired)", 0.05, 1.50, 0.30, 0.05,
                         key=f"{key_prefix}_d")
        pwr = c2.slider("Target power", 0.50, 0.99, 0.80, 0.01,
                        key=f"{key_prefix}_pwr")
        try:
            n_req = tp.solve_power(
                effect_size=float(d_in), alpha=float(alpha),
                power=float(pwr), alternative="two-sided",
            )
        except Exception as e:
            st.error(f"Power calc failed: {e}")
            return
        st.metric(
            "Required n_images",
            int(np.ceil(n_req)) if n_req and not np.isnan(n_req) else "n/a",
            help="Number of paired (image, dim) observations needed.",
        )

        # Curve over a range of d
        ds = np.arange(0.05, 1.51, 0.05)
        ns = [
            tp.solve_power(effect_size=float(d), alpha=float(alpha),
                           power=float(pwr), alternative="two-sided")
            for d in ds
        ]
        curve = pd.DataFrame({"cohens_d": ds, "n_required": np.ceil(ns).astype(int)})
        st.line_chart(curve.set_index("cohens_d"))
        st.caption(f"Curve fixes power={pwr:.2f}, α={float(alpha):.3f}.")
        _csv_download(curve, filename="power_curve.csv",
                      key=f"{key_prefix}_dl_curve")


def _render_pairwise_compare(
    per_img_run: pd.DataFrame,
    sel_models_disp: list[str],
    *,
    key_prefix: str,
) -> None:
    """Side-by-side comparison of two selected LLMs.

    Renders three sub-blocks:

    1. **Each model vs human** — paired t-test per dimension (image-mean
       LLM_X vs image-mean human) for Model A and Model B side-by-side.
    2. **Model A vs Model B** — direct paired t-test on the per-image
       LLM means (no human in the loop).
    3. **A − B per category** — mean delta and Cohen's d by category.
    """
    st.markdown("##### 🔀 Compare two LLMs head-to-head")
    if len(sel_models_disp) < 2:
        st.caption("Select at least two models in the sidebar to enable pairwise comparison.")
        return

    avail = sorted(per_img_run["model"].dropna().unique().tolist())
    if len(avail) < 2:
        st.caption("Not enough models in scope after filters.")
        return

    cA, cB = st.columns(2)
    a_default = 0
    b_default = 1 if len(avail) > 1 else 0
    model_a = cA.selectbox(
        "Model A", avail, index=a_default, key=f"{key_prefix}_a",
    )
    # Force B != A in the default unless impossible
    b_options = [m for m in avail if m != model_a]
    if not b_options:
        st.caption("Need a second model.")
        return
    model_b = cB.selectbox(
        "Model B", b_options,
        index=min(b_default, len(b_options) - 1),
        key=f"{key_prefix}_b",
    )

    sub_a = per_img_run[per_img_run["model"] == model_a]
    sub_b = per_img_run[per_img_run["model"] == model_b]
    if sub_a.empty or sub_b.empty:
        st.caption("One of the selected models has no rows in scope.")
        return

    # ── 1. Each model vs human (paired) ──
    rows = []
    for label, sub in (("A: " + model_a, sub_a), ("B: " + model_b, sub_b)):
        for dim, ssub in sub.groupby("dimension"):
            pairs = (ssub.groupby("image_id")
                         .agg(llm=("llm_mean", "mean"),
                              human=("human_value", "first"))
                         .dropna())
            if pairs.shape[0] < 3:
                continue
            stats = _ttest_one(pairs["llm"].to_numpy(), pairs["human"].to_numpy())
            d = _cohens_d_paired((pairs["llm"] - pairs["human"]).to_numpy())
            rows.append({
                "scope": label, "dimension": dim, "n_images": int(pairs.shape[0]),
                "llm_mean": round(float(pairs["llm"].mean()), 3),
                "human_mean": round(float(pairs["human"].mean()), 3),
                "mean_diff": round(float((pairs["llm"] - pairs["human"]).mean()), 3),
                "t": round(stats["t"], 3),
                "p": round(stats["p"], 4),
                "cohens_d": round(d, 3),
                "pearson_r": round(stats["r"], 3),
            })
    if rows:
        st.markdown("**Each model vs human**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 2. Direct paired t-test (Model A vs Model B) on shared images ──
    st.markdown("**Model A vs Model B (paired on shared images)**")
    rows_ab = []
    delta_long = []  # for category breakdown
    for dim in sorted(set(sub_a["dimension"]).intersection(sub_b["dimension"])):
        ax = (sub_a[sub_a["dimension"] == dim]
              .groupby("image_id")
              .agg(a=("llm_mean", "mean"),
                   category=("category", "first")))
        bx = (sub_b[sub_b["dimension"] == dim]
              .groupby("image_id")
              .agg(b=("llm_mean", "mean")))
        pairs = ax.join(bx, how="inner").dropna()
        if pairs.shape[0] < 3:
            continue
        diff = (pairs["a"] - pairs["b"]).to_numpy()
        stats = _ttest_one(pairs["a"].to_numpy(), pairs["b"].to_numpy())
        d = _cohens_d_paired(diff)
        rows_ab.append({
            "dimension": dim, "n_images": int(pairs.shape[0]),
            "mean_A": round(float(pairs["a"].mean()), 3),
            "mean_B": round(float(pairs["b"].mean()), 3),
            "mean_diff_A_minus_B": round(float(diff.mean()), 3),
            "sd_diff": round(float(diff.std(ddof=1)), 3),
            "t": round(stats["t"], 3),
            "p": round(stats["p"], 4),
            "cohens_d": round(d, 3),
            "pearson_r": round(stats["r"], 3),
        })
        # Long form for category breakdown / scatter
        tmp = pairs.reset_index().assign(dimension=dim)
        delta_long.append(tmp)

    if not rows_ab:
        st.info("No shared images between the two models for any dimension.")
        return

    df_ab = pd.DataFrame(rows_ab)
    kc = st.columns(4)
    kc[0].metric("Dimensions tested", len(df_ab))
    sig = int((df_ab["p"] < 0.05).sum())
    kc[1].metric("Significant (p<0.05)", sig)
    kc[2].metric("Max |Cohen's d|", f"{df_ab['cohens_d'].abs().max():.2f}")
    kc[3].metric("Max |Δ A−B|", f"{df_ab['mean_diff_A_minus_B'].abs().max():.2f}")
    st.dataframe(df_ab, use_container_width=True, hide_index=True)
    _csv_download(df_ab, filename="pairwise_AvsB.csv",
                  key=f"{key_prefix}_dl_ab")

    # ── 3. Per-category breakdown ──
    long = pd.concat(delta_long, ignore_index=True)
    long["delta"] = long["a"] - long["b"]
    cat_rows = []
    for (dim, cat), sub in long.groupby(["dimension", "category"]):
        if sub.shape[0] < 3:
            continue
        diff = sub["delta"].to_numpy()
        d = _cohens_d_paired(diff)
        cat_rows.append({
            "dimension": dim, "category": cat,
            "n_images": int(sub.shape[0]),
            "mean_A": round(float(sub["a"].mean()), 3),
            "mean_B": round(float(sub["b"].mean()), 3),
            "mean_diff_A_minus_B": round(float(diff.mean()), 3),
            "cohens_d": round(d, 3),
        })
    if cat_rows:
        st.markdown("**A − B by category**")
        st.dataframe(pd.DataFrame(cat_rows),
                     use_container_width=True, hide_index=True)

    # ── 4. Scatter A vs B ──
    try:
        import plotly.express as px
        fig = px.scatter(
            long, x="b", y="a", color="category", facet_col="dimension",
            hover_data=["image_id"],
            labels={"a": f"{model_a} (Model A)", "b": f"{model_b} (Model B)"},
            color_discrete_map=CATEGORY_PALETTE,
        )
        # y = x reference line
        for col_idx, dim in enumerate(sorted(long["dimension"].unique()), start=1):
            fig.add_shape(
                type="line", x0=1, y0=1, x1=7, y1=7,
                line={"dash": "dash", "color": "#888"},
                xref=f"x{col_idx}", yref=f"y{col_idx}",
            )
        fig.update_layout(height=380, margin={"t": 40, "b": 30})
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
