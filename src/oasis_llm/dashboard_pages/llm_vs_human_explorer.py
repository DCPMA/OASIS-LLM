"""LLM vs Human (OASIS) — interactive Explorer page.

Live, multi-axis exploration over completed runs joined with OASIS human
norms. Users can filter by:

- model (multi-select; "All" = pooled-LLM)
- OASIS category (Animal / Scene / Person / Object)
- dimension (valence / arousal)
- individual image (search)
- aggregation scope: pooled all-LLMs, by-model, by-category, model × category

The page surfaces the analyses we exposed in :mod:`oasis_llm.analyses`:
descriptives, paired t-tests, pooled t-test, regression, per-category
breakdown, distribution comparison, outlier images, inter-LLM agreement,
and a category × model ANOVA on absolute error.
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from oasis_llm import analyses as an
from oasis_llm.dashboard_pages._ui import connect_ro, page_header

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
def render() -> None:
    page_header(
        "LLM vs Human Explorer",
        "Live, scopable comparison of LLM ratings against OASIS human norms.",
        icon="🔬",
    )

    runs_meta = _load_run_meta()
    if runs_meta.empty:
        st.info("No runs found in the database yet.")
        return

    runs_done = runs_meta[runs_meta["status"] == "done"].copy()
    if runs_done.empty:
        st.info("No completed runs yet.")
        return

    # ── Sidebar filters ──
    with st.sidebar:
        st.markdown("### Explorer filters")

        image_sets = sorted(runs_done["image_set"].dropna().unique().tolist())
        default_set = image_sets[0] if image_sets else None
        sel_set = st.selectbox("Image set", image_sets, index=0 if default_set else None)

        runs_in_set = runs_done[runs_done["image_set"] == sel_set]
        models_avail = sorted(runs_in_set["model"].dropna().unique().tolist())
        sel_models = st.multiselect(
            "Models", models_avail,
            default=models_avail,
            help="Select one or more models. 'All' (pooled) = mean across the selected models.",
        )

        sel_dims = st.multiselect(
            "Dimensions", ["valence", "arousal"],
            default=["valence", "arousal"],
        )

        norms = _load_norms()
        cat_avail = sorted(norms["category"].dropna().unique().tolist())
        sel_cats = st.multiselect("Categories", cat_avail, default=cat_avail)

        scope = st.radio(
            "Aggregation scope",
            ["Pooled all-LLMs", "By model", "By category", "Model × Category"],
            index=0,
        )

        img_query = st.text_input(
            "Image search", "",
            help="Substring filter on image_id (e.g. 'Wolf', 'Flowers').",
        ).strip()

        n_boot = st.number_input(
            "Bootstrap replicates (0 to disable)",
            min_value=0, max_value=5000, value=0, step=500,
            help="Adds 95% CI columns to t-test outputs. >0 is slow.",
        )

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

    tabs = st.tabs([
        "📊 Descriptives",
        "🧪 t-tests",
        "📈 Regression",
        "🟦 Scatter",
        "📦 Distribution",
        "🚨 Outliers",
        "🔁 Inter-LLM agreement",
        "🧮 Cat × Model ANOVA",
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
        rows = _ttest_rows(per_img_run, scope=scope, n_boot=int(n_boot))
        if not rows:
            st.info("Not enough data for a paired t-test under the current filter.")
        else:
            df_tt = pd.DataFrame(rows).round(4)
            st.dataframe(df_tt, use_container_width=True, hide_index=True)
            _csv_download(df_tt, filename="ttests.csv", key="dl_ttests")

    # ── Tab 3: regression LLM = a + b·Human ──
    with tabs[2]:
        st.caption("OLS regression LLM = a + b · Human (perfect calibration: a=0, b=1).")
        rows = _regression_rows(per_img_run, scope=scope)
        if not rows:
            st.info("Not enough data for a regression fit.")
        else:
            df_reg = pd.DataFrame(rows).round(4)
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
        top_k = st.slider("Top K", 5, 100, 20, 5)
        if scope.startswith("Pooled"):
            agg = (
                per_img_run.groupby(["image_id", "category", "dimension"], as_index=False)
                .agg(llm_mean=("llm_mean", "mean"),
                     human_value=("human_value", "first"),
                     k_runs=("run_id", "nunique"))
            )
            agg["delta"] = agg["llm_mean"] - agg["human_value"]
        else:
            agg = per_img_run.copy()
            agg["delta"] = agg["llm_mean"] - agg["human_value"]
        agg["abs_delta"] = agg["delta"].abs()
        top = (
            agg.sort_values(["dimension", "abs_delta"], ascending=[True, False])
               .groupby("dimension", group_keys=False)
               .head(top_k)
               .reset_index(drop=True)
        )
        df_top = top.round(3)
        st.dataframe(df_top, use_container_width=True, hide_index=True)
        _csv_download(df_top, filename="outliers.csv", key="dl_outliers")

    # ── Tab 7: inter-LLM agreement ──
    with tabs[6]:
        st.caption("Across the currently selected models, per dimension.")
        rows = _inter_llm_rows(per_img_run)
        if not rows:
            st.info("Need ≥ 2 models with overlapping images.")
        else:
            df_inter = pd.DataFrame(rows).round(3)
            st.dataframe(df_inter, use_container_width=True, hide_index=True)
            _csv_download(df_inter, filename="inter_llm_agreement.csv", key="dl_inter")

    # ── Tab 8: ANOVA Category × Model ──
    with tabs[7]:
        try:
            import statsmodels.api as sm  # noqa: F401
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
        else:
            df_anova = pd.DataFrame(rows).round(4)
            st.dataframe(df_anova, use_container_width=True, hide_index=True)
            _csv_download(df_anova, filename="anova_category_model.csv", key="dl_anova")


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
