"""Compare Runs page: side-by-side scatter + pairwise correlation matrix.

Pick N runs (any provider, any dataset). For each dimension, a single scatter
plot with one color per run shows LLM rating (y) vs human norm (x). A pairwise
correlation matrix surfaces inter-rater agreement between runs on the
same images.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd
import streamlit as st

from oasis_llm.dashboard_pages._ui import (
    connect_ro, db_locked_warning, kpi, page_header,
)


HUMAN_NORMS_CSV = "OASIS/OASIS.csv"


def _all_runs_meta(con) -> list[dict]:
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
    """Return long-format means per (run_id, image_id, dimension)."""
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


def render():
    page_header(
        "Compare Runs",
        "Pick any runs, see ratings vs human norms side-by-side, plus pairwise correlations.",
        icon="🔀",
    )
    con = connect_ro()
    if con is None:
        db_locked_warning(); return

    runs = _all_runs_meta(con)
    if not runs:
        st.info("No runs yet."); return

    # ── Filters & multi-select ──────────────────────────────────────────────
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
        st.info("No runs match these filters (note: runs with 0 completed trials are hidden).")
        return

    label_to_id = {
        f"{r['run_id']}  ·  {r['model']}  ·  {r['done']} done": r["run_id"]
        for r in eligible
    }
    selected_labels = st.multiselect(
        "Select runs to compare (2+ recommended)",
        list(label_to_id.keys()),
        default=list(label_to_id.keys())[:min(3, len(label_to_id))],
    )
    selected_ids = [label_to_id[l] for l in selected_labels]
    if not selected_ids:
        st.info("Select at least one run."); return

    # ── Pull data ───────────────────────────────────────────────────────────
    with st.spinner("Aggregating per-image means…"):
        df = _per_image_means(con, selected_ids)
        norms = _human_norms()
    if df.empty:
        st.info("Selected runs have no completed trials."); return

    # Run → label lookup (model id is more readable than run_id)
    run_to_label = {r["run_id"]: r["model"] or r["run_id"] for r in eligible}
    df["run_label"] = df["run_id"].map(lambda rid: f"{run_to_label[rid]} · {rid[-6:]}")

    merged = df.merge(norms, on="image_id", how="inner")
    merged["human_value"] = merged.apply(
        lambda r: r["human_valence"] if r["dimension"] == "valence"
        else r["human_arousal"] if r["dimension"] == "arousal"
        else None, axis=1,
    )
    merged = merged.dropna(subset=["human_value"])

    # ── KPIs ────────────────────────────────────────────────────────────────
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
    tab_scatter, tab_corr, tab_table = st.tabs([
        "📈 Scatter vs human", "🔗 Pairwise correlations", "📋 Long table",
    ])

    # ── Scatter per dimension ───────────────────────────────────────────────
    with tab_scatter:
        for dim in sorted(merged["dimension"].unique()):
            sub = merged[merged["dimension"] == dim].copy()
            if sub.empty: continue
            st.markdown(f"#### {dim.title()}")
            chart_df = sub[["image_id", "run_label", "human_value", "mean_rating"]].rename(
                columns={"human_value": "human", "mean_rating": "llm"}
            )
            st.scatter_chart(
                chart_df, x="human", y="llm", color="run_label",
                height=360,
            )
            # Per-run summary
            summary = (
                sub.groupby("run_label")
                .apply(
                    lambda g: pd.Series({
                        "n": len(g),
                        "pearson_r": float(g["mean_rating"].corr(g["human_value"])) if g["mean_rating"].std() > 0 else float("nan"),
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

    # ── Pairwise correlations (LLM-vs-LLM) ─────────────────────────────────
    with tab_corr:
        if len(selected_ids) < 2:
            st.info("Pick 2+ runs for pairwise correlations.")
        else:
            for dim in sorted(df["dimension"].unique()):
                sub = df[df["dimension"] == dim]
                wide = sub.pivot(index="image_id", columns="run_label", values="mean_rating")
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
                    f"n images with overlap (all runs): {int(wide.dropna().shape[0])}"
                )

    # ── Raw long table ──────────────────────────────────────────────────────
    with tab_table:
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
