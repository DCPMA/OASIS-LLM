"""Compare 5 LLMs' valence/arousal ratings against OASIS human means.

Cohort: image_set == '20260426-uniform40' (40 images x 20 samples x 5 models).

Outputs (under analysis/llm_vs_human_uniform40/):
    descriptives.csv        Per-model + aggregate descriptive statistics.
    ttests_per_model.csv    Per-model paired (image-level) and Welch (trial-level) t-tests.
    ttests_aggregate.csv    All-LLMs-pooled vs humans, paired and Welch.
    per_image_means.csv     Wide table: image x (human, each model) means.
    plots/                  Distribution and comparison figures (PNG).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "llm_vs_human_uniform40"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "llm_runs.duckdb"
OASIS_CSV = ROOT / "OASIS" / "OASIS.csv"
IMAGE_SET = "20260426-uniform40"
DIMS = ["valence", "arousal"]

sns.set_theme(style="whitegrid", context="paper")


def load_trials() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        """
        SELECT t.run_id,
               json_extract_string(r.config_json,'$.model') AS model,
               t.image_id,
               t.dimension,
               t.sample_idx,
               t.rating
        FROM trials t
        JOIN runs r USING(run_id)
        WHERE json_extract_string(r.config_json,'$.image_set') = ?
          AND t.status = 'done'
          AND t.rating IS NOT NULL
        """,
        [IMAGE_SET],
    ).df()
    con.close()
    return df


def load_humans() -> pd.DataFrame:
    h = pd.read_csv(OASIS_CSV)
    return h.rename(columns={"Theme": "image_id"})[
        ["image_id", "Category", "Valence_mean", "Arousal_mean"]
    ]


def short_name(model: str) -> str:
    return model.split("/")[-1]


def descriptives(trials: pd.DataFrame, humans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in DIMS:
        human_col = "Valence_mean" if dim == "valence" else "Arousal_mean"
        h = humans[human_col].dropna()
        rows.append(
            {
                "source": "Human (OASIS)",
                "dimension": dim,
                "n": int(h.size),
                "mean": h.mean(),
                "sd": h.std(ddof=1),
                "median": h.median(),
                "min": h.min(),
                "max": h.max(),
            }
        )
        for model, sub in trials[trials.dimension == dim].groupby("model"):
            r = sub.rating.astype(float)
            rows.append(
                {
                    "source": short_name(model),
                    "dimension": dim,
                    "n": int(r.size),
                    "mean": r.mean(),
                    "sd": r.std(ddof=1),
                    "median": r.median(),
                    "min": r.min(),
                    "max": r.max(),
                }
            )
        all_llm = trials[trials.dimension == dim].rating.astype(float)
        rows.append(
            {
                "source": "All LLMs (pooled)",
                "dimension": dim,
                "n": int(all_llm.size),
                "mean": all_llm.mean(),
                "sd": all_llm.std(ddof=1),
                "median": all_llm.median(),
                "min": all_llm.min(),
                "max": all_llm.max(),
            }
        )
    return pd.DataFrame(rows)


def per_image_means(trials: pd.DataFrame, humans: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for dim in DIMS:
        human_col = "Valence_mean" if dim == "valence" else "Arousal_mean"
        sub = trials[trials.dimension == dim]
        wide = (
            sub.groupby(["image_id", "model"]).rating.mean().unstack("model")
        )
        wide.columns = [short_name(c) for c in wide.columns]
        wide["LLM_mean"] = wide.mean(axis=1)
        wide = wide.merge(
            humans.set_index("image_id")[[human_col, "Category"]],
            left_index=True,
            right_index=True,
            how="left",
        ).rename(columns={human_col: "Human_mean"})
        out[dim] = wide.reset_index()
    return out


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def cohens_d_welch(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    pooled = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def run_tests(trials: pd.DataFrame, image_means: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_model_rows = []
    agg_rows = []
    for dim in DIMS:
        human_col = "Valence_mean" if dim == "valence" else "Arousal_mean"
        wide = image_means[dim].dropna(subset=["Human_mean"])
        models = [c for c in wide.columns if c not in {"image_id", "Category", "Human_mean", "LLM_mean"}]
        for m in models:
            paired = wide[[m, "Human_mean"]].dropna()
            t_p, p_p = stats.ttest_rel(paired[m], paired["Human_mean"])
            d_p = cohens_d_paired(paired[m].to_numpy(), paired["Human_mean"].to_numpy())
            r_pearson, _ = stats.pearsonr(paired[m], paired["Human_mean"])

            trial_vals = trials[(trials.dimension == dim) & (trials.model.str.endswith(m))].rating.astype(float).to_numpy()
            human_vals = wide["Human_mean"].dropna().to_numpy()
            t_w, p_w = stats.ttest_ind(trial_vals, human_vals, equal_var=False)
            d_w = cohens_d_welch(trial_vals, human_vals)
            per_model_rows.append(
                {
                    "model": m,
                    "dimension": dim,
                    "n_images": len(paired),
                    "llm_image_mean": paired[m].mean(),
                    "human_image_mean": paired["Human_mean"].mean(),
                    "mean_diff": (paired[m] - paired["Human_mean"]).mean(),
                    "paired_t": t_p,
                    "paired_p": p_p,
                    "paired_cohens_d": d_p,
                    "pearson_r": r_pearson,
                    "n_trials": len(trial_vals),
                    "welch_t": t_w,
                    "welch_p": p_w,
                    "welch_cohens_d": d_w,
                }
            )

        # Aggregate (all LLMs pooled)
        paired = wide[["LLM_mean", "Human_mean"]].dropna()
        t_p, p_p = stats.ttest_rel(paired["LLM_mean"], paired["Human_mean"])
        d_p = cohens_d_paired(paired["LLM_mean"].to_numpy(), paired["Human_mean"].to_numpy())
        r_pearson, _ = stats.pearsonr(paired["LLM_mean"], paired["Human_mean"])
        trial_vals = trials[trials.dimension == dim].rating.astype(float).to_numpy()
        human_vals = wide["Human_mean"].dropna().to_numpy()
        t_w, p_w = stats.ttest_ind(trial_vals, human_vals, equal_var=False)
        d_w = cohens_d_welch(trial_vals, human_vals)
        agg_rows.append(
            {
                "comparison": "All LLMs (per-image mean) vs Human",
                "dimension": dim,
                "n_images": len(paired),
                "llm_image_mean": paired["LLM_mean"].mean(),
                "human_image_mean": paired["Human_mean"].mean(),
                "mean_diff": (paired["LLM_mean"] - paired["Human_mean"]).mean(),
                "paired_t": t_p,
                "paired_p": p_p,
                "paired_cohens_d": d_p,
                "pearson_r": r_pearson,
                "n_trials": len(trial_vals),
                "welch_t": t_w,
                "welch_p": p_w,
                "welch_cohens_d": d_w,
            }
        )
    return pd.DataFrame(per_model_rows), pd.DataFrame(agg_rows)


def plot_distributions(trials: pd.DataFrame, humans: pd.DataFrame) -> None:
    trials = trials.copy()
    trials["model_short"] = trials.model.map(short_name)
    for dim in DIMS:
        human_col = "Valence_mean" if dim == "valence" else "Arousal_mean"
        sub = trials[trials.dimension == dim].copy()
        # Per-model violin + human overlay
        fig, ax = plt.subplots(figsize=(9, 5))
        order = sorted(sub.model_short.unique())
        sns.violinplot(
            data=sub, x="model_short", y="rating", order=order,
            inner="quartile", cut=0, ax=ax, color="#4C78A8",
        )
        h = humans[human_col].dropna()
        ax.axhline(h.mean(), color="crimson", linestyle="--", label=f"Human mean = {h.mean():.2f}")
        ax.fill_between(
            ax.get_xlim(), h.mean() - h.std(ddof=1), h.mean() + h.std(ddof=1),
            color="crimson", alpha=0.08, label="Human ± 1 SD",
        )
        ax.set_title(f"{dim.capitalize()} ratings: LLMs vs OASIS human mean (n={len(h)} images)")
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{dim.capitalize()} rating (1–7)")
        ax.legend(loc="best", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(PLOTS / f"violin_{dim}.png", dpi=160)
        plt.close(fig)

        # Aggregate histogram: pooled LLM trials vs human image means
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(sub.rating, bins=np.arange(0.5, 8.5, 1), alpha=0.55, label="All LLMs (trials)", color="#4C78A8", density=True)
        ax.hist(h, bins=20, alpha=0.55, label="Humans (image means)", color="crimson", density=True)
        ax.set_xlabel(f"{dim.capitalize()} rating")
        ax.set_ylabel("Density")
        ax.set_title(f"{dim.capitalize()}: pooled LLMs vs OASIS humans")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"hist_aggregate_{dim}.png", dpi=160)
        plt.close(fig)


def plot_per_image(image_means: dict[str, pd.DataFrame]) -> None:
    for dim, wide in image_means.items():
        models = [c for c in wide.columns if c not in {"image_id", "Category", "Human_mean", "LLM_mean"}]
        # Scatter LLM_mean vs Human_mean
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(wide["Human_mean"], wide["LLM_mean"], color="#4C78A8", alpha=0.85)
        lims = [1, 7]
        ax.plot(lims, lims, "k--", lw=1, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        r, p = stats.pearsonr(wide["Human_mean"], wide["LLM_mean"])
        ax.set_title(f"{dim.capitalize()} per-image means: All-LLM vs Human (r={r:.2f}, p={p:.1e})")
        ax.set_xlabel("Human mean (OASIS)")
        ax.set_ylabel("Pooled LLM mean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"scatter_aggregate_{dim}.png", dpi=160)
        plt.close(fig)

        # Per-model scatter grid
        n = len(models)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, m in zip(axes, models):
            ax.scatter(wide["Human_mean"], wide[m], color="#4C78A8", alpha=0.8)
            ax.plot(lims, lims, "k--", lw=1)
            r, _ = stats.pearsonr(wide["Human_mean"], wide[m])
            ax.set_title(f"{m} (r={r:.2f})")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("Human mean")
            ax.set_ylabel("Model mean")
        for ax in axes[len(models):]:
            ax.axis("off")
        fig.suptitle(f"{dim.capitalize()} per-image means by model", y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"scatter_per_model_{dim}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # Boxplot of (LLM - Human) image-level differences
        diffs = pd.DataFrame({m: wide[m] - wide["Human_mean"] for m in models + ["LLM_mean"]})
        diffs = diffs.rename(columns={"LLM_mean": "All LLMs"})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=diffs, ax=ax, color="#4C78A8")
        sns.stripplot(data=diffs, ax=ax, color="black", alpha=0.35, size=2.5)
        ax.axhline(0, color="crimson", linestyle="--")
        ax.set_ylabel("Model mean − Human mean (per image)")
        ax.set_title(f"{dim.capitalize()}: per-image bias by model")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(PLOTS / f"bias_box_{dim}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    print("Loading trials from DuckDB...")
    trials = load_trials()
    print(f"  trials: {len(trials)}  models: {trials.model.nunique()}  images: {trials.image_id.nunique()}")
    humans = load_humans()
    print(f"  humans: {len(humans)} OASIS images")

    desc = descriptives(trials, humans)
    desc.to_csv(OUT / "descriptives.csv", index=False)
    print(desc.to_string(index=False))

    image_means = per_image_means(trials, humans)
    for dim, wide in image_means.items():
        wide.to_csv(OUT / f"per_image_means_{dim}.csv", index=False)

    per_model, agg = run_tests(trials, image_means)
    per_model.to_csv(OUT / "ttests_per_model.csv", index=False)
    agg.to_csv(OUT / "ttests_aggregate.csv", index=False)
    print("\nPer-model t-tests:")
    print(per_model.to_string(index=False))
    print("\nAggregate (all-LLM) t-tests:")
    print(agg.to_string(index=False))

    plot_distributions(trials, humans)
    plot_per_image(image_means)
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
