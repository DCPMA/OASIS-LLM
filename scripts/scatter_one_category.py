"""Scatter plots restricted to a single OASIS category.

Each dot is one (image × model) per-image mean.
Pass the category name as the first CLI arg (Animal / Scene / Person / Object).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "analysis" / "llm_vs_human_uniform40"
OUT = IN / "plots"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY_COLOR = {
    "Animal": "#D62728",
    "Scene":  "#2CA02C",
    "Person": "#1F77B4",
    "Object": "#E6A700",
}
MODEL_COLS = [
    "claude-sonnet-4.6",
    "gemma-4-31b-it",
    "gpt-5.4",
    "qwen3.6-plus",
    "grok-4.20",
]


def main() -> None:
    category = sys.argv[1] if len(sys.argv) > 1 else "Animal"
    color = CATEGORY_COLOR[category]
    slug = category.lower()

    sns.set_theme(style="whitegrid", context="paper")

    for dim in ("valence", "arousal"):
        wide = pd.read_csv(IN / f"per_image_means_{dim}.csv")
        wide = wide[wide["Category"] == category]
        models = [c for c in MODEL_COLS if c in wide.columns]
        long = wide.melt(
            id_vars=["image_id", "Category", "Human_mean"],
            value_vars=models,
            var_name="model",
            value_name="model_rating",
        ).dropna(subset=["model_rating", "Human_mean"])

        fig, ax = plt.subplots(figsize=(7.0, 6.5))
        r, p = stats.pearsonr(long["model_rating"], long["Human_mean"])
        ax.scatter(
            long["model_rating"], long["Human_mean"],
            s=48, alpha=0.78, color=color,
            edgecolor="white", linewidth=0.6,
            label=f"{category}  (r={r:.2f}, p={p:.1e}, n={len(long)})",
        )
        # annotate each animal image once near its centroid for readability
        centroids = long.groupby("image_id")[["model_rating", "Human_mean"]].mean()
        for img, row in centroids.iterrows():
            ax.annotate(
                img, (row["model_rating"], row["Human_mean"]),
                fontsize=7, alpha=0.65, xytext=(4, 3), textcoords="offset points",
            )

        lims = [1, 7]
        ax.plot(lims, lims, "k--", lw=1, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"LLM {dim} rating (per-image mean)")
        ax.set_ylabel(f"Human {dim} rating (OASIS image mean)")
        ax.set_title(
            f"{category} images only — per-image {dim}: each dot = (image × model)\n"
            f"{wide['image_id'].nunique()} {slug} images × {len(models)} models"
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
        fig.tight_layout()
        out_path = OUT / f"scatter_{slug}_{dim}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
