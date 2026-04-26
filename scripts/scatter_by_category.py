"""Scatter plots: per-image LLM rating vs Human rating, colored by OASIS category.

Each dot is one (image × model) pair. Color encodes the image's OASIS Category:
  Animal -> red, Scene -> green, Person -> blue, Object -> deep yellow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "analysis" / "llm_vs_human_uniform40"
OUT = IN / "plots"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY_PALETTE = {
    "Animal": "#D62728",  # red
    "Scene":  "#2CA02C",  # green
    "Person": "#1F77B4",  # blue
    "Object": "#E6A700",  # deep yellow
}

MODEL_COLS = [
    "claude-sonnet-4.6",
    "gemma-4-31b-it",
    "gpt-5.4",
    "qwen3.6-plus",
    "grok-4.20",
]


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")

    for dim in ("valence", "arousal"):
        wide = pd.read_csv(IN / f"per_image_means_{dim}.csv")
        models = [c for c in MODEL_COLS if c in wide.columns]
        long = wide.melt(
            id_vars=["image_id", "Category", "Human_mean"],
            value_vars=models,
            var_name="model",
            value_name="model_rating",
        ).dropna(subset=["model_rating", "Human_mean", "Category"])

        fig, ax = plt.subplots(figsize=(7.5, 7))
        for cat, color in CATEGORY_PALETTE.items():
            sub = long[long.Category == cat]
            if sub.empty:
                continue
            r, _ = stats.pearsonr(sub["model_rating"], sub["Human_mean"])
            ax.scatter(
                sub["model_rating"], sub["Human_mean"],
                s=44, alpha=0.78, color=color, edgecolor="white", linewidth=0.6,
                label=f"{cat}  (r={r:.2f}, n={len(sub)})",
            )
        lims = [1, 7]
        ax.plot(lims, lims, "k--", lw=1, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"LLM {dim} rating (per-image mean)")
        ax.set_ylabel(f"Human {dim} rating (OASIS image mean)")
        ax.set_title(
            f"Per-image {dim} ratings by OASIS category\n"
            f"each dot = one (image × model) pair, n={len(long)} across {len(models)} models"
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92, title="Category")
        fig.tight_layout()
        out_path = OUT / f"scatter_by_category_{dim}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
