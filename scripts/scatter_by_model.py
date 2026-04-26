"""Scatter plots: per-image LLM rating vs Human rating, colored by model.

Reads the per-image means produced by `llm_vs_human_ttest.py` and creates
two figures (valence, arousal) where each dot is one (image, model) pair.
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

DIM_HUMAN = {"valence": "Valence_mean", "arousal": "Arousal_mean"}

PALETTE = {
    "claude-sonnet-4.6": "#E45756",
    "gemma-4-31b-it":    "#54A24B",
    "gpt-5.4":           "#4C78A8",
    "qwen3.6-plus":      "#F58518",
    "grok-4.20":         "#9467BD",
}


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")

    for dim in ("valence", "arousal"):
        wide = pd.read_csv(IN / f"per_image_means_{dim}.csv")
        models = [c for c in PALETTE if c in wide.columns]
        long = wide.melt(
            id_vars=["image_id", "Category", "Human_mean"],
            value_vars=models,
            var_name="model",
            value_name="model_rating",
        ).dropna(subset=["model_rating", "Human_mean"])

        fig, ax = plt.subplots(figsize=(7.5, 7))
        for m in models:
            sub = long[long.model == m]
            r, _ = stats.pearsonr(sub["model_rating"], sub["Human_mean"])
            ax.scatter(
                sub["model_rating"], sub["Human_mean"],
                s=42, alpha=0.8, color=PALETTE[m], edgecolor="white", linewidth=0.6,
                label=f"{m}  (r={r:.2f}, n={len(sub)})",
            )
        lims = [1, 7]
        ax.plot(lims, lims, "k--", lw=1, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"LLM {dim} rating (per-image mean)")
        ax.set_ylabel(f"Human {dim} rating (OASIS image mean)")
        ax.set_title(f"Per-image {dim} ratings: each dot = one (image × model) pair")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        out_path = OUT / f"scatter_by_model_{dim}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
