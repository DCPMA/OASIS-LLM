"""Paper-style summaries and plots for OASIS-LLM runs."""
from __future__ import annotations

import json
from pathlib import Path
import random

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .db import DB_PATH
from .images import LONG_CSV


CATEGORY_COLORS = {
    "Animal": "seagreen",
    "Object": "royalblue",
    "Person": "palevioletred",
    "Scene": "goldenrod",
}


def export_paper_plots(run_id: str, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_summary = load_llm_image_summary(run_id)
    if llm_summary.empty:
        raise ValueError(f"No completed trials found for run '{run_id}'.")

    human_summary = load_human_image_summary()
    comparison = llm_summary.merge(
        human_summary,
        on=["image_id", "category"],
        how="left",
        validate="one_to_one",
    )

    llm_summary.to_csv(out_dir / "llm_image_summary.csv", index=False)
    comparison.to_csv(out_dir / "human_subset_comparison.csv", index=False)

    _plot_distributions(llm_summary, out_dir / "figure2_distributions.png")
    _plot_mean_sd_relationships(llm_summary, out_dir / "figure3_mean_sd.png")
    _plot_valence_arousal(llm_summary, out_dir / "figure4_valence_arousal.png")
    _plot_valence_arousal_by_category(llm_summary, out_dir / "figure5_by_category.png")
    _plot_human_overlap(comparison, out_dir / "human_overlap.png")

    summary = {
        "run_id": run_id,
        "image_count": int(len(llm_summary)),
        "valence_samples_per_image": _format_unique_ints(llm_summary["valence_n"]),
        "arousal_samples_per_image": _format_unique_ints(llm_summary["arousal_n"]),
        "valence_arousal_correlation": _format_float(
            _safe_corr(llm_summary, "valence_mean", "arousal_mean")
        ),
        "human_valence_correlation": _format_float(
            _safe_corr(comparison, "valence_mean", "human_valence_mean")
        ),
        "human_arousal_correlation": _format_float(
            _safe_corr(comparison, "arousal_mean", "human_arousal_mean")
        ),
    }
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2)
    return summary


def export_participant_style_dataset(
    run_id: str,
    out_dir: Path,
    *,
    images_per_participant: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_llm_trials(run_id)
    if trials.empty:
        raise ValueError(f"No completed trials found for run '{run_id}'.")
    run_meta = load_run_metadata(run_id)

    selected_images = _select_images_for_wide_export(
        trials[["image_id", "category"]].drop_duplicates(),
        n=images_per_participant,
        seed=seed,
    )
    selected_trials = trials[trials["image_id"].isin(selected_images)].copy()

    attempt_df = _build_attempt_wide(selected_trials, selected_images, run_meta)
    attempt_df.to_csv(out_dir / "attempt_wide.csv", index=False)

    legacy_wide = out_dir / "participant_wide.csv"
    legacy_summary = out_dir / "participant_wide_summary.json"
    if legacy_wide.exists():
        legacy_wide.unlink()
    if legacy_summary.exists():
        legacy_summary.unlink()

    distribution_counts = _build_distribution_counts(selected_trials)
    distribution_counts.to_csv(out_dir / "distribution_counts.csv", index=False)

    pd.DataFrame({"image_id": selected_images}).to_csv(
        out_dir / "selected_images.csv", index=False
    )

    _plot_distribution_grid(
        selected_trials,
        selected_images,
        dimension="valence",
        out_path=out_dir / "valence_distributions.png",
    )
    _plot_distribution_grid(
        selected_trials,
        selected_images,
        dimension="arousal",
        out_path=out_dir / "arousal_distributions.png",
    )
    _write_attempt_dataset_brief(
        out_path=out_dir / "implementation_brief.md",
        run_meta=run_meta,
        selected_images=selected_images,
        attempt_count=int(attempt_df.shape[0]),
        images_per_attempt=images_per_participant,
    )

    summary = {
        "run_id": run_id,
        "provider": run_meta["provider"],
        "model_id": run_meta["model_id"],
        "attempt_count": int(attempt_df.shape[0]),
        "images_per_attempt": int(images_per_participant),
        "response_columns": int(attempt_df.shape[1] - 5),
        "selected_images": selected_images,
        "note": "Rows are reconstructed from sample_idx because this run stores independent trials, not real participant sessions.",
    }
    pd.Series(summary).to_json(out_dir / "attempt_wide_summary.json", indent=2)
    return summary


def load_llm_image_summary(run_id: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    query = """
    WITH per_dimension AS (
        SELECT
            image_id,
            dimension,
            avg(rating) AS mean_rating,
            stddev_samp(rating) AS sd_rating,
            count(*) AS n
        FROM trials
        WHERE run_id = ? AND status = 'done'
        GROUP BY 1, 2
    ),
    wide AS (
        SELECT
            image_id,
            max(CASE WHEN dimension = 'valence' THEN mean_rating END) AS valence_mean,
            max(CASE WHEN dimension = 'valence' THEN sd_rating END) AS valence_sd,
            max(CASE WHEN dimension = 'valence' THEN n END) AS valence_n,
            max(CASE WHEN dimension = 'arousal' THEN mean_rating END) AS arousal_mean,
            max(CASE WHEN dimension = 'arousal' THEN sd_rating END) AS arousal_sd,
            max(CASE WHEN dimension = 'arousal' THEN n END) AS arousal_n
        FROM per_dimension
        GROUP BY 1
    ),
    meta AS (
        SELECT DISTINCT theme AS image_id, category
        FROM read_csv_auto(?)
    )
    SELECT
        wide.image_id,
        coalesce(meta.category, 'Unknown') AS category,
        wide.valence_mean,
        wide.valence_sd,
        wide.valence_n,
        wide.arousal_mean,
        wide.arousal_sd,
        wide.arousal_n
    FROM wide
    LEFT JOIN meta USING (image_id)
    ORDER BY wide.image_id
    """
    return con.execute(query, [run_id, str(LONG_CSV)]).fetchdf()


def load_llm_trials(run_id: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    query = """
    WITH meta AS (
        SELECT DISTINCT theme AS image_id, category
        FROM read_csv_auto(?)
    )
    SELECT
        t.run_id,
        t.image_id,
        t.dimension,
        t.sample_idx,
        t.rating,
        coalesce(meta.category, 'Unknown') AS category
    FROM trials t
    LEFT JOIN meta USING (image_id)
    WHERE t.run_id = ? AND t.status = 'done'
    ORDER BY t.sample_idx, t.image_id, t.dimension
    """
    return con.execute(query, [str(LONG_CSV), run_id]).fetchdf()


def load_run_metadata(run_id: str) -> dict[str, str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    row = con.execute(
        "SELECT config_json FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Run '{run_id}' not found in runs table.")
    config = json.loads(row[0])
    return {
        "run_id": run_id,
        "provider": str(config.get("provider", "unknown")),
        "model_id": str(config.get("model", "unknown")),
    }


def load_human_image_summary() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    query = """
    WITH per_dimension AS (
        SELECT
            theme AS image_id,
            category,
            lower(valar) AS dimension,
            avg(try_cast(rating AS DOUBLE)) AS mean_rating,
            stddev_samp(try_cast(rating AS DOUBLE)) AS sd_rating,
            count(try_cast(rating AS DOUBLE)) AS n
        FROM read_csv_auto(?)
        GROUP BY 1, 2, 3
    )
    SELECT
        image_id,
        category,
        max(CASE WHEN dimension = 'valence' THEN mean_rating END) AS human_valence_mean,
        max(CASE WHEN dimension = 'valence' THEN sd_rating END) AS human_valence_sd,
        max(CASE WHEN dimension = 'valence' THEN n END) AS human_valence_n,
        max(CASE WHEN dimension = 'arousal' THEN mean_rating END) AS human_arousal_mean,
        max(CASE WHEN dimension = 'arousal' THEN sd_rating END) AS human_arousal_sd,
        max(CASE WHEN dimension = 'arousal' THEN n END) AS human_arousal_n
    FROM per_dimension
    GROUP BY 1, 2
    ORDER BY 1
    """
    return con.execute(query, [str(LONG_CSV)]).fetchdf()


def _plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plots = [
        ("valence_mean", "Mean valence rating", "Valence ratings by image", (1, 7), np.arange(1, 7.01, 0.2)),
        ("arousal_mean", "Mean arousal rating", "Arousal ratings by image", (1, 7), np.arange(1, 7.01, 0.2)),
        ("valence_sd", "SD of valence rating", "SD of valence ratings by image", (0, 3), np.arange(0, 3.01, 0.1)),
        ("arousal_sd", "SD of arousal rating", "SD of arousal ratings by image", (0, 3), np.arange(0, 3.01, 0.1)),
    ]
    for ax, (column, xlabel, title, xlim, bins) in zip(axes.flat, plots, strict=True):
        values = df[column].dropna()
        if values.empty:
            _mark_missing(ax, title)
            continue
        ax.hist(values, bins=bins, color="lightblue", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_sd_relationships(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = [
        (axes[0], "valence_mean", "valence_sd", "Valence mean", "Valence SD", "Valence SD by valence mean", 3),
        (axes[1], "arousal_mean", "arousal_sd", "Arousal mean", "Arousal SD", "Arousal SD by arousal mean", 2),
    ]
    for ax, x_col, y_col, xlabel, ylabel, title, degree in configs:
        plot_df = df[[x_col, y_col]].dropna()
        if plot_df.empty:
            _mark_missing(ax, title)
            continue
        ax.scatter(plot_df[x_col], plot_df[y_col], color="lightblue")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 7)
        ax.set_ylim(0, 2.5)
        _draw_polyfit(ax, plot_df[x_col], plot_df[y_col], degree=degree, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_valence_arousal(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_df = df[["valence_mean", "arousal_mean"]].dropna()
    if plot_df.empty:
        _mark_missing(ax, "Mean valence and arousal ratings by image")
    else:
        ax.scatter(plot_df["valence_mean"], plot_df["arousal_mean"], color="lightblue")
        ax.set_title("Mean valence and arousal ratings by image")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.axhline(4, linestyle="--", color="grey", linewidth=1)
        ax.axvline(4, linestyle="--", color="grey", linewidth=1)
        corr = _safe_corr(plot_df, "valence_mean", "arousal_mean")
        if corr is not None:
            ax.text(1.2, 6.5, f"R = {corr:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_valence_arousal_by_category(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_df = df[["valence_mean", "arousal_mean", "category"]].dropna()
    if plot_df.empty:
        _mark_missing(ax, "Mean valence and arousal ratings by image category")
    else:
        for category, cat_df in plot_df.groupby("category"):
            color = CATEGORY_COLORS.get(category, "slategray")
            ax.scatter(
                cat_df["valence_mean"],
                cat_df["arousal_mean"],
                color=color,
                label=category,
            )
        ax.set_title("Mean valence and arousal ratings by image category")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.axhline(4, linestyle="--", color="grey", linewidth=1)
        ax.axvline(4, linestyle="--", color="grey", linewidth=1)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_human_overlap(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = [
        (axes[0], "valence_mean", "human_valence_mean", "LLM valence mean", "Human valence mean", "LLM vs human valence"),
        (axes[1], "arousal_mean", "human_arousal_mean", "LLM arousal mean", "Human arousal mean", "LLM vs human arousal"),
    ]
    for ax, x_col, y_col, xlabel, ylabel, title in configs:
        plot_df = df[[x_col, y_col, "category"]].dropna()
        if plot_df.empty:
            _mark_missing(ax, title)
            continue
        for category, cat_df in plot_df.groupby("category"):
            color = CATEGORY_COLORS.get(category, "slategray")
            ax.scatter(cat_df[x_col], cat_df[y_col], color=color, label=category)
        corr = _safe_corr(plot_df, x_col, y_col)
        if corr is not None:
            ax.text(plot_df[x_col].min(), plot_df[y_col].max(), f"R = {corr:.3f}")
        _draw_identity_line(ax, plot_df[x_col], plot_df[y_col])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_attempt_wide(
    trials: pd.DataFrame,
    selected_images: list[str],
    run_meta: dict[str, str],
) -> pd.DataFrame:
    frame = trials.copy()
    frame["attempt_id"] = frame["sample_idx"].map(
        lambda idx: f"{run_meta['run_id']}_attempt_{int(idx) + 1:02d}"
    )
    frame["column_name"] = frame["image_id"] + "_" + frame["dimension"]
    wide = frame.pivot(
        index="attempt_id",
        columns="column_name",
        values="rating",
    ).reset_index()
    wide.insert(0, "sample_idx", wide["attempt_id"].map(
        lambda value: int(str(value).rsplit("_", 1)[-1]) - 1
    ))
    wide.insert(0, "model_id", run_meta["model_id"])
    wide.insert(0, "provider", run_meta["provider"])
    wide.insert(0, "run_id", run_meta["run_id"])

    ordered_columns = [
        "run_id",
        "provider",
        "model_id",
        "attempt_id",
        "sample_idx",
    ] + [
        f"{image_id}_{dimension}"
        for image_id in selected_images
        for dimension in ("valence", "arousal")
    ]
    return wide.reindex(columns=ordered_columns)


def _write_attempt_dataset_brief(
    *,
    out_path: Path,
    run_meta: dict[str, str],
    selected_images: list[str],
    attempt_count: int,
    images_per_attempt: int,
) -> None:
    response_columns = images_per_attempt * 2
    image_list = "\n".join(f"- {image_id}" for image_id in selected_images)
    out_path.write_text(
        "# Attempt-Style Dataset Brief\n\n"
        "## Goal\n\n"
        "Export a stateless LLM run into a wide CSV where each row corresponds to one decoding attempt (`sample_idx`) instead of a human participant session.\n\n"
        "## Source Tables\n\n"
        "- `runs`: provides `run_id` and `config_json` for `provider` and `model_id`.\n"
        "- `trials`: provides one row per `(run_id, image_id, dimension, sample_idx)` with the numeric `rating`.\n"
        "- `data/derived/OASIS_data_long.csv`: provides image-to-category metadata; not required for the wide CSV itself, but required for the distribution outputs.\n\n"
        "## Row Semantics\n\n"
        f"- One row per `sample_idx` for run `{run_meta['run_id']}`.\n"
        f"- There are `{attempt_count}` attempt rows in the current export.\n"
        "- These rows are not real participants; they are repeated independent model calls grouped by `sample_idx`.\n\n"
        "## Required Output Columns\n\n"
        "Metadata columns:\n"
        "- `run_id`\n"
        "- `provider`\n"
        "- `model_id`\n"
        "- `attempt_id`\n"
        "- `sample_idx`\n\n"
        f"Response columns: `{images_per_attempt}` images x 2 dimensions = `{response_columns}` columns.\n"
        "Each response column must be named exactly `picture name_valence` or `picture name_arousal`.\n\n"
        "## Selected Images In This Export\n\n"
        f"{image_list}\n\n"
        "## Implementation Steps\n\n"
        "1. Filter `trials` to one completed `run_id`.\n"
        "2. Pick the fixed image subset used for the export.\n"
        "3. Pivot ratings wide on `(sample_idx, image_id, dimension)`.\n"
        "4. Prepend the metadata columns from `runs.config_json`.\n"
        "5. Order response columns as `image_1_valence`, `image_1_arousal`, `image_2_valence`, `image_2_arousal`, ...\n"
        "6. Separately count rating frequencies per `(image_id, dimension, rating)` to drive the distribution plots.\n\n"
        "## Acceptance Criteria\n\n"
        f"- CSV has `{attempt_count}` rows.\n"
        f"- CSV has `{response_columns}` response columns plus the 5 metadata columns.\n"
        "- No `participant_id` column appears anywhere in the export.\n"
        "- Distribution plots are produced separately for valence and arousal.\n",
        encoding="utf-8",
    )


def _build_distribution_counts(trials: pd.DataFrame) -> pd.DataFrame:
    counts = (
        trials.groupby(["image_id", "category", "dimension", "rating"])
        .size()
        .reset_index(name="count")
        .sort_values(["dimension", "image_id", "rating"])
    )
    return counts


def _plot_distribution_grid(
    trials: pd.DataFrame,
    selected_images: list[str],
    *,
    dimension: str,
    out_path: Path,
) -> None:
    dim_trials = trials[trials["dimension"] == dimension]
    n_images = len(selected_images)
    n_cols = 5
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.8 * n_rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, image_id in zip(axes_array, selected_images, strict=False):
        image_trials = dim_trials[dim_trials["image_id"] == image_id]
        counts = (
            image_trials["rating"].value_counts().reindex(range(1, 8), fill_value=0).sort_index()
        )
        ax.bar(counts.index.astype(str), counts.values, color="lightblue", edgecolor="white")
        ax.set_title(image_id, fontsize=10)
        ax.set_ylim(0, max(1, counts.max()))
    for ax in axes_array[n_images:]:
        ax.axis("off")
    for ax in axes_array:
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
    fig.suptitle(f"Per-picture {dimension} distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _select_images_for_wide_export(
    image_meta: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> list[str]:
    unique_meta = image_meta.drop_duplicates().copy()
    if len(unique_meta) < n:
        raise ValueError(f"Run only has {len(unique_meta)} images; cannot export {n} images per participant.")
    rng = random.Random(seed)
    by_category: dict[str, list[str]] = {}
    for row in unique_meta.itertuples(index=False):
        by_category.setdefault(str(row.category), []).append(str(row.image_id))
    categories = sorted(by_category)
    total = sum(len(by_category[category]) for category in categories)
    allocation = {
        category: max(1, round(n * len(by_category[category]) / total))
        for category in categories
    }
    while sum(allocation.values()) > n:
        reducible = [category for category in categories if allocation[category] > 1]
        target = max(reducible or categories, key=lambda category: allocation[category])
        allocation[target] -= 1
    while sum(allocation.values()) < n:
        target = min(categories, key=lambda category: allocation[category])
        allocation[target] += 1

    selected: list[str] = []
    for category in categories:
        pool = sorted(by_category[category])
        rng.shuffle(pool)
        selected.extend(sorted(pool[: allocation[category]]))
    return sorted(selected)


def _draw_polyfit(
    ax: plt.Axes,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    degree: int,
    color: str,
) -> None:
    if len(x_values) < degree + 1 or x_values.nunique() <= degree:
        return
    coeffs = np.polyfit(x_values.to_numpy(), y_values.to_numpy(), degree)
    x_line = np.linspace(x_values.min(), x_values.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    y_hat = np.polyval(coeffs, x_values.to_numpy())
    r_squared = _r_squared(y_values.to_numpy(), y_hat)
    ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
    ax.text(1.2, 2.25, f"R² = {r_squared:.3f}")


def _draw_identity_line(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    lower = min(x_values.min(), y_values.min())
    upper = max(x_values.max(), y_values.max())
    ax.plot([lower, upper], [lower, upper], color="grey", linestyle="--", linewidth=1)


def _mark_missing(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _r_squared(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    residual = ((y_true - y_hat) ** 2).sum()
    total = ((y_true - y_true.mean()) ** 2).sum()
    if total == 0:
        return 1.0
    return 1 - residual / total


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float | None:
    corr_df = df[[left, right]].dropna()
    if len(corr_df) < 2:
        return None
    value = corr_df[left].corr(corr_df[right])
    if pd.isna(value):
        return None
    return float(value)


def _format_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3f}"


def _format_unique_ints(series: pd.Series) -> str:
    values = sorted({int(v) for v in series.dropna().tolist()})
    return ", ".join(str(v) for v in values) if values else "-"