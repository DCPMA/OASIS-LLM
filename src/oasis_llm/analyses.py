"""Analysis entity: aggregate ratings across multiple runs that share a dataset.

An Analysis is a researcher-curated bundle of run_ids, all of which must rate
the same dataset_id. It exposes per-image aggregates (mean, SD, agreement) and
cross-run comparisons.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

import duckdb
import pandas as pd


@dataclass
class Analysis:
    analysis_id: str
    name: str
    description: str | None
    dataset_id: str
    created_at: datetime | None
    run_ids: list[str]


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", s).strip("-") or uuid.uuid4().hex[:8]


def _run_dataset(con, run_id: str) -> str | None:
    """Look up which dataset a run was executed against (from runs.config_json)."""
    row = con.execute(
        "SELECT config_json FROM runs WHERE run_id=?", [run_id]
    ).fetchone()
    if not row:
        return None
    import json
    try:
        return json.loads(row[0]).get("image_set")
    except Exception:
        return None


def create(con, name: str, dataset_id: str, *, description: str | None = None) -> str:
    aid = _slug(name)
    if con.execute("SELECT 1 FROM analyses WHERE analysis_id=?", [aid]).fetchone():
        aid = f"{aid}-{uuid.uuid4().hex[:6]}"
    con.execute(
        "INSERT INTO analyses (analysis_id, name, description, dataset_id) VALUES (?, ?, ?, ?)",
        [aid, name, description, dataset_id],
    )
    return aid


def list_all(con) -> list[Analysis]:
    rows = con.execute(
        """
        SELECT a.analysis_id, a.name, a.description, a.dataset_id, a.created_at
        FROM analyses a ORDER BY a.created_at DESC NULLS LAST, a.analysis_id
        """
    ).fetchall()
    return [_load(con, *r) for r in rows]


def get(con, analysis_id: str) -> Analysis | None:
    row = con.execute(
        """
        SELECT analysis_id, name, description, dataset_id, created_at
        FROM analyses WHERE analysis_id=?
        """, [analysis_id]
    ).fetchone()
    if row is None:
        return None
    return _load(con, *row)


def _load(con, aid, name, desc, dataset_id, created_at) -> Analysis:
    runs = [r[0] for r in con.execute(
        "SELECT run_id FROM analysis_runs WHERE analysis_id=? ORDER BY added_at",
        [aid],
    ).fetchall()]
    return Analysis(
        analysis_id=aid, name=name, description=desc, dataset_id=dataset_id,
        created_at=created_at, run_ids=runs,
    )


def add_run(con, analysis_id: str, run_id: str, *, label: str | None = None) -> None:
    a = get(con, analysis_id)
    if a is None:
        raise KeyError(analysis_id)
    rds = _run_dataset(con, run_id)
    if rds is None:
        raise KeyError(f"unknown run: {run_id}")
    if rds != a.dataset_id:
        raise ValueError(
            f"run {run_id} was executed against dataset '{rds}', "
            f"but this analysis is bound to '{a.dataset_id}'"
        )
    con.execute(
        "INSERT OR IGNORE INTO analysis_runs (analysis_id, run_id, label) VALUES (?, ?, ?)",
        [analysis_id, run_id, label],
    )


def remove_run(con, analysis_id: str, run_id: str) -> None:
    con.execute(
        "DELETE FROM analysis_runs WHERE analysis_id=? AND run_id=?",
        [analysis_id, run_id],
    )


def delete(con, analysis_id: str) -> None:
    con.execute("DELETE FROM analysis_runs WHERE analysis_id=?", [analysis_id])
    con.execute("DELETE FROM analyses WHERE analysis_id=?", [analysis_id])


def eligible_runs(con, dataset_id: str) -> list[dict]:
    """Return runs whose config.image_set == dataset_id (i.e. addable to an analysis)."""
    out = []
    for row in con.execute(
        "SELECT run_id, status, config_json, created_at FROM runs ORDER BY created_at DESC NULLS LAST"
    ).fetchall():
        rid, status, cfg_json, created_at = row
        import json
        try:
            cfg = json.loads(cfg_json)
        except Exception:
            cfg = {}
        if cfg.get("image_set") != dataset_id:
            continue
        out.append({
            "run_id": rid, "status": status,
            "model": cfg.get("model"), "provider": cfg.get("provider"),
            "temperature": cfg.get("temperature"),
            "samples_per_image": cfg.get("samples_per_image"),
            "created_at": created_at,
        })
    return out


def per_image_aggregate(con, analysis_id: str) -> pd.DataFrame:
    """Long-format DataFrame: one row per (run_id, image_id, dimension) with mean/SD/n."""
    a = get(con, analysis_id)
    if a is None or not a.run_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(a.run_ids))
    df = con.execute(
        f"""
        SELECT run_id, image_id, dimension,
               avg(rating)  AS mean_rating,
               stddev_samp(rating) AS sd_rating,
               count(*)     AS n
        FROM trials
        WHERE run_id IN ({placeholders}) AND status='done'
        GROUP BY run_id, image_id, dimension
        ORDER BY image_id, dimension, run_id
        """,
        a.run_ids,
    ).fetchdf()
    return df


def cross_run_correlations(con, analysis_id: str) -> dict:
    """Pairwise correlations of per-image mean ratings between runs, per dimension."""
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for dim, sub in df.groupby("dimension"):
        wide = sub.pivot(index="image_id", columns="run_id", values="mean_rating")
        out[dim] = wide.corr()
    return out


def vs_human_norms(con, analysis_id: str, norms_csv: str = "OASIS/OASIS.csv") -> pd.DataFrame:
    """Join per-run-per-image means with human valence/arousal norms."""
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return pd.DataFrame()
    norms = duckdb.connect(":memory:").execute(
        f"""
        SELECT "Theme" AS image_id,
               "Valence_mean" AS human_valence,
               "Arousal_mean" AS human_arousal
        FROM read_csv_auto('{norms_csv}', header=true)
        """
    ).fetchdf()
    merged = df.merge(norms, on="image_id", how="left")
    merged["human_value"] = merged.apply(
        lambda r: r["human_valence"] if r["dimension"] == "valence" else r["human_arousal"],
        axis=1,
    )
    merged["delta"] = merged["mean_rating"] - merged["human_value"]
    return merged


# ─── leaderboard ───────────────────────────────────────────────────────────
def leaderboard(
    con,
    norms_csv: str = "OASIS/OASIS.csv",
    *,
    min_images: int = 5,
) -> pd.DataFrame:
    """Score every run in the DB against OASIS human norms.

    Returns one row per ``(run_id, dimension)`` with columns:
    ``run_id, model, dimension, n_images, pearson_r, spearman_rho, mae, rmse,
    mean_pred, mean_human``.

    Only runs with at least ``min_images`` rated images per dimension and a
    matching human norm column (valence or arousal) appear.
    """
    import json
    import numpy as np

    # Per-run-per-image means
    df = con.execute(
        """
        SELECT t.run_id, t.image_id, t.dimension, avg(t.rating) AS mean_rating,
               count(*) AS n
        FROM trials t
        WHERE t.status='done'
        GROUP BY t.run_id, t.image_id, t.dimension
        """
    ).fetchdf()
    if df.empty:
        return pd.DataFrame()

    norms = duckdb.connect(":memory:").execute(
        f"""
        SELECT "Theme" AS image_id,
               "Valence_mean" AS human_valence,
               "Arousal_mean" AS human_arousal
        FROM read_csv_auto('{norms_csv}', header=true)
        """
    ).fetchdf()
    merged = df.merge(norms, on="image_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["human_value"] = merged.apply(
        lambda r: r["human_valence"] if r["dimension"] == "valence"
        else r["human_arousal"] if r["dimension"] == "arousal"
        else None,
        axis=1,
    )
    merged = merged.dropna(subset=["human_value"])

    # Run → model lookup
    run_meta = {}
    for rid, cfg_json in con.execute(
        "SELECT run_id, config_json FROM runs"
    ).fetchall():
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        run_meta[rid] = {
            "model": cfg.get("model"),
            "provider": cfg.get("provider"),
            "temperature": cfg.get("temperature"),
        }

    rows = []
    for (run_id, dim), sub in merged.groupby(["run_id", "dimension"]):
        if len(sub) < min_images:
            continue
        pred = sub["mean_rating"].astype(float).values
        human = sub["human_value"].astype(float).values
        # Pearson
        if pred.std() == 0 or human.std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(pred, human)[0, 1])
        # Spearman via pandas rank correlation (no scipy dependency)
        try:
            rho = float(
                pd.Series(pred).corr(pd.Series(human), method="spearman")
            )
        except Exception:
            rho = float("nan")
        mae = float(np.abs(pred - human).mean())
        rmse = float(np.sqrt(((pred - human) ** 2).mean()))
        meta = run_meta.get(run_id, {})
        rows.append({
            "run_id": run_id,
            "model": meta.get("model"),
            "provider": meta.get("provider"),
            "temperature": meta.get("temperature"),
            "dimension": dim,
            "n_images": int(len(sub)),
            "pearson_r": r,
            "spearman_rho": rho,
            "mae": mae,
            "rmse": rmse,
            "mean_pred": float(pred.mean()),
            "mean_human": float(human.mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["dimension", "pearson_r"], ascending=[True, False]).reset_index(drop=True)
    return out


# ─── multi-run analytics ────────────────────────────────────────────────────
def model_pair_deltas(con, analysis_id: str) -> pd.DataFrame:
    """Per-image, per-dimension, per-pair rating deltas.

    Returns long format with columns:
    ``run_a, run_b, dimension, n_overlap, mean_delta, abs_mean_delta,
    max_abs_delta, sd_delta``.

    ``delta = mean_rating[run_a] - mean_rating[run_b]`` averaged over images
    where both runs have a rating.
    """
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for dim, sub in df.groupby("dimension"):
        wide = sub.pivot(index="image_id", columns="run_id", values="mean_rating")
        run_ids = list(wide.columns)
        for i, ra in enumerate(run_ids):
            for rb in run_ids[i + 1:]:
                paired = wide[[ra, rb]].dropna()
                if paired.empty:
                    continue
                d = paired[ra] - paired[rb]
                rows.append({
                    "run_a": ra, "run_b": rb, "dimension": dim,
                    "n_overlap": int(len(d)),
                    "mean_delta": float(d.mean()),
                    "abs_mean_delta": float(d.abs().mean()),
                    "max_abs_delta": float(d.abs().max()),
                    "sd_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                })
    return pd.DataFrame(rows)


def icc_across_runs(con, analysis_id: str) -> pd.DataFrame:
    """Intraclass correlation across runs as raters.

    Returns one row per dimension with columns:
    ``dimension, n_images, k_runs, icc2_1, icc3_1``.

    - ``icc2_1`` = ICC(2,1), two-way random effects, single rater, absolute
      agreement. Use when runs are a sample from a population of models.
    - ``icc3_1`` = ICC(3,1), two-way mixed effects, single rater, consistency.
      Use when the specific runs/models are the only ones of interest.

    Implementation follows Shrout & Fleiss (1979) / McGraw & Wong (1996).
    Returns NaN where there are <2 runs or <2 images with full overlap.
    """
    import numpy as np
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for dim, sub in df.groupby("dimension"):
        wide = sub.pivot(index="image_id", columns="run_id", values="mean_rating").dropna()
        n, k = wide.shape  # n = images, k = raters/runs
        if n < 2 or k < 2:
            rows.append({
                "dimension": dim, "n_images": int(n), "k_runs": int(k),
                "icc2_1": float("nan"), "icc3_1": float("nan"),
            })
            continue
        x = wide.values.astype(float)
        grand = x.mean()
        row_means = x.mean(axis=1)  # per-image
        col_means = x.mean(axis=0)  # per-run
        ss_total = ((x - grand) ** 2).sum()
        ss_rows = k * ((row_means - grand) ** 2).sum()         # subjects
        ss_cols = n * ((col_means - grand) ** 2).sum()         # raters
        ss_err = ss_total - ss_rows - ss_cols                  # residual
        # Mean squares
        bms = ss_rows / (n - 1)
        jms = ss_cols / (k - 1) if k > 1 else 0.0
        ems = ss_err / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
        # ICC(2,1) absolute agreement
        denom2 = bms + (k - 1) * ems + k * (jms - ems) / n
        icc2_1 = (bms - ems) / denom2 if denom2 != 0 else float("nan")
        # ICC(3,1) consistency
        denom3 = bms + (k - 1) * ems
        icc3_1 = (bms - ems) / denom3 if denom3 != 0 else float("nan")
        rows.append({
            "dimension": dim, "n_images": int(n), "k_runs": int(k),
            "icc2_1": float(icc2_1), "icc3_1": float(icc3_1),
        })
    return pd.DataFrame(rows)
