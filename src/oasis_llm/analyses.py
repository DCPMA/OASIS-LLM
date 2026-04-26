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
        ccc_val = _ccc(pred, human)
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
            "ccc": ccc_val,
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


# ─── extended human-norms loader (with Category) ────────────────────────────
def _load_norms_with_category(norms_csv: str = "OASIS/OASIS.csv") -> pd.DataFrame:
    """Load OASIS norms including ``Category`` column.

    Returns columns: ``image_id, category, human_valence, human_arousal``.
    """
    return duckdb.connect(":memory:").execute(
        f"""
        SELECT "Theme"        AS image_id,
               "Category"     AS category,
               "Valence_mean" AS human_valence,
               "Arousal_mean" AS human_arousal
        FROM read_csv_auto('{norms_csv}', header=true)
        """
    ).fetchdf()


def _vs_human_with_category(con, analysis_id: str, norms_csv: str = "OASIS/OASIS.csv") -> pd.DataFrame:
    """Per-image (run × dimension) means joined with human norms + Category."""
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return pd.DataFrame()
    norms = _load_norms_with_category(norms_csv)
    merged = df.merge(norms, on="image_id", how="left")
    merged["human_value"] = merged.apply(
        lambda r: r["human_valence"] if r["dimension"] == "valence" else r["human_arousal"],
        axis=1,
    )
    merged["delta"] = merged["mean_rating"] - merged["human_value"]
    return merged


# ─── shared statistical helpers ─────────────────────────────────────────────
def _cohens_d_paired(diff: "np.ndarray") -> float:
    import numpy as np
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def _bootstrap_ci(
    arr1: "np.ndarray",
    arr2: "np.ndarray | None",
    stat_fn,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a paired/single-array statistic.

    If ``arr2`` is None, ``stat_fn(sample)`` is called on a single resample.
    Otherwise ``stat_fn(s1, s2)`` is called on paired indices.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(arr1)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if arr2 is None:
            stats[i] = stat_fn(arr1[idx])
        else:
            stats[i] = stat_fn(arr1[idx], arr2[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _ccc(pred: "np.ndarray", human: "np.ndarray") -> float:
    """Lin's concordance correlation coefficient."""
    import numpy as np
    p = np.asarray(pred, dtype=float)
    h = np.asarray(human, dtype=float)
    mp, mh = p.mean(), h.mean()
    vp, vh = p.var(ddof=0), h.var(ddof=0)
    cov = np.mean((p - mp) * (h - mh))
    denom = vp + vh + (mp - mh) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def _try_scipy():
    try:
        from scipy import stats as sps  # type: ignore
        return sps
    except Exception:  # pragma: no cover
        return None


# ─── (1) Paired t-test per run × dimension ─────────────────────────────────
def paired_ttest_per_run(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
    n_boot: int = 0,
) -> pd.DataFrame:
    """Per-image paired t-test of LLM mean vs human mean, for each run × dim.

    Columns: ``run_id, model, dimension, n_images, llm_mean, human_mean,
    mean_diff, sd_diff, t, df, p, cohens_d, pearson_r``. If ``n_boot > 0``,
    bootstrap percentile CIs are added: ``mean_diff_ci_lo/hi`` and
    ``cohens_d_ci_lo/hi``.
    """
    import numpy as np
    sps = _try_scipy()
    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    merged = merged.dropna(subset=["mean_rating", "human_value"])

    # run → model lookup
    import json
    run_meta = {}
    for rid, cfg_json in con.execute("SELECT run_id, config_json FROM runs").fetchall():
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        run_meta[rid] = cfg.get("model")

    rows = []
    for (rid, dim), sub in merged.groupby(["run_id", "dimension"]):
        pred = sub["mean_rating"].astype(float).to_numpy()
        human = sub["human_value"].astype(float).to_numpy()
        n = len(pred)
        if n < 2:
            continue
        diff = pred - human
        sd_diff = float(diff.std(ddof=1))
        if sps is not None:
            t_stat, p_val = sps.ttest_rel(pred, human)
            r_pearson = float(sps.pearsonr(pred, human)[0]) if pred.std() and human.std() else float("nan")
        else:
            se = sd_diff / np.sqrt(n) if sd_diff > 0 else float("nan")
            t_stat = float(diff.mean() / se) if se and se > 0 else float("nan")
            p_val = float("nan")
            r_pearson = float(np.corrcoef(pred, human)[0, 1]) if pred.std() and human.std() else float("nan")
        d = _cohens_d_paired(diff)
        row = {
            "run_id": rid,
            "model": run_meta.get(rid),
            "dimension": dim,
            "n_images": int(n),
            "llm_mean": float(pred.mean()),
            "human_mean": float(human.mean()),
            "mean_diff": float(diff.mean()),
            "sd_diff": sd_diff,
            "t": float(t_stat),
            "df": int(n - 1),
            "p": float(p_val),
            "cohens_d": d,
            "pearson_r": float(r_pearson),
        }
        if n_boot > 0:
            mlo, mhi = _bootstrap_ci(pred, human, lambda a, b: float(np.mean(a - b)), n_boot=n_boot)
            dlo, dhi = _bootstrap_ci(pred, human, lambda a, b: _cohens_d_paired(a - b), n_boot=n_boot)
            row.update({
                "mean_diff_ci_lo": mlo, "mean_diff_ci_hi": mhi,
                "cohens_d_ci_lo": dlo, "cohens_d_ci_hi": dhi,
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dimension", "p"]).reset_index(drop=True)
    return out


# ─── (3) Pooled all-LLM aggregate t-test ───────────────────────────────────
def pooled_ttest(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
    n_boot: int = 0,
) -> pd.DataFrame:
    """Average across runs to one LLM value per image, then paired t vs human.

    Columns: ``dimension, n_images, k_runs, llm_mean, human_mean, mean_diff,
    sd_diff, t, df, p, cohens_d, pearson_r`` (+ bootstrap CIs if requested).
    """
    import numpy as np
    sps = _try_scipy()
    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for dim, sub in merged.groupby("dimension"):
        pivot = sub.pivot_table(
            index="image_id", columns="run_id", values="mean_rating", aggfunc="mean",
        )
        human = sub.drop_duplicates("image_id").set_index("image_id")["human_value"]
        pooled = pivot.mean(axis=1)
        joined = pd.concat([pooled.rename("llm"), human.rename("human")], axis=1).dropna()
        if len(joined) < 2:
            continue
        pred = joined["llm"].to_numpy()
        h = joined["human"].to_numpy()
        diff = pred - h
        sd_diff = float(diff.std(ddof=1))
        if sps is not None:
            t_stat, p_val = sps.ttest_rel(pred, h)
            r_pearson = float(sps.pearsonr(pred, h)[0]) if pred.std() and h.std() else float("nan")
        else:
            se = sd_diff / np.sqrt(len(diff)) if sd_diff > 0 else float("nan")
            t_stat = float(diff.mean() / se) if se and se > 0 else float("nan")
            p_val = float("nan")
            r_pearson = float(np.corrcoef(pred, h)[0, 1]) if pred.std() and h.std() else float("nan")
        row = {
            "dimension": dim,
            "n_images": int(len(joined)),
            "k_runs": int(pivot.shape[1]),
            "llm_mean": float(pred.mean()),
            "human_mean": float(h.mean()),
            "mean_diff": float(diff.mean()),
            "sd_diff": sd_diff,
            "t": float(t_stat),
            "df": int(len(joined) - 1),
            "p": float(p_val),
            "cohens_d": _cohens_d_paired(diff),
            "pearson_r": float(r_pearson),
        }
        if n_boot > 0:
            mlo, mhi = _bootstrap_ci(pred, h, lambda a, b: float(np.mean(a - b)), n_boot=n_boot)
            dlo, dhi = _bootstrap_ci(pred, h, lambda a, b: _cohens_d_paired(a - b), n_boot=n_boot)
            row.update({
                "mean_diff_ci_lo": mlo, "mean_diff_ci_hi": mhi,
                "cohens_d_ci_lo": dlo, "cohens_d_ci_hi": dhi,
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ─── (4) Linear regression LLM on Human ────────────────────────────────────
def regress_llm_on_human(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
    pooled: bool = False,
) -> pd.DataFrame:
    """OLS regression ``LLM = a + b · Human`` per run × dimension.

    A perfectly calibrated model has ``b = 1, a = 0``. ``b > 1`` ⇒ scale
    stretch; ``a > 0`` ⇒ positive shift. If ``pooled=True``, regresses the
    cross-run pooled mean per image instead of per-run.
    Returns columns: ``[run_id|"pooled"], dimension, n, slope, intercept,
    r2, residual_sd``.
    """
    import numpy as np
    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    if pooled:
        groups = []
        for dim, sub in merged.groupby("dimension"):
            pivot = sub.pivot_table(index="image_id", columns="run_id", values="mean_rating", aggfunc="mean")
            human = sub.drop_duplicates("image_id").set_index("image_id")["human_value"]
            pooled_s = pivot.mean(axis=1)
            j = pd.concat([pooled_s.rename("y"), human.rename("x")], axis=1).dropna()
            groups.append(("pooled", dim, j))
    else:
        groups = []
        for (rid, dim), sub in merged.groupby(["run_id", "dimension"]):
            j = sub[["mean_rating", "human_value"]].dropna().rename(
                columns={"mean_rating": "y", "human_value": "x"}
            )
            groups.append((rid, dim, j))

    rows = []
    for label, dim, j in groups:
        if len(j) < 3:
            continue
        x = j["x"].to_numpy(dtype=float); y = j["y"].to_numpy(dtype=float)
        if x.std() == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({
            ("scope" if pooled else "run_id"): label,
            "dimension": dim,
            "n": int(len(j)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "residual_sd": float(np.sqrt(ss_res / max(len(j) - 2, 1))),
        })
    return pd.DataFrame(rows)


# ─── (5) CCC helper, exposed as public API ─────────────────────────────────
def ccc_score(pred, human) -> float:
    """Lin's concordance correlation coefficient. Public wrapper."""
    import numpy as np
    return _ccc(np.asarray(pred), np.asarray(human))


# ─── (6) Per-category breakdown ────────────────────────────────────────────
def category_breakdown(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
) -> pd.DataFrame:
    """LLM-vs-human stats per (run × dimension × OASIS category).

    Columns: ``run_id, model, dimension, category, n_images, llm_mean,
    human_mean, mean_diff, pearson_r, t, p, cohens_d``.
    """
    import json
    import numpy as np
    sps = _try_scipy()
    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    merged = merged.dropna(subset=["mean_rating", "human_value", "category"])

    run_meta = {}
    for rid, cfg_json in con.execute("SELECT run_id, config_json FROM runs").fetchall():
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        run_meta[rid] = cfg.get("model")

    rows = []
    for (rid, dim, cat), sub in merged.groupby(["run_id", "dimension", "category"]):
        pred = sub["mean_rating"].astype(float).to_numpy()
        human = sub["human_value"].astype(float).to_numpy()
        n = len(pred)
        if n < 2:
            continue
        diff = pred - human
        sd_diff = float(diff.std(ddof=1))
        if sps is not None:
            t_stat, p_val = sps.ttest_rel(pred, human) if n > 1 else (float("nan"), float("nan"))
            r_pearson = float(sps.pearsonr(pred, human)[0]) if pred.std() and human.std() else float("nan")
        else:
            se = sd_diff / np.sqrt(n) if sd_diff > 0 else float("nan")
            t_stat = float(diff.mean() / se) if se and se > 0 else float("nan")
            p_val = float("nan")
            r_pearson = float(np.corrcoef(pred, human)[0, 1]) if pred.std() and human.std() else float("nan")
        rows.append({
            "run_id": rid,
            "model": run_meta.get(rid),
            "dimension": dim,
            "category": cat,
            "n_images": int(n),
            "llm_mean": float(pred.mean()),
            "human_mean": float(human.mean()),
            "mean_diff": float(diff.mean()),
            "pearson_r": float(r_pearson),
            "t": float(t_stat),
            "p": float(p_val),
            "cohens_d": _cohens_d_paired(diff),
        })
    return pd.DataFrame(rows).sort_values(["dimension", "category", "model"]).reset_index(drop=True)


# ─── (10) Distribution comparison ─────────────────────────────────────────
def distribution_compare(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
) -> pd.DataFrame:
    """Compare each run's raw rating distribution to human image-mean dist.

    Returns: ``run_id, model, dimension, n_llm, n_human, ks_stat, ks_p,
    wasserstein, llm_kurtosis, llm_pct_extreme``.
    """
    import json
    import numpy as np
    sps = _try_scipy()
    if sps is None:
        return pd.DataFrame()  # KS/Wasserstein require scipy

    a = get(con, analysis_id)
    if a is None or not a.run_ids:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(a.run_ids))
    raw = con.execute(
        f"""
        SELECT run_id, dimension, rating
        FROM trials
        WHERE run_id IN ({placeholders}) AND status='done' AND rating IS NOT NULL
        """,
        a.run_ids,
    ).fetchdf()
    if raw.empty:
        return pd.DataFrame()

    norms = _load_norms_with_category(norms_csv)
    human_by_dim = {
        "valence": norms["human_valence"].dropna().to_numpy(),
        "arousal": norms["human_arousal"].dropna().to_numpy(),
    }

    run_meta = {}
    for rid, cfg_json in con.execute("SELECT run_id, config_json FROM runs").fetchall():
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        run_meta[rid] = cfg.get("model")

    rows = []
    for (rid, dim), sub in raw.groupby(["run_id", "dimension"]):
        ll = sub["rating"].astype(float).to_numpy()
        hh = human_by_dim.get(dim)
        if hh is None or len(hh) == 0 or len(ll) == 0:
            continue
        ks_stat, ks_p = sps.ks_2samp(ll, hh)
        try:
            w = float(sps.wasserstein_distance(ll, hh))
        except Exception:
            w = float("nan")
        try:
            kurt = float(sps.kurtosis(ll, fisher=True, bias=False))
        except Exception:
            kurt = float("nan")
        pct_extreme = float(np.mean((ll == ll.min()) | (ll == ll.max())))
        rows.append({
            "run_id": rid, "model": run_meta.get(rid), "dimension": dim,
            "n_llm": int(len(ll)), "n_human": int(len(hh)),
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "wasserstein": w, "llm_kurtosis": kurt,
            "llm_pct_extreme": pct_extreme,
        })
    return pd.DataFrame(rows).sort_values(["dimension", "wasserstein"]).reset_index(drop=True)


# ─── (13) Outlier image table ──────────────────────────────────────────────
def outlier_images(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
    top_k: int = 20,
    scope: str = "pooled",  # "pooled" | "per_run"
) -> pd.DataFrame:
    """Top-K images where LLM disagrees most with human norms.

    scope="pooled" uses the average across runs; scope="per_run" returns the
    largest |Δ| rows globally across runs.
    """
    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    if scope == "pooled":
        agg = (
            merged.groupby(["image_id", "category", "dimension"])
            .agg(llm_mean=("mean_rating", "mean"),
                 human_value=("human_value", "first"),
                 k_runs=("run_id", "nunique"))
            .reset_index()
        )
        agg["delta"] = agg["llm_mean"] - agg["human_value"]
        agg["abs_delta"] = agg["delta"].abs()
        out = (
            agg.sort_values(["dimension", "abs_delta"], ascending=[True, False])
            .groupby("dimension", group_keys=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return out
    # per_run
    merged["abs_delta"] = merged["delta"].abs()
    out = (
        merged.sort_values(["dimension", "abs_delta"], ascending=[True, False])
        .groupby("dimension", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return out[["run_id", "image_id", "category", "dimension", "mean_rating", "human_value", "delta", "abs_delta"]]


# ─── (15) Inter-LLM agreement across N runs ────────────────────────────────
def inter_llm_agreement(con, analysis_id: str) -> pd.DataFrame:
    """Per-dimension agreement metrics across the runs in the analysis.

    Columns: ``dimension, n_images, k_runs, mean_pairwise_r,
    median_pairwise_r, mean_per_image_sd, icc2_1, icc3_1``.
    """
    import numpy as np
    df = per_image_aggregate(con, analysis_id)
    if df.empty:
        return pd.DataFrame()
    icc_df = icc_across_runs(con, analysis_id).set_index("dimension")
    rows = []
    for dim, sub in df.groupby("dimension"):
        wide = sub.pivot(index="image_id", columns="run_id", values="mean_rating").dropna()
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue
        corr = wide.corr().values
        iu = np.triu_indices_from(corr, k=1)
        pairwise = corr[iu]
        per_image_sd = wide.std(axis=1, ddof=1)
        rows.append({
            "dimension": dim,
            "n_images": int(wide.shape[0]),
            "k_runs": int(wide.shape[1]),
            "mean_pairwise_r": float(np.nanmean(pairwise)),
            "median_pairwise_r": float(np.nanmedian(pairwise)),
            "mean_per_image_sd": float(per_image_sd.mean()),
            "icc2_1": float(icc_df.loc[dim, "icc2_1"]) if dim in icc_df.index else float("nan"),
            "icc3_1": float(icc_df.loc[dim, "icc3_1"]) if dim in icc_df.index else float("nan"),
        })
    return pd.DataFrame(rows)


# ─── (16) Category × Model 2-way ANOVA on the |error| ──────────────────────
def category_model_anova(
    con,
    analysis_id: str,
    *,
    norms_csv: str = "OASIS/OASIS.csv",
) -> pd.DataFrame:
    """Two-way ANOVA on |LLM-human delta| per image: Category × Model.

    Tests whether bias magnitude depends on category, on model, or their
    interaction. Implementation uses Type-III-style sums of squares for a
    balanced/unbalanced two-factor design (effects-coded contrasts via least
    squares; falls back to Type I ordering if statsmodels is unavailable).

    Columns: ``dimension, factor, df, ss, ms, F, p, eta_sq``.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception:
        return pd.DataFrame()  # statsmodels optional
    import json

    merged = _vs_human_with_category(con, analysis_id, norms_csv)
    if merged.empty:
        return pd.DataFrame()
    run_meta = {}
    for rid, cfg_json in con.execute("SELECT run_id, config_json FROM runs").fetchall():
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except Exception:
            cfg = {}
        run_meta[rid] = cfg.get("model") or rid
    merged = merged.dropna(subset=["mean_rating", "human_value", "category"]).copy()
    merged["model"] = merged["run_id"].map(run_meta)
    merged["abs_err"] = (merged["mean_rating"] - merged["human_value"]).abs()

    rows = []
    for dim, sub in merged.groupby("dimension"):
        if sub["category"].nunique() < 2 or sub["model"].nunique() < 2 or len(sub) < 8:
            continue
        # Type II ANOVA (statsmodels default with anova_lm typ=2)
        try:
            model = smf.ols("abs_err ~ C(category) + C(model) + C(category):C(model)", data=sub).fit()
            aov = sm.stats.anova_lm(model, typ=2)
        except Exception:
            continue
        ss_total = aov["sum_sq"].sum()
        for factor, r in aov.iterrows():
            rows.append({
                "dimension": dim,
                "factor": factor,
                "df": float(r["df"]),
                "ss": float(r["sum_sq"]),
                "ms": float(r["sum_sq"]) / float(r["df"]) if r["df"] else float("nan"),
                "F": float(r["F"]) if "F" in r and not pd.isna(r["F"]) else float("nan"),
                "p": float(r["PR(>F)"]) if "PR(>F)" in r and not pd.isna(r["PR(>F)"]) else float("nan"),
                "eta_sq": float(r["sum_sq"]) / ss_total if ss_total > 0 else float("nan"),
            })
    return pd.DataFrame(rows)

