"""Per-experiment export/import bundles.

A bundle is a zip with this layout:

    manifest.json
    experiment.json
    dataset.json                  (just the metadata row, no images)
    dataset_images.csv            (list of image_ids; no files)
    configs/<config_name>.yaml    (one per config in the experiment)
    runs.csv  + runs.parquet      (backing runs rows)
    trials.csv + trials.parquet   (all trials for those runs)

Images themselves are intentionally NOT bundled. The destination workspace must
already have the OASIS images (or any local cache) at the same image_ids — the
bundle only ships the *labels* attached to each image_id.

Schema version is bumped on any breaking change.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any

import duckdb
import pandas as pd
import yaml

SCHEMA_VERSION = 1


# ─── export ────────────────────────────────────────────────────────────────
def export_experiment(con: duckdb.DuckDBPyConnection, experiment_id: str) -> bytes:
    """Build a zip bundle for a single experiment.

    Returns the bundle as raw bytes (caller writes to disk or hands to
    Streamlit's download button).

    Raises ``KeyError`` if the experiment doesn't exist.
    """
    exp_row = con.execute(
        """
        SELECT experiment_id, name, description, dataset_id, status,
               created_at, finished_at
        FROM experiments WHERE experiment_id=?
        """,
        [experiment_id],
    ).fetchone()
    if exp_row is None:
        raise KeyError(f"unknown experiment: {experiment_id}")
    exp_dict = {
        "experiment_id": exp_row[0],
        "name": exp_row[1],
        "description": exp_row[2],
        "dataset_id": exp_row[3],
        "status": exp_row[4],
        "created_at": _iso(exp_row[5]),
        "finished_at": _iso(exp_row[6]),
    }
    dataset_id = exp_row[3]

    # Dataset metadata
    ds_row = con.execute(
        """
        SELECT dataset_id, name, description, status, source,
               generation_params, created_at, approved_at
        FROM datasets WHERE dataset_id=?
        """,
        [dataset_id],
    ).fetchone()
    dataset_dict: dict[str, Any] = {}
    if ds_row is not None:
        dataset_dict = {
            "dataset_id": ds_row[0],
            "name": ds_row[1],
            "description": ds_row[2],
            "status": ds_row[3],
            "source": ds_row[4],
            "generation_params": ds_row[5],
            "created_at": _iso(ds_row[6]),
            "approved_at": _iso(ds_row[7]),
        }
    dataset_images_df = con.execute(
        """
        SELECT dataset_id, image_id, excluded, note
        FROM dataset_images WHERE dataset_id=?
        ORDER BY image_id
        """,
        [dataset_id],
    ).fetchdf()

    # Experiment configs
    cfg_rows = con.execute(
        """
        SELECT config_name, config_json, run_id, position
        FROM experiment_configs WHERE experiment_id=?
        ORDER BY position, config_name
        """,
        [experiment_id],
    ).fetchall()
    config_files: dict[str, dict] = {}
    run_ids: list[str] = []
    for cfg_name, cfg_json, run_id, _pos in cfg_rows:
        config_files[cfg_name] = json.loads(cfg_json)
        run_ids.append(run_id)

    # Runs + trials for those run_ids
    runs_df = pd.DataFrame()
    trials_df = pd.DataFrame()
    if run_ids:
        placeholders = ",".join("?" * len(run_ids))
        runs_df = con.execute(
            f"""
            SELECT run_id, config_json, config_hash, status, created_at, finished_at
            FROM runs WHERE run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchdf()
        trials_df = con.execute(
            f"""
            SELECT * FROM trials WHERE run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchdf()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "run_ids": run_ids,
        "n_runs": len(run_ids),
        "n_trials": int(len(trials_df)),
        "n_dataset_images": int(len(dataset_images_df)),
    }

    # Build the zip in-memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("experiment.json", json.dumps(exp_dict, indent=2))
        z.writestr("dataset.json", json.dumps(dataset_dict, indent=2))
        z.writestr(
            "dataset_images.csv",
            dataset_images_df.to_csv(index=False) if not dataset_images_df.empty else "",
        )
        for cfg_name, payload in config_files.items():
            z.writestr(f"configs/{cfg_name}.yaml", yaml.safe_dump(payload, sort_keys=False))
        if not runs_df.empty:
            z.writestr("runs.csv", runs_df.to_csv(index=False))
            z.writestr("runs.parquet", _df_to_parquet(runs_df))
        if not trials_df.empty:
            z.writestr("trials.csv", trials_df.to_csv(index=False))
            z.writestr("trials.parquet", _df_to_parquet(trials_df))
    return buf.getvalue()


# ─── import ────────────────────────────────────────────────────────────────
def import_experiment(
    con: duckdb.DuckDBPyConnection,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a bundle produced by ``export_experiment``.

    By default skips any rows whose primary key already exists. Set
    ``overwrite=True`` to delete-then-replace the experiment + its runs/trials.

    Returns a summary dict.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported bundle schema_version: {manifest.get('schema_version')}"
                f" (this code supports v{SCHEMA_VERSION})"
            )
        exp = json.loads(z.read("experiment.json"))
        dataset = json.loads(z.read("dataset.json")) if "dataset.json" in names else {}
        dataset_images_csv = z.read("dataset_images.csv").decode() if "dataset_images.csv" in names else ""
        config_files: dict[str, dict] = {}
        for n in names:
            if n.startswith("configs/") and n.endswith(".yaml"):
                cfg_name = n[len("configs/"):-len(".yaml")]
                config_files[cfg_name] = yaml.safe_load(z.read(n))
        runs_df = (
            pd.read_parquet(io.BytesIO(z.read("runs.parquet")))
            if "runs.parquet" in names else pd.DataFrame()
        )
        trials_df = (
            pd.read_parquet(io.BytesIO(z.read("trials.parquet")))
            if "trials.parquet" in names else pd.DataFrame()
        )

    summary = {
        "experiment_id": exp["experiment_id"],
        "dataset_id": exp["dataset_id"],
        "imported_runs": 0,
        "imported_trials": 0,
        "imported_dataset_images": 0,
        "skipped": [],
    }
    exp_id = exp["experiment_id"]
    dataset_id = exp["dataset_id"]

    # Optional clean slate
    if overwrite:
        run_ids = manifest.get("run_ids") or []
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            con.execute(
                f"DELETE FROM trials WHERE run_id IN ({placeholders})", run_ids
            )
            con.execute(
                f"DELETE FROM runs WHERE run_id IN ({placeholders})", run_ids
            )
        con.execute("DELETE FROM experiment_configs WHERE experiment_id=?", [exp_id])
        con.execute("DELETE FROM experiments WHERE experiment_id=?", [exp_id])

    # Dataset (idempotent INSERT … SELECT WHERE NOT EXISTS)
    if dataset:
        existing = con.execute(
            "SELECT 1 FROM datasets WHERE dataset_id=?", [dataset_id]
        ).fetchone()
        if not existing:
            con.execute(
                """
                INSERT INTO datasets
                  (dataset_id, name, description, status, source,
                   generation_params, created_at, approved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    dataset["dataset_id"], dataset.get("name") or dataset_id,
                    dataset.get("description"), dataset.get("status") or "approved",
                    dataset.get("source") or "imported",
                    dataset.get("generation_params"),
                    dataset.get("created_at"), dataset.get("approved_at"),
                ],
            )
        else:
            summary["skipped"].append(f"dataset {dataset_id} already exists")

    # Dataset images
    if dataset_images_csv.strip():
        dimg = pd.read_csv(io.StringIO(dataset_images_csv))
        for _, row in dimg.iterrows():
            try:
                con.execute(
                    """
                    INSERT INTO dataset_images (dataset_id, image_id, excluded, note)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        row["dataset_id"], row["image_id"],
                        bool(row.get("excluded", False)), row.get("note"),
                    ],
                )
                summary["imported_dataset_images"] += 1
            except Exception as e:
                summary["skipped"].append(f"dataset_image {row['image_id']}: {e}")

    # Experiment row
    if not con.execute(
        "SELECT 1 FROM experiments WHERE experiment_id=?", [exp_id]
    ).fetchone():
        con.execute(
            """
            INSERT INTO experiments
              (experiment_id, name, description, dataset_id, status,
               created_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                exp_id, exp.get("name") or exp_id, exp.get("description"),
                dataset_id, exp.get("status") or "imported",
                exp.get("created_at"), exp.get("finished_at"),
            ],
        )
    else:
        summary["skipped"].append(f"experiment {exp_id} already exists")

    # Runs (must precede experiment_configs FK, and trials)
    if not runs_df.empty:
        for _, row in runs_df.iterrows():
            try:
                con.execute(
                    """
                    INSERT INTO runs (run_id, config_json, config_hash, status,
                                      created_at, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [row["run_id"], row["config_json"], row["config_hash"],
                     row["status"], row.get("created_at"), row.get("finished_at")],
                )
                summary["imported_runs"] += 1
            except Exception as e:
                summary["skipped"].append(f"run {row['run_id']}: {e}")

    # Experiment configs (one per config file)
    for pos, (cfg_name, payload) in enumerate(sorted(config_files.items())):
        run_id = f"{exp_id}__{cfg_name}"
        try:
            con.execute(
                """
                INSERT INTO experiment_configs
                  (experiment_id, config_name, config_json, run_id, position)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [exp_id, cfg_name, json.dumps(payload), run_id, pos],
            )
        except Exception as e:
            summary["skipped"].append(f"config {cfg_name}: {e}")

    # Trials (bulk via DataFrame)
    if not trials_df.empty:
        # ensure column order matches table
        cols = [
            "run_id", "image_id", "dimension", "sample_idx", "status",
            "rating", "raw_response", "reasoning", "prompt_hash", "latency_ms",
            "input_tokens", "output_tokens", "cost_usd", "error", "attempts",
            "claimed_at", "completed_at", "finish_reason", "response_id",
            "trace_id",
        ]
        present = [c for c in cols if c in trials_df.columns]
        sub = trials_df[present].copy()
        # Bulk insert via duckdb register
        con.register("__trials_in", sub)
        before = con.execute("SELECT count(*) FROM trials").fetchone()[0]
        col_list = ", ".join(present)
        con.execute(
            f"""
            INSERT INTO trials ({col_list})
            SELECT {col_list} FROM __trials_in
            ON CONFLICT DO NOTHING
            """
        )
        after = con.execute("SELECT count(*) FROM trials").fetchone()[0]
        con.unregister("__trials_in")
        summary["imported_trials"] = int(after - before)

    return summary


# ─── helpers ────────────────────────────────────────────────────────────────
def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _df_to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
