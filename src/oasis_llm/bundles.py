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


# ─── dataset export ────────────────────────────────────────────────────────
def export_dataset(
    con: duckdb.DuckDBPyConnection,
    dataset_id: str,
    *,
    include_images: bool = False,
) -> bytes:
    """Build a zip bundle for a single dataset.

    Layout:
        manifest.json
        dataset.json
        dataset_images.csv
        images/<image_id>.jpg     (only when include_images=True)

    ``include_images=True`` copies every active image file into the bundle so
    the receiving workspace doesn't need a local OASIS/images checkout. This
    can produce large zips (≥100 MB for the full pool).
    """
    ds_row = con.execute(
        """
        SELECT dataset_id, name, description, status, source,
               generation_params, created_at, approved_at
        FROM datasets WHERE dataset_id=?
        """,
        [dataset_id],
    ).fetchone()
    if ds_row is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
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
    images_df = con.execute(
        """
        SELECT dataset_id, image_id, excluded, note
        FROM dataset_images WHERE dataset_id=?
        ORDER BY image_id
        """,
        [dataset_id],
    ).fetchdf()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dataset",
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "n_images": int(len(images_df)),
        "images_bundled": include_images,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("dataset.json", json.dumps(dataset_dict, indent=2))
        z.writestr(
            "dataset_images.csv",
            images_df.to_csv(index=False) if not images_df.empty else "",
        )
        if include_images and not images_df.empty:
            from .images import IMAGES_DIR
            for iid in images_df["image_id"].tolist():
                p = IMAGES_DIR / f"{iid}.jpg"
                if p.exists():
                    z.write(p, arcname=f"images/{iid}.jpg")
    return buf.getvalue()


# ─── analysis export ───────────────────────────────────────────────────────
def export_analysis(con: duckdb.DuckDBPyConnection, analysis_id: str) -> bytes:
    """Build a zip bundle for a saved analysis.

    Layout:
        manifest.json
        analysis.json                 (spec row)
        analysis_runs.csv             (which runs are members + label)
        per_image_aggregate.csv       (computed by analysis.per_image_aggregate)
        cross_run_correlations.csv    (when ≥2 runs are present)
        vs_human_norms.csv            (when human norms are wired up)

    Computed tables are best-effort — if the analysis module raises, the
    error is recorded under ``manifest.computed_errors`` and other tables
    proceed.
    """
    a_row = con.execute(
        """
        SELECT analysis_id, name, description, dataset_id, created_at
        FROM analyses WHERE analysis_id=?
        """,
        [analysis_id],
    ).fetchone()
    if a_row is None:
        raise KeyError(f"unknown analysis: {analysis_id}")
    analysis_dict = {
        "analysis_id": a_row[0],
        "name": a_row[1],
        "description": a_row[2],
        "dataset_id": a_row[3],
        "created_at": _iso(a_row[4]),
    }
    runs_df = con.execute(
        """
        SELECT analysis_id, run_id, label, added_at
        FROM analysis_runs WHERE analysis_id=?
        ORDER BY added_at
        """,
        [analysis_id],
    ).fetchdf()

    computed: dict[str, pd.DataFrame] = {}
    extras: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        from . import analyses as an
    except Exception as e:
        an = None  # type: ignore[assignment]
        errors["import"] = f"{type(e).__name__}: {e}"
    if an is not None:
        try:
            df = an.per_image_aggregate(con, analysis_id)
            if df is not None and not df.empty:
                computed["per_image_aggregate"] = df
        except Exception as e:
            errors["per_image_aggregate"] = f"{type(e).__name__}: {e}"
        try:
            d = an.cross_run_correlations(con, analysis_id)
            if d:
                extras["cross_run_correlations"] = d
        except Exception as e:
            errors["cross_run_correlations"] = f"{type(e).__name__}: {e}"
        try:
            df = an.vs_human_norms(con, analysis_id)
            if df is not None and not df.empty:
                computed["vs_human_norms"] = df
        except Exception as e:
            errors["vs_human_norms"] = f"{type(e).__name__}: {e}"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "analysis",
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "analysis_id": analysis_id,
        "dataset_id": a_row[3],
        "n_runs": int(len(runs_df)),
        "computed_tables": list(computed.keys()),
        "computed_errors": errors,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("analysis.json", json.dumps(analysis_dict, indent=2))
        z.writestr(
            "analysis_runs.csv",
            runs_df.to_csv(index=False) if not runs_df.empty else "",
        )
        for name, df in computed.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
            try:
                z.writestr(f"{name}.parquet", _df_to_parquet(df))
            except Exception:
                pass  # Parquet is best-effort — CSV is the source of truth.
        if extras:
            z.writestr("extras.json", json.dumps(extras, indent=2, default=str))
    return buf.getvalue()


# ─── global export ─────────────────────────────────────────────────────────
def export_bundle(
    con: duckdb.DuckDBPyConnection,
    *,
    dataset_ids: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    analysis_ids: list[str] | None = None,
    include_images: bool = False,
) -> bytes:
    """Bundle multiple entities into a single ``.zip`` for backup / migration.

    Layout:
        manifest.json
        datasets/<dataset_id>.zip
        experiments/<experiment_id>.zip
        analyses/<analysis_id>.zip

    Each inner zip is exactly what ``export_*`` would emit individually, so
    receiving code can extract them and run the existing import path.
    """
    dataset_ids = list(dataset_ids or [])
    experiment_ids = list(experiment_ids or [])
    analysis_ids = list(analysis_ids or [])

    failures: list[dict] = []

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for did in dataset_ids:
            try:
                z.writestr(
                    f"datasets/{did}.zip",
                    export_dataset(con, did, include_images=include_images),
                )
            except Exception as e:
                failures.append({"kind": "dataset", "id": did, "error": str(e)})
        for eid in experiment_ids:
            try:
                z.writestr(f"experiments/{eid}.zip", export_experiment(con, eid))
            except Exception as e:
                failures.append({"kind": "experiment", "id": eid, "error": str(e)})
        for aid in analysis_ids:
            try:
                z.writestr(f"analyses/{aid}.zip", export_analysis(con, aid))
            except Exception as e:
                failures.append({"kind": "analysis", "id": aid, "error": str(e)})

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "global",
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "dataset_ids": dataset_ids,
            "experiment_ids": experiment_ids,
            "analysis_ids": analysis_ids,
            "n_datasets": len(dataset_ids),
            "n_experiments": len(experiment_ids),
            "n_analyses": len(analysis_ids),
            "images_bundled": include_images,
            "failures": failures,
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
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


# ─── dataset import ────────────────────────────────────────────────────────
def import_dataset(
    con: duckdb.DuckDBPyConnection,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a bundle produced by ``export_dataset``.

    By default skips datasets whose ``dataset_id`` already exists. With
    ``overwrite=True``, deletes the existing dataset row + its
    ``dataset_images`` and re-imports.

    Image files (``images/<id>.jpg``) are extracted into ``IMAGES_DIR`` only
    if the file isn't already there — never overwrites a local image.
    """
    summary: dict[str, Any] = {
        "kind": "dataset",
        "dataset_id": None,
        "imported_dataset_images": 0,
        "imported_image_files": 0,
        "skipped": [],
    }
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported bundle schema_version: {manifest.get('schema_version')}"
                f" (this code supports v{SCHEMA_VERSION})"
            )
        dataset = json.loads(z.read("dataset.json"))
        dataset_id = dataset["dataset_id"]
        summary["dataset_id"] = dataset_id
        images_csv = (
            z.read("dataset_images.csv").decode()
            if "dataset_images.csv" in names else ""
        )

        if overwrite:
            con.execute(
                "DELETE FROM dataset_images WHERE dataset_id=?", [dataset_id]
            )
            con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])

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
                    dataset_id, dataset.get("name") or dataset_id,
                    dataset.get("description"),
                    dataset.get("status") or "approved",
                    dataset.get("source") or "imported",
                    dataset.get("generation_params"),
                    dataset.get("created_at"), dataset.get("approved_at"),
                ],
            )
        else:
            summary["skipped"].append(f"dataset {dataset_id} already exists")

        if images_csv.strip():
            dimg = pd.read_csv(io.StringIO(images_csv))
            for _, row in dimg.iterrows():
                try:
                    con.execute(
                        """
                        INSERT INTO dataset_images
                          (dataset_id, image_id, excluded, note)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                        """,
                        [
                            row["dataset_id"], row["image_id"],
                            bool(row.get("excluded", False)),
                            row.get("note"),
                        ],
                    )
                    summary["imported_dataset_images"] += 1
                except Exception as e:
                    summary["skipped"].append(
                        f"dataset_image {row['image_id']}: {e}"
                    )

        # Optional bundled image files
        if any(n.startswith("images/") for n in names):
            from .images import IMAGES_DIR
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            for n in names:
                if not n.startswith("images/") or n.endswith("/"):
                    continue
                fname = n[len("images/"):]
                target = IMAGES_DIR / fname
                if target.exists():
                    continue
                target.write_bytes(z.read(n))
                summary["imported_image_files"] += 1

    return summary


# ─── analysis import ───────────────────────────────────────────────────────
def import_analysis(
    con: duckdb.DuckDBPyConnection,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a bundle produced by ``export_analysis``.

    Imports the analysis row and its ``analysis_runs`` membership. The runs
    themselves are NOT imported (analysis bundles intentionally don't carry
    them — a missing run_id is recorded under ``skipped``).
    """
    summary: dict[str, Any] = {
        "kind": "analysis",
        "analysis_id": None,
        "imported_runs": 0,
        "skipped": [],
    }
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported bundle schema_version: {manifest.get('schema_version')}"
                f" (this code supports v{SCHEMA_VERSION})"
            )
        analysis = json.loads(z.read("analysis.json"))
        runs_csv = (
            z.read("analysis_runs.csv").decode()
            if "analysis_runs.csv" in names else ""
        )

    aid = analysis["analysis_id"]
    summary["analysis_id"] = aid

    if overwrite:
        con.execute("DELETE FROM analysis_runs WHERE analysis_id=?", [aid])
        con.execute("DELETE FROM analyses WHERE analysis_id=?", [aid])

    existing = con.execute(
        "SELECT 1 FROM analyses WHERE analysis_id=?", [aid]
    ).fetchone()
    if not existing:
        con.execute(
            """
            INSERT INTO analyses
              (analysis_id, name, description, dataset_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                aid, analysis.get("name") or aid,
                analysis.get("description"),
                analysis.get("dataset_id"),
                analysis.get("created_at"),
            ],
        )
    else:
        summary["skipped"].append(f"analysis {aid} already exists")

    if runs_csv.strip():
        rdf = pd.read_csv(io.StringIO(runs_csv))
        for _, row in rdf.iterrows():
            run_exists = con.execute(
                "SELECT 1 FROM runs WHERE run_id=?", [row["run_id"]]
            ).fetchone()
            if not run_exists:
                summary["skipped"].append(
                    f"run {row['run_id']} not present locally; "
                    "membership row skipped"
                )
                continue
            try:
                con.execute(
                    """
                    INSERT INTO analysis_runs
                      (analysis_id, run_id, label, added_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        row["analysis_id"], row["run_id"],
                        row.get("label"), row.get("added_at"),
                    ],
                )
                summary["imported_runs"] += 1
            except Exception as e:
                summary["skipped"].append(
                    f"analysis_run {row['run_id']}: {e}"
                )

    return summary


# ─── global / multi-bundle import ──────────────────────────────────────────
def import_bundle(
    con: duckdb.DuckDBPyConnection,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a global bundle produced by ``export_bundle``.

    Iterates through ``datasets/*.zip``, ``experiments/*.zip``,
    ``analyses/*.zip`` (in that order, so referenced datasets exist before
    their experiments/analyses) and dispatches to the per-kind importers.
    Per-entry failures are caught and recorded under ``failures``; the import
    continues.
    """
    summary: dict[str, Any] = {
        "kind": "global",
        "datasets": [],
        "experiments": [],
        "analyses": [],
        "failures": [],
    }
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("kind") != "global":
            raise ValueError(
                f"not a global bundle (kind={manifest.get('kind')!r}); "
                "use import_any() for auto-detection"
            )

        def _process(prefix: str, fn, key: str) -> None:
            for n in sorted(names):
                if not n.startswith(prefix) or not n.endswith(".zip"):
                    continue
                inner = z.read(n)
                try:
                    sub = fn(con, inner, overwrite=overwrite)
                    summary[key].append(sub)
                except Exception as e:
                    summary["failures"].append(
                        {"entry": n, "error": f"{type(e).__name__}: {e}"}
                    )

        _process("datasets/", import_dataset, "datasets")
        _process("experiments/", import_experiment, "experiments")
        _process("analyses/", import_analysis, "analyses")

    return summary


def import_any(
    con: duckdb.DuckDBPyConnection,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Auto-detect the bundle kind from ``manifest.json`` and dispatch.

    Returns the summary from the per-kind importer, plus a ``kind`` key.
    Bundles without a ``kind`` field are treated as legacy experiment
    bundles.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        manifest = json.loads(z.read("manifest.json"))
    kind = manifest.get("kind") or "experiment"
    if kind == "global":
        return import_bundle(con, zip_bytes, overwrite=overwrite)
    if kind == "dataset":
        return import_dataset(con, zip_bytes, overwrite=overwrite)
    if kind == "analysis":
        return import_analysis(con, zip_bytes, overwrite=overwrite)
    if kind == "experiment":
        out = import_experiment(con, zip_bytes, overwrite=overwrite)
        out.setdefault("kind", "experiment")
        return out
    raise ValueError(f"unknown bundle kind: {kind!r}")


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
