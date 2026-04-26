"""Idempotent trial enqueue."""
from __future__ import annotations

import json

import duckdb

from .config import RunConfig
from .images import select_image_set


def upsert_run(con: duckdb.DuckDBPyConnection, cfg: RunConfig) -> tuple[str, bool]:
    """Insert run row if missing. Returns (run_id, created)."""
    cfg_json = cfg.model_dump_json()
    cfg_hash = cfg.canonical_hash()
    run_id = cfg.name
    existing = con.execute(
        "SELECT config_hash FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if existing is None:
        con.execute(
            "INSERT INTO runs (run_id, config_json, config_hash, status) VALUES (?, ?, ?, 'created')",
            [run_id, cfg_json, cfg_hash],
        )
        return run_id, True
    if existing[0] != cfg_hash:
        raise RuntimeError(
            f"Run '{run_id}' exists with different config hash "
            f"(stored={existing[0]}, new={cfg_hash}). Use a new --name or --new-run."
        )
    return run_id, False


def enqueue_trials(con: duckdb.DuckDBPyConnection, cfg: RunConfig) -> int:
    """Insert pending trial rows. Returns rows actually inserted."""
    image_ids = select_image_set(cfg.image_set, con=con)
    rows = []
    for img in image_ids:
        for dim in cfg.dimensions:
            for s in range(cfg.samples_per_image):
                rows.append((cfg.name, img, dim, s, "pending", 0))
    # Use a temp table + anti-join for idempotency
    con.execute(
        "CREATE TEMP TABLE _staging (run_id TEXT, image_id TEXT, dimension TEXT, sample_idx INTEGER, status TEXT, attempts INTEGER)"
    )
    con.executemany(
        "INSERT INTO _staging VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    inserted = con.execute(
        """
        INSERT INTO trials (run_id, image_id, dimension, sample_idx, status, attempts)
        SELECT s.run_id, s.image_id, s.dimension, s.sample_idx, s.status, s.attempts
        FROM _staging s
        LEFT JOIN trials t USING (run_id, image_id, dimension, sample_idx)
        WHERE t.run_id IS NULL
        RETURNING 1
        """
    ).fetchall()
    con.execute("DROP TABLE _staging")
    return len(inserted)
