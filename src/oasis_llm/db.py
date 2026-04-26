"""DuckDB schema and connection helpers."""
from __future__ import annotations

from pathlib import Path
import duckdb

DB_PATH = Path("data/llm_runs.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    config_json   TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trials (
    run_id        TEXT NOT NULL,
    image_id      TEXT NOT NULL,
    dimension     TEXT NOT NULL,
    sample_idx    INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    rating        INTEGER,
    raw_response  TEXT,
    reasoning     TEXT,
    prompt_hash   TEXT,
    latency_ms    INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      DOUBLE,
    error         TEXT,
    attempts      INTEGER DEFAULT 0,
    claimed_at    TIMESTAMP,
    completed_at  TIMESTAMP,
    finish_reason TEXT,
    response_id   TEXT,
    trace_id      TEXT,
    PRIMARY KEY (run_id, image_id, dimension, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(run_id, status);

CREATE TABLE IF NOT EXISTS captions (
    image_id      TEXT NOT NULL,
    captioner     TEXT NOT NULL,
    caption       TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id, captioner)
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    status        TEXT NOT NULL DEFAULT 'draft',  -- draft|approved|archived
    source        TEXT NOT NULL DEFAULT 'generated',  -- generated|builtin|imported
    generation_params TEXT,                       -- JSON: {strategy, n, seed, ...}
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_id    TEXT NOT NULL,
    image_id      TEXT NOT NULL,
    excluded      BOOLEAN NOT NULL DEFAULT FALSE,
    note          TEXT,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_images_active
    ON dataset_images(dataset_id, excluded);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    dataset_id    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'draft',  -- draft|running|done|archived
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_configs (
    experiment_id TEXT NOT NULL,
    config_name   TEXT NOT NULL,
    config_json   TEXT NOT NULL,         -- full RunConfig as JSON
    run_id        TEXT NOT NULL UNIQUE,  -- backing runs.run_id (composed: {exp}__{cfg})
    position      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (experiment_id, config_name)
);

CREATE INDEX IF NOT EXISTS idx_experiment_configs_run
    ON experiment_configs(run_id);

CREATE TABLE IF NOT EXISTS run_processes (
    run_id     TEXT PRIMARY KEY,
    pid        INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    dataset_id   TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_id TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    label       TEXT,
    added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (analysis_id, run_id)
);
"""


def connect(db_path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    # Lightweight migrations for columns added to existing schemas.
    _migrate(con)
    # Seed built-in datasets (idempotent). Imported lazily to avoid cycles.
    from .datasets import seed_builtins
    try:
        seed_builtins(con)
    except Exception:
        # If seeding fails (e.g. images dir missing in a test env), keep going.
        pass
    return con


def _migrate(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotent column-add migrations for older DBs."""
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'trials'"
    ).fetchall()}
    if "trace_id" not in cols:
        con.execute("ALTER TABLE trials ADD COLUMN trace_id TEXT")
    # Run-queue columns on `runs` table.
    runs_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'"
    ).fetchall()}
    if "queued_at" not in runs_cols:
        con.execute("ALTER TABLE runs ADD COLUMN queued_at TIMESTAMP")
    if "queue_priority" not in runs_cols:
        con.execute("ALTER TABLE runs ADD COLUMN queue_priority INTEGER DEFAULT 0")
    # Singleton settings table for queue paused-toggle + scheduler heartbeat.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduler_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def lock_holder_pid(db_path: Path | str = DB_PATH) -> int | None:
    """Return the PID of the OS process holding an exclusive DuckDB lock,
    or ``None`` if the file is not locked / not lockable.

    DuckDB raises ``IOException`` with a message like:

        Could not set lock on file "...": Conflicting lock is held in
        /opt/homebrew/.../Python (PID 44805) by user desmond.

    We parse the PID out of that string. If a fresh connection succeeds,
    we close it and return ``None``.
    """
    import re
    try:
        probe = duckdb.connect(str(db_path))
        probe.close()
        return None
    except duckdb.IOException as e:  # type: ignore[attr-defined]
        m = re.search(r"PID\s+(\d+)", str(e))
        return int(m.group(1)) if m else -1
    except Exception:
        return None

