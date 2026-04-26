"""DuckDB schema and connection helpers."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
import duckdb

DB_PATH = Path("data/llm_runs.duckdb")
RECOVERY_DIR = DB_PATH.parent / "recovery"
RECOVERY_KEEP = 5  # keep last N rotating snapshots

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
    # Daily counter for OpenRouter ``:free`` tier requests (1000/day cap).
    # One row per UTC date; all :free models share a single quota.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS or_free_daily (
            day   DATE PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # User-facing favourites: a flat star table over heterogeneous entities.
    # ``entity_type`` is one of {dataset, experiment, analysis, run}; the
    # entity_id is the corresponding primary key (no FK, since the table
    # spans heterogeneous targets and we don't want star-on-delete cascades
    # to surprise anyone — un-stars are cheap to recompute).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            entity_type TEXT NOT NULL,
            entity_id   TEXT NOT NULL,
            starred_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note        TEXT,
            PRIMARY KEY (entity_type, entity_id)
        )
        """
    )


def ensure_schema(con: duckdb.DuckDBPyConnection) -> bool:
    """Idempotently apply the full schema + migrations on an existing connection.

    Safe to call as often as you like — every statement uses
    ``CREATE TABLE/INDEX IF NOT EXISTS`` or column-presence guards. Returns
    ``True`` when the connection accepted the DDL (read-write), ``False`` for
    a read-only connection or transient DDL conflicts. Downstream code that
    must tolerate missing tables (e.g. on a stale RO connection attached to
    a DB another process hasn't migrated yet) should still wrap its own
    queries defensively.
    """
    try:
        con.execute(SCHEMA)
        _migrate(con)
        return True
    except Exception:
        return False


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


def snapshot_db(
    con: duckdb.DuckDBPyConnection,
    *,
    label: str = "snapshot",
    db_path: Path | str = DB_PATH,
    keep: int = RECOVERY_KEEP,
) -> Path | None:
    """Checkpoint and copy the live DB file to ``data/recovery/`` with a
    timestamped, never-overwriting name. Rotates so only ``keep`` files
    remain (oldest deleted first).

    Returns the new snapshot path, or ``None`` if the source file is missing.
    Failures are swallowed and reported via return value of ``None`` only
    when the source file doesn't exist; other I/O errors propagate so the
    caller can decide whether to abort the upcoming write.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    # Flush WAL → main file so the snapshot is self-contained.
    try:
        con.execute("CHECKPOINT")
    except Exception:
        # Best-effort: even an un-checkpointed copy is better than nothing.
        pass
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    dst = RECOVERY_DIR / f"{db_path.name}.{ts}.{safe_label}.bak"
    shutil.copy2(db_path, dst)
    # Rotate: keep only the newest `keep` files matching this stem.
    pattern = f"{db_path.name}.*.bak"
    snaps = sorted(RECOVERY_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in snaps[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass
    return dst

