"""Run queue: lightweight wrapper around `runs.status='queued'`.

Status flow:
  queued  →  running  →  done | failed | cancelled
            (pending may also appear; see runner)

A queued run is a run whose status is exactly ``queued`` (with ``queued_at``
set). The scheduler picks the oldest queued run when a parallelism slot is
free, sets its status to ``pending``, and spawns a subprocess.

Concepts:
- The queue is just a view over ``runs``. No separate table.
- ``queue_priority`` sorts within queued items (DESC, ties by ``queued_at`` ASC).
- ``scheduler_state`` table holds singleton k/v: ``paused``, ``max_parallel``,
  ``last_heartbeat``, ``scheduler_pid``.

Public API:
- enqueue(con, run_id, priority=0)
- dequeue_next(con) -> run_id | None       # also moves status → 'pending'
- cancel_queued(con, run_id)
- list_queued(con) -> list[QueueItem]
- bump(con, run_id, direction)             # +1 / -1 priority
- enqueue_experiment(con, experiment_id)   # all configs in position order
- is_paused(con) / set_paused(con, paused)
- max_parallel(con) / set_max_parallel(con, n)
- mark_heartbeat(con, pid) / heartbeat_age_s(con)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import duckdb


@dataclass(frozen=True)
class QueueItem:
    run_id: str
    queued_at: object
    priority: int
    config_name: str | None
    experiment_id: str | None


# ---------------------------------------------------------------------------
# core queue operations
# ---------------------------------------------------------------------------
def enqueue(
    con: duckdb.DuckDBPyConnection, run_id: str, *, priority: int = 0
) -> None:
    """Mark a run as queued. Idempotent on already-queued runs."""
    row = con.execute(
        "SELECT status FROM runs WHERE run_id=?", [run_id]
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown run: {run_id}")
    if row[0] == "running":
        raise RuntimeError(f"run {run_id} is currently running; cannot queue")
    con.execute(
        """
        UPDATE runs
        SET status='queued',
            queued_at=COALESCE(queued_at, CURRENT_TIMESTAMP),
            queue_priority=?
        WHERE run_id=?
        """,
        [priority, run_id],
    )


def cancel_queued(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """Remove a run from the queue (only if currently queued)."""
    con.execute(
        "UPDATE runs SET status='pending', queued_at=NULL WHERE run_id=? AND status='queued'",
        [run_id],
    )


def dequeue_next(con: duckdb.DuckDBPyConnection) -> str | None:
    """Atomically pick the next queued run and flip its status to 'pending'.

    Returns the run_id, or None if the queue is empty.
    Caller is responsible for spawning the subprocess afterwards.
    """
    row = con.execute(
        """
        SELECT run_id FROM runs
        WHERE status='queued'
        ORDER BY queue_priority DESC, queued_at ASC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    run_id = row[0]
    con.execute(
        "UPDATE runs SET status='pending', queued_at=NULL WHERE run_id=?",
        [run_id],
    )
    return run_id


def list_queued(con: duckdb.DuckDBPyConnection) -> list[QueueItem]:
    rows = con.execute(
        """
        SELECT r.run_id, r.queued_at, r.queue_priority,
               ec.config_name, ec.experiment_id
        FROM runs r
        LEFT JOIN experiment_configs ec USING(run_id)
        WHERE r.status='queued'
        ORDER BY r.queue_priority DESC, r.queued_at ASC
        """
    ).fetchall()
    return [QueueItem(*r) for r in rows]


def bump(con: duckdb.DuckDBPyConnection, run_id: str, direction: int) -> None:
    """Adjust priority by +1 (up) or -1 (down). Only affects queued items."""
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1")
    con.execute(
        "UPDATE runs SET queue_priority=COALESCE(queue_priority, 0)+? "
        "WHERE run_id=? AND status='queued'",
        [direction, run_id],
    )


def enqueue_experiment(
    con: duckdb.DuckDBPyConnection, experiment_id: str
) -> list[str]:
    """Queue every config of an experiment, preserving position order.

    Configs that have already completed (status='done' with no pending trials)
    are skipped. Configs currently running are also skipped. Returns the list
    of run_ids that were enqueued.
    """
    rows = con.execute(
        """
        SELECT ec.run_id, ec.position, r.status,
               (SELECT COUNT(*) FROM trials t
                WHERE t.run_id=ec.run_id AND t.status IN ('pending','failed')) AS pending
        FROM experiment_configs ec
        JOIN runs r USING(run_id)
        WHERE ec.experiment_id=?
        ORDER BY ec.position, ec.config_name
        """,
        [experiment_id],
    ).fetchall()
    enqueued = []
    # Use position-based descending priority so earlier configs run first
    # within the experiment (higher priority dequeued first).
    n = len(rows)
    for run_id, position, status, pending in rows:
        if status == "running":
            continue
        if status == "done" and (pending or 0) == 0:
            continue
        priority = (n - int(position or 0))  # earlier position → higher priority
        enqueue(con, run_id, priority=priority)
        enqueued.append(run_id)
    return enqueued


# ---------------------------------------------------------------------------
# scheduler_state singleton
# ---------------------------------------------------------------------------
def _get(con: duckdb.DuckDBPyConnection, key: str) -> str | None:
    row = con.execute(
        "SELECT value FROM scheduler_state WHERE key=?", [key]
    ).fetchone()
    return row[0] if row else None


def _set(con: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO scheduler_state (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
        [key, value],
    )


def is_paused(con: duckdb.DuckDBPyConnection) -> bool:
    return (_get(con, "paused") or "0") == "1"


def set_paused(con: duckdb.DuckDBPyConnection, paused: bool) -> None:
    _set(con, "paused", "1" if paused else "0")


def max_parallel(con: duckdb.DuckDBPyConnection) -> int:
    raw = _get(con, "max_parallel")
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


def set_max_parallel(con: duckdb.DuckDBPyConnection, n: int) -> None:
    _set(con, "max_parallel", str(max(1, int(n))))


def mark_heartbeat(con: duckdb.DuckDBPyConnection, pid: int) -> None:
    _set(con, "scheduler_pid", str(int(pid)))
    _set(con, "last_heartbeat", str(time.time()))


def heartbeat_age_s(con: duckdb.DuckDBPyConnection) -> float | None:
    raw = _get(con, "last_heartbeat")
    if not raw:
        return None
    try:
        return time.time() - float(raw)
    except ValueError:
        return None


def scheduler_pid(con: duckdb.DuckDBPyConnection) -> int | None:
    raw = _get(con, "scheduler_pid")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None
