"""Scheduler daemon: dequeues queued runs and spawns them as subprocesses.

Polling loop:
  1. Heartbeat
  2. Reap any dead/finished runs from `run_processes`
  3. If paused → skip
  4. While slots_free > 0 and queue non-empty → dequeue + spawn

Spawning uses the new ``oasis-llm run-id <run_id>`` CLI, which loads the
config from the runs table (no YAML needed for experiment-created runs).

The daemon opens a fresh writable DuckDB connection PER TICK and closes it
before sleeping. This is critical: the spawned runner subprocess needs to
acquire the same DuckDB lock to do its work, so we can never hold the lock
across the sleep window.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import duckdb

from . import queue as _q


def _alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _running_count(con: duckdb.DuckDBPyConnection) -> int:
    """How many run subprocesses are still alive?"""
    rows = con.execute("SELECT run_id, pid FROM run_processes").fetchall()
    n = 0
    for run_id, pid in rows:
        if _alive(int(pid)):
            n += 1
        else:
            con.execute("DELETE FROM run_processes WHERE run_id=?", [run_id])
    return n


def _spawn(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    """Spawn `oasis-llm run-id <run_id>` as a detached child."""
    pid = _spawn_detached(run_id)
    con.execute(
        "INSERT OR REPLACE INTO run_processes (run_id, pid) VALUES (?, ?)",
        [run_id, pid],
    )
    return pid


def _spawn_detached(run_id: str) -> int:
    """Just spawn the subprocess and return its pid. No DB I/O."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id.replace('/', '_')}.log"
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "oasis_llm.cli", "run-id", run_id],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


_stop_requested = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop_requested  # noqa: PLW0603
    _stop_requested = True


def run_daemon(
    db_path: str | Path | None = None,
    *,
    poll_interval_s: float = 5.0,
    max_iterations: int | None = None,
) -> None:
    """Run the scheduler loop. Blocks until SIGTERM/SIGINT or max_iterations.

    ``max_iterations`` is for tests.
    """
    from .db import connect

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pid = os.getpid()
    print(f"[scheduler] starting (pid={pid}, poll={poll_interval_s}s)", flush=True)

    iterations = 0
    while not _stop_requested:
        iterations += 1
        # Open a fresh connection per tick so the spawned runner can acquire
        # the DuckDB lock once we release it. Holding the connection across
        # the sleep would deadlock every spawn.
        con = None
        try:
            con = connect(db_path) if db_path else connect()
            _q.mark_heartbeat(con, pid)

            running = _running_count(con)
            cap = _q.max_parallel(con)
            paused = _q.is_paused(con)

            if not paused:
                while running < cap:
                    run_id = _q.dequeue_next(con)
                    if run_id is None:
                        break
                    # IMPORTANT: close the connection BEFORE spawning so the
                    # child process can immediately acquire the DuckDB lock
                    # (otherwise the runner crashes with "Conflicting lock").
                    con.close()
                    con = None
                    spawn_pid = _spawn_detached(run_id)
                    print(
                        f"[scheduler] dequeued {run_id} → pid {spawn_pid}",
                        flush=True,
                    )
                    # Re-open briefly to record the pid, then close again.
                    con = connect(db_path) if db_path else connect()
                    con.execute(
                        "INSERT OR REPLACE INTO run_processes (run_id, pid) VALUES (?, ?)",
                        [run_id, spawn_pid],
                    )
                    con.close()
                    con = None
                    running += 1
                    # Give the freshly-spawned runner a head start to grab the
                    # lock before we re-open for the next iteration.
                    time.sleep(2.0)
                    break  # one spawn per tick keeps things simple
        except Exception as e:  # noqa: BLE001
            print(f"[scheduler] tick error: {type(e).__name__}: {e}", flush=True)
        finally:
            if con is not None:
                try: con.close()
                except Exception: pass

        if max_iterations is not None and iterations >= max_iterations:
            break
        # Sleep in small chunks so signal handlers can interrupt promptly.
        slept = 0.0
        while slept < poll_interval_s and not _stop_requested:
            time.sleep(min(0.5, poll_interval_s - slept))
            slept += 0.5

    print("[scheduler] exiting", flush=True)
    try:
        con.close()
    except Exception:
        pass
