"""Run process control: start as background subprocess, pause/resume/cancel via DB.

The runner respects ``runs.status`` between trial claims:
  - ``pending``/``running``: keep working
  - ``paused`` / ``cancelled``: workers stop claiming and exit cleanly

This module spawns the CLI ``oasis-llm run <config>`` as a detached subprocess,
records its PID in ``run_processes``, and provides helpers for the dashboard.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import duckdb


def is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_pid(con: duckdb.DuckDBPyConnection, run_id: str) -> int | None:
    row = con.execute(
        "SELECT pid FROM run_processes WHERE run_id=?", [run_id]
    ).fetchone()
    if row is None:
        return None
    pid = int(row[0])
    if not is_alive(pid):
        con.execute("DELETE FROM run_processes WHERE run_id=?", [run_id])
        return None
    return pid


def start(con: duckdb.DuckDBPyConnection, run_id: str, config_path: str | Path) -> int:
    """Start (or restart) a run as a detached background process. Returns PID."""
    existing = get_pid(con, run_id)
    if existing is not None:
        raise RuntimeError(f"Run {run_id} already running (PID {existing}).")
    # Clear any stale paused/cancelled status so the runner can claim trials.
    con.execute("UPDATE runs SET status='pending' WHERE run_id=?", [run_id])
    log_dir = Path("data/logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id.replace('/', '_')}.log"
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "oasis_llm.cli", "run", str(config_path)],
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    con.execute(
        "INSERT OR REPLACE INTO run_processes (run_id, pid) VALUES (?, ?)",
        [run_id, proc.pid],
    )
    return proc.pid


def pause(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """Soft-pause: workers stop claiming new trials at the next check."""
    con.execute("UPDATE runs SET status='paused' WHERE run_id=?", [run_id])


def resume(con: duckdb.DuckDBPyConnection, run_id: str, config_path: str | Path) -> int:
    """Resume a paused run. Spawns a fresh subprocess if the previous one exited."""
    pid = get_pid(con, run_id)
    con.execute("UPDATE runs SET status='pending' WHERE run_id=?", [run_id])
    if pid is None:
        return start(con, run_id, config_path)
    return pid


def cancel(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """Mark cancelled and kill the subprocess if alive."""
    con.execute("UPDATE runs SET status='cancelled' WHERE run_id=?", [run_id])
    pid = get_pid(con, run_id)
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass
        con.execute("DELETE FROM run_processes WHERE run_id=?", [run_id])


def reset_failed(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    """Reset all failed trials in a run back to pending. Returns count."""
    n = con.execute(
        "SELECT count(*) FROM trials WHERE run_id=? AND status='failed'", [run_id]
    ).fetchone()[0]
    con.execute(
        "UPDATE trials SET status='pending', error=NULL, attempts=0 WHERE run_id=? AND status='failed'",
        [run_id],
    )
    return int(n)
