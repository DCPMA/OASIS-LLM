"""Run lifecycle administration: deletion + bulk purge.

Separated from runner.py so the dashboard can import without pulling litellm.
"""
from __future__ import annotations

import os
from typing import Iterable

import duckdb


def collect_trace_ids(con: duckdb.DuckDBPyConnection, run_id: str) -> list[str]:
    """All non-null trace_ids associated with a run's trials."""
    rows = con.execute(
        "SELECT DISTINCT trace_id FROM trials WHERE run_id=? AND trace_id IS NOT NULL",
        [run_id],
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def delete_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    delete_langfuse: bool = False,
) -> dict:
    """Cascade-delete a run.

    Removes trials, run_processes, analysis_runs, experiment_configs link, and
    the runs row. Optionally also deletes the matching Langfuse traces.

    Returns a summary dict {trials, langfuse_traces_deleted, langfuse_error}.
    """
    trace_ids = collect_trace_ids(con, run_id) if delete_langfuse else []
    n_trials = con.execute(
        "SELECT count(*) FROM trials WHERE run_id=?", [run_id]
    ).fetchone()[0]

    con.execute("DELETE FROM trials             WHERE run_id=?", [run_id])
    con.execute("DELETE FROM run_processes      WHERE run_id=?", [run_id])
    con.execute("DELETE FROM experiment_configs WHERE run_id=?", [run_id])
    con.execute("DELETE FROM analysis_runs      WHERE run_id=?", [run_id])
    con.execute("DELETE FROM runs               WHERE run_id=?", [run_id])

    summary = {"run_id": run_id, "trials": int(n_trials),
               "langfuse_traces_deleted": 0, "langfuse_error": None}

    if delete_langfuse and trace_ids:
        deleted, err = _delete_langfuse_traces(trace_ids)
        summary["langfuse_traces_deleted"] = deleted
        summary["langfuse_error"] = err
    return summary


def purge_runs(
    con: duckdb.DuckDBPyConnection,
    run_ids: Iterable[str],
    *,
    delete_langfuse: bool = False,
    drop_empty_experiments: bool = True,
) -> list[dict]:
    """Delete each run; optionally drop now-empty experiment shells."""
    summaries = []
    for rid in run_ids:
        summaries.append(delete_run(con, rid, delete_langfuse=delete_langfuse))
    if drop_empty_experiments:
        con.execute(
            """
            DELETE FROM experiments
            WHERE experiment_id NOT IN (
                SELECT DISTINCT experiment_id FROM experiment_configs
            )
            """
        )
    return summaries


def _delete_langfuse_traces(trace_ids: list[str]) -> tuple[int, str | None]:
    """Best-effort delete via Langfuse SDK. Returns (count, error_or_None)."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return 0, "Langfuse credentials not set"
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        # Batch in chunks to avoid huge payloads.
        n = 0
        chunk = 500
        for i in range(0, len(trace_ids), chunk):
            lf.client.trace.delete_multiple(trace_ids=trace_ids[i:i + chunk])
            n += len(trace_ids[i:i + chunk])
        return n, None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"
