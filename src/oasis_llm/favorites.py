"""Favourites / starred entities.

A flat ``favorites`` table tags arbitrary entities (datasets, experiments,
analyses, runs) so list pages can offer a "⭐ Starred only" filter and the
global Export page can bundle user-curated subsets.

Design notes
------------
* The table has no FK so star-on-delete cascades do not surprise users; a
  stale star is harmless and trivially cleaned up by :func:`prune`.
* All read paths return tuples / dicts so callers can dataframe-fy them
  without taking a dependency on this module's internals.
"""
from __future__ import annotations

from typing import Iterable

import duckdb

ENTITY_TYPES = ("dataset", "experiment", "analysis", "run")


def _validate_type(entity_type: str) -> None:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(
            f"unknown entity_type: {entity_type!r} (expected one of {ENTITY_TYPES})"
        )


def _ensure_table(con: duckdb.DuckDBPyConnection) -> bool:
    """Lazy CREATE for older DBs / read-only connections.

    Mirrors ``rate_limit.OpenRouterFreeTierLimiter._ensure_table``: tries to
    CREATE the table; on RO connections (which raise on DDL) falls back to a
    SELECT probe and returns False if the table genuinely isn't there. Reads
    treat False as "no favourites yet"; writes raise so callers see the
    underlying problem.
    """
    try:
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
        return True
    except Exception:
        try:
            con.execute("SELECT 1 FROM favorites LIMIT 1")
            return True
        except Exception:
            return False


def is_starred(con: duckdb.DuckDBPyConnection, entity_type: str, entity_id: str) -> bool:
    _validate_type(entity_type)
    if not _ensure_table(con):
        return False
    row = con.execute(
        "SELECT 1 FROM favorites WHERE entity_type=? AND entity_id=?",
        [entity_type, entity_id],
    ).fetchone()
    return row is not None


def starred_set(con: duckdb.DuckDBPyConnection, entity_type: str) -> set[str]:
    """Return the set of starred entity_ids for ``entity_type`` (cheap O(1) membership)."""
    _validate_type(entity_type)
    if not _ensure_table(con):
        return set()
    rows = con.execute(
        "SELECT entity_id FROM favorites WHERE entity_type=?", [entity_type],
    ).fetchall()
    return {r[0] for r in rows}


def add(
    con: duckdb.DuckDBPyConnection, entity_type: str, entity_id: str,
    *, note: str | None = None,
) -> None:
    _validate_type(entity_type)
    _ensure_table(con)
    con.execute(
        """
        INSERT INTO favorites (entity_type, entity_id, note) VALUES (?, ?, ?)
        ON CONFLICT (entity_type, entity_id) DO UPDATE SET note=excluded.note
        """,
        [entity_type, entity_id, note],
    )


def remove(con: duckdb.DuckDBPyConnection, entity_type: str, entity_id: str) -> None:
    _validate_type(entity_type)
    if not _ensure_table(con):
        return
    con.execute(
        "DELETE FROM favorites WHERE entity_type=? AND entity_id=?",
        [entity_type, entity_id],
    )


def toggle(con: duckdb.DuckDBPyConnection, entity_type: str, entity_id: str) -> bool:
    """Flip the star. Returns the new state (True = starred)."""
    if is_starred(con, entity_type, entity_id):
        remove(con, entity_type, entity_id)
        return False
    add(con, entity_type, entity_id)
    return True


def list_starred(
    con: duckdb.DuckDBPyConnection, entity_type: str | None = None,
) -> list[dict]:
    """Enumerate stars, newest-first."""
    if entity_type is not None:
        _validate_type(entity_type)
    if not _ensure_table(con):
        return []
    if entity_type is not None:
        rows = con.execute(
            "SELECT entity_type, entity_id, starred_at, note "
            "FROM favorites WHERE entity_type=? ORDER BY starred_at DESC",
            [entity_type],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT entity_type, entity_id, starred_at, note "
            "FROM favorites ORDER BY starred_at DESC"
        ).fetchall()
    return [
        {"entity_type": r[0], "entity_id": r[1], "starred_at": r[2], "note": r[3]}
        for r in rows
    ]


def prune(con: duckdb.DuckDBPyConnection) -> int:
    """Remove stars whose target no longer exists. Returns count removed."""
    if not _ensure_table(con):
        return 0
    targets = {
        "dataset":    "SELECT dataset_id    FROM datasets",
        "experiment": "SELECT experiment_id FROM experiments",
        "analysis":   "SELECT analysis_id   FROM analyses",
        "run":        "SELECT run_id        FROM runs",
    }
    n = 0
    for et, q in targets.items():
        try:
            live = {r[0] for r in con.execute(q).fetchall()}
        except Exception:
            # Source table missing on this connection — skip rather than
            # nuke stars for an entity type we can't verify.
            continue
        starred = starred_set(con, et)
        for stale in starred - live:
            remove(con, et, stale)
            n += 1
    return n
