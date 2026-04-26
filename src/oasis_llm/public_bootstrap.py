"""Bootstrap a DuckDB from committed `.zip` bundles for the public viewer.

Used by the public results-viewer entrypoint when running on a host
(Streamlit Cloud, Hugging Face Spaces, etc.) where ``data/llm_runs.duckdb``
is gitignored and therefore absent on the deployed filesystem.

Flow:

    1. If ``db_path`` already exists and contains data (any rows in ``runs``),
       return early — assume a real desktop install or a previously-bootstrapped
       cloud install. Idempotent.
    2. Otherwise, open a fresh DB at ``db_path`` (which calls ``ensure_schema``)
       and call :func:`bundles.import_any` on each ``*.zip`` in ``bundles_dir``,
       sorted by filename so iteration is deterministic.
    3. Return a summary dict for the caller to log.

This module never raises on a single bundle failure — failed imports are
recorded under ``failures`` and the next bundle is attempted, mirroring the
behaviour of :func:`bundles.import_bundle`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import bundles, db


def _is_populated(db_path: Path) -> bool:
    """True if the DB exists and has at least one row in ``runs``."""
    if not db_path.exists():
        return False
    try:
        con = db.connect(db_path)
        try:
            count = con.execute("SELECT COUNT(*) FROM runs").fetchone()
            return bool(count and count[0] > 0)
        finally:
            con.close()
    except Exception:
        return False


def bootstrap_from_bundles(
    db_path: Path | str = db.DB_PATH,
    bundles_dir: Path | str = "data/public",
) -> dict[str, Any]:
    """Populate ``db_path`` from every ``*.zip`` in ``bundles_dir``.

    Idempotent — if the DB already has data, returns immediately with
    ``{"skipped": True, ...}``. Otherwise creates the schema, imports each
    bundle in filename order, and returns the aggregated summary.
    """
    db_path = Path(db_path)
    bundles_dir = Path(bundles_dir)

    if _is_populated(db_path):
        return {
            "skipped": True,
            "reason": "db already populated",
            "db_path": str(db_path),
        }

    # Clear any orphan WAL pointing at a missing main file.
    wal = db_path.with_suffix(db_path.suffix + ".wal")
    if wal.exists() and not db_path.exists():
        wal.unlink()

    if not bundles_dir.exists():
        return {
            "skipped": False,
            "imported": [],
            "failures": [],
            "warning": f"bundles_dir does not exist: {bundles_dir}",
            "db_path": str(db_path),
        }

    zips = sorted(bundles_dir.glob("*.zip"))

    con = db.connect(db_path)
    imported: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for zp in zips:
            try:
                summary = bundles.import_any(con, zp.read_bytes(), overwrite=False)
                imported.append({"file": zp.name, "summary": summary})
            except Exception as e:
                failures.append(
                    {"file": zp.name, "error": f"{type(e).__name__}: {e}"}
                )
    finally:
        con.close()

    return {
        "skipped": False,
        "imported": imported,
        "failures": failures,
        "n_bundles": len(zips),
        "db_path": str(db_path),
    }
