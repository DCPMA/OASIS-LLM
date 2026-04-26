"""Tests for `oasis_llm.public_bootstrap`.

Two cases:
1. Empty bundles_dir → no-op success (creates empty schema, no imports).
2. Already-populated DB → bootstrap is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis_llm import db, public_bootstrap


def test_bootstrap_creates_empty_db_when_no_bundles(tmp_path: Path) -> None:
    db_path = tmp_path / "llm_runs.duckdb"
    bundles_dir = tmp_path / "public"
    bundles_dir.mkdir()

    summary = public_bootstrap.bootstrap_from_bundles(db_path, bundles_dir)

    assert summary["skipped"] is False
    assert summary["imported"] == []
    assert summary["failures"] == []
    assert summary["n_bundles"] == 0
    assert db_path.exists()

    # Schema is in place — runs table exists and is empty.
    con = db.connect(db_path)
    try:
        n = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_bootstrap_skips_when_db_already_populated(tmp_path: Path) -> None:
    db_path = tmp_path / "llm_runs.duckdb"
    bundles_dir = tmp_path / "public"
    bundles_dir.mkdir()

    # Pre-populate the DB with one run row.
    con = db.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs (run_id, config_json, config_hash, status) "
            "VALUES (?, ?, ?, ?)",
            ["test_run", json.dumps({"name": "test"}), "deadbeef", "done"],
        )
    finally:
        con.close()

    summary = public_bootstrap.bootstrap_from_bundles(db_path, bundles_dir)

    assert summary["skipped"] is True
    assert "already populated" in summary["reason"]


def test_bootstrap_handles_missing_bundles_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "llm_runs.duckdb"
    bundles_dir = tmp_path / "does_not_exist"

    summary = public_bootstrap.bootstrap_from_bundles(db_path, bundles_dir)

    assert summary["skipped"] is False
    assert "warning" in summary
    assert summary["imported"] == []
