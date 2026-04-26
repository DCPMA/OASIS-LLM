"""Validation tests for ``oasis_llm.estimates``.

These tests stand up an isolated in-memory DuckDB seeded with a small
synthetic schema (only the columns ``estimates`` reads) so we can verify
the math without depending on a real run.
"""
from __future__ import annotations

import json
import math

import duckdb
import pytest

from oasis_llm import estimates as e


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the minimum schema ``estimates`` queries."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE runs (
            run_id      TEXT PRIMARY KEY,
            config_json TEXT
        );
        CREATE TABLE trials (
            run_id        TEXT NOT NULL,
            image_id      TEXT NOT NULL,
            dimension     TEXT NOT NULL,
            sample_idx   INTEGER NOT NULL,
            status        TEXT NOT NULL,
            cost_usd      DOUBLE,
            latency_ms    INTEGER,
            completed_at  TIMESTAMP
        );
    """)
    return c


def _add_run(con, run_id: str, provider: str, model: str, *, max_concurrency: int = 1) -> None:
    con.execute(
        "INSERT INTO runs VALUES (?, ?)",
        [run_id, json.dumps({
            "provider": provider, "model": model,
            "max_concurrency": max_concurrency,
        })],
    )


def _add_trials(con, run_id: str, costs: list[float | None], latencies_ms: list[int | None] | None = None) -> None:
    """Insert N done trials. ``costs[i] is None`` ⇒ free trial (Ollama)."""
    if latencies_ms is None:
        latencies_ms = [1000] * len(costs)
    assert len(costs) == len(latencies_ms)
    for i, (c, lat) in enumerate(zip(costs, latencies_ms)):
        con.execute(
            "INSERT INTO trials VALUES (?, ?, ?, ?, 'done', ?, ?, NULL)",
            [run_id, f"img{i}", "valence", 0, c, lat],
        )


def _add_pending(con, run_id: str, n: int) -> None:
    for i in range(n):
        con.execute(
            "INSERT INTO trials VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL)",
            [run_id, f"pending{i}", "valence", 0],
        )


# ── trials_for_config ───────────────────────────────────────────────────────


def test_trials_for_config_basic():
    assert e.trials_for_config(10, 2, 5) == 100
    assert e.trials_for_config(0, 2, 5) == 0
    assert e.trials_for_config(30, 2, 1) == 60


# ── _empirical_cost_per_trial ──────────────────────────────────────────────


def test_empirical_returns_none_below_min_history(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.01, 0.02])  # only 2, threshold is 3
    mean, std, n = e._empirical_cost_per_trial(con, "openrouter", "gpt-4o")
    assert mean is None
    assert n == 2


def test_empirical_computes_mean_and_std(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.10, 0.20, 0.30, 0.40])  # mean=0.25, sample sd≈0.129
    mean, std, n = e._empirical_cost_per_trial(con, "openrouter", "gpt-4o")
    assert n == 4
    assert mean == pytest.approx(0.25, abs=1e-9)
    assert std == pytest.approx(math.sqrt(sum((x - 0.25) ** 2 for x in [0.10, 0.20, 0.30, 0.40]) / 3), rel=1e-6)


def test_empirical_filters_by_provider_and_model(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_run(con, "r2", "openrouter", "claude-3.5-sonnet")
    _add_trials(con, "r1", [0.10, 0.20, 0.30])
    _add_trials(con, "r2", [9.0, 9.0, 9.0])
    mean, _, n = e._empirical_cost_per_trial(con, "openrouter", "gpt-4o")
    assert n == 3
    assert mean == pytest.approx(0.20)


def test_empirical_skips_null_costs(con):
    _add_run(con, "r1", "ollama", "gemma3:12b")
    _add_trials(con, "r1", [None, None, None, None])  # local model, no cost
    mean, _, n = e._empirical_cost_per_trial(con, "ollama", "gemma3:12b")
    assert mean is None
    assert n == 0  # NULL costs aren't counted


def test_empirical_aggregates_across_runs(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_run(con, "r2", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.10, 0.20])
    _add_trials(con, "r2", [0.30, 0.40])  # combined: 4 trials, mean 0.25
    mean, _, n = e._empirical_cost_per_trial(con, "openrouter", "gpt-4o")
    assert n == 4
    assert mean == pytest.approx(0.25)


# ── estimate_cost_per_trial ────────────────────────────────────────────────


def test_ollama_is_always_free(con):
    est = e.estimate_cost_per_trial(con, "ollama", "gemma3:12b")
    assert est.source == "free"
    assert est.mean_usd == 0.0


def test_history_beats_fallback(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.05] * 5)
    est = e.estimate_cost_per_trial(con, "openrouter", "gpt-4o")
    assert est.source == "history"
    assert est.mean_usd == pytest.approx(0.05)
    assert est.n_trials == 5


def test_unknown_model_returns_unknown(con):
    est = e.estimate_cost_per_trial(con, "openrouter", "made-up/nonexistent-model-xyz")
    assert est.source in ("unknown", "fallback")  # fallback only if litellm has it


# ── project_run ────────────────────────────────────────────────────────────


def test_project_returns_none_for_unknown_run(con):
    # No-op: project_run never returns None today (always builds something
    # from counts). Use unknown run_id and check sane defaults.
    proj = e.project_run(con, "doesnt-exist")
    assert proj is not None
    assert proj.done == 0 and proj.total == 0
    assert proj.cost_per_trial_mean_usd is None
    assert proj.eta_seconds is None


def test_project_in_progress_run(con):
    _add_run(con, "r1", "openrouter", "gpt-4o", max_concurrency=4)
    _add_trials(con, "r1", [0.01, 0.02, 0.03], latencies_ms=[2000, 2000, 2000])
    _add_pending(con, "r1", 17)  # 3 done + 17 pending = 20 total
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.done == 3
    assert proj.total == 20
    assert proj.remaining == 17
    assert proj.cost_so_far_usd == pytest.approx(0.06)
    assert proj.cost_per_trial_mean_usd == pytest.approx(0.02)
    # projected = spent + remaining * mean = 0.06 + 17 * 0.02 = 0.40
    assert proj.projected_total_usd == pytest.approx(0.40)
    # eta = remaining * mean_latency / concurrency = 17 * 2.0 / 4 = 8.5 s
    assert proj.eta_seconds == pytest.approx(8.5)
    assert proj.latency_recent_mean_s == pytest.approx(2.0)


def test_project_completed_run_has_zero_remaining(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.05, 0.05, 0.05])
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.remaining == 0
    assert proj.eta_seconds is None  # Nothing left to wait for
    assert proj.projected_total_usd == pytest.approx(0.15)


def test_project_local_run_no_cost(con):
    _add_run(con, "r1", "ollama", "gemma3:12b", max_concurrency=2)
    _add_trials(con, "r1", [None, None, None], latencies_ms=[5000, 5000, 5000])
    _add_pending(con, "r1", 7)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.cost_so_far_usd == 0.0
    assert proj.cost_per_trial_mean_usd is None
    assert proj.projected_total_usd is None  # local = no cost projection
    # but we DO project ETA: 7 * 5.0 / 2 = 17.5s
    assert proj.eta_seconds == pytest.approx(17.5)


def test_project_concurrency_defaults_to_one(con):
    # Run row has no max_concurrency in config → fallback to 1.
    con.execute(
        "INSERT INTO runs VALUES (?, ?)",
        ["r1", json.dumps({"provider": "openrouter", "model": "gpt-4o"})],
    )
    _add_trials(con, "r1", [0.01], latencies_ms=[1000])
    _add_pending(con, "r1", 9)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.eta_seconds == pytest.approx(9.0)  # 9 * 1.0 / 1


def test_project_handles_zero_done(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_pending(con, "r1", 5)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.done == 0 and proj.total == 5
    assert proj.cost_per_trial_mean_usd is None
    assert proj.eta_seconds is None  # Can't extrapolate from zero data


# ── Formatting ──────────────────────────────────────────────────────────────


def test_format_cost():
    assert e.format_cost(None) == "—"
    assert e.format_cost(0) == "$0.00"
    assert e.format_cost(0.005) == "$0.0050"
    assert e.format_cost(0.42) == "$0.420"
    assert e.format_cost(12.5) == "$12.50"
    assert e.format_cost(1234.56) == "$1,234.56"


def test_format_cost_with_uncertainty():
    assert e.format_cost_with_uncertainty(None, None) == "—"
    assert e.format_cost_with_uncertainty(0.42, None) == "$0.420"
    assert e.format_cost_with_uncertainty(0.42, 0) == "$0.420"
    assert "±" in e.format_cost_with_uncertainty(0.42, 0.15)


def test_format_duration():
    assert e.format_duration(None) == "—"
    assert e.format_duration(-5) == "—"
    assert e.format_duration(45) == "45s"
    assert e.format_duration(125) == "2m 05s"
    assert e.format_duration(3725) == "1h 02m"
    assert e.format_duration(86400 + 7200) == "1d 02h"
