"""Validation tests for ``oasis_llm.estimates``.

Pre-launch cost estimates are pricing-only (no DB) and use:
    1. ``free`` for Ollama
    2. live OpenRouter pricing × calibrated tokens (560 in / 35 out)
    3. LiteLLM static pricing × calibrated tokens
    4. ``unknown``

Live run projections still hit the DB — those tests stand up an
in-memory DuckDB with the minimum schema.
"""
from __future__ import annotations

import json

import duckdb
import pytest

from oasis_llm import estimates as e


# ─────────────────────────────────────────────────────────────────────────────
# Cost-per-trial (pricing only, no DB)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_or_cache():
    """Reset the OpenRouter price cache before every test."""
    e._or_price_cache = None
    e._or_price_cache_at = 0.0
    yield
    e._or_price_cache = None
    e._or_price_cache_at = 0.0


def test_trials_for_config_basic():
    assert e.trials_for_config(10, 2, 5) == 100
    assert e.trials_for_config(0, 2, 5) == 0
    assert e.trials_for_config(30, 2, 1) == 60


def test_ollama_is_always_free():
    est = e.estimate_cost_per_trial("ollama", "gemma3:12b")
    assert est.source == "free"
    assert est.mean_usd == 0.0


def test_calibration_constants_are_reasonable():
    """Sanity-check the constants match the documented calibration."""
    assert 400 <= e.OASIS_INPUT_TOKENS <= 700
    assert 20 <= e.OASIS_OUTPUT_TOKENS <= 60


def test_openrouter_live_pricing_used_when_available(monkeypatch):
    """Live OpenRouter price × calibrated tokens, exact arithmetic."""
    monkeypatch.setattr(
        e, "_fetch_openrouter_prices",
        lambda: {"google/gemma-3-27b-it": (1e-6, 2e-6)},
    )
    est = e.estimate_cost_per_trial("openrouter", "google/gemma-3-27b-it")
    assert est.source == "openrouter"
    expected = 1e-6 * e.OASIS_INPUT_TOKENS + 2e-6 * e.OASIS_OUTPUT_TOKENS
    assert est.mean_usd == pytest.approx(expected)


def test_openrouter_strips_redundant_prefix(monkeypatch):
    """``openrouter/foo/bar`` falls back to lookup of ``foo/bar``."""
    monkeypatch.setattr(
        e, "_fetch_openrouter_prices",
        lambda: {"foo/bar": (1e-6, 1e-6)},
    )
    est = e.estimate_cost_per_trial("openrouter", "openrouter/foo/bar")
    assert est.source == "openrouter"
    assert est.mean_usd is not None and est.mean_usd > 0


def test_openrouter_falls_back_to_litellm_when_api_unreachable(monkeypatch):
    """API down → use LiteLLM static dict."""
    monkeypatch.setattr(e, "_fetch_openrouter_prices", lambda: None)
    import litellm
    monkeypatch.setitem(
        litellm.model_cost, "openrouter/test-model",
        {"input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6},
    )
    est = e.estimate_cost_per_trial("openrouter", "test-model")
    assert est.source == "litellm"
    assert est.mean_usd == pytest.approx(
        1e-6 * (e.OASIS_INPUT_TOKENS + e.OASIS_OUTPUT_TOKENS),
    )


def test_openrouter_unknown_model_returns_unknown(monkeypatch):
    monkeypatch.setattr(e, "_fetch_openrouter_prices", lambda: {})
    import litellm
    monkeypatch.setattr(litellm, "model_cost", {})
    est = e.estimate_cost_per_trial("openrouter", "made-up-xyz")
    assert est.source == "unknown"
    assert est.mean_usd is None


def test_or_price_cache_ttl(monkeypatch):
    """Repeated calls within TTL don't refetch."""
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return {"foo": (1e-6, 1e-6)}

    monkeypatch.setattr(e, "_fetch_openrouter_prices", fake_fetch)
    e._openrouter_prices()
    e._openrouter_prices()
    e._openrouter_prices()
    assert calls["n"] == 1


def test_or_price_cache_persists_when_fetch_fails(monkeypatch):
    """Once we have a cached price, transient API failures don't drop it."""
    monkeypatch.setattr(
        e, "_fetch_openrouter_prices", lambda: {"foo": (1e-6, 1e-6)},
    )
    e._openrouter_prices()
    monkeypatch.setattr(e, "_fetch_openrouter_prices", lambda: None)
    e._or_price_cache_at = 0.0  # Force expiry
    cached = e._openrouter_prices()
    assert cached == {"foo": (1e-6, 1e-6)}


def test_zero_priced_model_returns_unknown(monkeypatch):
    """Free-tier models priced at $0 shouldn't be reported as 'openrouter'."""
    monkeypatch.setattr(
        e, "_fetch_openrouter_prices",
        lambda: {"google/gemma-3-27b-it:free": (0.0, 0.0)},
    )
    import litellm
    monkeypatch.setattr(litellm, "model_cost", {})
    est = e.estimate_cost_per_trial("openrouter", "google/gemma-3-27b-it:free")
    assert est.source == "unknown"


def test_litellm_used_for_anthropic_provider(monkeypatch):
    """Non-OpenRouter providers skip the OR API and go straight to LiteLLM."""
    fetch_called = {"n": 0}

    def fake_fetch():
        fetch_called["n"] += 1
        return {"some-model": (1e-6, 1e-6)}

    monkeypatch.setattr(e, "_fetch_openrouter_prices", fake_fetch)
    import litellm
    monkeypatch.setitem(
        litellm.model_cost, "claude-test",
        {"input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6},
    )
    est = e.estimate_cost_per_trial("anthropic", "claude-test")
    assert est.source == "litellm"
    assert fetch_called["n"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Run projection (DB-backed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the minimum schema ``project_run`` queries."""
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


def test_project_returns_sane_defaults_for_unknown_run(con):
    proj = e.project_run(con, "doesnt-exist")
    assert proj is not None
    assert proj.done == 0 and proj.total == 0
    assert proj.cost_per_trial_mean_usd is None
    assert proj.eta_seconds is None


def test_project_in_progress_run(con):
    _add_run(con, "r1", "openrouter", "gpt-4o", max_concurrency=4)
    _add_trials(con, "r1", [0.01, 0.02, 0.03], latencies_ms=[2000, 2000, 2000])
    _add_pending(con, "r1", 17)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.done == 3
    assert proj.total == 20
    assert proj.remaining == 17
    assert proj.cost_so_far_usd == pytest.approx(0.06)
    assert proj.cost_per_trial_mean_usd == pytest.approx(0.02)
    assert proj.projected_total_usd == pytest.approx(0.40)  # 0.06 + 17*0.02
    assert proj.eta_seconds == pytest.approx(8.5)            # 17 * 2.0 / 4
    assert proj.latency_recent_mean_s == pytest.approx(2.0)


def test_project_completed_run_has_zero_remaining(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_trials(con, "r1", [0.05, 0.05, 0.05])
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.remaining == 0
    assert proj.eta_seconds is None
    assert proj.projected_total_usd == pytest.approx(0.15)


def test_project_local_run_no_cost(con):
    _add_run(con, "r1", "ollama", "gemma3:12b", max_concurrency=2)
    _add_trials(con, "r1", [None, None, None], latencies_ms=[5000, 5000, 5000])
    _add_pending(con, "r1", 7)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.cost_so_far_usd == 0.0
    assert proj.cost_per_trial_mean_usd is None
    assert proj.projected_total_usd is None
    assert proj.eta_seconds == pytest.approx(17.5)  # 7 * 5.0 / 2


def test_project_concurrency_defaults_to_one(con):
    con.execute(
        "INSERT INTO runs VALUES (?, ?)",
        ["r1", json.dumps({"provider": "openrouter", "model": "gpt-4o"})],
    )
    _add_trials(con, "r1", [0.01], latencies_ms=[1000])
    _add_pending(con, "r1", 9)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.eta_seconds == pytest.approx(9.0)


def test_project_handles_zero_done(con):
    _add_run(con, "r1", "openrouter", "gpt-4o")
    _add_pending(con, "r1", 5)
    proj = e.project_run(con, "r1")
    assert proj is not None
    assert proj.done == 0 and proj.total == 5
    assert proj.cost_per_trial_mean_usd is None
    assert proj.eta_seconds is None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────


def test_format_cost():
    assert e.format_cost(None) == "—"
    assert e.format_cost(0) == "$0.00"
    assert e.format_cost(0.005) == "$0.0050"
    assert e.format_cost(0.42) == "$0.420"
    assert e.format_cost(12.5) == "$12.50"
    assert e.format_cost(1234.56) == "$1,234.56"


def test_format_duration():
    assert e.format_duration(None) == "—"
    assert e.format_duration(-5) == "—"
    assert e.format_duration(45) == "45s"
    assert e.format_duration(125) == "2m 05s"
    assert e.format_duration(3725) == "1h 02m"
    assert e.format_duration(86400 + 7200) == "1d 02h"
