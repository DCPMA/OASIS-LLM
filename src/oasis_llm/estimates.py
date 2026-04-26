"""Cost & time estimates for experiments and runs.

Pre-launch cost is computed from **calibrated token constants** (see
``OASIS_INPUT_TOKENS`` / ``OASIS_OUTPUT_TOKENS`` below) multiplied by
**live OpenRouter pricing** (cached 1h). LiteLLM's static ``model_cost``
dict is the second-tier fallback for non-OpenRouter providers.

This is intentionally simple: actual token usage on the OASIS rating
task is extremely tight (n=10,598 trials gave input μ=544±51,
output μ=31±8), so per-model variance doesn't justify the
complexity of empirical-history tracking. Day-1 estimates on a fresh
machine are within ~10% of reality.

Live (in-progress) projections on the Runs page still use empirical
$/trial from each run's own completed trials — that data is free and
strictly more accurate once trials start landing.

All public functions are pure-Python; pricing functions are network-
cached, projection functions are DB-read-only. They never raise: on
failure they return ``None``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import duckdb


# ── Calibrated token constants ──────────────────────────────────────────────
#
# Derived from n=10,598 completed trials across 9 vision models on the
# OASIS valence/arousal rating task (single image + JSON output):
#
#   input  tokens: μ=544, median=529, σ=51,  range 379–652
#   output tokens: μ= 31, median= 32, σ= 8,  range   2–256
#
# Per-model means cluster tightly (449–606 input, 30–40 output), so a
# single global pair is accurate enough for budgeting. Slight upward
# padding on input avoids systematic under-quoting; output stays at
# the rounded mean since the 256 max is a parsing-failure outlier.
#
# If you change the prompt template or add multi-image support,
# re-calibrate by running the SQL in ``scripts/`` (or just inspect
# ``avg(input_tokens)`` / ``avg(output_tokens)`` from ``trials``).
OASIS_INPUT_TOKENS = 560
OASIS_OUTPUT_TOKENS = 35

# Rolling window size for in-progress projections. Recent trials reflect
# current model warmth / network conditions better than the run's full mean.
_ROLLING_WINDOW = 20

# Live OpenRouter pricing cache.
_OR_CACHE_TTL_S = 3600.0
_or_price_cache: dict[str, tuple[float, float]] | None = None
_or_price_cache_at: float = 0.0


@dataclass(frozen=True)
class CostEstimate:
    """Per-trial cost estimate.

    ``mean_usd`` is the only number consumers should display. ``source``
    is a short label for diagnostics (live OR pricing, LiteLLM, free, or
    unavailable). ``mean_usd`` is None iff source=="unknown".
    """

    mean_usd: float | None
    source: Literal["openrouter", "litellm", "free", "unknown"]


@dataclass(frozen=True)
class RunProjection:
    """Live projection for an in-progress (or completed) run."""

    done: int
    total: int
    remaining: int
    cost_so_far_usd: float
    cost_per_trial_mean_usd: float | None
    cost_per_trial_std_usd: float | None
    projected_total_usd: float | None
    latency_recent_mean_s: float | None
    eta_seconds: float | None


# ── Cost helpers ────────────────────────────────────────────────────────────


def _fetch_openrouter_prices() -> dict[str, tuple[float, float]] | None:
    """Fetch ``{model_id: (prompt_per_token, completion_per_token)}``.

    Hits ``https://openrouter.ai/api/v1/models``. Sends the API key when
    ``OPENROUTER_API_KEY`` is set so the user sees the same model set
    they have access to. Returns None on any error — the caller falls
    back to LiteLLM.
    """
    import json
    import urllib.request
    headers = {"User-Agent": "oasis-llm/1.0"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models", headers=headers,
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            payload = json.loads(r.read().decode())
    except Exception:
        return None
    out: dict[str, tuple[float, float]] = {}
    for m in payload.get("data") or []:
        mid = m.get("id")
        pricing = m.get("pricing") or {}
        try:
            p_in = float(pricing.get("prompt") or 0.0)
            p_out = float(pricing.get("completion") or 0.0)
        except (TypeError, ValueError):
            continue
        if mid:
            out[mid] = (p_in, p_out)
    return out or None


def _openrouter_prices() -> dict[str, tuple[float, float]] | None:
    """Cached wrapper around :func:`_fetch_openrouter_prices` (TTL 1h)."""
    global _or_price_cache, _or_price_cache_at
    now = time.monotonic()
    if _or_price_cache is not None and (now - _or_price_cache_at) < _OR_CACHE_TTL_S:
        return _or_price_cache
    fresh = _fetch_openrouter_prices()
    if fresh is not None:
        _or_price_cache = fresh
        _or_price_cache_at = now
    return _or_price_cache


def _openrouter_cost_per_trial(model: str) -> float | None:
    """Live OpenRouter price × calibrated tokens. None if unavailable."""
    prices = _openrouter_prices()
    if not prices:
        return None
    # OpenRouter ids are bare ("google/gemma-3-27b-it"), but our config
    # may store them with the "openrouter/" prefix — accept both.
    candidates = [model, model.removeprefix("openrouter/")]
    for key in candidates:
        if key in prices:
            p_in, p_out = prices[key]
            cost = p_in * OASIS_INPUT_TOKENS + p_out * OASIS_OUTPUT_TOKENS
            return cost if cost > 0 else None
    return None


def _litellm_cost_per_trial(provider: str, model: str) -> float | None:
    """Per-trial cost from LiteLLM's static model_cost dict + calibrated tokens.

    Used as a fallback for non-OpenRouter providers and when the live
    OpenRouter API is unreachable.
    """
    if provider == "ollama":
        return None
    try:
        import litellm  # type: ignore
    except Exception:
        return None
    candidates = []
    if provider == "openrouter":
        candidates.append(f"openrouter/{model}")
    candidates.append(model)
    info = None
    cost_dict = getattr(litellm, "model_cost", None) or {}
    for key in candidates:
        if key in cost_dict:
            info = cost_dict[key]
            break
    if info is None:
        return None
    in_rate = info.get("input_cost_per_token")
    out_rate = info.get("output_cost_per_token")
    if not in_rate and not out_rate:
        return None
    cost = (
        float(in_rate or 0.0) * OASIS_INPUT_TOKENS
        + float(out_rate or 0.0) * OASIS_OUTPUT_TOKENS
    )
    return cost if cost > 0 else None


def estimate_cost_per_trial(provider: str, model: str) -> CostEstimate:
    """Best available cost-per-trial estimate.

    Order: ``free`` (Ollama) → live OpenRouter pricing → LiteLLM static
    pricing → ``unknown``. All cost paths use the calibrated
    ``OASIS_INPUT_TOKENS`` / ``OASIS_OUTPUT_TOKENS`` constants.
    """
    if provider == "ollama":
        return CostEstimate(0.0, "free")
    if provider == "openrouter":
        live = _openrouter_cost_per_trial(model)
        if live is not None:
            return CostEstimate(live, "openrouter")
    fb = _litellm_cost_per_trial(provider, model)
    if fb is not None:
        return CostEstimate(fb, "litellm")
    return CostEstimate(None, "unknown")


def trials_for_config(num_images: int, num_dimensions: int, samples_per_image: int) -> int:
    """Total trials a config will enqueue. Formula matches ``enqueue.py``."""
    return int(num_images) * int(num_dimensions) * int(samples_per_image)


# ── Run projection (in-progress) ────────────────────────────────────────────


def project_run(
    con: duckdb.DuckDBPyConnection, run_id: str,
) -> RunProjection | None:
    """Live projection for ``run_id``. Returns None if the run is unknown."""
    counts = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE status='done')              AS done,
            count(*)                                            AS total,
            coalesce(sum(cost_usd) FILTER (WHERE status='done'), 0.0) AS spent
        FROM trials WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if counts is None:
        return None
    done, total, spent = int(counts[0] or 0), int(counts[1] or 0), float(counts[2] or 0.0)
    remaining = max(0, total - done)

    # Need at least 1 done trial to extrapolate; below that we still
    # return a projection with None fields so callers can render
    # "—" placeholders consistently.
    cost_mean: float | None = None
    cost_std: float | None = None
    projected_total: float | None = None
    if done > 0:
        # Only consider trials with non-NULL cost. Ollama has cost=NULL
        # (free), so cost_mean stays at None and projected_total stays None
        # for local runs.
        cstats = con.execute(
            """
            SELECT avg(cost_usd), stddev_samp(cost_usd), count(cost_usd)
            FROM trials
            WHERE run_id=? AND status='done' AND cost_usd IS NOT NULL
            """,
            [run_id],
        ).fetchone()
        if cstats is not None and cstats[2] and int(cstats[2]) > 0:
            cost_mean = float(cstats[0])
            cost_std = float(cstats[1]) if cstats[1] is not None else 0.0
            projected_total = spent + cost_mean * remaining

    # Latency from recent completed trials (rolling window).
    lat_recent_mean: float | None = None
    eta_seconds: float | None = None
    if done > 0 and remaining > 0:
        lat_row = con.execute(
            """
            SELECT avg(latency_ms) FROM (
                SELECT latency_ms FROM trials
                WHERE run_id=? AND status='done' AND latency_ms IS NOT NULL
                ORDER BY completed_at DESC NULLS LAST
                LIMIT ?
            )
            """,
            [run_id, _ROLLING_WINDOW],
        ).fetchone()
        if lat_row is not None and lat_row[0] is not None:
            lat_recent_mean = float(lat_row[0]) / 1000.0
            # Concurrency: we don't store this per-run live, so read it
            # off the run's config.
            cfg_row = con.execute(
                "SELECT config_json FROM runs WHERE run_id=?", [run_id],
            ).fetchone()
            concurrency = 1
            if cfg_row and cfg_row[0]:
                try:
                    import json
                    concurrency = max(
                        1, int(json.loads(cfg_row[0]).get("max_concurrency", 1) or 1),
                    )
                except Exception:
                    concurrency = 1
            eta_seconds = (remaining * lat_recent_mean) / concurrency

    return RunProjection(
        done=done,
        total=total,
        remaining=remaining,
        cost_so_far_usd=spent,
        cost_per_trial_mean_usd=cost_mean,
        cost_per_trial_std_usd=cost_std,
        projected_total_usd=projected_total,
        latency_recent_mean_s=lat_recent_mean,
        eta_seconds=eta_seconds,
    )


# ── Formatting helpers (UI-friendly) ────────────────────────────────────────


def format_cost(usd: float | None) -> str:
    """Compact USD formatting. Sub-cent values get extra decimals."""
    if usd is None:
        return "—"
    if usd == 0:
        return "$0.00"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    return f"${usd:,.2f}"


def format_duration(seconds: float | None) -> str:
    """Compact human duration. ``None → '—'``."""
    if seconds is None or seconds < 0:
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h < 24:
        return f"{h}h {m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d {h:02d}h"
