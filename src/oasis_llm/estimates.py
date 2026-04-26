"""Cost & time estimates for experiments and runs.

Two tiers of estimate:

- **Empirical**: extrapolate from completed trials of the same (provider,
  model). Requires history; uncertainty comes from per-trial std-dev.
- **Fallback**: use LiteLLM's pricing DB (``litellm.model_cost``) with a
  rough token assumption (≈1500 input + 150 output per trial). Used when
  no history exists for the model.

Two consumers:

- **Pre-launch** (Experiments page): per-config $ estimate + total before
  the user clicks "Create experiment". Time estimate intentionally
  omitted at this stage — too easy to mislead with cold-start latency.
- **In-progress** (Runs page): live ETA + projected total cost computed
  from a rolling window of recent completed trials.

All public functions are pure-Python and DB-read-only — no writes, no
mutation. They never raise: on failure they return ``None`` and let the
caller decide whether to render a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import duckdb


# Token-count assumptions used only when we have zero history for a model.
# Vision-LLM image trials are dominated by image tokens which vary widely;
# 1500 in / 150 out is a conservative middle-of-the-road guess for OASIS-style
# rating prompts.
_FALLBACK_INPUT_TOKENS = 1500
_FALLBACK_OUTPUT_TOKENS = 150

# Minimum completed trials for a stat to count as "history". Below this,
# variance is too high to trust the empirical mean.
_MIN_HISTORY_TRIALS = 3

# Rolling window size for in-progress projections. Recent trials reflect
# current model warmth / network conditions better than the run's full mean.
_ROLLING_WINDOW = 20


@dataclass(frozen=True)
class CostEstimate:
    """Per-trial cost estimate.

    ``mean_usd`` is None when the model is local (Ollama) or pricing
    couldn't be resolved. ``std_usd`` is None when source != "history".
    """

    mean_usd: float | None
    std_usd: float | None
    n_trials: int
    source: Literal["history", "fallback", "free", "unknown"]


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


def _empirical_cost_per_trial(
    con: duckdb.DuckDBPyConnection, provider: str, model: str
) -> tuple[float | None, float | None, int]:
    """Mean & std cost-per-trial across all done trials for (provider, model).

    Joins trials → runs → config_json so we filter on the *recorded*
    provider/model (not the current name in PROVIDERS list, which can drift).
    Returns ``(mean, std, n)``. ``mean`` is None when n < _MIN_HISTORY_TRIALS
    or every cost_usd is NULL.
    """
    row = con.execute(
        """
        SELECT
            avg(t.cost_usd) AS mean_cost,
            stddev_samp(t.cost_usd) AS std_cost,
            count(t.cost_usd) AS n
        FROM trials t
        JOIN runs r USING (run_id)
        WHERE t.status = 'done'
          AND t.cost_usd IS NOT NULL
          AND json_extract_string(r.config_json, '$.provider') = ?
          AND json_extract_string(r.config_json, '$.model')    = ?
        """,
        [provider, model],
    ).fetchone()
    if row is None:
        return None, None, 0
    mean_cost, std_cost, n = row
    n = int(n or 0)
    if n < _MIN_HISTORY_TRIALS or mean_cost is None:
        return None, None, n
    return float(mean_cost), float(std_cost) if std_cost is not None else 0.0, n


def _litellm_cost_per_trial(provider: str, model: str) -> float | None:
    """Per-trial cost from LiteLLM's model_cost dict + token assumption.

    Returns None if the model isn't in LiteLLM's catalogue or pricing
    is missing / zero. Local providers (Ollama) always return None.
    """
    if provider == "ollama":
        return None
    try:
        import litellm  # type: ignore
    except Exception:
        return None
    # LiteLLM keys differ by provider: openrouter routes through
    # ``openrouter/<id>``, anthropic and openai use bare ids.
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
    in_rate = float(in_rate or 0.0)
    out_rate = float(out_rate or 0.0)
    cost = (in_rate * _FALLBACK_INPUT_TOKENS) + (out_rate * _FALLBACK_OUTPUT_TOKENS)
    if cost <= 0:
        return None
    return cost


def estimate_cost_per_trial(
    con: duckdb.DuckDBPyConnection, provider: str, model: str,
) -> CostEstimate:
    """Best available cost-per-trial estimate.

    Order: empirical history → LiteLLM fallback → ``free`` (Ollama) →
    ``unknown``.
    """
    if provider == "ollama":
        # Local models are free; still return a CostEstimate so the UI
        # can label them clearly rather than rendering nothing.
        return CostEstimate(0.0, 0.0, 0, "free")

    mean, std, n = _empirical_cost_per_trial(con, provider, model)
    if mean is not None:
        return CostEstimate(mean, std, n, "history")

    fb = _litellm_cost_per_trial(provider, model)
    if fb is not None:
        return CostEstimate(fb, None, 0, "fallback")

    return CostEstimate(None, None, n, "unknown")


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


def format_cost_with_uncertainty(mean: float | None, std: float | None) -> str:
    """``$0.42 ± $0.15`` style. Drops the ± half when std is None or 0."""
    if mean is None:
        return "—"
    base = format_cost(mean)
    if std is None or std == 0 or std < mean * 0.001:
        return base
    return f"{base} ± {format_cost(std)}"


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
