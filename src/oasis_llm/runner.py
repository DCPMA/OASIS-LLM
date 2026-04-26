"""Async runner: claim pending trials, call LLM, persist results."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta

import duckdb
import litellm
from litellm import acompletion
from rich.console import Console

litellm.suppress_debug_info = True
litellm.set_verbose = False

from .config import RunConfig
from .images import image_data_url
from .prompts import RATING_SCHEMA, system_prompt, user_prompt
from .providers import litellm_model_id, setup_langfuse, setup_provider
from .rate_limit import (
    DailyQuotaExceeded,
    OPENROUTER_FREE_LIMITER,
    is_openrouter_free_model,
)

console = Console()

STALE_AFTER = timedelta(minutes=10)


def _prompt_hash(system: str, user: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()[:16]


def _build_prompts(
    cfg: "RunConfig",
    dimension: str,
    image_id: str | None = None,
    sample_idx: int | None = None,
) -> tuple[str, str]:
    sys_p = cfg.system_prompt_override or system_prompt(dimension)
    usr_p = user_prompt(dimension)
    if cfg.capture_reasoning:
        # Replace the paper's "respond with a single integer" instruction with a
        # JSON-with-reasoning instruction. Models that take the system prompt
        # literally (e.g. Gemma 4) otherwise emit just "6" and ignore reasoning.
        sys_p = sys_p.replace(
            "Please respond with a single integer from 1 to 7.",
            'Respond with a JSON object containing your integer `rating` (1-7) '
            'and a brief one-sentence `reasoning` (<=30 words) describing what '
            'about the image drove the rating.\n'
            'Example: {"rating": 5, "reasoning": "A bright sunset evokes mild positive affect."}',
        )
        usr_p = (
            f"{usr_p} Respond ONLY with a JSON object: "
            '{"rating": <int 1-7>, "reasoning": "<one-sentence rationale>"}.'
        )
    if cfg.format_hint_suffix:
        usr_p = f"{usr_p}\n\n{cfg.format_hint_suffix}"
    # Per-sample salt: forces a different decoding path even at temperature=0
    # without changing decoding hyperparameters. Placed at the END of the user
    # turn so the image prefix stays prompt-cacheable on providers that support
    # KV/prompt caching (Anthropic, OpenAI, vLLM, SGLang).
    if cfg.cache_buster and image_id is not None and sample_idx is not None:
        salt = hashlib.sha256(
            f"{cfg.name}|{image_id}|{dimension}|{sample_idx}".encode()
        ).hexdigest()[:10]
        usr_p = f"{usr_p}\n\n[trial-id: {salt}]"
    return sys_p, usr_p


def _build_messages(
    cfg: "RunConfig",
    dimension: str,
    image_id: str,
    sample_idx: int | None = None,
) -> list[dict]:
    sys_p, usr_p = _build_prompts(cfg, dimension, image_id, sample_idx)
    if cfg.modality == "vision":
        user_content = [
            {"type": "image_url", "image_url": {"url": image_data_url(image_id)}},
            {"type": "text", "text": usr_p},
        ]
    else:
        user_content = usr_p
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_content},
    ]


def _schema_for(cfg: "RunConfig") -> dict:
    """Return rating schema, with reasoning field stripped if capture_reasoning is False.

    Note: even when capture_reasoning is True we keep `reasoning` optional in the
    schema. Marking it required under strict-mode causes some models (e.g. Gemma)
    to enter degenerate output loops. The user prompt explicitly asks for it instead.
    """
    if cfg.capture_reasoning:
        return RATING_SCHEMA
    return {
        "type": "object",
        "properties": {"rating": RATING_SCHEMA["properties"]["rating"]},
        "required": ["rating"],
        "additionalProperties": False,
    }


def _parse_rating(content: str) -> tuple[int | None, str | None]:
    """Parse rating + optional reasoning from a model response."""
    content = (content or "").strip()
    # Try JSON first
    try:
        obj = json.loads(content)
        r = int(obj.get("rating"))
        if 1 <= r <= 7:
            return r, obj.get("reasoning")
    except Exception:
        pass
    # Try to extract first JSON object in text
    try:
        i = content.find("{")
        j = content.rfind("}")
        if i != -1 and j != -1:
            obj = json.loads(content[i : j + 1])
            r = int(obj.get("rating"))
            if 1 <= r <= 7:
                return r, obj.get("reasoning")
    except Exception:
        pass
    # Fallback: first integer 1-7 in the response
    import re
    m = re.search(r"\b([1-7])\b", content)
    if m:
        return int(m.group(1)), None
    return None, None


def _claim_one(con: duckdb.DuckDBPyConnection, run_id: str) -> dict | None:
    """Atomically claim one pending trial. Honours `runs.status` for pause/cancel."""
    # Honour external control: if the run was paused or cancelled, stop claiming.
    rstatus = con.execute(
        "SELECT status FROM runs WHERE run_id=?", [run_id]
    ).fetchone()
    if rstatus and rstatus[0] in ("paused", "cancelled"):
        return None
    # Reset stale claims first
    cutoff = datetime.utcnow() - STALE_AFTER
    con.execute(
        "UPDATE trials SET status='pending' WHERE run_id=? AND status='running' AND claimed_at < ?",
        [run_id, cutoff],
    )
    row = con.execute(
        """
        SELECT image_id, dimension, sample_idx, attempts
        FROM trials
        WHERE run_id = ? AND (status = 'pending' OR (status = 'failed' AND attempts < 3))
        ORDER BY attempts, sample_idx, image_id, dimension
        LIMIT 1
        """,
        [run_id],
    ).fetchone()
    if row is None:
        return None
    image_id, dim, sidx, attempts = row
    con.execute(
        "UPDATE trials SET status='running', claimed_at=CURRENT_TIMESTAMP WHERE run_id=? AND image_id=? AND dimension=? AND sample_idx=?",
        [run_id, image_id, dim, sidx],
    )
    return {"image_id": image_id, "dimension": dim, "sample_idx": sidx, "attempts": attempts}


def _record_result(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    trial: dict,
    *,
    rating: int | None,
    raw: str,
    reasoning: str | None,
    prompt_hash: str,
    latency_ms: int,
    in_tok: int | None,
    out_tok: int | None,
    cost: float | None,
    error: str | None,
    finish_reason: str | None = None,
    response_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    status = "done" if rating is not None and error is None else "failed"
    con.execute(
        """
        UPDATE trials
        SET status=?, rating=?, raw_response=?, reasoning=?, prompt_hash=?,
            latency_ms=?, input_tokens=?, output_tokens=?, cost_usd=?,
            error=?, finish_reason=?, response_id=?, trace_id=?,
            attempts=attempts+1, completed_at=CURRENT_TIMESTAMP
        WHERE run_id=? AND image_id=? AND dimension=? AND sample_idx=?
        """,
        [
            status, rating, raw, reasoning, prompt_hash,
            latency_ms, in_tok, out_tok, cost,
            error, finish_reason, response_id, trace_id,
            run_id, trial["image_id"], trial["dimension"], trial["sample_idx"],
        ],
    )


async def _call_model(
    cfg: RunConfig, dim: str, image_id: str, sample_idx: int | None = None,
    trace_id: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[str, int, int | None, int | None, float | None]:
    """Single LLM call. Returns (raw_text, latency_ms, in_tok, out_tok, cost_usd)."""
    # OpenRouter :free tier is rate-limited to 20 rpm + 1000/day. Block here
    # before issuing the request. Raises DailyQuotaExceeded if exhausted; the
    # runner converts that to a permanent error so the config gets cancelled.
    if con is not None and is_openrouter_free_model(cfg.provider, cfg.model):
        await OPENROUTER_FREE_LIMITER.acquire(con)
    provider_kwargs = setup_provider(cfg.provider, cfg.api_base)
    messages = _build_messages(cfg, dim, image_id, sample_idx)
    model_id = litellm_model_id(cfg.provider, cfg.model)
    call_kwargs = dict(
        model=model_id,
        messages=messages,
        timeout=cfg.request_timeout_s,
        metadata={
            "trace_name": "oasis-llm",
            "trace_id": trace_id,
            "session_id": cfg.name,  # Langfuse: groups all trials of a run
            "generation_name": f"{cfg.name}/{dim}",
            "tags": [
                "oasis-llm", cfg.provider, cfg.model, dim,
                # If run_id is `{exp}__{cfg}`, surface the experiment id as a tag.
                *([f"exp:{cfg.name.split('__', 1)[0]}"] if "__" in cfg.name else []),
            ],
            "trace_user_id": cfg.name,
        },
        **provider_kwargs,
        **cfg.extra_params,
    )
    if cfg.max_tokens is not None:
        call_kwargs["max_tokens"] = cfg.max_tokens
    # OpenRouter native cost reporting
    if cfg.provider == "openrouter":
        call_kwargs.setdefault("extra_body", {})
        call_kwargs["extra_body"].setdefault("usage", {"include": True})
    # Ollama: thinking-capable models (qwen3/3.5, gemma4, deepseek-r1, ...) emit
    # hidden reasoning tokens that consume max_tokens but don't appear in
    # `content`, producing empty responses. Default `think: False` unless the
    # user explicitly opted in via extra_params.
    if cfg.provider == "ollama" and "think" not in cfg.extra_params:
        call_kwargs["think"] = False
    if cfg.temperature is not None:
        call_kwargs["temperature"] = cfg.temperature
    # Strategy:
    #  - capture_reasoning=False: enforce strict JSON-schema; smaller models
    #    handle the simple {rating: int} shape reliably.
    #  - capture_reasoning=True: skip response_format entirely. Smaller models
    #    (e.g. Gemma 4) enter degenerate output loops when required-reasoning is
    #    forced under strict schemas. Rely on prompt + _parse_rating fallback.
    if cfg.capture_reasoning:
        resp = await acompletion(**call_kwargs)
    else:
        try:
            resp = await acompletion(
                **call_kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "rating", "schema": _schema_for(cfg), "strict": True},
                },
            )
        except Exception:
            resp = await acompletion(**call_kwargs)
    t1 = time.monotonic()
    raw = resp.choices[0].message.content or ""
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    response_id = getattr(resp, "id", None)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", None) if usage else None
    out_tok = getattr(usage, "completion_tokens", None) if usage else None
    cost = None
    try:
        cost = float(litellm.completion_cost(completion_response=resp))
    except Exception:
        cost = None
    # Fallback: OpenRouter native cost in usage.cost (when usage.include=true)
    if cost is None and usage is not None:
        native_cost = getattr(usage, "cost", None)
        if native_cost is None and hasattr(usage, "model_extra"):
            native_cost = (usage.model_extra or {}).get("cost")
        if native_cost is None and hasattr(usage, "__dict__"):
            native_cost = usage.__dict__.get("cost")
        if native_cost is not None:
            try:
                cost = float(native_cost)
            except (TypeError, ValueError):
                pass
    return raw, int((time.monotonic() - t1) * 1000), in_tok, out_tok, cost, finish_reason, response_id


async def _evict_ollama_model(cfg: RunConfig) -> bool:
    """POST keep_alive=0 to Ollama to tear down a stuck runner subprocess.

    Returns True if eviction was attempted (regardless of outcome). Use when
    consecutive Ollama timeouts/500s indicate a runner deadlock — the model
    stays resident in VRAM but generation produces nothing. Eviction kills
    the runner so the next request spawns a fresh one.
    """
    if cfg.provider != "ollama":
        return False
    import httpx
    base = (cfg.api_base or "http://localhost:11434").rstrip("/")
    # Strip the litellm provider prefix if present.
    model = cfg.model.split("/", 1)[-1] if "/" in cfg.model else cfg.model
    url = f"{base}/api/generate"
    payload = {"model": model, "keep_alive": 0, "prompt": "", "stream": False}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=payload)
        console.print(f"[yellow]\u2622 Evicted Ollama model {model} (consecutive stalls)[/]")
        return True
    except Exception as e:
        console.print(f"[yellow]eviction request failed: {type(e).__name__}: {e}[/]")
        return True


def _is_stall_error(exc: BaseException) -> bool:
    """Return True for the Ollama runner-deadlock signature."""
    s = f"{type(exc).__name__}: {exc}"
    return (
        "Timeout" in s
        or "APIConnectionError" in s
        or "Connection timed out" in s
        or "ollama" in s.lower() and ("500" in s or "internal server error" in s.lower())
    )


# Substrings of error strings that indicate a PERMANENT (non-transient) failure.
# These are terminal regardless of retries: model misconfigured, bad request,
# auth issue, content policy, schema mismatch, etc.
_PERMANENT_ERROR_MARKERS = (
    "BadRequestError",
    "NotFoundError",
    "AuthenticationError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
    "ContentPolicyViolationError",
    "DailyQuotaExceeded",            # OpenRouter :free daily cap reached
    "model_not_found",
    "invalid_request",
    "schema_validation_error",
    " 400 ", " 401 ", " 403 ", " 404 ", " 422 ",
    "status_code: 400", "status_code: 401", "status_code: 403",
    "status_code: 404", "status_code: 422",
)


def _is_permanent_error(error_str: str | None) -> bool:
    """True if the error is a permanent (non-transient) failure.

    Transient errors (timeouts, 5xx, 429, connection resets) return False —
    the retry loop handles those. Permanent errors include 4xx (except 429),
    schema/validation failures, auth/permission errors, and the
    OpenRouter free-tier daily-quota signal.
    """
    if not error_str:
        return False
    s = error_str
    # 429 (rate limit) is transient — exclude even if it shares a 4xx prefix.
    if " 429 " in s or "status_code: 429" in s or "RateLimitError" in s:
        return False
    return any(m in s for m in _PERMANENT_ERROR_MARKERS)


async def _worker(
    cfg: RunConfig,
    con: duckdb.DuckDBPyConnection,
    sem: asyncio.Semaphore,
    lock: asyncio.Lock,
    stall_state: dict,
):
    while True:
        # Fast exit if a sibling worker auto-cancelled the run (e.g. permanent
        # failure rate exceeded threshold).
        if stall_state.get("cancelled"):
            return
        async with lock:
            trial = _claim_one(con, cfg.name)
        if trial is None:
            return
        async with sem:
            t0 = time.monotonic()
            error = None
            finish_reason = response_id = None
            rating = None
            reasoning = None
            raw = ""
            in_tok = out_tok = cost = None
            # Stable per-trial id we control, used as the Langfuse trace_id so
            # the dashboard can deep-link back to the trace.
            trace_id = uuid.uuid4().hex
            sys_p, usr_p = _build_prompts(
                cfg, trial["dimension"], trial["image_id"], trial["sample_idx"]
            )
            ph = _prompt_hash(sys_p, usr_p, cfg.model)
            attempts = max(1, int(cfg.max_retries) + 1)
            base = max(0.0, float(cfg.retry_backoff_base_s))
            coef = max(1.0, float(cfg.retry_backoff_coef))
            for attempt in range(attempts):
                error = None
                try:
                    raw, _, in_tok, out_tok, cost, finish_reason, response_id = await _call_model(
                        cfg, trial["dimension"], trial["image_id"], trial["sample_idx"],
                        trace_id=trace_id, con=con,
                    )
                    rating, reasoning = _parse_rating(raw)
                    if not raw.strip():
                        error = "empty response from model"
                    elif rating is None:
                        error = "could not parse rating"
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                # Track Ollama stall streak across workers; trigger evict when
                # threshold hit. Successful trial resets the counter.
                if cfg.provider == "ollama":
                    if error and _is_stall_error(Exception(error)):
                        stall_state["streak"] = stall_state.get("streak", 0) + 1
                        threshold = int(cfg.ollama_evict_threshold or 0)
                        if threshold > 0 and stall_state["streak"] >= threshold:
                            async with stall_state["evict_lock"]:
                                # Re-check under lock so only one worker evicts.
                                if stall_state["streak"] >= threshold:
                                    await _evict_ollama_model(cfg)
                                    stall_state["streak"] = 0
                    elif rating is not None and error is None:
                        stall_state["streak"] = 0
                # Success or terminal error → stop retrying.
                if error is None and rating is not None:
                    break
                # Permanent (non-transient) errors won't recover with a retry —
                # bail immediately so we can update the failure ratio fast.
                if _is_permanent_error(error):
                    break
                if attempt + 1 < attempts:
                    delay = base * (coef ** attempt)
                    console.print(
                        f"[dim yellow]retry {attempt + 1}/{attempts - 1} in {delay:.1f}s "
                        f"({trial['image_id']} {trial['dimension']}): {error}[/]"
                    )
                    await asyncio.sleep(delay)
            latency_ms = int((time.monotonic() - t0) * 1000)
            async with lock:
                _record_result(
                    con, cfg.name, trial,
                    rating=rating, raw=raw, reasoning=reasoning,
                    prompt_hash=ph, latency_ms=latency_ms,
                    in_tok=in_tok, out_tok=out_tok, cost=cost, error=error,
                    finish_reason=finish_reason, response_id=response_id,
                    trace_id=trace_id,
                )
            status = "[green]ok[/]" if rating is not None else "[red]fail[/]"
            console.print(f"[dim]{cfg.name}[/] {trial['image_id']} {trial['dimension']}#{trial['sample_idx']} -> {status} rating={rating} ({latency_ms}ms)")

            # ── Permanent-failure ratio: cancel the config if it's broken ──
            # Track per-run attempt count + permanent failure count. After a
            # warmup floor (avoids early-flake cancellation on transient bursts),
            # if the permanent-failure ratio exceeds the configured threshold
            # the run is marked cancelled and all workers exit.
            stall_state["attempts"] = stall_state.get("attempts", 0) + 1
            if _is_permanent_error(error):
                stall_state["perm_failures"] = stall_state.get("perm_failures", 0) + 1
            threshold = float(cfg.max_permanent_failure_rate or 0.0)
            warmup = max(0, int(cfg.permanent_failure_warmup or 0))
            if (
                threshold > 0.0
                and stall_state["attempts"] >= max(1, warmup)
                and stall_state.get("perm_failures", 0) / stall_state["attempts"] > threshold
            ):
                async with lock:
                    if not stall_state.get("cancelled"):
                        stall_state["cancelled"] = True
                        ratio = stall_state["perm_failures"] / stall_state["attempts"]
                        reason = (
                            f"auto-cancelled: permanent-failure ratio "
                            f"{stall_state['perm_failures']}/{stall_state['attempts']} "
                            f"({ratio:.1%}) > {threshold:.1%}"
                        )
                        console.print(f"[red bold]\u26d4 {cfg.name} {reason}[/]")
                        con.execute(
                            "UPDATE runs SET status='cancelled' WHERE run_id=?",
                            [cfg.name],
                        )
                return


async def run(cfg: RunConfig, con: duckdb.DuckDBPyConnection) -> None:
    """Run the full worker pool until no pending trials remain."""
    if setup_langfuse():
        console.print("[dim]Langfuse tracing enabled[/]")
    con.execute("UPDATE runs SET status='running' WHERE run_id=?", [cfg.name])
    # Optional global cap via env (handy for shared API quotas).
    import os as _os
    cap = _os.getenv("OASIS_MAX_CONCURRENCY")
    effective = cfg.max_concurrency
    if cap:
        try:
            effective = max(1, min(int(cap), cfg.max_concurrency))
        except ValueError:
            pass
    if effective != cfg.max_concurrency:
        console.print(
            f"[yellow]concurrency capped {cfg.max_concurrency} → {effective} "
            f"by OASIS_MAX_CONCURRENCY[/]"
        )
    sem = asyncio.Semaphore(effective)
    lock = asyncio.Lock()
    stall_state = {"streak": 0, "evict_lock": asyncio.Lock()}
    workers = [
        asyncio.create_task(_worker(cfg, con, sem, lock, stall_state))
        for _ in range(effective)
    ]
    await asyncio.gather(*workers)
    # If we exited because the user paused/cancelled, leave that status alone.
    final = con.execute("SELECT status FROM runs WHERE run_id=?", [cfg.name]).fetchone()
    if final and final[0] in ("paused", "cancelled"):
        return
    # Otherwise compute final status from remaining trials.
    pending = con.execute(
        "SELECT count(*) FROM trials WHERE run_id=? AND status IN ('pending','running')",
        [cfg.name],
    ).fetchone()[0]
    if pending == 0:
        con.execute(
            "UPDATE runs SET status='done', finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
            [cfg.name],
        )
    else:
        con.execute("UPDATE runs SET status='pending' WHERE run_id=?", [cfg.name])
