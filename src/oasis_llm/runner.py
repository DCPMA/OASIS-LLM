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
) -> tuple[str, int, int | None, int | None, float | None]:
    """Single LLM call. Returns (raw_text, latency_ms, in_tok, out_tok, cost_usd)."""
    provider_kwargs = setup_provider(cfg.provider, cfg.api_base)
    messages = _build_messages(cfg, dim, image_id, sample_idx)
    model_id = litellm_model_id(cfg.provider, cfg.model)
    call_kwargs = dict(
        model=model_id,
        messages=messages,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout_s,
        metadata={
            "trace_name": "oasis-llm",
            "trace_id": trace_id,
            "generation_name": f"{cfg.name}/{dim}",
            "tags": ["oasis-llm", cfg.provider, cfg.model, dim],
            "trace_user_id": cfg.name,
        },
        **provider_kwargs,
        **cfg.extra_params,
    )
    # OpenRouter native cost reporting
    if cfg.provider == "openrouter":
        call_kwargs.setdefault("extra_body", {})
        call_kwargs["extra_body"].setdefault("usage", {"include": True})
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


async def _worker(cfg: RunConfig, con: duckdb.DuckDBPyConnection, sem: asyncio.Semaphore, lock: asyncio.Lock):
    while True:
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
            try:
                raw, _, in_tok, out_tok, cost, finish_reason, response_id = await _call_model(
                    cfg, trial["dimension"], trial["image_id"], trial["sample_idx"],
                    trace_id=trace_id,
                )
                rating, reasoning = _parse_rating(raw)
                if not raw.strip():
                    error = "empty response from model"
                elif rating is None:
                    error = "could not parse rating"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
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


async def run(cfg: RunConfig, con: duckdb.DuckDBPyConnection) -> None:
    """Run the full worker pool until no pending trials remain."""
    if setup_langfuse():
        console.print("[dim]Langfuse tracing enabled[/]")
    con.execute("UPDATE runs SET status='running' WHERE run_id=?", [cfg.name])
    sem = asyncio.Semaphore(cfg.max_concurrency)
    lock = asyncio.Lock()
    workers = [asyncio.create_task(_worker(cfg, con, sem, lock)) for _ in range(cfg.max_concurrency)]
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
