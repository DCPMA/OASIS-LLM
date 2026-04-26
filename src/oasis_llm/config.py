"""Run config schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str
    provider: Literal["openrouter", "ollama", "anthropic", "google", "openai"]
    model: str  # litellm-style model id, e.g. "openrouter/google/gemma-4-31b-it"
    modality: Literal["vision", "text"] = "vision"
    dimensions: list[Literal["valence", "arousal"]] = Field(default_factory=lambda: ["valence", "arousal"])
    image_set: str = "pilot_30"  # named subset or "full_900"
    samples_per_image: int = 5
    max_concurrency: int = 4
    request_timeout_s: int = 60
    max_retries: int = 3
    retry_backoff_base_s: float = 1.0
    retry_backoff_coef: float = 2.0
    ollama_evict_threshold: int = 3  # consecutive Ollama timeouts/500s before evict-and-reload (0=disable)
    # Auto-cancel a config when too many trials fail with PERMANENT (non-transient)
    # errors — e.g. 404 model not found, 400 bad request, 422 schema validation.
    # Avoids burning quota on a misconfigured config while the rest of the
    # experiment proceeds. Warmup floor prevents single early-flake cancellations.
    max_permanent_failure_rate: float = 0.05  # ratio threshold (0.0 = disable)
    permanent_failure_warmup: int = 20         # min attempts before ratio is checked
    temperature: float | None = None
    max_tokens: int | None = None  # None = don't send max_tokens (uncapped)
    capture_reasoning: bool = True
    cache_buster: bool = True  # Append per-sample salt to user prompt to force decoding variance even at temperature=0. Placed AFTER image so prefix-cacheability is preserved.
    system_prompt_override: str | None = None  # full replacement of paper-verbatim system prompt
    format_hint_suffix: str | None = None  # appended to user prompt; useful to enforce JSON-only output on weaker models
    api_base: str | None = None  # for ollama, e.g. "http://localhost:11434"
    extra_params: dict = Field(default_factory=dict)

    def canonical_hash(self) -> str:
        # Exclude fields that should NOT invalidate a run:
        #   - name, runtime knobs (concurrency/timeout/retries)
        #   - samples_per_image (we want to support progressive expansion)
        payload = self.model_dump(exclude={
            "name", "max_concurrency", "request_timeout_s", "max_retries",
            "retry_backoff_base_s", "retry_backoff_coef", "ollama_evict_threshold",
            "max_permanent_failure_rate", "permanent_failure_warmup",
            "samples_per_image",
        })
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
