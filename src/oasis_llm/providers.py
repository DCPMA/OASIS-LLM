"""Provider routing for litellm."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def setup_langfuse() -> bool:
    """Wire litellm -> Langfuse if env keys are present. Returns True if enabled."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return False
    # litellm reads LANGFUSE_HOST; user provided LANGFUSE_BASE_URL
    if not os.getenv("LANGFUSE_HOST"):
        host = os.getenv("LANGFUSE_BASE_URL")
        if host:
            os.environ["LANGFUSE_HOST"] = host
    import litellm
    # Validate the langfuse SDK is importable + compatible. On Python 3.14 + pydantic v2,
    # langfuse 2.x can raise pydantic.v1.errors.ConfigError. We swallow that and disable.
    try:
        import langfuse  # noqa: F401
        if not hasattr(langfuse, "version"):
            return False
    except Exception:
        return False
    if "langfuse" not in (litellm.success_callback or []):
        litellm.success_callback = (litellm.success_callback or []) + ["langfuse"]
    if "langfuse" not in (litellm.failure_callback or []):
        litellm.failure_callback = (litellm.failure_callback or []) + ["langfuse"]
    return True


def setup_provider(provider: str, api_base: str | None = None) -> dict:
    """Return litellm-compatible kwargs for a provider."""
    kwargs: dict = {}
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in env")
        kwargs["api_key"] = key
        kwargs["api_base"] = api_base or "https://openrouter.ai/api/v1"
    elif provider == "ollama":
        kwargs["api_base"] = api_base or "http://localhost:11434"
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
    elif provider == "google":
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
    return kwargs


def litellm_model_id(provider: str, model: str) -> str:
    """Return the model id with the litellm provider prefix."""
    if provider == "openrouter":
        return f"openrouter/{model}" if not model.startswith("openrouter/") else model
    if provider == "ollama":
        # litellm uses "ollama_chat/<model>" for chat models with multimodal
        if model.startswith("ollama_chat/") or model.startswith("ollama/"):
            return model
        return f"ollama_chat/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}" if not model.startswith("anthropic/") else model
    if provider == "google":
        return f"gemini/{model}" if not model.startswith("gemini/") else model
    if provider == "openai":
        return model
    return model


# ─── model-default discovery ─────────────────────────────────────────────────
# Cached per (provider, model, api_base). Returns ``{}`` when the provider
# doesn't expose recommended defaults — callers should treat that as "let
# the provider apply whatever it does internally".
_MODEL_DEFAULTS_CACHE: dict[tuple, dict] = {}


def fetch_model_defaults(
    provider: str, model: str, *, api_base: str | None = None,
    timeout_s: float = 4.0,
) -> dict:
    """Best-effort sampling-parameter defaults for ``(provider, model)``.

    Returns a dict possibly containing keys: ``temperature``, ``top_p``,
    ``top_k``, ``num_ctx``, ``max_tokens``. Any key may be missing.

    * **Ollama**: queries ``/api/show`` and parses the ``parameters`` block.
      This is the only provider that exposes per-model recommended defaults
      in a structured form.
    * **OpenRouter / Anthropic / Google / OpenAI**: return ``{}`` — the
      provider applies its own internal defaults when params are omitted.

    The result is cached for the lifetime of the process. Network errors
    return ``{}`` rather than raising; the caller can fall back to "send
    nothing → provider default".
    """
    key = (provider, model, api_base or "")
    cached = _MODEL_DEFAULTS_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict = {}
    try:
        if provider == "ollama":
            out = _fetch_ollama_defaults(model, api_base, timeout_s)
    except Exception:
        out = {}
    _MODEL_DEFAULTS_CACHE[key] = out
    return out


def _fetch_ollama_defaults(model: str, api_base: str | None, timeout_s: float) -> dict:
    """Parse Ollama's ``/api/show`` ``parameters`` text block.

    Ollama returns the ``Modelfile`` parameters as a newline-delimited string
    like::

        num_ctx 4096
        temperature 0.7
        top_k 40
        top_p 0.9
        stop "<|eot_id|>"

    We only surface the four sampling parameters we expose in the UI.
    """
    import json as _json
    import urllib.request

    base = (api_base or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434").rstrip("/")
    bare = model.split("/", 1)[-1] if model.startswith(("ollama_chat/", "ollama/")) else model
    req = urllib.request.Request(
        f"{base}/api/show",
        data=_json.dumps({"name": bare}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = _json.loads(r.read().decode())
    raw = body.get("parameters") or ""
    out: dict = {}
    for line in str(raw).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        k, v = parts[0], parts[1].strip().strip('"')
        if k == "temperature":
            try: out["temperature"] = float(v)
            except ValueError: pass
        elif k == "top_p":
            try: out["top_p"] = float(v)
            except ValueError: pass
        elif k == "top_k":
            try: out["top_k"] = int(v)
            except ValueError: pass
        elif k == "num_ctx":
            try: out["num_ctx"] = int(v)
            except ValueError: pass
    return out

