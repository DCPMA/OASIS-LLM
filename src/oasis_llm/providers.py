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
_MODEL_SUPPORTED_CACHE: dict[tuple, list[str]] = {}


# Static fallback for providers / failure cases. Empty list means "allow
# everything" — the UI shouldn't gate fields when we don't know.
_PROVIDER_PARAM_FALLBACK: dict[str, list[str]] = {
    # Hard-coded union of common params; used when the live API call fails.
    "openrouter": [
        "max_tokens", "temperature", "top_p", "top_k", "stop", "seed",
        "frequency_penalty", "presence_penalty", "repetition_penalty",
        "min_p", "response_format",
    ],
    # Ollama supports everything its Modelfile + generate API expose.
    "ollama": [
        "temperature", "top_p", "top_k", "max_tokens", "stop", "seed",
        "min_p", "frequency_penalty", "presence_penalty",
        "repeat_penalty", "repeat_last_n", "num_ctx", "num_predict",
        "mirostat", "mirostat_eta", "mirostat_tau", "tfs_z", "typical_p",
    ],
    # Other providers — leave empty (allow everything).
    "anthropic": [],
    "google": [],
    "openai": [],
}


def fetch_model_defaults(
    provider: str, model: str, *, api_base: str | None = None,
    timeout_s: float = 4.0,
) -> dict:
    """Best-effort sampling-parameter defaults for ``(provider, model)``.

    Returns a dict possibly containing keys: ``temperature``, ``top_p``,
    ``top_k``, ``min_p``, ``num_ctx``, ``num_predict``, ``repeat_penalty``,
    ``repeat_last_n``, ``mirostat``, ``mirostat_eta``, ``mirostat_tau``,
    ``tfs_z``, ``typical_p``, ``frequency_penalty``, ``presence_penalty``,
    ``seed``. Any key may be missing.

    * **Ollama**: queries ``/api/show`` and parses the full ``parameters``
      Modelfile block — every numeric line is surfaced.
    * **OpenRouter / Anthropic / Google / OpenAI**: return ``{}`` — those
      providers don't expose recommended-default endpoints. The UI uses
      generic fallback ranges when no defaults are detected.

    The result is cached for the lifetime of the process. Network errors
    return ``{}`` rather than raising.
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


def fetch_model_supported(
    provider: str, model: str, *, api_base: str | None = None,
    timeout_s: float = 4.0,
) -> list[str]:
    """Return the list of sampling parameter names this model supports.

    * **OpenRouter**: hits ``/api/v1/parameters/{model_slug}`` which returns
      ``{"data": {"supported_parameters": [...]}}``. ``:free``-tier models
      typically expose a smaller subset (e.g. no ``top_k``).
    * **Ollama**: returns the static union of Modelfile params (any model
      can accept all of them via the generate API).
    * Others: returns the static fallback list.

    On network failure or missing API key, falls back to the static
    ``_PROVIDER_PARAM_FALLBACK`` list. Empty list means "no info — let the
    UI show everything". Result is cached per ``(provider, model, api_base)``.
    """
    key = (provider, model, api_base or "")
    cached = _MODEL_SUPPORTED_CACHE.get(key)
    if cached is not None:
        return cached
    out: list[str] = []
    try:
        if provider == "openrouter":
            out = _fetch_openrouter_supported(model, api_base, timeout_s)
        elif provider == "ollama":
            out = list(_PROVIDER_PARAM_FALLBACK["ollama"])
    except Exception:
        out = []
    if not out:
        out = list(_PROVIDER_PARAM_FALLBACK.get(provider, []))
    _MODEL_SUPPORTED_CACHE[key] = out
    return out


def _fetch_openrouter_supported(
    model: str, api_base: str | None, timeout_s: float,
) -> list[str]:
    """Query OpenRouter's per-model supported_parameters endpoint.

    Endpoint: ``GET /api/v1/parameters/{model_slug}`` (requires Bearer auth).
    Returns the model-specific list, e.g. ``:free`` tiers usually drop
    ``top_k`` and the penalty knobs.
    """
    import json as _json
    import urllib.request

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return []
    base = (api_base or "https://openrouter.ai/api/v1").rstrip("/")
    slug = model[len("openrouter/"):] if model.startswith("openrouter/") else model
    req = urllib.request.Request(
        f"{base}/parameters/{slug}",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = _json.loads(r.read().decode())
    data = body.get("data") or {}
    sup = data.get("supported_parameters") or []
    return [str(s) for s in sup]


def _fetch_ollama_defaults(model: str, api_base: str | None, timeout_s: float) -> dict:
    """Parse Ollama's ``/api/show`` ``parameters`` text block.

    Ollama returns the ``Modelfile`` parameters as a newline-delimited string
    like::

        num_ctx 4096
        temperature 0.7
        top_k 40
        top_p 0.9
        stop "<|eot_id|>"

    We surface every numeric parameter; ``stop`` (string) is dropped.
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

    int_keys = {
        "top_k", "num_ctx", "num_predict", "num_keep", "repeat_last_n",
        "mirostat", "seed",
    }
    float_keys = {
        "temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty",
        "frequency_penalty", "mirostat_eta", "mirostat_tau", "tfs_z",
        "typical_p",
    }
    out: dict = {}
    for line in str(raw).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        k, v = parts[0], parts[1].strip().strip('"')
        if k in int_keys:
            try: out[k] = int(v)
            except ValueError: pass
        elif k in float_keys:
            try: out[k] = float(v)
            except ValueError: pass
    return out

