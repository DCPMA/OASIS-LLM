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
