"""Experiments page: create multi-config rating campaigns and trigger runs."""
from __future__ import annotations

import asyncio
import json
import os

import streamlit as st

from oasis_llm import datasets as ds
from oasis_llm import experiments as ex
from oasis_llm.config import RunConfig
from oasis_llm.dashboard_pages._ui import (
    connect_ro, connect_rw, db_locked_warning, kpi, page_header, status_pill,
)


PROVIDERS = ["ollama", "openrouter", "anthropic", "google", "openai"]


def _slug_from_model(model: str) -> str:
    """Derive a short slug suitable as a config name from a model id.

    ``openrouter/anthropic/claude-3.5-haiku`` → ``claude-3.5-haiku``
    ``gemma3:12b`` → ``gemma3-12b``
    """
    # Strip provider prefixes
    s = model.rsplit("/", 1)[-1]
    # Replace separators with dashes
    s = s.replace(":", "-")
    return s or "config"


# Curated suggestions per provider. Ollama is filled live by querying the local daemon.
KNOWN_MODELS: dict[str, list[str]] = {
    "openrouter": [
        "openrouter/google/gemma-3-27b-it",
        "openrouter/google/gemini-2.0-flash-exp:free",
        "openrouter/anthropic/claude-3.5-sonnet",
        "openrouter/openai/gpt-4o-mini",
    ],
    "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
    "google":    ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "openai":    ["gpt-4o", "gpt-4o-mini"],
}


@st.cache_data(ttl=300)
def _openrouter_models(vision_only: bool = False, free_only: bool = False) -> tuple[list[str], str | None]:
    """Query the OpenRouter live model catalogue. Cached for 5 minutes.

    Returns ``(models, error)``. ``error`` is None on success; on failure it is
    a short string the UI can render so listing failures are no longer silent.

    Sends an Authorization header when ``OPENROUTER_API_KEY`` is set so we get
    the same model set the user actually has access to (free models are
    sometimes hidden from anonymous responses).
    """
    import urllib.request
    headers = {"User-Agent": "oasis-llm/1.0"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models", headers=headers,
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    models = payload.get("data", []) or []
    if vision_only:
        models = [
            m for m in models
            if "image" in ((m.get("architecture") or {}).get("input_modalities") or [])
        ]
    if free_only:
        models = [m for m in models if str(m.get("id", "")).endswith(":free")]
    # Sort: Anthropic, Google, OpenAI first (most useful for our use case),
    # then alphabetical. Skip variant-prefix entries like "~anthropic/...".
    def _sort_key(m: dict) -> tuple:
        mid = m.get("id", "")
        head = mid.split("/", 1)[0]
        priority = {"anthropic": 0, "google": 1, "openai": 2}.get(head, 9)
        return (priority, mid.startswith("~"), mid)
    models.sort(key=_sort_key)
    return [f"openrouter/{m['id']}" for m in models if m.get("id")], None


@st.cache_data(ttl=30)
def _ollama_models() -> list[str]:
    """Query the local Ollama daemon for installed models. Cached for 30s."""
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as r:
            data = _json.loads(r.read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


@st.cache_data(ttl=300)
def _ollama_show(name: str) -> dict:
    """Query Ollama /api/show for a model. Cached 5 min. Returns ``{}`` on error."""
    import urllib.request, json as _json
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=_json.dumps({"name": name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            return _json.loads(r.read())
    except Exception:
        return {}


def _ollama_capabilities(name: str) -> list[str]:
    """Return Ollama's reported capabilities list for a model. May be empty.

    NOTE: This field is **unreliable**. e.g. ``gemma4:e4b-mlx-bf16`` reports
    no ``vision`` capability but actually accepts images. Use as informational
    badge only, never as a hard gate.
    """
    return _ollama_show(name).get("capabilities") or []


def _probe_ollama_model(model: str, *, disable_thinking: bool = True) -> dict:
    """Run one rating-style call against an Ollama model. Used by the Probe button.

    Picks the first available approved-dataset image; falls back to a tiny
    in-memory blank PNG if no DB image is available.
    """
    import asyncio, base64, time
    from litellm import acompletion

    # Try to find a real image; fall back to a 1x1 white PNG.
    img_url: str | None = None
    try:
        from oasis_llm import images as _img
        # Try a known small set first; user can re-probe with a richer set later.
        for candidate in ("Spider 1", "Dancing 8", "Socks 1", "Wedding 12"):
            try:
                img_url = _img.image_data_url(candidate)
                break
            except Exception:
                continue
    except Exception:
        img_url = None
    if img_url is None:
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        img_url = "data:image/png;base64," + base64.b64encode(png).decode()

    msgs = [
        {"role": "system", "content": "You are a rating assistant. Output only an integer 1-7."},
        {"role": "user", "content": [
            {"type": "text", "text": "On a 1-7 scale, rate the arousal of this image. Output only the integer."},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ]
    kwargs: dict = {
        "model": f"ollama/{model}",
        "messages": msgs,
        "max_tokens": 64,
        "timeout": 60,
    }
    if disable_thinking:
        kwargs["think"] = False

    t0 = time.monotonic()
    try:
        resp = asyncio.run(acompletion(**kwargs))
        dt = time.monotonic() - t0
        content = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        return {
            "ok": bool(content.strip()),
            "latency": dt,
            "finish": finish,
            "content": content[:200],
            "content_len": len(content),
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "latency": time.monotonic() - t0,
            "finish": None,
            "content": "",
            "content_len": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def _models_for(provider: str, *, vision_only: bool = False, free_only: bool = False) -> tuple[list[str], str | None]:
    """Return ``(models, error)``. ``error`` is None unless the live fetch failed."""
    if provider == "ollama":
        live = _ollama_models()
        return (live or ["qwen3-vl:8b", "gemma3:12b", "llava:latest"]), None
    if provider == "openrouter":
        live, err = _openrouter_models(vision_only=vision_only, free_only=free_only)
        return (live or KNOWN_MODELS["openrouter"]), err
    return KNOWN_MODELS.get(provider, []), None


def render():
    page_header(
        "Experiments",
        "A multi-config rating campaign against ONE approved dataset.",
        icon="🧪",
    )

    # Edit / duplicate mode takes over the page until the user saves or cancels.
    form_mode = st.session_state.get("exp_form_mode")
    if form_mode in ("edit", "duplicate"):
        _create_form(
            edit_exp_id=st.session_state.get("exp_edit_id") if form_mode == "edit" else None,
            mode=form_mode,
        )
        return

    detail_id = st.query_params.get("experiment")
    if detail_id:
        _render_detail(detail_id)
        return

    _render_list()


# ---------------------------------------------------------------------------
def _render_list():
    con_ro = connect_ro()
    rows = ex.list_all(con_ro) if con_ro is not None else []

    n_total = len(rows)
    n_running = sum(1 for r in rows if r.status == "running")
    n_done = sum(1 for r in rows if r.status == "done")
    cols = st.columns(3)
    cols[0].markdown(kpi("Experiments", n_total), unsafe_allow_html=True)
    cols[1].markdown(kpi("Running", n_running), unsafe_allow_html=True)
    cols[2].markdown(kpi("Completed", n_done), unsafe_allow_html=True)

    st.markdown("---")
    tab_list, tab_new, tab_import = st.tabs(
        ["📋  All experiments", "✨  Create new", "📥  Import bundle"]
    )

    with tab_list:
        if not rows:
            st.info("No experiments yet. Create one with the form on the right.")
        else:
            for e in rows:
                _exp_row(e)

    with tab_new:
        _create_form()

    with tab_import:
        _render_import()


def _render_import():
    """Upload + apply an experiment bundle zip."""
    from oasis_llm.bundles import import_experiment

    st.markdown(
        "Upload a `.zip` bundle exported from another workspace. The bundle "
        "ships experiment + configs + runs + trials + dataset metadata. "
        "Image files are **not** transferred — make sure the destination has "
        "the same OASIS images locally."
    )
    uploaded = st.file_uploader("Bundle file", type=["zip"], key="bundle_upload")
    overwrite = st.toggle(
        "Overwrite if experiment already exists",
        value=False,
        help="When on, deletes the existing experiment + its runs/trials before importing.",
    )
    if not uploaded:
        return
    if st.button("📥 Import bundle", type="primary"):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            summary = import_experiment(con, uploaded.read(), overwrite=overwrite)
        except Exception as e:
            st.error(f"Import failed: {e}")
            return
        st.success(
            f"Imported experiment `{summary['experiment_id']}` · "
            f"{summary['imported_runs']} runs · "
            f"{summary['imported_trials']} trials · "
            f"{summary['imported_dataset_images']} dataset images."
        )
        if summary["skipped"]:
            with st.expander(f"Skipped ({len(summary['skipped'])})"):
                for s in summary["skipped"]:
                    st.write(f"- {s}")


def _exp_row(e):
    cols = st.columns([3, 2, 2, 2, 2])
    with cols[0]:
        st.markdown(
            f"<div style='font-weight:600; font-size:1.05rem'>{e.name}</div>"
            f"<div style='color:#8a8aa0; font-size:0.78rem'>{e.experiment_id}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(status_pill(e.status), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f"<div style='color:#8a8aa0; font-size:0.78rem;'>dataset</div>"
            f"<div>{e.dataset_id}</div>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"<div style='color:#8a8aa0; font-size:0.78rem;'>configs</div>"
            f"<div>{len(e.configs)}</div>",
            unsafe_allow_html=True,
        )
    with cols[4]:
        if st.button("Open ›", key=f"openexp_{e.experiment_id}", width='stretch'):
            st.query_params["experiment"] = e.experiment_id
            st.rerun()
    st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def _config_json_to_form_dict(cfg_name: str, cj: dict) -> dict:
    """Convert a stored ``config_json`` row back into the editor's session shape."""
    extra = dict(cj.get("extra_params") or {})
    disable_thinking = bool(extra.get("think") is False) if "think" in extra else True
    return {
        "config_name": cfg_name,
        "config_name_auto": False,  # preserve existing slug verbatim
        "provider": cj.get("provider", "ollama"),
        "model": cj.get("model", ""),
        "temperature": float(cj.get("temperature") or 0.0),
        "samples_per_image": int(cj.get("samples_per_image", 5)),
        "capture_reasoning": bool(cj.get("capture_reasoning", True)),
        "cache_buster": bool(cj.get("cache_buster", True)),
        "max_concurrency": int(cj.get("max_concurrency", 4)),
        "max_retries": int(cj.get("max_retries", 3)),
        "retry_backoff_base_s": float(cj.get("retry_backoff_base_s", 1.0)),
        "retry_backoff_coef": float(cj.get("retry_backoff_coef", 2.0)),
        "ollama_evict_threshold": int(cj.get("ollama_evict_threshold", 3)),
        "max_permanent_failure_rate": float(cj.get("max_permanent_failure_rate", 0.05)),
        "permanent_failure_warmup": int(cj.get("permanent_failure_warmup", 20)),
        "request_timeout_s": int(cj.get("request_timeout_s", 60)),
        "disable_thinking": disable_thinking,
    }


def _create_form(edit_exp_id: str | None = None, mode: str = "create"):
    """Render the experiment editor.

    Modes:
      - ``create``: blank form (default) — calls :func:`ex.create` on submit.
      - ``edit``: pre-loads ``edit_exp_id``'s configs — calls
        :func:`ex.update_configs` on submit. Only valid for ``status='draft'``.
      - ``duplicate``: pre-loads source configs into a fresh draft. Submits
        via :func:`ex.create`.
    """
    con_ro = connect_ro()
    if con_ro is None:
        st.warning("No database yet. Create a dataset first.")
        return
    approved = [d for d in ds.list_all(con_ro) if d.status == "approved"]
    if not approved:
        st.warning("No approved datasets. Approve one first in the Datasets tab.")
        return

    # ── Pre-seed working state from an existing experiment when in edit/duplicate.
    if mode in ("edit", "duplicate") and not st.session_state.get("exp_configs_seeded"):
        src_id = edit_exp_id or st.session_state.get("exp_dup_src")
        if src_id:
            src = ex.get(con_ro, src_id)
            if src is not None:
                st.session_state["exp_configs"] = [
                    _config_json_to_form_dict(c.config_name, c.config_json)
                    for c in src.configs
                ]
                if mode == "edit":
                    st.session_state["exp_default_name"] = src.name
                    st.session_state["exp_edit_dataset"] = src.dataset_id
                    st.session_state["exp_edit_description"] = src.description or ""
                else:  # duplicate
                    st.session_state["exp_default_name"] = f"{src.name}-copy"
                    st.session_state["exp_edit_dataset"] = src.dataset_id
                    st.session_state["exp_edit_description"] = src.description or ""
                st.session_state["exp_configs_seeded"] = True

    if mode == "edit":
        st.info(f"✏️ Editing draft experiment `{edit_exp_id}` — saving will rebuild its configs and reset all pending trials.")
    elif mode == "duplicate":
        st.info("📋 Duplicating an existing experiment — submitting will create a fresh draft.")

    # Working state: list of config dicts
    if "exp_configs" not in st.session_state:
        default_provider = "ollama" if _ollama_models() else "openrouter"
        default_model = _models_for(default_provider)[0][0]
        import os as _os
        try:
            _default_to = int(_os.getenv("OASIS_DEFAULT_TIMEOUT_S", "60"))
        except ValueError:
            _default_to = 60
        st.session_state["exp_configs"] = [
            {
                "config_name": _slug_from_model(default_model),
                "config_name_auto": True,
                "provider": default_provider,
                "model": default_model,
                "temperature": 0.0, "samples_per_image": 5,
                "capture_reasoning": True, "cache_buster": True,
                "max_concurrency": 4,
                "max_retries": 3,
                "retry_backoff_base_s": 1.0,
                "retry_backoff_coef": 2.0,
                "ollama_evict_threshold": 3,
                "max_permanent_failure_rate": 0.05,
                "permanent_failure_warmup": 20,
                "request_timeout_s": _default_to,
                "disable_thinking": True,
            }
        ]

    # Default experiment name: today + short random hash, regenerated each visit.
    if "exp_default_name" not in st.session_state:
        from datetime import date
        import secrets
        st.session_state["exp_default_name"] = (
            f"{date.today().strftime('%Y%m%d')}-{secrets.token_hex(2)}"
        )

    st.markdown("### Experiment metadata")
    mc1, mc2 = st.columns([3, 2])
    name = mc1.text_input("Experiment name", value=st.session_state["exp_default_name"])
    # In edit mode, the dataset is locked (changing it would invalidate the
    # backing trial rows). In duplicate mode we default to the source dataset
    # but allow the user to switch.
    locked_ds = st.session_state.get("exp_edit_dataset") if mode == "edit" else None
    if locked_ds is not None:
        mc2.text_input(
            "Dataset (locked while editing)", value=locked_ds, disabled=True,
        )
        dataset_choice = locked_ds
    else:
        default_ds = st.session_state.get("exp_edit_dataset")
        ds_options = [d.dataset_id for d in approved]
        idx = ds_options.index(default_ds) if default_ds in ds_options else 0
        dataset_choice = mc2.selectbox(
            "Dataset (must be approved)",
            ds_options,
            index=idx,
            format_func=lambda x: x,
        )
    description = st.text_area(
        "Description (optional)",
        height=80,
        value=st.session_state.get("exp_edit_description", ""),
    )

    st.markdown("---")
    st.markdown("### Configs")
    st.caption("Each config becomes a separate run within the experiment.")

    to_remove = None
    for i, cfg in enumerate(st.session_state["exp_configs"]):
        with st.container(border=True):
            head = st.columns([5, 1])
            head[0].markdown(f"**Config {i + 1}**")
            if head[1].button("🗑️", key=f"rm_cfg_{i}", help="Remove this config"):
                to_remove = i
            cc1, cc2 = st.columns(2)
            cfg["provider"] = cc1.selectbox(
                "Provider", PROVIDERS,
                index=PROVIDERS.index(cfg["provider"]) if cfg["provider"] in PROVIDERS else 0,
                key=f"cfgprov_{i}",
            )
            # Model picker: dropdown of installed/known + free-text override.
            # Vision filter applies only to OpenRouter (Ollama caps are unreliable).
            vision_only = (
                cfg.get("_vision_only", True)
                if cfg["provider"] == "openrouter" else False
            )
            free_only = (
                cfg.get("_free_only", False)
                if cfg["provider"] == "openrouter" else False
            )
            suggestions, list_err = _models_for(
                cfg["provider"], vision_only=vision_only, free_only=free_only,
            )
            options = list(dict.fromkeys([cfg["model"], *suggestions, "✨ custom…"]))
            picked = cc2.selectbox(
                "Model", options,
                index=options.index(cfg["model"]) if cfg["model"] in options else 0,
                key=f"cfgmodpick_{i}",
                help=(
                    "All locally installed Ollama models. Capability badges below "
                    "come from /api/show but are unreliable for some MLX builds."
                    if cfg["provider"] == "ollama" else
                    "Live OpenRouter catalogue (cached 5min). "
                    "Toggle below to see all 350+ models."
                    if cfg["provider"] == "openrouter" else
                    "Pick a known model or choose Custom to type your own."
                ),
            )
            if picked == "✨ custom…":
                cfg["model"] = cc2.text_input(
                    "Custom model id", value=cfg["model"], key=f"cfgmodtxt_{i}",
                    label_visibility="collapsed",
                )
            else:
                cfg["model"] = picked
            # Auto-sync the config slug from the model id unless the user
            # has overridden it.
            if cfg.get("config_name_auto", True):
                cfg["config_name"] = _slug_from_model(cfg["model"])
            if cfg["provider"] == "openrouter":
                fcol1, fcol2, fcol3 = st.columns(3)
                cfg["_vision_only"] = fcol1.toggle(
                    "Vision-capable only", value=vision_only,
                    key=f"cfgvision_{i}",
                    help="Hide text-only models from the dropdown (recommended).",
                )
                cfg["_free_only"] = fcol2.toggle(
                    "Free tier only (`:free`)", value=free_only,
                    key=f"cfgfreeonly_{i}",
                    help=(
                        "Show only OpenRouter `:free`-tier models. These are "
                        "rate-limited to 20 requests/minute and 1000/day."
                    ),
                )
                if fcol3.button("🔄 Refresh", key=f"cfgrefresh_{i}",
                                help="Re-fetch the live OpenRouter catalogue (5-min cache otherwise)."):
                    _openrouter_models.clear()
                    st.rerun()
                if list_err:
                    st.warning(
                        f"OpenRouter listing failed — falling back to curated list. "
                        f"`{list_err}`"
                    )
                else:
                    st.caption(
                        f"📡 {len(suggestions)} OpenRouter models loaded "
                        f"(vision_only={vision_only}, free_only={free_only}). "
                        f"5-min cache."
                    )
            if cfg["provider"] == "ollama" and cfg["model"] in _ollama_models():
                caps = _ollama_capabilities(cfg["model"])
                badges = " ".join(f"`{c}`" for c in caps) if caps else "_(none reported)_"
                has_vision = "vision" in caps
                has_thinking = "thinking" in caps
                vision_note = (
                    "✅ vision reported" if has_vision
                    else "⚠️ no `vision` in capabilities — may still work for some MLX builds; use Probe to confirm"
                )
                think_note = (
                    "· 🧠 thinking-capable — keep `Disable thinking` ON to avoid empty responses"
                    if has_thinking else ""
                )
                st.caption(f"capabilities: {badges} · {vision_note} {think_note}")

                pcol1, pcol2 = st.columns([1, 4])
                if pcol1.button("🔬 Probe", key=f"cfgprobe_{i}",
                                help="Run one rating call against an OASIS image and show the result."):
                    with pcol2:
                        with st.spinner(f"Probing {cfg['model']}…"):
                            res = _probe_ollama_model(
                                cfg["model"],
                                disable_thinking=bool(cfg.get("disable_thinking", True)),
                            )
                        if res["ok"]:
                            st.success(
                                f"✅ {res['latency']:.1f}s · finish={res['finish']} · "
                                f"content={res['content']!r}"
                            )
                        else:
                            st.error(
                                f"❌ {res['latency']:.1f}s · finish={res['finish']} · "
                                f"content_len={res['content_len']} · {res.get('error', '')}"
                            )

            # Run id slug — auto from model, with optional override
            with st.expander(
                f"Run id: `{cfg['config_name']}`"
                + ("" if cfg.get("config_name_auto", True) else " (custom)"),
                expanded=False,
            ):
                auto = st.toggle(
                    "Auto-derive from model id", value=cfg.get("config_name_auto", True),
                    key=f"cfgname_auto_{i}",
                    help="When on, the run id slug follows the selected model.",
                )
                cfg["config_name_auto"] = auto
                if auto:
                    cfg["config_name"] = _slug_from_model(cfg["model"])
                    st.caption(f"→ `{cfg['config_name']}`")
                else:
                    cfg["config_name"] = st.text_input(
                        "Custom run id slug",
                        value=cfg["config_name"],
                        key=f"cfgname_text_{i}",
                    )

            cd1, cd2, cd3, cd4 = st.columns(4)
            cfg["temperature"] = cd1.number_input(
                "Temperature", min_value=0.0, max_value=2.0, value=float(cfg["temperature"]),
                step=0.1, key=f"cfgtemp_{i}",
            )
            cfg["samples_per_image"] = cd2.number_input(
                "Samples per image", min_value=1, max_value=50,
                value=int(cfg["samples_per_image"]), key=f"cfgsamp_{i}",
            )
            cfg["capture_reasoning"] = cd3.toggle(
                "Capture reasoning", value=bool(cfg["capture_reasoning"]), key=f"cfgreas_{i}",
            )
            cfg["cache_buster"] = cd4.toggle(
                "Cache buster", value=bool(cfg["cache_buster"]), key=f"cfgcb_{i}",
            )
            if cfg["provider"] == "ollama":
                cfg["disable_thinking"] = st.toggle(
                    "Disable thinking (Ollama)",
                    value=bool(cfg.get("disable_thinking", True)),
                    key=f"cfgthink_{i}",
                    help=(
                        "Sends `think: False` to Ollama. Required for thinking-capable "
                        "models (qwen3.5, gemma4, deepseek-r1) — otherwise hidden CoT "
                        "tokens consume the entire `max_tokens` budget and the response "
                        "comes back empty after a long latency."
                    ),
                )
            cfg["max_concurrency"] = st.slider(
                "Max concurrency",
                min_value=1, max_value=32,
                value=int(cfg.get("max_concurrency", 4)),
                key=f"cfgconc_{i}",
                help=(
                    "Per-config concurrent in-flight requests. The global "
                    "`OASIS_MAX_CONCURRENCY` env var can cap this at runtime."
                ),
            )
            cfg["max_retries"] = st.slider(
                "Max retries per trial",
                min_value=0, max_value=10,
                value=int(cfg.get("max_retries", 3)),
                key=f"cfgretries_{i}",
                help=(
                    "How many times the runner re-attempts a trial that errored "
                    "or returned an unparseable response. Set 0 to disable retries."
                ),
            )
            r1, r2 = st.columns(2)
            with r1:
                cfg["retry_backoff_base_s"] = st.number_input(
                    "Retry backoff base (s)",
                    min_value=0.0, max_value=60.0,
                    value=float(cfg.get("retry_backoff_base_s", 1.0)),
                    step=0.5,
                    key=f"cfgretrybase_{i}",
                    help="Initial delay before the first retry. delay_n = base × coef^n.",
                )
            with r2:
                cfg["retry_backoff_coef"] = st.number_input(
                    "Retry backoff coefficient",
                    min_value=1.0, max_value=10.0,
                    value=float(cfg.get("retry_backoff_coef", 2.0)),
                    step=0.5,
                    key=f"cfgretrycoef_{i}",
                    help="Multiplier applied each retry. 2.0 = exponential (1, 2, 4, 8s).",
                )
            cfg["ollama_evict_threshold"] = st.slider(
                "Ollama evict-on-stall threshold",
                min_value=0, max_value=10,
                value=int(cfg.get("ollama_evict_threshold", 3)),
                key=f"cfgevict_{i}",
                help=(
                    "After this many consecutive Ollama timeouts/500s, the "
                    "runner POSTs `keep_alive: 0` to evict the stuck Ollama "
                    "runner subprocess. Next request reloads cleanly. "
                    "0 = disable. Ignored for non-Ollama providers."
                ),
            )
            pf1, pf2 = st.columns(2)
            with pf1:
                cfg["max_permanent_failure_rate"] = st.number_input(
                    "Auto-cancel: permanent-fail rate threshold",
                    min_value=0.0, max_value=1.0,
                    value=float(cfg.get("max_permanent_failure_rate", 0.05)),
                    step=0.01,
                    format="%.2f",
                    key=f"cfgpfrate_{i}",
                    help=(
                        "If more than this fraction of trials fail with PERMANENT "
                        "(non-transient) errors — e.g. 404 model not found, 400 "
                        "malformed request, 422 schema validation, content policy "
                        "— the config is auto-cancelled and the experiment moves "
                        "to the next one. Set 0 to disable."
                    ),
                )
            with pf2:
                cfg["permanent_failure_warmup"] = st.number_input(
                    "Warmup attempts before threshold applies",
                    min_value=0, max_value=200,
                    value=int(cfg.get("permanent_failure_warmup", 20)),
                    step=5,
                    key=f"cfgpfwarmup_{i}",
                    help=(
                        "Minimum number of attempts before the permanent-fail "
                        "ratio is checked. Prevents single early-flake errors "
                        "from cancelling the whole config."
                    ),
                )
            cfg["request_timeout_s"] = st.slider(
                "Request timeout (s)",
                min_value=10, max_value=600,
                value=int(cfg.get("request_timeout_s", 60)),
                step=10,
                key=f"cfgtimeout_{i}",
                help=(
                    "Per-call timeout. Increase for large local models that "
                    "queue inside Ollama under concurrency (e.g. 27B/40B+ on "
                    "Apple Silicon often need 120-300s). Failed calls take the "
                    "FULL timeout × (max_retries+1) before giving up."
                ),
            )

    if to_remove is not None and len(st.session_state["exp_configs"]) > 1:
        st.session_state["exp_configs"].pop(to_remove)
        st.rerun()

    btns = st.columns([1, 1, 1, 2])
    with btns[0]:
        if st.button("➕ Add config", width='stretch'):
            n = len(st.session_state["exp_configs"]) + 1
            last = st.session_state["exp_configs"][-1]
            st.session_state["exp_configs"].append({
                "config_name": _slug_from_model(last["model"]),
                "config_name_auto": True,
                "provider": last["provider"],
                "model": last["model"],
                "temperature": 0.0, "samples_per_image": 5,
                "capture_reasoning": True, "cache_buster": True,
                "max_concurrency": int(last.get("max_concurrency", 4)),
                "max_retries": int(last.get("max_retries", 3)),
                "retry_backoff_base_s": float(last.get("retry_backoff_base_s", 1.0)),
                "retry_backoff_coef": float(last.get("retry_backoff_coef", 2.0)),
                "ollama_evict_threshold": int(last.get("ollama_evict_threshold", 3)),
                "max_permanent_failure_rate": float(last.get("max_permanent_failure_rate", 0.05)),
                "permanent_failure_warmup": int(last.get("permanent_failure_warmup", 20)),
                "request_timeout_s": int(last.get("request_timeout_s", 60)),
                "disable_thinking": bool(last.get("disable_thinking", True)),
            })
            st.rerun()
    with btns[1]:
        if st.button("Reset", width='stretch'):
            for k in (
                "exp_configs", "exp_default_name", "exp_configs_seeded",
                "exp_edit_dataset", "exp_edit_description",
            ):
                st.session_state.pop(k, None)
            st.rerun()
    with btns[2]:
        if mode in ("edit", "duplicate") and st.button("✖️ Cancel", width='stretch'):
            for k in (
                "exp_configs", "exp_default_name", "exp_configs_seeded",
                "exp_edit_dataset", "exp_edit_description",
                "exp_form_mode", "exp_edit_id", "exp_dup_src",
            ):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("---")
    cta_label = {
        "edit": "💾 Save changes",
        "duplicate": "📋 Create duplicate",
    }.get(mode, "✨ Create experiment")
    if st.button(cta_label, type="primary"):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            cfgs_payload = []
            for c in st.session_state["exp_configs"]:
                c2 = {k: v for k, v in c.items() if not k.startswith("_")}
                # Translate UI flag → litellm extra_params for Ollama.
                if c2.get("provider") == "ollama" and c2.pop("disable_thinking", True):
                    extra = dict(c2.get("extra_params") or {})
                    extra.setdefault("think", False)
                    c2["extra_params"] = extra
                else:
                    c2.pop("disable_thinking", None)
                cfgs_payload.append(c2)
            if mode == "edit" and edit_exp_id:
                ex.update_configs(
                    con, edit_exp_id, cfgs_payload,
                    name=name or None,
                    description=description or None,
                )
                exp_id = edit_exp_id
            else:
                exp_id = ex.create(
                    con, name, dataset_choice,
                    cfgs_payload,
                    description=description or None,
                )
        except Exception as e:
            st.error(f"Failed: {e}")
            return
        success_msg = {
            "edit": f"Updated experiment `{exp_id}`.",
            "duplicate": f"Duplicated to new experiment `{exp_id}`.",
        }.get(mode, f"Created experiment `{exp_id}`.")
        st.success(success_msg)
        for k in (
            "exp_configs", "exp_default_name", "exp_configs_seeded",
            "exp_edit_dataset", "exp_edit_description",
            "exp_form_mode", "exp_edit_id", "exp_dup_src",
        ):
            st.session_state.pop(k, None)
        st.query_params["experiment"] = exp_id
        st.rerun()


# ---------------------------------------------------------------------------
def _render_detail(exp_id: str):
    con_ro = connect_ro()
    if con_ro is None:
        st.error("No DB."); return
    e = ex.get(con_ro, exp_id)
    if e is None:
        st.error(f"Unknown experiment `{exp_id}`.")
        if st.button("← Back"):
            del st.query_params["experiment"]; st.rerun()
        return

    # Live polling: refresh every 2s while a config is running in this session.
    thread = st.session_state.get(f"exp_thread_{exp_id}")
    if thread is not None and not thread.is_alive():
        # Worker finished — clean up state and rerun once for the final view.
        st.session_state.pop(f"exp_thread_{exp_id}", None)
        thread = None
    is_running = e.status == "running" or thread is not None
    if is_running:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=2000, key=f"exp_refresh_{exp_id}")
        except ImportError:
            st.caption("⏱ Live — install `streamlit-autorefresh` for auto-update.")

    top = st.columns([6, 2])
    with top[0]:
        st.markdown(
            f"# 🧪 {e.name} &nbsp; {status_pill(e.status)}",
            unsafe_allow_html=True,
        )
        st.caption(f"`{e.experiment_id}` · dataset: **{e.dataset_id}** · "
                   f"{len(e.configs)} configs · created {str(e.created_at)[:19] if e.created_at else '—'}")
        if e.description:
            st.write(e.description)
    with top[1]:
        if st.button("← All experiments", width='stretch'):
            del st.query_params["experiment"]; st.rerun()
        # Export bundle
        from oasis_llm.bundles import export_experiment
        try:
            bundle_bytes = export_experiment(con_ro, exp_id)
            st.download_button(
                "📦 Export bundle (.zip)",
                data=bundle_bytes,
                file_name=f"experiment_{exp_id}.zip",
                mime="application/zip",
                width='stretch',
                help=(
                    "Bundles experiment + configs + runs + trials + dataset metadata. "
                    "Image files are NOT included — colleague needs the same image_ids locally."
                ),
            )
        except Exception as ex_err:
            st.caption(f"⚠️ Export unavailable: {ex_err}")

    prog = ex.progress(con_ro, exp_id)
    total_done = sum(p["done"] for p in prog)
    total_pending = sum(p["pending"] for p in prog)
    total_failed = sum(p["failed"] for p in prog)
    total_trials = sum(p["total"] for p in prog)
    total_cost = sum(p["cost_usd"] for p in prog)
    cols = st.columns(4)
    cols[0].markdown(
        kpi("Progress", f"{total_done}/{total_trials}",
            f"{total_pending} pending · {total_failed} failed"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(kpi("Configs", len(e.configs)), unsafe_allow_html=True)
    cols[2].markdown(kpi("Total cost", f"${total_cost:.4f}"), unsafe_allow_html=True)
    pct = (100 * total_done / total_trials) if total_trials else 0
    cols[3].markdown(kpi("% complete", f"{pct:.1f}%"), unsafe_allow_html=True)

    if total_trials:
        st.progress(min(1.0, total_done / total_trials))

    # ── DB lock-conflict chooser (set by _run_experiment pre-flight) ───────
    conflict_key = f"db_lock_conflict_{exp_id}"
    if st.session_state.get(conflict_key) is not None:
        _render_lock_conflict(exp_id, st.session_state[conflict_key])
        return

    # Run / re-run button
    btn_cols = st.columns([1, 1, 1, 1, 1, 1, 2])
    thread_active = st.session_state.get(f"exp_thread_{exp_id}") is not None
    with btn_cols[0]:
        if st.button("▶️ Run all configs", type="primary",
                     disabled=(total_pending + total_failed == 0) or thread_active):
            _run_experiment(exp_id)
            st.rerun()
    with btn_cols[1]:
        # Edit configs — only available while the experiment is in draft mode
        # (i.e. no trials have run yet). Clicking enters the edit form which
        # rebuilds the experiment's configs/runs/trials in place on save.
        edit_disabled = e.status != "draft" or thread_active
        if st.button(
            "✏️ Edit configs",
            disabled=edit_disabled,
            help=(
                "Available only while the experiment is a draft (no trials run). "
                "After running starts, configs are immutable — duplicate instead."
                if edit_disabled else
                "Modify this draft's configs. Saving rebuilds the backing trial rows."
            ),
        ):
            for k in (
                "exp_configs", "exp_default_name", "exp_configs_seeded",
                "exp_edit_dataset", "exp_edit_description",
            ):
                st.session_state.pop(k, None)
            st.session_state["exp_form_mode"] = "edit"
            st.session_state["exp_edit_id"] = exp_id
            st.rerun()
    with btn_cols[2]:
        # Duplicate — works regardless of status. Creates a new draft seeded
        # with this experiment's configs (trials are NOT copied).
        if st.button(
            "📋 Duplicate",
            disabled=thread_active,
            help="Create a new draft experiment seeded with these configs.",
        ):
            for k in (
                "exp_configs", "exp_default_name", "exp_configs_seeded",
                "exp_edit_dataset", "exp_edit_description",
            ):
                st.session_state.pop(k, None)
            st.session_state["exp_form_mode"] = "duplicate"
            st.session_state["exp_dup_src"] = exp_id
            st.session_state.pop("exp_edit_id", None)
            del st.query_params["experiment"]
            st.rerun()
    with btn_cols[3]:
        if thread_active and st.button("⏹️ Cancel"):
            ex.update_status(connect_rw(), exp_id, "cancelled")
            # The runner's _claim_one watches each run's status; flagging the
            # experiment is a soft signal — also flag each non-done run.
            con = connect_rw()
            for c in e.configs:
                con.execute(
                    "UPDATE runs SET status='cancelled' WHERE run_id=? AND status IN ('running','pending')",
                    [c.run_id],
                )
            st.session_state.pop(f"exp_thread_{exp_id}", None)
            st.rerun()
    with btn_cols[4]:
        if not thread_active and st.button("🗑️ Delete"):
            st.session_state[f"confirm_delexp_{exp_id}"] = True
        if st.session_state.get(f"confirm_delexp_{exp_id}"):
            if st.button("⚠️ Confirm delete"):
                con = connect_rw()
                if con is None: db_locked_warning(); return
                ex.delete(con, exp_id)
                del st.query_params["experiment"]
                st.session_state.pop(f"confirm_delexp_{exp_id}", None)
                st.rerun()

    st.markdown("---")
    st.markdown("### Per-config progress")
    by_name = {c.config_name: c.config_json for c in e.configs}
    table_rows = []
    for p in prog:
        cfg = by_name.get(p["config_name"], {})
        pct = (100 * p["done"] / p["total"]) if p["total"] else 0
        avg = p["avg_latency_ms"]
        table_rows.append({
            "Config": p["config_name"],
            "Model": cfg.get("model", "—"),
            "Temp": cfg.get("temperature", "—"),
            "Samples": cfg.get("samples_per_image", "—"),
            "Done": f"{p['done']}/{p['total']}",
            "Failed": p["failed"],
            "% done": f"{pct:.1f}%",
            "Cost": f"${p['cost_usd']:.4f}",
            # Coerce to string so Streamlit/Arrow doesn't choke when some
            # configs have measured latency (int) and others are still pending
            # ("—").
            "Avg ms": f"{avg:.0f}" if avg else "—",
            "run_id": p["run_id"],
        })
    st.dataframe(table_rows, width='stretch', hide_index=True)

    with st.expander("🔧 Raw config JSON"):
        st.json({c.config_name: c.config_json for c in e.configs})


def _render_lock_conflict(exp_id: str, holder_pid: int):
    """Render the chooser when another OS process holds the DuckDB file lock."""
    from oasis_llm.dashboard_pages._ui import render_lock_conflict
    render_lock_conflict(f"db_lock_conflict_{exp_id}", holder_pid)


def _run_experiment(exp_id: str):
    """Launch all pending configs in a background thread so the UI can refresh.

    The thread shares the Streamlit process's DuckDB connection; DuckDB
    serializes writes via its internal lock, and the asyncio loop runs in
    just one thread, so concurrent use with the main Streamlit thread is
    safe in practice. Live progress is polled by `streamlit-autorefresh`.
    """
    import threading

    from oasis_llm.db import lock_holder_pid
    from oasis_llm.runner import run as run_async

    # ── Pre-flight: is another OS process holding the DB lock? ─────────────
    # DuckDB allows only one process to hold the RW lock. If a CLI run or
    # another Python process is currently holding it, surface a chooser.
    holder = lock_holder_pid()
    if holder is not None and holder != os.getpid():
        st.session_state[f"db_lock_conflict_{exp_id}"] = holder
        st.rerun()
        return

    con = connect_rw()
    if con is None:
        db_locked_warning(); return
    e = ex.get(con, exp_id)
    if e is None:
        st.error("Vanished?"); return

    state_key = f"exp_thread_{exp_id}"
    if st.session_state.get(state_key) is not None:
        st.warning("Already running."); return

    ex.update_status(con, exp_id, "running")

    def _worker():
        # Background thread: use its OWN connection to avoid racing the
        # main Streamlit thread on the cached connection object. Both
        # connections point at the same DB file with identical config,
        # which DuckDB allows in-process.
        from oasis_llm.db import connect as _connect
        try:
            worker_con = _connect()
        except Exception as e_open:  # noqa: BLE001
            print(f"[experiment {exp_id}] could not open DB: {e_open}")
            return
        try:
            for c in e.configs:
                # Honor experiment-level cancellation between configs.
                exp_status = worker_con.execute(
                    "SELECT status FROM experiments WHERE experiment_id=?", [exp_id]
                ).fetchone()
                if exp_status and exp_status[0] == "cancelled":
                    break
                payload = dict(c.config_json)
                payload["name"] = c.run_id
                payload["image_set"] = e.dataset_id
                cfg = RunConfig(**payload)
                pending = worker_con.execute(
                    "SELECT count(*) FROM trials WHERE run_id=? AND status IN ('pending','failed')",
                    [c.run_id],
                ).fetchone()[0]
                if pending > 0:
                    try:
                        # Reset 'failed' trials to 'pending' so we retry them.
                        worker_con.execute(
                            "UPDATE trials SET status='pending', error=NULL "
                            "WHERE run_id=? AND status='failed'",
                            [c.run_id],
                        )
                        asyncio.run(run_async(cfg, worker_con))
                    except Exception as ex_err:  # noqa: BLE001
                        worker_con.execute(
                            "UPDATE runs SET status='failed' WHERE run_id=?",
                            [c.run_id],
                        )
                        print(f"[experiment {exp_id}] config {c.config_name} crashed: {ex_err}")
            # Final status
            final = worker_con.execute(
                "SELECT status FROM experiments WHERE experiment_id=?", [exp_id]
            ).fetchone()
            if final and final[0] != "cancelled":
                ex.update_status(worker_con, exp_id, "done")
        except Exception as outer:  # noqa: BLE001
            print(f"[experiment {exp_id}] worker crashed: {outer}")
        finally:
            try:
                worker_con.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name=f"exp-{exp_id}", daemon=True)
    t.start()
    st.session_state[state_key] = t
    st.toast(f"▶️ Started experiment {exp_id} in background — page will auto-refresh.")
