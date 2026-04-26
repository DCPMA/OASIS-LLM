"""Shared UI primitives: theme injection, layout helpers, common widgets."""
from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st

from oasis_llm.db import DB_PATH


CSS = """
<style>
/* === base reset === */
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* === sidebar === */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e0e18 0%, #0a0a12 100%);
  border-right: 1px solid #1f1f2e;
}
[data-testid="stSidebar"] .stRadio > div { gap: 0.25rem; }
[data-testid="stSidebar"] .stRadio label {
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  transition: background 0.15s ease;
  width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1a1a26; }

/* === cards === */
.kpi-card {
  background: linear-gradient(135deg, #15151f 0%, #1a1a26 100%);
  border: 1px solid #2a2a3e;
  border-radius: 12px;
  padding: 1rem 1.25rem;
}
.kpi-card .label {
  font-size: 0.75rem;
  color: #8a8aa0;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin: 0;
}
.kpi-card .value {
  font-size: 1.75rem;
  font-weight: 600;
  color: #e7e7ee;
  margin: 0.15rem 0 0;
  line-height: 1.1;
}
.kpi-card .delta {
  font-size: 0.78rem;
  color: #a78bfa;
  margin: 0.25rem 0 0;
}

/* === status pills === */
.pill {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.pill-draft    { background: #3a3a1a; color: #ffd76b; }
.pill-approved { background: #1a3a25; color: #6bff9a; }
.pill-archived { background: #2a2a35; color: #8a8aa0; }
.pill-running  { background: #1a2a3a; color: #6baeff; }
.pill-done     { background: #1a3a25; color: #6bff9a; }
.pill-failed   { background: #3a1a1a; color: #ff6b6b; }
.pill-builtin  { background: #2a1a3a; color: #c89aff; }
.pill-generated{ background: #1a3a3a; color: #6bdcff; }

/* === section dividers === */
hr {
  border: none;
  border-top: 1px solid #2a2a3e;
  margin: 1.5rem 0;
}

/* === buttons === */
.stButton > button {
  border-radius: 8px;
  border: 1px solid #2a2a3e;
  transition: all 0.15s ease;
}
.stButton > button:hover {
  border-color: #a78bfa;
}

/* === dataframe === */
[data-testid="stDataFrame"] {
  border: 1px solid #2a2a3e;
  border-radius: 8px;
  overflow: hidden;
}

/* === tabs === */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 1px solid #2a2a3e;
}
.stTabs [data-baseweb="tab"] {
  padding: 0.5rem 1rem;
  border-radius: 8px 8px 0 0;
}

/* === image grid hover === */
.stImage img {
  border-radius: 8px;
  border: 1px solid #2a2a3e;
  transition: transform 0.15s ease, border-color 0.15s ease;
}
.stImage img:hover {
  transform: scale(1.02);
  border-color: #a78bfa;
}

/* === expander === */
[data-testid="stExpander"] {
  border: 1px solid #2a2a3e;
  border-radius: 8px;
  background: #15151f;
}

/* === metric override === */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #15151f 0%, #1a1a26 100%);
  border: 1px solid #2a2a3e;
  border-radius: 12px;
  padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] {
  color: #8a8aa0;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
  font-size: 1.75rem;
  font-weight: 600;
}
</style>
"""


def apply_theme():
    """Inject custom CSS once per render."""
    st.markdown(CSS, unsafe_allow_html=True)


def status_pill(status: str) -> str:
    """Return HTML for a coloured status pill."""
    return f'<span class="pill pill-{status}">{status}</span>'


def kpi(label: str, value, delta: str | None = None) -> str:
    """Return HTML for a KPI card. Use with st.markdown(..., unsafe_allow_html=True)."""
    delta_html = f'<p class="delta">{delta}</p>' if delta else ""
    return (
        f'<div class="kpi-card">'
        f'<p class="label">{label}</p>'
        f'<p class="value">{value}</p>'
        f'{delta_html}'
        f'</div>'
    )


def page_header(title: str, subtitle: str | None = None, icon: str = ""):
    """Standard page header."""
    head = f"{icon} {title}" if icon else title
    st.markdown(f"# {head}")
    if subtitle:
        st.caption(subtitle)


def star_button(entity_type: str, entity_id: str, *, key_suffix: str = "") -> bool:
    """Render a ⭐ / ☆ toggle button. Returns the new starred state.

    Streamlit re-renders inside ``st.button``'s click branch, so the caller
    typically does not need the return value — but it's available so the
    page can update an in-memory set without re-querying.
    """
    from oasis_llm import favorites as _fav
    con = get_con()
    if con is None:
        return False
    starred = _fav.is_starred(con, entity_type, entity_id)
    label = "⭐" if starred else "☆"
    help_text = "Remove from favourites" if starred else "Add to favourites"
    key = f"star_{entity_type}_{entity_id}_{key_suffix}".replace("/", "_")
    if st.button(label, key=key, help=help_text):
        new_state = _fav.toggle(con, entity_type, entity_id)
        st.rerun()
        return new_state
    return starred


def starred_filter_toggle(entity_type: str, *, key: str | None = None) -> bool:
    """Render a 'Show starred only' toggle. Returns its current state.

    Pages call this once per render and use the return value to filter their
    list.
    """
    return st.toggle(
        "⭐ Starred only",
        value=False,
        key=key or f"starred_only_{entity_type}",
        help="Filter the list to show only items you have starred.",
    )


# ─── Model blinding ─────────────────────────────────────────────────────────
def blind_models_toggle(*, key: str = "blind_models") -> bool:
    """Sidebar toggle that requests anonymisation of model identifiers.

    When ``True``, callers should pass model names through :func:`apply_blinding`
    before rendering them in any chart, table, or download. The mapping is
    consistent within a session and shuffled when the toggle is re-enabled
    after being off.
    """
    return st.checkbox(
        "🙈 Blind models",
        value=False,
        key=key,
        help=(
            "Replace model identifiers with anonymised labels (Model A, "
            "Model B, …) across every analytic on this page. Use the "
            "🔓 Reveal mapping expander to un-blind."
        ),
    )


def apply_blinding(models: list[str], *, on: bool, session_key: str = "_blind_map") -> dict[str, str]:
    """Return a dict mapping each real model id to its display label.

    When ``on`` is False, the mapping is the identity. When True, models are
    sorted deterministically and assigned ``Model A``, ``Model B``, ... A
    consistent mapping is cached on ``st.session_state[session_key]`` so the
    same model keeps the same anonymised label for the duration of the
    session, even as the user navigates between tabs.
    """
    if not on:
        return {m: m for m in models}
    cache = st.session_state.setdefault(session_key, {})
    # Refresh mapping if the set of models changes
    cached_set = set(cache.keys())
    new_set = set(models)
    if cached_set != new_set:
        # Stable, deterministic ordering for reproducibility within a session
        ordered = sorted(new_set)
        cache = {}
        for i, m in enumerate(ordered):
            # 26 letters then double-letter: Model A..Z, AA, AB, …
            if i < 26:
                label = f"Model {chr(ord('A') + i)}"
            else:
                a, b = divmod(i, 26)
                label = f"Model {chr(ord('A') + a - 1)}{chr(ord('A') + b)}"
            cache[m] = label
        st.session_state[session_key] = cache
    return cache


def reveal_blinding_expander(mapping: dict[str, str]) -> None:
    """Render an expander that reveals the active blinding mapping."""
    if not mapping:
        return
    is_blinded = any(real != label for real, label in mapping.items())
    if not is_blinded:
        return
    with st.expander("🔓 Reveal mapping (real model ↔ anonymised label)"):
        rows = sorted(mapping.items(), key=lambda kv: kv[1])
        for real, label in rows:
            st.markdown(f"- `{label}` → **{real}**")


def bounded_number_input(
    label: str,
    *,
    value,
    min_value,
    max_value,
    step=None,
    key: str,
    help: str | None = None,
    format: str | None = None,
    use_slider: bool = True,
):
    """Slider with an "✏️ Manual entry" escape hatch for out-of-range values.

    The label-side widget is a ``st.slider`` (or ``st.number_input`` when
    ``use_slider=False`` or the existing ``value`` is already outside the
    ``[min_value, max_value]`` band). A small toggle next to the label
    lets the user switch to a free ``number_input`` with no upper bound,
    so they can dial a value beyond the slider's recommended range when
    needed.

    Returns the chosen numeric value, type-matched to ``min_value``.
    """
    is_int = isinstance(min_value, int) and isinstance(max_value, int) and not isinstance(value, float)
    out_of_band = (value is not None) and (value < min_value or value > max_value)

    manual_key = f"{key}__manual"
    # Sticky default: stay in manual mode once the user opts in OR if the
    # incoming value is already out of band.
    if manual_key not in st.session_state:
        st.session_state[manual_key] = bool(out_of_band)

    head = st.columns([5, 1])
    with head[1]:
        st.session_state[manual_key] = st.toggle(
            "✏️",
            value=st.session_state[manual_key],
            key=f"{key}__manual_toggle",
            help="Type a value manually (allows beyond-recommended-range).",
        )
    with head[0]:
        if st.session_state[manual_key] or out_of_band:
            kwargs = dict(
                label=label,
                value=value,
                key=key,
                help=help,
            )
            if step is not None:
                kwargs["step"] = step
            if format is not None:
                kwargs["format"] = format
            # No min/max bounds — that's the whole point of manual mode.
            return st.number_input(**kwargs)
        else:
            if use_slider:
                kwargs = dict(
                    label=label,
                    min_value=min_value,
                    max_value=max_value,
                    value=value if value is not None else min_value,
                    key=key,
                    help=help,
                )
                if step is not None:
                    kwargs["step"] = step
                return st.slider(**kwargs)
            else:
                kwargs = dict(
                    label=label,
                    min_value=min_value,
                    max_value=max_value,
                    value=value if value is not None else min_value,
                    key=key,
                    help=help,
                )
                if step is not None:
                    kwargs["step"] = step
                if format is not None:
                    kwargs["format"] = format
                return st.number_input(**kwargs)


def get_con():
    """Single shared DuckDB connection for the Streamlit session, cached.

    DuckDB requires all in-process connections to share configuration, so we
    use one read-write connection for everything. Returns ``None`` if another
    OS process holds an exclusive lock on the file.
    """
    def _is_lock_error(exc: BaseException) -> bool:
        message = str(exc)
        return (
            "Could not set lock on file" in message
            or "Conflicting lock is held" in message
        )

    def _render_open_error(exc: BaseException) -> None:
        message = str(exc)
        wal_path = Path(f"{DB_PATH}.wal")
        if "Failure while replaying WAL file" in message:
            st.error(
                "DuckDB could not replay the local WAL file. "
                "The main database file may be stale while newer writes remain only in the WAL."
            )
            st.caption(
                f"Back up both `{DB_PATH}` and `{wal_path}` before recovery. "
                "Removing the WAL can discard any imports or writes that were not checkpointed yet."
            )
        else:
            st.error(f"Failed to open database: {type(exc).__name__}: {message}")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {message}")

    @st.cache_resource(show_spinner=False)
    def _open():
        from oasis_llm.db import connect
        return connect()
    try:
        con = _open()
    except duckdb.IOException as exc:
        st.cache_resource.clear()
        if not _is_lock_error(exc):
            _render_open_error(exc)
            st.stop()
        return None
    except duckdb.InternalException as exc:
        st.cache_resource.clear()
        _render_open_error(exc)
        st.stop()
        return None
    # Auto-heal: re-run lightweight migrations on every retrieval so a
    # cached connection picks up tables added by newer code without
    # forcing the user to restart Streamlit. CREATE TABLE IF NOT EXISTS
    # is a no-op when the table already exists, so cost is negligible.
    try:
        from oasis_llm.db import ensure_schema
        ensure_schema(con)
    except Exception:
        # Don't let migration probing take down the page; downstream code
        # has its own defensive guards.
        pass
    return con


# Backwards-compat shims; both now return the shared cached connection.
connect_ro = get_con
connect_rw = get_con


def db_locked_warning():
    """Render a friendly DB-locked banner with a kill/skip chooser.

    Falls back to a plain error if we can't identify the holder.
    """
    import os
    from oasis_llm.db import lock_holder_pid
    holder = lock_holder_pid()
    if holder is not None and holder != os.getpid():
        render_lock_conflict("global_lock_conflict", holder)
        return
    st.error(
        "🔒 Database is locked by another OS process (e.g. an active `oasis-llm run`). "
        "Stop it and click *Rerun* in the top-right."
    )


def render_lock_conflict(state_key: str, holder_pid: int) -> None:
    """Render a chooser when another OS process holds the DuckDB lock.

    Stores its state under ``st.session_state[state_key]`` so callers can
    show the chooser by setting that key and gating subsequent UI on it.
    """
    import os
    import signal
    import subprocess
    import time

    proc_label = f"PID {holder_pid}"
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(holder_pid)],
            capture_output=True, text=True, timeout=2,
        )
        cmd = out.stdout.strip()
        if cmd:
            proc_label = f"PID {holder_pid} — `{cmd[:80]}`"
    except Exception:
        pass

    st.error(
        "🔒 **Database is locked by another OS process.** "
        f"DuckDB only allows one writer at a time.\n\n{proc_label}"
    )
    st.caption(
        "Usually a `oasis-llm run` CLI process or another Python script "
        "is currently using the database file."
    )
    cols = st.columns(3)
    if cols[0].button("⏭️ Skip", width='stretch',
                      key=f"skip_{state_key}"):
        st.session_state.pop(state_key, None)
        st.rerun()
    if cols[1].button(f"💀 Terminate PID {holder_pid} & retry",
                      width='stretch', type="primary",
                      key=f"kill_{state_key}"):
        try:
            os.kill(holder_pid, signal.SIGTERM)
        except ProcessLookupError:
            st.warning("Process already gone — retrying…")
        except PermissionError:
            st.error(f"Permission denied killing PID {holder_pid}.")
            return
        from oasis_llm.db import lock_holder_pid
        for _ in range(20):
            time.sleep(0.2)
            if lock_holder_pid() is None:
                break
        else:
            try:
                os.kill(holder_pid, signal.SIGKILL)
            except Exception:
                pass
            time.sleep(0.5)
        st.session_state.pop(state_key, None)
        st.cache_resource.clear()
        st.toast(f"Killed PID {holder_pid}. Click the action again to retry.")
        st.rerun()
    if cols[2].button("🔄 Recheck", width='stretch',
                      key=f"recheck_{state_key}"):
        from oasis_llm.db import lock_holder_pid
        new_holder = lock_holder_pid()
        if new_holder is None or new_holder == os.getpid():
            st.session_state.pop(state_key, None)
        else:
            st.session_state[state_key] = new_holder
        st.rerun()
