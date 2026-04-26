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


def get_con():
    """Single shared DuckDB connection for the Streamlit session, cached.

    DuckDB requires all in-process connections to share configuration, so we
    use one read-write connection for everything. Returns ``None`` if another
    OS process holds an exclusive lock on the file.

    Retry-with-backoff: when the scheduler daemon is running, it briefly holds
    the lock during each tick (~1s) and releases between ticks. We retry for
    up to ~6s so the dashboard reliably hits a free window.
    """
    import time

    @st.cache_resource(show_spinner=False)
    def _open():
        from oasis_llm.db import connect
        return connect()

    delays = [0.0, 0.5, 1.0, 1.5, 2.0, 1.5]  # ~6.5s total
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return _open()
        except duckdb.IOException:
            st.cache_resource.clear()
            continue
    return None


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
