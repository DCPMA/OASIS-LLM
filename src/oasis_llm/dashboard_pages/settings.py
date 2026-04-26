"""Settings page: DB maintenance, runtime knobs, env inspection."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from oasis_llm.dashboard_pages._ui import (
    connect_rw, db_locked_warning, page_header,
)

ENV_FILE = Path(".env")


def render():
    page_header("Settings", "DB maintenance, runtime knobs, env inspection.")

    tab_db, tab_runtime, tab_env = st.tabs([
        "🗄️ Database", "⚙️ Runtime", "🔐 Environment",
    ])

    with tab_db:
        _render_db()
    with tab_runtime:
        _render_runtime()
    with tab_env:
        _render_env()


# ─── DB maintenance ────────────────────────────────────────────────────────
def _render_db():
    st.subheader("Index repair")
    st.markdown(
        "Drops and recreates the user-defined indexes on the trials, "
        "experiment_configs, and dataset_images tables. Use this when a "
        "delete fails with `Failed to delete all rows from index`."
    )
    if st.button("🔧 Rebuild indexes", type="secondary"):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            con.execute("CHECKPOINT")
            for stmt in [
                "DROP INDEX IF EXISTS idx_trials_status",
                "DROP INDEX IF EXISTS idx_experiment_configs_run",
                "DROP INDEX IF EXISTS idx_dataset_images_active",
            ]:
                con.execute(stmt)
            con.execute("CHECKPOINT")
            for stmt in [
                "CREATE INDEX IF NOT EXISTS idx_trials_status "
                "ON trials(run_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_experiment_configs_run "
                "ON experiment_configs(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_dataset_images_active "
                "ON dataset_images(dataset_id, excluded)",
            ]:
                con.execute(stmt)
            con.execute("CHECKPOINT")
        except Exception as e:
            st.error(f"Failed: {e}")
        else:
            st.success("Indexes rebuilt.")

    st.markdown("---")
    st.subheader("Manual checkpoint")
    st.caption(
        "Forces DuckDB to flush WAL → main file. Useful before backups."
    )
    if st.button("💾 CHECKPOINT", type="secondary"):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            con.execute("CHECKPOINT")
        except Exception as e:
            st.error(f"Failed: {e}")
        else:
            st.success("Checkpoint complete.")


# ─── Runtime knobs ─────────────────────────────────────────────────────────
def _render_runtime():
    st.subheader("Global concurrency cap")
    st.markdown(
        "Sets `OASIS_MAX_CONCURRENCY` for **future runs only** (already-running "
        "subprocesses are unaffected). The runner caps every run's "
        "`max_concurrency` at this value. Leave at 0 to disable the cap."
    )
    current = os.getenv("OASIS_MAX_CONCURRENCY", "")
    try:
        current_int = int(current) if current else 0
    except ValueError:
        current_int = 0
    new_val = st.slider(
        "OASIS_MAX_CONCURRENCY",
        min_value=0, max_value=64, value=current_int,
        help="0 = no cap.",
    )
    if st.button("Apply", type="primary"):
        if new_val == 0:
            os.environ.pop("OASIS_MAX_CONCURRENCY", None)
            _patch_env_file("OASIS_MAX_CONCURRENCY", None)
            st.success("Cap removed.")
        else:
            os.environ["OASIS_MAX_CONCURRENCY"] = str(new_val)
            _patch_env_file("OASIS_MAX_CONCURRENCY", str(new_val))
            st.success(
                f"Set to {new_val}. New runs will be capped at this. "
                "(Persisted to .env for future processes.)"
            )


# ─── Environment inspection ────────────────────────────────────────────────
def _render_env():
    st.subheader("API credentials")
    rows = []
    for key in [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL", "LANGFUSE_HOST", "LANGFUSE_PROJECT_ID",
        "OLLAMA_HOST", "OASIS_MAX_CONCURRENCY",
    ]:
        val = os.getenv(key, "")
        if not val:
            display = "—"
        elif "KEY" in key or "SECRET" in key:
            display = f"…{val[-4:]}" if len(val) > 4 else "set"
        else:
            display = val
        rows.append({"variable": key, "value": display})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption(
        f".env file: `{ENV_FILE}` "
        + ("(present)" if ENV_FILE.exists() else "(missing)")
    )


# ─── Helpers ───────────────────────────────────────────────────────────────
def _patch_env_file(key: str, value: str | None) -> None:
    """Add/update/remove a single key=value in .env. Best-effort, idempotent."""
    if not ENV_FILE.exists():
        if value is None:
            return
        ENV_FILE.write_text(f"{key}={value}\n")
        return
    lines = ENV_FILE.read_text().splitlines()
    out = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            if value is not None:
                out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found and value is not None:
        out.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(out) + ("\n" if out else ""))
