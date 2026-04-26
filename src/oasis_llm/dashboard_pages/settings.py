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

    st.markdown("---")
    st.subheader("Rebuild trials primary key")
    st.markdown(
        "Use when a write fails with `INTERNAL Error: Failed to append to "
        "PRIMARY_trials_*: PRIMARY KEY violation` — usually after `kill -9` "
        "of a runner mid-write. Rebuilds the table via CTAS to clean the "
        "PK index. Row count is preserved."
    )
    if st.button("🛠️ Rebuild trials PK", type="secondary"):
        con = connect_rw()
        if con is None:
            db_locked_warning(); return
        try:
            n_before = con.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            con.execute("BEGIN")
            con.execute("CREATE TABLE trials_new AS SELECT * FROM trials")
            con.execute("DROP TABLE trials")
            con.execute(
                """
                CREATE TABLE trials (
                  run_id VARCHAR NOT NULL,
                  image_id VARCHAR NOT NULL,
                  dimension VARCHAR NOT NULL,
                  sample_idx INTEGER NOT NULL,
                  status VARCHAR NOT NULL DEFAULT 'pending',
                  rating INTEGER,
                  raw_response VARCHAR,
                  reasoning VARCHAR,
                  prompt_hash VARCHAR,
                  latency_ms INTEGER,
                  input_tokens INTEGER,
                  output_tokens INTEGER,
                  cost_usd DOUBLE,
                  error VARCHAR,
                  attempts INTEGER DEFAULT 0,
                  claimed_at TIMESTAMP,
                  completed_at TIMESTAMP,
                  finish_reason VARCHAR,
                  response_id VARCHAR,
                  trace_id VARCHAR,
                  PRIMARY KEY(run_id, image_id, dimension, sample_idx)
                )
                """
            )
            con.execute("INSERT INTO trials SELECT * FROM trials_new")
            con.execute("DROP TABLE trials_new")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_trials_status "
                "ON trials(run_id, status)"
            )
            con.execute("COMMIT")
            con.execute("CHECKPOINT")
            n_after = con.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        except Exception as e:
            try: con.execute("ROLLBACK")
            except Exception: pass
            st.error(f"Failed: {e}")
        else:
            st.success(f"Rebuilt. Rows: {n_before} → {n_after}")


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

    st.markdown("---")
    st.subheader("Default request timeout (s)")
    st.markdown(
        "Default `request_timeout_s` for **new** experiment configs. "
        "Existing configs keep their saved value. Bump to 120-300 if you "
        "regularly run medium/large local Ollama models — under concurrency "
        "they queue inside Ollama and trip the default 60s ceiling."
    )
    cur_to = os.getenv("OASIS_DEFAULT_TIMEOUT_S", "")
    try:
        cur_to_int = int(cur_to) if cur_to else 60
    except ValueError:
        cur_to_int = 60
    new_to = st.slider(
        "OASIS_DEFAULT_TIMEOUT_S",
        min_value=10, max_value=600, step=10, value=cur_to_int,
        help="Per-call timeout. Failed calls eat the FULL value × (max_retries+1).",
    )
    if st.button("Apply timeout", type="secondary"):
        os.environ["OASIS_DEFAULT_TIMEOUT_S"] = str(new_to)
        _patch_env_file("OASIS_DEFAULT_TIMEOUT_S", str(new_to))
        st.success(f"Default timeout set to {new_to}s.")

    st.markdown("---")
    st.subheader("Ollama parallelism (advice)")
    st.markdown(
        "By default Ollama serves **one** request at a time per model "
        "instance. Concurrent requests beyond that queue inside the Ollama "
        "server, which is why a config's `max_concurrency=4` can still trip "
        "60s timeouts even though each individual call only takes 6-7s.\n\n"
        "Two env vars on the **Ollama server** (not this process) control "
        "real parallelism:\n"
        "- `OLLAMA_NUM_PARALLEL` — concurrent requests served per loaded model "
        "(default 1; try 2-4 on a Mac with enough RAM).\n"
        "- `OLLAMA_MAX_LOADED_MODELS` — how many distinct models stay in VRAM "
        "(default 1).\n\n"
        "Restart `ollama serve` after setting these. "
        "Example macOS launch: "
        "`OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 ollama serve`"
    )
    # Quick check: what is the running Ollama server doing?
    try:
        import requests as _rq
        _rq.get("http://localhost:11434/api/version", timeout=1).raise_for_status()
        st.caption("✅ Ollama is reachable on localhost:11434. The env vars above "
                   "must be set in the *server* environment, not here.")
    except Exception:
        st.caption("⚠️ Ollama is not reachable on localhost:11434.")


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
