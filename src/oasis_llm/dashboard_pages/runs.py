"""Runs page: inspect, control, export individual runs."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from oasis_llm import run_control as rc
from oasis_llm.dashboard_pages._ui import (
    connect_rw, db_locked_warning, kpi, page_header, status_pill,
)
from oasis_llm.images import image_path


CONFIGS_DIRS = [Path("configs/runs"), Path(".")]


def _find_config_for_run(run_id: str) -> Path | None:
    """Best-effort lookup of the YAML used to launch a run.

    We search for a YAML whose ``name`` field matches ``run_id``.
    """
    import yaml
    for d in CONFIGS_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.yaml"):
            try:
                data = yaml.safe_load(p.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and data.get("name") == run_id:
                return p
    return None


def render():
    page_header(
        "Runs",
        "Each run = one config × one dataset. Inspect, control, export.",
        icon="📊",
    )
    con = connect_rw()
    if con is None:
        db_locked_warning()
        return

    runs = con.execute(
        """
        SELECT r.run_id, r.status, r.created_at, r.finished_at,
               count(t.image_id) AS total,
               sum(CASE WHEN t.status='done'    THEN 1 ELSE 0 END) AS done,
               sum(CASE WHEN t.status='failed'  THEN 1 ELSE 0 END) AS failed,
               sum(CASE WHEN t.status='pending' THEN 1 ELSE 0 END) AS pending,
               sum(CASE WHEN t.status='running' THEN 1 ELSE 0 END) AS running,
               COALESCE(sum(t.cost_usd), 0) AS cost,
               r.config_json
        FROM runs r LEFT JOIN trials t USING (run_id)
        GROUP BY r.run_id, r.status, r.created_at, r.finished_at, r.config_json
        ORDER BY r.created_at DESC NULLS LAST
        """
    ).fetchall()
    if not runs:
        st.info("No runs yet.")
        return

    # Per-run model id (parsed from config_json) for filtering
    def _model_of(cfg_json: str | None) -> str:
        if not cfg_json:
            return "?"
        try:
            return json.loads(cfg_json).get("model") or "?"
        except Exception:
            return "?"

    run_models = {r[0]: _model_of(r[10]) for r in runs}
    available_models = sorted({m for m in run_models.values() if m != "?"})

    fc1, fc2 = st.columns([1, 1])
    model_filter = fc1.multiselect(
        "Filter by model",
        available_models,
        default=[],
        help="Leave empty to show all runs.",
    )
    status_filter = fc2.multiselect(
        "Filter by status",
        ["running", "paused", "done", "failed", "pending"],
        default=[],
    )

    def _passes(r) -> bool:
        if model_filter and run_models.get(r[0]) not in model_filter:
            return False
        if status_filter and r[1] not in status_filter:
            return False
        return True

    visible_runs = [r for r in runs if _passes(r)]
    if not visible_runs:
        st.info("No runs match the selected filters.")
        return

    options = {
        f"{r[0]}  ·  {run_models.get(r[0], '?')}  ({r[1]}, {r[5] or 0}/{r[4] or 0})": r[0]
        for r in visible_runs
    }
    label = st.selectbox("Select run", list(options.keys()))
    run_id = options[label]

    row = next(r for r in runs if r[0] == run_id)
    _, status, created, finished, total, done, failed, pending, running, cost, _cfg_json = row
    total = total or 0
    done = done or 0
    failed = failed or 0
    pending = pending or 0
    running = running or 0

    # Auto-refresh while live
    if status in ("running", "paused") or rc.get_pid(con, run_id) is not None:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=2000, key=f"refresh_{run_id}")
        except Exception:
            st.caption("⏱ Run is live — click 🔄 *Rerun* to refresh.")

    # ── KPIs ────────────────────────────────────────────────────────────────
    cols = st.columns(4)
    cols[0].markdown(kpi("Status", status), unsafe_allow_html=True)
    cols[1].markdown(
        kpi("Trials", f"{done}/{total}", f"{pending} pending · {failed} failed · {running} running"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(kpi("Cost", f"${cost:.4f}"), unsafe_allow_html=True)
    cols[3].markdown(
        kpi("Created", str(created)[:19] if created else "—"),
        unsafe_allow_html=True,
    )

    # ── Progress bar ────────────────────────────────────────────────────────
    pct = (done / total) if total else 0.0
    st.progress(min(1.0, pct), text=f"{done}/{total} done · {pct*100:.1f}%")

    # ── Controls ────────────────────────────────────────────────────────────
    pid = rc.get_pid(con, run_id)
    config_path = _find_config_for_run(run_id)
    _render_controls(con, run_id, status, pid, config_path, pending, failed)

    # ── Config / metadata ───────────────────────────────────────────────────
    cfg_row = con.execute(
        "SELECT config_json FROM runs WHERE run_id=?", [run_id]
    ).fetchone()
    cfg_dict = json.loads(cfg_row[0]) if cfg_row and cfg_row[0] else {}
    with st.expander("🔧 Config"):
        st.json(cfg_dict)

    st.markdown("---")
    tab_lat, tab_samples, tab_fails, tab_export = st.tabs([
        "⏱️  Latency", "🖼️  Samples", "⚠️  Failures", "📤  Export",
    ])

    with tab_lat:
        _render_latency(con, run_id)

    with tab_samples:
        _render_samples(con, run_id)

    with tab_fails:
        _render_failures(con, run_id)

    with tab_export:
        _render_export(con, run_id, cfg_dict)


# ─── controls ──────────────────────────────────────────────────────────────
def _render_controls(con, run_id: str, status: str, pid, config_path, pending: int, failed: int):
    with st.container(border=True):
        st.markdown("### 🎛️ Controls")
        # If a previous Start attempt detected a foreign DB lock holder, show
        # the chooser instead of the regular controls.
        if st.session_state.get("runs_lock_conflict") is not None:
            from oasis_llm.dashboard_pages._ui import render_lock_conflict
            render_lock_conflict(
                "runs_lock_conflict", st.session_state["runs_lock_conflict"]
            )
            return
        if config_path is None:
            st.warning(
                "Couldn't auto-discover the YAML config for this run. "
                "Start/Resume from the UI is unavailable. The run can still be controlled via "
                f"`uv run oasis-llm run <yaml>` from the CLI."
            )
        cs = st.columns(6)
        # Start
        with cs[0]:
            can_start = (
                pid is None and status not in ("running",) and pending + failed > 0
                and config_path is not None
            )
            if st.button("▶️ Start", disabled=not can_start, width='stretch', key="rc_start"):
                # Pre-flight: ensure no other OS process holds the DuckDB lock.
                # The CLI subprocess we're about to spawn would otherwise crash
                # immediately. The dashboard itself also holds an open
                # connection, so the spawned subprocess will fail unless we
                # release it first.
                from oasis_llm.db import lock_holder_pid
                holder = lock_holder_pid()
                if holder is not None and holder != os.getpid():
                    st.session_state["runs_lock_conflict"] = holder
                    st.rerun()
                else:
                    try:
                        new_pid = rc.start(con, run_id, config_path)
                        st.success(f"Started PID {new_pid}.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        # Queue
        with cs[1]:
            can_queue = (
                pid is None and status not in ("running", "queued")
                and pending + failed > 0
            )
            if st.button("⏭️ Queue", disabled=not can_queue,
                         width='stretch', key="rc_queue",
                         help="Add this run to the scheduler queue."):
                from oasis_llm import queue as _q
                try:
                    _q.enqueue(con, run_id)
                    st.success("Queued. See the Queue page.")
                except Exception as e:
                    st.error(str(e))
                st.rerun()
        # Pause
        with cs[2]:
            if st.button("⏸️ Pause", disabled=(status != "running"),
                         width='stretch', key="rc_pause"):
                rc.pause(con, run_id)
                st.info("Paused (workers will drain).")
                st.rerun()
        # Resume
        with cs[3]:
            can_resume = status == "paused" and config_path is not None
            if st.button("⏯️ Resume", disabled=not can_resume,
                         width='stretch', key="rc_resume"):
                rc.resume(con, run_id, config_path)
                st.success("Resumed.")
                st.rerun()
        # Cancel
        with cs[4]:
            if st.button("⏹️ Cancel", disabled=(status not in ("running", "paused", "queued") and pid is None),
                         width='stretch', key="rc_cancel"):
                if status == "queued":
                    from oasis_llm import queue as _q
                    _q.cancel_queued(con, run_id)
                    st.warning("Removed from queue.")
                else:
                    rc.cancel(con, run_id)
                    st.warning("Cancelled.")
                st.rerun()
        # Reset failures
        with cs[5]:
            if st.button("🔁 Reset failed", disabled=(failed == 0),
                         width='stretch', key="rc_reset"):
                n = rc.reset_failed(con, run_id)
                st.info(f"Reset {n} failed trials.")
                st.rerun()
        if pid is not None:
            st.caption(f"🟢 Background process running · PID {pid}")

    # ── Danger zone ────────────────────────────────────────────────────────
    with st.expander("⚠️ Danger zone — delete this run"):
        st.markdown(
            "Permanently removes the run and **all its trials**. If this run "
            "belongs to an experiment, only this run (one config) is deleted; "
            "the experiment row is kept. Cannot be undone."
        )
        if status == "running" or pid is not None:
            st.caption("Cancel the run first before deleting.")
        else:
            also_lf = st.toggle(
                "Also delete matching Langfuse traces",
                value=True,
                key=f"del_lf_{run_id}",
                help="Best-effort; requires LANGFUSE_PUBLIC_KEY/SECRET_KEY.",
            )
            confirm = st.text_input(
                "Type the run id to confirm",
                key=f"del_confirm_{run_id}",
                placeholder=run_id,
            )
            if st.button(
                "🗑️ Delete run permanently",
                type="secondary",
                disabled=(confirm.strip() != run_id),
                key=f"del_btn_{run_id}",
            ):
                from oasis_llm.run_admin import delete_run as _del
                try:
                    # DuckDB index can corrupt; drop+recreate around the delete
                    # to avoid the "Failed to delete all rows from index" fatal.
                    con.execute("CHECKPOINT")
                    con.execute("DROP INDEX IF EXISTS idx_trials_status")
                    summary = _del(con, run_id, delete_langfuse=also_lf)
                    con.execute(
                        "CREATE INDEX IF NOT EXISTS idx_trials_status "
                        "ON trials(run_id, status)"
                    )
                    con.execute("CHECKPOINT")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
                else:
                    msg = (
                        f"Deleted run `{run_id}` "
                        f"({summary['trials']} trials"
                    )
                    if summary.get("langfuse_traces_deleted"):
                        msg += f", {summary['langfuse_traces_deleted']} Langfuse traces"
                    msg += ")."
                    st.success(msg)
                    if summary.get("langfuse_error"):
                        st.warning(f"Langfuse: {summary['langfuse_error']}")
                    st.session_state.pop("selected_run", None)
                    st.rerun()


def _delete_run(con, run_id: str) -> None:
    """Legacy shim — delegates to run_admin.delete_run (no Langfuse)."""
    from oasis_llm.run_admin import delete_run as _del
    _del(con, run_id, delete_langfuse=False)


# ─── latency ───────────────────────────────────────────────────────────────
def _render_latency(con, run_id: str):
    df = con.execute(
        """
        SELECT latency_ms FROM trials
        WHERE run_id=? AND status='done' AND latency_ms IS NOT NULL
        ORDER BY completed_at
        """,
        [run_id],
    ).fetchdf()
    if df.empty:
        st.caption("No latency data.")
        return
    cols = st.columns(4)
    cols[0].markdown(kpi("p50", f"{int(df['latency_ms'].median())} ms"), unsafe_allow_html=True)
    cols[1].markdown(kpi("p95", f"{int(df['latency_ms'].quantile(0.95))} ms"), unsafe_allow_html=True)
    cols[2].markdown(kpi("max", f"{int(df['latency_ms'].max())} ms"), unsafe_allow_html=True)
    cols[3].markdown(kpi("mean", f"{int(df['latency_ms'].mean())} ms"), unsafe_allow_html=True)
    st.line_chart(df["latency_ms"].reset_index(drop=True), height=200)


# ─── samples ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _norms_lookup() -> dict[str, dict]:
    """{image_id: {valence_mean, arousal_mean, category}} from OASIS.csv"""
    p = Path("OASIS/OASIS.csv")
    if not p.exists():
        return {}
    import duckdb
    df = duckdb.connect(":memory:").execute(
        f"""
        SELECT "Theme" AS image_id, "Category" AS category,
               "Valence_mean" AS valence_mean, "Arousal_mean" AS arousal_mean
        FROM read_csv_auto('{p}', header=true)
        """
    ).fetchdf()
    return {r["image_id"]: r.to_dict() for _, r in df.iterrows()}


def _langfuse_base_url() -> str | None:
    """Return the Langfuse base URL for trace deep-linking, or None."""
    base = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if not base:
        # Try to load from project .env without polluting global env
        try:
            from dotenv import dotenv_values
            from pathlib import Path as _P
            vals = dotenv_values(_P(".env"))
            base = vals.get("LANGFUSE_BASE_URL") or vals.get("LANGFUSE_HOST")
        except Exception:
            return None
    if not base:
        return None
    return base.rstrip("/")


def _langfuse_creds() -> tuple[str | None, str | None]:
    """Return (public_key, secret_key) from env or .env."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if pk and sk:
        return pk, sk
    try:
        from dotenv import dotenv_values
        from pathlib import Path as _P
        vals = dotenv_values(_P(".env"))
        return vals.get("LANGFUSE_PUBLIC_KEY"), vals.get("LANGFUSE_SECRET_KEY")
    except Exception:
        return None, None


@st.cache_data(ttl=3600)
def _langfuse_project_id() -> str | None:
    """Discover the Langfuse project id via the public projects endpoint.

    Cached for an hour. Returns None on failure.
    """
    base = _langfuse_base_url()
    if not base:
        return None
    pk, sk = _langfuse_creds()
    if not (pk and sk):
        # Allow override
        pid = os.getenv("LANGFUSE_PROJECT_ID")
        return pid or None
    pid_env = os.getenv("LANGFUSE_PROJECT_ID")
    if pid_env:
        return pid_env
    try:
        import urllib.request
        import base64
        import json as _json
        req = urllib.request.Request(f"{base}/api/public/projects")
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = _json.loads(r.read())
        projects = data.get("data") or []
        if projects:
            return projects[0]["id"]
    except Exception:
        return None
    return None


def _trace_url(trace_id: str | None) -> str | None:
    """Build a Langfuse trace deep-link, or None if unavailable."""
    if not trace_id:
        return None
    base = _langfuse_base_url()
    if not base:
        return None
    pid = _langfuse_project_id()
    if pid:
        return f"{base}/project/{pid}/traces/{trace_id}"
    # Fall back to the (sometimes-broken) shortcut route
    return f"{base}/trace/{trace_id}"


def _render_samples(con, run_id: str):
    norms = _norms_lookup()
    cs = st.columns([1, 2, 2])
    n_show = cs[0].slider("Images to show", 1, 30, 6, key=f"samp_n_{run_id}")
    img_filter = cs[1].text_input("Filter image_id contains", value="",
                                  key=f"samp_filter_{run_id}")
    dim_choice = cs[2].selectbox(
        "Dimension", ["all", "valence", "arousal", "dominance"],
        index=0, key=f"samp_dim_{run_id}",
    )

    where = "AND image_id ILIKE ?" if img_filter else ""
    dim_where = "" if dim_choice == "all" else "AND dimension=?"
    params = [run_id]
    if img_filter:
        params.append(f"%{img_filter}%")
    if dim_choice != "all":
        params.append(dim_choice)
    params.append(n_show)
    sample_imgs = con.execute(
        f"""
        SELECT DISTINCT image_id FROM trials
        WHERE run_id=? AND status='done' {where} {dim_where}
        ORDER BY image_id LIMIT ?
        """,
        params,
    ).fetchall()
    if not sample_imgs:
        st.caption("No completed trials match.")
        return
    for (img_id,) in sample_imgs:
        with st.container(border=True):
            ic1, ic2 = st.columns([1, 3])
            with ic1:
                p = image_path(img_id)
                if p.exists():
                    st.image(str(p), width='stretch')
                norm = norms.get(img_id)
                if norm:
                    st.markdown(
                        f"<div style='font-size:0.85rem;'>"
                        f"<b>{img_id}</b> · {norm.get('category', '?')}<br>"
                        f"<span style='color:#a78bfa;'>Human V {norm['valence_mean']:.2f} · "
                        f"A {norm['arousal_mean']:.2f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(img_id)
            with ic2:
                dim_clause = "" if dim_choice == "all" else "AND dimension=?"
                qparams = [run_id, img_id]
                if dim_choice != "all":
                    qparams.append(dim_choice)
                samples = con.execute(
                    f"""
                    SELECT dimension, sample_idx, rating, latency_ms,
                           trace_id
                    FROM trials WHERE run_id=? AND image_id=? AND status='done'
                          {dim_clause}
                    ORDER BY dimension, sample_idx
                    """,
                    qparams,
                ).fetchdf()
                # Per-dimension LLM means + delta vs human
                if not samples.empty:
                    by_dim = samples.groupby("dimension")["rating"].agg(["mean", "std", "count"])
                    info_cols = st.columns(len(by_dim))
                    for col, (dim, vals) in zip(info_cols, by_dim.iterrows()):
                        human_key = f"{dim}_mean"
                        human_val = (norms.get(img_id) or {}).get(human_key)
                        delta = ""
                        if human_val is not None:
                            delta = f"Δ {vals['mean'] - human_val:+.2f}"
                        col.markdown(
                            f"<div class='kpi-card' style='padding:0.6rem 0.85rem;'>"
                            f"<p class='label'>{dim} · n={int(vals['count'])}</p>"
                            f"<p class='value' style='font-size:1.25rem;'>{vals['mean']:.2f}"
                            f" <span style='font-size:0.75rem; color:#8a8aa0;'>±{vals['std']:.2f}</span></p>"
                            f"<p class='delta'>{delta}</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                # Render trace_id as a Langfuse link if any are present
                display = samples.copy()
                has_trace = (
                    "trace_id" in display.columns
                    and display["trace_id"].notna().any()
                )
                if has_trace:
                    display["trace_id"] = display["trace_id"].apply(_trace_url)
                    st.dataframe(
                        display, width='stretch', hide_index=True,
                        column_config={
                            "trace_id": st.column_config.LinkColumn(
                                "trace", display_text="open in langfuse →",
                            ),
                        },
                    )
                else:
                    # Drop the empty trace column entirely for cleaner display
                    if "trace_id" in display.columns:
                        display = display.drop(columns=["trace_id"])
                    st.dataframe(display, width='stretch', hide_index=True)


# ─── failures ──────────────────────────────────────────────────────────────
def _render_failures(con, run_id: str):
    fails = con.execute(
        """
        SELECT image_id, dimension, sample_idx, attempts, error
        FROM trials WHERE run_id=? AND status='failed'
        ORDER BY image_id, dimension, sample_idx
        """,
        [run_id],
    ).fetchdf()
    if fails.empty:
        st.success("No failures.")
    else:
        st.dataframe(fails, width='stretch', hide_index=True)


# ─── export ────────────────────────────────────────────────────────────────
def _render_export(con, run_id: str, cfg_dict: dict):
    st.markdown(
        "Export trials in **attempt-wide** format: each row is "
        "`(run_id, provider, model_id, attempt_id, sample_idx)` followed by per-image "
        "`<image>_valence` / `<image>_arousal` columns."
    )
    n_per_attempt = st.number_input("Images per attempt row", min_value=1, value=20, step=1)
    if st.button("Generate attempt_wide.csv"):
        from oasis_llm.analysis import export_participant_style_dataset
        out_dir = Path("outputs/participant_dataset") / run_id
        try:
            summary = export_participant_style_dataset(
                run_id=run_id, out_dir=out_dir,
                images_per_participant=int(n_per_attempt),
            )
        except Exception as e:
            st.error(f"Export failed: {e}")
            return
        st.session_state[f"exported_{run_id}"] = (out_dir, summary)
        st.success(f"Wrote to `{out_dir}` · {summary['attempt_count']} attempts × {summary['response_columns']} response cols")

    exported = st.session_state.get(f"exported_{run_id}")
    if exported:
        out_dir, summary = exported
        wide = out_dir / "attempt_wide.csv"
        if wide.exists():
            data = wide.read_bytes()
            st.download_button(
                "⬇️ Download attempt_wide.csv",
                data=data, file_name=f"{run_id.replace('/', '_')}_attempt_wide.csv",
                mime="text/csv", width='stretch',
            )

    st.markdown("---")
    st.markdown("#### Long-format CSV (one row per trial)")
    if st.button("Build long CSV"):
        df = con.execute(
            """
            SELECT run_id, image_id, dimension, sample_idx, status, rating,
                   latency_ms, input_tokens, output_tokens, cost_usd,
                   error, completed_at
            FROM trials WHERE run_id=?
            ORDER BY image_id, dimension, sample_idx
            """,
            [run_id],
        ).fetchdf()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download trials_long.csv",
            data=buf.getvalue().encode(),
            file_name=f"{run_id.replace('/', '_')}_trials_long.csv",
            mime="text/csv", width='stretch',
        )
