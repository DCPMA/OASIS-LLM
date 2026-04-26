"""Queue page: scheduler controls + queued/running runs management."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import streamlit as st

from oasis_llm import queue as q
from oasis_llm.dashboard_pages._ui import (
    connect_rw, db_locked_warning, kpi, page_header, status_pill,
)


def _alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_scheduler() -> int:
    """Spawn the scheduler daemon as a detached subprocess. Returns PID."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "scheduler.log"
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "oasis_llm.cli", "scheduler"],
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def _stop_scheduler(pid: int) -> bool:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def render():
    page_header(
        "Queue",
        "Stack runs to execute sequentially. The scheduler daemon dequeues "
        "items up to `max_parallel` at a time.",
        icon="⏭️",
    )
    con = connect_rw()
    if con is None:
        db_locked_warning(); return

    # ── Scheduler status row ────────────────────────────────────────────
    sch_pid = q.scheduler_pid(con)
    hb = q.heartbeat_age_s(con)
    sch_alive = _alive(sch_pid) and (hb is not None and hb < 30)
    paused = q.is_paused(con)
    cap = q.max_parallel(con)
    queued = q.list_queued(con)

    n_running = con.execute(
        "SELECT COUNT(*) FROM runs WHERE status IN ('running','pending') "
        "AND run_id IN (SELECT run_id FROM run_processes)"
    ).fetchone()[0]

    cols = st.columns(4)
    cols[0].markdown(
        kpi(
            "Scheduler",
            "🟢 running" if sch_alive else ("🟡 paused" if paused and sch_alive else "🔴 stopped"),
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(kpi("Queued", str(len(queued))), unsafe_allow_html=True)
    cols[2].markdown(kpi("Running", str(n_running)), unsafe_allow_html=True)
    cols[3].markdown(kpi("Max parallel", str(cap)), unsafe_allow_html=True)

    st.markdown("---")

    # ── Controls ────────────────────────────────────────────────────────
    ctl = st.columns([1, 1, 1, 2])
    if not sch_alive:
        if ctl[0].button("▶️ Start scheduler", type="primary", width="stretch"):
            pid = _start_scheduler()
            st.success(f"Scheduler launched (pid={pid}). Refresh in a few seconds.")
            st.rerun()
    else:
        if ctl[0].button("⏹ Stop scheduler", width="stretch"):
            if sch_pid and _stop_scheduler(sch_pid):
                st.success("Scheduler signalled to stop.")
            else:
                st.warning("Could not stop scheduler (already gone?).")
            st.rerun()

    new_paused = ctl[1].toggle(
        "⏸ Pause queue",
        value=paused,
        key="q_pause",
        help="When ON, the scheduler keeps running but does NOT dequeue new items. "
        "Use to stack items without auto-start.",
    )
    if new_paused != paused:
        q.set_paused(con, new_paused)
        st.rerun()

    new_cap = ctl[2].number_input(
        "Max parallel",
        min_value=1, max_value=8, value=int(cap),
        key="q_cap",
        help="Maximum concurrent runs the scheduler will keep in flight.",
    )
    if int(new_cap) != int(cap):
        q.set_max_parallel(con, int(new_cap))
        st.toast(f"max_parallel = {int(new_cap)}")
        st.rerun()

    if sch_alive and hb is not None:
        ctl[3].caption(f"💓 last heartbeat {hb:.1f}s ago · pid {sch_pid}")
    elif sch_pid and hb is not None:
        ctl[3].caption(
            f"⚠️ stale heartbeat ({hb:.0f}s ago) · pid {sch_pid} "
            "— may have died; restart."
        )

    st.markdown("---")

    # ── Queue list ──────────────────────────────────────────────────────
    st.subheader("Queued items")
    if not queued:
        st.info(
            "Queue is empty. Use the Experiments page → **Queue experiment** "
            "or the Runs page → **Queue this run** to add items."
        )
    else:
        for i, item in enumerate(queued):
            with st.container(border=True):
                cols = st.columns([4, 2, 1, 1, 1, 1])
                cols[0].markdown(
                    f"**`{item.run_id}`**  \n"
                    f"<span style='color:#8a8aa0; font-size:0.78rem;'>"
                    f"queued at {item.queued_at} · priority {item.priority}"
                    + (f" · exp `{item.experiment_id}`" if item.experiment_id else "")
                    + "</span>",
                    unsafe_allow_html=True,
                )
                cols[1].markdown(status_pill("queued"), unsafe_allow_html=True)
                if cols[2].button("⬆", key=f"qup_{i}", help="Raise priority"):
                    q.bump(con, item.run_id, +1); st.rerun()
                if cols[3].button("⬇", key=f"qdn_{i}", help="Lower priority"):
                    q.bump(con, item.run_id, -1); st.rerun()
                if cols[4].button("✕", key=f"qrm_{i}", help="Remove from queue"):
                    q.cancel_queued(con, item.run_id); st.rerun()
                if cols[5].button("🚀 Now", key=f"qnow_{i}",
                                  help="Bypass scheduler — start this immediately."):
                    # Only safe if scheduler isn't picking it up at the same time.
                    from oasis_llm.scheduler import _spawn  # type: ignore
                    spawn_pid = _spawn(con, item.run_id)
                    st.success(f"Started {item.run_id} as pid {spawn_pid}.")
                    st.rerun()

    st.markdown("---")

    # ── Currently running ────────────────────────────────────────────────
    st.subheader("Currently running")
    rows = con.execute(
        """
        SELECT rp.run_id, rp.pid, rp.started_at, r.status,
               (SELECT COUNT(*) FROM trials t WHERE t.run_id=rp.run_id AND t.status='done') AS done,
               (SELECT COUNT(*) FROM trials t WHERE t.run_id=rp.run_id) AS total
        FROM run_processes rp
        LEFT JOIN runs r USING(run_id)
        ORDER BY rp.started_at DESC
        """
    ).fetchall()
    if not rows:
        st.info("Nothing running right now.")
    else:
        for run_id, pid, started_at, status, done, total in rows:
            alive = _alive(int(pid))
            cols = st.columns([4, 2, 2, 2])
            cols[0].markdown(f"**`{run_id}`**")
            cols[1].markdown(status_pill(status or "?"), unsafe_allow_html=True)
            pct = (done or 0) / (total or 1) * 100
            cols[2].markdown(
                f"<span style='color:#8a8aa0; font-size:0.85rem;'>"
                f"{done or 0}/{total or 0} ({pct:.0f}%)</span>",
                unsafe_allow_html=True,
            )
            cols[3].markdown(
                f"<span style='color:#8a8aa0; font-size:0.78rem;'>"
                f"pid {pid} {'🟢' if alive else '🔴 dead'}</span>",
                unsafe_allow_html=True,
            )

    # ── Tail of scheduler log ───────────────────────────────────────────
    st.markdown("---")
    log_path = Path("data/logs/scheduler.log")
    if log_path.exists():
        with st.expander("📜 Scheduler log (last 30 lines)"):
            try:
                lines = log_path.read_text().splitlines()[-30:]
                st.code("\n".join(lines), language="text")
            except Exception as e:
                st.caption(f"could not read log: {e}")
