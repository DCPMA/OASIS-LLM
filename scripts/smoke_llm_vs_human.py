"""Smoke test for the new analytics module + Explorer helpers.

Runs each new function against a *copy* of the live DuckDB so it can run
even while the dashboard holds the writer lock. Builds a temporary
Analysis binding the runs in image_set 20260426-uniform40, runs every
analytic, then deletes the temporary analysis from the copy.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from oasis_llm import analyses as an
from oasis_llm.db import DB_PATH
from oasis_llm.dashboard_pages import llm_vs_human_explorer as exp


def main() -> int:
    src = Path(DB_PATH)
    if not src.exists():
        print(f"DB {src} not found", file=sys.stderr)
        return 1
    tmpdir = Path(tempfile.mkdtemp(prefix="oasis_smoke_"))
    dst = tmpdir / "llm_runs.duckdb"
    shutil.copy2(src, dst)
    # WAL might also need copying; copy if present
    wal_src = src.with_suffix(src.suffix + ".wal")
    if wal_src.exists():
        try:
            shutil.copy2(wal_src, dst.with_suffix(dst.suffix + ".wal"))
        except Exception:
            pass
    print(f"Working against DB copy at {dst}")

    con = duckdb.connect(str(dst))

    # Find runs in 20260426-uniform40 and create a smoke analysis
    rows = con.execute(
        """
        SELECT run_id, json_extract_string(config_json, '$.image_set') AS image_set
        FROM runs WHERE status='done'
        """
    ).fetchall()
    target = [r for r in rows if r[1] == "20260426-uniform40"]
    if not target:
        print("No completed runs in image_set 20260426-uniform40 — aborting.", file=sys.stderr)
        return 1

    aid = an.create(con, "smoke-llm-vs-human", dataset_id="20260426-uniform40",
                    description="auto-generated smoke test")
    try:
        for rid, _ in target:
            an.add_run(con, aid, rid)

        print(f"\n=== smoke test for analysis '{aid}' (n={len(target)} runs) ===\n")

        sections: list[tuple[str, pd.DataFrame]] = []

        sections.append(("paired_ttest_per_run",
                         an.paired_ttest_per_run(con, aid, n_boot=0)))
        sections.append(("paired_ttest_per_run (n_boot=200)",
                         an.paired_ttest_per_run(con, aid, n_boot=200)))
        sections.append(("pooled_ttest", an.pooled_ttest(con, aid, n_boot=200)))
        sections.append(("regress_llm_on_human (per run)",
                         an.regress_llm_on_human(con, aid)))
        sections.append(("regress_llm_on_human (pooled)",
                         an.regress_llm_on_human(con, aid, pooled=True)))
        sections.append(("category_breakdown", an.category_breakdown(con, aid)))
        sections.append(("distribution_compare", an.distribution_compare(con, aid)))
        sections.append(("outlier_images (pooled)",
                         an.outlier_images(con, aid, top_k=10, scope="pooled")))
        sections.append(("outlier_images (per_run)",
                         an.outlier_images(con, aid, top_k=10, scope="per_run")))
        sections.append(("inter_llm_agreement", an.inter_llm_agreement(con, aid)))
        sections.append(("category_model_anova", an.category_model_anova(con, aid)))

        # Updated leaderboard (now with CCC)
        sections.append(("leaderboard (with CCC)", an.leaderboard(con).head(15)))

        for name, df in sections:
            print(f"--- {name} ---")
            if df is None or df.empty:
                print("(empty)")
            else:
                with pd.option_context("display.width", 200, "display.max_columns", 20):
                    print(df.head(10).to_string(index=False))
            print()

        # Explorer-level helpers (in-page t-test rows for each scope)
        per_img_run = an.per_image_aggregate(con, aid)
        norms = an._load_norms_with_category()
        per_img_run = per_img_run.merge(norms, on="image_id", how="left")
        per_img_run["human_value"] = per_img_run.apply(
            lambda r: r["human_valence"] if r["dimension"] == "valence" else r["human_arousal"],
            axis=1,
        )
        # add model column
        run_to_model = {}
        for rid, cfg_json in con.execute("SELECT run_id, config_json FROM runs").fetchall():
            try:
                cfg = json.loads(cfg_json) if cfg_json else {}
            except Exception:
                cfg = {}
            run_to_model[rid] = cfg.get("model")
        per_img_run["model"] = per_img_run["run_id"].map(run_to_model)
        per_img_run = per_img_run.rename(columns={"mean_rating": "llm_mean"})

        for scope in ("Pooled all-LLMs", "By model", "By category", "Model × Category"):
            rows = exp._ttest_rows(per_img_run, scope=scope, n_boot=0)
            print(f"--- explorer t-tests / scope={scope} ---")
            df = pd.DataFrame(rows)
            if df.empty:
                print("(empty)")
            else:
                with pd.option_context("display.width", 200, "display.max_columns", 20):
                    print(df.head(10).round(3).to_string(index=False))
            print()

        return 0
    finally:
        an.delete(con, aid)
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
