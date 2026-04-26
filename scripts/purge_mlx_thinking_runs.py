"""One-off purge of the 2 cancelled MLX runs that failed due to the Ollama
thinking-mode bug. Safe to run again; will no-op if rows already gone."""
from __future__ import annotations

import sys

import duckdb

from oasis_llm.run_admin import purge_runs

DB = "data/llm_runs.duckdb"
RUNS = [
    "20260426-488c__gemma4-e4b-mlx-bf16",
    "20260426-cba2__qwen3-5-4b-mlx-bf16",
]


def main() -> int:
    con = duckdb.connect(DB)
    # Drop user indexes first (DuckDB delete-all-rows-from-index workaround)
    for idx in ("idx_trials_status", "idx_experiment_configs_run"):
        try:
            con.execute(f"DROP INDEX IF EXISTS {idx}")
        except Exception as e:
            print(f"  warn: drop {idx}: {e}")
    con.execute("CHECKPOINT")

    results = purge_runs(con, RUNS, delete_langfuse=True, drop_empty_experiments=True)
    for r in results:
        print(r)

    # Recreate indexes
    con.execute("CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiment_configs_run "
        "ON experiment_configs(run_id)"
    )
    con.execute("CHECKPOINT")
    con.close()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
