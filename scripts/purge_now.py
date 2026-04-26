"""Index repair + purge. One-shot. Safe to delete after running."""
from dotenv import load_dotenv
load_dotenv()

from oasis_llm.db import connect
from oasis_llm.run_admin import purge_runs

TO_DELETE = [
    "20260425-ee2b__config1",
    "20260425-2191__config1",
    "20260425-180a__config1",
    "my-experiment__qwen3-5",
]

con = connect()
print("Before:")
print("  runs:", con.execute("SELECT count(*) FROM runs").fetchone()[0])
print("  trials:", con.execute("SELECT count(*) FROM trials").fetchone()[0])

# DuckDB sometimes corrupts user-created indexes ("Failed to delete all rows
# from index"). Workaround: drop, delete, recreate.
con.execute("CHECKPOINT")
con.execute("DROP INDEX IF EXISTS idx_trials_status")
con.execute("DROP INDEX IF EXISTS idx_experiment_configs_run")
con.execute("DROP INDEX IF EXISTS idx_dataset_images_active")
con.execute("CHECKPOINT")
print("dropped user indexes; running purge...")

summaries = purge_runs(
    con,
    TO_DELETE,
    delete_langfuse=True,
    drop_empty_experiments=True,
)
for s in summaries:
    print(s)

con.execute(
    "CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(run_id, status)"
)
con.execute(
    "CREATE INDEX IF NOT EXISTS idx_experiment_configs_run "
    "ON experiment_configs(run_id)"
)
con.execute(
    "CREATE INDEX IF NOT EXISTS idx_dataset_images_active "
    "ON dataset_images(dataset_id, excluded)"
)
con.execute("CHECKPOINT")
print("recreated indexes")

print("After:")
print("  runs:", con.execute("SELECT count(*) FROM runs").fetchone()[0])
print("  experiments:", con.execute("SELECT count(*) FROM experiments").fetchone()[0])
print("  trials:", con.execute("SELECT count(*) FROM trials").fetchone()[0])
