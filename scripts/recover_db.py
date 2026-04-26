"""One-shot recovery: quarantine the corrupt DB+WAL, create a fresh DB,
re-import the bundle in import/, verify, and CHECKPOINT.

Usage: uv run python scripts/recover_db.py [bundle.zip]
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "llm_runs.duckdb"
WAL = DATA / "llm_runs.duckdb.wal"
RECOVERY = DATA / "recovery"
IMPORT_DIR = ROOT / "import"


def main() -> int:
    if len(sys.argv) >= 2:
        bundle = Path(sys.argv[1])
    else:
        zips = sorted(IMPORT_DIR.glob("*.zip"))
        if not zips:
            print(f"No .zip bundles found in {IMPORT_DIR}", file=sys.stderr)
            return 2
        # newest first
        bundle = max(zips, key=lambda p: p.stat().st_mtime)

    if not bundle.exists():
        print(f"Bundle not found: {bundle}", file=sys.stderr)
        return 2

    print(f"Bundle: {bundle} ({bundle.stat().st_size:,} B)")

    # 1. Quarantine
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qdir = DATA / "quarantine" / ts
    qdir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in (DB, WAL):
        if p.exists():
            dst = qdir / p.name
            shutil.move(str(p), str(dst))
            moved.append(dst)
    if RECOVERY.exists():
        dst = qdir / "recovery"
        shutil.move(str(RECOVERY), str(dst))
        moved.append(dst)
    print(f"Quarantined {len(moved)} item(s) → {qdir}")
    for m in moved:
        print(f"  · {m.relative_to(ROOT)}")

    # 2. Fresh DB with schema (lazy import so quarantine works even if
    #    the package import would otherwise touch the db).
    from oasis_llm.db import connect
    from oasis_llm.bundles import import_any

    con = connect(DB)
    print("Fresh DB initialised. Tables:")
    for (t,) in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY 1"
    ).fetchall():
        print(f"  · {t}")

    # 3. Re-import (transaction-wrapped)
    print(f"\nImporting {bundle.name}…")
    blob = bundle.read_bytes()
    con.execute("BEGIN")
    try:
        summary = import_any(con, blob, overwrite=False)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("CHECKPOINT")

    print("\nImport summary:")
    import json
    print(json.dumps(summary, indent=2, default=str))

    # 4. Sanity counts
    print("\nRow counts after import:")
    for t in ("datasets", "dataset_images", "experiments",
              "experiment_configs", "runs", "trials", "analyses",
              "analysis_runs"):
        try:
            n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"  {t:20s} {n:>10,}")
        except Exception as e:
            print(f"  {t:20s} ERROR {e}")

    con.close()
    print(f"\nDone. New DB: {DB} ({DB.stat().st_size:,} B)")
    print(f"Quarantined originals preserved at: {qdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
