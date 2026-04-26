import duckdb, os
p = "data/llm_runs.duckdb"
print("size:", os.path.getsize(p))
print("wal size:", os.path.getsize(p + ".wal") if os.path.exists(p + ".wal") else None)
try:
    con = duckdb.connect(p)
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall()
    print("tables:", tables)
    for (t,) in tables:
        try:
            n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} rows")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
except Exception as e:
    print("OPEN FAILED:", type(e).__name__, e)
