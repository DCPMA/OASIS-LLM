"""Streamlit Community Cloud entrypoint.

Streamlit Cloud expects a top-level script. This shim simply imports and
runs the same dashboard module that ``oasis-llm dashboard`` launches.

Set the following secrets in the Streamlit Cloud app settings:

- ``OASIS_LLM_READONLY = "1"`` (recommended for the public deployment)
- DuckDB file: place ``data/llm_runs.duckdb`` in the repo, or have
  ``streamlit_app.py`` download it from object storage at startup.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

# Default to read-only mode for public deployment
os.environ.setdefault("OASIS_LLM_READONLY", "1")

# Ensure src/ is importable when running outside an installed package
_repo_root = Path(__file__).parent
_src = _repo_root / "src"
if _src.exists():
    import sys
    sys.path.insert(0, str(_src))

# Run the dashboard module as if invoked by streamlit
runpy.run_module("oasis_llm.dashboard", run_name="__main__")
