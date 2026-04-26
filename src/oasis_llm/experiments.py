"""Experiment entity: a multi-config rating campaign against ONE dataset.

An Experiment contains 1..N named configs. Each config is materialised as a
backing `runs` row (with run_id = ``{experiment_id}__{config_name}``) so the
existing trial pipeline does not need to change.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb
import yaml

from .config import RunConfig
from .enqueue import enqueue_trials, upsert_run


@dataclass
class ExperimentConfig:
    config_name: str
    config_json: dict
    run_id: str
    position: int


@dataclass
class Experiment:
    experiment_id: str
    name: str
    description: str | None
    dataset_id: str
    status: str
    created_at: datetime | None
    finished_at: datetime | None
    configs: list[ExperimentConfig]


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", s).strip("-") or uuid.uuid4().hex[:8]


def _materialise_config(
    exp_id: str, cfg_name: str, base: dict, dataset_id: str,
) -> RunConfig:
    """Apply the experiment's defaults onto a per-config dict and return a RunConfig."""
    payload = dict(base)
    payload["name"] = f"{exp_id}__{cfg_name}"
    payload["image_set"] = dataset_id
    return RunConfig(**payload)


def create(
    con: duckdb.DuckDBPyConnection,
    name: str,
    dataset_id: str,
    configs: list[dict],
    *,
    description: str | None = None,
) -> str:
    """Create a draft experiment with N configs.

    Each config dict is a full RunConfig payload missing ``name`` and ``image_set``;
    those are injected. Each config must have a unique ``config_name`` field that
    is stripped before validation.
    """
    if not configs:
        raise ValueError("experiment must have at least one config")
    # validate dataset exists
    row = con.execute(
        "SELECT 1 FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown dataset: {dataset_id}")
    exp_id = _slug(name)
    if con.execute("SELECT 1 FROM experiments WHERE experiment_id=?", [exp_id]).fetchone():
        exp_id = f"{exp_id}-{uuid.uuid4().hex[:6]}"
    seen_names = set()
    parsed: list[tuple[str, RunConfig, dict]] = []
    for i, c in enumerate(configs):
        c = dict(c)  # shallow copy
        cfg_name = c.pop("config_name", None) or c.get("name") or f"config{i}"
        cfg_name = _slug(cfg_name)
        if cfg_name in seen_names:
            raise ValueError(f"duplicate config_name: {cfg_name}")
        seen_names.add(cfg_name)
        rc = _materialise_config(exp_id, cfg_name, c, dataset_id)
        parsed.append((cfg_name, rc, c))
    # write experiment + configs + backing runs + trials
    con.execute(
        """
        INSERT INTO experiments (experiment_id, name, description, dataset_id, status)
        VALUES (?, ?, ?, ?, 'draft')
        """,
        [exp_id, name, description, dataset_id],
    )
    for pos, (cfg_name, rc, raw) in enumerate(parsed):
        upsert_run(con, rc)            # creates runs row
        enqueue_trials(con, rc)        # creates pending trial rows
        con.execute(
            """
            INSERT INTO experiment_configs
              (experiment_id, config_name, config_json, run_id, position)
            VALUES (?, ?, ?, ?, ?)
            """,
            [exp_id, cfg_name, json.dumps(raw), rc.name, pos],
        )
    return exp_id


def list_all(con: duckdb.DuckDBPyConnection) -> list[Experiment]:
    rows = con.execute(
        """
        SELECT e.experiment_id, e.name, e.description, e.dataset_id,
               e.status, e.created_at, e.finished_at
        FROM experiments e
        ORDER BY e.created_at DESC NULLS LAST, e.experiment_id
        """
    ).fetchall()
    return [_load_experiment(con, *r) for r in rows]


def get(con: duckdb.DuckDBPyConnection, experiment_id: str) -> Experiment | None:
    row = con.execute(
        """
        SELECT experiment_id, name, description, dataset_id,
               status, created_at, finished_at
        FROM experiments WHERE experiment_id=?
        """,
        [experiment_id],
    ).fetchone()
    if row is None:
        return None
    return _load_experiment(con, *row)


def _load_experiment(con, exp_id, name, desc, dataset_id, status, created_at, finished_at) -> Experiment:
    cfgs = con.execute(
        """
        SELECT config_name, config_json, run_id, position
        FROM experiment_configs WHERE experiment_id=?
        ORDER BY position, config_name
        """,
        [exp_id],
    ).fetchall()
    return Experiment(
        experiment_id=exp_id, name=name, description=desc, dataset_id=dataset_id,
        status=status, created_at=created_at, finished_at=finished_at,
        configs=[
            ExperimentConfig(
                config_name=c[0], config_json=json.loads(c[1]),
                run_id=c[2], position=int(c[3]),
            )
            for c in cfgs
        ],
    )


def progress(con: duckdb.DuckDBPyConnection, experiment_id: str) -> list[dict]:
    """Return per-config trial counts: done/pending/failed/total/cost."""
    rows = con.execute(
        """
        SELECT ec.config_name, ec.run_id,
               sum(CASE WHEN t.status='done' THEN 1 ELSE 0 END) AS done,
               sum(CASE WHEN t.status='pending' THEN 1 ELSE 0 END) AS pending,
               sum(CASE WHEN t.status='running' THEN 1 ELSE 0 END) AS running,
               sum(CASE WHEN t.status='failed' THEN 1 ELSE 0 END) AS failed,
               count(t.image_id) AS total,
               round(sum(t.cost_usd), 4) AS cost,
               round(avg(t.latency_ms)) AS avg_latency_ms
        FROM experiment_configs ec
        LEFT JOIN trials t ON t.run_id = ec.run_id
        WHERE ec.experiment_id=?
        GROUP BY ec.config_name, ec.run_id, ec.position
        ORDER BY ec.position
        """,
        [experiment_id],
    ).fetchall()
    return [
        {
            "config_name": r[0], "run_id": r[1],
            "done": int(r[2] or 0), "pending": int(r[3] or 0),
            "running": int(r[4] or 0), "failed": int(r[5] or 0),
            "total": int(r[6] or 0),
            "cost_usd": float(r[7] or 0.0),
            "avg_latency_ms": int(r[8]) if r[8] is not None else None,
        }
        for r in rows
    ]


def update_status(con, experiment_id: str, status: str) -> None:
    if status == "done":
        con.execute(
            "UPDATE experiments SET status='done', finished_at=CURRENT_TIMESTAMP "
            "WHERE experiment_id=?",
            [experiment_id],
        )
    else:
        con.execute(
            "UPDATE experiments SET status=? WHERE experiment_id=?",
            [status, experiment_id],
        )


def archive(con, experiment_id: str) -> None:
    con.execute(
        "UPDATE experiments SET status='archived' WHERE experiment_id=?",
        [experiment_id],
    )


def delete(con, experiment_id: str) -> None:
    cfgs = con.execute(
        "SELECT run_id FROM experiment_configs WHERE experiment_id=?", [experiment_id]
    ).fetchall()
    for (run_id,) in cfgs:
        con.execute("DELETE FROM trials WHERE run_id=?", [run_id])
        con.execute("DELETE FROM runs WHERE run_id=?", [run_id])
    con.execute("DELETE FROM experiment_configs WHERE experiment_id=?", [experiment_id])
    con.execute("DELETE FROM experiments WHERE experiment_id=?", [experiment_id])


def from_yaml(path) -> tuple[str, str, list[dict], str | None]:
    """Parse an experiment YAML; returns (name, dataset_id, configs, description)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    name = data["name"]
    dataset_id = data.get("dataset") or data.get("dataset_id")
    if not dataset_id:
        raise ValueError("experiment YAML must specify `dataset:` (a dataset_id)")
    configs = data.get("configs") or []
    if not configs:
        raise ValueError("experiment YAML must have a non-empty `configs:` list")
    return name, dataset_id, configs, data.get("description")


def get_by_run_id(con, run_id: str) -> Experiment | None:
    """Resolve which experiment a run_id belongs to (if any)."""
    row = con.execute(
        "SELECT experiment_id FROM experiment_configs WHERE run_id=?", [run_id]
    ).fetchone()
    if row is None:
        return None
    return get(con, row[0])
