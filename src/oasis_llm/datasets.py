"""Dataset entity: researcher-curated image sets.

A Dataset is a named, reviewed list of image_ids. Workflow:
  generate (draft) -> review (toggle excluded, add notes) -> approve (locked) -> use in runs.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb

from .images import _stratified_sample, _uniform_sample, all_image_ids, image_categories


# Built-in datasets that are auto-seeded on first connect. They mirror the
# legacy hardcoded image_set names so existing run YAMLs keep working.
BUILTINS: dict[str, dict] = {
    "pilot_30":  {"strategy": "stratified", "n": 30, "seed": 42},
    "pilot_10":  {"strategy": "stratified", "n": 10, "seed": 42},
    "smoke_3":   {"strategy": "stratified", "n": 3,  "seed": 42},
    "full_900":  {"strategy": "all",        "n": None, "seed": None},
}


@dataclass
class Dataset:
    dataset_id: str
    name: str
    description: str | None
    status: str
    source: str
    generation_params: dict | None
    created_at: datetime | None
    approved_at: datetime | None
    image_count: int
    active_count: int  # excluded=false


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", s).strip("-") or uuid.uuid4().hex[:8]


def seed_builtins(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotently insert built-in datasets as approved + status=builtin."""
    for ds_id, params in BUILTINS.items():
        existing = con.execute(
            "SELECT 1 FROM datasets WHERE dataset_id = ?", [ds_id]
        ).fetchone()
        if existing is not None:
            continue
        if params["strategy"] == "all":
            ids = all_image_ids()
        else:
            ids = _stratified_sample(params["n"], seed=params["seed"])
        con.execute(
            """
            INSERT INTO datasets
              (dataset_id, name, description, status, source, generation_params, approved_at)
            VALUES (?, ?, ?, 'approved', 'builtin', ?, CURRENT_TIMESTAMP)
            """,
            [ds_id, ds_id, f"Built-in {ds_id}", json.dumps(params)],
        )
        con.executemany(
            "INSERT INTO dataset_images (dataset_id, image_id) VALUES (?, ?)",
            [(ds_id, iid) for iid in ids],
        )


def generate(
    con: duckdb.DuckDBPyConnection,
    name: str,
    n: int,
    *,
    strategy: str = "stratified",
    seed: int = 42,
    description: str | None = None,
) -> str:
    """Generate a draft dataset by sampling from the full pool. Returns dataset_id.

    Strategies:
      - ``stratified``: proportional allocation across categories (default).
      - ``uniform``: equal count per category. ``n`` must be a multiple of the
        number of categories (currently 4 → must be a multiple of 4).
      - ``random``: uniform random across all images, ignoring categories.
      - ``all``: every image in the pool.
    """
    if strategy not in {"stratified", "uniform", "random", "all"}:
        raise ValueError(f"unknown strategy: {strategy}")
    dataset_id = _slug(name)
    if con.execute("SELECT 1 FROM datasets WHERE dataset_id=?", [dataset_id]).fetchone():
        # disambiguate
        dataset_id = f"{dataset_id}-{uuid.uuid4().hex[:6]}"
    if strategy == "all":
        ids = all_image_ids()
    elif strategy == "stratified":
        ids = _stratified_sample(n, seed=seed)
    elif strategy == "uniform":
        ids = _uniform_sample(n, seed=seed)
    else:  # random
        import random
        rng = random.Random(seed)
        pool = all_image_ids()
        ids = sorted(rng.sample(pool, min(n, len(pool))))
    params = {"strategy": strategy, "n": n, "seed": seed}
    con.execute(
        """
        INSERT INTO datasets (dataset_id, name, description, status, source, generation_params)
        VALUES (?, ?, ?, 'draft', 'generated', ?)
        """,
        [dataset_id, name, description, json.dumps(params)],
    )
    con.executemany(
        "INSERT INTO dataset_images (dataset_id, image_id) VALUES (?, ?)",
        [(dataset_id, iid) for iid in ids],
    )
    return dataset_id


def list_all(con: duckdb.DuckDBPyConnection) -> list[Dataset]:
    rows = con.execute(
        """
        SELECT d.dataset_id, d.name, d.description, d.status, d.source,
               d.generation_params, d.created_at, d.approved_at,
               count(di.image_id) AS total,
               sum(CASE WHEN NOT di.excluded THEN 1 ELSE 0 END) AS active
        FROM datasets d LEFT JOIN dataset_images di USING (dataset_id)
        GROUP BY d.dataset_id, d.name, d.description, d.status, d.source,
                 d.generation_params, d.created_at, d.approved_at
        ORDER BY d.created_at DESC NULLS LAST, d.dataset_id
        """
    ).fetchall()
    return [
        Dataset(
            dataset_id=r[0], name=r[1], description=r[2], status=r[3], source=r[4],
            generation_params=json.loads(r[5]) if r[5] else None,
            created_at=r[6], approved_at=r[7],
            image_count=int(r[8] or 0), active_count=int(r[9] or 0),
        )
        for r in rows
    ]


def get(con: duckdb.DuckDBPyConnection, dataset_id: str) -> Dataset | None:
    for d in list_all(con):
        if d.dataset_id == dataset_id:
            return d
    return None


def images(con: duckdb.DuckDBPyConnection, dataset_id: str) -> list[dict]:
    """Return all rows in the dataset (including excluded)."""
    rows = con.execute(
        """
        SELECT image_id, excluded, note, added_at
        FROM dataset_images WHERE dataset_id=? ORDER BY image_id
        """,
        [dataset_id],
    ).fetchall()
    return [
        {"image_id": r[0], "excluded": bool(r[1]), "note": r[2], "added_at": r[3]}
        for r in rows
    ]


def active_image_ids(con: duckdb.DuckDBPyConnection, dataset_id: str) -> list[str]:
    """Return image_ids with excluded=false. This is what runs consume."""
    rows = con.execute(
        """
        SELECT image_id FROM dataset_images
        WHERE dataset_id=? AND NOT excluded
        ORDER BY image_id
        """,
        [dataset_id],
    ).fetchall()
    return [r[0] for r in rows]


def _ensure_mutable(con: duckdb.DuckDBPyConnection, dataset_id: str) -> str:
    row = con.execute(
        "SELECT status, source FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()
    if row is None:
        raise KeyError(f"dataset not found: {dataset_id}")
    status, source = row
    if status == "approved":
        raise PermissionError(
            f"dataset {dataset_id} is approved; archive or duplicate it before editing"
        )
    if source == "builtin":
        raise PermissionError(f"dataset {dataset_id} is built-in and immutable")
    return status


def set_excluded(
    con: duckdb.DuckDBPyConnection, dataset_id: str, image_id: str,
    excluded: bool, note: str | None = None,
) -> None:
    _ensure_mutable(con, dataset_id)
    if note is None:
        con.execute(
            "UPDATE dataset_images SET excluded=? WHERE dataset_id=? AND image_id=?",
            [excluded, dataset_id, image_id],
        )
    else:
        con.execute(
            "UPDATE dataset_images SET excluded=?, note=? WHERE dataset_id=? AND image_id=?",
            [excluded, note, dataset_id, image_id],
        )


def add_image(con: duckdb.DuckDBPyConnection, dataset_id: str, image_id: str) -> None:
    _ensure_mutable(con, dataset_id)
    if image_id not in set(all_image_ids()):
        raise ValueError(f"unknown image_id: {image_id}")
    con.execute(
        """
        INSERT INTO dataset_images (dataset_id, image_id, excluded)
        VALUES (?, ?, FALSE)
        ON CONFLICT (dataset_id, image_id) DO NOTHING
        """,
        [dataset_id, image_id],
    )


def remove_image(con: duckdb.DuckDBPyConnection, dataset_id: str, image_id: str) -> None:
    _ensure_mutable(con, dataset_id)
    con.execute(
        "DELETE FROM dataset_images WHERE dataset_id=? AND image_id=?",
        [dataset_id, image_id],
    )


def regenerate_image(
    con: duckdb.DuckDBPyConnection,
    dataset_id: str,
    image_id: str,
    *,
    same_category: bool = True,
    seed: int | None = None,
) -> str:
    """Replace ``image_id`` in the dataset with a fresh draw from the OASIS pool.

    By default the replacement is sampled from the same category (so uniform
    stratification is preserved). The original image_id is excluded from the
    candidate pool, as are any images already present in the dataset.

    Returns the new image_id. Raises ValueError if no candidates remain.
    """
    import random as _r

    _ensure_mutable(con, dataset_id)
    cats = image_categories()
    src_cat = cats.get(image_id)
    pool_all = set(all_image_ids())
    in_set = {
        r[0] for r in con.execute(
            "SELECT image_id FROM dataset_images WHERE dataset_id=?", [dataset_id]
        ).fetchall()
    }
    if same_category and src_cat:
        candidates = [
            i for i in pool_all
            if cats.get(i) == src_cat and i not in in_set and i != image_id
        ]
    else:
        candidates = [i for i in pool_all if i not in in_set and i != image_id]
    if not candidates:
        raise ValueError(
            f"no replacement image available for {image_id} "
            f"({'same category' if same_category else 'any category'})"
        )
    rng = _r.Random(seed) if seed is not None else _r.Random()
    new_id = rng.choice(sorted(candidates))
    # Atomic swap: delete old, insert new.
    con.execute(
        "DELETE FROM dataset_images WHERE dataset_id=? AND image_id=?",
        [dataset_id, image_id],
    )
    con.execute(
        "INSERT INTO dataset_images (dataset_id, image_id, excluded) VALUES (?, ?, FALSE)",
        [dataset_id, new_id],
    )
    return new_id


def approve(con: duckdb.DuckDBPyConnection, dataset_id: str) -> None:
    row = con.execute(
        "SELECT status FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()
    if row is None:
        raise KeyError(f"dataset not found: {dataset_id}")
    n_active = con.execute(
        "SELECT count(*) FROM dataset_images WHERE dataset_id=? AND NOT excluded",
        [dataset_id],
    ).fetchone()[0]
    if n_active == 0:
        raise ValueError(f"dataset {dataset_id} has 0 active images; cannot approve")
    con.execute(
        "UPDATE datasets SET status='approved', approved_at=CURRENT_TIMESTAMP WHERE dataset_id=?",
        [dataset_id],
    )


def archive(con: duckdb.DuckDBPyConnection, dataset_id: str) -> None:
    row = con.execute("SELECT source FROM datasets WHERE dataset_id=?", [dataset_id]).fetchone()
    if row is None:
        raise KeyError(f"dataset not found: {dataset_id}")
    if row[0] == "builtin":
        raise PermissionError("cannot archive built-in datasets")
    con.execute("UPDATE datasets SET status='archived' WHERE dataset_id=?", [dataset_id])


def duplicate(
    con: duckdb.DuckDBPyConnection, dataset_id: str, new_name: str,
) -> str:
    """Clone a dataset (active images only) into a new draft. Useful for editing approved sets."""
    src = get(con, dataset_id)
    if src is None:
        raise KeyError(dataset_id)
    new_id = _slug(new_name)
    if con.execute("SELECT 1 FROM datasets WHERE dataset_id=?", [new_id]).fetchone():
        new_id = f"{new_id}-{uuid.uuid4().hex[:6]}"
    params = {"derived_from": dataset_id}
    con.execute(
        """
        INSERT INTO datasets (dataset_id, name, description, status, source, generation_params)
        VALUES (?, ?, ?, 'draft', 'generated', ?)
        """,
        [new_id, new_name, f"Cloned from {dataset_id}", json.dumps(params)],
    )
    active = active_image_ids(con, dataset_id)
    con.executemany(
        "INSERT INTO dataset_images (dataset_id, image_id) VALUES (?, ?)",
        [(new_id, iid) for iid in active],
    )
    return new_id


def delete(con: duckdb.DuckDBPyConnection, dataset_id: str) -> None:
    row = con.execute("SELECT source FROM datasets WHERE dataset_id=?", [dataset_id]).fetchone()
    if row is None:
        raise KeyError(dataset_id)
    if row[0] == "builtin":
        raise PermissionError("cannot delete built-in datasets")
    con.execute("DELETE FROM dataset_images WHERE dataset_id=?", [dataset_id])
    con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
