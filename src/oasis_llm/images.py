"""Image loading + selection."""
from __future__ import annotations

import base64
import random
from functools import lru_cache
from pathlib import Path

import duckdb

IMAGES_DIR = Path("OASIS/images")
LONG_CSV = Path("data/derived/OASIS_data_long.csv")


@lru_cache(maxsize=1)
def all_image_ids() -> list[str]:
    """Image IDs are filename stems."""
    return sorted(p.stem for p in IMAGES_DIR.glob("*.jpg"))


def image_path(image_id: str) -> Path:
    p = IMAGES_DIR / f"{image_id}.jpg"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def encode_image_base64(image_id: str) -> str:
    return base64.b64encode(image_path(image_id).read_bytes()).decode()


def image_data_url(image_id: str) -> str:
    return f"data:image/jpeg;base64,{encode_image_base64(image_id)}"


@lru_cache(maxsize=1)
def image_categories() -> dict[str, str]:
    """Return {image_id: category} from the long CSV."""
    if not LONG_CSV.exists():
        return {}
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"SELECT DISTINCT theme, category FROM read_csv_auto('{LONG_CSV}')"
    ).fetchdf()
    return dict(zip(df["theme"].astype(str), df["category"].astype(str)))


def select_image_set(
    name: str,
    seed: int = 42,
    con: "duckdb.DuckDBPyConnection | None" = None,
) -> list[str]:
    """Return the list of image_ids for a named image set.

    Resolution order:
      1. Explicit ``dataset:<id>`` prefix → DB lookup (active images only).
      2. Bare name matching a dataset_id in the datasets table → DB lookup.
      3. Legacy hardcoded names (full_900, pilot_30, pilot_10, smoke_3) →
         deterministic stratified sampling. (These are also auto-seeded as
         built-in datasets, so step 2 normally handles them.)
      4. Path to a newline-delimited file of image_ids.
    """
    # Step 1+2: try database. Reuse the caller's connection if provided so we
    # don't deadlock against an existing read-write handle in the same process.
    explicit = name.startswith("dataset:")
    lookup_id = name.split(":", 1)[1] if explicit else name
    owns_con = False
    db_con = con
    try:
        from .db import DB_PATH
        if db_con is None and Path(DB_PATH).exists():
            db_con = duckdb.connect(str(DB_PATH), read_only=True)
            owns_con = True
        if db_con is not None:
            row = db_con.execute(
                "SELECT 1 FROM datasets WHERE dataset_id=?", [lookup_id]
            ).fetchone()
            if row is not None:
                ids = [
                    r[0]
                    for r in db_con.execute(
                        "SELECT image_id FROM dataset_images "
                        "WHERE dataset_id=? AND NOT excluded ORDER BY image_id",
                        [lookup_id],
                    ).fetchall()
                ]
                if ids:
                    return ids
    except Exception:
        pass
    finally:
        if owns_con and db_con is not None:
            db_con.close()
    if explicit:
        raise ValueError(f"unknown dataset: {lookup_id}")
    # Step 3: legacy fallback
    if name == "full_900":
        return all_image_ids()
    if name == "pilot_30":
        return _stratified_sample(30, seed=seed)
    if name == "pilot_10":
        return _stratified_sample(10, seed=seed)
    if name == "smoke_3":
        return _stratified_sample(3, seed=seed)
    # Step 4: file path
    p = Path(name)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    raise ValueError(f"unknown image set: {name}")


def _stratified_sample(n: int, seed: int = 42) -> list[str]:
    cats = image_categories()
    if not cats:
        # Fall back to uniform random across all images
        rng = random.Random(seed)
        ids = all_image_ids()
        return sorted(rng.sample(ids, min(n, len(ids))))
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = {}
    for iid, cat in cats.items():
        by_cat.setdefault(cat, []).append(iid)
    # Proportional allocation, at least 1 per category, sum to n
    cats_sorted = sorted(by_cat)
    total = sum(len(v) for v in by_cat.values())
    alloc = {c: max(1, round(n * len(by_cat[c]) / total)) for c in cats_sorted}
    # Fix rounding drift
    while sum(alloc.values()) > n:
        biggest = max(alloc, key=alloc.get)
        alloc[biggest] -= 1
    while sum(alloc.values()) < n:
        smallest = min(alloc, key=alloc.get)
        alloc[smallest] += 1
    out: list[str] = []
    # Filter to image ids that actually have files on disk
    available = set(all_image_ids())
    for c in cats_sorted:
        # Sort pool deterministically before shuffling — DuckDB DISTINCT does not guarantee order
        pool = sorted([i for i in by_cat[c] if i in available])
        rng.shuffle(pool)
        out.extend(pool[: alloc[c]])
    return sorted(out)


def category_count() -> int:
    """Number of distinct OASIS categories. Used for uniform-strategy validation."""
    return len(set(image_categories().values()))


def _uniform_sample(n: int, seed: int = 42) -> list[str]:
    """Uniform stratification: equal count per category. Requires n %% n_cats == 0."""
    cats = image_categories()
    if not cats:
        raise ValueError("uniform sampling requires the long CSV with category labels")
    by_cat: dict[str, list[str]] = {}
    for iid, cat in cats.items():
        by_cat.setdefault(cat, []).append(iid)
    cats_sorted = sorted(by_cat)
    n_cats = len(cats_sorted)
    if n % n_cats != 0:
        raise ValueError(
            f"uniform strategy requires n to be a multiple of {n_cats} (number of categories); got {n}"
        )
    per_cat = n // n_cats
    rng = random.Random(seed)
    available = set(all_image_ids())
    out: list[str] = []
    for c in cats_sorted:
        pool = sorted([i for i in by_cat[c] if i in available])
        if len(pool) < per_cat:
            raise ValueError(
                f"category '{c}' only has {len(pool)} images; cannot sample {per_cat}"
            )
        rng.shuffle(pool)
        out.extend(pool[:per_cat])
    return sorted(out)
