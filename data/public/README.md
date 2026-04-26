# `data/public/` — bundles for the public results viewer

This directory holds `.zip` bundles produced by the desktop CLI:

```bash
oasis-llm experiment export <experiment_id> -o data/public/<name>.zip
oasis-llm analysis export   <analysis_id>   -o data/public/<name>.zip
```

When the public results viewer (Phase 4) boots and finds no
`data/llm_runs.duckdb`, it calls
`oasis_llm.public_bootstrap.bootstrap_from_bundles()` to populate a fresh
DuckDB from every `.zip` in this directory.

## Conventions

- One bundle per logical unit (one experiment, or one analysis).
- Filename is the bundle's intended public slug:
  `<topic>-<scope>-<date>.zip` works well, e.g. `llm-vs-human-uniform40-2025-04.zip`.
- Each bundle should be referenced in the catalog below with its source
  experiment/analysis id, generation date, n trials, and intended use.
- The `OASIS/images/` set itself is **not** in any bundle (the destination
  workspace must already have the same `image_id` set on disk; the bundle
  ships only the labels).

## Catalog

_Empty until Phase 2.5 — the user picks which experiments/analyses are
public and runs the export commands above. Each entry should follow the
template:_

```
### <bundle-filename>.zip

- **Source:** experiment / analysis id `<id>`
- **Generated:** YYYY-MM-DD
- **Models:** model1, model2, …
- **Image set:** N OASIS images × M samples
- **Total trials:** ~X
- **Intended use:** "Paper figure 3", "Methodology illustration", etc.
- **Notes:** Anything important about scope, exclusions, known issues.
```
