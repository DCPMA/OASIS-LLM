# Results

Published runs of the OASIS-LLM harness. Each subdirectory is a self-contained result with narrative, figures, and CSVs of all numerical outputs. Browse here to read a result; clone the repo to inspect the underlying data files locally.

> **This is a temporary surface.** Once the dedicated result-page renderer ships in the [docs site](https://dcpma.mintlify.app), individual results will be linked from a `/results` group there. The Markdown copies here will continue to exist as a raw-evidence mirror.

## Index

| Run | Cohort | Headline | Date |
| --- | --- | --- | ---: |
| [llm_vs_human_uniform40](llm_vs_human_uniform40/) | 5 frontier VLMs × 40 images × 20 samples (n=7,960) | LLMs match humans on valence (*r* ≈ 0.95); over-rate arousal by ~+0.36 on a 1–7 scale (*p* = 0.064 paired, *p* = 0.009 raw-trial) | 2026-04-26 |

## Reading guide

Each result page follows the same structure:

1. **TL;DR** — three sentences a busy reader can take away
2. **Cohort** — what was tested
3. **Headline figures** — a small number of plots that carry the inference
4. **Numerical results** — t-test tables, descriptives
5. **Methodology** — how it was produced (linked to platform docs for full protocol)
6. **Choices made** — the design decisions specific to this run
7. **Restrictions and caveats** — where the conclusions stop
8. **Files** — pointer to every CSV and PNG in the directory
9. **Reproducing** — exact command(s) to regenerate
10. **Citation** — what to cite if used

## Adding a new result

For now, each result lives as a directory under `results/`. Treat them as immutable post-publication: a re-run goes in a new directory with a new name (date-stamped or cohort-stamped).

When the Mintlify-side renderer ships, this directory's contents will be lifted into MDX pages with proper component embedding; the directory layout here is intentionally portable to make that transformation mechanical.
