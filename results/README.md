# Results

Published runs of the OASIS-LLM harness. Each subdirectory is a self-contained result with narrative, figures, and CSVs of all numerical outputs. Browse here to read a result; clone the repo to inspect the underlying data files locally.

## Index

| Run                                               | Cohort                                             | Headline                                                                                                                          |       Date |
| ------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llm_vs_human_uniform40](llm_vs_human_uniform40/) | 5 frontier VLMs × 40 images × 20 samples (n=7,960) | LLMs match humans on valence (_r_ ≈ 0.95); over-rate arousal by ~+0.36 on a 1–7 scale (_p_ = 0.064 paired, _p_ = 0.009 raw-trial) | 2026-04-26 |

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

Each result lives as a directory under `results/`. Treat them as immutable post-publication: a re-run goes in a new directory with a new name (date-stamped or cohort-stamped).
