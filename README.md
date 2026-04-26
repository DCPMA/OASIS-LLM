# OASIS-LLM

Adapting [Kurdi, Lozano, & Banaji (2017)](OASIS/Kurdi_BRM_2017.full.txt)'s OASIS image-norming procedure to LLM agents. The question this pilot answers: **how do agents feel about images?** — i.e. how do contemporary vision LLMs rate the 900 OASIS images on valence and arousal?

See [PLAN.md](PLAN.md) for the methodology and roadmap.

## Quick start

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." >> .env

# Validate a vision model end-to-end (3 stratified images, valence only)
uv run oasis-llm smoke configs/runs/smoke-openrouter-gemma4.yaml

# Run a full pilot config (resumable, idempotent)
uv run oasis-llm run configs/runs/pilot-gemma4-31b.yaml

# Inspect status
uv run oasis-llm status
uv run oasis-llm status pilot-gemma4-31b

# Open the dashboard
uv run oasis-llm dashboard

# Export to CSV
uv run oasis-llm export pilot-gemma4-31b

# Write paper-style pilot plots for a run
uv run oasis-llm paper-plots pilot30-qwen35-local
```

## Validated vision models

| Provider       | Model ID                          | Status | Notes                               |
| -------------- | --------------------------------- | ------ | ----------------------------------- |
| openrouter     | `google/gemma-4-31b-it`           | ✅     | Free; ~2s/call                      |
| openrouter     | `google/gemini-3.1-pro-preview`   | ✅     | $2/$12 + $20/img; ~5s/call          |
| openrouter     | `qwen/qwen3.6-plus`               | ✅     | $0.32/$1.95; ~10s/call              |
| ollama (local) | `qwen3-vl:8b`                     | ✅     | Free local; ~10s/call; needs retry  |
| ~~openrouter~~ | ~~`minimax/minimax-m2.7`~~        | ❌     | No vision endpoint despite metadata |
| ~~openrouter~~ | ~~`nvidia/nemotron-3-super-...`~~ | ❌     | Text-only                           |

`moonshotai/kimi-k2.6` and `anthropic/claude-sonnet-4.5` are not yet smoke-tested — add a config and run `oasis-llm smoke <config>` to validate before promoting.

## Architecture

- **Stateless single-trial calls.** Each trial = `(instructions + 1 image + 1 question) → 1 rating`. No within-session context.
- **Idempotent + resumable.** `trials` table keyed on `(run_id, image_id, dimension, sample_idx)`. Re-running `oasis-llm run` picks up only `pending`/`failed` trials. Failed trials cap at 3 attempts.
- **Progressive sampling.** Bump `samples_per_image` in YAML to add more samples; existing rows untouched.
- **Provider-agnostic** via [litellm](https://github.com/BerriAI/litellm).
- **Storage:** DuckDB at `data/llm_runs.duckdb`.
- **Dashboard:** Streamlit (`oasis-llm dashboard`).
- **Paper-style plots:** `oasis-llm paper-plots <run_id>` writes the original-study-style figures for the completed images in that run.

## Project layout

```
configs/runs/*.yaml     # one file per run
src/oasis_llm/
  cli.py                # `oasis-llm run|smoke|status|export|dashboard`
  config.py             # pydantic RunConfig
  db.py                 # DuckDB schema
  enqueue.py            # idempotent trial insertion
  runner.py             # asyncio worker pool, retry, claim
  prompts.py            # paper-verbatim Kurdi 2017 instructions
  providers.py          # provider routing (openrouter, ollama, …)
  images.py             # image loading, stratified sampling
  dashboard.py          # Streamlit app
data/llm_runs.duckdb    # gitignored
```

## Out of scope (v1)

- Demographic / persona conditioning ("silicon-sample" investigation) — future work.
- Statistical comparison to human OASIS norms — exported CSV is in long format compatible with [scripts/OASIS.R](scripts/OASIS.R) for later analysis.
- Caption-then-rate fallback for text-only models — designed in but not yet implemented; v1 is vision-models-only.
