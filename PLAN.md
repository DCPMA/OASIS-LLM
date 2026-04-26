# OASIS-LLM: Adapting OASIS to Run with LLM Agents

Pilot study adapting [Kurdi, Lozano, & Banaji (2017)](OASIS/Kurdi_BRM_2017.full.txt)'s OASIS norming procedure to LLM "agents". The question: **how do agents feel about images?**

## Research Question (v1)

How do contemporary LLM agents rate the OASIS image set on **valence** and **arousal**, and how do those ratings vary across models?

Out of scope for v1 (deferred to future work, noted in README):

- Demographic/persona conditioning (silicon-sample work)
- Comparison to human norms as a primary statistical claim (we'll plot it for sanity, not center the study on it)
- Cross-modality analysis (vision-direct vs caption-mediated) beyond a basic check

## Conceptual Design

We drop the "participant" abstraction from the original study. In Kurdi et al., one MTurk participant rated 225 images sequentially within a single session. For LLM agents we instead use **stateless single-trial calls**: each trial is one independent API request with `(instructions + 1 image + 1 question) → 1 rating`.

Rationale:

- Removes within-session order/adaptation effects we don't want to model in v1
- Trivially parallel and idempotent
- Works identically for tiny local models and frontier models, regardless of context window

A "run" is a configured batch: one model, one set of parameters, one image subset, one or both dimensions, N samples per image. Runs are independent, named, version-controlled, resumable.

## Procedure (per trial)

1. Load image bytes from [OASIS/images](OASIS/images).
2. Build prompt:
   - System: paper-verbatim image-focused instructions (Appendix 1 of Kurdi 2017) for the assigned dimension.
   - User: image (inline base64 for vision models; pre-generated caption for text-only models) + the rating question with the 7 verbal anchors.
3. Call model via litellm with structured-output schema: `{"rating": int 1-7, "reasoning": str (optional, capped)}`.
4. Persist trial row to DuckDB with rating, raw response, latency, tokens, cost.

Default temperature: keep at provider default (do **not** force 0). Repeat sampling per image is how we estimate variance, not a single deterministic point.

## Sampling Strategy (per-run, configurable)

Two knobs per run config:

- `image_set`: `pilot_30` (stratified across animals/objects/people/scenes) | `full_900` | named subset file
- `samples_per_image`: integer; supports progressive expansion (add more samples to a run later without re-running existing trials, thanks to idempotency)

Suggested defaults:

- **Local (Ollama)** runs: `full_900 × samples_per_image=3` — cheap, run overnight.
- **Remote (OpenRouter / direct API)** runs: start at `pilot_30 × samples=5`, inspect cost & response quality in dashboard, then optionally bump samples or expand to `full_900` incrementally.

## Models (initial config, user-provided IDs)

| Provider              | Model ID                                       | Modality | Notes                        |
| --------------------- | ---------------------------------------------- | -------- | ---------------------------- |
| ollama (local)        | `qwen3.5:9b`                                   | TBD      | check vision support on pull |
| ollama (local)        | `gemma4:e4b`                                   | TBD      | check vision support on pull |
| openrouter            | `google/gemma-4-31b-it`                        | TBD      |                              |
| openrouter            | `nvidia/nemotron-3-super-120b-a12b`            | TBD      | likely text-only → captions  |
| openrouter            | `google/gemini-3.1-pro-preview`                | vision   |                              |
| openrouter (optional) | `minimax/minimax-m2.7`                         | TBD      |                              |
| openrouter (optional) | `moonshotai/kimi-k2.6`                         | TBD      |                              |
| openrouter (optional) | `qwen/qwen3.6-plus`                            | TBD      |                              |
| anthropic (direct)    | `claude-sonnet-4.5` (or current GA equivalent) | vision   | added as a strong vision ref |

Each model's vision capability is declared in its run config. Text-only models route through the **caption-then-rate** pipeline (see below). One designated strong vision model generates captions once for all 900 images; captions are cached in `data/derived/captions.parquet` and reused.

## Tech Stack

- **Language / env:** Python 3.12, `uv` for dependency management.
- **LLM clients:** [`litellm`](https://github.com/BerriAI/litellm) (unified across openrouter / ollama / anthropic / google).
- **Structured output:** `pydantic` schema, enforced via litellm's `response_format=json_schema` (falls back to JSON-mode + parse for providers that don't support full schema).
- **Concurrency:** `asyncio` worker pool, `tenacity` retry with exponential backoff. Per-run `max_concurrency` configurable (low for Ollama, high for OpenRouter).
- **Storage:** DuckDB single-file database at `data/llm_runs.duckdb`. Schema designed so `trials` can be exported as a long CSV matching [data/derived/OASIS_data_long.csv](data/derived/OASIS_data_long.csv) for later R analysis.
- **Captions cache:** `data/derived/captions.parquet` keyed by `image_id`.
- **Dashboard / controller:** Streamlit at `src/dashboard.py`.
- **CLI:** `oasis-llm` entry point (Typer).
- **Cost / observability:** built-in cost log in DuckDB; optional Langfuse hook later.

## Data Model

```
runs
  run_id        TEXT PRIMARY KEY    -- slug, e.g. "pilot-sonnet-valence-2026-04-25"
  config_json   JSON                -- frozen YAML config
  config_hash   TEXT                -- sha256 of canonical config
  status        TEXT                -- created | running | paused | done | failed
  created_at    TIMESTAMP
  finished_at   TIMESTAMP NULL

trials
  run_id        TEXT
  image_id      TEXT                -- matches OASIS image filename stem
  dimension     TEXT                -- 'valence' | 'arousal'
  sample_idx    INTEGER             -- 0..samples_per_image-1
  status        TEXT                -- pending | running | done | failed
  rating        INTEGER NULL        -- 1..7
  raw_response  TEXT NULL
  reasoning     TEXT NULL
  prompt_hash   TEXT                -- so we can detect prompt drift
  latency_ms    INTEGER NULL
  input_tokens  INTEGER NULL
  output_tokens INTEGER NULL
  cost_usd      DOUBLE NULL
  error         TEXT NULL
  attempts      INTEGER DEFAULT 0
  claimed_at    TIMESTAMP NULL      -- for stale-claim recovery
  completed_at  TIMESTAMP NULL
  PRIMARY KEY (run_id, image_id, dimension, sample_idx)

captions
  image_id      TEXT
  captioner     TEXT                -- model id used
  caption       TEXT
  created_at    TIMESTAMP
  PRIMARY KEY (image_id, captioner)
```

### Idempotency & Resumability

- **Enqueue:** `INSERT ... ON CONFLICT DO NOTHING` for each `(run_id, image_id, dim, sample_idx)`. Re-running enqueue is a no-op.
- **Claim:** worker atomically sets `status='running', claimed_at=now()` only `WHERE status='pending'`.
- **Resume:** re-running `oasis-llm run <name>` picks up `pending` + `failed` trials.
- **Stale claims:** trials stuck in `running` past a configurable timeout get reset to `pending`.
- **Progressive expansion:** bumping `samples_per_image` from 5→10 in the YAML re-enqueues only the new `sample_idx` rows (5..9). Existing rows untouched.
- **Config drift guard:** if `config_hash` changes (model, prompt, params), CLI refuses unless `--new-run` flag is passed (forks to a new `run_id`).

## Project Layout (proposed)

```
pyproject.toml
README.md
PLAN.md                             # this file
configs/
  runs/
    pilot-sonnet-valence.yaml
    pilot-gemini-arousal.yaml
    local-gemma-full.yaml
  models.yaml                       # shared model definitions (provider, modality, defaults)
src/oasis_llm/
  __init__.py
  cli.py                            # `oasis-llm run|resume|status|export|caption|dashboard`
  config.py                         # pydantic models for run/model configs
  db.py                             # DuckDB connection + migrations
  enqueue.py                        # idempotent trial enqueue
  runner.py                         # async worker pool, claims, retries
  providers.py                      # litellm wiring per provider
  prompts.py                        # paper-verbatim instructions, schemas
  captions.py                       # vision-model captioning pipeline
  images.py                         # image loader, base64, validation
  export.py                         # DuckDB → OASIS_data_long-style CSV
  dashboard.py                      # Streamlit app
data/
  llm_runs.duckdb                   # gitignored
  derived/
    captions.parquet                # gitignored
scripts/
  smoke_test.py                     # 3-image end-to-end check
```

## Dashboard (Streamlit)

Single-page app. Sections:

1. **Runs table:** run_id, model, dimension, image_set, progress %, mean rating, total cost, status. Buttons: pause/resume.
2. **Live progress:** auto-refreshing trial counts per run.
3. **Sample inspector:** randomly sample N completed trials, show image + rating + reasoning side-by-side. Filter by run, dimension, rating value.
4. **Distributions:** rating histograms per run, latency histogram, cost over time.
5. **Cross-run comparison (later):** scatter plot of mean rating per image across two selected runs.
6. **Export:** one-click CSV/parquet download in `OASIS_data_long.csv` schema.

No auth. Local only.

## Phased Plan

| Phase                            | Goal                                                                                                        | Cost ceiling       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------ |
| **0. Scaffold**                  | Project skeleton, DuckDB schema, CLI, prompts, smoke test (3 images, 1 model)                               | <$1                |
| **1. Pilot remote**              | `pilot_30 × 5 samples` on 1 OpenRouter vision model, both dims                                              | <$5                |
| **2. Captions + text-only path** | Generate captions for all 900 images with one vision model; verify quality; enable text-only model runs     | ~$10 one-time      |
| **3. Dashboard**                 | Streamlit dashboard wired to DuckDB; sample inspector + progress                                            | —                  |
| **4. Multi-model pilot**         | All 3–4 primary models, `pilot_30 × 5`, both dims                                                           | <$30               |
| **5. Local full-set**            | Ollama runs at `full_900 × 3`, overnight                                                                    | $0 (local compute) |
| **6. Selective expansion**       | Promote best models to `full_900` on remote based on dashboard inspection                                   | user-decided       |
| **Later**                        | Persona/demographic conditioning · cross-model agreement analysis · comparison to human norms · paper draft | —                  |

## Decisions Locked

- ✅ Stateless single-trial calls (drop participant abstraction)
- ✅ No persona/demographics in v1 (README note for future)
- ✅ Configurable runs, idempotent, resumable, progressive sampling
- ✅ litellm + DuckDB + Streamlit + Typer CLI
- ✅ Caption-then-rate fallback for text-only models, captions cached
- ✅ Deliverables for v1: setup code, dataset (DuckDB + CSV export), dashboard. Analysis & paper later.

## Open Questions Before Scaffolding

1. **Caption model:** which vision model do we trust to caption all 900 images once? Suggest `gemini-3.1-pro-preview` (cheap, strong vision) or `claude-sonnet-4.5`.
2. **Reasoning capture:** ask models for a 1-sentence rationale alongside the rating? Useful for the dashboard sample inspector but adds tokens/cost. Suggest yes, capped at ~100 tokens.
3. **Anthropic Claude in lineup?** Not in your list but adds a strong vision reference point. Include or skip?
4. **Run naming convention:** suggest `<scope>-<model-slug>-<dim>-<date>`, e.g. `pilot-gemini31pro-both-2026-04-25`.
5. **License / privacy:** any concern about sending OASIS images (some are explicit/violent/distressing per the paper) to remote APIs? Most providers' ToS allow this but worth a flag.
