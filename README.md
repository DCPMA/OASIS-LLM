# OASIS-LLM

Adapting [Kurdi, Lozano, & Banaji (2017)](OASIS/Kurdi_BRM_2017.full.txt)'s OASIS image-norming procedure to LLM agents. The question this pilot answers: **how do agents feel about images?** — i.e. how do contemporary vision LLMs rate the 900 OASIS images on valence and arousal?

See [PLAN.md](PLAN.md) for the methodology and roadmap.

## Quick start

```bash
uv sync
uv run oasis-llm dashboard
```

That installs dependencies and opens the dashboard against the data already in the repo.
