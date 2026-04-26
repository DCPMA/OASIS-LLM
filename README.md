# OASIS-LLM

A research harness that adapts the **Open Affective Standardized Image Set (OASIS)** image-norming procedure of [Kurdi, Lozano, & Banaji (2017)](OASIS/Kurdi_BRM_2017.full.txt) to vision-language models. The question it answers: **how do contemporary LLMs feel about images?** — i.e. how do their valence and arousal ratings on the 900 OASIS images compare to the human MTurk norms (n=822)?

> **Status**: Active research project. Methodology is documented and stable; results pages are added as bundles are published.
>
> **Live**: [Documentation](https://github.com/DCPMA/AI-Psy) · [Published results](https://github.com/DCPMA/AI-Psy)
> _(URLs updated when hosting goes live — see [`IMPLEMENTATION_PUBLIC_RELEASE.md`](IMPLEMENTATION_PUBLIC_RELEASE.md).)_

## What's in here

- An async LLM rating pipeline that supports OpenRouter, OpenAI, Anthropic, Google, and local Ollama via [LiteLLM](https://docs.litellm.ai/).
- An 8-page Streamlit dashboard for designing experiments, browsing the OASIS image set, monitoring runs, and analysing results against human norms.
- A bundle import/export system for sharing experiment results reproducibly.
- A pre-launch cost calculator calibrated against n=10,598 historical trials.
- Research-report-style documentation of every non-trivial discovery this harness has run into (the "Discoveries" section of the docs).

See [PLAN.md](PLAN.md) for the research roadmap and [site/docs/](site/docs/) for the full documentation site (rendered at `<project>.mintlify.app` once hosting is live).

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/DCPMA/AI-Psy.git
cd AI-Psy
cp .env.example .env          # then fill in at least OPENROUTER_API_KEY
uv sync                       # installs all deps from uv.lock

# Launch the desktop dashboard:
uv run oasis-llm dashboard
```

The dashboard opens at `http://localhost:8501`. From there you can generate a dataset, define an experiment, and launch a 100-trial pilot in a few minutes.

### Run a pilot from the CLI

```bash
uv run oasis-llm run configs/runs/pilot30-qwen35-local.yaml
uv run oasis-llm status
uv run oasis-llm export <run_id> outputs/<run_id>.csv
```

### Image set

The 900 OASIS images themselves are licensed under **CC BY-NC-SA 4.0** by the original authors and are _not_ redistributable from this repo. Download them from [osf.io/6pnd7](https://osf.io/6pnd7) and unpack into `OASIS/images/` (gitignored).

## Repository layout

```
src/oasis_llm/        # Python package (CLI, runner, dashboard, analyses)
  dashboard_pages/    # 8 Streamlit pages: home, datasets, explorer, …
configs/runs/         # YAML run configurations
data/
  raw/                # OASIS_data.csv + codebook (the human norms)
  derived/            # OASIS_data_long.csv (denormalised)
  public/             # Committed .zip bundles for the public results viewer
scripts/              # Analysis & maintenance scripts
site/docs/            # Mintlify documentation source
tests/                # Pytest suite (currently focused on cost estimates)
streamlit_app.py      # Streamlit Cloud entrypoint (full desktop dashboard)
```

## Documentation

The docs site is published from [site/docs/](site/docs/). Local preview:

```bash
npm i -g mint                 # one-time, requires Node ≥ 20.17
cd site/docs
mint dev                      # http://localhost:3000
```

## Citation

If you use this work in academic research, please cite both the harness and the underlying OASIS paper. See [`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this repository" button in the sidebar.

## License

The code in this repository is released under the [MIT license](LICENSE). The OASIS image set retains its original [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license and is **not** covered by the MIT terms.

## Acknowledgements

This project replicates and extends Kurdi, Lozano, & Banaji (2017). All credit for the original image set, the rating procedure, and the human norms belongs to the original authors.
