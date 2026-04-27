# OASIS-LLM

A research harness that adapts the **Open Affective Standardized Image Set (OASIS)** image-norming procedure of [Kurdi, Lozano, & Banaji (2017)](OASIS/Kurdi_BRM_2017.full.txt) to vision-language models. The question it answers: **how do contemporary LLMs feel about images?** — i.e. how do their valence and arousal ratings on the 900 OASIS images compare to the human MTurk norms (n=822)?

> **Status**: Active research project. Methodology is documented and stable; results pages are added as bundles are published.
>
> **Live**: [Documentation](https://dcpma.mintlify.app) · [Published results](results/)

## Current results

The first published pilot result is in [`results/llm_vs_human_uniform40/`](results/llm_vs_human_uniform40/) — 5 frontier vision-language models, 40 OASIS images, 20 samples per (image, model) pair, 7,960 trials total. Headline: LLMs track human valence ratings tightly (Pearson *r* ≈ 0.95) but systematically over-rate arousal (+0.36 on a 1–7 scale).

Further runs land in [`results/`](results/) as separate dated directories. Once the dedicated result renderer ships in the docs site, the canonical surface will move there; the Markdown copies in this folder remain as raw-evidence mirrors.

## What's in here

- An async LLM rating pipeline that supports OpenRouter, OpenAI, Anthropic, Google, and local Ollama via [LiteLLM](https://docs.litellm.ai/).
- An 8-page Streamlit dashboard for designing experiments, browsing the OASIS image set, monitoring runs, and analysing results against human norms.
- A bundle import/export system for sharing experiment results reproducibly.
- A pre-launch cost calculator calibrated against n=10,598 historical trials.
- Research-report-style documentation of every non-trivial discovery this harness has run into (the "Discoveries" section of the docs).

See [site/docs/](site/docs/) for the documentation source (rendered live at [dcpma.mintlify.app](https://dcpma.mintlify.app)).

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/DCPMA/OASIS-LLM.git
cd OASIS-LLM
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

### Image set & human norms

The 900 OASIS images and the human-norms CSV are licensed under **CC BY-NC-SA 4.0** by the original authors and are _not_ redistributed from this repository. Before running anything you need to populate two paths:

- **Images** — download from [osf.io/6pnd7](https://osf.io/6pnd7) and unpack into `OASIS/images/*.jpg`.
- **Human norms** — download `OASIS_data.csv` and `OASIS_codebook.txt` from the same OSF page and place them in `data/raw/`.

Both paths are gitignored. The dashboard and CLI will surface clear errors if either is missing.

## Repository layout

```
src/oasis_llm/        # Python package (CLI, runner, dashboard, analyses)
  dashboard_pages/    # 8 Streamlit pages: home, datasets, explorer, …
configs/runs/         # YAML run configurations
data/
  raw/                # OASIS_data.csv + codebook  (user-populated; gitignored)
  derived/            # OASIS_data_long.csv         (built locally; gitignored)
  public/             # Committed .zip bundles for the public results viewer
OASIS/                # Reference images + paper PDFs (user-populated; gitignored)
scripts/              # Analysis & maintenance scripts
site/docs/            # Mintlify documentation source
tests/                # Pytest suite
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
