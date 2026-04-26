# Public Release — Implementation Plan

> **Scope**: Take OASIS-LLM from a private local project to a public open-source repo with hosted documentation and a hosted **results viewer** for published experiments. The full desktop dashboard remains a local-install workflow.
>
> **Status**: Draft, awaiting approval of Phase 4 option (β vs γ).
>
> **Owner**: Desmond
>
> **Related docs**: [PLAN.md](PLAN.md) (research roadmap), [site/docs/](site/docs/) (the docs site itself)

---

## TL;DR

Five phases. Phases 1–3 + 5 are unchanged from the original deployment plan (repo hardening, demo bundle infrastructure, Mintlify hosting, docs alignment). **Phase 4 is replaced** by a slim "results viewer" deployment so we publish curated experiment results without exposing the full editor / runner / Settings surface.

The public artefact is a **separate Streamlit entrypoint** (`streamlit_results.py`) — _or_ a set of MDX pages auto-generated from the same bundles into the Mintlify docs site. Both options share Phase 2's bundle infrastructure. The choice is a UX-vs-infra tradeoff documented in §Phase 4.

---

## Architectural principles

1. **Separation of public from local.** The desktop dashboard (`streamlit_app.py`, full 8 pages) stays as-is. The public surface is a new, narrower entrypoint that imports the same modules but registers fewer pages. There is **no flag-on-flag-off mode switching** in the desktop app.
2. **Bundles are the publication boundary.** Anything published goes through `oasis-llm experiment-export` / `oasis-llm analysis-export` to produce a `.zip` in `data/public/`. If a result is not in a bundle, it is not public.
3. **Single source of truth for shared UI.** The Home / Runs / Analysis / Image-Explorer pages are imported by both entrypoints. No copy-paste forks.
4. **Reversibility everywhere except the public-flip.** Every step except "GitHub repo → public" can be undone by reverting commits or deleting cloud apps.
5. **No new runtime dependencies.** Everything in this plan uses libraries already in `pyproject.toml`.

---

## Phase 1 — Repo hardening

**Goal**: Make the repo safe to flip public — no leaked secrets, proper licensing, citation metadata, contributor-friendly README.

### Tasks

| #   | File / action         | Detail                                                                                                                                                    |
| --- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | `LICENSE` (new)       | MIT. `Copyright (c) 2025 Desmond Wong`.                                                                                                                   |
| 1.2 | `CITATION.cff` (new)  | YAML; references DOI from [4qspg-datacite.json](4qspg-datacite.json) and the OASIS paper as `preferred-citation`.                                         |
| 1.3 | `README.md` (rewrite) | Expand 12 → ~80 lines: motivation, install (`uv sync`), 100-trial pilot, links to docs + live results viewer (URLs filled in Phase 5), citation, license. |
| 1.4 | `.gitignore` (edit)   | Add `.streamlit/secrets.toml`, `data/snapshot.duckdb`, `**/.DS_Store`. Whitelist `!data/public/*.zip`, `!data/public/README.md`.                          |
| 1.5 | Secret-history audit  | `gitleaks detect --source . -v` and `git log --all -p -- .env`. If a leak is found, rewrite history with `git filter-repo` BEFORE flipping public.        |
| 1.6 | `.env.example` (new)  | All required env vars (LANGFUSE\_\*, OPENROUTER_API_KEY, OPENAI_API_KEY, etc.) with empty values + one-line comments.                                     |

### Expected outcome

- Repo passes `gitleaks` clean.
- GitHub will render a "Cite this repository" button when public.
- A new contributor can `git clone && uv sync && uv run oasis-llm dashboard` from README alone.

### Success criteria

- [ ] `gitleaks detect` exits 0
- [ ] `git log --all --full-history -- .env` returns empty
- [ ] `cffconvert -i CITATION.cff --validate` passes
- [ ] `LICENSE` byte-matches the [SPDX MIT template](https://spdx.org/licenses/MIT.html)
- [ ] All README links resolve in `mint dev` preview

### Risks

- **Historical secret leak** → destructive history rewrite required. Mitigation: dry-run with `git filter-repo --analyze`, back up `.git/` before rewriting.

---

## Phase 2 — Demo bundle infrastructure

**Goal**: Make the codebase capable of bootstrapping a usable DuckDB from committed `.zip` bundles. Used by both Phase 4 options.

### Tasks

| #   | File / action                             | Detail                                                                                                                                                                              |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | `data/public/` (new dir, committed)       | Holds `.zip` bundles plus a README cataloging them.                                                                                                                                 |
| 2.2 | `data/public/README.md` (new)             | Per-bundle: name, source experiment id, generation date, n trials, models, intended use ("paper figure 3", etc).                                                                    |
| 2.3 | `src/oasis_llm/public_bootstrap.py` (new) | Module with `bootstrap_from_bundles(db_path, bundles_dir)` — idempotent: if DB exists, no-op; else create schema, iterate `*.zip`, call `bundles.import_any()` for each. ~30 lines. |
| 2.4 | `tests/test_public_bootstrap.py` (new)    | Two cases: (a) bootstrap creates DB + imports a fixture bundle; (b) re-running is a no-op.                                                                                          |
| 2.5 | Bundle generation (manual, local)         | User selects which experiments/analyses are public; runs `oasis-llm experiment-export <id> -o data/public/<name>.zip` for each.                                                     |

### Expected outcome

`bootstrap_from_bundles()` populates a fresh DuckDB from committed bundles within ~5 seconds. Removing/adding bundles + deleting the DB + re-bootstrapping reflects the change.

### Success criteria

- [ ] `pytest tests/test_public_bootstrap.py` passes
- [ ] On a fresh clone with no DB, calling `bootstrap_from_bundles()` produces a DB whose `runs` and `trials` tables match the bundle contents
- [ ] Second call exits early without re-importing
- [ ] No dependency added to `pyproject.toml`

### Risks

- **Bundle schema drift** between desktop export and public import → bootstrap fails. Mitigation: `bundles.py` already version-checks; bootstrap propagates that error.
- **Stale `.wal` from a prior process** → DuckDB connection failure. Mitigation: bootstrap deletes orphan `.wal` if main file is absent.

---

## Phase 3 — Public flip + Mintlify connect

**Goal**: Repo is public; docs site live at `<project>.mintlify.app`.

### Tasks

| #   | Action                                                           | Detail                                                                                             |
| --- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 3.1 | GitHub: Settings → General → Visibility → Public                 | **Irreversible.** Requires manual confirmation.                                                    |
| 3.2 | Mintlify: dashboard.mintlify.com → Add new project → select repo | Uses your existing Mintlify account.                                                               |
| 3.3 | Mintlify: Settings → Deployment → Git Settings                   | Enable "Set up as monorepo"; path = `/site/docs`; branch = `main`.                                 |
| 3.4 | Trigger first deploy                                             | Push a no-op commit; verify build logs in dashboard.                                               |
| 3.5 | (Optional) Custom domain                                         | Add `docs.<yourdomain>` in Mintlify; create CNAME `docs → cname.mintlify-dns.com`; wait for HTTPS. |

### Expected outcome

`https://<project>.mintlify.app` (or custom domain) serves live docs that auto-redeploy on every push to `main` touching `site/docs/`.

### Success criteria

- [ ] First Mintlify build succeeds (logs visible in dashboard)
- [ ] Each existing page (`/`, `/quickstart`, `/experiment-design`, …) renders without 404
- [ ] All internal links resolve
- [ ] `mint dev` from `site/docs/` matches the hosted output
- [ ] (If custom domain) HTTPS cert provisioned within 24 h

### Risks

- **`docs.json` v2 schema rejection** at hosted build time. Mitigation: run `mint validate` locally before push.
- **Mintlify Hobby tier excludes PR previews** — main-branch pushes deploy directly. Mitigation: develop on feature branches, merge once `mint dev` is clean.

---

## Phase 4 — Results viewer (CHOOSE ONE)

**Goal**: Publish curated experiment results with similar look-and-feel to the desktop dashboard, without exposing the runner / editor / settings surface.

> **Open decision**: Choose between Option β (Mintlify-embedded static results) and Option γ (slim Streamlit viewer). They have the same Phase 2 dependency and the same export workflow; they differ in UX richness and infra cost.

### Option β — Mintlify-embedded static results

**Idea**: Each published bundle is rendered to a Mintlify MDX page with embedded figures (PNG/SVG) and tables (data dumped to JSON, rendered with `<Frame>` and `<table>`). Lives at `<project>.mintlify.app/results/<bundle-name>`. **No new hosting platform.**

#### What ships

| File                                 | Purpose                                                                                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/oasis_llm/cli.py` (edit)        | New subcommand `oasis-llm results-export <bundle.zip> -o site/docs/results/<name>.mdx`                                                          |
| `src/oasis_llm/results_md.py` (new)  | Bundle → MDX rendering: title, methodology block, figures (saved as PNG via plotly's `to_image()`), tables (rendered as MDX), provenance footer |
| `site/docs/results/` (new directory) | One MDX per bundle + an `index.mdx` listing all published results                                                                               |
| `site/docs/results/_assets/`         | Static images / figures referenced by the MDX pages                                                                                             |
| `site/docs/docs.json` (edit)         | New "Published Results" tab/group                                                                                                               |

#### Pros

- Zero new hosting infra — leverages Mintlify deploy from Phase 3
- Pages are SEO-indexable, linkable, citeable
- Survives forever (static), no sleep, no memory cap
- ~200 lines of new code (CLI subcommand + MDX renderer)

#### Cons

- Static — no interactive filters, no drill-down
- Look-and-feel is Mintlify (clean, but not the desktop-dashboard look)
- Plotly figures become PNG (no hover, no zoom)

#### Success criteria

- [ ] Each bundle in `data/public/` produces a single MDX page when CLI is run
- [ ] Generated MDX validates with `mint dev` (no broken refs)
- [ ] Figures render correctly in dark and light Mintlify themes
- [ ] Per-page generation completes in < 5 s on a typical bundle
- [ ] No new runtime deps (Plotly + Pillow already present)

---

### Option γ — Slim Streamlit results viewer

**Idea**: A new entrypoint `streamlit_results.py` that registers only Home + Image Explorer + Runs + Analysis pages. Bootstraps DB from `data/public/*.zip` on first boot (via Phase 2). Hosted on Streamlit Community Cloud at `<subdomain>.streamlit.app`. **Reuses the existing dashboard pages — no UI rewrite.**

#### What ships

| File                                                             | Purpose                                                                                                                                                 |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `streamlit_results.py` (new, repo root)                          | Entrypoint: calls `bootstrap_from_bundles()` then registers a 4-page subset of `dashboard_pages/`                                                       |
| `src/oasis_llm/dashboard_pages/results_home.py` (new, ~50 lines) | Public-friendly landing page listing the bundles + their provenance — replaces full Home which links to admin pages                                     |
| `src/oasis_llm/dashboard_pages/_ui.py` (edit)                    | Add `is_results_viewer()` helper for any page that needs to omit a write button                                                                         |
| `src/oasis_llm/dashboard_pages/runs.py` (edit, minimal)          | Hide run-control buttons when `is_results_viewer()` is True (currently keyed off `OASIS_LLM_READONLY` — extend to also recognise the viewer entrypoint) |
| `pyproject.toml` (no edit)                                       | All deps already present                                                                                                                                |

#### Pros

- Same desktop look-and-feel — the same pages, same filters, same plotly interactivity
- Drill-down preserved (click a run → see trials, click a trial → see prompt + reasoning)
- Imports the same code paths as desktop, so they cannot drift

#### Cons

- Requires a second hosting platform (Streamlit Community Cloud)
- Free tier: 2.7 GB memory ceiling, 12-hour idle sleep
- Cold-start ~30s on wake-from-sleep
- Custom domain not on free tier (hosted at `<subdomain>.streamlit.app`)

#### Success criteria

- [ ] Cold-start completes within 60s, no OOM
- [ ] Memory after first navigation < 2.5 GB
- [ ] All four registered pages render and reflect the imported bundles
- [ ] Settings / Datasets / Experiments / Import-Export pages do **not** appear in nav
- [ ] No write buttons (start, pause, edit, delete) appear anywhere
- [ ] Wake-from-sleep on a fresh visit completes within 90s

---

### Comparison

| Dimension                  | β (Mintlify static)             | γ (Streamlit slim)                                   |
| -------------------------- | ------------------------------- | ---------------------------------------------------- |
| Hosting platforms          | 1 (Mintlify)                    | 2 (Mintlify + Streamlit Cloud)                       |
| New code                   | ~200 lines                      | ~80 lines (mostly entrypoint + small UI conditional) |
| UX similarity to desktop   | Low                             | High                                                 |
| Interactivity              | None                            | Full                                                 |
| SEO / citeability          | Excellent                       | Poor (Streamlit not crawler-friendly)                |
| Cold-start latency         | None                            | ~30–90 s after sleep                                 |
| Custom domain on free tier | ✅                              | ❌                                                   |
| Maintenance                | Regenerate MDX on bundle change | Zero — bundles auto-bootstrap                        |

**Default recommendation**: **β** for "this is a research artefact people will cite and link to from papers." **γ** for "people should be able to play with the data the way I do on my desktop." Either can be added later if the other is chosen first — they share Phase 2.

---

## Phase 5 — Docs alignment + commit

**Goal**: Make the docs site match what shipped in Phases 1–4.

### Tasks

| #   | File                                                             | Detail                                                                                                                                  |
| --- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1 | `site/docs/index.mdx`                                            | Full restructure of card group around the real surface: 8 dashboard pages + CLI + Bundles + Discoveries + Live results + GitHub source. |
| 5.2 | `site/docs/deployment.mdx` (new)                                 | Document the deploy process from this plan so contributors can fork-deploy. ~120 lines. Add to `docs.json` Reference group.             |
| 5.3 | `site/docs/quickstart.mdx`                                       | Add "Try the live results" call-out at top with the published-results URL (Mintlify if β, Streamlit if γ).                              |
| 5.4 | `site/docs/cost-estimation-saga.mdx`                             | Fix `19 tests` → `27 tests`.                                                                                                            |
| 5.5 | `site/docs/discoveries-methodology.mdx`, `ollama-operations.mdx` | No changes needed — already accurate.                                                                                                   |
| 5.6 | (β only) `site/docs/results/index.mdx` + per-bundle pages        | Generated by the new CLI subcommand from Phase 4β.                                                                                      |
| 5.7 | Single commit + push                                             | `docs: research-report restructure + deployment guide + published results`                                                              |

### Expected outcome

Mintlify auto-redeploys; new homepage + deployment page (+ results pages if β) live within ~1 minute of push.

### Success criteria

- [ ] All four "Discoveries" pages render from the homepage cards
- [ ] `deployment.mdx` URLs match what's actually live (no placeholders)
- [ ] (β) Each `data/public/*.zip` has a corresponding `/results/<name>` page
- [ ] `mint dev` shows zero broken links before push
- [ ] Mintlify build succeeds

### Risks

- **Mintlify build failure** → live site stays on previous good build. Mitigation: `mint dev` locally before push.

---

## Cross-phase summary

| Phase | What ships                                          | Reversible?                    | External dependency                        |
| ----- | --------------------------------------------------- | ------------------------------ | ------------------------------------------ |
| 1     | Hardened repo (LICENSE, CITATION, README, gitleaks) | Yes                            | None                                       |
| 2     | Demo-bundle bootstrap + curated bundles             | Yes                            | At least one usable run/analysis to export |
| 3     | Repo public + Mintlify live                         | **No** (public = irreversible) | Mintlify account                           |
| 4β    | Mintlify-embedded static results                    | Yes                            | None beyond Phase 3                        |
| 4γ    | Slim Streamlit results viewer hosted                | Yes (delete app)               | Streamlit Cloud account                    |
| 5     | Docs reflect reality                                | Yes                            | None                                       |

Total scope after Phase 4 decision:

| If 4β | ~10 new files, ~6 edits, ~1 manual platform setup (Mintlify)             |
| ----- | ------------------------------------------------------------------------ |
| If 4γ | ~6 new files, ~7 edits, ~2 manual platform setups (Mintlify + Streamlit) |

---

## Decision log

| Date    | Decision                                                  | Rationale                                                                                                    |
| ------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| _today_ | License = **MIT**                                         | User selection. Common academic-code default.                                                                |
| _today_ | Subset publishing strategy = **C1 (bundles + bootstrap)** | Reuses existing `bundles.py`. Per-bundle granularity matches "publish curated subsets".                      |
| _today_ | Homepage scope = **Full restructure into feature-grid**   | Real product surface (8 pages + CLI + bundles + analyses) is larger than the existing 7-card group reflects. |
| _open_  | Phase 4 option = **β or γ**                               | Awaiting decision. Default recommendation: β for citeable static results; γ for desktop-like interactivity.  |
| _open_  | Optional add-ons (CI, pre-commit, UptimeRobot, Sponsors)  | Awaiting decision. None required for launch.                                                                 |
| _open_  | Execution order                                           | Awaiting decision. Default: 1 → 2 → 3 → 4 → 5 (with explicit pause before Phase 3 public-flip).              |

---

## How to use this file

This is the canonical implementation reference. When working on the public release:

1. Re-read the relevant phase before starting.
2. Mark tasks complete in the tracked todo list (managed via the agent's todo tool — see `manage_todo_list` notes in agent memory).
3. Update the **Decision log** above when a choice is made.
4. If a phase reveals new constraints, edit this file in the same commit as the affected code change, with a one-line note in the Decision log.
