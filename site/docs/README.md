# OASIS-LLM docs (Mintlify)

This is a [Mintlify](https://mintlify.com) v2 documentation site for the
`oasis-llm` experiment harness in this repo.

## Preview locally

```bash
# one-time install
npm i -g mintlify

# from this directory
cd site/docs
mintlify dev
```

Open <http://localhost:3000>. The dev server hot-reloads on save.

## Deploy

Mintlify deploys directly from a Git branch. Point a Mintlify project at this
repo with **content path** set to `site/docs`. The config is `docs.json`
(Mintlify v2 — _not_ the older `mint.json`).

## Editing

- All pages are `.mdx`. YAML frontmatter is required (`title`, `description`).
- Mermaid diagrams use ` ```mermaid ` fenced code blocks; Mintlify renders them natively.
- Mintlify components used: `<Card>`, `<CardGroup>`, `<Note>`, `<Warning>`,
  `<Tip>`, `<Steps>` / `<Step>`, `<Tabs>`, `<CodeGroup>`,
  `<Accordion>` / `<AccordionGroup>`.
- Navigation lives in `docs.json` under `navigation.tabs[].groups[].pages`.
  Each entry is the page filename without the `.mdx` suffix.
- Page slugs are filename-based: `experiment-design.mdx` → `/experiment-design`.

## Source-code references

The docs reference Python modules in `src/oasis_llm/`. These are described in
inline code spans rather than linked, because Mintlify won't render relative
links to files outside the docs root. If you move or rename a referenced
module, grep this directory for the old path.

## File map

| File                    | What's in it                                              |
| ----------------------- | --------------------------------------------------------- |
| `docs.json`             | Mintlify v2 config (theme, navigation, colors).           |
| `index.mdx`             | Landing page with navigation cards.                       |
| `quickstart.mdx`        | Minimal install + `oasis-llm dashboard`.                  |
| `experiment-design.mdx` | 7-point Likert, paper-verbatim prompts, sample-size math. |
| `image-set.mdx`         | Named subsets, stratified sampling, custom image lists.   |
| `workflow.mdx`          | Mermaid sequence + state diagrams, trial schema.          |
| `configuration.mdx`     | Every `RunConfig` field, canonical-hash semantics.        |
| `discrepancies.mdx`     | Between- vs within-subject, N per cell, etc.              |
| `cost-latency.mdx`      | Pilot numbers, full-set extrapolation, cost capture.      |
| `reasoning-capture.mdx` | The Gemma 4 saga + the prompt-rewrite fix.                |
| `cache-buster.mdx`      | Per-sample salt, KV-cache placement, limitations.         |
| `glossary.mdx`          | Valence, arousal, ICC(2,k), Spearman-Brown, etc.          |
