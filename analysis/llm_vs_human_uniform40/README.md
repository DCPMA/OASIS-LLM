# LLM vs Human (OASIS) — t-test analysis

**Cohort.** Image set `20260426-uniform40` (40 OASIS images × 20 samples × 5 LLMs):
`anthropic/claude-sonnet-4.6`, `google/gemma-4-31b-it`, `openai/gpt-5.4`,
`qwen/qwen3.6-plus`, `x-ai/grok-4.20`. Total completed trials: **7,960**
(qwen3.6-plus is missing 1 image → 39 paired images, 780 trials).

**Reference.** OASIS `Valence_mean` / `Arousal_mean` per image (1–7 scale).

## Methods

For each dimension (valence, arousal):

1. **Per-model paired t-test** on the 40 image-level means
   (`LLM_image_mean` vs `Human_mean`). Reports paired *t*, *p*, Cohen’s *d*
   (mean diff / SD of diffs), and the Pearson *r* between LLM and human
   per-image means.
2. **Per-model Welch t-test** on raw LLM trial ratings (n=800) vs the 40
   human image means.
3. **Aggregate** comparison: pooled-LLM image mean (average of the 5 model
   means per image) vs the human image mean — paired t-test on 40 images,
   plus Welch on all 3,980 LLM trials vs human image means.

## Headline results

### Valence
- All 5 models track human valence very closely per image
  (**Pearson r = 0.93 – 0.96**).
- Paired t-tests are non-significant for 4 / 5 models; only **gpt-5.4** drifts
  slightly low (mean diff −0.27, *p* = 0.039, *d* = −0.34).
- Pooled LLMs vs humans: mean diff −0.03, *t*(39) = −0.26, *p* = 0.79.
  → **No detectable bias on valence in aggregate.**

### Arousal
- Per-image correlations are still strong but weaker
  (**Pearson r = 0.76 – 0.83**).
- LLMs systematically **rate arousal higher** than humans:
  - claude-sonnet-4.6: +0.59, *t*(39) = 3.45, *p* = 0.0014, *d* = 0.54
  - grok-4.20:        +0.65, *t*(39) = 3.14, *p* = 0.0032, *d* = 0.50
  - gpt-5.4:          +0.41, *t*(39) = 2.07, *p* = 0.045, *d* = 0.33
  - gemma-4-31b-it / qwen3.6-plus: not significant.
- Pooled LLMs vs humans: mean diff **+0.36** per image, paired
  *t*(39) = 1.91, *p* = 0.064; Welch on raw trials *p* = 0.0087.
  → **Marginal upward bias on arousal in aggregate, driven by Claude, Grok,
  and GPT-5.**

See `descriptives.csv`, `ttests_per_model.csv`, `ttests_aggregate.csv`,
`per_image_means_*.csv`, and `plots/` for full numbers and figures
(violin, histograms, per-image scatter, bias boxplots).

## Files
- `descriptives.csv` — N, mean, SD, median, min, max per source/dimension.
- `per_image_means_{valence,arousal}.csv` — wide table: image × (each model, LLM_mean, Human_mean, Category).
- `ttests_per_model.csv` — paired + Welch t-tests for every model × dimension.
- `ttests_aggregate.csv` — pooled-LLM vs human, paired + Welch.
- `plots/violin_{dim}.png` — per-model rating distributions vs human mean ± 1 SD.
- `plots/hist_aggregate_{dim}.png` — pooled LLM trials vs human image means.
- `plots/scatter_aggregate_{dim}.png` — per-image LLM_mean vs Human_mean (with y=x and Pearson r).
- `plots/scatter_per_model_{dim}.png` — same scatter, broken out per model.
- `plots/bias_box_{dim}.png` — boxplot of (model − human) per-image differences.

Reproduce: `python scripts/llm_vs_human_ttest.py`
