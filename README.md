# LongRoPE diagnostic — does the |V|=128 prompt cross Phi-3.5's 4096-token seam?

Standalone, tokenizer-only diagnostic supporting one specific finding in the
submitted paper: a non-monotonic success-rate collapse observed for
Phi-3.5-mini-instruct at exactly `|V|=128` in the cross-family generalization
check. The paper reports this as an investigated near-miss, not a confirmed
explanation, and this notebook is the artifact behind that investigation.

## What it checks

Phi-3.5-mini-instruct uses LongRoPE positional scaling, which switches between
short- and long-context interpolation factors at
`original_max_position_embeddings = 4096` tokens. The paper pre-registers a
falsifiable hypothesis: if the actual `|V|=128` prompts land within a declared
tolerance of that 4096-token boundary, a regime switch at generation time is a
plausible (not confirmed) explanation for the anomaly.

This notebook answers that with a fact, not a guess: it reconstructs the three
actual prompts the original experiment evaluated at `|V|=128` (same task IDs,
same seeds, same candidate-set construction, same system prompt), tokenizes
each one with Phi-3.5-mini-instruct's real tokenizer via
`apply_chat_template`, and reports the exact token count against the declared
±300-token tolerance.

## What it does NOT do

- Does not load any model weights.
- Does not run generation.
- Does not require a GPU.
- Does not reproduce the anomaly itself, only measures the one input property
  (prompt length in tokens) the pre-registered hypothesis depends on.

## How to run it

**Kaggle:** Settings → Accelerator → **None (CPU)**. Settings → Internet →
**On** (needed to fetch the small BFCL data files and the tokenizer, both
small downloads). Run All. Finishes in under a minute.

**Local Jupyter:** `pip install transformers==5.16.1`, then Run All. Same
runtime, no GPU required either way.

## What the output means

The notebook prints, per case, the exact token count of the real `|V|=128`
prompt Phi-3.5-mini-instruct actually saw during evaluation. Compare this
directly against 4096 ± 300 (the paper's pre-registered tolerance) to see
whether the measurement falls inside or outside that declared window. The
paper reports the result and its interpretation directly in the corresponding
section — this notebook exists so that specific number can be independently
reproduced in under a minute rather than taken on trust.

## Data and license note
## verify_headline_numbers_bundle/

A second standalone notebook, independent of the first. Reconstructs four
of the paper's headline numbers directly from bundled raw per-case CSVs,
using code written fresh for this purpose — not a re-run of the paper's
own analysis functions.

**What it reconstructs:**
- All four fitted crossover points from Table 1 (1.5B, 3B, 7B, 14B)
- The α scaling exponent (log-log fit across those four points)
- One H9 filtering effect size (7B, at |V|=234)

**How to run it:** open `verify_headline_numbers.ipynb` and run all cells
in order. No GPU, no model weights, no network access required — everything
needed is in the accompanying `data/` folder. Runs in well under a minute.

**What "independent" means here, precisely:** this notebook was written by
the same overall process that produced the paper, using a different code
path than the original analysis (not the same functions re-executed) —
not by a separate human or system. It lets you check these specific numbers
yourself, with code short enough to read in full, rather than trusting a
self-report. It does not cover every number in the paper, and it is
explicitly not a substitute for full external re-derivation. Scope and
limitations are stated in the notebook's own first cell.

**Expected output:** every reconstructed value should print `[MATCH]`
against the value the paper claims. If anything prints `[** MISMATCH **]`,
that's worth reporting.
BFCL task data is fetched at runtime from its public source
(github.com/ShishirPatil/gorilla, Apache 2.0) and is not redistributed here.
The Phi-3.5-mini-instruct tokenizer is fetched at runtime from its public
Hugging Face model card.
