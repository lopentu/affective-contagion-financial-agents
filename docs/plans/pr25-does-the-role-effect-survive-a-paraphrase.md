# PR25: does the role effect survive a paraphrase of the role prompts?

- Status: **frozen 2026-08-09**, before any data existed. Analysis `e50_analysis.py`,
  SHA-256 `f827baabd73f6be4`, hashed before the run was queued.
- Experiment: E50.

## The question, in plain words

The paper's central claim is that the reader's assigned role changes the response. But each role
is one exact prompt string, so the claim could be about five strings rather than five roles. We give
each reader role two more wordings that say the same thing in different words, and ask whether the role
differences appear under every wording.

**Which research question this serves**: RQ1's central evidence (the role-heterogeneity result,
E36) — this is a validity check on the paper's headline claim, on the soundness axis.

**Why it is worth the GPU time**: about five hours buys the answer. Either answer improves the paper:
survival strengthens the central claim past its
main attack; failure narrows the claim and promotes the role-vs-wording confound from limitation to
finding.

**What a null would mean**: with the gate below passing, a flat spread under a new wording is a
real "the wording drives it", not "we could not tell" — the sensitivity analysis in the frozen
script shows E36-scale spreads sit 4-6 sigma above the detection floor at this design size.

## Design

Three rosters, identical except the five READER system prompts (the treated sender's prompt and
the model block are byte-identical; stimuli are pre-generated so the sender prompt is inert):

- **Wording A**: `configs/models/ladder/qwen3_4b.yaml` (the original, E36's roster).
- **Wording B**: `configs/models/paraphrase/qwen3_4b_wording_b.yaml`.
- **Wording C**: `configs/models/paraphrase/qwen3_4b_wording_c.yaml`.

Both paraphrase files were written today, before any data, by paraphrasing each prompt to
preserve its content (including the risk manager's rhetoric clause, whose meaning is kept in
different words in both).

Per wording: the same 59 matched deposit-outflow pairs (the standing gate's survivors, seeds as
in every experiment since E36), 5 reader roles x 2 contexts (bare, embedded) x 2 arms x 3 draws,
1 replicate = **3,540 readings**; 10,620 total. Runner: `scripts/run_context_comparison.py`, the
E36 runner, with `--replicates 1`. Model, revision, decoding, lexicon: pinned as everywhere.

## Outcome, estimator, decision rules

All fixed in the hashed script; summary:

- Estimator identical to the registry's `role_effect` pairing (draws to cell, arms differenced
  within (seed, role, context), contexts pooled within pair, decision vocabulary scrubbed). The
  estimator was checked against a known answer before freezing: fed the released E36 readings it
  returns every registered value exactly (trader +0.0228, fundamental analyst -0.0052, spread
  0.0107, p 0.0005).
- **Gate (run health, not a finding)**: wording A's trader estimate within +-0.008 of +0.0228 and
  wording A's spread permutation p < 0.05. Gate failure means the run is invalid, full stop.
- **H1**: spread permutation test per new wording, Holm over {B, C}. Both survive -> heterogeneity
  is not one string's artifact. One -> partial, claim only that wording. None -> the central claim
  narrows to the original strings and the confound becomes a result.
- **Secondary, descriptive only**: per-role tables with bootstrap intervals per wording; no rank
  test across wordings at n=5 (E39's lesson).

## Exclusions

None beyond the standing 59-pair gate, already applied to the stimulus set. Reading counts other
than exactly 3,540 per wording fail the queue's artifact check and mark the run failed.

## What each result licenses

Exactly what the decision rules above say, and nothing about: which role responds most (three
prior checks reordered the roles), other models, other events, distributed (non-announced)
affect, or human perception.
