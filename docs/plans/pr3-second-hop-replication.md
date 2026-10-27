# PR3 · Independent replication of the second hop

- Written: 2026-08-05, **before the seed-36-and-up runs existed**. Committed before the first run.
- Governs: E18. Does **not** govern, modify, or pool with E16's 28-pair result.
- Status: **frozen**, with one dated arithmetic correction below.

## Correction, 2026-08-06: the sign threshold as written was impossible

The rule below said the positive-seed count must reach "`sign_test_threshold(n)` — **37 of 60** if all
pairs survive". That parenthetical is wrong and is corrected here rather than quietly restated.

This plan runs **seeds 36–67, which is 32 pairs**, and says twice that the two blocks are *never pooled*.
So:

- `sign_test_threshold(32) = 22`. The operative threshold is **22** if all 32 pairs survive the pairing
  gate, and `sign_test_threshold(n)` on whatever number actually survives otherwise.
- **37 > 32.** Applied literally, the rule could not have been satisfied by any data whatsoever, so the
  replication would have been reported NOT SUPPORTED regardless of what it found.

The "37 of 60" figure belonged to the earlier design that pooled these pairs with E16's 28 toward 60 —
the very design this plan exists to reject as outcome-triggered continuation. The rule's own surviving
clause, "computed from the surviving n and not from a target", was always the operative instruction; the
parenthetical contradicted it.

**Why correcting this now is not outcome-triggered.** At the time of correction the stimulus pairs were
still generating and **no reader had run**, so no outcome of this experiment existed in any form. This
fixes an arithmetic slip against the plan's own stated intent, before data — it does not move a threshold
after seeing a result. The corrected threshold is the same rule PR2 used — an exact 5% one-sided
binomial test — evaluated at the n this plan actually runs. It is 22 of 32 (69%) against E16's 19 of 28
(68%), so it is no easier a bar than the one E16 failed.

## Why this is a new plan and not more data for the old one

After [PR2](pr2-isolated-reader.md)'s arm B came in at +0.0113, CI [+0.0003, +0.0224], **17 of 28 signs
against the 19 its own rule requires**, I queued 32 extra stimulus pairs to "fix" it. This is
outcome-triggered continuation: adding observations *because* a result failed and then
retesting at the same threshold inflates the false-positive rate. That is precisely the mistake
[PR2 B1.2](pr2-isolated-reader.md) withdrew when it removed the two-stage 12-then-30 design.

I had told myself I was "adding data rather than relaxing the rule". Both are ways of getting a second
chance at the same threshold, and the second one is harder to notice.

**So:** E16's 28-pair analysis is frozen exactly as reported, with its NOT SUPPORTED verdict. This plan
covers a **separate replication on new seeds**, analysed on its own and reported on its own. The two are
never pooled.

## Design (fixed)

Identical to PR2 as amended, except the seed range.

| | |
| --- | --- |
| Stimulus pairs | seeds **36-67**, none previously executed; gated by `check_pairing.py` before any reading |
| Reader roles | the 5 untreated roles |
| Draws | 3 per (pair, role, condition) |
| Arms | `real` and `silent` only. The matched arm is **not** repeated: PR2 B4.3 established it has one stimulus pair, so more seeds add nothing to it |
| Model | Qwen3-4B-Instruct-2507 @ `cdbee75f…` |
| Everything else | as PR2 |

## Primary outcome and rule (fixed, unchanged from PR2)

Reader arousal, aggregated draws → cell → arms → seed, by-seed interval, and:

1. the 95% interval excludes zero on the positive side, **and**
2. the count of positive seeds reaches `sign_test_threshold(n)`, computed from the surviving n and not
   from a target — **22 if all 32 pairs survive**. (This line originally read "37 of 60"; see the
   dated correction at the top. 37 was impossible at n=32 and belonged to a withdrawn pooled design.)

Deliberately the same rule as PR2, so this is a genuine replication rather than a fresh attempt with a
friendlier bar.

## What each outcome licenses (fixed in advance)

| E16 (28 pairs) | E18 (this) | Licensed statement |
| --- | --- | --- |
| not supported | not supported | no detectable reader effect across two independent blocks; report the pooled *estimate* descriptively with both intervals, and make no significance claim |
| not supported | supported | E16 was underpowered; E18 stands as the primary evidence, and E16's failure is reported alongside it |
| not supported | supported, opposite sign | treat both as noise and say so |

Note there is no row where E16 becomes supported. Its verdict is fixed.

## Declared in advance: the "all tone" claim is retracted, not retested

PR2's write-up said arm A (+0.0105) and arm B (+0.0113) agreeing meant "essentially all of the reader
effect is tone, not content". That is **withdrawn**. Verified: the two arms are uncorrelated at cell level
(r = −0.05) and seed level (r = −0.08), with sign agreement 15 of 28, which is chance. Equal means with
zero correlation is coincidence, not mediation.

Worse for the claim, splitting arm B by PR2's content gate runs the wrong way: **wording-only pairs
+0.0081 (n=7), content-differing pairs +0.0124 (n=21)**. If tone were doing the work, the wording-only
pairs would show the larger effect. They show the smaller one.

E18 will report that split again on new seeds as a **declared secondary**, but no version of the "all
tone" claim may be revived from it.

## What is not claimed regardless of outcome

Everything PR2 excludes still applies: lexical arousal coupling rather than anxiety, one scenario, one
model, one treated role, one hop. Plus:

- **The silent baseline is not a clean no-message control.** It omits the entire team-messages section of
  the prompt rather than presenting an empty one, so it differs structurally and not only in content. It
  is reported as a descriptive floor and used for **no** inference. Fixing it properly needs a prompt
  variant that keeps the section and empties it, which is out of scope here.
- **"Propagation" is the wrong word for a first hop.** The first hop is the sender's own message arousal.
  Only a reader outcome speaks to propagation.
