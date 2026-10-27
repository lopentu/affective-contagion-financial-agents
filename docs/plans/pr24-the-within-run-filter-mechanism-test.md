# PR24 · The filter's leak, tested inside one run

- Governs E49. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **soundness**: the paper currently says the filter's leak is its own regenerated wording,
  and flags in Limitations that this "compares two experiments; its within-run test has not been
  built." This run builds it, so the claim either becomes a within-run result or gets corrected.

## The question in one sentence

Inside one run, does the filtered message move readers where the no-closing message does not — the
gap that can only be the rewrite's own wording?

## Why it is worth the GPU time

E45 showed a no-closing message moves the trader
not at all; E41 showed the
filtered message moves it within 7% of anxious. The regenerated-wording conclusion subtracts one
experiment from another. Generation here is intermittently non-reproducible, so a cross-run
difference could in principle be a run difference. One design with all four arms closes that.

## What is held fixed, and what varies

**Fixed.** E41's design and gate exactly: the 59 meridian pairs, five reader roles, both contexts,
three draws, fresh same-model rewrites gated slot-by-slot (pairs with a failed rewrite excluded
from every arm), `Qwen/Qwen3-4B-Instruct-2507` pinned.

**Varies.** Four arms per pair: `neutral`, `anxiety_like`, `filtered`, and `no_closing` — the frame
plus verbatim fact block with no self-referential sentence, byte-identical to E45's arm. The runner
gains this arm behind a flag that defaults off, with a test pinning the arm to
`payload.realize("no_closing")`. At full preservation: 59 × 5 × 2 × 4 × 3 = **7,080 readings**,
fewer only if rewrites are rejected (the E41 preservation gate and its 40-of-59 floor are
unchanged).

## The contrasts, in the order they will be read

1. **Reference gate:** `anxiety_like` − `neutral` at the trader must be positive and clear of zero,
   or the run is a failed reproduction and no verdict is issued.
2. **The leak, within-run:** `filtered` − `no_closing`, per role, Holm across five. The two texts
   share the absence of a closing sentence; they differ only in the rewrite's paraphrase of the
   frame and facts. Positive means the rewrite's own wording carries arousal.
3. **The E45 replication:** `no_closing` − `neutral`, per role — expected ≈ 0 at the trader and
   bear researcher, negative at the other three, matching E45 on fresh generations.
4. **The E41 replication:** `filtered` − `anxiety_like` and `filtered` − `neutral`, descriptive,
   read against E41's table.

## The decision rule, fixed now

| Outcome | What it licenses |
| --- | --- |
| Contrast 2 positive and Holm-surviving at the trader | **The regenerated-wording claim becomes a within-run result** and the Limitations caveat comes out. |
| Contrast 2 containing zero at the trader while contrast 4 reproduces E41 | The two earlier runs disagree with this one; the paper's claim reverts to "unexplained" and the Limitations caveat strengthens. Report prominently. |
| Contrast 3 failing to reproduce E45's pattern | E45's ablation is run-sensitive; both E45's entry and the paper get dated corrections. |
| Preservation below 40 of 59 | Failed filter pass; everything descriptive, as PR17 fixed. |

## Sensitivity, computed before the run

The quantity in contrast 2 was ≈ +0.023 at the trader across E41/E45 (0.0229 minus 0.0000); per-role
precision at 59 pairs and three draws is ±0.003–0.005, so an effect of that size is unmissable and
even a quarter of it is detectable.

## Cost

59 rewrites plus up to 7,080 readings ≈ 4 hours, queued behind E48 overnight.

## What it cannot answer

- One filter prompt, one model, one event — E41's scope limits carry over unchanged.
- It does not test a register-aware gate; it establishes whether one is needed.
