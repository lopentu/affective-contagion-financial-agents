# PR23 · Does alarm cost the recommendation its force in other model families?

- Governs E48. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **soundness of the paper's newest headline**. "Alarm costs the recommendation its force"
  (E46, replicated across polarity by
  E47) is a one-model result, and this project has
  already published one effect that turned out to be one model's alone — the sign reversal, which
  E39 found in Qwen and neither other family. The action claim must not repeat that history
  unchecked.

## The question in one sentence

Do Granite and Gemma readers also follow an alarmed sender's SELL recommendation less?

## What is held fixed, and what varies

**Fixed.** E46's design exactly: the 59 meridian pairs, five reader roles, two arms, three draws,
cognition fields on, position size the primary outcome, the anchor check read first. The stimuli
stay byte-identical (still written by the primary model), so a difference is the reading family.

**Varies.** The reader model only: `ibm-granite/granite-4.1-3b`, then `google/gemma-4-E4B-it` —
E39's two families at their pinned revisions. 1,770 readings each, counts fixed in advance.

## Outcomes and the decision rule, fixed now

Per family: parse health (≥ 95% carrying the three fields — a family that cannot produce the
cognition grammar is reported as unmeasurable, not null); the anchor check; the departure rate;
then position size per role with Holm across five, exactly E46's estimator.

| Outcome | What it licenses |
| --- | --- |
| At least one Holm-surviving negative size contrast in a family | The force-loss direction appears in that family; the paper may say the action effect is not the primary model's alone, scoped to however many families show it. |
| All intervals containing zero in a family | Uninformative or bounding, by width: if the family's trader interval excludes E46's −0.1034, that family responds less than Qwen and the paper says so; otherwise underpowered, and says that. |
| Holm-surviving **positive** size contrast | Direction reversal across families — reported with the same prominence E39 gave the sign reversal, and the paper's claim narrows to the primary model. |
| Parse floor missed | That family is unmeasurable under this grammar; not a null. |

Belief fields are recorded and reported per role as secondary, directions read descriptively;
`expected_return` decides nothing (unit mixing, per E46).

## Sensitivity, computed before the run

E46's Qwen effects ran −0.026 to −0.103 with CI half-widths ±0.02–0.05 at 59 pairs. The same design
at the same pair count has the same precision, so a Qwen-sized trader effect is comfortably
detectable in each family; effects below ~0.03 per role are not, and are bounds.

## Cost

2 × 1,770 readings ≈ 2.5–3 hours, on an idle GPU, chained Granite then Gemma.

## What it cannot answer

- Stimuli remain Qwen-written; whether other families *produce* force-costing alarm is separate.
- One event (the SELL); the BUY replication in other families is another run.
- The discounting-vs-defensiveness mechanism is not retested here.
