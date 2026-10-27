# PR20 · Is it the alarm, or the missing calm?

- Governs E45. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **soundness** first: it turns the one hypothesis the paper currently states without a test
  into a tested claim, in either direction.

## The question in one sentence

When a reader responds to the anxious rendering of a message, is it responding to the alarm that
was added, or to the reassurance that was removed?

## Why it is worth the GPU time

E41 found that a filtered
message --- alarm deleted, facts intact, no self-referential sentence left --- still moved the most
responsive reader to within 7% of the anxious original. The reading that explains it is that the
calm arm's closing (*"I am composed and unworried as I write this"*) actively pulls readers down,
so most of what we had measured as a response to alarm at that role is a response to reassurance
going missing. That reading was formed after seeing E41's result. The paper states it as a
post-hoc hypothesis in three places. This run is the designed test E41's entry promised, and it
changes what the mitigation section is allowed to conclude: a filter cannot restore calm it never
removed, but only if the calm was doing the work in the first place.

## What is held fixed, and what varies

**Fixed.** The same 59 matched stimulus pairs (the deposit-outflow event), the same five reader
roles, both contexts, three draws, the same lexicon, scrub and aggregation as E36, E40 and E41, so
the numbers sit on the same scale as everything else here. Reader model `Qwen/Qwen3-4B-Instruct-2507`
at the pinned revision. Seed base 1,000,000, colliding with no earlier design.

**Varies.** Readers see one of three texts, differing only in what follows the verbatim fact block:

| Arm | What ends the message |
| --- | --- |
| `neutral` | *"I am composed and unworried as I write this."* |
| `anxiety_like` | *"I am alarmed, anxious and frightened, and I write this filled with dread."* |
| `no_closing` | nothing --- the frame and the fact block, then the message ends |

The runner refuses to start unless the `no_closing` text is a strict prefix of both other arms.

## The two contrasts, and the reference gate

Per role, anxious-minus-calm must first reproduce: **the trader's `anxiety_like` − `neutral` must
be positive with an interval clear of zero.** If it does not, the run is reported as a failed
reproduction and no decomposition verdict is issued for any role.

Then, per role, paired within pair, Holm across the five roles within each contrast separately:

1. **Removing the calm:** `no_closing` − `neutral`. Positive means the calm closing's absence alone
   raises the reply.
2. **Adding the alarm:** `anxiety_like` − `no_closing`. Positive means the alarm adds something
   beyond the missing reassurance.

The two sum to the reference effect by construction, so the interesting quantity is the split. For
each role whose reference effect is clear of zero, report the **calm-absence share**
(`no_closing` − `neutral`) / (`anxiety_like` − `neutral`) with a bootstrap interval.

## The decision rule, fixed now

Verdicts are issued per role, only where the reference effect is clear of zero:

| Outcome | Verdict |
| --- | --- |
| Contrast 1 clear of zero (positive), contrast 2 containing zero | **Calm-absence confirmed** at that role. The paper's hypothesis becomes a designed-test result there. |
| Contrast 2 clear of zero (positive), contrast 1 containing zero | **Alarm-driven.** The calm-absence reading is refuted at that role and comes out of the abstract entirely. |
| Both clear of zero (positive) | **Both components real.** Report the share; the abstract states the decomposition with its split, not as either pure story. |
| Neither clear of zero | **Uninformative** at that role --- the design lacked the precision, and that is what is reported. Not a refutation. |
| Either contrast negative and clear of zero | Reported as-is; E41's fundamental-analyst flattening makes this a live possibility for the role whose reference effect is negative. |

The interpretation of reassurance follows the trader's verdict, since the trader's gap is
the reference for E41's 7% comparison. Report the result regardless of its direction.

## Sensitivity, computed before the run

E41 measured per-role contrasts on these pairs with intervals of roughly ±0.003 to ±0.005. If
E41's decomposition is roughly right, the trader splits into about +0.023 (calm-absence) against
about +0.002 (alarm), which this design separates cleanly. The bull researcher's predicted split
(~+0.003 against ~+0.005) sits at the precision floor and will likely be uninformative; that is
written here so it is not presented afterwards as a finding about that role. Roles whose reference
effect is near zero (risk manager) are uninformative by construction.

## Cost

59 pairs × 5 roles × 2 contexts × 3 arms × 3 draws = **5,310 readings**, no rewrite pass. About
three hours, on the same throughput as E41. The reading count is fixed in advance: anything other
than 5,310 lines is a failed run.

## What it cannot answer

- **One closing sentence per arm.** The alarm and calm sentences are single fixed token sequences;
  the decomposition is of *these* sentences, not of alarm and reassurance in general.
- **One event, one model, readers only.** Same limits as E36/E41.
- It does not test the filter again. E41's preservation and attenuation numbers stand regardless.
- It cannot say whether *adding* calibrated calm would mitigate --- that is an intervention on the
  sender, not an ablation of the stimulus, and it is a different run.
