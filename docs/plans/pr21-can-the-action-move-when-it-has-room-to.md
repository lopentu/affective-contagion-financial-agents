# PR21 · Can the action move, when it has room to?

- Governs E46. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **excitement** first. Words move words and stated beliefs, not actions. An action consequence
  "would move the paper from 'words move words' to 'words move decisions'".

## The question in one sentence

When the recommended trade is a real position rather than doing nothing, does the anxious rendering
change the position readers take?

## Why it is worth the GPU time

E30 found beliefs moving and actions not moving —
but its event recommends **HOLD at size 0.00**, so a reader could only act by disobeying an explicit
instruction, and 0 of 720 did. That is a ceiling the design built, not a fact about the agents. The
deposit-outflow event recommends **SELL at size 0.15**: size becomes a number a reader can shade up
or down without disobeying anything. If anxious wording makes readers sell harder, that is the
"words move decisions" result. If readers copy 0.15 exactly whatever the tone, the anchor dominates
even when graded deviation is available — an informative negative this time, because the design gave
them room.

Either outcome also buys a free replication: the same readings carry the three belief fields, so
E30's belief result is tested on a second event at 2.5× the pairs.

## What is held fixed, and what varies

**Fixed.** The 59 matched pairs of the deposit-outflow event (the same s30 batch as E36/E41/E45,
pairing-gate checked before any generation), five reader roles, three draws, the full-inbox
isolated-reader design with price pinned per pair, `Qwen/Qwen3-4B-Instruct-2507` at the pinned
revision, cognition fields on (ADR 0007).

**Varies.** The treated message is the payload's calm or high-arousal rendering — the same two
texts as every meridian experiment. 59 × 5 × 2 × 3 = **1,770 readings**, a count fixed in advance.

## Outcomes, in the order they will be read

1. **Primary: position size.** Per (pair, role, arm), draws averaged; anxious minus calm per
   (pair, role); per-role mean with 95% interval; Holm across the five roles. Reported per role
   with the mean after the table, as always.
2. **The anchor check, before the primary is interpreted.** Per arm: the share of parsed readings
   whose action is SELL and whose size is within 0.005 of the recommended 0.15.
3. **Departure rate**, per arm: share of readings with action ≠ SELL. Analogue of E30's 0-of-360.
4. **Secondary: the belief fields.** Downside risk, confidence, expected return — same estimator
   as size, compared in direction against E30 (+ downside, − confidence, expected return null).
5. Parse health first, as run health: PR7 required 95% of readings to carry the fields; same floor.

## The decision rule, fixed now

| Outcome | What it licenses |
| --- | --- |
| Anchor check ≥ 95% in **both** arms and the size contrast's intervals contain zero | **Uninformative by anchor is not available here** — a 95% anchor with a null is the informative negative: *given room to shade a recommended position, readers copy it regardless of tone*. Reported as a finding about instruction-following, scoped to this model and event. |
| Any role's size contrast clear of zero after Holm | **Words move decisions** at that role. Reported per role; no role named as generally most responsive; the paper's action claim changes from "no room to appear" to a measured effect. |
| Parse floor missed (< 95% carrying fields) | Failed run health; everything descriptive only. |
| Belief fields: downside and confidence intervals clear of zero in E30's directions | The belief result replicates on a second event and the paper says so. Opposite signs or nulls are reported with equal weight; a null here at 59 pairs bounds the effect below E30's estimate. |

There is no outcome under which this run weakens an existing claim: the current paper claims no
action effect and a one-event belief effect. It can only sharpen the negative or replace it.

## Sensitivity, computed before the run

Size is bounded in [0, 1] and recommended at 0.15. If readers behave as E30's did (exact
compliance), within-arm variance is ~0 and any nonzero contrast would be detectable at tiny
magnitudes; the risk is not power but the anchor, which outcome 2 measures directly. For the belief
fields, E30 detected +0.0227 (downside) at 24 pairs; 59 pairs gives roughly 1.6× tighter intervals,
so a same-size effect is comfortably detectable and a null is informative down to about half E30's
estimate.

## Cost

1,770 readings, no rewrite pass, roughly one hour. Queued behind E45 on the same GPU.

## What it cannot answer

- **The anchor is still in the message.** The recommendation and size are payload slots, identical
  across arms by design — that is the price of holding facts fixed. This run loosens the anchor
  (deviation is graded, not disobedience); it cannot remove it. A design with a non-directive
  recommendation slot is the next step if this one anchors at 95%.
- **Isolated readers, one event, one model.** Same limits as E30/E36.
- It does not connect team-level transmission to action — readers here receive the stimulus
  directly. The chain gap the Discussion names stays open either way.

---

## Amendment A, 2026-08-08 · The anchor size was a test fixture's value, not the scenario's

**Made while E46 was still queued behind E45 — no reading existed and none had been generated.**

The frozen analysis set the recommended size at 0.15, taken from the payload fixture in
`tests/test_payload.py`. The deposit-outflow scenario actually recommends **SELL at 0.50**
(`configs/scenarios/meridian_deposit_outflow.yaml`, confirmed against the rendered stimulus text).
With the wrong constant, the anchor check would have read a fully anchored run as 0% anchored and
issued the wrong verdict. The constant is corrected to 0.50; nothing else changed. Analysis hash
after the amendment: `cab108b4d901a965` (was `5df47e9610d35f9e`).

Caught while assembling the paper's appendix, because the appendix quotes the stimulus verbatim and
the verbatim text says 0.50. A constant that names a fact about an artefact should be read from the
artefact; the queue script's input checks verify files exist, not that constants match their
contents.
