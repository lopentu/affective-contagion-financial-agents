# PR22 · Does alarm cost the BUY too?

- Governs E47. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **soundness** first: it is the designed test that separates the two readings of
  E46, written the hour E46 landed and before its
  numbers travelled anywhere but its own entry.

## The question in one sentence

When the alarmed sender recommends BUYING, do readers buy less — and do their stated beliefs
contradict the alarm again, or follow it this time?

## Why it is worth the GPU time

E46 found the anxious rendering of a SELL recommendation costs it force: fewer sells, smaller
positions, and — against E30's directions — *lower* stated downside and *higher* confidence. Two
readings fit, and they disagree about a BUY event:

| Reading | Size on a BUY | Downside | Confidence |
| --- | --- | --- | --- |
| **Discounting** (the alarmed sender is credited less) | smaller | **lower** | **higher** |
| **Risk aversion** (alarm makes readers defensive) | smaller | **higher** | **lower** |

Both predict smaller positions, so the belief directions are the discriminator, and they are
opposite. A third outcome — sizes *larger* under alarm — would say E46's direction is not even
action-general, and that is worth knowing too. The BUY event's stimulus pairs already exist
(24 pairs, seeds 500–523, generated for the polarity check), so the run costs ~25 minutes.

## What is held fixed, and what varies

**Fixed.** The 24 matched harborline pairs (pairing-gate checked before any reading), five reader
roles, three draws, the full-inbox isolated-reader design with price pinned, cognition fields on,
`Qwen/Qwen3-4B-Instruct-2507` at the pinned revision. The event recommends **BUY at size 0.50** —
the same size as E46's SELL, so the two runs are symmetric.

**Varies.** The treated message is the payload's calm or high-arousal rendering.
24 × 5 × 2 × 3 = **720 readings**, fixed in advance.

## Outcomes, in the order they will be read

1. Parse health: ≥ 95% of readings carry the three fields, PR7's floor.
2. Anchor check per arm (action BUY, size within 0.005 of 0.50), read before the primary.
3. Departure rate per arm (action ≠ BUY).
4. **Primary: position size**, anxious − calm, per role, Holm across five.
5. **The discriminator: downside risk and confidence**, same estimator, directions read against the
   table above. `expected_return` is reported but E46 showed the field mixes units; it decides
   nothing.

## The decision rule, fixed now

Per role, only where the size contrast survives Holm; the run-level reading requires at least two
Holm-surviving roles agreeing:

| Size | Downside | Confidence | Verdict |
| --- | --- | --- | --- |
| negative | negative (clear of zero) | positive (clear of zero) | **DISCOUNTING supported** — alarm costs the recommendation its force regardless of direction |
| negative | positive (clear of zero) | negative (clear of zero) | **RISK AVERSION supported** — E46's belief signs were about the SELL, not the sender |
| negative | mixed or containing zero | mixed or containing zero | Sizes shrink, mechanism unresolved; report both readings as open |
| positive (clear of zero) | — | — | **Direction is event-specific**; E46's "away from the sender" does not generalise and the paper must scope it to one event |
| containing zero everywhere | — | — | Uninformative at 24 pairs unless the interval excludes E46's trader effect (−0.1034), in which case the BUY event responds less than the SELL event did |

## Sensitivity, computed before the run

24 pairs against E46's 59: intervals roughly 1.6× wider. E46's trader size effect (−0.1034, CI
half-width ±0.047) would show here with half-width ≈ ±0.075 — detectable. Effects the size of
E46's smaller roles (≈ −0.05) sit at the detection floor and may be uninformative per role; that is
why the run-level reading needs only two Holm-surviving roles, and why a null here is a bound, not
a refutation.

## Cost

720 readings, ~25 minutes, on a GPU that is currently idle.

## What it cannot answer

- One BUY event, one model, isolated readers; the causal chain to team-level action stays open.
- If neither belief pattern is clear, the mechanism question stays open — this run separates the
  two named readings; it does not enumerate all possible ones.
