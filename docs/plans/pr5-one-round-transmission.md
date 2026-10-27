# PR5 · How much emotional wording reaches a teammate in one round

- Written 2026-08-06, before any run existed. **Frozen on commit.**
- Governs E27. Replaces PR4, which is withdrawn — see ADR 0008.
- Does not govern, modify, or pool with E15, E16 or E18.

## The question

An agent reads an anxious version of a report and writes a message. Its teammates read that message. How
much of the agent's emotional wording reaches them one round later?

```
dose      = the treated agent's own arousal, anxious minus neutral, at the shock round
peer_sum  = the sum of its five teammates' arousal differences one round later
transfer  = peer_sum / dose
```

`transfer = 1` means the teammates' combined response matched what the sender emitted. This is **not** the
conserved/damped/amplified question, which needs a separate plan and a fitted VAR.

## Why the message is injected rather than generated

The treated agent's message is **replaced**, after it has decided and traded, with either the neutral or
the anxious rendering of one identical payload. Everything else in the run is held fixed.

This buys two things nothing else does.

**The price channel is shut, by construction, at the round that matters.** Each agent's trade is computed
from its own decision before the delivery step runs, the outbox is swapped only after every agent has
spoken, and the market clears after that. So every trade at the shock round is identical across arms, and
the price the teammates see one round later is identical too. The only thing that differs is the text.
That removes the need for a no-communication arm at this horizon, which halves the runs and avoids the
precision cost of a second subtraction.

**The dose is set rather than estimated.** E15 found that a sender
given anxious input rewrites the *facts* in 21 of 28 pairs, so a generated message varies in content as
well as tone. An injected message cannot.

ADR 0003 considered and rejected output injection on
the grounds that neither governing document described it. That reasoning does not survive the
identification argument above, and ADR 0003 is amended rather than contradicted.

## Design

| | |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Topology | `full`, six agents |
| Rounds | 12, injection at round 8, measured at rounds 9, 10, 11 |
| Arms | **N** neutral rendering, **A** anxious rendering |
| Scenarios | all four shipped |
| Units | **144** matched A/N pairs: 4 scenarios × 36 seeds |
| Treated role | balanced — each of the six roles is the sender for 6 seeds per scenario |
| Cost | 288 runs × 72 generations ≈ **8 GPU-hours** at the measured 1.4 s/generation |

## Sensitivity, computed before running

Measured on 62 existing matched pairs: the teammate sum one round after the shock has **SD 0.129**.

| Units | Smallest detectable `peer_sum` | Smallest detectable `transfer` |
| --- | --- | --- |
| 48 | 0.0374 | **1.02** |
| 96 | 0.0265 | 0.72 |
| **144** | **0.0216** | **0.59** |

144 units is the sample size, and its limit is stated plainly: **a transfer below about 0.6 will not be
distinguished from zero.** If the answer is that a little gets through, this design will report a null,
and that null will be labelled uninformative rather than negative. Reaching 0.4 would need roughly 300
units and about 17 GPU-hours; that is the next run, not this one.

**Prior exposure.** Those 62 pairs come from the E15 and E18 stimulus batches. I used them to get the
spread, and in doing so I saw their mean. They are one scenario, they lack the injection, and **none of
their seeds is used here**. E27 runs seeds 300–335.

## Analysis, fixed now

- One arousal value per agent per round, decision vocabulary removed by `scrub.scrub_decision_vocabulary`.
- `peer_sum` is the sum over the five untreated agents at round 9 only.
- Aggregate draws → agent → seed, then bootstrap over scenarios and seeds with
  `gain.clustered_bootstrap_interval` in `stratified` mode: 2000 resamples, percentile, every scenario
  present in every resample because the four scenarios are the fixed set being averaged over.
- Non-finite resamples are dropped and counted. Below 95% usable, no ratio is reported.
- Rounds 10 and 11 are reported as a curve and are **descriptive only**: trades diverge after round 9, so
  price is open again and those rounds mix text with price.

## Decision rule

In order. Any step can stop the inference.

1. **Injection check.** Every injected message must carry its intended payload, and the anxious rendering
   must score higher on arousal than the neutral one. Otherwise: stimulus failure, stop.
2. **Pairing check.** At least 80% of pairs must pass `check_pairing`.
   E18 ran at 97%, so this is achievable. Below it, report
   a run-health failure and read nothing.
3. **Direction.** If `dose` and `peer_sum` point opposite ways, or either interval sits wholly below zero,
   report a counter-response and stop.
4. **Dose.** `dose` must be wholly above zero and at least 0.010.
5. **Transmission.** If the `peer_sum` interval includes zero, report no detectable transmission together
   with the 0.0216 floor, and label it uninformative rather than negative.
6. **Only then** report `transfer` with its interval.

## Memory

Agents keep nothing between rounds (ADR 0009). That is a
defect and it will be fixed, but it does not touch this measurement: the hop measured here runs outward
from an injected message to its readers, and needs no memory on anyone's part.

It does bound the horizon curve. Rounds 10 and 11 show decay driven by price and inbox alone, because no
agent can carry its own affect forward. Read them as that, and not as persistence.

## What this cannot support

- **Not the trichotomy.** No output may describe `transfer` as conserved, damped or amplified.
- **Not anxiety.** NRC VAD scores anxiety and urgency almost identically, so the reading is lexical
  arousal (ADR 0004).
- **Not behaviour.** Trades are reported descriptively. E16 found
  identical SELL 0.5 in 24 of 24, and one scenario here recommends HOLD at size 0.00 precisely so a
  departure would be visible.
- **Not general.** Four scenarios, one model, one topology.
- **Not the firewall.** RQ4 needs a third arm and the payload-preservation check, neither of which is
  ready. It is a separate run under this same design.
