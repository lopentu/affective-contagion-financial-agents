# PR8 · Does the effect hold on a second financial event?

- Written 2026-08-06, before any run on a second scenario existed. **Frozen on commit.**
- **Amended 2026-08-06, before this plan produced a single run**, to add a dominance prediction. The
  amendment is recorded below with the reason it is legitimate.
- Governs E31.

## Why this exists, and why it was missing

Every headline number in this project comes from a **single financial event**: a bank deposit outflow at
a fictional lender. The sender effect (+0.0366), the teammate effect (+0.0150), and both of their
decision-vocabulary checks all used `meridian_deposit_outflow` and nothing else.

E27 did use four scenarios, but it measures a
different thing — a message *handed* to the agent rather than one the agent wrote. The belief pilot uses
one scenario. So nothing has ever tested whether the effects that carry the paper survive a change of
event.

If the effect is specific to that one event, the entire claim is an anecdote. Three further scenarios
have been shipped and validated since,
and have never been run. This is the cheapest remaining way to find out.

## Design

Identical to the design that produced E15 and E18, changing only the scenario.

| | |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Scenarios | **`harborline_capacity_award`** and **`pallasridge_covenant_note`** |
| Seeds | **exactly 24 per scenario**, fresh: 500–523 |
| Arms | neutral and anxious rendering of one identical payload, interleaved per seed |
| Outcomes | the sender's own message arousal, and the teammates' at shock+1 |
| Scrub | decision vocabulary removed, as in E15 and E18 |
| Cost | 96 runs ≈ **2 GPU-hours**, plus a reader pass if the sender effect holds |

**Why those two scenarios.** `harborline` inverts the polarity — good news, an upside forecast, a BUY
recommendation — so it tests whether the effect needs bad news to work. `pallasridge` keeps the direction
but drops the stakes by an order of magnitude, 0.5–1% over two quarters rather than 5–8% over a week. Between
them they vary the two things most likely to matter. `calderon` is held back because the belief pilot is
using it, and mixing would confuse which run produced what.

## What is measured

Per scenario, separately, never pooled with `meridian` and never pooled with each other:

1. **Sender effect** — the treated agent's own message arousal, anxious minus neutral, at the shock round.
2. **Teammate effect** — the untreated agents' arousal at shock+1.

Each reported as a paired difference by seed with a 95% interval, plus the count of seeds in the predicted
direction against `sign_test_threshold(n)`.

## Amendment: a dominance prediction, added before any data

E27 found, after the fact, that teammates reading
an anxious message write language scoring lower on **dominance** — the sense of being in control. That
matters because it is the one dimension separating anxiety from urgency: both raise arousal, only anxiety
lowers dominance.

That finding is exploratory. It was noticed after seeing the arousal result, on a dimension we had been
recording and not analysing, and it replicates only partly across designs. Left alone it stays a fishing
expedition.

**This plan has produced no runs, so the prediction can be registered rather than fished.** Adding it is
legitimate for one reason and it is worth stating precisely: nothing is being changed about an existing
rule after seeing a result. A *new* prediction is being written down before the data that would test it
exists. If it were the other way round — softening a threshold, or adding an outcome after a null — it
would be the outcome-triggered continuation this project has corrected once already.

**The prediction, fixed now:** in both scenarios, the treated agent's own message and its teammates'
messages will show **lower dominance** in the anxious arm, with 95% intervals excluding zero.

**Reported as a secondary**, never promoted over the arousal outcome, and reported whichever way it goes.
A failure here is informative: it would mean E27's dominance result does not generalise beyond an
injected message carrying explicit anxiety words, and that the anxiety-versus-urgency question is still
entirely open.

## The rule, fixed now

1. **Stimulus gate.** Both renderings must pass the payload check and differ in arousal in the intended
   direction, as every run already requires.
2. **Pairing gate.** At least 80% of pairs must pass `check_pairing`.
3. **Read each scenario on its own.** A scenario counts as replicating if its interval excludes zero on
   the positive side **and** its sign count reaches the threshold — the same two-part rule E15 and E18 used.

**What each outcome licenses, fixed in advance:**

| `harborline` | `pallasridge` | What may be said |
| --- | --- | --- |
| replicates | replicates | The effect is not specific to one event, and not specific to bad news |
| fails | replicates | The effect needs adverse content; the earlier results are about bearish framing rather than framing |
| replicates | fails | The effect needs material stakes; report the magnitude dependence |
| fails | fails | **Every earlier headline number is a single-scenario finding and must be reported that way** |

The last row is the one worth writing down in advance, because it is the outcome that would most change
what the paper says, and it is the one it would be most tempting to explain away afterwards.

## What counts as nothing

At 24 seeds and the spread measured in E15 (SD 0.0413), the smallest detectable sender effect is about
**0.017** — below the +0.0366 seen on `meridian`, so a real effect of that size would be caught. An effect
half that size would not be reliably detected, and a null at 24 seeds means "smaller than 0.017 or absent",
not "absent".

If a scenario returns a null, the honest report is that the effect did not replicate at this sample size,
with the floor stated. No seeds are added afterwards to chase it — that is the outcome-triggered
continuation this project has already corrected once.

## What this cannot support

- **Not generality.** Three scenarios total is not "financial language". It tests whether one anecdote is
  three anecdotes.
- **Not anxiety.** Same lexicon limitation as everywhere else.
- **Not behaviour.** Actions reported descriptively only.
- **Not a second model.** Qwen3-4B throughout.
