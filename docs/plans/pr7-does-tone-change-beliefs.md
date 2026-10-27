# PR7 · Does anxious wording change what an agent believes?

- Written 2026-08-06, before any run with belief fields switched on existed. **Frozen on commit.**
- Governs E30, a pilot. A full run follows only if the pilot says beliefs move at all.

## The question

Every result so far is about words. An agent reads an anxious report and writes a more activated message;
its teammates do too. None of that says anything changed inside the agent.

Does an agent that reads an anxious version of identical facts hold a **different view** — a lower
expected return, less confidence, a larger downside?

This is the second research question, and it has never been measured. The fields for it were built in
ADR 0007 and have never once been switched on: across every
run ever done, including E27, `expected_return`, `confidence` and `downside_risk` are 0% populated.

## Why this comes before capability and memory

Propagation, model scale and memory are machinery for a claim that
has never been made. "Wording moves an arousal score by 0.015" is a linguistic observation. "An agent
reading an anxious version of the same facts expects a worse return" is the claim this project exists to
make.

It is also the cheapest thing in the queue: about three hours, against sixteen for the capability ladder
as originally scoped.

## What an exploratory look already suggests

The 1395 reader messages from E18 contain belief
vocabulary in 40% of cases, so they can be counted for free. Counting them:

| Word family | Anxious minus neutral, per message | Seeds positive |
| --- | --- | --- |
| risk, downside, exposure | **+0.168** [−0.030, +0.366] | 19/31 |
| confidence, certain, convinced | **−0.047** [−0.102, +0.007] | 5/31 |
| may, might, uncertain | +0.002 | 1/31 |
| expect, forecast, likely | −0.030 | 1/31 |

**This tests nothing.** The word families were chosen after reading the corpus, there are four
comparisons with no correction, and neither interval excludes zero. It is here because it points
somewhere: more risk language and less confidence language, which are the same direction substantively.
That is what makes a properly measured version worth three hours.

Word counting is a poor proxy anyway. The structured fields ask the agent directly.

## Design

| | |
| --- | --- |
| Base design | the isolated-reader setup from [PR3](pr3-second-hop-replication.md), which reached a 97% pairing rate |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Scenarios | **`calderon_mixed_guidance`**, whose recommendation is HOLD at size 0.00 |
| Seeds | 24, fresh: 400–423 |
| Arms | reader sees the neutral rendering, or the anxious rendering, of one identical report |
| Belief fields | **on** — this is their first use |
| Cost | ~3 GPU-hours |

**Why that scenario.** In every other scenario the report recommends an action, and
E16 found agents copy it: identical SELL at size 0.5 in 24 of 24.
An agent that is told what to do and does it tells us nothing. `calderon` recommends **HOLD at zero**, so
any trade at all is a visible departure, and its evidence is genuinely two-sided so the agent has room to
form a view.

## What is measured

Ranked before any data exists, and the ranking does not change afterwards. Holm correction across the
three.

1. **`expected_return`** — the belief the mechanism predicts should move.
2. **`downside_risk`** — the risk assessment named in the research question.
3. **`confidence`**.

Each reported as a paired difference with its interval, aggregated draws → reader → seed as in
[`clustered.py`](../../src/affective_contagion/clustered.py).

Reported alongside, not as the test: action and size, because an agent departing from a recommended
HOLD is worth seeing whichever way the belief fields go.

## The rule, fixed now

1. **Parse health.** At least 95% of decisions must carry all three fields as finite numbers. Below that,
   report a format failure and nothing else — the fields have never run, so this may simply not work.
2. **Pairing.** At least 80% of pairs pass `check_pairing`.
3. **Read the primary** with its Holm-corrected interval.
4. **State the direction.** Anxious wording should lower expected return and raise downside risk. A
   significant move the *other* way is reported as such and not as support.

**Identical actions with moved beliefs is a positive result**, not a failure, and this is written down in
advance so it cannot be reinterpreted later. The research question asks about beliefs, risk assessments,
*or* actions. A shift in what an agent thinks, with no shift in what it does, answers it.

## What counts as nothing

Sensitivity cannot be computed in advance: no run has ever produced these fields, so their spread is
unknown. That is stated rather than guessed. The pilot's job is to measure that spread and decide whether
a full run is worth it — which is why 24 seeds and one scenario, not 144 and four.

If the fields come back with no variation at all — every agent answering the same number — that is itself
the finding, and it means the grammar is forcing an answer rather than eliciting one.

## What this cannot support

- **Not four scenarios.** One payload, and a deliberately unusual one chosen because it recommends
  inaction.
- **Not a team.** Isolated readers, no market, no feedback.
- **Not anxiety.** The treatment is anxious wording; the word list behind every arousal number cannot
  separate anxiety from urgency.
- **Not behaviour at scale.** One model, and E19 found the sender
  effect fading as models grow.
