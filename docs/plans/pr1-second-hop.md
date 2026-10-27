# PR1 · Second-hop transmission to untreated peers

- Written: 2026-08-05, before any run in the seed range below existed. Committed with Amendment A1;
  see A1 for the precise ordering, which is weaker than "committed before the first run".
- Governs: E13 (and any later entry citing a second-hop effect on this design)
- Status: **frozen**. Changes to this file after the first run of E13 must be recorded as amendments
  with dates, never as edits.

## Why this exists

the study protocol forbids retrofitting hypotheses after examining treatment outcomes, and
E7 already shows why: its second-hop aggregate moved from
+0.0195 (d = 1.33, 6/6) on round 5 alone to −0.0058 on round 6 alone. With no window specified in
advance, either number could have been reported as the finding. Both were withdrawn. This document
fixes every analysis choice so that E13's result cannot be selected after the fact.

## Prior-exposure disclosure

I have seen the round-5 second-hop number on seeds 0–5 (E11, since withdrawn). The window choice below
is therefore **not** blind, even though it is justified on mechanism. The mitigation is that E13 runs on
**seeds 6–17, which have never been executed on this design**. E13 is accordingly an out-of-sample test
of a hypothesis formed on seeds 0–5, and must be described that way, not as a plan written blind.
Seeds 0–5 are excluded from the primary analysis for this reason.

## Design (fixed)

Identical to E11 in every respect except the seed range.

| | |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ pinned revision in `configs/models/qwen3_4b_shared.yaml` |
| Agents | 6, roles as in `configs/experiments/debug_affective.yaml` |
| Topology | `full` |
| Rounds | 7, shock at round 4, `shock_duration: 1` |
| Treated role | `sentiment_analyst` (single treated sender) |
| Arms | `neutral` vs `affective`, payload-matched per ADR 0003 |
| Seeds | **6–17** (12 seeds), paired across arms |
| Decoding | constrained, `constrained_decision_format: true` |
| Instrument | NRC VAD v2.1, `signed` scale, `match=surface_then_suffix`, arousal |

## Primary outcome (one, fixed)

**Mean arousal of the five untreated agents' messages at round 5**, averaged within a run, then paired
across arms by seed.

Round 5 and not 4–6 pooled, for a mechanical reason: the treated agent writes its message at round 4,
peers read only the *previous* round's messages, so round 5 is the first and only round in which a
peer's message can reflect the treated message through exactly **one** hop. Round 6 mixes that with
peer-to-peer re-transmission and is not a single-hop quantity.

Round 5 excludes the treated agent itself. Its own round-5 message is not part of this outcome.

### The primary outcome is not purely linguistic, and this is the main threat to it

Verified in `simulation.py:276-345`: within a round every agent sees `outbox` (the *previous* round's
messages) and `market.price`; at the end of the round `outbox` is replaced and `market.clear(orders)`
updates price. So a round-5 peer message can be influenced by the treatment through **two** channels,
not one:

1. **language** — the treated agent's round-4 message, which is the channel of interest; and
2. **price** — the treated agent's round-4 *trade*, which moved price before peers observed it.

A price-mediated effect is real transmission but not *affective* transmission, and the research
question is about language. The design cannot separate them by construction, so the following check is
fixed in advance rather than left to interpretation:

- Report the paired difference in **round-4 order imbalance** and in **price after round-4 clearing**
  between arms, with the same estimator and interval as the primary outcome.
- If either interval excludes zero, the primary result is **confounded** and must be reported as
  "transmission through an unseparated language-or-price path", not as affective transmission.
- If both intervals contain zero, the price channel is not detectably active and the linguistic
  reading is *permitted but not proven* — absence of a detectable price difference at n=12 is not
  proof of no price effect, and the achieved MDD on imbalance must be reported alongside.

Prior expectation, recorded now: E9 measured a single
agent's trade moving price 0.167% against 0.3% per-round noise, so the price channel is expected to sit
below noise. That is a prediction this check can falsify, not an assumption it may rely on.

A clean separation requires a design where peers cannot observe price, or where the treated agent is
barred from trading in the shock round. Neither is in scope here; both are recorded as follow-ups.

## Estimator (fixed)

Paired on seed, as everywhere in this project: seed dominates run-to-run variance far more than
condition does.

- Point estimate: paired mean difference, affective − neutral
- Uncertainty: two-sided 95% t interval on the paired mean, df = 11
- Standardised: Cohen's d_z
- Sign consistency: count of seeds with a positive difference, out of 12

## Decision rule (fixed)

Second-hop transmission is **supported** only if both hold:

1. the 95% interval excludes zero, and
2. at least **9 of 12** seeds share the positive sign.

Anything else is **not supported**. In that case the achieved minimum detectable difference must be
reported alongside, because a null is uninterpretable without it: if the MDD exceeds the first-hop
separation measured in the same runs, the design could not have detected full onward transmission and
the result is "underpowered", not "absent". Neither outcome may be described with the word "rejected"
absent an equivalence test, which this plan does not include.

## Secondary and exploratory (declared in advance, no claims)

All of the following are **exploratory**. No multiplicity correction is applied, and none may be
promoted to a headline finding.

- First hop: treated agent's own round-4 message, same estimator. This is an out-of-sample replication
  of E11 and is the check that the manipulation reached the channel at all in these seeds.
- Round 6 peers (compounded transmission).
- Per-role breakdown of the round-5 peer effect.
- Valence and dominance alongside arousal.
- Parse failure rate, HOLD rate, and price path.

If the first hop fails to replicate in seeds 6–17, the primary outcome is **uninterpretable** rather
than negative, and must be reported as such: peers cannot receive a manipulation the sender never
transmitted.

## Exclusions (fixed)

- A run with a parse failure rate above **0.10** is excluded. Its seed is then dropped from **both**
  arms, to keep the design paired. The count and the seeds dropped must be reported.
- A run that fails to complete is rerun once with an identical config; if it fails again its seed is
  dropped from both arms and recorded as a failure.
- No other exclusion is permitted. In particular, no seed may be dropped after its outcome is seen.

## What is not being claimed regardless of outcome

- Not causal evidence about human emotion, and not a claim that any model feels anything.
- Not generalisable across models: one checkpoint. Not across scenarios: one scenario. Not across
  topologies: `full` only.
- Under `full` topology a peer's round-5 message reflects all five other agents, so this measures
  transmission in a fully connected group, not a single sender's isolated influence. A per-edge
  claim requires `chain` and is out of scope here.
- Nothing about market instability. The market is weakly coupled at this calibration
  (E9).

---

# Amendment A1 · 2026-08-05, before any outcome was examined

**Ordering, stated precisely.** The text above was written before any seed 6–17 run existed, but the
file was **not committed** before the runs launched — it was untracked in the working tree while the
first cells executed. It is committed with this amendment. No outcome from any seed 6–17 run had been
read when this amendment was written; the numbers below all come from E11's seeds 0–5. I cannot claim
the stronger property of a commit that precedes the first run, and do not.

## A1.1 The primary outcome is underpowered, and the original null-handling rule concealed it

The plan fixed n=12 without stating an expected SD. That SD is recoverable from data I had already
seen: E11's round-5 peer cell for this model is **+0.0032 at d_z +0.20**, implying an SD of paired
peer-mean differences of **0.0160**. Therefore:

| | |
| --- | --- |
| SD of paired peer-mean differences (from E11 seeds 0–5) | 0.0160 |
| Minimum detectable difference at n=12, t(11) | **0.0142** |
| Effect actually observed on seeds 0–5 | **+0.0032** |
| Power at n=12 against +0.0032 | **≈8%** |
| Seeds required to detect +0.0032 | **≈196** |

The original rule said a null is "underpowered" only if the MDD exceeds the first-hop separation
(0.0411). Since 0.0142 < 0.0411 that condition can never fire, so the plan would have reported a null
with the sentence "the design could detect no less than 0.0142" — technically true, and misleading in
exactly the way the E7 title was misleading. Zero attenuation across a hop is not a plausible
alternative; the plausible alternative is the exposure sample's own 0.0032, against which this design
is blind.

**Amended rule.** A second benchmark is fixed now: **25% onward transmission = 0.0103**. A null must be
reported as uninformative unless the achieved MDD is below that. At the expected SD it will not be, so:

**E13's primary outcome is declared underpowered in advance.** E13 is therefore repositioned, before
its outcomes are read, from a test of second-hop transmission to:

1. a well-powered **out-of-sample replication of the first hop** (MDD 0.0235 against an expected
   0.0411 — this part is adequately powered and remains a real test);
2. a **fresh variance estimate** for the peer-mean measure, which is what can size a real test; and
3. the **price-channel and payload checks** below, which are diagnostic and do not need this power.

A confirmatory second-hop test needs a different design, not more seeds at this one. The isolated-reader
design in A1.5 is the intended route and is recorded as the next experiment.

## A1.2 The price-channel check was the wrong instrument

The original check tested the paired mean of shock-round imbalance and price for exclusion of zero.
That is wrong for a quantity that is **deterministic within a pair**: verified against E11's runs, the
two arms are byte-identical before the shock and their round-4 *peer* messages are byte-identical, so
market noise differences out exactly and any price difference is entirely attributable to the treated
agent's own trade. A mean test can pass while a third of pairs each carried an open non-linguistic
channel. E9's "impact is below a typical noise draw" does **not** license ignoring it here, because that
comparison is across runs and this design shares the noise stream. Citing E9 for this was an error.

**Amended checks**, all fixed now:

- Report the **per-seed** difference in shock-round order imbalance and price, and the **count of seeds
  where it is non-zero**. Report the counts, not only a mean test.
- Pre-specify a **sensitivity analysis on the clean subset**: the seeds where the treated agent's
  shock-round action *and* size match across arms. On that subset price is bit-identical, so the peer
  difference is purely linguistic. This is the cleanest estimate the design admits.
- Add a **placebo**: the treated agent's own round-5 message. Under `full` topology it cannot read its
  own round-4 message, so its round-5 prompt differs across arms *only* through price. Any arousal
  difference there is a pure price-channel effect with no linguistic component. A placebo as large as
  the peer effect invalidates the linguistic reading.

## A1.3 What peers receive is not payload-matched, and the plan must say so

ADR 0003's matching guarantees apply to the **stimulus**. They do not apply to the treated agent's own
round-4 message, which is what peers actually read. That message may differ across arms in forecast
strength, stated confidence, or recommended size, not only in tone. If it does, a peer difference is
transmission of *information plus affect*.

**Fixed now:** report the treated agent's shock-round action and size by arm and seed, and apply a
residue check to its round-4 message analogous to `payload.matched_realizations` — digits, direction
terms, and size recommendations present in one arm and not the other. If the round-4 messages differ in
payload, the peer result may **not** be called affective transmission, exactly as with price.

## A1.4 Instrument and measurement-cell defects

- **Action words carry arousal.** ADR 0004 already records that VAD cannot separate anxiety from
  urgency. A peer switching HOLD→SELL moves its arousal by roughly the whole MDD. Fixed now: report a
  robustness score with BUY/SELL/HOLD and a small closed list of direction tokens removed, plus the
  top contributing tokens per arm. Without it a positive result cannot be attributed to affect.
- **Zero lexicon coverage silently scores 0.5**, which sits inside the range peer messages occupy, so
  one unscoreable message among five shifts the peer mean by about the entire MDD. Fixed now: report
  mean and minimum coverage per cell. `second_hop.py` refuses a zero-coverage message outright.
- **The parse-failure rule was at the wrong granularity.** A run-level 0.10 ceiling over 42 records
  permits four failures, while a single failure at the treated agent in round 4, or at any peer in
  round 5, corrupts the primary outcome directly — `fallback_hold` both scores and *delivers* raw text.
  Fixed now: **any** parse failure in a measurement cell — (shock round, treated) or (shock+1, any
  peer) — drops that seed from both arms. E7 and E11 report 0.000 failures, so this is expected to cost
  nothing.
- **The sign rule is restated as a proportion for every reachable n**, because the code computed
  `ceil(0.75n)` while the text said "9 of 12"; these agree at n=12 and diverge after an exclusion. The
  rule is `positive_seeds >= ceil(0.75 * n)`. Noted for the record: 9/12 is *not* a 0.05 sign test
  (one-sided p = 0.073); it is an outlier-robustness requirement layered on the interval, and is
  described as such rather than as a significance threshold.
- **A directional reversal now has its own verdict.** The original rule reported peers *damping* under
  the affective arm — an interval excluding zero on the negative side — as "failure to detect", which
  would have been a false statement. `second_hop.py` gains a REVERSED verdict.

## A1.5 Disclosure widened, and the next experiment named

The original disclosure covered only the round-5 window. The accurate statement is that **everything
except the seed range was chosen on seeds 0–5**: the model (Qwen3-4B was selected *because* E11 showed a
first hop), the instrument and channel (arousal rather than valence or dominance), the treated role, and
the one-sided direction. Only the seed range is fresh. The window choice is the least of these, and is
in any case nearly forced — with 7 rounds and a shock at 4, round 5 is the only single-hop round.

The inconsistency this amendment removes: seeds 0–5 were consulted to fix the window but not to size the
study, even though they yield the SD that shows the design cannot test its primary outcome. Using the
exposure for the choice that makes the plan look pre-specified while ignoring it for the calculation that
undermines it is the residual bias. A1.1 uses it for both.

**Next experiment (not part of E13).** An isolated-reader design: take each seed's treated round-4
message from both arms, present it to each peer role with price held **identical** across arms, and
sample many independent generations. Because agents are stateless, this faithfully reproduces what a
round-5 peer sees, closes the price channel *by construction* rather than by test, directly tests
payload equivalence by checking whether isolated readers recover the same forecast and recommendation,
and replaces 12 run-level pairs with hundreds of independent draws — the only realistic route to power
against an effect near 0.003. the study protocol already requires an isolated-reader check as manipulation
validation, so this is owed regardless.

---

# Amendment A2 · 2026-08-05, before any outcome was examined

## A2.1 A supported result is lexical arousal coupling, not demonstrated anxiety transmission

ADR 0004 already records that on NRC VAD anxiety and urgency are near-identical
(arousal 0.573 vs 0.555), and that the instrument **cannot** answer RQ1 alone. Messages are capped at
60 words, and arousal is a mean over matched tokens, so a peer switching HOLD to SELL moves its arousal
by roughly the entire minimum detectable difference — action words carry arousal.

**Fixed now:** whatever this experiment shows, the licensed description is *lexical arousal coupling
between agents*. "Affective contagion", "anxiety transmission", and any phrasing implying the
construct rather than the measure are forbidden in reporting E13, in the index row, and in any
manuscript text citing it, unless the A1.4 action-token robustness score and the payload check in A1.3
both hold. This is a reporting constraint, not a caveat to be added later.

## A2.2 The estimand is group transmission, not a sender-to-peer edge

Under `full` topology a round-5 peer message reflects all five other agents. The plan said this in its
"not claimed" section while the rest of the document spoke of "second-hop transmission" as though it
were one edge. **Fixed now:** the quantity is *transmission within a fully connected group of six*. A
per-edge claim requires the `chain` topology and is not in scope. The phrase "second hop" is retained
only as a shorthand for the round index.

## A2.3 The first-hop gate is deliberately asymmetric with the primary criteria

The gate applies the interval alone, not the sign rule that the primary outcome must also satisfy. That
asymmetry is intentional and is frozen here rather than left to be noticed later: the gate exists only
to establish that *some* manipulation entered the channel, so requiring the stricter outlier-robustness
condition of it would discard runs whose sender demonstrably transmitted. It is a precondition, not a
finding.

## A2.4 Imbalance is latent; only price is observed

The plan's prose said "peers observe price" in a sentence that also discussed order imbalance. Peers
never see imbalance — it is not in `GenerationContext`, and it reaches them only through price. The
distinction matters because it identifies price as the sole non-linguistic channel, so the placebo in
A1.2 is a complete test of that channel rather than a partial one.

## A2.5 What A1 and A2 together mean for what E13 can produce

Recorded now so that reading the outcomes cannot change it:

| Outcome | Status | May be reported as |
| --- | --- | --- |
| First hop on seeds 6–17 | adequately powered (MDD 0.0235 vs expected 0.0411) | a genuine out-of-sample replication or failure to replicate |
| Peer-mean arousal at shock+1 | **declared underpowered in advance** | a variance estimate; a null here is uninformative and must say so |
| Price/imbalance per-seed counts, clean subset, placebo | diagnostic | whether the non-linguistic channel was open |
| Coverage, parse health, HOLD rate | diagnostic | run health |
| Anything about anxiety as a construct | not measured | nothing |
