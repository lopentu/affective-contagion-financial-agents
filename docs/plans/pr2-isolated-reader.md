# PR2 · Do isolated readers of an anxious message write more aroused messages?

- Written: 2026-08-05, **before any isolated-reader generation existed**. Committed before the first run.
- Governs: E15, and any later entry citing an isolated-reader effect
- Status: **frozen**. Changes after the first run go in dated amendments, never edits.

This is the analysis plan written down before looking at the data. PR1 exists because E7's second-hop
number was +0.0195 (d = 1.33, 6/6) on round 5, −0.0058 on round 6, and +0.0069 pooled — three
defensible choices, and picking after the fact would have turned noise into a finding.

## Why this design and not the simulation

The simulation cannot answer this question, for three separate reasons established in E13 and E14.

1. **Power.** A 7-round run yields one number per arm, so 12 seeds give 12 paired observations. The
   peer effect looks like ~0.003 and 12 pairs detect no less than 0.014, so ~200 seeds would be needed
   ([PR1 A1.1](pr1-second-hop.md)). Here the unit is one generation, so the same GPU time buys
   hundreds of observations.
2. **The price channel.** In the simulation a peer sees the treated agent's message *and* a price its
   trade may have moved. Here price is an input held identical across arms, closing that channel by
   construction rather than testing it.
3. **Reproducibility.** A simulation pair must stay byte-identical for four rounds before the
   treatment lands, and intermittently does not (E13: 6/42 cells on one rerun; E14: the failure is
   environmental and comes and goes). Here there is no history, so nothing can desynchronise.

## Design (fixed)

Take the treated agent's shock-round message from an existing matched pair of runs, show it to one
untreated peer role in isolation, and record what that peer writes.

| | |
| --- | --- |
| Stimulus messages | The treated agent's shock-round message from **30 freshly generated matched pairs**, seeds 6-35, each verified with `check_pairing.py`. See amendment B1 |
| Reader roles | The **5 untreated roles**: fundamental_analyst, bull_researcher, bear_researcher, trader, risk_manager |
| Draws per (seed, role, arm) | **13** (see amendment B1 for the sample-size decision) |
| Total generations | 30 × 5 × 13 × 2 arms = **3,900** |
| Price and fundamental | **Identical across arms**, fixed at `fundamental_value` 100.0 and price 100.0 for every reading |
| Reader input | Exactly one inbox message (the stimulus) and no news; the reader never sees the report |
| Models | **Two arms of the same experiment**, see below |
| Instrument | NRC VAD v2.1, `signed`, `match=surface_then_suffix`, arousal |
| Decoding | constrained, temperature 0.7, `ModelSpec.deterministic` on |

### Two models, declared in advance

- **Qwen3-4B-Instruct-2507** @ `cdbee75f17c01a7cc42f958dc650907174af0554`, bfloat16. The pipeline model,
  where the first hop is established.
- **Qwen3.6-27B**, GGUF `UD-Q5_K_XL`, via a llama.cpp backend that does not yet exist.

Both are **co-primary**, and this is a decision about what a null would mean rather than a bonus
robustness run. E11's lesson was that a null can be a capability floor rather than an absence: Qwen2.5-3B
showed no first hop and Qwen3-4B showed a large one, on an identical design. If only 4B were tested and
its readers showed nothing, "isolated readers do not transmit affect" would be indistinguishable from
"not at 4B" — the E7 error repeating. Testing both makes a joint null informative and a split result a
capability finding.

If the 27B backend is not working when the 4B arm completes, the 4B arm is reported alone and **explicitly
labelled as unable to distinguish absence from a capability floor**. It does not become the headline by
default.

## Primary outcome (one, fixed)

**Arousal of the reader's own message.** Aggregated as specified in "Estimator", per model.

## Estimator (fixed)

The observations are nested — 13 draws inside a (seed, role) cell, and a cell's two arms read the *same
pair* of messages — so treating 780 draws as 780 independent observations would understate uncertainty
by roughly the square root of the cluster size. Implemented in `clustered.py`:

1. average the 13 draws within each (seed, role) cell, per arm;
2. difference the arms within each cell, removing any cell-level effect exactly;
3. average the 5 role differences within each seed, so each independently seeded message pair
   contributes one number;
4. report the paired mean over the 12 seeds, with a two-sided 95% t interval, df = 11.

**By seed is primary** because different seeds' message pairs really are independent, each coming from a
separately seeded run. A **by-cell** interval (n = 60) is also reported, and is *not* the headline: it
assumes roles reading the same message are independent, which they are not. The gap between the two is
itself reported, because it measures how much apparent precision would have come from double-counting.

Extra draws buy measurement precision, not degrees of freedom. A test pins this: the by-seed interval
must be unchanged whether each cell holds 2 draws or 20.

## Power, computed before running

The relevant variance is between *seeds*, since that is the primary unit. Two anchors from measured data:

- E11's peer-mean SD of paired differences was **0.0164**, on the matched runs. That is a within-simulation
  peer measure over 5 agents, the closest available analogue.
- Averaging 13 draws per cell and 5 roles per seed removes most measurement noise, so the by-seed SD
  should be **at or below** 0.0164. Taking 0.0164 as a conservative ceiling:

| | |
| --- | --- |
| Minimum detectable difference, n = 12, t(11) | **0.0146** (z-approximation: 0.0133) |
| PR1's "worth caring about" benchmark (25% of the first hop) | 0.0103 |
| Effect seen in the simulation's peers | ~0.003 |

**So 12 stimulus pairs are still not enough for the small effect, and this is stated before running
rather than discovered afterwards.** The seed count, not the draw count, is the binding constraint —
which is exactly the mistake PR1 made by adding seeds to a design whose unit was wrong.

**Therefore the design is extended now, before any data exists:** if the by-seed interval on 12 pairs
does not exclude zero, generate additional stimulus pairs by running further matched simulation seeds
(18–41, giving 30 pairs) and rerun the same frozen analysis. 30 pairs give an MDD of **0.0087**, below
the 0.0103 benchmark. This extension is pre-authorised and is **not** optional stopping: the analysis,
outcome, and decision rule are unchanged, only n increases, and the extension triggers on the interval
rather than on the effect's direction or size. The extended result supersedes the 12-pair result rather
than being reported alongside it as a second look.

## Decision rule (fixed)

Per model, transmission to an isolated reader is **supported** only if:

1. the by-seed 95% interval excludes zero on the positive side, and
2. at least `ceil(0.75 n)` seeds share the positive sign.

Otherwise **not supported**, and the achieved MDD must be reported. If the MDD exceeds 0.0103 the null
is **uninformative** and must say so — the PR1 A1.1 rule, carried forward. The word "rejected" is not
available without an equivalence test, which this plan does not include.

If the two models disagree, that is reported as a **capability-dependent effect**, not as one model
being wrong.

## Manipulation checks (fixed, and gating)

- **The stimulus pairs must actually differ in affect.** Report the paired arousal difference between the
  12 message pairs. This is E13's first hop and is known to be +0.0339, CI [+0.0110, +0.0567]; it is
  restated here because if it did not hold, no reader could receive anything.
- **Payload equivalence of what readers see.** PR1 A1.3 flagged that ADR 0003's matching guarantees cover
  the *stimulus*, not the treated agent's own message. Report, for each pair: the treated agent's action
  and size, and whether the anxious message contains digits, direction terms, or size recommendations
  absent from the neutral one. **If the pairs differ in payload, a reader effect may not be called
  affective** — it is transmission of information plus affect.
- **Isolated-reader recovery.** Report whether readers of the two arms produce the same action and size
  distribution. the study protocol requires this as manipulation validation, and this design is the first that can
  supply it.

## Exclusions (fixed)

- A reading whose message has **zero lexicon coverage** is excluded and counted; its arousal would be the
  neutral 0.5 default rather than a measurement.
- A cell that loses more than 3 of its 13 draws to exclusions drops that (seed, role) cell from both arms.
- A parse failure is **kept**, not excluded: the fallback text is what the agent actually produced and is
  scored as such. The parse-failure rate is reported per arm, and a rate differing by more than 5 points
  between arms is itself a finding about the treatment rather than a nuisance.
- No seed or role may be dropped after its outcome is seen.

## Secondary and exploratory (declared, no claims)

All exploratory; none may become a headline, and no multiplicity correction is applied.

- Per-role breakdown. Two role prompts explicitly instruct affect-discounting (`sentiment_analyst`,
  `risk_manager`), and one of the five readers is `risk_manager`, so heterogeneous receptivity is
  expected rather than surprising.
- Valence and dominance alongside arousal.
- Action and size distributions by arm (beyond the gating check above).
- An **action-token robustness score**: arousal recomputed with BUY/SELL/HOLD and a small closed list of
  direction terms removed. PR1 A2.1 requires this before any result is described as affective, because
  action words carry arousal and a reader switching HOLD to SELL moves arousal by about one MDD.
- Message length and lexicon coverage by arm.

## What is not being claimed, whatever the result

- **Not anxiety.** Per PR1 A2.1 the licensed reading is *lexical arousal coupling*. On NRC VAD, anxiety and
  urgency are near-identical (0.573 vs 0.555), and ADR 0004 records that the instrument cannot answer RQ1
  alone.
- **Not team dynamics.** One hop, one reader, no peers, no market, no feedback. This deliberately cannot
  speak to escalation, and a positive result here does not imply the simulation would show one.
- **Not the firewall.** RQ4 is untested; `ValidatedPayloadFirewall` is still unreachable from a config.
- **Not general across scenarios.** One scenario. Not across topologies — there is no topology here.
- The 27B arm is **quantized** (Q5_K_XL). A behavioural difference between models confounds scale,
  training generation, and quantization, and may not be attributed to scale alone.

---

# Amendment B1 · 2026-08-05, before any reading was generated

Two changes, both made before any isolated-reader generation existed, and both narrowing what the plan
allows rather than widening it.

## B1.1 The stimulus messages are regenerated, not reused

As written, the plan reused the treated agent's shock-round messages from E13. Those runs had broken
pairing: 1 of 12 pairs fully matched, and 99 of 288 pre-treatment generations were byte-identical. So a
stimulus pair differed between arms because of the report **and** because of accumulated nondeterministic
drift in the sender's own inbox. Reusing them would have meant every reader effect was measured against
stimuli that were not cleanly matched.

**Fixed:** stimuli come from 30 freshly generated matched pairs, seeds 6–35, each verified with
`scripts/check_pairing.py`. Any pair that diverged before the treatment landed is **excluded before any
reading is generated**, and the count of exclusions is reported. Seeds 6–17 of E13 are not reused.

## B1.2 Fixed sample size of 30 pairs; the two-stage extension is withdrawn

The plan pre-authorised testing at 12 pairs and extending to 30 if the interval included zero, arguing
this was not optional stopping because the trigger was the interval rather than the effect size. That
argument is wrong: testing at one sample size and then testing again at a larger one raises the
false-positive rate above the nominal 5% regardless of what triggers the second look, because there are
two opportunities to cross the same threshold.

**Fixed:** 30 pairs, one analysis, one test. No interim look at 12. This was affordable — the pairs cost
about an hour of GPU — and it removes the need to defend a sequential procedure at all. 30 pairs give a
smallest-spottable difference of **0.0087**, below the 0.0103 that PR2 set as worth detecting, so the
design is adequately powered before it runs rather than after.

Everything else in the plan — the outcome, the aggregation order, the decision rule, the exclusions, the
manipulation checks, the two models — is unchanged.

---

# Amendment B2 · 2026-08-05, before any reading was generated

## B2.1 The sign rule could never detect the effect the plan calls worth detecting — at any sample size

Carried over from PR1 A1.4: `positive_seeds >= ceil(0.75 n)`. Holding the *proportion* fixed imposes a
floor that no sample size removes. To get 75% of seeds positive the true effect must be at least
`Phi^-1(0.75) = 0.674` standard deviations, so with a spread of 0.0164 the smallest detectable effect is
**0.0111** — above the **0.0103** this plan set as worth detecting.

Worse, verified by simulation: the chance of passing the rule when the true effect is exactly 0.0103
**falls as the sample grows**, because the observed proportion converges to `Phi(0.0103/0.0164) = 0.735`,
which is below 0.75.

| pairs | chance of passing the sign rule at the benchmark effect |
| --- | --- |
| 12 | 0.605 |
| 30 | 0.441 |
| 120 | 0.399 |
| 1000 | **0.151** |

A rule that gets weaker with more evidence is broken. **Fixed:** the sign criterion is now an exact
one-sided binomial test at 0.05, which scales properly — 12 pairs need 10, 22 need 16, **30 need 20**,
60 need 37. PR1's rule is superseded for PR2 and flagged for PR1 should it ever be reused.

## B2.2 Thirteen draws per cell was the worst use of the budget

Extra draws shrink only measurement noise; they cannot touch the pair-to-pair spread, which is the term
that sets precision. That noise was measured directly from data already on disk — round 0 of the
E13 runs has byte-identical prompts across all 24 runs and differs only by seed — giving a per-draw
spread of **0.0162**. Under that, the best allocation is roughly **3 draws**, all **5 roles**, and as many
pairs as the budget allows.

**Fixed:** **3 draws** per (pair, role, arm), not 13. With 30 pairs and 5 roles that is 900 readings per
model instead of 3,900 — cheaper *and* more precise, and the freed budget pays for B2.4 and B2.5.

## B2.3 The reader's price was a state no run ever occupies

The plan pinned price at 100.0, giving `Deviation: +0.000%` in the prompt. No seed is at that value at
the round in question, and readers visibly anchor on that string. **Fixed:** use each pair's own real
post-shock price. E13 verified price is bit-identical across arms in all 12 seeds, so this keeps the
price channel closed *exactly* while removing an artificial state. Strictly better at no cost.

## B2.4 Add the no-communication arm that the study protocol requires

The plan had only neutral-versus-anxious, with no baseline for a reader that received nothing. Without it
we cannot say whether the anxious arm raises arousal or the neutral arm lowers it, and we have no
reference for how close readers sit to a ceiling. **Fixed:** a third arm with an empty inbox, same roles,
same draws. Affordable because of B2.2.

## B2.5 Report the diluted and undiluted contrast, and stop claiming fidelity

B1's code change already gives the reader the whole five-message inbox, which is what a real peer reads.
The single-message version is retained as a **declared exploratory contrast**, because the difference
between them measures how much dilution matters. But the claim in PR1 A1.5 that isolation "faithfully
reproduces what a round-5 peer sees" is **withdrawn**: the full-inbox arm reproduces the inbox, not the
round-5 state, since the reader has no history and a pinned price.

## B2.6 "Nothing can desynchronise" was false

Measured: at round 0 of the E13 runs, where prompts and generation seeds are byte-identical, **15 of 72
cells produced different text — 21% of single calls.** This design is immune to *cascading* divergence,
not to nondeterminism. **Fixed:** the claim is restated as "no cascading divergence", and the single-call
reproduction rate is measured as a pre-run check.

## B2.7 Each pair needs its own reader seed

Generation seeds come from `(run_seed, round_index, agent_index)`. With one `run_seed` across all pairs,
all 30 supposedly independent units would share the same draw seeds, correlating the noise the interval
assumes independent. **Fixed:** `run_seed` is a function of the stimulus seed, identical across arms
within a pair (so the arms share draw noise, which reduces variance) and distinct across pairs.

## B2.8 The payload gate must be checked before any reading, not after

On the old stimuli, **9 of 12 pairs failed the plan's own payload check** — spot-verified: seed 6 went
`sell` to `rising, sell`, seed 14 went `bearish` to `downside, rising, sell` and doubled in length. So the
plan was pre-committed to a verdict of "may not be called affective" before it ran. This is only partly a
contamination artefact: the treated agent rewords freely under a 60-word cap, so it may persist.

**Fixed:** the payload check runs on the regenerated 30 pairs and is **reported before any reading is
generated**. If most pairs still fail, the correct response is a slot-exact stimulus — hand-written or
firewall-regenerated matched messages — not a reader experiment on mismatched inputs. That decision is
made on the check, not on any outcome.

## B2.9 Smaller corrections

- **Size on the standardised effect, not an absolute threshold.** Dilution changes the scale of both the
  effect and its spread, so an absolute benchmark imported from the simulation is not comparable.
  Targets are stated as a fraction of the first hop's standardised effect (E11 1.41, E13 0.94).
- **The 27B arm is demoted to declared future work.** The backend does not exist, no revision is pinned,
  and llama.cpp uses a different grammar engine and sampler — so a split result would confound scale,
  training generation, quantization *and* backend, destroying the capability-floor logic that justified
  co-primacy. The capability-floor caveat on a single-model result stands. Replaced by B2.10.
- **A seed counts only if all 5 roles survive**, otherwise the seed is dropped — the estimator averages
  roles unweighted, so a seed with 4 roles is not comparable to one with 5.
- **Zero-coverage readings are reported both ways** rather than excluded, since excluding on an outcome is
  selection. A coverage difference between arms is itself a finding.
- **Sender index comes from the roster**, not hardcoded, so moving the treated role cannot silently break.
- **Stimuli are taken from `delivery.delivered_text`**, not `decision.message`. Identical under the
  identity intervention, different under the firewall this design will eventually test.
- **Provenance:** the 30 pairs are written to `outputs/runs/` with manifests, and the SHA-256 of all 60
  stimulus texts is recorded here before any reading is generated. They currently live only in a session
  scratch directory.

## B2.10 Replacing the 27B: a ladder of small models

The reason for a second model was never scale for its own sake — it was that a null on one model cannot be
distinguished from a capability floor. Several small, recent, well-supported models answer that better
than one large quantized one, because they run through the same transformers path with no backend change:

| Model | Params | bf16 VRAM | Released | Downloads | Role in the ladder |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B | ~8 GB | 2025-08 | 3.2M | baseline; the first hop is established here |
| `Qwen/Qwen3.5-2B` | 2.3B | ~4.5 GB | 2026-02 | 2.6M | small rung, current generation |
| `Qwen/Qwen3.5-4B` | 4.7B | ~9.3 GB | 2026-02 | 6.7M | **same size as the baseline, one generation newer** |
| `Qwen/Qwen3.5-9B` | 9.7B | ~19.3 GB | 2026-02 | 12.4M | large rung; fits 24 GB with little headroom |
| `google/gemma-4-E2B-it` | 5.1B | ~10 GB | 2026-03 | 4.2M | second family, small |
| `google/gemma-4-E4B-it` | 8.0B | ~16 GB | 2026-03 | 5.8M | second family, large |
| `ibm-granite/granite-4.1-3b` | 3.4B | ~7 GB | 2026-04 | 0.3M | third family, small; already downloaded |
| `ibm-granite/granite-4.1-8b` | 8.8B | ~18 GB | 2026-04 | 4.2M | third family, large; already downloaded |

This separates two things a single large model would have confounded:

- **Size, family held constant.** Qwen3.5 at 2B, 4B and 9B is a three-point ladder inside one current
  release, so a trend across it is about capacity rather than training recipe.
- **Generation, size held constant.** `Qwen3-4B-Instruct-2507` against `Qwen3.5-4B` are both roughly 4B
  from the same lab six months apart. If the effect appears in one and not the other at equal size, that
  is about the recipe, not capacity.
- **Family, size roughly held.** Three labs at a small and a large rung each.

Every model runs through the same transformers path, so no backend difference confounds the comparison -
which is the property the 27B GGUF arm could not have offered.

All Apache-2.0 except Gemma, which carries Google's Gemma terms. Nothing blocks publishing results.

Two earlier candidates were dropped after checking dates rather than reputation.
`microsoft/Phi-4-mini-instruct` is from 2025-02 with no general successor since, so it would age the
ladder rather than extend it. `allenai/Olmo-3-1025-7B` is open in weights *and* data, which is
attractive, but at 0.1M downloads it is far less exercised; it stays recorded as an optional addition if
open data becomes the point. `Qwen3.5-27B` needs about 56 GB at bfloat16 and does not fit, and
`gemma-4-12B` needs about 24 GB, which is why the 8 GB `E4B` is the Gemma large rung.

**Implementation note, checked before committing to the ladder.** `Qwen3_5ForConditionalGeneration` and
`Gemma4ForConditionalGeneration` are multimodal wrappers whose text layers nest under
`model.language_model.layers`, not the `model.layers` that `backends.huggingface._decoder_layers`
searches. This experiment needs no internal activations, so it is unaffected, but E12's layer-discovery
survey does not cover these checkpoints and the helper needs the extra path before any activation work
uses them. `Qwen3.5-4B` was verified to ship `chat_template.jinja`, so it is instruction-tuned despite
carrying no `-Instruct` suffix; the `-Base` variants in the same collection are not and must not be used.

---

# Amendment B3 · 2026-08-05, before any reading was generated

Growing the ladder from two models to eight created a multiplicity problem that B2.10 did not address.
Recorded and fixed before any data exists.

## B3.1 The problem

If every model receives its own supported / not-supported verdict, the chance of at least one false
"supported" somewhere in the ladder is:

| models | per-model rate | chance of at least one false positive |
| --- | --- | --- |
| 2 | 0.022 | 0.044 |
| **8** | 0.022 | **0.163** |
| 8 | 0.05 | 0.337 |

The 0.022 is the measured rate of the conjunction actually used here - a positive interval excluding
zero *and* the sign test. So roughly a one-in-six chance of a spurious hit that would have been written
up as a result.

## B3.2 Why Holm-Bonferroni is not the fix

Holm controls the chance of any false positive across a family of tests, and it is the right tool when
the question is "did at least one of these work?". That is not this question. Eight models are not eight
attempts at the same hypothesis; they are a designed ladder whose purpose is to distinguish a genuine
absence from a capability floor. That is a question about a *trend*, and Holm answers it by making every
rung harder to pass, which is both the wrong target and a needless loss of sensitivity.

## B3.3 The fix: one primary, seven descriptive, one trend test

- **Primary, named now:** `Qwen/Qwen3-4B-Instruct-2507`. The single confirmatory test, using the
  estimator and decision rule already frozen above. One test, one threshold, no correction required.
  It is chosen because it is the model where the first hop is established, so a reader null there is
  interpretable rather than ambiguous.
- **The other seven are descriptive.** Their effects and intervals are reported in full, but they carry
  **no supported / not-supported verdict**, and no claim of the form "at least one model showed the
  effect" may be made from them. Nothing needs correcting because nothing is being claimed per model.
- **One trend test:** Spearman rank correlation between each model's effect size and its parameter
  count, across all eight, one-sided, at 0.05. This is the test that actually addresses the
  capability-floor question, and it is a single number.
- **Declared fallback:** if per-model verdicts are ever wanted, Holm-Bonferroni at family-wise 0.05,
  comparing the sorted p-values against 0.05/8, 0.05/7 … 0.05/1. Written down now so the choice cannot
  be made after seeing which models looked good.

## B3.4 What each outcome then licenses

| Primary model | Trend across the ladder | Licensed statement |
| --- | --- | --- |
| supported | positive | readers take up the affect, and more capable models take up more of it |
| supported | flat or absent | readers take up the affect; no evidence it depends on capacity in this range |
| not supported | positive | consistent with a capability floor - the primary model may simply be too small |
| not supported | flat or absent | the strongest available evidence against reader take-up, and the only combination that licenses "no detectable effect" as a general statement about this range |

Note that only the last row supports a broad negative claim, and even then only across 2-10B parameters,
one scenario, and one instrument. The word "rejected" remains unavailable without an equivalence test.

---

# Amendment B4 · 2026-08-05, during the first run, before any result was examined

Three defects were found in the *running* code. Two were live bugs, fixed and the affected
output discarded; one is a limit of the design that must change how an arm is analysed. One further
claim was checked and rejected.

## B4.1 The no-communication arm was not silent (fixed, output discarded)

`read_in_isolation` inserted the treated sender's slot unconditionally, so an empty message still
arrived as *a message from the sentiment analyst containing nothing*. That is a different condition from
receiving nothing at all, and not the baseline the study protocol asks for. Worse, the arm emitted both a
`neutral` and an `anxiety_like` condition with identical empty content: **225 of 237 cells produced
byte-identical output**, so half the arm's GPU time was spent twice.

Fixed: an empty message contributes nothing to the inbox, matching the simulation, which drops empty
outbox entries. The silent arm now emits a single condition. The 942 readings generated before the fix
are discarded rather than analysed.

## B4.2 The treated sender's index was hardcoded to zero (fixed before it mattered)

`treated_index` was computed by a fallback that always returned 0. Correct today only because
`sentiment_analyst` happens to be the first role — and it would have silently mislabelled every reading
in the **treated-role sweep**, where the treated agent is deliberately not role 0. Fixed to read from the
roster.

## B4.3 The matched arm has one stimulus pair, not thirty — and the analysis must say so

This is not a bug; it is a fact about the design that the plan failed to state. There is one scenario, so
`matched_realizations` yields exactly **one** payload-matched pair, and substituting it for every seed
means all thirty "pairs" carry identical stimulus text. The by-seed estimator would then treat one
stimulus as thirty independent ones.

**Fixed, before any matched reading exists:**

- The matched arm's unit of analysis is the **reader**, not the stimulus: clustering is by role, with
  seed and draw both functioning as reader-side replication. Its interval is computed over 5 roles, not
  30 seeds.
- It may support the statement *"this particular pair of messages, differing only in a closing sentence,
  moves readers by X"*. It may **not** support any statement generalising across message content, because
  it contains no variation in content at all.
- It remains a bench test that bypasses the sender entirely, per the ADR 0003 discussion above.

The real arm is unaffected: its thirty stimulus pairs are genuinely distinct, because each is an agent's
own paraphrase.

## B4.4 Price was identical across arms

B2.3's "keeps the price channel closed exactly" was checked against all 28 pairs: **0 of 28** differ
in shock-round price. B2.3 stands as written.
