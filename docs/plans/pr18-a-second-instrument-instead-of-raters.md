# PR18 · Ask a fear-specific instrument, instead of asking twelve people

- Governs E43. Analysis fixed and hashed **before it was run**, on data that **already existed**.
- Serves **soundness**. It attacks the limitation stated on every result in this project.
- **No GPU and no raters.** Re-scores the committed generations on CPU.

## Prior exposure, stated first

The generations are E36's, already analysed and
published, and now committed to `data/generations/`. **This is a post-hoc analysis of existing data.**
What is frozen is the rule: the outcome, the decision and the reading of every result are written here
before the script ran once. Cite it as a robustness check, not an independent finding.

## The problem it addresses

Every number here is scored with the NRC VAD lexicon, which scores *anxious* and *urgent* almost
identically. So every claim is about **lexical arousal**, and the word *anxiety* is not licensed. The
route planned for this was [PR10](pr10-do-people-read-it-as-anxiety.md), human ratings. Amendment A to
that plan already found its first population close to tautological, and the second needs people we do
not have.

There are two ways past the instrument that need nobody. This plan is the first: **use a second
instrument that separates the two constructs.** The other is to manipulate urgency independently of
affect, which is a separate experiment under its own plan.

## The instrument, and why it can do what the word list cannot

`SamLowe/roberta-base-go_emotions`, MIT, pinned at revision `d75048347613a25d77de8cf6412eaae9fa7b26be`.
A classifier over the 28 GoEmotions labels, trained on 58k Reddit comments annotated by people.

It has separate `fear` and `nervousness` labels. It has **no urgency label at all**, because urgency
is not an emotion. That asymmetry is the whole point: a measure with no way to express urgency cannot
mistake urgency for apprehension.

Three quantities, defined now:

| | |
| --- | --- |
| **anxiety** | `fear` + `nervousness` |
| **aroused-not-anxious** | `excitement` + `surprise` + `anger` + `annoyance` |
| **discriminant** | anxiety minus aroused-not-anxious |

The discriminant is the quantity the word list cannot produce. A rise in it cannot be explained by
text merely sounding more activated, because activation without apprehension is subtracted.

**It separates the constructs on clear cases.** On four probe sentences written for the check:

| | VAD arousal | classifier anxiety |
| --- | --- | --- |
| anxious minus calm | **+0.208** | **+1.037** |
| urgent minus routine | **+0.058** | **+0.000** |

Urgency moves the current instrument by 28% of what anxiety moves it. It moves the classifier by
nothing. That is a sanity check on hand-written sentences, not a validation, and it is reported as
one.

## What this is not

**It is not a replacement for people rating our text**, and no result from it may be worded as though
it were. Three limits, all stated before any number exists:

- The classifier is a **compressed human judgement**, trained on human labels. It is a different
  instrument from our word list, not an escape from human judgement, and its own errors are unknown
  on this text.
- **The training domain is Reddit, not financial reporting.** That shift is real and unmeasured here.
  A financial message that a person would read as measured concern may fall outside what the
  classifier saw.
- Findings are worded **"a classifier trained on human emotion labels says X"**, never "people say X".

## The question and the rule

Reader messages from E36, anxious arm minus calm arm, paired within stimulus pair, per role, draws
averaged to the cell first, exactly as every other analysis here aggregates.

**Primary.** Does the reader effect appear in the **discriminant**?

| Outcome | What it licenses |
| --- | --- |
| Discriminant positive, interval excluding zero, on the roles whose arousal effect was clear | **The propagating thing is apprehension, not activation in general.** The paper may then say that readers of a high-arousal message produce more apprehensive language, on a measure that cannot confuse it with urgency, and the word *anxiety-like* becomes defensible for the outcome with this instrument named. |
| Discriminant interval containing zero while anxiety alone is positive | Readers produce more apprehensive language, but not more than they produce other activated language. Weaker, and reported as such. |
| Discriminant interval containing zero and anxiety too | **The effect does not appear on this instrument.** Every claim stays at lexical arousal permanently, and this is reported as a failure of the arousal result to generalise across instruments, which is a more damaging result than a null and is written here so it cannot be quietly dropped. |
| Discriminant negative | Readers produce activated-but-not-apprehensive language. That would mean the arousal effect is closer to urgency or irritation than to anxiety, and the framing of the project changes. |

**Secondary, reported whatever the primary shows.**

1. **Per role**, with Holm across five. E36 and
   E40 both found roles differing enough to
   cancel, so no aggregate is reported without the per-role table beside it.
2. **Agreement between instruments.** Spearman correlation across the five roles between the arousal
   effect and the discriminant effect. Two instruments ranking the roles the same way is evidence
   about the roles; ranking them differently is evidence that role effects are instrument-dependent,
   which E40 already showed once and would be worth knowing twice.
3. **The full 28-label profile**, anxious minus calm, so what actually moved is visible rather than
   summarised. If `sadness` or `disappointment` moves more than `fear`, that is a fact about the
   effect and belongs in the write-up.

## Sensitivity

The classifier's outputs are probabilities in `[0,1]` and their paired differences have an unknown
spread on this text, so a detectable effect size cannot be stated in advance the way it can for the
lexicon. **The bootstrap interval on the primary is therefore what decides informativeness**, and a
discriminant interval that contains zero *and* an effect as large as the probe difference is reported
as uninformative rather than as absence.

## Reporting

E43 in `docs/experiments/`. Any number reaching a document is registered in
`docs/verified-numbers.toml` first, with the classifier and revision named in the estimator.
