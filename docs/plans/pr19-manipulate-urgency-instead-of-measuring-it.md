# PR19 · Settle anxiety against urgency by manipulating both, not by measuring better

- Governs E44. Frozen 2026-08-08, **before any reading exists**. Analysis hashed before queueing.
- Serves **soundness**. It is the strongest available answer to the criticism, and it needs **no
  human raters**.

## The question in one sentence

Our word list cannot tell anxious from urgent. So instead of finding a better ruler, vary the two
things separately and see which one the reader actually responds to.

## Why this beats both a better instrument and a rating study

E43 took the measurement route: score
the same messages with a classifier that has fear and nervousness labels and no urgency label. It
worked, and it has a limit it cannot escape — it is still an instrument, and instruments can be
wrong about text unlike their training data.

This is the design route. If the reader's response tracks the sender's **affect** and not the
sender's **urgency**, no instrument needs to distinguish them, because the manipulation already has.
That is a causal answer rather than a measurement one, and it is the standard remedy when a measure
cannot separate two constructs.

It also costs nobody's afternoon. Twelve raters on eight items were never going to arrive.

## The design: a 2×2, and it is available because of a decision made months ago

Urgency is already a **payload slot** here, not part of the treatment.
`FinancialPayload.fact_block` puts "Review timing: ..." inside the block both arms embed verbatim,
and the docstring says why: varying urgency with the treatment would make the two collinear and RQ1
unanswerable.

That decision is exactly what makes this run possible. Urgency can be varied as its own factor,
orthogonal to affect.

| | routine timing | immediate timing |
| --- | --- | --- |
| **calm closing** | cell A | cell C |
| **anxious closing** | cell B | cell D |

- **Affect** varies in the closing sentence, self-referential as always: "I am composed and unworried"
  against "I am alarmed, anxious and frightened".
- **Urgency** varies in the review-timing slot: *"review at the next scheduled committee meeting"*
  against *"review before this session closes"*. Both are legitimate values of a payload fact. Neither
  says anything about the speaker's state.

Within the affect contrast the fact block is byte-identical, so the existing slot gate applies
unchanged. Across the urgency contrast exactly one slot differs, by design, and the gate is applied
per-contrast rather than across the whole square.

## What is held fixed

The same 59 matched stimulus pairs, five reader roles, `bare` context only, three draws,
`Qwen/Qwen3-4B-Instruct-2507` at the pinned revision. Bare rather than both contexts because the
square already quadruples the arms; E36 measured the context effect separately and it is not what
this asks.

59 pairs × 5 roles × 4 cells × 3 draws = **3,540 readings**, roughly ninety minutes.

## Outcomes

Both instruments, on every cell, because they answer different halves:

- **VAD arousal**, so the result sits on the same scale as every published number here.
- **The E43 discriminant**, apprehension minus activation-without-apprehension, which is the outcome
  that can distinguish the two constructs on the response as well as in the stimulus.

## The rule, fixed now

Per role and pooled, draws averaged to the cell first, contexts not applicable.

| | Contrast |
| --- | --- |
| **affect main effect** | (B + D) − (A + C) |
| **urgency main effect** | (C + D) − (A + B) |
| **interaction** | (D − C) − (B − A) |

The two main effects are separate questions and are corrected separately, not as one family.

| Outcome | What it licenses |
| --- | --- |
| Affect effect clear of zero, urgency effect containing zero | **The response is to the affect, not the urgency.** This is the result that retires the project's oldest caveat. The paper may say the effect is not an urgency effect, on a causal rather than a measurement basis. |
| Both clear of zero, affect larger | Both matter. Report both sizes; every claim carries "and urgency also raises it, by X". |
| Urgency effect comparable or larger | **We have been manipulating urgency and calling it affect.** The framing of the whole project changes. Written here first, because it is the outcome that would hurt most. |
| Both containing zero | The design failed to move the reader at all, which would contradict E36 on its own stimuli and means something is broken. Check the manipulation before reading anything else. |
| Interaction clear of zero | Affect and urgency are not additive: anxious wording matters more (or less) when the deadline is close. Interesting, reported, and not something any current claim depends on. |

**Per role always.** E36,
E40 and E43 have now each found role effects
that cancel or reorder. A main effect averaged over roles is reported only with its per-role table
beside it.

## Sensitivity, before the run

E36 measured the affect contrast at these pairs and draws with per-role intervals of roughly ±0.003
on arousal. This design has the same pairs and draws and twice the arms, so each main effect is
estimated from all 59 pairs at both levels of the other factor — comparable precision.

**The urgency manipulation's size is not known in advance**, and that is the honest gap. The probe in
PR18 put urgent-minus-routine at **+0.058** on VAD arousal in the *stimulus*, against +0.208 for
affect. If reader response is proportional to stimulus dose, an urgency effect around a quarter of the
affect effect would be expected under pure arousal transmission — roughly +0.005 pooled, which this
design can detect. **An urgency effect much below that will not be distinguishable from zero**, and a
null is then "smaller than about 0.005", never "absent".

## What it cannot answer

- **One urgency contrast.** "Next scheduled meeting" against "before this session closes" is one
  operationalisation. A stronger urgency manipulation might behave differently, and a null is about
  this contrast.
- **Urgency stays inside the fact block**, so it is stated rather than performed. The same criticism
  PR10's Amendment A made of the affect manipulation applies here: this varies *stated* urgency.
- It does not license the word *anxiety*. It rules out one confound. Sadness and disappointment,
  which E43 found moving more than fear, are untouched by it.

---

## Amendment A, 2026-08-08 · The first urgency wording could not have failed

**Made before any reading exists**, while building the renderer. No outcome has been seen.

The wording above — *"review before this session closes"* against *"review at the next scheduled
committee meeting"* — moves the stimulus's arousal by **+0.0001**. The affect contrast moves it by
+0.0383. A design in which one factor has three hundred times the dose of the other cannot find an
urgency effect, and the null it would produce would say nothing about urgency.

This is the same error caught in [PR10's Amendment A](pr10-do-people-read-it-as-anxiety.md): a test
that cannot fail is not evidence. It was invisible until the renderer was actually run, because the
review-timing slot is a few words inside a ninety-word fact block and arousal is a mean over tokens.

**The fix is to match the dose.** The comparison worth making is not "affect against whatever urgency
happens to score", it is **"affect against urgency at the same measured arousal"**. Then a reader
responding to one and not the other is a real dissociation rather than an artefact of one factor being
louder.

Revised urgency levels:

- **routine** — *"review at the next scheduled committee meeting, at the usual time"*
- **immediate** — *"urgent immediate review demanded now, before the deadline expires this session"*

Both are statements about review timing. Neither describes the speaker's state, and neither asserts an
outcome, a severity or a spread; the residue gate is unchanged and both levels pass it.

The square now looks like this to the two instruments:

| Main effect in the stimulus | VAD arousal | E43 discriminant |
| --- | --- | --- |
| **affect** | **+0.0353** | **+0.8966** |
| **urgency** | **+0.0309** | **+0.0014** |

**The two factors are within 13% of each other on the instrument that cannot tell them apart, and
utterly different on the one that can.** That is the design this run needs, and it is what makes the
primary contrast interpretable:

- Reader responds to affect and not urgency, **at matched arousal dose** → the response is to the
  affect. This is now a fair test rather than a foregone one.
- Reader responds to both about equally → reader arousal tracks stimulus arousal, whatever produced
  it, and our outcome measure is doing exactly what its critics say. That would be the most damaging
  outcome available, and matching the dose is what makes it possible to observe.

The sensitivity paragraph above is superseded. Under matched dose, **an urgency effect proportional to
its arousal dose would be about 87% of the affect effect** — for the trader, roughly +0.020 against
+0.0228. That is comfortably detectable at 59 pairs. An urgency effect below about 0.005 is not
distinguishable from zero, and a null is reported as "smaller than about a fifth of the affect
effect", never as absent.
