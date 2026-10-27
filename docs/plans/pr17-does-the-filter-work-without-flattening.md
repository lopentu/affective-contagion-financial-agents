# PR17 · Does a filter remove the tone without removing the message?

- Governs E41. Frozen 2026-08-08, **before any reading existed**. The analysis script is hashed
  before the run is queued and the hash recorded in the entry.
- Serves **excitement** first and **soundness** second. It is the one contribution here that is
  useful to someone building a system rather than only to someone evaluating one.

## The question in one sentence

If an agent writes a message in a tense register, and we pass that message through a model asked to
rewrite it neutrally while keeping every fact, does the next agent stop picking up the tone — and
does the message still say what it said?

## Why it is worth the GPU time

**This is the mitigation the project has promised since the blueprint and has never run.** It is the
largest hole in the work. Both entry points list it. RQ4 has
no evidence at all behind it, of any kind, and that is the only research question in that position.

The reason it stayed unrun was defensible and is now spent. The argument was that a mitigation on an
unvalidated construct answers a question nobody has accepted we can ask. But
E40 has since ruled out the copying explanation
for the effect the filter would target, and the reader effect itself replicates across designs. There
is now something real to filter.

## What is held fixed, and what varies

**Fixed.** The same 59 matched stimulus pairs, the same five reader roles, both contexts, three
draws, the same lexicon, scrub and aggregation as E36
and E40, so the numbers sit on the same scale as everything else here.

**Varies.** One arm is added. Readers see one of three texts:

| Arm | What it is |
| --- | --- |
| `neutral` | the calm rendering of the payload |
| `anxiety_like` | the high-arousal rendering of the same payload |
| `filtered` | the high-arousal rendering, rewritten by the model with the neutralising instruction |

**The filter is the same model the agents use**, not a stronger one brought in from outside. A filter
is a component of the system being studied. Using a larger model to clean up a smaller one's output
would answer a different and easier question, and the result would not transfer to anyone running a
homogeneous team.

## The gate that decides whether any of this means anything

A filter that reduces the reader's response **by deleting the bad news** is not a filter. It would
look identical in the outcome measure, so it has to be excluded before the outcome is computed.

Every rewrite is checked against the payload before any reader sees it. It must retain every number
in the fact block, and the asset, observed event, forecast direction, expected magnitude, confidence,
time horizon and recommended action, each verbatim up to case and spacing. A rewrite that fails is
written to the rewrite log with what it dropped, and **its whole pair is excluded from every arm** —
not only from the filtered arm, because dropping a pair from one arm would leave the others comparing
different sets of pairs.

**The preservation rate is reported before any effect, and it is itself a result.** A filter that
keeps the payload in 60% of messages is a different object from one that keeps it in 99%, and the
second number is the one that would make this deployable.

**If fewer than 40 of the 59 rewrites pass, the run is reported as a failed filter rather than as a
mitigation result.** That threshold is fixed now: 40 pairs is roughly where the design retains enough
precision to separate the two contrasts below, and a filter that fails a third of the time has
answered the practical question in the negative regardless of what the survivors do.

The check is mechanical, and its limit is stated rather than discovered later: a rewrite that keeps
every number while reversing the meaning would pass it. Bidirectional entailment supersedes this when
it exists.

## Two contrasts, and they must not be collapsed

Conflating these is how a mitigation result becomes meaningless, so both are primary and both are
reported whatever they show.

**1. Does the filter reduce what the reader writes?** `filtered` minus `anxiety_like`, per role,
paired within pair. Negative means the filter removed some of the effect.

**2. Does it land at the calm baseline, or past it?** `filtered` minus `neutral`, per role. Near zero
means the filter neutralised the message. **Clearly negative means the filter flattened it** — it
produced something duller than a calm human-equivalent rendering of the same facts, which is a cost
and not a success.

## The decision rule, fixed now

Per role, then Holm across the five roles within each contrast. The two contrasts are separate
questions, not one family, and are corrected separately.

| Outcome | What it licenses |
| --- | --- |
| Contrast 1 negative, contrast 2 containing zero | **The filter works.** It removes the tone and lands where a calm rendering lands. This is the strongest available result and the one the paper would lead the mitigation section with. |
| Contrast 1 negative, contrast 2 clearly negative | **The filter over-corrects.** It works and it costs something. Report the size of the flattening; do not call it a success without it. |
| Contrast 1 containing zero | **No detectable reduction.** With preservation healthy this is a real negative about this filter, not about filtering; a rewrite instruction is one implementation of an idea. |
| Preservation below 40 of 59 | **Failed filter.** Reported as such. Contrasts are descriptive only. |

Role heterogeneity applies here as everywhere: E36 and E40 both found roles differing enough to
cancel, so **a filter effect averaged over roles is not reported without the per-role table beside
it**. If the filter helps three roles and harms two, the average is the least informative number in
the study.

## Sensitivity, computed before the run

E36 measured per-role effects on these pairs with intervals of roughly ±0.003 to ±0.004 at 59 pairs
and three draws. The quantity here is a difference between two arms measured on the same pairs, so
its precision is comparable, with the caveat that the filtered arm carries the extra variance of its
own rewrite generation.

The `anxiety_like` minus `neutral` effect this filter would be removing runs from +0.0228 down to
−0.0052 by role. **A filter that removes half of the trader's +0.0228 is detectable here. One that
removes half of the risk manager's +0.0031 is not**, and for roles whose underlying effect is near
zero the result will be uninformative by construction. That is written here so it is not presented
afterwards as a finding about those roles.

## Cost

59 rewrites, then at most 59 pairs × 5 roles × 2 contexts × 3 arms × 3 draws = **5,310 readings**,
fewer if rewrites are rejected. Roughly three hours, comparable to E36.

## What it cannot answer

- **One filter, one instruction.** A null is about this rewrite prompt, not about the idea of
  filtering. Say "this filter" every time.
- **One event and one model.** Same limits as everything else here.
- It says nothing about whether the filter would preserve *urgency* where urgency is decision-bearing.
  Urgency is a payload slot in this design and is checked as one, but the deeper question of whether a
  neutralising filter suppresses a warning that should have been heard is not measured, and is the
  first thing to say in the limitations.
- It is measured on reader **language**. Whether a filter changes reader **decisions** needs the belief
  outcomes, which this run does not collect.

---

## Amendment A, 2026-08-08 · The gate rejected all 59 rewrites, and the rewrites were right

**Made after the rewrite pass and before any reading existed** — the run refused to generate the
reading pass precisely because the gate failed, so no outcome data has been seen and none exists.

The first E41 attempt produced 59 rewrites and the preservation gate rejected **every one**, all on
the same two slots. Reading them showed the filter had done exactly what it was asked: every number
survived, the entity, action, confidence, horizon and review timing survived verbatim, and the affect
sentence was gone. What failed was the gate's phrasing requirement. The model wrote "the forecast
indicates a further decline of 5 to 8 percent" where the slot strings are `down` and `a further 5 to
8 percent` — semantically exact, verbatim mismatch.

A gate that fails correct behaviour is worse than a permissive one, because the next person weakens
it in a hurry. So the check now matches what each slot *means*:

- **Numbers**: verbatim, unchanged — a magnitude is the one thing context cannot restore.
- **Direction**: a closed, documented synonym list per direction (`down` accepts *decline*, *fall*,
  *drop*…), so paraphrase passes and deletion still fails.
- **Magnitude**: its content tokens — the numbers and the unit words — rather than the phrase, so
  "5 to 8 percent" reworded passes while "5 to 8 basis points" fails.
- Everything else stays verbatim up to case and spacing.

Checked three ways before relaunch: the amended gate passes all 59 real rewrites; it still catches a
rewrite that deletes the direction and magnitude, and one that swaps the action; and the new
paraphrase test goes red on the old gate. The 40-of-59 floor and both contrasts are unchanged.

One thing this run already established, worth keeping whatever the reading pass shows: **the queue's
artifact check worked.** It reported "0 of 59 preserved", refused to spend three GPU-hours reading a
broken arm, and exited loudly.
