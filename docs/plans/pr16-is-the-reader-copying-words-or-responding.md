# PR16 · Is the reader copying words, or responding?

- Governs E40. Analysis fixed and hashed **before it was run**, on data that **already existed**.
- Serves **soundness** by checking whether word reuse explains the measured response.

## Prior exposure, stated first because it changes what this can claim

The generations this uses were made for E36 and are
already analysed and published. **This is a post-hoc analysis of existing data.** It is not frozen
before its data existed, and it cannot be, because the criticism that prompted it arrived after the
run.

What is frozen is the analysis itself: the outcome, the rule and the reading of every result are
written here before the script was executed once. Nothing below was chosen after seeing a number. The
result is a robustness check on E36 and must be cited as one, never as an independent finding.

## The criticism

> The word list helps select and verify the manipulation, and then measures the response. The result
> may be reuse of lexicon-recognised words rather than an affective response at all.

Put plainly: we show an agent a message full of tense words, and then we count tense words in its
reply. If the agent simply repeats some of what it read, our number goes up and nothing has
propagated. This is a better objection than the one about anxiety versus urgency, because it survives
a human validation study. Even a perfectly validated stimulus leaves the copying explanation open.

## The test

Delete from every reply every word that appeared in the message the reader saw, then score what is
left. If the reply's arousal rose only because it reused the sender's vocabulary, the rise disappears.
If it rose because the reader wrote its own tense sentences, the rise survives.

**The deleted set is the union of both arms' inboxes, and this is the design decision that matters.**
The calm and high-arousal arms have different inboxes, so deleting each arm's own words would delete a
larger and more aroused set from the high-arousal reply than from its match. That biases the effect
downward by construction and breaks the matching the whole design rests on. Taking the union means the
same words are removed from both replies of a pair, so the two remain comparable and the only thing
that changed is that shared vocabulary is gone.

The per-arm version is reported too, as a **conservative bound**. It cannot be the primary for the
reason just given.

Affect-naming words are removed as well, using the existing closed `ECHO_TERMS` list, so a reader that
writes "the sentiment analyst sounds alarmed" does not score as alarmed itself. That distinction was
already drawn in E27 and this reuses its list
rather than inventing a second one.

## Outcome and rule

Everything is E36's estimator applied to scrubbed text: draws averaged to the (pair, role, context)
cell, arms differenced within pair, roles pooled over context.

**Primary, two conditions, both required.**

1. The permutation test on the spread of the five role effects still rejects at *p* < 0.05, role
   labels shuffled within pair, 2,000 draws.
2. The fundamental analyst's estimate is still negative with an interval excluding zero.

**Both hold** → copying is not the explanation. The reader is producing tense language of its own, and
the role difference is a difference in how agents respond rather than in how much they echo.

**Neither holds** → the copying explanation is live, and the paper must say so. This is the outcome
that would cost us the most, which is why the rule is written before the script runs. It would not
retract E36, which measured what it measured; it would change what E36 *means*, from "roles respond
differently" to "roles echo differently", and the paper would have to be rewritten around that.

**One holds and not the other** → reported as such, no verdict. Most likely case is the spread
surviving while the fundamental analyst's interval widens across zero, since removing words costs
precision. That would support heterogeneity and leave the sign reversal unconfirmed under the control.

## What must be reported alongside, whatever the answer

Three descriptive tables, per arm and per role:

- **Message length**, in words. If high-arousal replies are systematically longer, arousal per token
  could be a length artefact and this shows it.
- **Lexicon coverage**, the fraction of tokens the word list scores. If coverage differs by arm, the
  two arms' scores are averages over different denominators.
- **Verbatim overlap with the input**, the fraction of a reply's word types that appeared in its
  inbox. This is the direct measure of copying, and it is worth having as a number regardless of what
  the primary shows.

None of these three has a threshold attached. They are context for reading the primary, and inventing
a bar for them now, having never measured them, would be a bar chosen out of the air.

## What this cannot do

- It cannot rule out **paraphrase**. A reader that reads "alarming" and writes "worrying" copies the
  idea without repeating the word, and this test scores that as a response. Nothing available here
  separates paraphrase from independent production; the honest limitation is that this rules out
  verbatim reuse only.
- It cannot say anything about **anxiety**. It is the same word list with a smaller input.
- It is **one model and one event**, since that is what E36 was.

## Reporting

E40 in `docs/experiments/`, cross-linked from E36. Any number that reaches a document is registered in
`docs/verified-numbers.toml` first.
