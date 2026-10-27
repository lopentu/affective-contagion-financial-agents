- Written 2026-08-07, before the breakdown was computed. **Frozen on commit.**
- Governs E38.
- **No GPU.** Every generation already exists.

# PR14 · Re-report the completed runs per role

## Why this exists

E36 established that reader roles respond differently
enough to cancel: `trader` at +0.0228, `fundamental_analyst` at −0.0052, shuffling test *p* = 0.0005.
The document says every average over agents is suspect and continues to headline seven of them.
Either re-report the published numbers per role, or label them uninterpretable.

The generations exist. This is arithmetic on data already collected.

## What this is, and what it is not

**It is descriptive re-reporting.** The same completed runs, the same outcome, grouped by reader role
instead of averaged over roles. No new data, no test, no threshold, no verdict.

**It is not a test of role heterogeneity.** E36 already did that, on a design built for it, with the
comparison fixed before its data existed. Nothing here can add to or subtract from that finding, and a
role that looks large in one of these breakdowns is not thereby established.

**It is not a re-analysis that supersedes any entry.** Every experiment keeps its number and its
verdict. This adds a breakdown beneath them.

Saying this in advance matters because the last time completed data was sliced by role after the fact
(E32's five-way split at 19 per cell), it got the direction of two
roles backwards, and E36 later contradicted it. That split was labelled a lead rather than a finding,
which is the only reason it did no damage. The same label applies here.

## What gets re-reported

Every completed run whose headline number averages over reader roles:

| Experiment | Headline average | Roles available |
| --- | --- | --- |
| E27 | +0.0221 per teammate | 5 untreated roles × 4 scenarios |
| E18 | +0.0150 teammate effect | 5 untreated roles |
| E31 | +0.0007 and +0.0019 teammate effects | 5 untreated roles × 2 scenarios |
| E30 | belief outcomes | 5 reader roles |

For each: the effect per role, its interval, and the count of seeds in the positive direction, using
the same outcome definition and the same aggregation the original entry used, with only the final
average-over-roles step removed.

## What is fixed now

1. **The outcome and aggregation come from the original entry.** No re-definition, no re-scrubbing, no
   alternative estimator. If a breakdown cannot be produced without changing the estimator, it is
   reported as not available rather than approximated.
2. **Roles are reported in a fixed order** — alphabetical — so no ordering can be chosen after seeing
   the values.
3. **Every role is reported, including uninteresting ones.** No selection, no highlighting of the
   largest, no commentary on individual roles beyond stating the number.
4. **The average is reported alongside**, so the breakdown and the published figure can be compared
   directly and neither replaces the other.
5. **No intervals are used to declare anything significant.** They are shown because a number without
   its spread is unreadable, not to support a claim about any role.

## What each result licenses

| | |
| --- | --- |
| Roles vary widely within an experiment | The published average is a mixture, exactly as E36 predicts. The entry's number stands and its interpretation gains "mean response" |
| Roles are tightly clustered | That experiment's average is a reasonable summary after all, and it is worth saying so |
| A role reverses sign | Descriptive only. It matches E36's pattern; it does not independently establish it |

**None of these outcomes changes a verdict.** If the breakdown is uniformly boring, that is worth
knowing and gets reported.

**The script is hashed before it was run**: `e38_analysis.py`, SHA-256 `571f07b147c4eafd`. An earlier hash `913ae56cdbb86725` covered a version missing
the E18 breakdown that this plan lists; it was added before any result was read.

## What this cannot support

- **Not a new finding about roles.** E36 is the finding; this is bookkeeping.
- **Not a correction to any experiment's arithmetic**, which was right.
- **Not anything about scale, beliefs, actions or the trichotomy.**
- Same model, scenarios and lexicon limitations as the entries it re-reports.
