# PR12 · Is the average hiding the effect, and does any of it reproduce?

- Written 2026-08-07, before any run under this plan existed. **Frozen on commit.**
- Governs E36.

## Why this is the most important run outstanding

Two things turned up on 2026-08-06 that call the measurement itself into question, not just one result.

**Aggregate effects may be averages over roles pulling in opposite directions.** Slicing
E32's hop 1 by which agent happened to read first gave
`bear_researcher` +0.0406, `fundamental_analyst` +0.0391, `bull_researcher` −0.0228, `risk_manager`
−0.0261. If that pattern is real, then E32's and E33's nulls
are not nulls — they are cancellations — and every result in this project that averages over agents is
measuring the wrong thing. That slice was chosen after seeing a null, at 19 observations per cell, with
no correction for five looks, so it is a lead and nothing more. This plan turns it into a test.

**And two designs that should agree gave opposite signs.** E32's `fundamental_analyst` readers gave
+0.0391; E35's bare condition, the same role in what should
be the same situation, gave −0.0178. Both intervals excluded zero. Until that is understood, **every
effect size in this repository carries an unknown amount of run-to-run instability**, and
E13/E14 already found generation here is
intermittently non-reproducible.

Nothing further is worth running until these are settled, because both determine whether existing
numbers mean anything.

## Design: role × context, run twice

A full crossing, so all three questions are answered by one run rather than three.

| Factor | Levels |
| --- | --- |
| **Reader role** | all five non-sender roles |
| **Context** | `bare` (message alone, price at fundamental) and `embedded` (plus four peers, real post-shock price), exactly as E35 defined them |
| **Arm** | calm and anxious rendering of one payload |
| **Replicate** | **two independent repeats of the whole design**, differing only in the generation seed base |

| | |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Stimulus | the `meridian_deposit_outflow` matched rendering, dose +0.0404 — the same one E27, E32 and E35 used |
| Pairs | the **59** `s30` pairs that pass the pairing gate |
| Draws | 3 per cell |
| Cost | 5 × 2 × 2 × 2 × 59 × 3 = **7,080 generations**, roughly four hours |

The replicate factor is what makes this more than a bigger role sweep. Two independent passes over an
identical design either agree or they do not, and that is measured rather than assumed.

## What is measured

**Primary — is there role heterogeneity at all?** A permutation test on the *spread* of the five role
effects: shuffle role labels within pair, recompute the spread, 2,000 times. This asks one question
once, rather than five questions with a correction, and it is the question the lead actually raises.

**Secondary, in this order:**

1. **Per-role effects**, Holm-corrected across the five roles. Which roles, and in which direction.
2. **Stability** — the same per-role effect in replicate A against replicate B. Reported as the
   agreement in sign and the size of the discrepancy, not as a test.
3. **Context**, repeating E35's primary on fresh generations: embedded minus bare.

## Sensitivity, computed before running

Seed-level SD is 0.0430, measured in E32. At 59 pairs per cell the smallest detectable effect per
role is **0.0116**; pooling the two replicates for a per-role estimate gives about **0.0082**.

The lead's role effects range from +0.0406 to −0.0261. If they are real at anything like that size,
this design sees them comfortably. If the true spread is half that, the permutation test is
underpowered and the run reports an uninformative null rather than "roles do not matter".

**What would count as nothing:** a permutation test that does not reject, with per-role intervals all
containing zero, means the role slice in E32 was noise. That is a genuinely useful answer and it is
written here so it cannot be presented as a disappointment.

## The rule, fixed now

1. **Parse health** ≥ 95% in every cell, reported per role and per context.
2. **Read the permutation test first.** If it does not reject, the per-role numbers are descriptive
   only and no role may be singled out.
3. **Report stability whatever it shows.** If the two replicates disagree in sign for any role, that
   is reported at the top of the entry, not in a limitations list — it would mean the instrument is
   not yet reliable enough for the effect sizes being chased.

**What each outcome licenses, fixed in advance:**

| Result | What may be said |
| --- | --- |
| heterogeneity rejects, replicates agree | **Roles differ, and reliably.** Every aggregate result here must be re-read as an average over cancelling effects, and future designs report per role |
| heterogeneity rejects, replicates disagree | The instrument is too unstable to trust the pattern. Fix reproducibility before any further effect estimates |
| no heterogeneity, replicates agree | E32's role slice was noise. Aggregate reporting stands, and the nulls stay nulls |
| no heterogeneity, replicates disagree | The worst case: neither the pattern nor the measurement can be relied on, and reproducibility becomes the only priority |

The last two rows both remove work rather than adding it, which is the point of running this before
anything else.

**The analysis is frozen and hashed before the run was queued**: `e36_analysis.py`, SHA-256
`773eaad96e5c297d`. All branches were exercised on synthetic data — real heterogeneity, no heterogeneity,
and a deliberate sign flip between passes — and the first two attempts at that check were themselves
broken, which is why it was worth doing. The stability check originally counted a sign flip whenever
two passes disagreed, so two cells that were both indistinguishable from zero would flip about half the
time and it would have reported the instrument as unreliable on pure noise. **A flip now counts only
when both passes are clear of zero**, and cells that differ in sign while either pass includes zero are
reported separately as expected noise.

## What this cannot support

- **Not a rescue or a refutation of any earlier result.** E32, E33 and E35 stand as reported. This
  measures whether their *aggregation* was hiding structure.
- **Not the trichotomy.** One hop throughout.
- **Not the treated role.** The sender stays `sentiment_analyst`; this varies who *reads*.
  E17 looked at the sender and was explicitly exploratory.
- **Not anxiety.** Same lexicon limitation as everywhere.
- **One model, one scenario, no memory.**
