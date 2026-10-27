# Analysis plans

This folder holds the analysis plans for the paper's experiments. Each plan fixed the outcomes,
comparisons, exclusions, statistical tests and interpretation rules before its analysis was run.
Most were written before their data existed. PR14, PR16 and PR18 analyse generations that already
existed, and PR1 was committed after its first runs had started; each states this in its own text.
Amendments are dated and say what had and had not been seen when they were written.

[PROVENANCE.md](PROVENANCE.md) gives the date and source commit at which each plan was fixed. The
plans describe what was planned, not what was found; the paper reports the results.

Some plans refer to records that are not part of this repository: the experiment log
(`docs/experiments/`), a number registry (`docs/verified-numbers.toml`), per-experiment analysis
scripts such as `e50_analysis.py`, and plans for experiments the paper does not report. File hashes
recorded in a plan identify the version of that file when the plan was fixed. The E51 files listed
in PR26 are unchanged and still match; other files here are later versions. Use the
[generation inventory](../../data/generations/README.md) for the recomputation commands.
