# Notes for coding agents

This repository is the code release for *How LLM Financial Agents Respond to Stated Affect Depends
on Their Assigned Role* (FinNLP 2026). It holds the code, configurations, analysis plans and model
generations behind the paper. Read `README.md` first: its table maps each paper result to its
experiment, plan, script and data.

## Commands

Use `uv` for everything. Python 3.13.

```bash
uv sync --group dev                    # package and tests; no model dependencies
uv sync --group dev --extra experiment # adds torch, transformers and outlines for generation
uv run pytest                          # runs without a model or GPU; coverage must stay >= 85%
uvx ruff check . && uvx ruff format --check .
uvx ty check
uv run pre-commit run --all-files      # the three checks above, as the repository runs them
```

Ruff runs with every rule enabled (`select = ["ALL"]`), so public functions need docstrings. The
line limit is 100.

## Where things are

- `src/affective_contagion/` is the package. Model inference sits behind `GenerationBackend`
  (`generation.py`). Only `backends/huggingface.py` (generation) and `emotion_probe.py` (the
  emotion classifier) load a model. Everything else runs without one, and the tests inject fakes.
- `scripts/` holds command-line entry points. The `run_*.py` scripts generate data and need a GPU.
  `analyze_reader_outcomes.py`, `check_pairing.py`, `first_hop_check.py`, `second_hop_check.py`
  and `replay_run.py` do not.
- `configs/` holds the run configurations. `experiments/pr1/` is the generated team design and
  `experiments/pr5/` the injected design. These files are written by `scripts/generate_*.py`, so
  edit the generator and regenerate rather than editing a YAML file by hand.
- `data/generations/` holds the released generations, each with a SHA-256 manifest.
- `docs/plans/` holds the analysis plans, each fixed before its analysis was run. Its README says
  which ones came after their data.
- `docs/examples/sample-run/` is one complete run from the injected design.

## Identifiers

`E36`, `E51` and the like name experiments; released data files carry the same names. `PR12`,
`PR26` and the like name the plan that governed an experiment. A plan can mention files that are
not in this repository: plans and records for experiments the paper does not report, and
per-experiment analysis scripts (such as `e50_analysis.py`) that were not released.
`docs/plans/README.md` and `docs/reproducibility.md` say what is and is not here.

## Rules that keep the results valid

- **A matched pair differs only in its wording.** `payload.py` renders one structured payload in
  calm and alarmed wording and refuses a pair whose fact block differs. Don't add a code path that
  changes facts, numbers or the recommendation between arms.
- **Paired runs must stay identical until the treatment.** Matched arms share their seeds and
  random streams. Run `scripts/check_pairing.py` before any paired analysis of team runs.
- **Model settings live in the roster** (`configs/models/`): checkpoint, pinned revision,
  decoding, dtype. Don't duplicate them in experiment configs or code.
- **Read what varied from the data.** The treated role, scenario, seed and model vary across runs.
  Take them from each run's manifest or record. Don't assume one value: in the injected design the
  sender rotates over all six roles.
- **Don't edit the analysis plans** in `docs/plans/`. They record what was fixed in advance.
- **Don't edit the five E51 files whose hashes PR26 records:** `src/affective_contagion/trade_identity.py`,
  `scripts/e51_trade_identity.py`, `scripts/e51_queue.sh`, `scripts/e51_replication.py` and
  `scripts/e51_teammates.py`. Their hashes let a reader check them against the plan.
- **Don't change released data.** `affective_contagion.verification.verify_release` checks a
  reader archive against its manifest hashes, and
  `affective_contagion.injected_verification.verify_injected_release` checks the E51 archive. `scripts/export_release.py` writes new archives and refuses
  to overwrite an existing one.
- **Write run outputs under `outputs/`.** It is ignored by git. Don't write them to a temporary
  directory that the system may clean.

## What must not be committed

- **The NRC VAD lexicon.** Its terms forbid redistribution. Runs read it from `data/raw/`, which is
  ignored by git.
- **Model weights, caches, credentials and API keys.**

## Describing results

- The paper measures lexical arousal and a classifier's emotion labels in model output. Don't write
  that agents feel anxious or experience emotions.
- The finding is that the response depends on the role instruction. Different checks put the roles
  in different orders, so don't call any one role the most or least responsive.
- Every scenario is synthetic, and every company in `configs/scenarios/` is fictional.
