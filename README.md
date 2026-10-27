# Affective contagion in financial LLM-agent teams

Code, configurations, tests and released model generations for the paper *How LLM Financial Agents
Respond to Stated Affect Depends on Their Assigned Role* (FinNLP 2026).

The paper asks whether the wording of a financial report changes what LLM agents write and do when
every financial fact in the report is held fixed. This repository holds the code that renders the
matched reports, runs the agent team and the isolated readers, and checks the results, together
with the generations behind the paper's results.

[Reproduction notes](docs/reproducibility.md) list which results can be recomputed from this
repository and how.

## What is here

| Path | What it holds |
| --- | --- |
| `src/affective_contagion/` | The package. `payload.py` renders one structured financial payload in calm and alarmed wording and refuses to emit a pair whose fact block differs. `prompts.py` builds what an agent sees. `simulation.py` runs the six-agent team. `isolated_reader.py` shows one reader its inbox outside the team; `first_hop.py` and `second_hop.py` measure the reader's and teammates' responses in the team designs. `firewall.py` rewrites a message without its affect and checks the facts survived. `text_affect.py` scores lexical arousal, `scrub.py` removes decision words before scoring, and `copy_control.py` removes words the reader saw. `emotion_probe.py` is the second scoring instrument. `verification.py`, `reader_verification.py` and `injected_verification.py` recompute results from the released generations. |
| `configs/` | Four synthetic financial scenarios, model rosters with pinned checkpoint revisions, and the run configurations for the two team designs (`pr1/` generated, `pr5/` injected). The companies do not exist. |
| `scripts/` | Main entry points include `run_experiment.py` for a team run; `run_isolated_readers.py`, `run_context_comparison.py`, `run_factorial_readers.py`, `run_firewall_readers.py` and `run_closing_ablation_readers.py` for reader designs; `check_pairing.py` for the matched-pair gate; and `replay_run.py` to render a run as a page. |
| `data/generations/` | Released model outputs supporting paper results, with SHA-256 manifests. The E51 archive also retains its run manifests, market histories and lifecycle records. See its README. |
| `docs/examples/` | One run from the injected design and its replay page, so the pipeline can be read without a GPU. |
| `docs/plans/` | The analysis plans for the paper's experiments, each fixed before its analysis was run; most before their data existed. |
| `tests/` | The test suite. Inference is injected, so the tests run without a model. |

## Where each result lives

Each experiment has an identifier (`E36`) and an analysis plan in `docs/plans/` (`PR12`). Team runs
use `scripts/run_experiment.py` with the configs named below. Reader runs take their stimulus
pairs from those team runs. "Released" means the generations are in `data/generations/` under the
experiment's name.

| Paper section | Result | Experiment | Plan | How it was run | Data |
| --- | --- | --- | --- | --- | --- |
| 4.1 | The sender's own message, deposit outflow | E15 | PR2, PR3 | team runs, `configs/experiments/pr1/` seeds 6–35; `scripts/first_hop_check.py` | sender messages kept in the E36 inboxes |
| 4.1 | Teammates of a sender who wrote its own message | E18 | PR3 | team runs, `pr1/` seeds 36–67, then `scripts/run_isolated_readers.py --arms real silent`, aggregated with `affective_contagion.clustered` | not released |
| 4.1 | Sender and teammates on two further events | E31 | PR8 | team runs, `pr1/` seeds 500–523; `scripts/first_hop_check.py`, `scripts/second_hop_check.py` | capacity-award sender messages kept in the E47 inboxes |
| 4.1, D | Teammates of an injected message (injected design) | E51 | PR5, PR26 | team runs, `configs/experiments/pr5/`, `scripts/e51_queue.sh` | released, complete runs |
| 4.2 | Responses differ by role (central result) | E36 | PR12 | `scripts/run_context_comparison.py` | released |
| 4.2 | The role differences in two further model families | E39 | PR15 | `run_context_comparison.py`, `configs/models/ladder/` | released |
| 4.2, G | Copy control: words the reader saw removed | E40 | PR16 | `affective_contagion.copy_control` on E36 | uses E36 |
| 4.2, H | Second instrument: emotion classifier | E43 | PR18 | `affective_contagion.emotion_probe` on E36 | uses E36 |
| 4.2 | Affect crossed with urgency | E44 | PR19 | `scripts/run_factorial_readers.py` | released |
| 4.2 | Role instructions reworded twice | E50 | PR25 | `run_context_comparison.py`, `configs/models/paraphrase/` | released |
| 4.3 | Beliefs on the mixed-guidance event | E30 | PR7 | `scripts/run_isolated_readers.py --cognition`, `pr1/` seeds 400–423 | not released |
| 4.3 | Decisions on a SELL and a BUY event | E46, E47 | PR21, PR22 | `run_isolated_readers.py --cognition`; analysed by `scripts/analyze_reader_outcomes.py` | released |
| 4.3 | Decisions in two further families | E48 | PR23 | as E46 | released |
| 4.3 | Filter that rewrites the message without affect | E41 | PR17 | `scripts/run_firewall_readers.py` | released |
| 4.3, E | Removing or adding the closing sentence | E45 | PR20 | `scripts/run_closing_ablation_readers.py` | released |
| 4.3 | Filter and no-closing arms in one run | E49 | PR24 | `run_firewall_readers.py --include-no-closing` | released |
| D | Per-role breakdowns of the team runs | E38 | PR14 | from E30 and E31 | not released |
| I | Model scale, exploratory | E19 | PR2 | team runs, `pr1/` seeds 6–17, `scripts/generate_ladder_rosters.py` | not released |

Letters are appendices. [Reproduction notes](docs/reproducibility.md) say what can be recomputed
from the released data, and how.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev              # core package and tests, no model dependencies
uv run pytest
uv sync --extra experiment       # adds torch, transformers and friends for generation
```

To see a run without installing anything, open `docs/examples/sample-run.html`.

## Running an experiment

A run needs an experiment config, a roster, a scenario and the NRC VAD lexicon. All four are
required. This runs one generated-design team on the deposit-outflow scenario:

```bash
uv run python scripts/run_experiment.py \
  --experiment configs/experiments/pr1/affective_s6.yaml \
  --roster configs/models/qwen3_4b_shared.yaml \
  --scenario configs/scenarios/meridian_deposit_outflow.yaml \
  --vad-lexicon data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt \
  --vad-lexicon-scale signed \
  --run-id my-first-run
```

The lexicon is not redistributed here. Obtain NRC VAD v2.1 from the National Research Council
Canada under its research terms and place it under `data/raw/`.

Every run writes an immutable directory under `outputs/runs/` with a manifest recording the config,
seeds, model repository and revision, git commit and package versions.

## Reading the code

Identifiers such as `E36` name an experiment, and the generations in `data/generations/` are named
by them. Identifiers such as `PR12` name the analysis plan in `docs/plans/` that governed it;
`docs/plans/PROVENANCE.md` gives the date each plan was fixed.

## Licenses

We release our code, synthetic stimuli, analysis plans and generated texts under the repository's
[MIT License](LICENSE). Model weights are obtained separately under their own licenses and are not
redistributed here. These are the exact model-card revisions used for released results.

| Model | License | Source |
| --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | [Pinned model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/README.md) |
| `ibm-granite/granite-4.1-3b` | Apache-2.0 | [Pinned model card](https://huggingface.co/ibm-granite/granite-4.1-3b/blob/c0650403e44e78ec0262dab1c90914c65b196c4e/README.md) |
| `google/gemma-4-E4B-it` | Apache-2.0 | [Pinned model card](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md) |
| `SamLowe/roberta-base-go_emotions` | MIT | [Pinned model card](https://huggingface.co/SamLowe/roberta-base-go_emotions/blob/d75048347613a25d77de8cf6412eaae9fa7b26be/README.md) |

Google's [Gemma terms](https://ai.google.dev/gemma/terms) direct Gemma 4 users to the
[Apache-2.0 license](https://ai.google.dev/gemma/apache_2). The
[GoEmotions source repository](https://github.com/google-research/google-research/tree/master/goemotions)
uses [Apache-2.0](https://github.com/google-research/google-research/blob/master/LICENSE).

The NRC VAD lexicon is available separately for non-commercial research and educational use. Its
[terms prohibit redistribution](https://saifmohammad.com/WebPages/nrc-vad.html). Obtain it from the
author to reproduce VAD scores; the lexicon is not included here.

## Citation

Da-Chen Lian, Yu-Lin Chang, Yen-Hsi Huang and Shu-Kai Hsieh. How LLM Financial Agents Respond to
Stated Affect Depends on Their Assigned Role. In *Proceedings of the FinNLP 2026 Workshop*, 2026.
