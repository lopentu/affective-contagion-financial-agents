# Sample run

One complete run directory from the paper's injected design, and its replay page, so the pipeline
can be read without a GPU, a model download or the VAD lexicon.

```bash
# nothing to install; open the page in a browser
xdg-open docs/examples/sample-run.html
```

## What this is

The affective arm of one E51 pair: the deposit-outflow scenario at seed 306. Six agents trade for
12 rounds. At round 8 the sentiment analyst reads the alarmed rendering of the analyst report,
decides and trades, and its outgoing message is then replaced by the report itself. Its five
teammates read that message at round 9. The same run is in the E51 archive in `data/generations/`.

| | |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`, bfloat16 |
| Config | `configs/experiments/pr5/affective_s306.yaml` |
| Scenario | `configs/scenarios/meridian_deposit_outflow.yaml`; synthetic, and Meridian Bank does not exist |
| Affect instrument | NRC VAD v2.1 unigrams, `scale=signed`, `match=surface_then_suffix` |
| Decoding | grammar-constrained; parse failure 0.000 |

## What to look at

Open `sample-run.html`, press `→ shock`, and you land on the sentiment analyst's turn at round 8.
Each turn has two collapsible sections: the exact prompt sent to the model, and the raw model output
before parsing. Step to round 9 to see the teammates' prompts, which carry the report as the
sentiment analyst's message.

The pipeline panel is read from `trace.jsonl`, which the simulation writes as it runs, so it shows
the stages that actually ran.

## Files

| File | Contents |
| --- | --- |
| `manifest.json` | resolved config, roster, payload, stimuli, run health, provenance |
| `records.jsonl` | one row per agent per round: prompt, raw output, decision, delivery, VAD |
| `trace.jsonl` | executed pipeline stages, in order |
| `market.jsonl` | per-round imbalance, price, noise draw |
| `*.npy` | the saved series: activation, message arousal, positions, prices |

`a_internal.npy` is all-NaN, and the manifest records `"affect_direction_captured": false`. No
activation measurement was used in the paper.

## Licensing

The generated text is model output on a synthetic scenario. The NRC VAD lexicon is **not**
redistributed here, only arousal values derived from it. To re-score anything, obtain the lexicon
from the [NRC VAD page](https://saifmohammad.com/WebPages/nrc-vad.html) and cite it.

## Regenerating

`tests/test_sample_run.py` fails if the sample stops loading. To produce the run again:

```bash
uv run python scripts/run_experiment.py \
  --experiment configs/experiments/pr5/affective_s306.yaml \
  --roster configs/models/qwen3_4b_shared.yaml \
  --scenario configs/scenarios/meridian_deposit_outflow.yaml \
  --vad-lexicon data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt \
  --vad-lexicon-scale signed \
  --outputs outputs/runs --run-id sample-run

uv run python scripts/replay_run.py outputs/runs/sample-run --output sample-run.html
```

Generation is not always reproducible on GPU, so a new run can differ in wording from this one.
