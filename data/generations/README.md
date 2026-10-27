# Released generations

This folder holds the model outputs behind the paper's isolated-reader results and the complete
runs of its injected design (E51).

The complete generated-design team runs (E15, E18, E31), the belief run (E30) and the model-scale
sweep (E19) were not exported. The sender's messages from some of those team runs survive in the
E36 and E47 inbox files; the [reproduction notes](../../docs/reproducibility.md) show which numbers
they reproduce.

## What is here

There are two export formats. Each release's manifest identifies the format, filenames, hashes and
export commit.

- **Schema 1, isolated-reader releases:** a compressed readings file, a compressed inbox file and a
  manifest. The inbox file records the context supplied to each reader.
- **Schema 2, E51:** a compressed complete-run archive and a manifest. The archive retains each
  run's manifest, generation records, market history and lifecycle records. Sender identity comes
  from the run manifest.

| Release | Experiment | Readings | Reader model |
| --- | --- | --- | --- |
| `e36-role-context` | E36, the central role result | 7,080 | Qwen3-4B |
| `e39-family-granite` | E39 | 3,540 | Granite-4.1-3B |
| `e39-family-gemma` | E39 | 3,540 | Gemma-4-E4B |
| `e44-affect-by-urgency` | E44, the 2×2 | 3,540 | Qwen3-4B |
| `e41-firewall` | E41, the filter | 5,310 | Qwen3-4B |
| `e45-closing-ablation` | E45 | 5,310 | Qwen3-4B |
| `e46-action-sell` | E46 | 1,770 | Qwen3-4B |
| `e47-action-buy` | E47 | 720 | Qwen3-4B |
| `e48-action-granite` | E48 | 1,770 | Granite-4.1-3B |
| `e48-action-gemma` | E48 | 1,770 | Gemma-4-E4B |
| `e49-fourarm-filter` | E49 | 7,080 | Qwen3-4B |
| `e50-paraphrase-wording-a` | E50 | 3,540 | Qwen3-4B |
| `e50-paraphrase-wording-b` | E50 | 3,540 | Qwen3-4B |
| `e50-paraphrase-wording-c` | E50 | 3,540 | Qwen3-4B |
| `e51-injected-rerun` | E51, the injected design: 288 complete runs (schema 2, every round, market history and manifests) | 288 runs | Qwen3-4B |

The models and scenarios vary across releases. Read each manifest rather than applying E36's
single-model, single-event description to the whole folder. Only `e36-role-context` is wired into
`affective_contagion.verification`. E51 has a separate independent verifier described below. The
other archives are inspectable and hash-verifiable but are not all wired into one public
paper-number checker.

## Why it is committed when nothing else is

Run directories elsewhere in this project are gigabytes and stay out of git. These releases are
small enough to commit and contain the inputs needed for independent recomputation.

## Checking it yourself

The action tables (E46, E47, E48) require no model inference:

```bash
uv run python scripts/analyze_reader_outcomes.py data/generations/e46-action-sell.manifest.json data/generations/e47-action-buy.manifest.json data/generations/e48-action-granite.manifest.json data/generations/e48-action-gemma.manifest.json
uv run python scripts/analyze_reader_outcomes.py data/generations/e47-action-buy.manifest.json --buy-mechanism
```

The analysis uses Student t intervals and tests with Holm correction across roles per outcome.
Size is the numeric field in every reply, including HOLD; no trades are executed. The separate
`reader_verification.reader_numbers` function verifies these quantities independently.

`affective_contagion.verification.verify_release` checks a schema-1 release against its manifest
hashes. The same module recomputes the central E36 result from the released generations, and
`affective_contagion.copy_control.copy_control_by_inbox` recomputes the copy control (E40) from the
same archive. Both need the NRC VAD lexicon described in the repository README:

```bash
uv run python -c 'from pathlib import Path; from affective_contagion.configuration import load_yaml_model; from affective_contagion.copy_control import copy_control_by_inbox; from affective_contagion.payload import FinancialPayload, matched_realizations; from affective_contagion.text_affect import VADLexicon; lex = VADLexicon.from_tsv(Path("data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"), scale="signed", match="surface_then_suffix"); payload = load_yaml_model(Path("configs/scenarios/meridian_deposit_outflow.yaml"), FinancialPayload); r = copy_control_by_inbox(Path("data/generations/e36-role-context.manifest.json"), lex, matched_realizations(payload)); print(r.spread_controlled, r.permutation_p, r.mean_controlled); [print(x) for x in r.roles]'
```

`affective_contagion.injected_verification.verify_injected_release` independently checks the E51
schema-2 archive, reconstructs pairs and sender identity from archived manifests, verifies the
pairing gate, and derives the E51 outcomes. This command prints the archive summary, pooled run
health and the six-role table reported in Appendix D:

```bash
uv run python -c 'from pathlib import Path; from affective_contagion.injected_verification import verify_injected_release; r = verify_injected_release(Path("data/generations/e51-injected-rerun.manifest.json"), Path("data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt")); print(r.archive); print(r.identity.pooled); [print(role, row) for role, row in sorted(r.outcome_6_surface_then_suffix.items())]'
```

That route reproduces the E51 Appendix D role table from the released archive. It also checks the
125 of 144 identical requested trades, the 128 of 144 identical filled positions and next-round
prices, all 144 pairing gates, and zero parser fallbacks. It does not run the optional GoEmotions
classifier outcome unless a compatible scorer is supplied.

Two other statements in Appendix D are descriptive breakdowns: the `-0.0069` to `+0.0102` role
range for two self-written-message events, and the `+0.0635` against `+0.0207` belief comparison.
Their per-role readings were not retained, so they cannot be recomputed from the files here.

## Licenses

We release our code, synthetic stimuli, analysis plans and generated texts under the repository's
[MIT License](../../LICENSE). Model weights are obtained separately under their own licenses. The
repository README lists each pinned model-card revision and its license.

The NRC VAD lexicon is available separately for non-commercial research and educational use. Its
[terms prohibit redistribution](https://saifmohammad.com/WebPages/nrc-vad.html). Obtain it from the
author to reproduce VAD scores; it is not included in this release.

## Provenance

The models, scenarios and export commits vary by release and are recorded in each manifest. All
scenarios use fictional companies and synthetic financial events. The releases contain no personal
data, licensed corpus or real market data.

Re-export with `scripts/export_release.py`, which writes gzip with a zeroed timestamp so identical
content always hashes identically.
