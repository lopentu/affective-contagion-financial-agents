#!/usr/bin/env bash
# E51: rerun the PR5 injected design (288 runs) and verify every run's artifact before moving on.
# Governed by docs/plans/pr26-was-the-treated-agents-trade-identical.md. Outputs under
# outputs/runs/e51, never a scratch directory (E27's runs were lost to a /tmp cleanup).
set -u
cd "$(dirname "$0")/.."
OUT=outputs/runs/e51
LEX=data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt
ROSTER=configs/models/qwen3_4b_shared.yaml
mkdir -p "$OUT"
for f in "$LEX" "$ROSTER" configs/experiments/pr5/neutral_s300.yaml configs/scenarios/meridian_deposit_outflow.yaml; do
  [ -f "$f" ] || { echo "E51 ABORT: missing input $f"; exit 1; }
done
done_runs=0
for scenario in meridian_deposit_outflow harborline_capacity_award pallasridge_covenant_note calderon_mixed_guidance; do
  for seed in $(seq 300 335); do
    for arm in neutral affective; do
      id="e51-${scenario}-${arm}-s${seed}"
      if [ -f "$OUT/$id/records.jsonl" ]; then done_runs=$((done_runs + 1)); continue; fi
      echo "[$(date -u +%FT%TZ)] START $id"
      uv run python scripts/run_experiment.py \
        --experiment "configs/experiments/pr5/${arm}_s${seed}.yaml" \
        --roster "$ROSTER" \
        --scenario "configs/scenarios/${scenario}.yaml" \
        --vad-lexicon "$LEX" --vad-lexicon-scale signed \
        --outputs "$OUT" --run-id "$id"
      if [ ! -f "$OUT/$id/records.jsonl" ]; then
        echo "E51 FAILED: $id produced no records.jsonl; stopping"; exit 1
      fi
      done_runs=$((done_runs + 1))
      echo "[$(date -u +%FT%TZ)] DONE $id ($done_runs/288)"
    done
  done
done
n=$(find "$OUT" -mindepth 2 -maxdepth 2 -name records.jsonl | wc -l)
if [ "$n" -eq 288 ]; then echo "E51 QUEUE COMPLETE: 288 runs with records"; else echo "E51 QUEUE ENDED WITH $n OF 288"; exit 1; fi
