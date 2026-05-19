#!/usr/bin/env bash
set -euo pipefail
SCENARIO_FILTER="${1:-all}"
RATES=(0.0 0.1 0.2 0.3 0.4 0.5)
for DATASET in lanl optc; do
  for METHOD in gnn topogcl; do
    SCRIPT="ids_${METHOD}_${DATASET}_baseline.py"
    for SC in clean low_volume missing_structure interference; do
      [[ "$SCENARIO_FILTER" != "all" && "$SCENARIO_FILTER" != "$SC" ]] && continue
      OUTDIR="results/${SC}"; mkdir -p "$OUTDIR"
      for RATE in "${RATES[@]}"; do
        python "$SCRIPT" --train_scenario clean --test_scenario "$SC" --train_degradation_rate 0.0 --test_degradation_rate "$RATE" --out_json "$OUTDIR/${DATASET}_${METHOD}_clean_to_${SC}_${RATE}.json" || true
        python "$SCRIPT" --train_scenario "$SC" --test_scenario "$SC" --train_degradation_rate "$RATE" --test_degradation_rate "$RATE" --out_json "$OUTDIR/${DATASET}_${METHOD}_${SC}_to_${SC}_${RATE}.json" || true
      done
      python "$SCRIPT" --train_scenario "$SC" --test_scenario "$SC" --train_degradation_rate 0.2 --test_degradation_rate 0.4 --out_json "$OUTDIR/${DATASET}_${METHOD}_mild_to_severe_${SC}.json" || true
    done
  done
done
# TODO: add IoT experiments when IoT dataset scripts are added.
