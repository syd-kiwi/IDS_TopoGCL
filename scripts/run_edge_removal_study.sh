#!/usr/bin/env bash
set -euo pipefail

RATES=(0.0 0.1 0.2 0.3 0.4 0.5)
mkdir -p results/edge_removal

for DATASET in optc lanl; do
  if [[ "$DATASET" == "optc" ]]; then
    AUTH="datasets/OPTC/auth_optc.txt"
    RED="datasets/OPTC/redteam_optc.txt"
  else
    AUTH="datasets/LANL/auth_reformat.txt"
    RED="datasets/LANL/redteam_reformat.txt"
  fi

  for RATE in "${RATES[@]}"; do
    python ids_topogcl_${DATASET}_baseline.py \
      --auth_path "$AUTH" --red_path "$RED" \
      --corruption_type edges --corruption_rate "$RATE" \
      --out_json "results/edge_removal/${DATASET}_topogcl_edge_${RATE}.json"

    python ids_gnn_${DATASET}_baseline.py \
      --auth_path "$AUTH" --red_path "$RED" \
      --out_json "results/edge_removal/${DATASET}_gnn_edge_${RATE}.json"

    python ids_svm_baseline.py \
      --auth_path "$AUTH" --red_path "$RED" \
      --edge_drop_rate "$RATE" \
      --out_json "results/edge_removal/${DATASET}_svm_edge_${RATE}.json"
  done
done
