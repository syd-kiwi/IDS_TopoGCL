#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/fast_lanl

AUTH_PATH="datasets/LANL/auth_reformat.txt"
RED_PATH="datasets/LANL/redteam_reformat.txt"
RATE="0.5"
SEED="42"

# OPTC is excluded from the fast paper run because the LANL implementation is the controlled completed evaluation path.

COMMON_ARGS=(
  --auth_path "$AUTH_PATH"
  --red_path "$RED_PATH"
  --epochs 1
  --hidden_dim 32
  --emb_dim 16
  --benign_limit 120
  --mal_limit 120
  --random_seed "$SEED"
  --train_degradation_rate "$RATE"
  --test_degradation_rate "$RATE"
  --low_volume_mode events
  --missing_structure_mode edges
  --interference_mode feature_mask
)

for scenario in low_volume missing_structure interference; do
  python ids_gnn_lanl_baseline.py "${COMMON_ARGS[@]}" \
    --train_scenario clean --test_scenario "$scenario" \
    --out_json "results/fast_lanl/lanl_gnn_clean_to_${scenario}_${RATE}_seed${SEED}.json"

  python ids_gnn_lanl_baseline.py "${COMMON_ARGS[@]}" \
    --train_scenario "$scenario" --test_scenario "$scenario" \
    --out_json "results/fast_lanl/lanl_gnn_${scenario}_to_${scenario}_${RATE}_seed${SEED}.json"

  python ids_topogcl_lanl_baseline.py "${COMMON_ARGS[@]}" \
    --train_scenario clean --test_scenario "$scenario" \
    --out_json "results/fast_lanl/lanl_topogcl_clean_to_${scenario}_${RATE}_seed${SEED}.json"

  python ids_topogcl_lanl_baseline.py "${COMMON_ARGS[@]}" \
    --train_scenario "$scenario" --test_scenario "$scenario" \
    --out_json "results/fast_lanl/lanl_topogcl_${scenario}_to_${scenario}_${RATE}_seed${SEED}.json"
done
