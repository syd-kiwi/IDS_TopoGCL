#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/results/edge_removal"

RATES=(0.3)
auth="$ROOT_DIR/datasets/LANL/auth.txt"
red="$ROOT_DIR/datasets/LANL/redteam.txt"
topogcl_py="$ROOT_DIR/scripts/ids_topogcl_lanl_baseline.py"
gnn_py="$ROOT_DIR/scripts/ids_gnn_lanl_baseline.py"

for rate in "${RATES[@]}"; do
  python "$topogcl_py" --auth_path "$auth" --red_path "$red" --corruption_type edges --corruption_rate "$rate" --out_json "$ROOT_DIR/results/edge_removal/LANL_topogcl_${rate}.json"
  python "$gnn_py" --auth_path "$auth" --red_path "$red" --scenario edge_drop --scenario_rate "$rate" --out_json "$ROOT_DIR/results/edge_removal/LANL_gnn_${rate}.json"
  python "$ROOT_DIR/scripts/ids_svm_baseline.py" --auth_path "$auth" --red_path "$red" --edge_drop_rate "$rate" --out_json "$ROOT_DIR/results/edge_removal/LANL_svm_${rate}.json"
done

printf 'Done. LANL results at %s/results/edge_removal\n' "$ROOT_DIR"
