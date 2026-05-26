#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/edge_removal

RATES=(0.3)
auth="datasets/OPTC/auth_optc.txt"
red="datasets/OPTC/redteam_optc.txt"
topogcl_py="scripts/ids_topogcl_optc_baseline.py"
gnn_py="scripts/ids_gnn_optc_baseline.py"

for rate in "${RATES[@]}"; do
  python "$topogcl_py" --auth_path "$auth" --red_path "$red" --corruption_type edges --corruption_rate "$rate" --out_json "results/edge_removal/OPTC_topogcl_${rate}.json"
  python "$gnn_py" --auth_path "$auth" --red_path "$red" --scenario edge_drop --scenario_rate "$rate" --out_json "results/edge_removal/OPTC_gnn_${rate}.json"
  python scripts/ids_svm_baseline.py --auth_path "$auth" --red_path "$red" --edge_drop_rate "$rate" --out_json "results/edge_removal/OPTC_svm_${rate}.json"
done

printf 'Done. OPTC results at results/edge_removal\n'
