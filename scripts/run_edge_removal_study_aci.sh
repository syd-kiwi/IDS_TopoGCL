#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/edge_removal

RATES=(0.3)

auth="datasets/ACI/auth_aci.txt"
red="datasets/ACI/redteam_aci.txt"
topogcl_py="scripts/ids_topogcl_lanl_baseline.py"
gnn_py="scripts/ids_gnn_lanl_baseline.py"

for rate in "${RATES[@]}"; do
  python "$topogcl_py" --auth_path "$auth" --red_path "$red" --corruption_type edges --corruption_rate "$rate" --out_json "results/edge_removal/ACI_topogcl_${rate}.json"
  python "$gnn_py" --auth_path "$auth" --red_path "$red" --scenario edge_drop --scenario_rate "$rate" --out_json "results/edge_removal/ACI_gnn_${rate}.json"
  python scripts/ids_svm_baseline.py --auth_path "$auth" --red_path "$red" --edge_drop_rate "$rate" --out_json "results/edge_removal/ACI_svm_${rate}.json"
done

printf 'Done. ACI results at results/edge_removal\n'
