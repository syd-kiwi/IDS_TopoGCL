#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/edge_removal

DATASETS=("OPTC" "LANL" "ACI")
RATES=(0.3)

pick_first_existing_file() {
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

for ds in "${DATASETS[@]}"; do
  if [[ "$ds" == "OPTC" ]]; then
    auth="datasets/OPTC/auth_optc.txt"
    red="datasets/OPTC/redteam_optc.txt"
    topogcl_py="scripts/ids_topogcl_optc_baseline.py"
    gnn_py="scripts/ids_gnn_optc_baseline.py"
  elif [[ "$ds" == "LANL" ]]; then
    auth="datasets/LANL/auth_reformat.txt"
    red="datasets/LANL/redteam_reformat.txt"
    topogcl_py="scripts/ids_topogcl_lanl_baseline.py"
    gnn_py="scripts/ids_gnn_lanl_baseline.py"
  else
    auth=$(pick_first_existing_file \
      "datasets/ACI/auth_aci.txt" \
      "datasets/ACI/auth_reformat.txt" \
      "datasets/ACI/auth.txt")
    red=$(pick_first_existing_file \
      "datasets/ACI/redteam_aci.txt" \
      "datasets/ACI/redteam_reformat.txt" \
      "datasets/ACI/redteam.txt")
    # The LANL variants are the generic non-OPTC TopoGCL/GNN runners in this repo.
    topogcl_py="scripts/ids_topogcl_lanl_baseline.py"
    gnn_py="scripts/ids_gnn_lanl_baseline.py"
  fi

  for rate in "${RATES[@]}"; do
    python "$topogcl_py" --auth_path "$auth" --red_path "$red" --corruption_type edges --corruption_rate "$rate" --out_json "results/edge_removal/${ds}_topogcl_${rate}.json"
    python "$gnn_py" --auth_path "$auth" --red_path "$red" --scenario edge_drop --scenario_rate "$rate" --out_json "results/edge_removal/${ds}_gnn_${rate}.json"
    python scripts/ids_svm_baseline.py --auth_path "$auth" --red_path "$red" --edge_drop_rate "$rate" --out_json "results/edge_removal/${ds}_svm_${rate}.json"
  done
done

printf 'Done. Results at results/edge_removal\n'
