# IDS_TopoGCL

## Usage

> ⚠️ Dataset files in this repo may be Git LFS pointers in a fresh clone. If runs parse zero windows, resolve data first:
>
> ```bash
> git lfs install
> git lfs pull
> ```

To run baseline GNN on OPTC (reproduce ids_results.json):

```bash
python ids_gnn_optc_baseline.py --auth_path ./datasets/OPTC/auth_optc.txt --red_path ./datasets/OPTC/redteam_optc.txt
```

> GNN default: `--hidden_dim 32`, `--emb_dim 16`

To run TopoGCL on OPTC (reproduce ids_topogcl_results.json):

```bash
python ids_topogcl_optc_baseline.py --auth_path ./datasets/OPTC/auth_optc.txt --red_path ./datasets/OPTC/redteam_optc.txt
```

> TopoGCL default: `--hidden_dim 64`, `--emb_dim 32`

To run baseline GNN on LANL, 30% malware data (reproduce lanl_gnn_30p.json):

```bash
python ids_gnn_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --benign_limit 600 \
  --mal_limit 90 \
  --epochs 3 \
  --out_json lanl_gnn_30p.json
```

> GNN default: `--hidden_dim 32`, `--emb_dim 16`

To run TopoGCL on LANL, 30% malware data (reproduce lanl_topogcl_30p.json):

```bash
python ids_topogcl_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --benign_limit 600 \
  --mal_limit 90 \
  --epochs 3 \
  --out_json lanl_topogcl_30p.json
```

> TopoGCL default: `--hidden_dim 64`, `--emb_dim 32`

To run baseline GNN on LANL, 100% malware data (reproduce lanl_ids_results.json):

```bash
python ids_gnn_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --benign_limit 600 \
  --mal_limit 300 \
  --epochs 3 \
  --out_json lanl_ids_results.json
```

> GNN default: `--hidden_dim 32`, `--emb_dim 16`

To run TopoGCL on LANL, 100% malware data (reproduce lanl_topogcl_results.json):

```bash
python ids_topogcl_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --benign_limit 600 \
  --mal_limit 300 \
  --epochs 3 \
  --out_json lanl_topogcl_results.json
```

> TopoGCL default: `--hidden_dim 64`, `--emb_dim 32`

## Corruption experiments (OPTC + LANL)

Both TopoGCL scripts now support one configurable corruption pipeline (`--corruption_type`) with reproducible sampling (`--random_seed`):

- `none` (default): clean data
- `node_features`: post-graph feature masking
- `edges`: post-graph edge dropping
- `temporal`: pre-graph raw-event dropping

Common flags:

```bash
--corruption_type {none,node_features,edges,temporal}
--corruption_rate FLOAT
--node_mask_mode {element,dimension}
--temporal_drop_mode {random,window}
--temporal_window_size INT
--random_seed INT
```

Example: OPTC node-feature element masking at 20%

```bash
python ids_topogcl_optc_baseline.py \
  --auth_path ./datasets/OPTC/auth_optc.txt \
  --red_path ./datasets/OPTC/redteam_optc.txt \
  --corruption_type node_features \
  --node_mask_mode element \
  --corruption_rate 0.2 \
  --random_seed 42
```

Example: LANL temporal contiguous window dropping

```bash
python ids_topogcl_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --corruption_type temporal \
  --temporal_drop_mode window \
  --temporal_window_size 50 \
  --corruption_rate 0.3 \
  --random_seed 42
```

Rate sweep (0.0, 0.1, 0.2, 0.3, 0.4):

```bash
for rate in 0.0 0.1 0.2 0.3 0.4; do
  python ids_topogcl_optc_baseline.py \
    --auth_path ./datasets/OPTC/auth_optc.txt \
    --red_path ./datasets/OPTC/redteam_optc.txt \
    --corruption_type edges \
    --corruption_rate "${rate}" \
    --out_json "optc_topogcl_edges_${rate}.json"
done
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tactical imperfect data experiments
- **low_volume**: IDS observes less data (event/edge/window subsampling).
- **missing_structure**: nodes/edges are missing.
- **interference**: noisy, delayed, or corrupted observations.

```bash
bash scripts/run_clean_results.sh
bash scripts/run_low_volume_sweep.sh
bash scripts/run_missing_structure_sweep.sh
bash scripts/run_interference_sweep.sh
bash scripts/run_all_experiments.sh
python scripts/aggregate_results.py
python scripts/plot_results.py
```
