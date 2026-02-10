# IDS_TopoGCL

## Usage

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
python3 ids_gnn_lanl_baseline.py \
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
python3 ids_topogcl_lanl_baseline.py \
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
python3 ids_gnn_lanl_baseline.py \
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
python3 ids_topogcl_lanl_baseline.py \
  --auth_path ./datasets/LANL/auth_reformat.txt \
  --red_path ./datasets/LANL/redteam_reformat.txt \
  --window_size 30 \
  --benign_limit 600 \
  --mal_limit 300 \
  --epochs 3 \
  --out_json lanl_topogcl_results.json
```

> TopoGCL default: `--hidden_dim 64`, `--emb_dim 32`
