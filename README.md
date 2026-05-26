# IDS_TopoGCL (Focused Study: Clean vs Missing Edges)

This repository is now focused on exactly three research questions:

1. Does **TopoGCL** outperform **SVM** and **GNN** under clean graph observations?
2. Does TopoGCL maintain stronger detection performance when communication edges are removed?
3. How does missing-edge rate affect the TopoGCL performance gap vs baselines?

## What remains in scope

- `ids_topogcl_optc_baseline.py`, `ids_topogcl_lanl_baseline.py`
- `ids_gnn_optc_baseline.py`, `ids_gnn_lanl_baseline.py`
- `ids_svm_baseline.py` (new one-class SVM baseline)
- `src/data_corruptions.py` (edge drop utilities)
- `scripts/run_edge_removal_study.sh`
- `scripts/summarize_edge_removal.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If datasets are LFS pointers, resolve them first:

```bash
git lfs install
git lfs pull
```

## Run the full study

```bash
bash scripts/run_edge_removal_study.sh
python scripts/summarize_edge_removal.py
```

Outputs:
- per-run JSONs: `results/edge_removal/*.json`
- combined metrics: `results/edge_removal_summary.csv`
- AUROC performance gaps: `results/edge_removal_gaps.csv`

## Interpretation

- **Q1 (clean):** inspect rows where `edge_drop_rate == 0.0` in `edge_removal_summary.csv`.
- **Q2 (robustness):** compare methods as `edge_drop_rate` increases.
- **Q3 (gap trend):** inspect `gap_topogcl_minus_gnn` and `gap_topogcl_minus_svm` in `edge_removal_gaps.csv`.
