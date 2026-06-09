# IDS_TopoGCL

This folder contains a compact, reproducible workflow for graph-based intrusion detection experiments using the `NF-BoT-IoT` and `NF-ToN-IoT` datasets.

## Repository layout

- `scripts/build_ids_graph_classification_dataset.py`  
  Builds graph classification datasets from tabular IDS data.
- `scripts/train_ids_binary.py`
  Trains GNN, GraphSAGE, GraphCL, TopoGCL, and internal InfoGraph baselines, and can still launch RGCL when selected, then writes the same JSON and summary CSV format.
- `scripts/train_ids_supervised_baselines.py`
  Trains the supervised GNN and GraphSAGE baselines separately from the unsupervised binary IDS pipeline.
- `requirements.txt`  
  Python dependencies for dataset construction and training.
- `results/`  
  Experiment outputs (JSON metrics and summary CSVs), currently including:
  - `results/nf_bot_iot/`
  - `results/nf_ton_iot/`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Build graph classification datasets

Run the dataset builder script (use `--help` to see all options):

```bash
python scripts/build_ids_graph_classification_dataset.py --help
```

Typical usage:

```bash
python scripts/build_ids_graph_classification_dataset.py
```

## 2) Train and evaluate models

Run the training/evaluation pipeline (use `--help` for available flags):

```bash
python scripts/train_ids_binary.py --help
```

Typical usage:

```bash
python scripts/train_ids_binary.py
```

InfoGraph is implemented internally for the local IDS `.npz` graph datasets and does not use `TUDataset` or download TU archives. If `rgcl` is selected, RGCL is launched from the external repository supplied with `--rgcl-dir`; by default, the command is equivalent to:

```bash
cd /home/kiwi-pandas/Documents/IDS_TopoGCL/RGCL/unsupervised_TU
python rgcl.py --seed $seed --DS $dataset
```

To run only the supervised baselines separately:

```bash
python scripts/train_ids_supervised_baselines.py
```

## Outputs

After running experiments, check:

- `results/<dataset>/results.json` for detailed run metrics.
- `results/<dataset>/summary.csv` for summarized performance tables.

## Notes

- If your datasets are tracked with Git LFS, run:

```bash
git lfs install
git lfs pull
```

- For reproducibility, keep Python/package versions consistent with `requirements.txt`.
