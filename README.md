# IDS_TopoGCL

This folder contains a compact, reproducible workflow for graph-based intrusion detection experiments using the `NF-BoT-IoT` and `NF-ToN-IoT` datasets.

## Repository layout

- `scripts/build_ids_graph_classification_dataset.py`  
  Builds graph classification datasets from tabular IDS data.
- `scripts/train_ids_graph_all_models.py`  
  Trains/evaluates graph-based models across the prepared datasets.
- `scripts/train_graphsage_gin_append_results.py`  
  Trains GraphSAGE and GIN on the same graph files/splits recorded in existing result JSON files and appends/upserts their metrics into the matching JSON/CSV outputs.
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
python scripts/train_ids_graph_all_models.py --help
```

Typical usage:

```bash
python scripts/train_ids_graph_all_models.py
```

To add GraphSAGE and GIN results to all existing experiment outputs without flags:

```bash
python scripts/train_graphsage_gin_append_results.py
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
