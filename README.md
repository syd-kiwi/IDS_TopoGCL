# IDS_TopoGCL

This folder contains a compact, reproducible workflow for graph-based intrusion detection experiments using the `NF-BoT-IoT` and `NF-ToN-IoT` datasets.

## Repository layout

- `scripts/build_ids_graph_classification_dataset.py`  
  Builds graph classification datasets from tabular IDS data.
- `scripts/train_ids_graph_all_models.py`  
  Trains/evaluates graph-based models across the prepared datasets.
- `scripts/run_infograph_baseline.py`  
  Trains only the InfoGraph baseline and appends its summary row to the configured existing results CSV while updating the matching JSON.
- `scripts/train_graphsage_gin_append_results.py`  
  Trains GraphSAGE and GIN on the configured graph files/splits and appends/upserts their metrics into the configured JSON/CSV outputs.
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

To train only InfoGraph and append its row to the configured existing summary CSV without flags:

```bash
python scripts/run_infograph_baseline.py
```

The script defaults to `NF-ToN-IoT` with `ton_results_25%.json` and `ton_summary_25%.csv`. Use `--dataset bot` for the `NF-BoT-IoT` 5% paths, or pass `--graph-dir`, `--out-json`, and `--out-csv` to override paths.

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
