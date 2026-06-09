# IDS_TopoGCL

This folder contains a compact, reproducible workflow for graph-based intrusion detection experiments using the `NF-BoT-IoT` and `NF-ToN-IoT` datasets.

## Repository layout

- `scripts/build_ids_graph_classification_dataset.py`  
  Builds graph classification datasets from tabular IDS data.
- `scripts/train_ids_binary.py`
  Trains GraphCL/TopoGCL and launches external InfoGraph/RGCL baselines, then writes the same JSON and summary CSV format.
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

InfoGraph and RGCL are launched from the external repositories supplied with `--infograph-dir` and `--rgcl-dir`. By default, the commands are equivalent to:

```bash
cd /home/kiwi-pandas/Documents/IDS_TopoGCL/InfoGraph/unsupervised
python main.py --DS DATASET_NAME --lr 0.001 --num-gc-layers 3

cd /home/kiwi-pandas/Documents/IDS_TopoGCL/RGCL/unsupervised_TU
python rgcl.py --seed $seed --DS $dataset
```

To run the removed supervised baselines separately:

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
