# TopIDS

End-to-end graph intrusion detection prototype based on **TopoGCL** (Topological Graph Contrastive Learning, AAAI 2024). A single Python script trains on benign provenance graphs and scores test graphs as benign or malicious.

## Repository

```
topIDS/
├── data/
│   ├── streamspot/     # StreamSpot provenance graphs (all.tsv)
│   └── grasec-iot/     # GraSec-IoT flow-graph snapshots (JSON)
└── topids_topogcl_prototype.py
```

## Pipeline

1. **Load data** — Parse real graph edges and node features (no kNN re-wiring). Large graphs are capped at 512 nodes.
2. **Train TopoGCL** (benign only) — Two channels trained with symmetric InfoNCE and in-batch negatives:
   - **Graph channel (H):** 2-layer GIN encoder; readout = concat(sum, mean, max); augmented views via node drop + feature mask.
   - **Topology channel (Z):** 0-dim extended persistence (degree + closeness filtrations) → Extended Persistence Landscapes (EPL) → 5-layer MLP (ETL). EPL is computed once per graph; contrastive views use dropout + Gaussian noise on standardized EPL.
   - **Loss:** `L = α·InfoNCE(H, H') + β·InfoNCE(Z, Z')`
3. **Score** — kNN distance from each graph embedding to benign train embeddings (joint H+Z).
4. **Detect** — Pick the threshold on a **balanced 1:1 validation set** (benign : malicious) that maximizes F1; apply it to a **balanced 1:1 test set**.

Training uses benign graphs only. Malicious graphs appear in validation (threshold tuning) and test (evaluation).

## Datasets

| Dataset | Description | Labels |
|---------|-------------|--------|
| **StreamSpot** | 600 provenance graphs | gid 300–399 = attack |
| **GraSec-IoT** | IoT flow-graph snapshots | `connection.Label` index 6 = benign |

## Run

**Dependencies:** Python 3.10+, NumPy, SciPy, scikit-learn, NetworkX, PyTorch.

```bash
python3 topids_topogcl_prototype.py --dataset all --epochs 12
```

Per-dataset settings (`DATASET_CONFIG` in the script):

| Dataset | α | β | kNN k |
|---------|---|---|-------|
| streamspot | 1.0 | 0.0 | 5 |
| grasec | 0.5 | 0.5 | 5 |

Other useful flags: `--dataset streamspot|grasec`, `--seed 42`, `--epochs 12`.

## Key Results

Balanced test sets (1:1 benign : malicious), F1-optimal threshold from validation, 12 epochs, seed 42.

| Dataset | Test size | ROC-AUC | F1 | Recall | FPR |
|---------|-----------|---------|-----|--------|-----|
| StreamSpot | 80 + 80 | 0.9527 | 0.9455 | 0.9750 | 0.0875 |
| GraSec-IoT | 22 + 22 | 0.8822 | 0.8571 | 0.8182 | 0.0909 |

**StreamSpot** — threshold = 0.113 (val F1 = 0.927); confusion matrix: TN=73, FP=7, FN=2, TP=78.

**GraSec-IoT** — threshold = 0.502 (val F1 = 0.727); confusion matrix: TN=20, FP=2, FN=4, TP=18.
