# TRACE

**Topology-Aware Graph Contrastive Learning for Resilient Intrusion Detection**

Accepted at **ARTMAN 2026**, co-located with IEEE ACSAC. The workshop takes place on **December 8, 2026, in Los Angeles, California**.

[Workshop and program](https://artman-workshop.gitlab.io/) · [Paper source](paper/main.tex) · [Run guide](docs/RUNNING.md) · [Saved results](results/README.md)

## What TRACE does

TRACE learns representations of benign activity graphs using graph contrastive learning and topological summaries. It scores new graphs by their distance from benign training embeddings in the projected representation space.

Representation learning uses benign graphs only. **Labeled benign and malicious validation graphs are used to select the detection threshold.** This is not a completely label-free detection pipeline.

## Quick start

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
git clone https://github.com/syd-kiwi/IDS_TopoGCL.git
cd IDS_TopoGCL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the datasets separately and place them at the paths in the [run guide](docs/RUNNING.md). Then run TRACE on one dataset:

```bash
python scripts/topids_topogcl_improved.py \
  --dataset streamspot --epochs 12 --seeds 42,43,44 \
  --mode graph_topo --out-dir outputs/trace_streamspot
```

New runs should write to `outputs/` to keep the saved experimental results intact. TRACE currently runs on CPU; the baseline runners can use CUDA when available.

## Repository guide

| Location | Contents |
| --- | --- |
| `scripts/topids_topogcl_improved.py` | TRACE training and evaluation |
| `scripts/topids_topogcl_prototype.py` | Earlier prototype, retained for historical comparisons |
| `scripts/train_ids_gnn.py` | Supervised GNN baseline |
| `scripts/train_ids_cl_baselines.py` | Benign-only GraphCL baseline |
| `scripts/ids_graph_data.py` | Shared data loading and evaluation helpers |
| `paper/` | LaTeX manuscript and figures |
| `results/` | Saved experiment outputs and plots |
| `utils/` | Compatibility classes for legacy dataset pickles |

Older filenames retain `topids` and `topogcl` for compatibility. The proposed method is **TRACE**.

## Recorded TRACE results

Mean ± standard deviation of test F1, read from the saved joint graph and topology runs:

| Dataset | Seeds | F1 |
| --- | --- | --- |
| StreamSpot | 42, 43, 44 | 0.923 ± 0.019 |
| GraSec-IoT | 42, 43, 44 | 0.790 ± 0.065 |
| Unicorn Wget | 42–46 | 0.583 ± 0.053 |

See the [result file guide](results/README.md) for exact sources and the [manuscript](paper/4_Evaluation.tex) for baseline comparisons and limitations. These are archived measurements, not results from a new run. TRACE does not outperform the baselines uniformly across datasets.
