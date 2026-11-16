# Running TRACE

Run commands from the repository root with the virtual environment activated. Python 3.10+ is required. Dependencies are listed in `requirements.txt`; they are not version-pinned, so this is not a frozen reproduction environment.

## Dataset paths

Datasets are not included in this repository. Use the dataset references in [the manuscript](../paper/4_Evaluation.tex) and its [bibliography](../paper/ref.bib) to obtain the source data. The loaders expect:

| CLI name | Dataset | Path under the data directory |
| --- | --- | --- |
| `streamspot` | StreamSpot | `streamspot/all.tsv` |
| `grasec` | GraSec-IoT | `grasec-iot/graph_json/Graph_JSON/{train,eval,test}/data_*.json` |
| `wget` | Unicorn Wget | `wget/graphs.pkl` |

The default data directory is `data/` at the repository root. TRACE and both current baseline runners accept `--data-root /path/to/data`. The older prototype uses the repository's `data/` directory.

## TRACE

Use the improved script for TRACE. The prototype is an earlier experiment, not the primary entry point.

```bash
python scripts/topids_topogcl_improved.py \
  --dataset streamspot --epochs 12 --seeds 42,43,44 \
  --mode graph_topo --out-dir outputs/trace_streamspot

python scripts/topids_topogcl_improved.py \
  --dataset grasec --epochs 12 --seeds 42,43,44 \
  --mode graph_topo --out-dir outputs/trace_grasec

python scripts/topids_topogcl_improved.py \
  --dataset wget --epochs 12 --seeds 42,43,44,45,46 \
  --mode graph_topo --out-dir outputs/trace_wget
```

Use `--dataset all` to run all three datasets with the same seed list. Use `--mode graph_only` or `--mode topo_only` for channel ablations. Run `python scripts/topids_topogcl_improved.py --help` for all options.

Each run writes configuration and per-seed metrics to JSON, aggregate metrics to a summary CSV, and validation/test anomaly scores to a scores CSV. Output filenames retain the historical `topids_` prefix. Reusing the same output directory and dataset/mode will overwrite those files, so choose a new directory for each experiment.

The encoder is trained on benign graphs. Validation includes labeled attacks for F1 threshold selection; test labels are used for evaluation. TRACE's loader constructs balanced validation/test sets. Inspect saved split counts before comparing runs across methods or revisions; the archived baseline and TRACE results should not be assumed to use identical test samples merely because they share a seed.

TRACE currently keeps its models and tensors on CPU. Reserving a GPU alone will not accelerate this script. The GNN and GraphCL baseline runners select CUDA when available.

## Baselines

These examples run StreamSpot with three seeds. Change `--dataset` and the seed list as needed. Separate output paths protect archived results.

```bash
python scripts/train_ids_gnn.py \
  --dataset streamspot --data-root data --epochs 10 --seeds 42,43,44 \
  --out-json outputs/gnn/metrics.json \
  --out-csv outputs/gnn/summary.csv \
  --out-scores-csv outputs/gnn/scores.csv

python scripts/train_ids_cl_baselines.py \
  --dataset streamspot --data-root data --models graphcl \
  --epochs-graphcl 10 --seeds 42,43,44 \
  --out-json outputs/graphcl/metrics.json \
  --out-csv outputs/graphcl/summary.csv \
  --out-scores-csv outputs/graphcl/scores.csv
```

Only `graphcl` is supported by the current contrastive baseline runner. Some archived outputs also contain InfoGraph from earlier experiments.

## Slurm

From the repository root:

```bash
mkdir -p logs
sbatch scripts/run_ids_gpu.sbatch
```

The batch script runs both current baselines and TRACE, writes to `outputs/slurm_JOBID/`, and uses the submission directory as the project root. Adjust Slurm partition/account/resource settings for your cluster. It expects `.venv/` and `data/` in that directory. Its GPU reservation serves the baseline runners; TRACE remains on CPU.

## Paper and maintenance

The manuscript entry point is `paper/main.tex`. With a LaTeX installation and `latexmk`:

```bash
cd paper
latexmk -pdf main.tex
```

The checked-in manuscript is a source snapshot. Verify it against your final accepted version before redistributing it as the camera-ready paper. Publication identifiers are not supplied here because no DOI or proceedings URL has been verified.

Keep saved results unchanged. Add new experiments under `outputs/`, and retain the commands, data version, environment, and seeds before promoting any new results. Review repository instructions in [AGENTS.md](../AGENTS.md) before making research changes.
