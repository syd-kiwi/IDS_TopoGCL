#!/bin/bash
#SBATCH --job-name=trace
#SBATCH --output=logs/trace_%j.out
#SBATCH --error=logs/trace_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Submit from the repository root after creating logs/.
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
source .venv/bin/activate
RUN_DIR="outputs/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python - <<'CHECK'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
CHECK

python scripts/train_ids_gnn.py \
  --dataset all --data-root data --epochs 10 --seeds 42,43,44 \
  --out-json "$RUN_DIR/gnn/metrics.json" \
  --out-csv "$RUN_DIR/gnn/summary.csv" \
  --out-scores-csv "$RUN_DIR/gnn/scores.csv"

python scripts/train_ids_cl_baselines.py \
  --dataset all --data-root data --models graphcl \
  --epochs-graphcl 10 --seeds 42,43,44 \
  --out-json "$RUN_DIR/graphcl/metrics.json" \
  --out-csv "$RUN_DIR/graphcl/summary.csv" \
  --out-scores-csv "$RUN_DIR/graphcl/scores.csv"

# TRACE currently runs on CPU; the GPU reservation is for the baselines.
python scripts/topids_topogcl_improved.py \
  --dataset all --data-root data --epochs 12 --seeds 42,43,44 \
  --mode graph_topo --out-dir "$RUN_DIR/trace"
