#!/bin/bash
#SBATCH --job-name=ids_topogcl
#SBATCH --output=logs/ids_topogcl_%j.out
#SBATCH --error=logs/ids_topogcl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p logs

cd /home/kiwi-pandas/Documents/IDS_TopoGCL

source .venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

python3 train_ids_gnn.py \
  --dataset all \
  --data-root data \
  --epochs 10 \
  --seeds 42,43,44

python3 train_ids_cl_baselines.py \
  --dataset all \
  --data-root data \
  --models graphcl,infograph \
  --epochs-graphcl 10 \
  --epochs-infograph 10 \
  --seeds 42,43,44

python3 topids_topogcl_prototype.py \
  --dataset all \
  --epochs 12

python3 topids_topogcl_improved.py \
  --dataset all \
  --epochs 12 \
  --seeds 42,43,44 \
  --mode graph_topo
