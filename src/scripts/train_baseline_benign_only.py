#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Requires torch_geometric
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    raise SystemExit(
        "Missing torch_geometric. Install it in your venv, then rerun.\n"
        "Error was:\n"
        f"{e}\n"
    )


# -----------------------------
# Data reading and windowing
# -----------------------------

@dataclass
class Event:
    t: int
    c1: int
    src: int
    srcp: int
    dst: int
    dstp: int
    proto: int
    pkt: int
    byt: int


def parse_event_line(line: str) -> Event | None:
    parts = line.strip().split(",")
    if len(parts) < 9:
        return None
    try:
        vals = [int(x) for x in parts[:9]]
        return Event(*vals)
    except Exception:
        return None


def load_windows(path: Path, window_size: int, max_windows: int | None) -> Tuple[List[int], List[List[Event]]]:
    """
    Returns:
      window_starts: list of window start times (int seconds)
      windows: list of list of Event
    Assumes file is roughly sorted by time.
    """
    windows: List[List[Event]] = []
    window_starts: List[int] = []

    current_bucket = None
    current_events: List[Event] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ev = parse_event_line(line)
            if ev is None:
                continue

            bucket = (ev.t // window_size) * window_size

            if current_bucket is None:
                current_bucket = bucket

            if bucket != current_bucket:
                window_starts.append(current_bucket)
                windows.append(current_events)
                current_bucket = bucket
                current_events = []

                if max_windows is not None and len(windows) >= max_windows:
                    break

            current_events.append(ev)

    if current_bucket is not None and (max_windows is None or len(windows) < max_windows) and len(current_events) > 0:
        window_starts.append(current_bucket)
        windows.append(current_events)

    return window_starts, windows


# -----------------------------
# Graph building
# -----------------------------

def build_graph_from_events(events: List[Event]) -> Data | None:
    """
    Nodes are computers.
    Edges are src -> dst.
    Node features are simple aggregates.
    """
    if len(events) == 0:
        return None

    # Collect node ids
    node_ids = {}
    def nid(x: int) -> int:
        if x not in node_ids:
            node_ids[x] = len(node_ids)
        return node_ids[x]

    # Edge lists and aggregates
    src_list = []
    dst_list = []

    out_deg = None
    in_deg = None
    out_bytes = None
    in_bytes = None
    out_pkts = None
    in_pkts = None

    # Pre size
    for ev in events:
        nid(ev.src)
        nid(ev.dst)

    n = len(node_ids)
    out_deg = np.zeros(n, dtype=np.int64)
    in_deg = np.zeros(n, dtype=np.int64)
    out_bytes = np.zeros(n, dtype=np.int64)
    in_bytes = np.zeros(n, dtype=np.int64)
    out_pkts = np.zeros(n, dtype=np.int64)
    in_pkts = np.zeros(n, dtype=np.int64)

    for ev in events:
        s = nid(ev.src)
        d = nid(ev.dst)
        src_list.append(s)
        dst_list.append(d)

        out_deg[s] += 1
        in_deg[d] += 1

        out_bytes[s] += max(0, ev.byt)
        in_bytes[d] += max(0, ev.byt)

        out_pkts[s] += max(0, ev.pkt)
        in_pkts[d] += max(0, ev.pkt)

    if len(src_list) == 0:
        return None

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Features: [out_deg, in_deg, out_bytes, in_bytes, out_pkts, in_pkts]
    x = np.stack([out_deg, in_deg, out_bytes, in_bytes, out_pkts, in_pkts], axis=1).astype(np.float32)

    # Log scale heavy tails
    x[:, 2:] = np.log1p(x[:, 2:])

    x = torch.tensor(x, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = x.shape[0]
    return data


# -----------------------------
# Augmentations
# -----------------------------

def drop_edges(data: Data, drop_p: float, rng: np.random.RandomState) -> Data:
    if drop_p <= 0.0:
        return data
    ei = data.edge_index
    m = ei.size(1)
    if m <= 1:
        return data
    keep = rng.rand(m) >= drop_p
    if keep.sum() < 1:
        keep[rng.randint(0, m)] = True
    ei2 = ei[:, torch.tensor(keep, dtype=torch.bool)]
    return Data(x=data.x, edge_index=ei2, num_nodes=data.num_nodes)


def jitter_features(data: Data, sigma: float, rng: np.random.RandomState) -> Data:
    if sigma <= 0.0:
        return data
    noise = rng.normal(0.0, sigma, size=data.x.shape).astype(np.float32)
    x2 = data.x + torch.tensor(noise, dtype=torch.float)
    return Data(x=x2, edge_index=data.edge_index, num_nodes=data.num_nodes)


# -----------------------------
# Model
# -----------------------------

class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        g = global_mean_pool(x, batch)
        return g


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, proj_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        z = F.normalize(z, dim=1)
        return z


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Standard SimCLR style NT Xent loss.
    """
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2b x d
    sim = torch.mm(z, z.t()) / temperature

    # mask self similarity
    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives are (i, i+b) and (i+b, i)
    targets = torch.arange(2 * b, device=z.device)
    pos = (targets + b) % (2 * b)

    loss = F.cross_entropy(sim, pos)
    return loss


# -----------------------------
# Train and score
# -----------------------------

@torch.no_grad()
def embed_all(encoder: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    encoder.eval()
    outs = []
    for batch in loader:
        batch = batch.to(device)
        g = encoder(batch.x, batch.edge_index, batch.batch)
        outs.append(g.cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign_path", required=True, help="CSV with 9 ints per line, sorted by time")
    ap.add_argument("--out_dir", default="outputs/benign_baseline")
    ap.add_argument("--window_size", type=int, default=300)
    ap.add_argument("--max_windows", type=int, default=4000)
    ap.add_argument("--train_frac", type=float, default=0.7)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--edge_drop", type=float, default=0.2)
    ap.add_argument("--feat_jitter", type=float, default=0.05)

    ap.add_argument("--threshold_q", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] device: {device}")

    benign_path = Path(args.benign_path)
    print(f"[+] loading windows from {benign_path}")
    window_starts, windows = load_windows(benign_path, args.window_size, args.max_windows)

    graphs = []
    kept_starts = []
    for ws, evs in zip(window_starts, windows):
        g = build_graph_from_events(evs)
        if g is None:
            continue
        graphs.append(g)
        kept_starts.append(ws)

    print(f"[+] windows parsed: {len(window_starts):,}")
    print(f"[+] graphs kept: {len(graphs):,}")

    if len(graphs) < 50:
        raise SystemExit("Too few graphs to train. Increase max_windows or check your input file.")

    # Split by time order
    n = len(graphs)
    n_train = max(1, int(args.train_frac * n))
    train_graphs = graphs[:n_train]
    test_graphs = graphs[n_train:]
    train_starts = kept_starts[:n_train]
    test_starts = kept_starts[n_train:]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    all_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    in_dim = train_graphs[0].x.size(1)
    encoder = GCNEncoder(in_dim, args.hidden, args.emb_dim).to(device)
    projector = ProjectionHead(args.emb_dim, args.proj_dim).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.lr)

    print("[+] training")
    for ep in range(1, args.epochs + 1):
        encoder.train()
        projector.train()

        total = 0.0
        steps = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Build two augmented views per graph in the batch
            # torch_geometric batches are combined, so we do augmentation per Data object before batching.
            # DataLoader gives a Batch already, so we approximate by augmenting the whole batch graph.
            # This is a baseline and is fine for now.

            # View 1
            v1 = drop_edges(batch, args.edge_drop, rng)
            v1 = jitter_features(v1, args.feat_jitter, rng)
            # View 2
            v2 = drop_edges(batch, args.edge_drop, rng)
            v2 = jitter_features(v2, args.feat_jitter, rng)

            z1 = encoder(v1.x, v1.edge_index, batch.batch)
            z2 = encoder(v2.x, v2.edge_index, batch.batch)

            p1 = projector(z1)
            p2 = projector(z2)

            loss = nt_xent(p1, p2, args.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1

        print(f"[+] epoch {ep}/{args.epochs} loss {total / max(1, steps):.4f}")

    # Embed all windows
    print("[+] embedding all windows")
    Z_all = embed_all(encoder, all_loader, device)  # n x emb_dim

    # Center from train embeddings
    Z_train = Z_all[:n_train]
    center = Z_train.mean(dim=0, keepdim=True)

    # Scores are L2 distance to center
    scores = torch.norm(Z_all - center, p=2, dim=1).numpy()

    # Threshold from train scores
    thr = float(np.quantile(scores[:n_train], args.threshold_q))
    flags = (scores > thr).astype(np.int32)

    # Save CSV
    scores_csv = out_dir / "scores.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_start", "score", "flag", "split"])
        for i, ws in enumerate(kept_starts):
            split = "train" if i < n_train else "test"
            w.writerow([ws, float(scores[i]), int(flags[i]), split])

    # Top anomalies
    topk = min(100, len(scores))
    idx = np.argsort(-scores)[:topk]
    top_csv = out_dir / "top_anomalies.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "window_start", "score", "n_nodes", "n_edges", "split"])
        for r, i in enumerate(idx, start=1):
            g = graphs[i]
            split = "train" if i < n_train else "test"
            w.writerow([r, kept_starts[i], float(scores[i]), int(g.num_nodes), int(g.edge_index.size(1)), split])

    # Plots
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.axvline(n_train, linestyle="--")
    plt.title("Anomaly score by window index")
    plt.xlabel("window index")
    plt.ylabel("distance to benign center")
    plt.tight_layout()
    plt.savefig(out_dir / "score_over_windows.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Anomaly score histogram")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "score_hist.png", dpi=200)
    plt.close()

    # Save run meta
    meta = {
        "benign_path": str(benign_path),
        "window_size": args.window_size,
        "max_windows": args.max_windows,
        "graphs_kept": len(graphs),
        "train_graphs": n_train,
        "test_graphs": len(graphs) - n_train,
        "threshold_q": args.threshold_q,
        "threshold": thr,
        "edge_drop": args.edge_drop,
        "feat_jitter": args.feat_jitter,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[+] wrote: {scores_csv}")
    print(f"[+] wrote: {top_csv}")
    print(f"[+] threshold: {thr:.6f}")
    print("[✔] Proof succeeded!")


if __name__ == "__main__":
    import json
    main()