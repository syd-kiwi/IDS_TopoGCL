import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Tuple, Optional

import torch


@dataclass
class GraphWindow:
    """Graph data for a 300s window.

    Attributes:
        x: Node features [num_nodes, num_features].
        a_hat: Symmetrically normalized adjacency with self-loops [num_nodes, num_nodes].
        num_nodes: Number of nodes.
    """

    x: torch.Tensor
    a_hat: torch.Tensor
    num_nodes: int


def parse_lines_to_windows(
    file_path: str, window_size: int = 300
) -> Generator[List[Tuple[int, int, int, int, int, int, int, int, int]], None, None]:
    """Yield raw rows per 300s window from a CSV log.

    Row schema (all ints):
        0: time, 1: src, 2: src_attr, 3: flag, 4: dst, 5: dst_attr, 6: code6, 7: code7, 8: code8
    Lines must be time-sorted.
    """

    current_start = None
    rows: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                t = int(parts[0])
                r = (
                    t,
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    int(parts[7]),
                    int(parts[8]),
                )
            except ValueError:
                continue
            if current_start is None:
                current_start = t
            if t >= current_start + window_size:
                if rows:
                    yield rows
                rows = []
                current_start = (t // window_size) * window_size
            rows.append(r)
    if rows:
        yield rows


def build_graph(rows: List[Tuple[int, int, int, int, int, int, int, int, int]]) -> GraphWindow:
    """Construct node features and normalized adjacency from raw rows.

    Features per node: [deg_in, deg_out, deg_total, out_mean(c2,c3,c5,c6,c7,c8), in_mean(c2,c3,c5,c6,c7,c8)].
    Adjacency: undirected with self-loops, symmetric norm D^{-1/2}(A+I)D^{-1/2}.
    """

    if not rows:
        x = torch.ones(1, 3)
        return GraphWindow(x=x, a_hat=torch.eye(1), num_nodes=1)
    node_ids: Dict[int, int] = {}
    for _, s, _, _, d, *_ in rows:
        if s not in node_ids:
            node_ids[s] = len(node_ids)
        if d not in node_ids:
            node_ids[d] = len(node_ids)
    n = len(node_ids)
    deg_in = torch.zeros(n)
    deg_out = torch.zeros(n)
    # adjacency as counts
    a = torch.zeros((n, n))
    # aggregations: sums for outgoing and incoming for columns [2,3,5,6,7,8]
    idx_cols = [2, 3, 5, 6, 7, 8]
    out_sums = torch.zeros((n, len(idx_cols)))
    in_sums = torch.zeros((n, len(idx_cols)))
    out_cnt = torch.zeros(n)
    in_cnt = torch.zeros(n)
    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]
        a[u, v] += 1.0
        a[v, u] += 1.0
        deg_out[u] += 1.0
        deg_in[v] += 1.0
        # outgoing aggregates at src
        out_sums[u] += torch.tensor([float(c2), float(c3), float(c5), float(c6), float(c7), float(c8)])
        out_cnt[u] += 1.0
        # incoming aggregates at dst
        in_sums[v] += torch.tensor([float(c2), float(c3), float(c5), float(c6), float(c7), float(c8)])
        in_cnt[v] += 1.0
    deg = deg_in + deg_out
    # means
    out_mean = out_sums / out_cnt.clamp_min(1.0).unsqueeze(1)
    in_mean = in_sums / in_cnt.clamp_min(1.0).unsqueeze(1)
    x = torch.cat([deg_in.unsqueeze(1), deg_out.unsqueeze(1), deg.unsqueeze(1), out_mean, in_mean], dim=1)
    # per-node L1 norm
    x_norm = x.sum(dim=1, keepdim=True).clamp_min(1.0)
    x = x / x_norm
    a = a + torch.eye(n)
    d = a.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
    a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    return GraphWindow(x=x, a_hat=a_hat, num_nodes=n)


class GCNEmbedder(torch.nn.Module):
    """Minimal 2-layer GCN with mean pooling to graph embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(a_hat @ self.w1(x))
        h = a_hat @ self.w2(h)
        return h.mean(dim=0)  # graph embedding


def iter_graphs(
    file_path: str,
    limit: Optional[int],
) -> Iterable[GraphWindow]:
    count = 0
    for edges in parse_lines_to_windows(file_path):
        yield build_graph(edges)
        count += 1
        if limit is not None and count >= limit:
            break


def compute_center(model: GCNEmbedder, graphs: Iterable[GraphWindow]) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    with torch.no_grad():
        for g in graphs:
            embs.append(model(g.x, g.a_hat))
    if not embs:
        return torch.zeros(model.w2.out_features)
    return torch.stack(embs, dim=0).mean(dim=0)


def train_one_class(
    model: GCNEmbedder,
    graphs: List[GraphWindow],
    center: torch.Tensor,
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    center = center.detach()
    for e in range(epochs):
        total = 0.0
        last = -1
        m = max(len(graphs), 1)
        for i, g in enumerate(graphs):
            z = model(g.x, g.a_hat)
            loss = torch.mean((z - center).pow(2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            p = int(100 * (i + 1) / m)
            if p != last and p % 10 == 0:
                print(f"epoch {e + 1}/{epochs} {p}%", flush=True)
                last = p
        print(f"epoch {e + 1}/{epochs} loss={total / m:.6f}", flush=True)


def distances(
    model: GCNEmbedder, graphs: List[GraphWindow], center: torch.Tensor, tag: Optional[str] = None
) -> List[float]:
    ds: List[float] = []
    m = max(len(graphs), 1)
    last = -1
    with torch.no_grad():
        for i, g in enumerate(graphs):
            z = model(g.x, g.a_hat)
            ds.append(float(torch.norm(z - center, p=2)))
            if tag is not None:
                p = int(100 * (i + 1) / m)
                if p != last and p % 10 == 0:
                    print(f"{tag} {p}%", flush=True)
                    last = p
    return ds


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(sorted(values))
    idx = min(max(int(math.ceil(q * (len(t) - 1))), 0), len(t) - 1)
    return float(t[idx])


def roc_auc(scores: List[float], labels: List[int]) -> float:
    # AUC via rank statistic; higher score = more anomalous
    if not scores or len(set(labels)) < 2:
        return 0.5
    t_scores = torch.tensor(scores)
    order = torch.argsort(t_scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = torch.tensor([i for i, y in enumerate(labels) if y == 1])
    n_pos = len(pos)
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = ranks[pos].sum().item()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN-based IDS baseline (single-file)")
    parser.add_argument(
        "--auth_path",
        type=str,
        default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/OPTC/auth_optc.txt",
    )
    parser.add_argument(
        "--red_path",
        type=str,
        default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/OPTC/redteam_optc.txt",
    )
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--benign_limit", type=int, default=2000)
    parser.add_argument("--mal_limit", type=int, default=2000)
    parser.add_argument("--threshold_q", type=float, default=0.99)
    parser.add_argument("--out_json", type=str, default="ids_results.json")
    args = parser.parse_args()

    assert os.path.exists(args.auth_path), "auth file not found"
    assert os.path.exists(args.red_path), "redteam file not found"

    # Load graphs with percent progress
    def collect_graphs(path: str, limit: Optional[int], tag: str) -> List[GraphWindow]:
        out: List[GraphWindow] = []
        last = -1
        for g in iter_graphs(path, limit=limit):
            out.append(g)
            if limit is not None and limit > 0:
                p = int(100 * len(out) / limit)
                if p != last and p <= 100 and p % 10 == 0:
                    print(f"loading {tag} {p}%", flush=True)
                    last = p
        return out

    benign_graphs = collect_graphs(args.auth_path, args.benign_limit, "benign")
    mal_graphs = collect_graphs(args.red_path, args.mal_limit, "malicious")
    if not benign_graphs or not mal_graphs:
        raise RuntimeError("Insufficient graphs parsed from inputs")

    in_dim = benign_graphs[0].x.shape[1]
    model = GCNEmbedder(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.emb_dim)

    # Center from initial pass
    center = compute_center(model, benign_graphs)
    train_one_class(model, benign_graphs, center, epochs=args.epochs, lr=args.lr)

    # Recompute center after training for stability
    center = compute_center(model, benign_graphs)
    d_benign = distances(model, benign_graphs, center, tag="score benign")
    d_mal = distances(model, mal_graphs, center, tag="score malicious")
    thr = percentile(d_benign, args.threshold_q)

    y_true = [0] * len(d_benign) + [1] * len(d_mal)
    y_score = d_benign + d_mal
    y_pred = [1 if s > thr else 0 for s in y_score]

    tp = sum(1 for yp, yt in zip(y_pred[len(d_benign) :], y_true[len(d_benign) :]) if yp == 1 and yt == 1)
    fn = sum(1 for yp, yt in zip(y_pred[len(d_benign) :], y_true[len(d_benign) :]) if yp == 0 and yt == 1)
    fp = sum(1 for yp, yt in zip(y_pred[: len(d_benign)], y_true[: len(d_benign)]) if yp == 1 and yt == 0)
    tn = sum(1 for yp, yt in zip(y_pred[: len(d_benign)], y_true[: len(d_benign)]) if yp == 0 and yt == 0)

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    auc = roc_auc(y_score, y_true)

    results = {
        "num_benign_windows": len(d_benign),
        "num_mal_windows": len(d_mal),
        "threshold_q": round(args.threshold_q, 2),
        "threshold": round(thr, 2),
        "metrics": {
            "accuracy": round(acc, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tpr": round(tpr, 2),
            "fpr": round(fpr, 2),
            "auroc": round(auc, 2),
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "emb_dim": args.emb_dim,
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


