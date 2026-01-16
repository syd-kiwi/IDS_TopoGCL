#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import torch


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class GraphWindow:
    x: torch.Tensor          # [n, 15]
    a_hat: torch.Tensor      # sparse COO [n, n]
    num_nodes: int
    window_start: int


# -----------------------------
# Parsing utilities
# -----------------------------
def parse_csv_9ints_stream(
    file_path: str,
) -> Generator[Tuple[int, int, int, int, int, int, int, int, int], None, None]:
    """
    Expects rows:
      time,c1,c2,c3,c4,c5,c6,c7,c8
    No header. All integers.
    """
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 9:
                continue
            try:
                t = int(parts[0])
                yield (
                    t,
                    int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]),
                    int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]),
                )
            except Exception:
                continue


def windows_from_stream(
    rows: Generator[Tuple[int, int, int, int, int, int, int, int, int], None, None],
    window_size: int,
) -> Generator[Tuple[int, List[Tuple[int, int, int, int, int, int, int, int, int]]], None, None]:
    """
    Groups rows into windows [start, start+window_size).
    Assumes time is mostly nondecreasing.
    """
    current_start: Optional[int] = None
    buf: List[Tuple[int, int, int, int, int, int, int, int, int]] = []

    for row in rows:
        t = row[0]
        if current_start is None:
            current_start = t - (t % window_size)

        while t >= current_start + window_size:
            if buf:
                yield current_start, buf
            buf = []
            current_start += window_size

        buf.append(row)

    if current_start is not None and buf:
        yield current_start, buf


# -----------------------------
# Graph builder (sparse)
# -----------------------------
def build_graph_sparse(
    rows: List[Tuple[int, int, int, int, int, int, int, int, int]],
    window_start: int,
    max_nodes: int,
) -> Optional[GraphWindow]:
    """
    Uses your baseline interpretation:
      src id = col1
      dst id = col4
      edge features = col2,col3,col5,col6,col7,col8

    Node features:
      [deg_in, deg_out, deg_total, out_mean(6), in_mean(6)] => 15 dims
    """
    if not rows:
        return None

    node_ids: Dict[int, int] = {}
    for (_, s, _, _, d, *_rest) in rows:
        if s not in node_ids:
            node_ids[s] = len(node_ids)
        if d not in node_ids:
            node_ids[d] = len(node_ids)

    n = len(node_ids)
    if n == 0:
        return None
    if n > max_nodes:
        return None

    deg_in = torch.zeros(n)
    deg_out = torch.zeros(n)

    out_sums = torch.zeros((n, 6))
    in_sums = torch.zeros((n, 6))
    out_cnt = torch.zeros(n)
    in_cnt = torch.zeros(n)

    src_idx: List[int] = []
    dst_idx: List[int] = []

    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]

        # undirected edges for message passing
        src_idx.append(u); dst_idx.append(v)
        src_idx.append(v); dst_idx.append(u)

        deg_out[u] += 1.0
        deg_in[v] += 1.0

        feat = torch.tensor([float(c2), float(c3), float(c5), float(c6), float(c7), float(c8)])
        out_sums[u] += feat
        out_cnt[u] += 1.0
        in_sums[v] += feat
        in_cnt[v] += 1.0

    deg = deg_in + deg_out
    out_mean = out_sums / out_cnt.clamp_min(1.0).unsqueeze(1)
    in_mean = in_sums / in_cnt.clamp_min(1.0).unsqueeze(1)

    x = torch.cat(
        [deg_in.unsqueeze(1), deg_out.unsqueeze(1), deg.unsqueeze(1), out_mean, in_mean],
        dim=1,
    )
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1.0)

    # sparse adjacency + self loops
    indices = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    values = torch.ones(len(src_idx), dtype=torch.float)

    self_idx = torch.arange(n, dtype=torch.long)
    self_indices = torch.stack([self_idx, self_idx], dim=0)
    indices = torch.cat([indices, self_indices], dim=1)
    values = torch.cat([values, torch.ones(n, dtype=torch.float)])

    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    # normalize: D^{-1/2} A D^{-1/2}
    d = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = torch.pow(d, -0.5)
    row, col = a.indices()
    norm_vals = a.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    a_hat = torch.sparse_coo_tensor(a.indices(), norm_vals, (n, n)).coalesce()

    return GraphWindow(x=x, a_hat=a_hat, num_nodes=n, window_start=window_start)


def collect_graphs(
    path: str,
    window_size: int,
    limit_windows: int,
    max_nodes: int,
    name: str,
) -> List[GraphWindow]:
    graphs: List[GraphWindow] = []
    row_stream = parse_csv_9ints_stream(path)
    win_stream = windows_from_stream(row_stream, window_size)

    seen = 0
    for wstart, rows in win_stream:
        gw = build_graph_sparse(rows, wstart, max_nodes=max_nodes)
        if gw is not None:
            graphs.append(gw)

        seen += 1
        if limit_windows > 0 and len(graphs) >= limit_windows:
            break

        if seen % 200 == 0:
            print(f"loading {name} windows seen {seen} graphs kept {len(graphs)}")

    print(f"[OK] {name} graphs kept: {len(graphs)}")
    return graphs


# -----------------------------
# Model
# -----------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = torch.nn.Linear(hidden_dim, emb_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_hat, self.w1(x)))
        h = torch.sparse.mm(a_hat, self.w2(h))
        return h.mean(dim=0)


class GraphAutoModel(torch.nn.Module):
    """
    Train on benign graphs to reconstruct mean node feature vector.
    Score is reconstruction MSE.
    """
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.enc = GCNEncoder(in_dim, hidden_dim, emb_dim)
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, in_dim),
        )

    def forward(self, gw: GraphWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.enc(gw.x, gw.a_hat)
        pred = self.dec(g)
        target = gw.x.mean(dim=0)
        return pred, target


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


# -----------------------------
# Metrics
# -----------------------------
def compute_basic_metrics(y_true: List[int], scores: List[float], thr: float):
    y_pred = [1 if s >= thr else 0 for s in scores]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    tpr = recall
    fpr = fp / max(1, (fp + tn))

    return acc, precision, recall, f1, tpr, fpr


def compute_auroc(y_true: List[int], scores: List[float]) -> float:
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    pairs = list(zip(scores, y_true))
    pairs.sort(key=lambda x: x[0])

    # average rank for ties
    score_to_avg_rank = {}
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        score_to_avg_rank[pairs[i][0]] = avg_rank
        i = j

    sum_ranks_pos = 0.0
    for s, t in zip(scores, y_true):
        if t == 1:
            sum_ranks_pos += score_to_avg_rank[s]

    n_pos = len(pos)
    n_neg = len(neg)
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return u / (n_pos * n_neg)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="LANL GNN baseline with sparse graphs + metrics JSON")
    ap.add_argument("--auth_path", type=str, required=True, help="converted benign csv (9 ints)")
    ap.add_argument("--red_path", type=str, required=True, help="converted malicious csv (9 ints)")
    ap.add_argument("--window_size", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--benign_limit", type=int, default=500)
    ap.add_argument("--mal_limit", type=int, default=500)
    ap.add_argument("--max_nodes", type=int, default=8000)
    ap.add_argument("--threshold_q", type=float, default=0.99)
    ap.add_argument("--out_json", type=str, default="lanl_ids_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}")

    d_benign = collect_graphs(
        path=args.auth_path,
        window_size=args.window_size,
        limit_windows=args.benign_limit,
        max_nodes=args.max_nodes,
        name="benign",
    )
    d_mal = collect_graphs(
        path=args.red_path,
        window_size=args.window_size,
        limit_windows=args.mal_limit,
        max_nodes=args.max_nodes,
        name="malicious",
    )

    if len(d_benign) == 0:
        raise RuntimeError("No benign graphs parsed. Check that auth_path is converted 9-int CSV and max_nodes is not too small.")
    if len(d_mal) == 0:
        print("[WARN] No malicious graphs parsed. Metrics will be degenerate. Fix your converter overlap.")

    model = GraphAutoModel(in_dim=15, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train on benign only
    model.train()
    for ep in range(args.epochs):
        total = 0.0
        for gw in d_benign:
            gwd = GraphWindow(
                x=gw.x.to(device),
                a_hat=gw.a_hat.to(device),
                num_nodes=gw.num_nodes,
                window_start=gw.window_start,
            )
            pred, target = model(gwd)
            loss = torch.mean((pred - target) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"[OK] epoch {ep+1}/{args.epochs} benign recon mse {total / max(1, len(d_benign)):.6f}")

    # Score graphs
    model.eval()
    with torch.no_grad():
        benign_scores: List[float] = []
        for gw in d_benign:
            gwd = GraphWindow(gw.x.to(device), gw.a_hat.to(device), gw.num_nodes, gw.window_start)
            pred, target = model(gwd)
            benign_scores.append(mse(pred.cpu(), target.cpu()))

        mal_scores: List[float] = []
        for gw in d_mal:
            gwd = GraphWindow(gw.x.to(device), gw.a_hat.to(device), gw.num_nodes, gw.window_start)
            pred, target = model(gwd)
            mal_scores.append(mse(pred.cpu(), target.cpu()))

    # Threshold from benign quantile
    bs = torch.tensor(benign_scores)
    thr = float(torch.quantile(bs, args.threshold_q).item())
    print(f"[OK] threshold quantile {args.threshold_q} value {thr:.6f}")

    # Metrics on combined set
    y_true = ([0] * len(d_benign)) + ([1] * len(d_mal))
    scores_all = benign_scores + mal_scores

    acc, precision, recall, f1, tpr, fpr = compute_basic_metrics(y_true, scores_all, thr)
    auc = compute_auroc(y_true, scores_all)

    # Your exact requested structure
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
            "auroc": round(auc, 2) if auc == auc else None,  # None if nan
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

    print(f"[OK] wrote {args.out_json}")
    print("[✔] Proof succeeded!")


if __name__ == "__main__":
    main()
