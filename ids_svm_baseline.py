#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch
from sklearn.svm import OneClassSVM

from scripts.data_corruptions import drop_edges


def score_stats(values: List[float]):
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(float(min(values)), 8),
        "mean": round(float(sum(values) / len(values)), 8),
        "max": round(float(max(values)), 8),
    }


@dataclass
class GraphWindow:
    x: torch.Tensor
    a_hat: torch.Tensor
    num_nodes: int


def parse_lines_to_windows(file_path: str, window_size: int = 300) -> Generator[List[Tuple[int, int, int, int, int, int, int, int, int]], None, None]:
    current_start = None
    rows: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                t = int(parts[0])
                row = (t, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]))
            except ValueError:
                continue
            if current_start is None:
                current_start = t
            if t >= current_start + window_size:
                if rows:
                    yield rows
                rows = []
                current_start = (t // window_size) * window_size
            rows.append(row)
    if rows:
        yield rows


def build_graph(rows: List[Tuple[int, int, int, int, int, int, int, int, int]]) -> GraphWindow:
    if not rows:
        x = torch.zeros(1, 15)
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
    a = torch.zeros((n, n))
    out_sums = torch.zeros((n, 6))
    in_sums = torch.zeros((n, 6))
    out_cnt = torch.zeros(n)
    in_cnt = torch.zeros(n)
    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]
        a[u, v] += 1.0
        a[v, u] += 1.0
        deg_out[u] += 1.0
        deg_in[v] += 1.0
        feat = torch.tensor([float(c2), float(c3), float(c5), float(c6), float(c7), float(c8)])
        out_sums[u] += feat
        in_sums[v] += feat
        out_cnt[u] += 1.0
        in_cnt[v] += 1.0
    deg = deg_in + deg_out
    out_mean = out_sums / out_cnt.clamp_min(1.0).unsqueeze(1)
    in_mean = in_sums / in_cnt.clamp_min(1.0).unsqueeze(1)
    x = torch.cat([deg_in.unsqueeze(1), deg_out.unsqueeze(1), deg.unsqueeze(1), out_mean, in_mean], dim=1)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1.0)
    a = a + torch.eye(n)
    d = a.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
    a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    return GraphWindow(x=x, a_hat=a_hat, num_nodes=n)


def iter_graphs(file_path: str, limit: Optional[int], window_size: int) -> List[GraphWindow]:
    out: List[GraphWindow] = []
    for rows in parse_lines_to_windows(file_path, window_size=window_size):
        out.append(build_graph(rows))
        if limit is not None and len(out) >= limit:
            break
    return out


def apply_edge_corruption(g: GraphWindow, rate: float, rng: torch.Generator) -> GraphWindow:
    if rate <= 0.0 or g.num_nodes <= 1:
        return g
    edge_set = []
    for u in range(g.num_nodes):
        for v in range(u + 1, g.num_nodes):
            if g.a_hat[u, v] > 0:
                edge_set.append((u, v))
    if not edge_set:
        return g
    edge_index = torch.tensor(edge_set, dtype=torch.long).t().contiguous()
    edge_index, _, _ = drop_edges(edge_index, rate=rate, rng=rng)
    a = torch.zeros((g.num_nodes, g.num_nodes))
    if edge_index.numel() > 0:
        src, dst = edge_index
        a[src, dst] = 1.0
        a[dst, src] = 1.0
    a = a + torch.eye(g.num_nodes)
    d = a.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
    a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    return GraphWindow(x=g.x, a_hat=a_hat, num_nodes=g.num_nodes)


def graph_to_vector(g: GraphWindow) -> List[float]:
    deg = g.a_hat.sum(dim=1)
    return [
        float(g.num_nodes),
        float(g.a_hat.sum().item()),
        float(deg.mean().item()),
        float(deg.std().item()) if g.num_nodes > 1 else 0.0,
        float(g.x.mean().item()),
        float(g.x.std().item()),
    ]


def roc_auc(scores_pos: List[float], scores_neg: List[float]) -> float:
    if not scores_pos or not scores_neg:
        return float("nan")
    greater = 0.0
    ties = 0.0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                greater += 1.0
            elif p == n:
                ties += 1.0
    return (greater + 0.5 * ties) / (len(scores_pos) * len(scores_neg))


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(sorted(values))
    idx = min(max(int(math.ceil(q * (len(t) - 1))), 0), len(t) - 1)
    return float(t[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth_path", required=True)
    ap.add_argument("--red_path", required=True)
    ap.add_argument("--window_size", type=int, default=300)
    ap.add_argument("--benign_limit", type=int, default=600)
    ap.add_argument("--mal_limit", type=int, default=300)
    ap.add_argument("--edge_drop_rate", type=float, default=0.0)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--out_json", default="svm_results.json")
    args = ap.parse_args()

    train_graphs = iter_graphs(args.auth_path, args.benign_limit, args.window_size)
    benign_test = iter_graphs(args.auth_path, args.mal_limit, args.window_size)
    mal_test = iter_graphs(args.red_path, args.mal_limit, args.window_size)

    rng = torch.Generator().manual_seed(args.random_seed)
    benign_test = [apply_edge_corruption(g, args.edge_drop_rate, rng) for g in benign_test]
    mal_test = [apply_edge_corruption(g, args.edge_drop_rate, rng) for g in mal_test]

    x_train = [graph_to_vector(g) for g in train_graphs]
    x_benign = [graph_to_vector(g) for g in benign_test]
    x_mal = [graph_to_vector(g) for g in mal_test]

    model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
    model.fit(x_train)
    benign_scores = (-model.decision_function(x_benign)).tolist()
    mal_scores = (-model.decision_function(x_mal)).tolist()

    thr = percentile(benign_scores, 0.95)
    tp = sum(s > thr for s in mal_scores)
    fp = sum(s > thr for s in benign_scores)
    fn = len(mal_scores) - tp
    tn = len(benign_scores) - fp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    out = {
        "dataset": "unknown",
        "method": "svm",
        "edge_drop_rate": args.edge_drop_rate,
        "metrics": {
            "accuracy": round(acc, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "fpr": round(fpr, 6),
            "auroc": round(roc_auc(mal_scores, benign_scores), 6),
        },
        "score_stats": {
            "benign": score_stats(benign_scores),
            "malicious": score_stats(mal_scores),
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
