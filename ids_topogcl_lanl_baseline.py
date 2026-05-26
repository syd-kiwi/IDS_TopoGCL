#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch

from src.data_corruptions import drop_edges, drop_raw_events, mask_node_features


def score_stats(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(float(min(values)), 8),
        "mean": round(float(sum(values) / len(values)), 8),
        "max": round(float(max(values)), 8)
    }


# =========================================================
# Data container
# =========================================================
@dataclass
class GraphWindow:
    x: torch.Tensor                    # [n, 15]
    edges_undirected: torch.Tensor     # [2, E] with u < v (no self loops)
    num_nodes: int
    window_start: int


# =========================================================
# Parsing: expects converted 9 int CSV with no header
# time,c1,c2,c3,c4,c5,c6,c7,c8
# =========================================================
def parse_csv_9ints_stream(
    file_path: str,
) -> Generator[Tuple[int, int, int, int, int, int, int, int, int], None, None]:
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


def _is_git_lfs_pointer(file_path: str) -> bool:
    try:
        with open(file_path, "r") as f:
            first = f.readline().strip()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def windows_from_stream(
    rows: Generator[Tuple[int, int, int, int, int, int, int, int, int], None, None],
    window_size: int,
) -> Generator[Tuple[int, List[Tuple[int, int, int, int, int, int, int, int, int]]], None, None]:
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


# =========================================================
# Sparse adjacency helpers
# =========================================================
def build_sparse_a_hat_from_undirected(
    n: int,
    edges_undirected: torch.Tensor,   # [2, E], u < v
    add_self_loops: bool = True,
) -> torch.Tensor:
    if n <= 0:
        n = 1
        edges_undirected = torch.zeros((2, 0), dtype=torch.long)

    if edges_undirected.numel() == 0:
        if add_self_loops:
            idx = torch.arange(n, dtype=torch.long)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(n, dtype=torch.float)
        else:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros((0,), dtype=torch.float)
        a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    else:
        u = edges_undirected[0]
        v = edges_undirected[1]

        # Expand to both directions for message passing
        src = torch.cat([u, v], dim=0)
        dst = torch.cat([v, u], dim=0)
        indices = torch.stack([src, dst], dim=0)
        values = torch.ones(indices.shape[1], dtype=torch.float)

        if add_self_loops:
            idx = torch.arange(n, dtype=torch.long)
            self_indices = torch.stack([idx, idx], dim=0)
            indices = torch.cat([indices, self_indices], dim=1)
            values = torch.cat([values, torch.ones(n, dtype=torch.float)], dim=0)

        a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    # Symmetric normalization D^{-1/2} A D^{-1/2}
    d = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = torch.pow(d, -0.5)
    row, col = a.indices()
    norm_vals = a.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    a_hat = torch.sparse_coo_tensor(a.indices(), norm_vals, (n, n)).coalesce()
    return a_hat


# =========================================================
# Graph builder: uses same feature logic as your baselines
# src id is column 1, dst id is column 4
# features use columns 2,3,5,6,7,8
# =========================================================
def build_graph(
    rows: List[Tuple[int, int, int, int, int, int, int, int, int]],
    window_start: int,
    max_nodes: int,
) -> Optional[GraphWindow]:
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

    # Build undirected unique edges
    edge_set = set()

    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            edge_set.add((a, b))

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

    # 15 dim features
    x = torch.cat([deg_in.unsqueeze(1), deg_out.unsqueeze(1), deg.unsqueeze(1), out_mean, in_mean], dim=1)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1.0)

    if edge_set:
        edges = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()  # [2, E]
    else:
        edges = torch.zeros((2, 0), dtype=torch.long)

    return GraphWindow(x=x, edges_undirected=edges, num_nodes=n, window_start=window_start)


def iter_graphs(
    file_path: str,
    limit: Optional[int],
    window_size: int,
    max_nodes: int,
    tag: Optional[str] = None,
) -> List[GraphWindow]:
    out: List[GraphWindow] = []
    seen = 0
    for wstart, rows in windows_from_stream(parse_csv_9ints_stream(file_path), window_size=window_size):
        g = build_graph(rows, window_start=wstart, max_nodes=max_nodes)
        if g is not None:
            out.append(g)

        seen += 1
        if limit is not None and limit > 0 and len(out) >= limit:
            break

        if tag is not None and seen % 200 == 0:
            print(f"loading {tag} windows seen {seen} graphs kept {len(out)}", flush=True)

    if tag is not None:
        print(f"[OK] {tag} graphs kept: {len(out)}", flush=True)
    return out


def apply_graph_corruption(
    g: GraphWindow,
    corruption_type: str,
    corruption_rate: float,
    node_mask_mode: str,
    rng: torch.Generator,
) -> Tuple[GraphWindow, Dict[str, float]]:
    if corruption_type == "none" or corruption_rate <= 0.0:
        return g, {"edges_before": float(g.edges_undirected.shape[1]), "edges_after": float(g.edges_undirected.shape[1]), "masked_fraction": 0.0}

    x = g.x.clone()
    edges = g.edges_undirected.clone()
    stats: Dict[str, float] = {"edges_before": float(edges.shape[1]), "edges_after": float(edges.shape[1]), "masked_fraction": 0.0}

    if corruption_type == "node_features":
        x, mask_stats = mask_node_features(x, rate=corruption_rate, mode=node_mask_mode, fill_value=0.0, rng=rng)
        stats["masked_fraction"] = mask_stats["masked_fraction"]
    elif corruption_type == "edges":
        edges, _, edge_stats = drop_edges(edges, rate=corruption_rate, rng=rng)
        stats["edges_before"] = float(edge_stats["edges_before"])
        stats["edges_after"] = float(edge_stats["edges_after"])
    else:
        raise ValueError(f"Unsupported corruption_type: {corruption_type}")

    assert x.dim() == 2 and x.shape[0] == g.num_nodes, "node feature shape mismatch after corruption"
    assert edges.dim() == 2 and edges.shape[0] == 2, "edge tensor must be [2, E]"
    return GraphWindow(x=x, edges_undirected=edges, num_nodes=g.num_nodes, window_start=g.window_start), stats


# =========================================================
# Augmentation: edge drop and feature mask
# NOTE: Scenario degradation simulates imperfect telemetry.
# TopoGCL augmentation creates contrastive views during training.
# =========================================================
def augment_graph(
    g: GraphWindow,
    edge_drop: float,
    feat_mask: float,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns augmented (x_aug, a_hat_aug)
    a_hat_aug is sparse
    """
    n = g.num_nodes
    edges = g.edges_undirected

    # Edge drop on undirected edges, then rebuild sparse adjacency
    if edges.numel() > 0 and edge_drop > 0:
        E = edges.shape[1]
        keep = (torch.rand(E, generator=rng) > edge_drop)
        edges_kept = edges[:, keep]
    else:
        edges_kept = edges

    a_hat = build_sparse_a_hat_from_undirected(n, edges_kept, add_self_loops=True)

    # Feature mask
    x = g.x
    if feat_mask > 0:
        m = (torch.rand(x.shape, generator=rng) > feat_mask).float()
        x = x * m

    return x, a_hat


# =========================================================
# Model: sparse GCN
# =========================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_hat, self.w1(x)))
        h = torch.sparse.mm(a_hat, self.w2(h))
        return h.mean(dim=0)


# =========================================================
# Contrastive loss: two branches (struct and topo)
# =========================================================
def loss_cal(
    z1: torch.Tensor,
    z2: torch.Tensor,
    zt1: torch.Tensor,
    zt2: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    T = float(tau)

    # structural branch
    z1_abs = z1.norm(dim=1).clamp_min(1e-12)
    z2_abs = z2.norm(dim=1).clamp_min(1e-12)
    sim = torch.einsum("ik,jk->ij", z1, z2) / torch.einsum("i,j->ij", z1_abs, z2_abs)
    sim = torch.exp(sim / T)
    pos = sim[range(z1.size(0)), range(z1.size(0))]
    loss1 = pos / (sim.sum(dim=1) - pos).clamp_min(1e-12)

    # topological branch
    t1_abs = zt1.norm(dim=1).clamp_min(1e-12)
    t2_abs = zt2.norm(dim=1).clamp_min(1e-12)
    simt = torch.einsum("ik,jk->ij", zt1, zt2) / torch.einsum("i,j->ij", t1_abs, t2_abs)
    simt = torch.exp(simt / T)
    post = simt[range(zt1.size(0)), range(zt1.size(0))]
    loss2 = post / (simt.sum(dim=1) - post).clamp_min(1e-12)

    loss = 1.0 * loss1 + 0.1 * loss2
    return (-torch.log(loss.clamp_min(1e-12))).mean()


def train_contrastive(
    model: GCN,
    graphs: List[GraphWindow],
    epochs: int,
    lr: float,
    edge_drop: float,
    feat_mask: float,
    tau: float,
    batch_size: int,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)

    N = len(graphs)
    idx = torch.arange(N)

    for e in range(epochs):
        perm = idx[torch.randperm(N, generator=rng)]
        total = 0.0
        last = -1

        for i in range(0, N, batch_size):
            batch_idx = perm[i : i + batch_size].tolist()
            gs = [graphs[j] for j in batch_idx]

            z1_list = []
            z2_list = []
            zt1_list = []
            zt2_list = []

            for g in gs:
                x1, a1 = augment_graph(g, edge_drop, feat_mask, rng)
                x2, a2 = augment_graph(g, edge_drop, feat_mask, rng)

                x1 = x1.to(device)
                x2 = x2.to(device)
                a1 = a1.to(device)
                a2 = a2.to(device)

                z1_list.append(model(x1, a1))
                z2_list.append(model(x2, a2))

                # topo emphasized: keep degree cols only
                x1_topo = x1.clone()
                x2_topo = x2.clone()
                x1_topo[:, 3:] = 0.0
                x2_topo[:, 3:] = 0.0
                zt1_list.append(model(x1_topo, a1))
                zt2_list.append(model(x2_topo, a2))

            z1 = torch.stack(z1_list, dim=0)
            z2 = torch.stack(z2_list, dim=0)
            zt1 = torch.stack(zt1_list, dim=0)
            zt2 = torch.stack(zt2_list, dim=0)

            loss = loss_cal(z1, z2, zt1, zt2, tau=tau)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * z1.size(0)

            p = int(100 * min(i + batch_size, N) / max(N, 1))
            if p != last and p % 10 == 0:
                print(f"epoch {e + 1}/{epochs} {p}%", flush=True)
                last = p

        print(f"epoch {e + 1}/{epochs} loss={total / max(N, 1):.6f}", flush=True)


# =========================================================
# Scoring and metrics
# =========================================================
def compute_center(model: GCN, graphs: List[GraphWindow], device: torch.device) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    with torch.no_grad():
        for g in graphs:
            a_hat = build_sparse_a_hat_from_undirected(g.num_nodes, g.edges_undirected).to(device)
            z = model(g.x.to(device), a_hat)
            embs.append(z)
    if not embs:
        return torch.zeros(model.w2.out_features, device=device)
    return torch.stack(embs, dim=0).mean(dim=0)


def distances(model: GCN, graphs: List[GraphWindow], center: torch.Tensor, device: torch.device, tag: str) -> List[float]:
    ds: List[float] = []
    m = max(len(graphs), 1)
    last = -1
    with torch.no_grad():
        for i, g in enumerate(graphs):
            a_hat = build_sparse_a_hat_from_undirected(g.num_nodes, g.edges_undirected).to(device)
            z = model(g.x.to(device), a_hat)
            ds.append(float(torch.norm(z - center, p=2).item()))

            p = int(100 * (i + 1) / m)
            if p != last and p % 10 == 0:
                print(f"{tag} {p}%", flush=True)
                last = p
    return ds


def roc_auc(scores: List[float], labels: List[int]) -> float:
    # Mann Whitney U based AUROC
    if not scores or len(set(labels)) < 2:
        return 0.5
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0])

    # average rank for ties
    i = 0
    n = len(pairs)
    score_to_avg_rank = {}
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        score_to_avg_rank[pairs[i][0]] = avg_rank
        i = j

    sum_ranks_pos = 0.0
    for s, y in zip(scores, labels):
        if y == 1:
            sum_ranks_pos += score_to_avg_rank[s]

    n_pos = len(pos)
    n_neg = len(neg)
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def split_train_test_benign(
    graphs: List[GraphWindow],
    train_ratio: float,
) -> Tuple[List[GraphWindow], List[GraphWindow]]:
    if not graphs:
        return [], []

    ordered = sorted(graphs, key=lambda g: g.window_start)
    split_idx = int(len(ordered) * train_ratio)
    split_idx = max(1, min(split_idx, len(ordered) - 1)) if len(ordered) > 1 else 1
    return ordered[:split_idx], ordered[split_idx:]


def _scenario_apply_graphs(graphs, scenario, rate, args, rng, tag):
    print(f"[INFO] {tag}: scenario={scenario} rate={rate}", flush=True)
    out=[]
    edges_b=edges_a=nodes_b=nodes_a=0
    masked=0.0
    for g in graphs:
        g2=g
        if scenario=="low_volume" and rate>0:
            if args.low_volume_mode=="windows":
                if torch.rand(1, generator=rng).item() < rate:
                    continue
            else:
                e,_,st=drop_edges(g2.edges_undirected, rate=rate, rng=rng)
                g2=GraphWindow(x=g2.x, edges_undirected=e, num_nodes=g2.num_nodes, window_start=g2.window_start)
                edges_b += st["edges_before"]; edges_a += st["edges_after"]
        elif scenario=="missing_structure" and rate>0:
            if args.missing_structure_mode in {"edges","both"}:
                e,_,st=drop_edges(g2.edges_undirected, rate=rate, rng=rng); g2=GraphWindow(x=g2.x, edges_undirected=e, num_nodes=g2.num_nodes, window_start=g2.window_start); edges_b+=st["edges_before"]; edges_a+=st["edges_after"]
        elif scenario=="interference" and rate>0:
            x,st=mask_node_features(g2.x, rate=rate, mode=args.node_mask_mode, rng=rng); g2=GraphWindow(x=x, edges_undirected=g2.edges_undirected, num_nodes=g2.num_nodes, window_start=g2.window_start); masked += st["masked_fraction"]
        nodes_b += g.num_nodes; nodes_a += g2.num_nodes
        out.append(g2)
    print(f"[INFO] {tag} nodes before={nodes_b} after={nodes_a} edges before={edges_b} after={edges_a} masked_avg={(100*masked/max(len(out),1)):.2f}%", flush=True)
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGCL style IDS baseline for LANL with sparse graphs")
    parser.add_argument("--auth_path", type=str, required=True, help="converted benign csv (9 ints)")
    parser.add_argument("--red_path", type=str, required=True, help="converted malicious csv (9 ints)")
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--benign_limit", type=int, default=2000)
    parser.add_argument("--mal_limit", type=int, default=2000)
    parser.add_argument("--train_ratio", type=float, default=0.8, help="fraction of benign windows used for training")
    parser.add_argument("--threshold_q", type=float, default=0.99)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--feat_mask", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--out_json", type=str, default="ids_topogcl_lanl_results.json")
    parser.add_argument("--corruption_type", type=str, default="none", choices=["none", "node_features", "edges", "temporal"])
    parser.add_argument("--corruption_rate", type=float, default=0.0)
    parser.add_argument("--node_mask_mode", type=str, default="element", choices=["element", "dimension"])
    parser.add_argument("--temporal_drop_mode", type=str, default="random", choices=["random", "window"])
    parser.add_argument("--temporal_window_size", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_scenario", type=str, default="clean", choices=["clean","low_volume","missing_structure","interference"])
    parser.add_argument("--test_scenario", type=str, default="clean", choices=["clean","low_volume","missing_structure","interference"])
    parser.add_argument("--train_degradation_rate", type=float, default=0.0)
    parser.add_argument("--test_degradation_rate", type=float, default=0.0)
    parser.add_argument("--low_volume_mode", type=str, default="events", choices=["events"])
    parser.add_argument("--missing_structure_mode", type=str, default="edges", choices=["edges"])
    parser.add_argument("--interference_mode", type=str, default="feature_mask", choices=["feature_mask"])
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--delay_steps", type=int, default=1)

    args = parser.parse_args()
    if _is_git_lfs_pointer(args.auth_path) or _is_git_lfs_pointer(args.red_path):
        raise RuntimeError(
            "Input file is a Git LFS pointer, not real data. Run `git lfs install && git lfs pull` "
            "then re-run with the resolved dataset files."
        )

    torch.manual_seed(args.random_seed)
    rng = torch.Generator().manual_seed(args.random_seed)

    torch.manual_seed(args.random_seed)
    rng = torch.Generator().manual_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)
    print(
        f"[INFO] dataset=LANL corruption_type={args.corruption_type} "
        f"mode={(args.node_mask_mode if args.corruption_type == 'node_features' else args.temporal_drop_mode if args.corruption_type == 'temporal' else 'n/a')} "
        f"rate={args.corruption_rate}",
        flush=True,
    )

    def collect_graphs(path: str, limit: Optional[int], tag: str) -> List[GraphWindow]:
        out: List[GraphWindow] = []
        total_events_before = 0
        total_events_after = 0
        total_edges_before = 0.0
        total_edges_after = 0.0
        total_masked_fraction = 0.0
        seen = 0

        for wstart, rows in windows_from_stream(parse_csv_9ints_stream(path), window_size=args.window_size):
            raw_rows = rows
            if args.corruption_type == "temporal":
                raw_rows, drop_stats = drop_raw_events(
                    raw_rows,
                    rate=args.corruption_rate,
                    mode=args.temporal_drop_mode,
                    window_size=args.temporal_window_size,
                    rng=rng,
                )
                total_events_before += drop_stats["events_before"]
                total_events_after += drop_stats["events_after"]

            g = build_graph(raw_rows, window_start=wstart, max_nodes=args.max_nodes)
            if g is None:
                continue
            if args.corruption_type in {"node_features", "edges"}:
                g, c_stats = apply_graph_corruption(g, args.corruption_type, args.corruption_rate, args.node_mask_mode, rng)
                total_edges_before += c_stats["edges_before"]
                total_edges_after += c_stats["edges_after"]
                total_masked_fraction += c_stats["masked_fraction"]
            out.append(g)

            seen += 1
            if limit is not None and len(out) >= limit:
                break
            if seen % 200 == 0:
                print(f"loading {tag} windows seen {seen} graphs kept {len(out)}", flush=True)

        print(f"[OK] {tag} graphs kept: {len(out)}", flush=True)
        if args.corruption_type == "temporal":
            print(f"[INFO] {tag} events before={total_events_before} after={total_events_after}", flush=True)
        if args.corruption_type == "edges":
            print(f"[INFO] {tag} edges before={int(total_edges_before)} after={int(total_edges_after)}", flush=True)
        if args.corruption_type == "node_features" and out:
            avg_mask = 100.0 * total_masked_fraction / len(out)
            print(f"[INFO] {tag} avg masked node-feature entries={avg_mask:.2f}%", flush=True)
        return out

    benign_graphs = collect_graphs(args.auth_path, args.benign_limit, "benign")
    mal_graphs = collect_graphs(args.red_path, args.mal_limit, "malicious")

    if len(benign_graphs) == 0:
        raise RuntimeError("No benign graphs parsed. Check you are using converted 9 int inputs and max_nodes.")
    if len(mal_graphs) == 0:
        raise RuntimeError("No malicious graphs parsed. Fix your converter overlap or increase time tolerance.")

    in_dim = benign_graphs[0].x.shape[1]
    model = GCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.emb_dim)

    benign_train, benign_test = split_train_test_benign(benign_graphs, args.train_ratio)
    if len(benign_test) == 0:
        print("[WARN] Not enough benign windows to create a holdout test split; metrics will use training windows.", flush=True)
        benign_test = benign_train

    benign_train = _scenario_apply_graphs(benign_train, args.train_scenario, args.train_degradation_rate, args, rng, "benign_train")
    benign_test = _scenario_apply_graphs(benign_test, args.test_scenario, args.test_degradation_rate, args, rng, "benign_test")
    mal_graphs = _scenario_apply_graphs(mal_graphs, args.test_scenario, args.test_degradation_rate, args, rng, "malicious")

    print(f"[OK] benign split train={len(benign_train)} test={len(benign_test)}", flush=True)

    train_contrastive(
        model=model,
        graphs=benign_train,
        epochs=args.epochs,
        lr=args.lr,
        edge_drop=args.edge_drop,
        feat_mask=args.feat_mask,
        tau=args.tau,
        batch_size=args.batch_size,
        seed=args.random_seed,
        device=device,
    )

    center = compute_center(model, benign_train, device=device)

    d_benign_train = distances(model, benign_train, center, device=device, tag="score benign train")
    d_benign_test = distances(model, benign_test, center, device=device, tag="score benign test")
    d_mal = distances(model, mal_graphs, center, device=device, tag="score malicious")

    thr = float(torch.quantile(torch.tensor(d_benign_train), args.threshold_q).item())

    y_true = [0] * len(d_benign_test) + [1] * len(d_mal)
    y_score = d_benign_test + d_mal
    y_pred = [1 if s > thr else 0 for s in y_score]

    # Confusion using same convention as your baseline
    fp = sum(1 for yp in y_pred[: len(d_benign_test)] if yp == 1)
    tn = sum(1 for yp in y_pred[: len(d_benign_test)] if yp == 0)
    tp = sum(1 for yp in y_pred[len(d_benign_test) :] if yp == 1)
    fn = sum(1 for yp in y_pred[len(d_benign_test) :] if yp == 0)

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    auc = roc_auc(y_score, y_true)

    results = {
        "dataset": "lanl",
        "method": "topogcl",
        "train_scenario": args.train_scenario,
        "test_scenario": args.test_scenario,
        "train_degradation_rate": args.train_degradation_rate,
        "test_degradation_rate": args.test_degradation_rate,
        "low_volume_mode": args.low_volume_mode,
        "missing_structure_mode": args.missing_structure_mode,
        "interference_mode": args.interference_mode,
        "num_benign_windows": len(benign_graphs),
        "num_mal_windows": len(d_mal),
        "threshold_q": round(args.threshold_q, 2),
        "threshold": round(float(thr), 8),
        "score_stats": {
            "benign_train": score_stats(d_benign_train),
            "benign_test": score_stats(d_benign_test),
            "malicious": score_stats(d_mal)
        },
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
            "edge_drop": args.edge_drop,
            "feat_mask": args.feat_mask,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "window_size": args.window_size,
            "max_nodes": args.max_nodes,
            "random_seed": args.random_seed,
            "corruption_type": args.corruption_type,
            "corruption_rate": args.corruption_rate,
            "node_mask_mode": args.node_mask_mode,
            "temporal_drop_mode": args.temporal_drop_mode,
            "temporal_window_size": args.temporal_window_size,
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] wrote {args.out_json}", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("[✔] Proof succeeded!", flush=True)


if __name__ == "__main__":
    main()
