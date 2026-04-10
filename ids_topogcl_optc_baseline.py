import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Tuple, Optional

import torch

from src.data_corruptions import drop_edges, drop_raw_events, mask_node_features


@dataclass
class GraphWindow:
    x: torch.Tensor
    a_hat: torch.Tensor
    num_nodes: int


def parse_lines_to_windows(
    file_path: str, window_size: int = 300
) -> Generator[List[Tuple[int, int, int, int, int, int, int, int, int]], None, None]:
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
                    int(parts[1]), int(parts[2]), int(parts[3]),
                    int(parts[4]), int(parts[5]), int(parts[6]),
                    int(parts[7]), int(parts[8]),
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


def _is_git_lfs_pointer(file_path: str) -> bool:
    try:
        with open(file_path, "r") as f:
            first = f.readline().strip()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def build_graph(rows: List[Tuple[int, int, int, int, int, int, int, int, int]]) -> GraphWindow:
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


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(a_hat @ self.w1(x))
        h = a_hat @ self.w2(h)
        return h.mean(dim=0)


def iter_graphs(file_path: str, limit: Optional[int], window_size: int) -> Iterable[GraphWindow]:
    count = 0
    for rows in parse_lines_to_windows(file_path, window_size=window_size):
        yield build_graph(rows)
        count += 1
        if limit is not None and count >= limit:
            break


def apply_graph_corruption(
    g: GraphWindow,
    corruption_type: str,
    corruption_rate: float,
    node_mask_mode: str,
    rng: torch.Generator,
) -> Tuple[GraphWindow, Dict[str, float]]:
    if corruption_type == "none" or corruption_rate <= 0.0:
        return g, {"edges_before": float(g.a_hat.numel()), "edges_after": float(g.a_hat.numel()), "masked_fraction": 0.0}

    x = g.x.clone()
    a_hat = g.a_hat.clone()
    stats: Dict[str, float] = {"edges_before": float((a_hat > 0).sum().item()), "edges_after": float((a_hat > 0).sum().item()), "masked_fraction": 0.0}

    if corruption_type == "node_features":
        x, mask_stats = mask_node_features(x, rate=corruption_rate, mode=node_mask_mode, fill_value=0.0, rng=rng)
        stats["masked_fraction"] = mask_stats["masked_fraction"]
    elif corruption_type == "edges":
        n = g.num_nodes
        if n > 1:
            edge_set = []
            for u in range(n):
                for v in range(u + 1, n):
                    if a_hat[u, v] > 0:
                        edge_set.append((u, v))
            if edge_set:
                edge_index = torch.tensor(edge_set, dtype=torch.long).t().contiguous()
                edge_index, _, edge_stats = drop_edges(edge_index, rate=corruption_rate, rng=rng)
                a = torch.zeros((n, n))
                if edge_index.numel() > 0:
                    src, dst = edge_index
                    a[src, dst] = 1.0
                    a[dst, src] = 1.0
                a = a + torch.eye(n)
                d = a.sum(dim=1)
                d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
                a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
                stats["edges_before"] = float(edge_stats["edges_before"])
                stats["edges_after"] = float(edge_stats["edges_after"])
    else:
        raise ValueError(f"Unsupported corruption_type: {corruption_type}")

    assert x.dim() == 2, "node feature tensor must be 2D"
    assert a_hat.shape[0] == a_hat.shape[1] == g.num_nodes, "adjacency shape mismatch after corruption"
    return GraphWindow(x=x, a_hat=a_hat, num_nodes=g.num_nodes), stats


def augment_graph(g: GraphWindow, edge_drop: float, feat_mask: float, rng: torch.Generator) -> GraphWindow:
    n = g.num_nodes
    a = g.a_hat.clone()
    if n > 1 and edge_drop > 0:
        m = torch.rand((n, n), generator=rng)
        keep = (m > edge_drop).float()
        keep = torch.triu(keep, 0)
        keep = keep + keep.T - torch.diag(torch.diag(keep))
        a = a * keep
    a = a + torch.eye(n)
    d = a.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
    a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    x = g.x.clone()
    if feat_mask > 0:
        m = (torch.rand(x.shape, device=x.device, generator=rng) > feat_mask).float()
        x = x * m
    return GraphWindow(x=x, a_hat=a_hat, num_nodes=n)


def loss_cal(x: torch.Tensor, x_aug: torch.Tensor, x_topo: torch.Tensor, x_topo_aug: torch.Tensor) -> torch.Tensor:
    T = 0.2
    batch_size, _ = x.size()
    # structural branch
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    # topological branch
    x_topo_abs = x_topo.norm(dim=1)
    x_topo_aug_abs = x_topo_aug.norm(dim=1)
    sim_matrix_topo = torch.einsum('ik,jk->ij', x_topo, x_topo_aug) / torch.einsum('i,j->ij', x_topo_abs, x_topo_aug_abs)
    sim_matrix_topo = torch.exp(sim_matrix_topo / T)
    pos_sim_topo = sim_matrix_topo[range(batch_size), range(batch_size)]
    loss2 = pos_sim_topo / (sim_matrix_topo.sum(dim=1) - pos_sim_topo)
    loss = 1.0 * loss1 + 0.1 * loss2
    loss = -torch.log(loss).mean()
    return loss


def compute_center(model: GCN, graphs: Iterable[GraphWindow]) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    with torch.no_grad():
        for g in graphs:
            embs.append(model(g.x, g.a_hat))
    if not embs:
        return torch.zeros(model.w2.out_features)
    return torch.stack(embs, dim=0).mean(dim=0)


def train_contrastive(
    model: GCN,
    graphs: List[GraphWindow],
    epochs: int,
    lr: float,
    edge_drop: float,
    feat_mask: float,
    tau: float,
    batch_size: int = 64,
    seed: int = 42,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    N = len(graphs)
    idx = torch.arange(N)
    for e in range(epochs):
        perm = idx[torch.randperm(N, generator=rng)]
        total = 0.0
        last = -1
        for i in range(0, N, batch_size):
            batch_idx = perm[i : i + batch_size]
            gs = [graphs[j] for j in batch_idx.tolist()]
            z1 = []
            z2 = []
            zt1 = []
            zt2 = []
            for g in gs:
                g1 = augment_graph(g, edge_drop, feat_mask, rng)
                g2 = augment_graph(g, edge_drop, feat_mask, rng)
                # structural embeddings
                z1.append(model(g1.x, g1.a_hat))
                z2.append(model(g2.x, g2.a_hat))
                # topology-emphasized embeddings: keep degree cols, zero others
                x1_topo = g1.x.clone(); x1_topo[:, 3:] = 0.0
                x2_topo = g2.x.clone(); x2_topo[:, 3:] = 0.0
                zt1.append(model(x1_topo, g1.a_hat))
                zt2.append(model(x2_topo, g2.a_hat))
            z1 = torch.stack(z1, dim=0)
            z2 = torch.stack(z2, dim=0)
            zt1 = torch.stack(zt1, dim=0)
            zt2 = torch.stack(zt2, dim=0)
            loss = loss_cal(z1, z2, zt1, zt2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * z1.size(0)
            p = int(100 * min(i + batch_size, N) / max(N, 1))
            if p != last and p % 10 == 0:
                print(f"epoch {e + 1}/{epochs} {p}%", flush=True)
                last = p
        print(f"epoch {e + 1}/{epochs} loss={total / max(N,1):.6f}", flush=True)


def distances(model: GCN, graphs: List[GraphWindow], center: torch.Tensor, tag: Optional[str] = None) -> List[float]:
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
    parser = argparse.ArgumentParser(description="TopoGCL-inspired IDS baseline (single-file)")
    parser.add_argument("--auth_path", type=str, default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/OPTC/auth_optc.txt")
    parser.add_argument("--red_path", type=str, default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/OPTC/redteam_optc.txt")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--benign_limit", type=int, default=2000)
    parser.add_argument("--mal_limit", type=int, default=2000)
    parser.add_argument("--threshold_q", type=float, default=0.99)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--feat_mask", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_json", type=str, default="optc_topogcl_results.json")
    parser.add_argument("--corruption_type", type=str, default="none", choices=["none", "node_features", "edges", "temporal"])
    parser.add_argument("--corruption_rate", type=float, default=0.0)
    parser.add_argument("--node_mask_mode", type=str, default="element", choices=["element", "dimension"])
    parser.add_argument("--temporal_drop_mode", type=str, default="random", choices=["random", "window"])
    parser.add_argument("--temporal_window_size", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    assert os.path.exists(args.auth_path), "auth file not found"
    assert os.path.exists(args.red_path), "redteam file not found"
    if _is_git_lfs_pointer(args.auth_path) or _is_git_lfs_pointer(args.red_path):
        raise RuntimeError(
            "Input file is a Git LFS pointer, not real data. Run `git lfs install && git lfs pull` "
            "then re-run with the resolved dataset files."
        )

    torch.manual_seed(args.random_seed)
    rng = torch.Generator().manual_seed(args.random_seed)

    print(
        f"[INFO] dataset=OPTC corruption_type={args.corruption_type} "
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
        last = -1
        for rows in parse_lines_to_windows(path, window_size=args.window_size):
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
            g = build_graph(raw_rows)
            if args.corruption_type in {"node_features", "edges"}:
                g, c_stats = apply_graph_corruption(g, args.corruption_type, args.corruption_rate, args.node_mask_mode, rng=rng)
                total_edges_before += c_stats["edges_before"]
                total_edges_after += c_stats["edges_after"]
                total_masked_fraction += c_stats["masked_fraction"]
            out.append(g)
            if limit is not None and limit > 0:
                p = int(100 * len(out) / limit)
                if p != last and p <= 100 and p % 10 == 0:
                    print(f"loading {tag} {p}%", flush=True)
                    last = p
            if limit is not None and len(out) >= limit:
                break
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
    if not benign_graphs or not mal_graphs:
        raise RuntimeError(
            "Insufficient graphs parsed from inputs. Confirm files contain comma-separated 9-int rows "
            "(not headers/LFS pointers) and that --window_size/limits are valid."
        )

    in_dim = benign_graphs[0].x.shape[1]
    model = GCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.emb_dim)

    train_contrastive(
        model,
        benign_graphs,
        epochs=args.epochs,
        lr=args.lr,
        edge_drop=args.edge_drop,
        feat_mask=args.feat_mask,
        tau=args.tau,
        batch_size=args.batch_size,
        seed=args.random_seed,
    )

    center = compute_center(model, benign_graphs)
    d_benign = distances(model, benign_graphs, center, tag="score benign")
    d_mal = distances(model, mal_graphs, center, tag="score malicious")
    thr = percentile(d_benign, args.threshold_q)

    y_true = [0] * len(d_benign) + [1] * len(d_mal)
    y_score = d_benign + d_mal
    y_pred = [1 if s > thr else 0 for s in y_score]

    tp = sum(1 for yp, yt in zip(y_pred[len(d_benign):], y_true[len(d_benign):]) if yp == 1 and yt == 1)
    fn = sum(1 for yp, yt in zip(y_pred[len(d_benign):], y_true[len(d_benign):]) if yp == 0 and yt == 1)
    fp = sum(1 for yp, yt in zip(y_pred[:len(d_benign)], y_true[:len(d_benign)]) if yp == 1 and yt == 0)
    tn = sum(1 for yp, yt in zip(y_pred[:len(d_benign)], y_true[:len(d_benign)]) if yp == 0 and yt == 0)

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
        "threshold": thr,
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tpr": tpr,
            "fpr": fpr,
            "auroc": auc,
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
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
