#!/usr/bin/env python3
import csv
import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.svm import SVC

DATASET = "OPTC"
OUTPUT_DIR = "results/structural_subtlety"
AUTH_PATH = "datasets/OPTC/auth_optc.txt"
RED_PATH = "datasets/OPTC/redteam_optc.txt"
WINDOW_SIZE = 300
SEED = 42
SUBTLETY_Z_THRESHOLD = 1.5


@dataclass
class GraphWindow:
    window_id: str
    x: torch.Tensor
    a_hat: torch.Tensor
    num_nodes: int
    edge_count: int


def parse_lines_to_windows(file_path: str, prefix: str, window_size: int = 300) -> Generator[Tuple[str, List[Tuple[int, int, int, int, int, int, int, int, int]]], None, None]:
    current_start = None
    rows: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
    idx = 0
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                t = int(parts[0])
                r = (t, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]))
            except ValueError:
                continue
            if current_start is None:
                current_start = t
            if t >= current_start + window_size:
                if rows:
                    window_id = f"{prefix}_w{idx:06d}_t{current_start}"
                    yield window_id, rows
                    idx += 1
                rows = []
                current_start = (t // window_size) * window_size
            rows.append(r)
    if rows:
        window_id = f"{prefix}_w{idx:06d}_t{current_start}"
        yield window_id, rows


def build_graph(window_id: str, rows):
    if not rows:
        return GraphWindow(window_id=window_id, x=torch.zeros(1, 15), a_hat=torch.eye(1), num_nodes=1, edge_count=0)
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
    undirected_edges = set()
    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]
        a[u, v] += 1.0
        a[v, u] += 1.0
        undirected_edges.add((u, v) if u <= v else (v, u))
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
    return GraphWindow(window_id=window_id, x=x, a_hat=a_hat, num_nodes=n, edge_count=len(undirected_edges))


def load_graphs(path: str, prefix: str) -> List[GraphWindow]:
    return [build_graph(window_id, rows) for window_id, rows in parse_lines_to_windows(path, prefix, WINDOW_SIZE)]


def graph_to_vector(g: GraphWindow) -> np.ndarray:
    deg = g.a_hat.sum(dim=1)
    return np.array([g.num_nodes, float(g.a_hat.sum()), float(deg.mean()), float(deg.std() if g.num_nodes > 1 else 0.0), float(g.x.mean()), float(g.x.std())], dtype=float)


class GCN(torch.nn.Module):
    def __init__(self, in_dim=15, hidden=32, out_dim=16):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden)
        self.w2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, a_hat):
        h = torch.relu(a_hat @ self.w1(x))
        h = a_hat @ self.w2(h)
        return h.mean(dim=0)


def embed_graphs(model: GCN, graphs: Sequence[GraphWindow]) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([model(g.x, g.a_hat) for g in graphs], dim=0)


def choose_threshold(y_true, scores):
    cand = np.unique(scores)
    best_thr, best_f1 = float(cand[0]), -1.0
    for thr in cand:
        pred = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def recall_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def evaluate(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "recall_at_5_percent_fpr": recall_at_fpr(y_true, scores, 0.05),
    }


def connected_components_count(adj_bool: np.ndarray) -> int:
    n = adj_bool.shape[0]
    seen = np.zeros(n, dtype=bool)
    cc = 0
    for i in range(n):
        if seen[i]:
            continue
        cc += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            nbrs = np.where(adj_bool[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return cc


def graph_stats(g: GraphWindow) -> Dict[str, float]:
    n = g.num_nodes
    m = g.edge_count
    avg_degree = (2.0 * m / n) if n else 0.0
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    a = g.a_hat.detach().cpu().numpy()
    adj = a > 0
    np.fill_diagonal(adj, False)
    cc = connected_components_count(adj) if n else 0
    centrality = a.sum(axis=1)
    return {
        "num_nodes": float(n),
        "num_edges": float(m),
        "average_degree": float(avg_degree),
        "density": float(density),
        "number_connected_components": float(cc),
        "average_betweenness": float(np.mean(centrality / max(centrality.sum(), 1.0))),
        "average_closeness": float(np.mean(1.0 / np.clip(np.sqrt(np.maximum(centrality, 1e-6)), 1e-6, None))),
    }


def select_structurally_subtle(mal_test: List[GraphWindow], benign_test: List[GraphWindow]) -> List[GraphWindow]:
    keys = ["num_nodes", "num_edges", "average_degree", "density"]
    benign_stats = [graph_stats(g) for g in benign_test]
    means = {k: float(np.mean([s[k] for s in benign_stats])) for k in keys}
    stds = {k: float(np.std([s[k] for s in benign_stats]) + 1e-8) for k in keys}

    selected = []
    for g in mal_test:
        s = graph_stats(g)
        z = np.array([abs((s[k] - means[k]) / stds[k]) for k in keys], dtype=float)
        if float(np.linalg.norm(z)) <= SUBTLETY_Z_THRESHOLD:
            selected.append(g)
    return selected


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benign = load_graphs(AUTH_PATH, "benign")
    malicious = load_graphs(RED_PATH, "malicious")

    b_train_end, b_val_end = int(0.6 * len(benign)), int(0.8 * len(benign))
    m_train_end, m_val_end = int(0.6 * len(malicious)), int(0.8 * len(malicious))
    b_train, b_val, b_test = benign[:b_train_end], benign[b_train_end:b_val_end], benign[b_val_end:]
    m_train, m_val, m_test = malicious[:m_train_end], malicious[m_train_end:m_val_end], malicious[m_val_end:]

    hard_malicious = select_structurally_subtle(m_test, b_test)

    with open(os.path.join(OUTPUT_DIR, "hard_malicious_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window_id"])
        w.writeheader()
        for g in hard_malicious:
            w.writerow({"window_id": g.window_id})

    train_graphs = b_train + m_train
    y_train = np.array([0] * len(b_train) + [1] * len(m_train))
    val_graphs = b_val + m_val
    y_val = np.array([0] * len(b_val) + [1] * len(m_val))
    test_graphs = b_test + m_test
    y_test = np.array([0] * len(b_test) + [1] * len(m_test))

    subtle_test_graphs = b_test + hard_malicious
    y_test_subtle = np.array([0] * len(b_test) + [1] * len(hard_malicious))

    rows = []

    # SVM
    x_train = np.stack([graph_to_vector(g) for g in train_graphs])
    x_val = np.stack([graph_to_vector(g) for g in val_graphs])
    x_test = np.stack([graph_to_vector(g) for g in test_graphs])
    x_subtle = np.stack([graph_to_vector(g) for g in subtle_test_graphs])
    svm = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(x_train, y_train)
    thr = choose_threshold(y_val, svm.predict_proba(x_val)[:, 1])
    rows.append({"model": "svm", "test_set": "full", **evaluate(y_test, svm.predict_proba(x_test)[:, 1], thr)})
    rows.append({"model": "svm", "test_set": "structural_subtlety", **evaluate(y_test_subtle, svm.predict_proba(x_subtle)[:, 1], thr)})

    # GNN
    enc = GCN(); clf = torch.nn.Linear(16, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
    for _ in range(10):
        for g, y in zip(train_graphs, y_train):
            z = enc(g.x, g.a_hat)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(z), torch.tensor([float(y)]))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val_scores = torch.sigmoid(clf(embed_graphs(enc, val_graphs))).squeeze(1).numpy()
        test_scores = torch.sigmoid(clf(embed_graphs(enc, test_graphs))).squeeze(1).numpy()
        subtle_scores = torch.sigmoid(clf(embed_graphs(enc, subtle_test_graphs))).squeeze(1).numpy()
    thr = choose_threshold(y_val, val_scores)
    rows.append({"model": "gnn", "test_set": "full", **evaluate(y_test, test_scores, thr)})
    rows.append({"model": "gnn", "test_set": "structural_subtlety", **evaluate(y_test_subtle, subtle_scores, thr)})

    # TopoGCL
    enc = GCN(); pre_epochs = 30
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for _ in range(pre_epochs):
        for g in train_graphs:
            mask = (torch.rand_like(g.x) > 0.2).float()
            z1 = enc(g.x * mask, g.a_hat)
            z2 = enc(g.x, g.a_hat)
            loss = 1 - torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    clf = torch.nn.Linear(16, 1)
    opt2 = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(20):
        for g, y in zip(train_graphs, y_train):
            z = enc(g.x, g.a_hat).detach()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(z), torch.tensor([float(y)]))
            opt2.zero_grad(); loss.backward(); opt2.step()
    with torch.no_grad():
        val_scores = torch.sigmoid(clf(embed_graphs(enc, val_graphs))).squeeze(1).numpy()
        test_scores = torch.sigmoid(clf(embed_graphs(enc, test_graphs))).squeeze(1).numpy()
        subtle_scores = torch.sigmoid(clf(embed_graphs(enc, subtle_test_graphs))).squeeze(1).numpy()
    thr = choose_threshold(y_val, val_scores)
    rows.append({"model": "topogcl", "test_set": "full", **evaluate(y_test, test_scores, thr)})
    rows.append({"model": "topogcl", "test_set": "structural_subtlety", **evaluate(y_test_subtle, subtle_scores, thr)})

    fields = ["dataset", "model", "test_set", "hard_malicious_count", "full_malicious_count", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "recall_at_5_percent_fpr"]
    with open(os.path.join(OUTPUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {"dataset": DATASET, "hard_malicious_count": len(hard_malicious), "full_malicious_count": len(m_test), **r}
            w.writerow(out)


if __name__ == "__main__":
    main()
