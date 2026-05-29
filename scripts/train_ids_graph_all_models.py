#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier


# =========================================================
# Data container for existing .npz graph files
# =========================================================
@dataclass
class GraphWindow:
    x: torch.Tensor
    edges_undirected: torch.Tensor
    num_nodes: int
    window_start: int
    label: int
    file_name: str


# =========================================================
# Load existing .npz graph files
# Expected keys:
# node_features, edge_index, edge_features, label
# =========================================================
def edge_index_to_undirected(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
    edge_index = np.asarray(edge_index)

    if edge_index.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E] or [E, 2], got {edge_index.shape}")

    edge_set = set()

    for u, v in edge_index.T:
        u = int(u)
        v = int(v)

        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue

        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))

    if not edge_set:
        return torch.zeros((2, 0), dtype=torch.long)

    edges = torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()
    return edges


def load_npz_graphs(
    graph_dir: Path,
    max_graphs: Optional[int],
    max_nodes: int,
) -> List[GraphWindow]:
    graphs: List[GraphWindow] = []
    paths = sorted(graph_dir.glob("*.npz"))

    if not paths:
        raise FileNotFoundError(f"No .npz graph files found in {graph_dir}")

    for path in paths:
        try:
            arr = np.load(path, allow_pickle=True)

            node_features = arr["node_features"].astype(np.float32)
            label = int(np.array(arr["label"]).reshape(-1)[0])
            edge_index = arr["edge_index"].astype(np.int64)

            if node_features.ndim == 1:
                node_features = node_features.reshape(-1, 1)

            node_features = np.nan_to_num(
                node_features,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            num_nodes = int(node_features.shape[0])

            if num_nodes == 0:
                continue

            if num_nodes > max_nodes:
                continue

            edges = edge_index_to_undirected(edge_index, num_nodes=num_nodes)

            graphs.append(
                GraphWindow(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edges_undirected=edges,
                    num_nodes=num_nodes,
                    window_start=len(graphs),
                    label=label,
                    file_name=path.name,
                )
            )

            if max_graphs is not None and max_graphs > 0 and len(graphs) >= max_graphs:
                break

        except Exception as exc:
            print(f"[WARN] skipped {path.name}: {exc}", flush=True)

    if not graphs:
        raise RuntimeError(f"No usable .npz graphs loaded from {graph_dir}")

    return graphs


def standardize_from_train(
    train_graphs: List[GraphWindow],
    all_graphs: List[GraphWindow],
) -> None:
    xs = torch.cat([g.x for g in train_graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)

    for g in all_graphs:
        g.x = (g.x - mean) / std
        g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)


def split_for_all_models(
    graphs: List[GraphWindow],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    benign_limit: Optional[int],
    mal_limit: Optional[int],
) -> Tuple[List[GraphWindow], List[GraphWindow], List[GraphWindow]]:
    rng = random.Random(seed)

    benign = [g for g in graphs if g.label == 0]
    malicious = [g for g in graphs if g.label == 1]

    rng.shuffle(benign)
    rng.shuffle(malicious)

    if benign_limit is not None and benign_limit > 0:
        benign = benign[:benign_limit]

    if mal_limit is not None and mal_limit > 0:
        malicious = malicious[:mal_limit]

    if len(benign) < 3:
        raise RuntimeError("Need at least three benign graphs.")
    if len(malicious) < 2:
        raise RuntimeError("Need at least two malicious graphs.")

    all_selected = benign + malicious
    y = np.array([g.label for g in all_selected], dtype=np.int64)
    idx = np.arange(len(all_selected))

    test_size = 1.0 - train_ratio - val_ratio
    if test_size <= 0:
        raise RuntimeError("TRAIN_RATIO + VAL_RATIO must be less than 1.0")

    strat = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 3 else None

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=strat,
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_size)
    strat_temp = y_temp if len(np.unique(y_temp)) > 1 and np.min(np.bincount(y_temp)) >= 2 else None

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=strat_temp,
    )

    train_graphs = [all_selected[i] for i in train_idx]
    val_graphs = [all_selected[i] for i in val_idx]
    test_graphs = [all_selected[i] for i in test_idx]

    if not any(g.label == 0 for g in train_graphs):
        raise RuntimeError("Training split has no benign graphs. Increase data size or adjust seed.")

    return train_graphs, val_graphs, test_graphs


# =========================================================
# Sparse adjacency helpers
# =========================================================
def build_sparse_a_hat_from_undirected(
    n: int,
    edges_undirected: torch.Tensor,
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

    d = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = torch.pow(d, -0.5)
    row, col = a.indices()
    norm_vals = a.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    a_hat = torch.sparse_coo_tensor(a.indices(), norm_vals, (n, n)).coalesce()
    return a_hat


# =========================================================
# Graph summary vector for SVM
# =========================================================
def graph_to_summary_vector(g: GraphWindow) -> np.ndarray:
    x = g.x.detach().cpu().numpy()

    if x.shape[0] == 0:
        mean = np.zeros(x.shape[1], dtype=np.float32)
        std = np.zeros(x.shape[1], dtype=np.float32)
    else:
        mean = x.mean(axis=0)
        std = x.std(axis=0)

    edge_count = float(g.edges_undirected.shape[1])
    density = 0.0
    if g.num_nodes > 1:
        density = (2.0 * edge_count) / (g.num_nodes * (g.num_nodes - 1))

    return np.concatenate(
        [
            mean.astype(np.float32),
            std.astype(np.float32),
            np.array([g.num_nodes, edge_count, density], dtype=np.float32),
        ]
    )


# =========================================================
# TopoGCL augmentation: missing edges and missing node features
# =========================================================
def augment_graph(
    g: GraphWindow,
    edge_drop: float,
    feat_mask: float,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = g.num_nodes
    edges = g.edges_undirected

    if edges.numel() > 0 and edge_drop > 0:
        e = edges.shape[1]
        keep = torch.rand(e, generator=rng) > edge_drop
        edges_kept = edges[:, keep]
    else:
        edges_kept = edges

    a_hat = build_sparse_a_hat_from_undirected(n, edges_kept, add_self_loops=True)

    x = g.x
    if feat_mask > 0:
        mask = (torch.rand(x.shape, generator=rng) > feat_mask).float()
        x = x * mask

    return x, a_hat


# =========================================================
# Sparse GCN encoder for TopoGCL
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
# Supervised GNN classifier baseline
# =========================================================
class SupervisedGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, g: GraphWindow, device: torch.device) -> torch.Tensor:
        a_hat = build_sparse_a_hat_from_undirected(g.num_nodes, g.edges_undirected).to(device)
        z = self.encoder(g.x.to(device), a_hat)
        return self.head(z)


def train_supervised_gnn(
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = SupervisedGNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        shuffled = train_graphs[:]
        rng.shuffle(shuffled)
        total_loss = 0.0

        for g in shuffled:
            logits = model(g, device).unsqueeze(0)
            target = torch.tensor([g.label], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        print(f"gnn epoch {epoch + 1}/{epochs} loss={total_loss / max(len(shuffled), 1):.6f}", flush=True)

    model.eval()
    probs = []

    with torch.no_grad():
        for g in test_graphs:
            logits = model(g, device)
            prob = torch.softmax(logits, dim=0)[1].item()
            probs.append(prob)

    return np.array(probs, dtype=np.float32)


# =========================================================
# TopoGCL contrastive loss
# =========================================================
def loss_cal(
    z1: torch.Tensor,
    z2: torch.Tensor,
    zt1: torch.Tensor,
    zt2: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    t = float(tau)

    z1_abs = z1.norm(dim=1).clamp_min(1e-12)
    z2_abs = z2.norm(dim=1).clamp_min(1e-12)
    sim = torch.einsum("ik,jk->ij", z1, z2) / torch.einsum("i,j->ij", z1_abs, z2_abs)
    sim = torch.exp(sim / t)
    pos = sim[range(z1.size(0)), range(z1.size(0))]
    loss1 = pos / (sim.sum(dim=1) - pos).clamp_min(1e-12)

    t1_abs = zt1.norm(dim=1).clamp_min(1e-12)
    t2_abs = zt2.norm(dim=1).clamp_min(1e-12)
    simt = torch.einsum("ik,jk->ij", zt1, zt2) / torch.einsum("i,j->ij", t1_abs, t2_abs)
    simt = torch.exp(simt / t)
    post = simt[range(zt1.size(0)), range(zt1.size(0))]
    loss2 = post / (simt.sum(dim=1) - post).clamp_min(1e-12)

    loss = loss1 + 0.1 * loss2
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
    seed: int,
    device: torch.device,
) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)

    n = len(graphs)
    idx = torch.arange(n)

    for epoch in range(epochs):
        perm = idx[torch.randperm(n, generator=rng)]
        total = 0.0

        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size].tolist()
            batch_graphs = [graphs[j] for j in batch_idx]

            z1_list = []
            z2_list = []
            zt1_list = []
            zt2_list = []

            for g in batch_graphs:
                x1, a1 = augment_graph(g, edge_drop, feat_mask, rng)
                x2, a2 = augment_graph(g, edge_drop, feat_mask, rng)

                x1 = x1.to(device)
                x2 = x2.to(device)
                a1 = a1.to(device)
                a2 = a2.to(device)

                z1_list.append(model(x1, a1))
                z2_list.append(model(x2, a2))

                x1_topo = x1.clone()
                x2_topo = x2.clone()

                if x1_topo.shape[1] > 3:
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

            total += float(loss.item()) * len(batch_graphs)

        print(f"topogcl epoch {epoch + 1}/{epochs} loss={total / max(n, 1):.6f}", flush=True)


# =========================================================
# Scoring and metrics
# =========================================================
def compute_embeddings(
    model: GCN,
    graphs: List[GraphWindow],
    device: torch.device,
    tag: str,
) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for i, g in enumerate(graphs):
            a_hat = build_sparse_a_hat_from_undirected(g.num_nodes, g.edges_undirected).to(device)
            z = model(g.x.to(device), a_hat)
            embs.append(z)

            if (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"{tag}: {i + 1}/{len(graphs)}", flush=True)

    if not embs:
        return torch.empty((0, model.w2.out_features), device=device)

    return torch.stack(embs, dim=0)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": safe_auc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
    }


def summarize_metrics(metrics_list: List[dict]) -> dict:
    metric_names = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    summary = {}
    for name in metric_names:
        values = np.array([m[name] for m in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(values))
        summary[f"{name}_std"] = float(np.nanstd(values))
    return summary


def train_svm_baseline(train_graphs: List[GraphWindow], test_graphs: List[GraphWindow]) -> np.ndarray:
    x_train = np.stack([graph_to_summary_vector(g) for g in train_graphs])
    y_train = np.array([g.label for g in train_graphs], dtype=np.int64)

    x_test = np.stack([graph_to_summary_vector(g) for g in test_graphs])

    svm = SVC(kernel="sigmoid", probability=True, random_state=42)
    svm.fit(x_train, y_train)

    return svm.predict_proba(x_test)[:, 1]


def train_xgboost_baseline(train_graphs: List[GraphWindow], test_graphs: List[GraphWindow], seed: int) -> np.ndarray:
    x_train = np.stack([graph_to_summary_vector(g) for g in train_graphs])
    y_train = np.array([g.label for g in train_graphs], dtype=np.int64)

    x_test = np.stack([graph_to_summary_vector(g) for g in test_graphs])

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    xgb.fit(x_train, y_train)

    return xgb.predict_proba(x_test)[:, 1]


def train_topogcl_scores(
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    hidden_dim: int,
    emb_dim: int,
    epochs: int,
    lr: float,
    edge_drop: float,
    feat_mask: float,
    tau: float,
    batch_size: int,
    threshold_q: float,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    train_benign = [g for g in train_graphs if g.label == 0]

    if len(train_benign) < 2:
        raise RuntimeError("TopoGCL needs at least two benign graphs in the train split.")

    model = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=emb_dim)

    train_contrastive(
        model=model,
        graphs=train_benign,
        epochs=epochs,
        lr=lr,
        edge_drop=edge_drop,
        feat_mask=feat_mask,
        tau=tau,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    train_emb = compute_embeddings(model, train_benign, device=device, tag="embed topogcl train benign")
    test_emb = compute_embeddings(model, test_graphs, device=device, tag="embed topogcl test")

    center = train_emb.mean(dim=0)
    train_dist = torch.norm(train_emb - center, p=2, dim=1).detach().cpu().numpy()
    test_dist = torch.norm(test_emb - center, p=2, dim=1).detach().cpu().numpy()

    threshold = float(np.quantile(train_dist, threshold_q))
    return test_dist, threshold


def main() -> None:
    # ========================================================= 
    # Hardcoded configuration
    # Change values here only if needed.
    # =========================================================
    #GRAPH_DIR = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph")
    GRAPH_DIR = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph")


    #OUT_JSON = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_results_05%.json")
    #OUT_CSV = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_summary_05%.csv")

    OUT_JSON = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_results_25%.json")
    OUT_CSV = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_summary_25%.csv")

    EPOCHS_TOPOGCL = 25
    EPOCHS_GNN = 1

    LR_TOPOGCL = 1e-3
    LR_GNN = 1e-3

    HIDDEN_DIM = 16
    EMB_DIM = 16

    EDGE_DROP = 0.001
    FEAT_MASK = 0.05
    TAU = 0.2
    BATCH_SIZE = 8

    THRESHOLD_Q = 0.99

    #TRAIN_RATIO = 0.60
    #VAL_RATIO = 0.15
    TRAIN_RATIO = 0.25
    VAL_RATIO = 0.15

    # Set these to 0 to use all available graphs.
    BENIGN_LIMIT = 0
    MAL_LIMIT = 0
    MAX_GRAPHS = 0

    MAX_NODES = 50000
    SEEDS = [42]
    #SEEDS = [42, 43, 44, 45, 46]
    STANDARDIZE = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)

    max_graphs = MAX_GRAPHS if MAX_GRAPHS > 0 else None
    benign_limit = BENIGN_LIMIT if BENIGN_LIMIT > 0 else None
    mal_limit = MAL_LIMIT if MAL_LIMIT > 0 else None

    graphs = load_npz_graphs(
        graph_dir=GRAPH_DIR,
        max_graphs=max_graphs,
        max_nodes=MAX_NODES,
    )

    print(f"[OK] graph dir: {GRAPH_DIR}", flush=True)
    print(f"[OK] total graphs loaded: {len(graphs)}", flush=True)

    rows = []
    details = {}
    per_model_metrics = {"svm": [], "xgboost": [], "gnn": [], "topogcl": []}
    split_info = {}
    labels_info = {}

    for seed in SEEDS:
        print(f"\n[RUN] seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_graphs, val_graphs, test_graphs = split_for_all_models(
            graphs=graphs,
            seed=seed,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            benign_limit=benign_limit,
            mal_limit=mal_limit,
        )

        if STANDARDIZE:
            standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)

        in_dim = train_graphs[0].x.shape[1]
        y_test = np.array([g.label for g in test_graphs], dtype=np.int64)

        if not split_info:
            split_info = {
                "train": len(train_graphs),
                "val": len(val_graphs),
                "test": len(test_graphs),
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
            }
            labels_info = {
                "train_benign": int(sum(g.label == 0 for g in train_graphs)),
                "train_malicious": int(sum(g.label == 1 for g in train_graphs)),
                "test_benign": int(sum(g.label == 0 for g in test_graphs)),
                "test_malicious": int(sum(g.label == 1 for g in test_graphs)),
            }

        per_model_metrics["svm"].append(compute_metrics(y_test, train_svm_baseline(train_graphs, test_graphs), threshold=0.5))
        per_model_metrics["xgboost"].append(
            compute_metrics(
                y_test,
                train_xgboost_baseline(train_graphs, test_graphs, seed=seed),
                threshold=0.25,
            )
        )

        gnn_scores = train_supervised_gnn(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            epochs=EPOCHS_GNN,
            lr=LR_GNN,
            seed=seed,
            device=device,
        )
        per_model_metrics["gnn"].append(compute_metrics(y_test, gnn_scores, threshold=0.25))

        topogcl_scores, topogcl_threshold = train_topogcl_scores(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            emb_dim=EMB_DIM,
            epochs=EPOCHS_TOPOGCL,
            lr=LR_TOPOGCL,
            edge_drop=EDGE_DROP,
            feat_mask=FEAT_MASK,
            tau=TAU,
            batch_size=BATCH_SIZE,
            threshold_q=THRESHOLD_Q,
            seed=seed,
            device=device,
        )
        per_model_metrics["topogcl"].append(compute_metrics(y_test, topogcl_scores, threshold=topogcl_threshold))

    for model_name, metrics_list in per_model_metrics.items():
        summary = summarize_metrics(metrics_list)
        rows.append({"model": model_name, **summary})
        details[model_name] = {"runs": metrics_list, "summary": summary}

    results = {
        "dataset": "NF-BoT-IoT",
        "graph_dir": str(GRAPH_DIR),
        "device": str(device),
        "num_graphs_loaded": len(graphs),
        "split": split_info,
        "labels": labels_info,
        "training": {
            "epochs_topogcl": EPOCHS_TOPOGCL,
            "epochs_gnn": EPOCHS_GNN,
            "lr_topogcl": LR_TOPOGCL,
            "lr_gnn": LR_GNN,
            "hidden_dim": HIDDEN_DIM,
            "emb_dim": EMB_DIM,
            "edge_drop": EDGE_DROP,
            "feat_mask": FEAT_MASK,
            "tau": TAU,
            "batch_size": BATCH_SIZE,
            "threshold_q": THRESHOLD_Q,
            "standardized": STANDARDIZE,
            "seeds": SEEDS,
            "benign_limit": BENIGN_LIMIT,
            "mal_limit": MAL_LIMIT,
            "max_graphs": MAX_GRAPHS,
            "max_nodes": MAX_NODES,
        },
        "models": details,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)

    with OUT_CSV.open("w", newline="") as f:
        fieldnames = ["model", "accuracy_mean", "accuracy_std", "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] wrote {OUT_JSON}", flush=True)
    print(f"[OK] wrote {OUT_CSV}", flush=True)

    print("\nFinal summary:")
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
