#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import warnings

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

try:
    from torch_geometric.data import Data  # type: ignore
except Exception:
    class Data:  # minimal fallback to keep script runnable
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class SimpleGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        n = x.size(0)
        a = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        a[edge_index[0], edge_index[1]] = 1.0
        a = a + torch.eye(n, dtype=x.dtype, device=x.device)
        deg = a.sum(1)
        deg_inv_sqrt = torch.pow(deg.clamp_min(1.0), -0.5)
        a_hat = deg_inv_sqrt[:, None] * a * deg_inv_sqrt[None, :]
        h = torch.relu(a_hat @ self.lin1(x))
        h = a_hat @ self.lin2(h)
        graph_embed = h.mean(0)
        return self.head(graph_embed)


def graph_to_summary_vector(g: Data) -> np.ndarray:
    x = g.x.numpy()
    edge_attr = g.edge_attr.numpy()
    return np.concatenate(
        [
            x.mean(axis=0),
            x.std(axis=0),
            edge_attr.mean(axis=0),
            edge_attr.std(axis=0),
            np.array([x.shape[0], g.edge_index.shape[1]], dtype=np.float32),
        ]
    )


def load_graphs(graph_dir: Path) -> list[Data]:
    graphs = []
    for path in sorted(graph_dir.glob("*.npz")):
        arr = np.load(path, allow_pickle=True)
        node_features = arr["node_features"].astype(np.float32)
        edge_index = arr["edge_index"].astype(np.int64)
        edge_features = arr["edge_features"].astype(np.float32)
        label = int(np.array(arr["label"]).reshape(-1)[0])
        graphs.append(
            Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                y=torch.tensor([label], dtype=torch.long),
                attack_type=str(np.array(arr["attack_type"]).reshape(-1)[0]),
                file_name=path.name,
            )
        )
    return graphs


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, model_name: str) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) < 2:
        warnings.warn(f"{model_name}: AUROC/AUPRC skipped (only one class in evaluation split).")
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def train_gnn(train_graphs: list[Data], train_y: np.ndarray, eval_graphs: list[Data], seed: int = 42) -> np.ndarray:
    torch.manual_seed(seed)
    model = SimpleGNN(in_dim=train_graphs[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(30):
        for g, y in zip(train_graphs, train_y):
            logits = model(g).unsqueeze(0)
            target = torch.tensor([int(y)], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for g in eval_graphs:
            prob = torch.softmax(model(g), dim=0)[1].item()
            probs.append(prob)
    return np.array(probs, dtype=np.float32)


def main() -> None:
    graph_dir = Path("datasets/NF-BoT-IoT/Graph")
    out_csv = Path("results/nf_bot_iot_graph_debug/summary.csv")
    seed = 42

    graphs = load_graphs(graph_dir)
    if not graphs:
        raise FileNotFoundError(f"No .npz graph files found in {graph_dir}")

    y = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    n_nodes = np.array([g.x.shape[0] for g in graphs], dtype=np.int64)
    n_edges = np.array([g.edge_index.shape[1] for g in graphs], dtype=np.int64)

    print(f"number of graphs: {len(graphs)}")
    print(f"number of benign graphs: {(y == 0).sum()}")
    print(f"number of malicious graphs: {(y == 1).sum()}")
    print(f"node feature dimension: {graphs[0].x.shape[1]}")
    print(f"edge feature dimension: {graphs[0].edge_attr.shape[1]}")
    print(f"min, mean, max number of nodes: {n_nodes.min()}, {n_nodes.mean():.2f}, {n_nodes.max()}")
    print(f"min, mean, max number of edges: {n_edges.min()}, {n_edges.mean():.2f}, {n_edges.max()}")

    idx = np.arange(len(graphs))
    strat = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 3 else None
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.30, random_state=seed, stratify=strat
    )
    strat_temp = y_temp if len(np.unique(y_temp)) > 1 and np.min(np.bincount(y_temp)) >= 2 else None
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=seed, stratify=strat_temp
    )

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    X_train = np.stack([graph_to_summary_vector(g) for g in train_graphs])
    X_test = np.stack([graph_to_summary_vector(g) for g in test_graphs])

    svm = SVC(kernel="rbf", probability=True, random_state=seed)
    svm.fit(X_train, y_train)
    svm_test_scores = svm.predict_proba(X_test)[:, 1]

    gnn_test_scores = train_gnn(train_graphs, y_train, test_graphs, seed=seed)

    topo_train = np.stack([graph_to_summary_vector(g) for g in train_graphs])
    topo_test = np.stack([graph_to_summary_vector(g) for g in test_graphs])
    topogcl = SVC(kernel="linear", probability=True, random_state=seed)
    topogcl.fit(topo_train, y_train)
    topogcl_test_scores = topogcl.predict_proba(topo_test)[:, 1]

    rows = [
        {"model": "svm", **compute_metrics(y_test, svm_test_scores, "SVM")},
        {"model": "gnn", **compute_metrics(y_test, gnn_test_scores, "GNN")},
        {"model": "topogcl", **compute_metrics(y_test, topogcl_test_scores, "TopoGCL")},
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"split sizes -> train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
    print(f"saved results: {out_csv}")


if __name__ == "__main__":
    main()
