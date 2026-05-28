#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from train_ids_graph_all_models import (
    GraphWindow,
    compute_metrics,
    load_npz_graphs,
    split_for_all_models,
    standardize_from_train,
    summarize_metrics,
)


# =========================================================
# Hardcoded configuration: no command-line flags.
# The script discovers existing TopoGCL result files and appends
# GraphSAGE/GIN metrics using each file's graph directory and split setup.
# =========================================================
RESULT_FILES = sorted(Path("results").glob("*/*_results_*.json"))

EPOCHS_GRAPHSAGE_GIN = 100
LR_GRAPHSAGE_GIN = 1e-3
HIDDEN_DIM_DEFAULT = 64
DROPOUT = 0.2
STANDARDIZE_DEFAULT = True
THRESHOLD = 0.25

METRIC_FIELDNAMES = [
    "model",
    "accuracy_mean",
    "accuracy_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "auroc_mean",
    "auroc_std",
    "auprc_mean",
    "auprc_std",
]


# =========================================================
# Sparse adjacency helpers for GraphSAGE and GIN
# =========================================================
def build_sparse_mean_adj(g: GraphWindow, device: torch.device) -> torch.Tensor:
    """Symmetric graph with self loops, row-normalized for mean aggregation."""
    n = max(g.num_nodes, 1)
    edges = g.edges_undirected

    if edges.numel() == 0:
        idx = torch.arange(n, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(n, dtype=torch.float32)
    else:
        u = edges[0]
        v = edges[1]
        src = torch.cat([u, v], dim=0)
        dst = torch.cat([v, u], dim=0)
        idx = torch.arange(n, dtype=torch.long)
        indices = torch.cat([torch.stack([src, dst], dim=0), torch.stack([idx, idx], dim=0)], dim=1)
        values = torch.ones(indices.shape[1], dtype=torch.float32)

    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    row, _ = adj.indices()
    norm_values = adj.values() / degree[row]
    return torch.sparse_coo_tensor(adj.indices(), norm_values, (n, n)).coalesce().to(device)


def build_sparse_sum_adj_no_self(g: GraphWindow, device: torch.device) -> torch.Tensor:
    """Symmetric graph without self loops for GIN sum aggregation."""
    n = max(g.num_nodes, 1)
    edges = g.edges_undirected

    if edges.numel() == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
    else:
        u = edges[0]
        v = edges[1]
        src = torch.cat([u, v], dim=0)
        dst = torch.cat([v, u], dim=0)
        indices = torch.stack([src, dst], dim=0)
        values = torch.ones(indices.shape[1], dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(device)


# =========================================================
# Supervised graph classifiers
# =========================================================
class GraphSAGELayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_mean, x)
        return torch.relu(self.linear(torch.cat([x, neigh], dim=1)))


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, g: GraphWindow, device: torch.device) -> torch.Tensor:
        x = g.x.to(device)
        adj = build_sparse_mean_adj(g, device)
        h = self.dropout(self.layer1(x, adj))
        h = self.layer2(h, adj)
        return self.head(h.mean(dim=0))


class GINLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, adj_sum: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_sum, x)
        return torch.relu(self.mlp((1.0 + self.eps) * x + neigh))


class GINClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = GINLayer(in_dim, hidden_dim)
        self.layer2 = GINLayer(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, g: GraphWindow, device: torch.device) -> torch.Tensor:
        x = g.x.to(device)
        adj = build_sparse_sum_adj_no_self(g, device)
        h = self.dropout(self.layer1(x, adj))
        h = self.layer2(h, adj)
        return self.head(h.mean(dim=0))


def train_supervised_graph_classifier(
    model: torch.nn.Module,
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    model_name: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
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

        print(
            f"{model_name} epoch {epoch + 1}/{epochs} "
            f"loss={total_loss / max(len(shuffled), 1):.6f}",
            flush=True,
        )

    model.eval()
    probs = []
    with torch.no_grad():
        for g in test_graphs:
            logits = model(g, device)
            probs.append(torch.softmax(logits, dim=0)[1].item())

    return np.array(probs, dtype=np.float32)


def summary_csv_for(result_path: Path) -> Path:
    return result_path.with_name(result_path.name.replace("_results_", "_summary_").replace(".json", ".csv"))


def load_existing_csv_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_upserted_summary_rows(csv_path: Path, new_rows: Iterable[dict]) -> None:
    rows_by_model = {row["model"]: row for row in load_existing_csv_rows(csv_path)}
    ordered_models = list(rows_by_model)

    for row in new_rows:
        model = row["model"]
        if model not in rows_by_model:
            ordered_models.append(model)
        rows_by_model[model] = {field: row.get(field, "") for field in METRIC_FIELDNAMES}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_by_model[model] for model in ordered_models)


def train_for_result_file(result_path: Path, device: torch.device) -> None:
    with result_path.open() as f:
        results = json.load(f)

    training = results.get("training", {})
    graph_dir = Path(results["graph_dir"])
    train_ratio = float(results.get("split", {}).get("train_ratio", 0.25))
    val_ratio = float(results.get("split", {}).get("val_ratio", 0.15))
    seeds = [int(seed) for seed in training.get("seeds", [42, 43, 44, 45, 46])]
    hidden_dim = int(training.get("hidden_dim", HIDDEN_DIM_DEFAULT))
    max_nodes = int(training.get("max_nodes", 50000))
    max_graphs_cfg = int(training.get("max_graphs", 0) or 0)
    benign_limit_cfg = int(training.get("benign_limit", 0) or 0)
    mal_limit_cfg = int(training.get("mal_limit", 0) or 0)
    standardize = bool(training.get("standardized", STANDARDIZE_DEFAULT))

    max_graphs = max_graphs_cfg if max_graphs_cfg > 0 else None
    benign_limit = benign_limit_cfg if benign_limit_cfg > 0 else None
    mal_limit = mal_limit_cfg if mal_limit_cfg > 0 else None

    print(f"\n[RESULT] {result_path}", flush=True)
    print(f"[OK] graph dir: {graph_dir}", flush=True)

    base_graphs = load_npz_graphs(graph_dir=graph_dir, max_graphs=max_graphs, max_nodes=max_nodes)
    print(f"[OK] total graphs loaded: {len(base_graphs)}", flush=True)

    per_model_metrics = {"graphsage": [], "gin": []}

    for seed in seeds:
        print(f"\n[RUN] {result_path.name} seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        graphs = load_npz_graphs(graph_dir=graph_dir, max_graphs=max_graphs, max_nodes=max_nodes)
        train_graphs, val_graphs, test_graphs = split_for_all_models(
            graphs=graphs,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            benign_limit=benign_limit,
            mal_limit=mal_limit,
        )

        if standardize:
            standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)

        in_dim = train_graphs[0].x.shape[1]
        y_test = np.array([g.label for g in test_graphs], dtype=np.int64)

        graphsage_scores = train_supervised_graph_classifier(
            model=GraphSAGEClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=DROPOUT),
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            epochs=EPOCHS_GRAPHSAGE_GIN,
            lr=LR_GRAPHSAGE_GIN,
            seed=seed,
            device=device,
            model_name="graphsage",
        )
        per_model_metrics["graphsage"].append(compute_metrics(y_test, graphsage_scores, threshold=THRESHOLD))

        gin_scores = train_supervised_graph_classifier(
            model=GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=DROPOUT),
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            epochs=EPOCHS_GRAPHSAGE_GIN,
            lr=LR_GRAPHSAGE_GIN,
            seed=seed,
            device=device,
            model_name="gin",
        )
        per_model_metrics["gin"].append(compute_metrics(y_test, gin_scores, threshold=THRESHOLD))

    new_rows = []
    results.setdefault("models", {})
    for model_name, metrics_list in per_model_metrics.items():
        summary = summarize_metrics(metrics_list)
        results["models"][model_name] = {"runs": metrics_list, "summary": summary}
        new_rows.append({"model": model_name, **summary})

    training["epochs_graphsage_gin"] = EPOCHS_GRAPHSAGE_GIN
    training["lr_graphsage_gin"] = LR_GRAPHSAGE_GIN
    training["dropout_graphsage_gin"] = DROPOUT
    training["threshold_graphsage_gin"] = THRESHOLD
    results["training"] = training
    results["device"] = str(device)
    results["num_graphs_loaded"] = len(base_graphs)

    with result_path.open("w") as f:
        json.dump(results, f, indent=2)

    csv_path = summary_csv_for(result_path)
    write_upserted_summary_rows(csv_path, new_rows)

    print(f"\n[OK] appended GraphSAGE/GIN results to {result_path}", flush=True)
    print(f"[OK] appended GraphSAGE/GIN rows to {csv_path}", flush=True)


def main() -> None:
    if not RESULT_FILES:
        raise FileNotFoundError("No existing result files matched results/*/*_results_*.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)

    for result_path in RESULT_FILES:
        train_for_result_file(result_path, device=device)


if __name__ == "__main__":
    main()
