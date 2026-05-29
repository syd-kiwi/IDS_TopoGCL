#!/usr/bin/env python3
"""Train an InfoGraph-only baseline and append its summary to an existing CSV.

The model follows the PyGCL InfoGraph example structure: a multi-layer GIN encoder
produces node-level and graph-level embeddings, projection heads map both views to
one contrastive space, and a graph-to-local JSD objective trains the encoder before a
linear SVM evaluates graph embeddings on the IDS train/test split.
"""
from __future__ import annotations

import argparse
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


@dataclass
class GraphWindow:
    x: torch.Tensor
    edges_undirected: torch.Tensor
    num_nodes: int
    window_start: int
    label: int
    file_name: str


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
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes or u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))

    if not edge_set:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()


def load_npz_graphs(graph_dir: Path, max_graphs: Optional[int], max_nodes: int) -> List[GraphWindow]:
    graphs: List[GraphWindow] = []
    paths = sorted(graph_dir.glob("*.npz"))

    if not paths:
        raise FileNotFoundError(f"No .npz graph files found in {graph_dir}")

    for path in paths:
        try:
            arr = np.load(path, allow_pickle=True)
            node_features = arr["node_features"].astype(np.float32)
            edge_index = arr["edge_index"].astype(np.int64)
            label = int(np.array(arr["label"]).reshape(-1)[0])

            if node_features.ndim == 1:
                node_features = node_features.reshape(-1, 1)

            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            num_nodes = int(node_features.shape[0])

            if num_nodes == 0 or num_nodes > max_nodes:
                continue

            graphs.append(
                GraphWindow(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edges_undirected=edge_index_to_undirected(edge_index, num_nodes=num_nodes),
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


def standardize_from_train(train_graphs: List[GraphWindow], all_graphs: List[GraphWindow]) -> None:
    xs = torch.cat([g.x for g in train_graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)

    for graph in all_graphs:
        graph.x = torch.nan_to_num((graph.x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def clone_graphs(graphs: List[GraphWindow]) -> List[GraphWindow]:
    return [
        GraphWindow(
            x=graph.x.clone(),
            edges_undirected=graph.edges_undirected.clone(),
            num_nodes=graph.num_nodes,
            window_start=graph.window_start,
            label=graph.label,
            file_name=graph.file_name,
        )
        for graph in graphs
    ]


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

    selected = benign + malicious
    y = np.array([g.label for g in selected], dtype=np.int64)
    idx = np.arange(len(selected))
    test_size = 1.0 - train_ratio - val_ratio
    if test_size <= 0:
        raise RuntimeError("TRAIN_RATIO + VAL_RATIO must be less than 1.0")

    strat = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 3 else None
    train_idx, temp_idx, _, y_temp = train_test_split(
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

    train_graphs = [selected[i] for i in train_idx]
    val_graphs = [selected[i] for i in val_idx]
    test_graphs = [selected[i] for i in test_idx]

    if len(np.unique([g.label for g in train_graphs])) < 2:
        raise RuntimeError("InfoGraph SVM evaluation needs both classes in the training split.")

    return train_graphs, val_graphs, test_graphs


class GINLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim),
        )
        self.batch_norm = torch.nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregated = x.new_zeros(x.shape)
        if edge_index.numel() > 0:
            src, dst = edge_index
            aggregated.index_add_(0, dst, x[src])
        h = self.mlp(x + aggregated)
        return self.batch_norm(torch.relu(h))


class InfoGraphEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(GINLayer(layer_input_dim, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        local_embeddings = []
        graph_embeddings = []

        for layer in self.layers:
            z = layer(z, edge_index)
            local_embeddings.append(z)
            pooled = z.new_zeros((num_graphs, z.shape[1]))
            pooled.index_add_(0, batch, z)
            graph_embeddings.append(pooled)

        return torch.cat(local_embeddings, dim=1), torch.cat(graph_embeddings, dim=1)


class ProjectionHead(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.linear(x)


class InfoGraphModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        projection_dim = hidden_dim * num_layers
        self.encoder = InfoGraphEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.local_fc = ProjectionHead(projection_dim)
        self.global_fc = ProjectionHead(projection_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, edge_index, batch, num_graphs)

    def project(self, z: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.local_fc(z), self.global_fc(g)


def make_directed_edges(edges_undirected: torch.Tensor) -> torch.Tensor:
    if edges_undirected.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    u, v = edges_undirected
    return torch.cat([torch.stack([u, v]), torch.stack([v, u])], dim=1)


def make_batch(graphs: List[GraphWindow], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    edge_parts = []
    batch_parts = []
    offset = 0

    for graph_idx, graph in enumerate(graphs):
        xs.append(graph.x)
        edge_index = make_directed_edges(graph.edges_undirected)
        if edge_index.numel() > 0:
            edge_parts.append(edge_index + offset)
        batch_parts.append(torch.full((graph.num_nodes,), graph_idx, dtype=torch.long))
        offset += graph.num_nodes

    x = torch.cat(xs, dim=0).to(device)
    batch = torch.cat(batch_parts, dim=0).to(device)
    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1).to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    return x, edge_index, batch


def infograph_jsd_loss(local_z: torch.Tensor, global_g: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(local_z, global_g.t())
    positive_mask = torch.zeros_like(scores, dtype=torch.bool)
    positive_mask[torch.arange(batch.numel(), device=batch.device), batch] = True
    negative_mask = ~positive_mask

    positive_expectation = torch.nn.functional.softplus(-scores[positive_mask]).mean()
    negative_expectation = torch.nn.functional.softplus(scores[negative_mask]).mean()
    return positive_expectation + negative_expectation


def train_infograph(
    model: InfoGraphModel,
    graphs: List[GraphWindow],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        shuffled = graphs[:]
        rng.shuffle(shuffled)
        total_loss = 0.0
        total_graphs = 0
        model.train()

        for start in range(0, len(shuffled), batch_size):
            batch_graphs = shuffled[start : start + batch_size]
            if len(batch_graphs) < 2:
                continue

            x, edge_index, batch = make_batch(batch_graphs, device=device)
            z, g = model(x, edge_index, batch, num_graphs=len(batch_graphs))
            z, g = model.project(z, g)
            loss = infograph_jsd_loss(z, g, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_graphs)
            total_graphs += len(batch_graphs)

        print(f"infograph epoch {epoch + 1}/{epochs} loss={total_loss / max(total_graphs, 1):.6f}", flush=True)


def compute_infograph_embeddings(
    model: InfoGraphModel,
    graphs: List[GraphWindow],
    batch_size: int,
    device: torch.device,
    tag: str,
) -> torch.Tensor:
    embeddings = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start : start + batch_size]
            x, edge_index, batch = make_batch(batch_graphs, device=device)
            _, g = model(x, edge_index, batch, num_graphs=len(batch_graphs))
            embeddings.append(g.detach().cpu())
            print(f"{tag}: {min(start + batch_size, len(graphs))}/{len(graphs)}", flush=True)

    return torch.cat(embeddings, dim=0)


def train_infograph_scores(
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    model = InfoGraphModel(input_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    train_infograph(
        model=model,
        graphs=train_graphs,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    train_embeddings = compute_infograph_embeddings(
        model,
        train_graphs,
        batch_size=batch_size,
        device=device,
        tag="embed infograph train",
    ).numpy()
    test_embeddings = compute_infograph_embeddings(
        model,
        test_graphs,
        batch_size=batch_size,
        device=device,
        tag="embed infograph test",
    ).numpy()

    y_train = np.array([g.label for g in train_graphs], dtype=np.int64)
    svm = SVC(kernel="linear", probability=True, random_state=seed)
    svm.fit(train_embeddings, y_train)
    return svm.predict_proba(test_embeddings)[:, 1]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
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
    summary = {}
    for name in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
        values = np.array([metrics[name] for metrics in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(values))
        summary[f"{name}_std"] = float(np.nanstd(values))
    return summary


def append_summary_csv(out_csv: Path, row: dict) -> None:
    fieldnames = [
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with out_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def update_results_json(out_json: Path, payload: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        with out_json.open() as handle:
            results = json.load(handle)
    else:
        results = {
            "dataset": payload["dataset"],
            "graph_dir": payload["graph_dir"],
            "models": {},
        }

    results["dataset"] = payload["dataset"]
    results["graph_dir"] = payload["graph_dir"]
    results["device"] = payload["device"]
    results["num_graphs_loaded"] = payload["num_graphs_loaded"]
    results["split"] = payload["split"]
    results["labels"] = payload["labels"]
    training = results.setdefault("training", {})
    training.update(payload["training"])
    results.setdefault("models", {})["infograph"] = payload["model"]

    with out_json.open("w") as handle:
        json.dump(results, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train only InfoGraph and append its summary to an existing CSV.")
    parser.add_argument("--dataset", choices=["ton", "bot"], default="ton")
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--no-standardize", action="store_true")
    return parser


def default_paths(dataset: str) -> Tuple[str, Path, Path, Path]:
    if dataset == "bot":
        return (
            "NF-BoT-IoT",
            Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph"),
            Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_results_05%.json"),
            Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_summary_05%.csv"),
        )

    return (
        "NF-ToN-IoT",
        Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph"),
        Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_results_25%.json"),
        Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_summary_25%.csv"),
    )


def main() -> None:
    args = build_parser().parse_args()
    dataset_name, graph_dir, out_json, out_csv = default_paths(args.dataset)
    graph_dir = args.graph_dir or graph_dir
    out_json = args.out_json or out_json
    out_csv = args.out_csv or out_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)

    graphs = load_npz_graphs(
        graph_dir=graph_dir,
        max_graphs=args.max_graphs if args.max_graphs > 0 else None,
        max_nodes=args.max_nodes,
    )
    print(f"[OK] graph dir: {graph_dir}", flush=True)
    print(f"[OK] total graphs loaded: {len(graphs)}", flush=True)

    metrics_list = []
    split_info = {}
    labels_info = {}

    for seed in args.seeds:
        print(f"\n[RUN] seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        seed_graphs = clone_graphs(graphs)
        train_graphs, val_graphs, test_graphs = split_for_all_models(
            graphs=seed_graphs,
            seed=seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            benign_limit=args.benign_limit if args.benign_limit > 0 else None,
            mal_limit=args.mal_limit if args.mal_limit > 0 else None,
        )

        if not args.no_standardize:
            standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)

        if not split_info:
            split_info = {
                "train": len(train_graphs),
                "val": len(val_graphs),
                "test": len(test_graphs),
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            }
            labels_info = {
                "train_benign": int(sum(g.label == 0 for g in train_graphs)),
                "train_malicious": int(sum(g.label == 1 for g in train_graphs)),
                "test_benign": int(sum(g.label == 0 for g in test_graphs)),
                "test_malicious": int(sum(g.label == 1 for g in test_graphs)),
            }

        scores = train_infograph_scores(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=train_graphs[0].x.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=seed,
            device=device,
        )
        y_test = np.array([g.label for g in test_graphs], dtype=np.int64)
        metrics = compute_metrics(y_test, scores, threshold=args.threshold)
        metrics_list.append(metrics)
        print(f"[OK] seed={seed} metrics={metrics}", flush=True)

    summary = summarize_metrics(metrics_list)
    row = {"model": "infograph", **summary}
    payload = {
        "dataset": dataset_name,
        "graph_dir": str(graph_dir),
        "device": str(device),
        "num_graphs_loaded": len(graphs),
        "split": split_info,
        "labels": labels_info,
        "training": {
            "epochs_infograph": args.epochs,
            "lr_infograph": args.lr,
            "hidden_dim_infograph": args.hidden_dim,
            "num_layers_infograph": args.num_layers,
            "batch_size_infograph": args.batch_size,
            "threshold_infograph": args.threshold,
            "standardized": not args.no_standardize,
            "seeds": args.seeds,
            "benign_limit": args.benign_limit,
            "mal_limit": args.mal_limit,
            "max_graphs": args.max_graphs,
            "max_nodes": args.max_nodes,
        },
        "model": {"runs": metrics_list, "summary": summary},
    }

    update_results_json(out_json, payload)
    append_summary_csv(out_csv, row)

    print(f"\n[OK] updated {out_json}", flush=True)
    print(f"[OK] appended {out_csv}", flush=True)
    print("\nFinal summary:", flush=True)
    print(row, flush=True)


if __name__ == "__main__":
    main()
