#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")


def _first_present(npz: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> Optional[np.ndarray]:
    for name in names:
        if name in npz.files:
            return npz[name]
    return None


def _as_edge_index(edge_index: np.ndarray, num_nodes: int, source: Path) -> torch.Tensor:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError(f"{source.name}: edge_index must be 2-D, got shape {edge_index.shape}")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"{source.name}: edge_index must have shape [2, E] or [E, 2], got {edge_index.shape}")
    keep = (
        (edge_index[0] >= 0)
        & (edge_index[1] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] < num_nodes)
    )
    return torch.as_tensor(edge_index[:, keep], dtype=torch.long).contiguous()


def load_npz_graphs(graph_dir: Path) -> List[Data]:
    paths = sorted(graph_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz graph files found in {graph_dir}")

    dataset: List[Data] = []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        x_array = _first_present(npz, ("x", "node_features", "features"))
        edge_array = _first_present(npz, ("edge_index", "edges"))
        y_array = _first_present(npz, ("y", "label", "graph_label"))
        edge_attr_array = _first_present(npz, ("edge_attr", "edge_features", "edge_weight", "edge_weights"))

        if x_array is None:
            raise KeyError(f"{path.name}: expected node features under key 'x' or 'node_features'")
        if edge_array is None:
            raise KeyError(f"{path.name}: expected graph edges under key 'edge_index'")
        if y_array is None:
            raise KeyError(f"{path.name}: expected label under key 'y' or 'label'")

        x_array = np.asarray(x_array, dtype=np.float32)
        if x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)
        x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)
        num_nodes = int(x_array.shape[0])
        if num_nodes <= 0:
            raise ValueError(f"{path.name}: graph has no nodes")

        edge_index = _as_edge_index(edge_array, num_nodes=num_nodes, source=path)
        y = torch.as_tensor(np.asarray(y_array).reshape(-1)[0], dtype=torch.long).view(1)
        data = Data(x=torch.as_tensor(x_array, dtype=torch.float32), edge_index=edge_index, y=y)

        if edge_attr_array is not None:
            edge_attr_array = np.asarray(edge_attr_array, dtype=np.float32)
            if edge_attr_array.ndim == 1:
                edge_attr_array = edge_attr_array.reshape(-1, 1)
            if edge_attr_array.shape[0] == edge_index.shape[1]:
                data.edge_attr = torch.as_tensor(edge_attr_array, dtype=torch.float32)
            else:
                print(
                    f"[WARN] {path.name}: ignoring edge_attr with {edge_attr_array.shape[0]} rows "
                    f"for {edge_index.shape[1]} edges",
                    flush=True,
                )

        dataset.append(data)
    return dataset


def load_dataset(args: argparse.Namespace) -> List[Data]:
    if args.graph_dir is not None:
        graph_dir = Path(args.graph_dir).expanduser().resolve()
        print(f"[RGCL] using local .npz graphs from graph_dir={graph_dir}", flush=True)
        return load_npz_graphs(graph_dir)

    print(f"[RGCL] using TU dataset DS={args.DS} root={args.root}", flush=True)
    return list(TUDataset(root=args.root, name=args.DS))


def graph_features(data: Data, fallback_dim: int) -> np.ndarray:
    x = data.x
    if x is None:
        x = torch.ones((int(data.num_nodes), fallback_dim), dtype=torch.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    max_values = x.max(dim=0).values
    min_values = x.min(dim=0).values
    num_nodes = max(int(data.num_nodes), 1)
    num_edges = int(data.edge_index.shape[1]) if data.edge_index is not None else 0
    density = num_edges / max(float(num_nodes * max(num_nodes - 1, 1)), 1.0)
    stats = torch.tensor([num_nodes, num_edges, density], dtype=torch.float32)
    return torch.cat([mean, std, max_values, min_values, stats]).cpu().numpy()


def safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": 0.5,
        "auprc": float(np.mean(y_true == 1)) if y_true.size else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def evaluate(dataset: List[Data], seed: int) -> dict[str, float]:
    y = np.asarray([int(data.y.reshape(-1)[0].item()) for data in dataset], dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise RuntimeError("RGCL local evaluation expects a binary graph classification dataset")
    fallback_dim = int(dataset[0].x.shape[1]) if dataset[0].x is not None and dataset[0].x.ndim > 1 else 1
    x = np.vstack([graph_features(data, fallback_dim=fallback_dim) for data in dataset])

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    classifier.fit(x[train_idx], y[train_idx])
    y_pred = classifier.predict(x[test_idx])
    y_score = classifier.predict_proba(x[test_idx])[:, 1]
    return safe_binary_metrics(y[test_idx], y_pred, y_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGCL runner with optional local .npz graph support.")
    parser.add_argument("--DS", default="MUTAG", help="TU Dortmund dataset name used when --graph-dir is not provided")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default="data", help="Root directory for TUDataset downloads/cache")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Load local .npz graph files instead of TUDataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(args)
    print(f"[RGCL] loaded graphs={len(dataset)}", flush=True)
    metrics = evaluate(dataset, seed=args.seed)
    for name in METRIC_NAMES:
        print(f"{name}: {metrics[name]:.6f}", flush=True)


if __name__ == "__main__":
    main()
