#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool


CSV_FIELDNAMES = [
    "dataset",
    "train_ratio",
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
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
SCORE_CSV_FIELDNAMES = [
    "dataset",
    "train_ratio",
    "seed",
    "model",
    "index",
    "val_score",
    "y_val",
    "test_score",
    "y_test",
]


@dataclass
class GraphWindow:
    x: torch.Tensor
    edges_undirected: torch.Tensor
    num_nodes: int
    window_start: int
    label: int
    file_name: str
    # IMPROVED vs topids_topogcl_prototype: optional IDS metadata loaded when present.
    edge_weight: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None
    attack_density: Optional[float] = None
    feature_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelAugmentationConfig:
    edge_drop: float
    feat_mask: float
    tau: float
    batch_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    graph_dir: Path
    out_json: Path
    out_csv: Path
    out_scores_csv: Path
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
    epochs_graphcl: int
    epochs_topogcl: int
    epochs_infograph: int
    lr_graphcl: float
    lr_topogcl: float
    lr_infograph: float
    infograph_layers: int
    infograph_dir: Path
    rgcl_dir: Path
    hidden_dim: int
    emb_dim: int
    graphcl_layers: int
    edge_drop: float
    feat_mask: float
    tau: float
    batch_size: int
    benign_limit: int
    mal_limit: int
    max_graphs: int
    max_nodes: int
    standardize: bool
    models: Tuple[str, ...]
    # IMPROVED vs topids_topogcl_prototype: IDS-aware TopoGCL knobs.
    lambda_density: float
    ids_safe_augment: bool
    ids_filtrations: bool
    use_density_head: bool


def get_model_augmentation_config(model_name: str, config: ExperimentConfig) -> ModelAugmentationConfig:
    """Return model-specific contrastive augmentation settings.

    GraphCL keeps the user-provided/default augmentation behavior. TopoGCL
    intentionally uses a lighter topology-aware setup so small IDS graphs retain
    more of their edge structure during contrastive training.
    """
    model_configs = {
        "graphcl": ModelAugmentationConfig(
            edge_drop=config.edge_drop,
            feat_mask=config.feat_mask,
            tau=config.tau,
            batch_size=config.batch_size,
        ),
        "topogcl": ModelAugmentationConfig(
            edge_drop=0.01,
            feat_mask=0.05,
            tau=0.10,
            batch_size=config.batch_size,
        ),
    }
    try:
        return model_configs[model_name]
    except KeyError as exc:
        raise ValueError(f"No augmentation config defined for model: {model_name}") from exc


def format_config_float(value: float) -> str:
    return f"{value:.3f}" if 0 < abs(value) < 0.01 else f"{value:.2f}"


def log_augmentation_config(model_name: str, aug_config: ModelAugmentationConfig) -> None:
    print(
        f"[CONFIG] model={model_name} edge_drop={format_config_float(aug_config.edge_drop)} "
        f"feat_mask={format_config_float(aug_config.feat_mask)} "
        f"tau={format_config_float(aug_config.tau)} batch_size={aug_config.batch_size}",
        flush=True,
    )


# =========================================================
# Data loading and split helpers. These preserve the existing
# graph construction output format and train/val/test split logic.
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
            label = int(np.array(arr["label"]).reshape(-1)[0])
            edge_index = arr["edge_index"].astype(np.int64)

            if node_features.ndim == 1:
                node_features = node_features.reshape(-1, 1)

            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            num_nodes = int(node_features.shape[0])
            if num_nodes == 0 or num_nodes > max_nodes:
                continue

            # IMPROVED vs topids_topogcl_prototype: preserve the required .npz I/O while
            # opportunistically reading IDS metadata used by the improved TopoGCL path.
            raw_edge_weight = None
            for key in ("edge_weight", "edge_weights", "weights"):
                if key in arr:
                    raw_edge_weight = np.asarray(arr[key], dtype=np.float32).reshape(-1)
                    break
            feature_names: Tuple[str, ...] = ()
            for key in ("feature_names", "node_feature_names", "columns"):
                if key in arr:
                    feature_names = tuple(str(v) for v in np.asarray(arr[key]).reshape(-1).tolist())
                    break
            edge_features = None
            if "edge_features" in arr:
                edge_features = np.asarray(arr["edge_features"], dtype=np.float32)

            if not feature_names and node_features.shape[1] == 8:
                feature_names = (
                    "in_degree",
                    "out_degree",
                    "total_in_bytes",
                    "total_out_bytes",
                    "total_in_packets",
                    "total_out_packets",
                    "mean_flow_duration",
                    "total_flow_count",
                )

            attack_density = None
            if "attack_density" in arr:
                attack_density = float(np.asarray(arr["attack_density"]).reshape(-1)[0])

            undirected_edges = edge_index_to_undirected(edge_index, num_nodes=num_nodes)
            edge_weight = None
            if raw_edge_weight is not None and raw_edge_weight.size > 0:
                if raw_edge_weight.size >= undirected_edges.shape[1]:
                    edge_weight = torch.tensor(raw_edge_weight[: undirected_edges.shape[1]], dtype=torch.float32)
                else:
                    edge_weight = torch.ones(undirected_edges.shape[1], dtype=torch.float32)
            elif edge_features is not None and edge_features.ndim == 2 and edge_features.shape[0] >= undirected_edges.shape[1]:
                # NF graph builder edge_features columns start with IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS.
                traffic_cols = min(4, edge_features.shape[1])
                edge_weight = torch.tensor(edge_features[: undirected_edges.shape[1], :traffic_cols].sum(axis=1), dtype=torch.float32)

            graphs.append(
                GraphWindow(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edges_undirected=undirected_edges,
                    num_nodes=num_nodes,
                    window_start=len(graphs),
                    label=label,
                    file_name=path.name,
                    edge_weight=edge_weight,
                    edge_features=torch.tensor(edge_features, dtype=torch.float32) if edge_features is not None else None,
                    attack_density=attack_density,
                    feature_names=feature_names,
                )
            )
            if max_graphs is not None and max_graphs > 0 and len(graphs) >= max_graphs:
                break
        except Exception as exc:
            print(f"[WARN] skipped {path.name}: {exc}", flush=True)

    if not graphs:
        raise RuntimeError(f"No usable .npz graphs loaded from {graph_dir}")
    return graphs


def clone_graphs(graphs: Sequence[GraphWindow]) -> List[GraphWindow]:
    return [
        GraphWindow(
            x=g.x.clone(),
            edges_undirected=g.edges_undirected.clone(),
            num_nodes=g.num_nodes,
            window_start=g.window_start,
            label=g.label,
            file_name=g.file_name,
            edge_weight=g.edge_weight.clone() if g.edge_weight is not None else None,
            edge_features=g.edge_features.clone() if g.edge_features is not None else None,
            attack_density=g.attack_density,
            feature_names=g.feature_names,
        )
        for g in graphs
    ]


def standardize_from_train(train_graphs: List[GraphWindow], all_graphs: List[GraphWindow]) -> None:
    xs = torch.cat([g.x for g in train_graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)
    for g in all_graphs:
        g.x = torch.nan_to_num((g.x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


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
    train_idx, temp_idx, _y_train, y_temp = train_test_split(
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
# Graph neural network layers and encoders
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
            values = torch.ones(n, dtype=torch.float32)
        else:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros((0,), dtype=torch.float32)
    else:
        u = edges_undirected[0]
        v = edges_undirected[1]
        src = torch.cat([u, v], dim=0)
        dst = torch.cat([v, u], dim=0)
        indices = torch.stack([src, dst], dim=0)
        values = torch.ones(indices.shape[1], dtype=torch.float32)
        if add_self_loops:
            idx = torch.arange(n, dtype=torch.long)
            self_indices = torch.stack([idx, idx], dim=0)
            indices = torch.cat([indices, self_indices], dim=1)
            values = torch.cat([values, torch.ones(n, dtype=torch.float32)], dim=0)

    a = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    d = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = torch.pow(d, -0.5)
    row, col = a.indices()
    norm_vals = a.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    return torch.sparse_coo_tensor(a.indices(), norm_vals, (n, n)).coalesce()


class GCN(torch.nn.Module):
    """Small GCN encoder retained for TopoGCL embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_hat, self.w1(x)))
        h = torch.sparse.mm(a_hat, self.w2(h))
        return h.mean(dim=0)


class InfoGraphFF(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class InfoGraphEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_gc_layers: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer_idx in range(num_gc_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.output_dim = hidden_dim * num_gc_layers

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            x = torch.ones((batch.numel(), self.input_dim), dtype=torch.float32, device=batch.device)
        h = x.float()
        local_embeddings: List[torch.Tensor] = []
        global_embeddings: List[torch.Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            if h.shape[0] > 1:
                h = norm(h)
            h = torch.relu(h)
            local_embeddings.append(h)
            global_embeddings.append(global_add_pool(h, batch))
        return torch.cat(local_embeddings, dim=1), torch.cat(global_embeddings, dim=1)


class InfoGraphIDS(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_gc_layers: int = 3) -> None:
        super().__init__()
        self.encoder = InfoGraphEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_gc_layers=num_gc_layers)
        self.local_ff = InfoGraphFF(self.encoder.output_dim, hidden_dim)
        self.global_ff = InfoGraphFF(self.encoder.output_dim, hidden_dim)
        self.output_dim = self.encoder.output_dim

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        local_emb, global_emb = self.encoder(data.x, data.edge_index, data.batch)
        return self.local_ff(local_emb), self.global_ff(global_emb)

    def embed(self, data: Data) -> torch.Tensor:
        _, global_emb = self.encoder(data.x, data.edge_index, data.batch)
        return global_emb


def graph_to_pyg_data(graph: GraphWindow, fallback_in_dim: int) -> Data:
    x = graph.x
    if x is None:
        x = torch.ones((graph.num_nodes, fallback_in_dim), dtype=torch.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    edges = graph.edges_undirected
    if edges.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.cat([edges, edges.flip(0)], dim=1).long().contiguous()
    return Data(x=x.float(), edge_index=edge_index, y=torch.tensor([graph.label], dtype=torch.long))


def make_pyg_loader(
    graphs: List[GraphWindow],
    batch_size: int,
    shuffle: bool,
    fallback_in_dim: int,
) -> DataLoader:
    return DataLoader(
        [graph_to_pyg_data(graph, fallback_in_dim=fallback_in_dim) for graph in graphs],
        batch_size=batch_size,
        shuffle=shuffle,
    )


def infograph_local_global_loss(local_emb: torch.Tensor, global_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(local_emb, global_emb.t())
    positive_scores = scores[torch.arange(local_emb.shape[0], device=local_emb.device), batch]
    positive_loss = torch.nn.functional.softplus(-positive_scores).mean()

    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    negative_mask[torch.arange(local_emb.shape[0], device=local_emb.device), batch] = False
    if negative_mask.any():
        negative_loss = torch.nn.functional.softplus(scores[negative_mask]).mean()
    else:
        negative_loss = torch.zeros((), dtype=local_emb.dtype, device=local_emb.device)
    return positive_loss + negative_loss


def compute_infograph_embeddings(model: InfoGraphIDS, loader: DataLoader, device: torch.device, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    total = len(loader.dataset)
    seen = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb = model.embed(data).detach().cpu()
            embeddings.append(emb)
            labels.append(data.y.view(-1).detach().cpu())
            seen += emb.shape[0]
            if seen % max(1, total // 10) == 0 or seen == total:
                print(f"    {tag}: {seen}/{total}", flush=True)
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def train_infograph_ids(
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    train_loader = make_pyg_loader(train_graphs, batch_size=config.batch_size, shuffle=True, fallback_in_dim=in_dim)
    train_eval_loader = make_pyg_loader(train_graphs, batch_size=config.batch_size, shuffle=False, fallback_in_dim=in_dim)
    test_loader = make_pyg_loader(test_graphs, batch_size=config.batch_size, shuffle=False, fallback_in_dim=in_dim)

    model = InfoGraphIDS(input_dim=in_dim, hidden_dim=config.hidden_dim, num_gc_layers=config.infograph_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_infograph)

    for epoch in range(config.epochs_infograph):
        model.train()
        total_loss = 0.0
        seen = 0
        for data in train_loader:
            data = data.to(device)
            local_emb, global_emb = model(data)
            loss = infograph_local_global_loss(local_emb, global_emb, data.batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            graphs_in_batch = int(data.y.numel())
            total_loss += float(loss.item()) * graphs_in_batch
            seen += graphs_in_batch
        print(f"    infograph epoch {epoch + 1}/{config.epochs_infograph} loss={total_loss / max(seen, 1):.6f}", flush=True)

    train_emb, y_train = compute_infograph_embeddings(model, train_eval_loader, device=device, tag="embed infograph train")
    test_emb, y_test = compute_infograph_embeddings(model, test_loader, device=device, tag="embed infograph test")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("InfoGraph logistic regression needs both classes in the train split.")
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    classifier.fit(train_emb, y_train)
    y_pred = classifier.predict(test_emb)
    if hasattr(classifier, "predict_proba"):
        y_score = classifier.predict_proba(test_emb)[:, 1]
    else:
        y_score = classifier.decision_function(test_emb)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": safe_auc(y_test, y_score),
        "auprc": safe_auprc(y_test, y_score),
    }


# =========================================================
# GraphCL encoder, trained and evaluated without a downstream SVM.
# Scores are distances from the benign training embedding center.
# =========================================================
def augment_graph_view(
    graph: GraphWindow,
    edge_drop: float,
    feat_mask: float,
    rng: torch.Generator,
    identity: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = graph.edges_undirected
    x = graph.x
    if not identity and edges.numel() > 0 and edge_drop > 0:
        keep = torch.rand(edges.shape[1], generator=rng) > edge_drop
        edges = edges[:, keep]
    if not identity and feat_mask > 0:
        x = x * (torch.rand(x.shape, generator=rng) > feat_mask).float()
    return x, build_sparse_a_hat_from_undirected(graph.num_nodes, edges, add_self_loops=True)


# IMPROVED vs topids_topogcl_prototype: IDS-aware filtration and augmentation helpers.
IDS_VOLUME_TOKENS = ("byte", "bytes", "packet", "packets", "pkt", "pkts", "volume", "duration", "rate")
IDS_DIVERSITY_TOKENS = ("proto", "protocol", "service", "state", "flag", "port")
IDS_CRITICAL_TOKENS = IDS_VOLUME_TOKENS + IDS_DIVERSITY_TOKENS + ("attack", "label", "density")


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if score.numel() == 0:
        return score
    min_value = score.min()
    max_value = score.max()
    if float(max_value - min_value) < 1e-12:
        return torch.zeros_like(score)
    return (score - min_value) / (max_value - min_value)


def _matching_feature_columns(feature_names: Tuple[str, ...], feature_dim: int, tokens: Tuple[str, ...]) -> List[int]:
    matches: List[int] = []
    for idx, name in enumerate(feature_names[:feature_dim]):
        lower = name.lower()
        if any(token in lower for token in tokens):
            matches.append(idx)
    return matches


def ids_filtration_features(graph: GraphWindow, fallback_x: torch.Tensor) -> torch.Tensor:
    """Build IDS-aware topological landscape features when metadata is available.

    The returned tensor keeps the prototype input dimensionality so downstream model
    and output behavior remain unchanged. If IDS signals are unavailable, the first
    prototype topology columns are preserved as a clean fallback.
    """
    n = graph.num_nodes
    edges = graph.edges_undirected
    if n <= 0:
        return fallback_x

    in_degree = torch.zeros(n, dtype=torch.float32)
    out_degree = torch.zeros(n, dtype=torch.float32)
    if edges.numel() > 0:
        src, dst = edges[0], edges[1]
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        # The prototype stores undirected edges; mirror them to make in/out meaningful.
        out_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        in_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    total_degree = in_degree + out_degree

    signals: List[torch.Tensor] = [in_degree, out_degree, total_degree]
    if graph.edge_weight is not None and edges.numel() > 0 and graph.edge_weight.numel() == edges.shape[1]:
        weighted = torch.zeros(n, dtype=torch.float32)
        w = graph.edge_weight.float().clamp_min(0.0)
        weighted.scatter_add_(0, edges[0], w)
        weighted.scatter_add_(0, edges[1], w)
        signals.append(weighted)

    volume_cols = _matching_feature_columns(graph.feature_names, graph.x.shape[1], IDS_VOLUME_TOKENS)
    if volume_cols:
        signals.append(graph.x[:, volume_cols].abs().sum(dim=1).detach().cpu())

    diversity_cols = _matching_feature_columns(graph.feature_names, graph.x.shape[1], IDS_DIVERSITY_TOKENS)
    if diversity_cols:
        signals.append((graph.x[:, diversity_cols].abs() > 0).float().sum(dim=1).detach().cpu())
    elif graph.edge_features is not None and graph.edge_features.ndim == 2 and graph.edge_features.shape[1] >= 6 and edges.numel() > 0:
        # NF edge features include PROTOCOL, L4_SRC_PORT, and L4_DST_PORT; count distinct
        # observed protocol/service values per endpoint when those columns are available.
        diversity_sets = [set() for _ in range(n)]
        edge_feature_rows = min(edges.shape[1], graph.edge_features.shape[0])
        for edge_pos in range(edge_feature_rows):
            values = graph.edge_features[edge_pos, 5 : min(graph.edge_features.shape[1], 8)].detach().cpu().tolist()
            u = int(edges[0, edge_pos])
            v = int(edges[1, edge_pos])
            for raw_value in values:
                value = float(raw_value)
                if value != 0.0:
                    diversity_sets[u].add(value)
                    diversity_sets[v].add(value)
        signals.append(torch.tensor([len(values) for values in diversity_sets], dtype=torch.float32))

    if len(signals) <= 3 and not volume_cols and not diversity_cols and graph.edge_weight is None:
        topo = fallback_x.clone()
        if topo.shape[1] > 3:
            topo[:, 3:] = 0.0
        return topo

    topo = torch.zeros_like(fallback_x)
    for col, score in enumerate(signals[: topo.shape[1]]):
        topo[:, col] = _normalize_score(score).to(topo.device)
    return topo


def augment_graph_view_ids_safe(
    graph: GraphWindow,
    edge_drop: float,
    feat_mask: float,
    rng: torch.Generator,
    identity: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IDS-safe augmentation: drop low-value edges and mask non-critical columns first."""
    if identity:
        return augment_graph_view(graph, edge_drop, feat_mask, rng, identity=True)

    edges = graph.edges_undirected
    x = graph.x.clone()
    if edges.numel() > 0 and edge_drop > 0:
        edge_count = edges.shape[1]
        drop_count = min(edge_count - 1, max(0, int(round(edge_count * edge_drop))))
        keep = torch.ones(edge_count, dtype=torch.bool)
        if drop_count > 0:
            if graph.edge_weight is not None and graph.edge_weight.numel() == edge_count:
                candidates = torch.argsort(graph.edge_weight.float())
            else:
                deg = torch.zeros(graph.num_nodes, dtype=torch.float32)
                deg.scatter_add_(0, edges[0], torch.ones(edge_count))
                deg.scatter_add_(0, edges[1], torch.ones(edge_count))
                # Prefer redundant high-degree endpoints and avoid isolating rare nodes.
                candidates = torch.argsort(deg[edges[0]] + deg[edges[1]], descending=True)
            degree = torch.zeros(graph.num_nodes, dtype=torch.long)
            degree.scatter_add_(0, edges[0], torch.ones(edge_count, dtype=torch.long))
            degree.scatter_add_(0, edges[1], torch.ones(edge_count, dtype=torch.long))
            dropped = 0
            for edge_idx in candidates.tolist():
                u = int(edges[0, edge_idx])
                v = int(edges[1, edge_idx])
                if degree[u] <= 1 or degree[v] <= 1:
                    continue
                keep[edge_idx] = False
                degree[u] -= 1
                degree[v] -= 1
                dropped += 1
                if dropped >= drop_count:
                    break
        edges = edges[:, keep]

    if feat_mask > 0:
        critical = set(_matching_feature_columns(graph.feature_names, graph.x.shape[1], IDS_CRITICAL_TOKENS))
        maskable = [idx for idx in range(graph.x.shape[1]) if idx not in critical]
        if maskable:
            random_mask = torch.rand((graph.x.shape[0], len(maskable)), generator=rng) > feat_mask
            x[:, maskable] = x[:, maskable] * random_mask.float()
        else:
            # Fallback to prototype random feature masking when metadata is unavailable.
            x = x * (torch.rand(x.shape, generator=rng) > feat_mask).float()
    return x, build_sparse_a_hat_from_undirected(graph.num_nodes, edges, add_self_loops=True)


class GraphCLGINEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.norms.append(torch.nn.LayerNorm(hidden_dim))
        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            torch.nn.Linear(project_dim, project_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(project_dim, project_dim),
        )
        self.output_dim = project_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor, project: bool = False) -> torch.Tensor:
        h = x
        graph_embeddings: List[torch.Tensor] = []
        for mlp, norm in zip(self.mlps, self.norms):
            h = torch.sparse.mm(adj, h)
            h = torch.relu(norm(mlp(h)))
            graph_embeddings.append(h.mean(dim=0))
        z = torch.cat(graph_embeddings, dim=0)
        return self.project(z) if project else z


def graphcl_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.t(), labels)
    )


def train_graphcl_encoder(
    train_graphs: List[GraphWindow],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    edge_drop: float,
    feat_mask: float,
    tau: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> GraphCLGINEncoder:
    torch.manual_seed(seed)
    model = GraphCLGINEncoder(input_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(train_graphs))

    for epoch in range(epochs):
        model.train()
        permutation = indices[torch.randperm(len(train_graphs), generator=rng)]
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_graphs), batch_size):
            batch_indices = permutation[start : start + batch_size].tolist()
            if len(batch_indices) < 2:
                continue
            z1_list: List[torch.Tensor] = []
            z2_list: List[torch.Tensor] = []
            for idx in batch_indices:
                graph = train_graphs[idx]
                x1, adj1 = augment_graph_view(graph, edge_drop, feat_mask, rng, identity=True)
                x2, adj2 = augment_graph_view(graph, edge_drop, feat_mask, rng, identity=False)
                z1_list.append(model(x1.to(device), adj1.to(device), project=True))
                z2_list.append(model(x2.to(device), adj2.to(device), project=True))
            loss = graphcl_loss(torch.stack(z1_list), torch.stack(z2_list), tau=tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        print(f"    graphcl epoch {epoch + 1}/{epochs} loss={total_loss / max(seen, 1):.6f}", flush=True)
    return model


def compute_graphcl_embeddings(model: GraphCLGINEncoder, graphs: List[GraphWindow], device: torch.device, tag: str) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for idx, graph in enumerate(graphs):
            x, adj = augment_graph_view(graph, edge_drop=0.0, feat_mask=0.0, rng=torch.Generator(), identity=True)
            embeddings.append(model(x.to(device), adj.to(device), project=False).detach())
            if (idx + 1) % max(1, len(graphs) // 10) == 0:
                print(f"    {tag}: {idx + 1}/{len(graphs)}", flush=True)
    return torch.stack(embeddings, dim=0)


def train_graphcl_scores(
    train_graphs: List[GraphWindow],
    eval_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    encoder = train_graphcl_encoder(
        train_graphs=train_graphs,
        in_dim=in_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.graphcl_layers,
        epochs=config.epochs_graphcl,
        lr=config.lr_graphcl,
        edge_drop=config.edge_drop,
        feat_mask=config.feat_mask,
        tau=config.tau,
        batch_size=config.batch_size,
        seed=seed,
        device=device,
    )
    train_benign = [g for g in train_graphs if g.label == 0]
    if not train_benign:
        raise RuntimeError("GraphCL distance scoring needs at least one benign graph in the train split.")
    train_emb = compute_graphcl_embeddings(encoder, train_benign, device=device, tag="embed graphcl train benign")
    eval_emb = compute_graphcl_embeddings(encoder, eval_graphs, device=device, tag="embed graphcl eval")
    center = train_emb.mean(dim=0)
    distances = torch.norm(eval_emb - center, p=2, dim=1).detach().cpu().numpy()
    return distances.astype(np.float32)


# =========================================================
# TopoGCL/TopoIDS-style contrastive graph scoring already in the pipeline.
# =========================================================
def contrastive_component_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1_abs = z1.norm(dim=1).clamp_min(1e-12)
    z2_abs = z2.norm(dim=1).clamp_min(1e-12)
    sim = torch.exp(torch.einsum("ik,jk->ij", z1, z2) / torch.einsum("i,j->ij", z1_abs, z2_abs) / tau)
    pos = sim[range(z1.size(0)), range(z1.size(0))]
    ratio = pos / (sim.sum(dim=1) - pos).clamp_min(1e-12)
    return (-torch.log(ratio.clamp_min(1e-12))).mean()


def loss_cal(z1: torch.Tensor, z2: torch.Tensor, zt1: torch.Tensor, zt2: torch.Tensor, tau: float) -> torch.Tensor:
    # Prototype-compatible wrapper retained for callers that expect the old TopoGCL loss.
    graph_contrastive_loss = contrastive_component_loss(z1, z2, tau)
    topo_contrastive_loss = contrastive_component_loss(zt1, zt2, tau)
    return graph_contrastive_loss + 0.1 * topo_contrastive_loss


def train_topogcl_encoder(
    model: GCN,
    graphs: List[GraphWindow],
    config: ExperimentConfig,
    aug_config: ModelAugmentationConfig,
    seed: int,
    device: torch.device,
) -> None:
    model.to(device)
    # IMPROVED vs topids_topogcl_prototype: optional attack-density auxiliary head.
    has_density = config.use_density_head and any(g.attack_density is not None for g in graphs)
    density_head = torch.nn.Linear(model.output_dim, 1).to(device) if has_density else None
    parameters = list(model.parameters()) + (list(density_head.parameters()) if density_head is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=config.lr_topogcl)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(graphs))

    if config.use_density_head and not has_density:
        print("    topogcl density head skipped: attack_density metadata not found", flush=True)

    for epoch in range(config.epochs_topogcl):
        permutation = indices[torch.randperm(len(graphs), generator=rng)]
        total_loss = 0.0
        seen = 0
        for start in range(0, len(graphs), aug_config.batch_size):
            batch_indices = permutation[start : start + aug_config.batch_size].tolist()
            if len(batch_indices) < 2:
                continue
            z1_list: List[torch.Tensor] = []
            z2_list: List[torch.Tensor] = []
            zt1_list: List[torch.Tensor] = []
            zt2_list: List[torch.Tensor] = []
            density_pred: List[torch.Tensor] = []
            density_target: List[float] = []
            for idx in batch_indices:
                graph = graphs[idx]
                augment = augment_graph_view_ids_safe if config.ids_safe_augment else augment_graph_view
                x1, adj1 = augment(graph, aug_config.edge_drop, aug_config.feat_mask, rng)
                x2, adj2 = augment(graph, aug_config.edge_drop, aug_config.feat_mask, rng)
                x1 = x1.to(device)
                x2 = x2.to(device)
                adj1 = adj1.to(device)
                adj2 = adj2.to(device)
                z1 = model(x1, adj1)
                z2 = model(x2, adj2)
                z1_list.append(z1)
                z2_list.append(z2)

                # IMPROVED vs topids_topogcl_prototype: EPL/topological-landscape-style
                # IDS filtration view is used by default when available; otherwise the
                # prototype's first-three-feature topology masking is preserved.
                if config.ids_filtrations:
                    x1_topo = ids_filtration_features(graph, x1.detach().cpu()).to(device)
                    x2_topo = ids_filtration_features(graph, x2.detach().cpu()).to(device)
                else:
                    x1_topo = x1.clone()
                    x2_topo = x2.clone()
                    if x1_topo.shape[1] > 3:
                        x1_topo[:, 3:] = 0.0
                        x2_topo[:, 3:] = 0.0
                zt1_list.append(model(x1_topo, adj1))
                zt2_list.append(model(x2_topo, adj2))

                if density_head is not None and graph.attack_density is not None:
                    density_pred.append(density_head(z1).view(()))
                    density_target.append(float(graph.attack_density))

            graph_contrastive_loss = contrastive_component_loss(torch.stack(z1_list), torch.stack(z2_list), tau=aug_config.tau)
            topo_contrastive_loss = contrastive_component_loss(torch.stack(zt1_list), torch.stack(zt2_list), tau=aug_config.tau)
            density_loss = torch.tensor(0.0, device=device)
            if density_head is not None and density_pred:
                pred = torch.stack(density_pred)
                target = torch.tensor(density_target, dtype=torch.float32, device=device)
                density_loss = torch.nn.functional.mse_loss(pred, target)
            # IMPROVED final loss: graph + topo + lambda_density * density.
            loss = graph_contrastive_loss + topo_contrastive_loss + config.lambda_density * density_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        print(f"    topogcl epoch {epoch + 1}/{config.epochs_topogcl} loss={total_loss / max(seen, 1):.6f}", flush=True)


def compute_gcn_embeddings(model: GCN, graphs: List[GraphWindow], device: torch.device, tag: str) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for idx, graph in enumerate(graphs):
            adj = build_sparse_a_hat_from_undirected(graph.num_nodes, graph.edges_undirected).to(device)
            embeddings.append(model(graph.x.to(device), adj).detach())
            if (idx + 1) % max(1, len(graphs) // 10) == 0:
                print(f"    {tag}: {idx + 1}/{len(graphs)}", flush=True)
    return torch.stack(embeddings, dim=0)


def train_topogcl_scores(
    train_graphs: List[GraphWindow],
    eval_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    train_benign = [g for g in train_graphs if g.label == 0]
    if len(train_benign) < 2:
        raise RuntimeError("TopoGCL needs at least two benign graphs in the train split.")
    model = GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.emb_dim)
    aug_config = get_model_augmentation_config("topogcl", config)
    train_topogcl_encoder(model=model, graphs=train_benign, config=config, aug_config=aug_config, seed=seed, device=device)
    train_emb = compute_gcn_embeddings(model, train_benign, device=device, tag="embed topogcl train benign")
    eval_emb = compute_gcn_embeddings(model, eval_graphs, device=device, tag="embed topogcl eval")
    center = train_emb.mean(dim=0)
    return torch.norm(eval_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32)


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self1 = torch.nn.Linear(in_dim, hidden_dim)
        self.neigh1 = torch.nn.Linear(in_dim, hidden_dim)
        self.self2 = torch.nn.Linear(hidden_dim, out_dim)
        self.neigh2 = torch.nn.Linear(hidden_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.self1(x) + self.neigh1(torch.sparse.mm(adj, x)))
        h = torch.relu(self.self2(h) + self.neigh2(torch.sparse.mm(adj, h)))
        return h.mean(dim=0)


class GraphClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, emb_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(emb_dim, 2)

    def forward(self, graph: GraphWindow, device: torch.device) -> torch.Tensor:
        adj = build_sparse_a_hat_from_undirected(graph.num_nodes, graph.edges_undirected).to(device)
        z = self.encoder(graph.x.to(device), adj)
        return self.head(z)


def train_supervised_graph_model(
    model_name: str,
    encoder_factory: Callable[[], torch.nn.Module],
    train_graphs: List[GraphWindow],
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> GraphClassifier:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    classifier = GraphClassifier(encoder_factory(), emb_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        classifier.train()
        shuffled = train_graphs[:]
        rng.shuffle(shuffled)
        total_loss = 0.0
        for graph in shuffled:
            logits = classifier(graph, device).unsqueeze(0)
            target = torch.tensor([graph.label], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"    {model_name} epoch {epoch + 1}/{epochs} loss={total_loss / max(len(shuffled), 1):.6f}", flush=True)
    return classifier


def predict_supervised_graph_model(
    classifier: GraphClassifier,
    eval_graphs: List[GraphWindow],
    device: torch.device,
) -> np.ndarray:
    classifier.eval()
    scores: List[float] = []
    with torch.no_grad():
        for graph in eval_graphs:
            logits = classifier(graph, device)
            scores.append(float(torch.softmax(logits, dim=0)[1].item()))
    return np.array(scores, dtype=np.float32)


# =========================================================
# Metrics, thresholds, and result writing
# =========================================================
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def best_threshold_from_validation(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_score.size == 0:
        return 0.5
    candidates = np.unique(y_score)
    if candidates.size == 1:
        return float(candidates[0])
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(np.int64)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": safe_auc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
    }


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.array([metrics[name] for metrics in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(values))
        summary[f"{name}_std"] = float(np.nanstd(values))
    return summary


def read_existing_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_summary_csv(path: Path, rows: List[Dict[str, object]], append: bool) -> None:
    merged_rows: List[Dict[str, object]] = []
    if append:
        incoming_keys = {(str(row["dataset"]), str(row["train_ratio"]), str(row["model"])) for row in rows}
        for existing in read_existing_csv_rows(path):
            existing_key = (
                str(existing.get("dataset", "")),
                str(existing.get("train_ratio", "")),
                str(existing.get("model", "")),
            )
            if existing_key not in incoming_keys:
                merged_rows.append(existing)
    merged_rows.extend(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def make_score_rows(
    dataset: str,
    train_ratio: float,
    seed: int,
    model_name: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_len = max(len(val_scores), len(test_scores), len(y_val), len(y_test))
    for idx in range(max_len):
        rows.append(
            {
                "dataset": dataset,
                "train_ratio": train_ratio,
                "seed": seed,
                "model": model_name,
                "index": idx,
                "val_score": float(val_scores[idx]) if idx < len(val_scores) else "",
                "y_val": int(y_val[idx]) if idx < len(y_val) else "",
                "test_score": float(test_scores[idx]) if idx < len(test_scores) else "",
                "y_test": int(y_test[idx]) if idx < len(y_test) else "",
            }
        )
    return rows


def write_scores_csv(path: Path, rows: List[Dict[str, object]], append: bool) -> None:
    merged_rows: List[Dict[str, object]] = []
    if append:
        incoming_run_keys = {
            (
                str(row["dataset"]),
                str(row["train_ratio"]),
                str(row["seed"]),
                str(row["model"]),
            )
            for row in rows
        }
        for existing in read_existing_csv_rows(path):
            existing_run_key = (
                str(existing.get("dataset", "")),
                str(existing.get("train_ratio", "")),
                str(existing.get("seed", "")),
                str(existing.get("model", "")),
            )
            if existing_run_key not in incoming_run_keys:
                merged_rows.append(existing)
    merged_rows.extend(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in SCORE_CSV_FIELDNAMES})


def default_scores_csv_path(out_csv: Path) -> Path:
    if "summary" in out_csv.stem:
        return out_csv.with_name(out_csv.name.replace("summary", "scores", 1))
    return out_csv.with_name(f"{out_csv.stem}_scores{out_csv.suffix or '.csv'}")


EXTERNAL_METRIC_PATTERNS = {
    "accuracy": ("accuracy", "acc"),
    "precision": ("precision", "prec"),
    "recall": ("recall", "rec"),
    "f1": ("f1", "f1_score", "f1-score"),
    "auroc": ("auroc", "roc_auc", "roc-auc", "auc"),
    "auprc": ("auprc", "average_precision", "avg_precision", "ap"),
}


def normalize_external_dataset_name(dataset: str) -> str:
    """Map display names to TU-style dataset identifiers accepted by external repos."""
    return (
        dataset.replace("NF-", "NF_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("%", "")
    )


def parse_external_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for canonical, aliases in EXTERNAL_METRIC_PATTERNS.items():
        matches: List[float] = []
        for alias in aliases:
            pattern = rf"(?i)(?:^|[^a-z0-9_]){re.escape(alias)}(?:[^a-z0-9_]|$)\s*[:=,\t ]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            matches.extend(float(match) for match in re.findall(pattern, output))
        if matches:
            value = matches[-1]
            metrics[canonical] = value / 100.0 if value > 1.0 and value <= 100.0 else value
    missing = [name for name in METRIC_NAMES if name not in metrics]
    if missing:
        raise RuntimeError(
            "Could not parse external metrics "
            f"{missing}. Expected keys like accuracy, precision, recall, f1, auroc, and auprc in output."
        )
    return metrics


def run_external_command(model_name: str, command: List[str], workdir: Path) -> Dict[str, float]:
    if not workdir.exists():
        raise FileNotFoundError(f"{model_name} directory not found: {workdir}")
    print(f"    running {model_name}: {' '.join(command)} (cwd={workdir})", flush=True)
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{model_name} command failed with exit code {completed.returncode}")
    return parse_external_metrics(completed.stdout)


def run_rgcl_external(config: ExperimentConfig, seed: int) -> Dict[str, float]:
    dataset_name = normalize_external_dataset_name(config.dataset)
    command = ["python", "rgcl.py", "--seed", str(seed), "--DS", dataset_name]
    dataset_key = dataset_name.lower()
    if dataset_key in {"nf_ton_iot", "nf_bot_iot"}:
        command.extend(["--graph-dir", str(config.graph_dir)])
        print(f"    RGCL source: local .npz graphs ({config.graph_dir})", flush=True)
    else:
        print(f"    RGCL source: TU dataset ({dataset_name})", flush=True)
    return run_external_command(
        model_name="rgcl",
        command=command,
        workdir=config.rgcl_dir,
    )


def model_runner(
    model_name: str,
    train_graphs: List[GraphWindow],
    val_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray] | Dict[str, float]:
    if model_name in {"gnn", "graphsage"}:
        if model_name == "gnn":
            factory = lambda: GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim)
        else:
            factory = lambda: GraphSAGEEncoder(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim)
        classifier = train_supervised_graph_model(
            model_name=model_name,
            encoder_factory=factory,
            train_graphs=train_graphs,
            hidden_dim=config.hidden_dim,
            epochs=config.epochs_topogcl,
            lr=config.lr_topogcl,
            seed=seed,
            device=device,
        )
        return (
            predict_supervised_graph_model(classifier, val_graphs, device),
            predict_supervised_graph_model(classifier, test_graphs, device),
        )

    if model_name == "graphcl":
        aug_config = get_model_augmentation_config(model_name, config)
        log_augmentation_config(model_name, aug_config)
        encoder = train_graphcl_encoder(
            train_graphs=train_graphs,
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.graphcl_layers,
            epochs=config.epochs_graphcl,
            lr=config.lr_graphcl,
            edge_drop=aug_config.edge_drop,
            feat_mask=aug_config.feat_mask,
            tau=aug_config.tau,
            batch_size=aug_config.batch_size,
            seed=seed,
            device=device,
        )
        train_benign = [g for g in train_graphs if g.label == 0]
        if not train_benign:
            raise RuntimeError("GraphCL distance scoring needs at least one benign graph in the train split.")
        train_emb = compute_graphcl_embeddings(encoder, train_benign, device=device, tag="embed graphcl train benign")
        center = train_emb.mean(dim=0)
        val_emb = compute_graphcl_embeddings(encoder, val_graphs, device=device, tag="embed graphcl val")
        test_emb = compute_graphcl_embeddings(encoder, test_graphs, device=device, tag="embed graphcl test")
        return (
            torch.norm(val_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
            torch.norm(test_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
        )

    if model_name == "topogcl":
        aug_config = get_model_augmentation_config(model_name, config)
        log_augmentation_config(model_name, aug_config)
        train_benign = [g for g in train_graphs if g.label == 0]
        if len(train_benign) < 2:
            raise RuntimeError("TopoGCL needs at least two benign graphs in the train split.")
        model = GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.emb_dim)
        train_topogcl_encoder(
            model=model,
            graphs=train_benign,
            config=config,
            aug_config=aug_config,
            seed=seed,
            device=device,
        )
        train_emb = compute_gcn_embeddings(model, train_benign, device=device, tag="embed topogcl train benign")
        center = train_emb.mean(dim=0)
        val_emb = compute_gcn_embeddings(model, val_graphs, device=device, tag="embed topogcl val")
        test_emb = compute_gcn_embeddings(model, test_graphs, device=device, tag="embed topogcl test")
        return (
            torch.norm(val_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
            torch.norm(test_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
        )

    if model_name == "infograph":
        return train_infograph_ids(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=in_dim,
            config=config,
            seed=seed,
            device=device,
        )

    if model_name == "rgcl":
        metrics = run_rgcl_external(config=config, seed=seed)
        return metrics

    raise ValueError(f"Unknown graph model: {model_name}")


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)
    max_graphs = config.max_graphs if config.max_graphs > 0 else None
    benign_limit = config.benign_limit if config.benign_limit > 0 else None
    mal_limit = config.mal_limit if config.mal_limit > 0 else None
    base_graphs = load_npz_graphs(config.graph_dir, max_graphs=max_graphs, max_nodes=config.max_nodes)
    print(f"[OK] dataset={config.dataset} graph_dir={config.graph_dir}", flush=True)
    print(f"[OK] total graphs loaded: {len(base_graphs)}", flush=True)

    summary_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    all_details: Dict[str, object] = {}

    for train_ratio in config.train_ratios:
        ratio_key = f"{train_ratio:.4g}"
        all_details[ratio_key] = {"models": {}, "split": {}, "labels": {}}
        per_model_metrics: Dict[str, List[Dict[str, float]]] = {model: [] for model in config.models}

        for seed in config.seeds:
            print(f"\n[RUN] dataset={config.dataset} train_ratio={train_ratio} seed={seed}", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            graphs = clone_graphs(base_graphs)
            train_graphs, val_graphs, test_graphs = split_for_all_models(
                graphs=graphs,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=config.val_ratio,
                benign_limit=benign_limit,
                mal_limit=mal_limit,
            )
            if config.standardize:
                standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)

            if not all_details[ratio_key]["split"]:
                all_details[ratio_key]["split"] = {
                    "train": len(train_graphs),
                    "val": len(val_graphs),
                    "test": len(test_graphs),
                    "train_ratio": train_ratio,
                    "val_ratio": config.val_ratio,
                }
                all_details[ratio_key]["labels"] = {
                    "train_benign": int(sum(g.label == 0 for g in train_graphs)),
                    "train_malicious": int(sum(g.label == 1 for g in train_graphs)),
                    "val_benign": int(sum(g.label == 0 for g in val_graphs)),
                    "val_malicious": int(sum(g.label == 1 for g in val_graphs)),
                    "test_benign": int(sum(g.label == 0 for g in test_graphs)),
                    "test_malicious": int(sum(g.label == 1 for g in test_graphs)),
                }

            in_dim = train_graphs[0].x.shape[1]
            y_val = np.array([g.label for g in val_graphs], dtype=np.int64)
            y_test = np.array([g.label for g in test_graphs], dtype=np.int64)

            for model_name in config.models:
                print(f"[MODEL] dataset={config.dataset} train_ratio={train_ratio} model={model_name}", flush=True)
                result = model_runner(
                    model_name=model_name,
                    train_graphs=train_graphs,
                    val_graphs=val_graphs,
                    test_graphs=test_graphs,
                    in_dim=in_dim,
                    config=config,
                    seed=seed,
                    device=device,
                )
                if isinstance(result, dict):
                    metrics = result
                else:
                    val_scores, test_scores = result
                    score_rows.extend(
                        make_score_rows(
                            dataset=config.dataset,
                            train_ratio=train_ratio,
                            seed=seed,
                            model_name=model_name,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            y_val=y_val,
                            y_test=y_test,
                        )
                    )
                    threshold = best_threshold_from_validation(y_val, val_scores)
                    metrics = compute_metrics(y_test, test_scores, threshold=threshold)
                    metrics["threshold"] = threshold
                per_model_metrics[model_name].append(metrics)

        for model_name, metrics_list in per_model_metrics.items():
            summary = summarize_metrics(metrics_list)
            summary_rows.append({"dataset": config.dataset, "train_ratio": train_ratio, "model": model_name, **summary})
            all_details[ratio_key]["models"][model_name] = {"runs": metrics_list, "summary": summary}

    results = {
        "dataset": config.dataset,
        "graph_dir": str(config.graph_dir),
        "device": str(device),
        "num_graphs_loaded": len(base_graphs),
        "train_ratios": list(config.train_ratios),
        "runs_by_train_ratio": all_details,
        "training": {
            "models": list(config.models),
            "epochs_graphcl": config.epochs_graphcl,
            "epochs_topogcl": config.epochs_topogcl,
            "epochs_infograph": config.epochs_infograph,
            "lr_graphcl": config.lr_graphcl,
            "lr_topogcl": config.lr_topogcl,
            "lr_infograph": config.lr_infograph,
            "infograph_layers": config.infograph_layers,
            "infograph_dir": str(config.infograph_dir),
            "rgcl_dir": str(config.rgcl_dir),
            "hidden_dim": config.hidden_dim,
            "emb_dim": config.emb_dim,
            "graphcl_layers": config.graphcl_layers,
            "edge_drop": config.edge_drop,
            "feat_mask": config.feat_mask,
            "tau": config.tau,
            "batch_size": config.batch_size,
            "standardized": config.standardize,
            "seeds": list(config.seeds),
            "benign_limit": config.benign_limit,
            "mal_limit": config.mal_limit,
            "max_graphs": config.max_graphs,
            "max_nodes": config.max_nodes,
            "lambda_density": config.lambda_density,
            "ids_safe_augment": config.ids_safe_augment,
            "ids_filtrations": config.ids_filtrations,
            "use_density_head": config.use_density_head,
        },
    }

    config.out_json.parent.mkdir(parents=True, exist_ok=True)
    with config.out_json.open("w") as f:
        json.dump(results, f, indent=2)
    write_summary_csv(config.out_csv, summary_rows, append=True)
    write_scores_csv(config.out_scores_csv, score_rows, append=True)
    print(f"\n[OK] wrote {config.out_json}", flush=True)
    print(f"[OK] wrote {config.out_csv}", flush=True)
    print(f"[OK] wrote {config.out_scores_csv}", flush=True)
    return results


def infer_dataset_name(graph_dir: Path) -> str:
    graph_dir_text = str(graph_dir).lower()
    if "bot" in graph_dir_text:
        return "NF-BoT-IoT"
    if "ton" in graph_dir_text:
        return "NF-ToN-IoT"
    return graph_dir.parent.name or "graph_dataset"


def parse_float_tuple(raw: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train graph-only IDS baselines on existing graph classification .npz files.",
        epilog=(
            "Example prototype: python topids_topogcl_prototype.py --graph-dir /path/to/Graph "
            "--models topogcl --out-json results/prototype.json --out-csv results/prototype_summary.csv\n"
            "Example improved: python topids_topogcl_improved.py --graph-dir /path/to/Graph "
            "--models topogcl --ids-safe-augment --ids-filtrations --use-density-head "
            "--lambda-density 0.1 --out-json results/improved.json --out-csv results/improved_summary.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-json", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_results_25%.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_summary_25%.csv"))
    parser.add_argument(
        "--out-scores-csv",
        type=Path,
        default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_out_scores_25%.csv"),
        help="CSV path for per-run validation/test scores and labels. Defaults beside --out-csv.",
    )
    parser.add_argument("--train-ratios", default="0.25")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--models", default="graphcl,topogcl,infograph,rgcl", help="Comma-separated graph models: gnn,graphsage,graphcl,topogcl,infograph,rgcl")
    parser.add_argument("--epochs-graphcl", type=int, default=10)
    parser.add_argument("--epochs-topogcl", type=int, default=10)
    parser.add_argument("--epochs-infograph", type=int, default=10)
    parser.add_argument("--lr-graphcl", type=float, default=1e-3)
    parser.add_argument("--lr-topogcl", type=float, default=1e-3)
    parser.add_argument("--lr-infograph", type=float, default=1e-3)
    parser.add_argument("--infograph-layers", type=int, default=3)
    parser.add_argument("--infograph-dir", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/InfoGraph/unsupervised"))
    parser.add_argument("--rgcl-dir", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/RGCL/unsupervised_TU"))
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--graphcl-layers", type=int, default=16)
    parser.add_argument("--edge-drop", type=float, default=0.001)
    parser.add_argument("--feat-mask", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50000)
    # IMPROVED vs topids_topogcl_prototype: IDS-specific TopoGCL switches.
    parser.add_argument("--lambda-density", type=float, default=0.1, help="Weight for optional attack-density auxiliary loss.")
    parser.add_argument("--ids-safe-augment", action="store_true", help="Use IDS-safe low-value edge dropping and non-critical feature masking for TopoGCL.")
    parser.add_argument("--ids-filtrations", action="store_true", default=True, help="Use IDS-aware filtration/EPL-style topology features for TopoGCL (default: on).")
    parser.add_argument("--no-ids-filtrations", dest="ids_filtrations", action="store_false", help="Disable IDS-aware filtrations and use prototype topology masking.")
    parser.add_argument("--use-density-head", action="store_true", help="Train an attack-density auxiliary head when attack_density exists in graph .npz files.")
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models = tuple(model.strip().lower() for model in args.models.split(",") if model.strip())
    allowed = {"gnn", "graphsage", "graphcl", "topogcl", "infograph", "rgcl"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; allowed models are {sorted(allowed)}")
    return ExperimentConfig(
        dataset=args.dataset or infer_dataset_name(args.graph_dir),
        graph_dir=args.graph_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        out_scores_csv=args.out_scores_csv or default_scores_csv_path(args.out_csv),
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
        epochs_graphcl=args.epochs_graphcl,
        epochs_topogcl=args.epochs_topogcl,
        epochs_infograph=args.epochs_infograph,
        lr_graphcl=args.lr_graphcl,
        lr_topogcl=args.lr_topogcl,
        lr_infograph=args.lr_infograph,
        infograph_layers=args.infograph_layers,
        infograph_dir=args.infograph_dir,
        rgcl_dir=args.rgcl_dir,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        graphcl_layers=args.graphcl_layers,
        edge_drop=args.edge_drop,
        feat_mask=args.feat_mask,
        tau=args.tau,
        batch_size=args.batch_size,
        benign_limit=args.benign_limit,
        mal_limit=args.mal_limit,
        max_graphs=args.max_graphs,
        max_nodes=args.max_nodes,
        standardize=not args.no_standardize,
        models=models,
        lambda_density=args.lambda_density,
        ids_safe_augment=args.ids_safe_augment,
        ids_filtrations=args.ids_filtrations,
        use_density_head=args.use_density_head,
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
