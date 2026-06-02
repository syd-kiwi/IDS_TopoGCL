#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
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
from sklearn.model_selection import train_test_split


CSV_FIELDNAMES = [
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

MODEL_INFO = {
    "gnn": ("GNN", "basic supervised graph model"),
    "graphsage": ("GraphSAGE", "stronger supervised graph model"),
    "graphcl": ("GraphCL", "standard graph contrastive learning baseline"),
    "topoids": ("TopoIDS", "topology-aware IDS model already present in the pipeline"),
}
MODEL_ALIASES = {
    "gnn": "gnn",
    "graphsage": "graphsage",
    "sage": "graphsage",
    "graphcl": "graphcl",
    "topoids": "topoids",
}


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown graph model: {model_name}")
    return MODEL_ALIASES[key]


def canonical_model_name(model_name: str) -> str:
    return MODEL_INFO[normalize_model_name(model_name)][0]


def model_description(model_name: str) -> str:
    return MODEL_INFO[normalize_model_name(model_name)][1]


@dataclass
class GraphWindow:
    x: torch.Tensor
    edges_undirected: torch.Tensor
    num_nodes: int
    window_start: int
    label: int
    file_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    graph_dir: Path
    out_json: Path
    out_csv: Path
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
    epochs_supervised: int
    epochs_graphcl: int
    epochs_topoids: int
    lr_supervised: float
    lr_graphcl: float
    lr_topoids: float
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


def clone_graphs(graphs: Sequence[GraphWindow]) -> List[GraphWindow]:
    return [
        GraphWindow(
            x=g.x.clone(),
            edges_undirected=g.edges_undirected.clone(),
            num_nodes=g.num_nodes,
            window_start=g.window_start,
            label=g.label,
            file_name=g.file_name,
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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_hat, self.w1(x)))
        h = torch.sparse.mm(a_hat, self.w2(h))
        return h.mean(dim=0)


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


# =========================================================
# Supervised GNN and GraphSAGE training
# =========================================================
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
# TopoIDS-style contrastive graph scoring already in the pipeline.
# =========================================================
def loss_cal(z1: torch.Tensor, z2: torch.Tensor, zt1: torch.Tensor, zt2: torch.Tensor, tau: float) -> torch.Tensor:
    z1_abs = z1.norm(dim=1).clamp_min(1e-12)
    z2_abs = z2.norm(dim=1).clamp_min(1e-12)
    sim = torch.exp(torch.einsum("ik,jk->ij", z1, z2) / torch.einsum("i,j->ij", z1_abs, z2_abs) / tau)
    pos = sim[range(z1.size(0)), range(z1.size(0))]
    loss1 = pos / (sim.sum(dim=1) - pos).clamp_min(1e-12)

    t1_abs = zt1.norm(dim=1).clamp_min(1e-12)
    t2_abs = zt2.norm(dim=1).clamp_min(1e-12)
    simt = torch.exp(torch.einsum("ik,jk->ij", zt1, zt2) / torch.einsum("i,j->ij", t1_abs, t2_abs) / tau)
    post = simt[range(zt1.size(0)), range(zt1.size(0))]
    loss2 = post / (simt.sum(dim=1) - post).clamp_min(1e-12)
    return (-torch.log((loss1 + 0.1 * loss2).clamp_min(1e-12))).mean()


def train_topoids_encoder(
    model: GCN,
    graphs: List[GraphWindow],
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_topoids)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(graphs))

    for epoch in range(config.epochs_topoids):
        permutation = indices[torch.randperm(len(graphs), generator=rng)]
        total_loss = 0.0
        seen = 0
        for start in range(0, len(graphs), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size].tolist()
            if len(batch_indices) < 2:
                continue
            z1_list: List[torch.Tensor] = []
            z2_list: List[torch.Tensor] = []
            zt1_list: List[torch.Tensor] = []
            zt2_list: List[torch.Tensor] = []
            for idx in batch_indices:
                graph = graphs[idx]
                x1, adj1 = augment_graph_view(graph, config.edge_drop, config.feat_mask, rng)
                x2, adj2 = augment_graph_view(graph, config.edge_drop, config.feat_mask, rng)
                x1 = x1.to(device)
                x2 = x2.to(device)
                adj1 = adj1.to(device)
                adj2 = adj2.to(device)
                z1_list.append(model(x1, adj1))
                z2_list.append(model(x2, adj2))
                x1_topo = x1.clone()
                x2_topo = x2.clone()
                if x1_topo.shape[1] > 3:
                    x1_topo[:, 3:] = 0.0
                    x2_topo[:, 3:] = 0.0
                zt1_list.append(model(x1_topo, adj1))
                zt2_list.append(model(x2_topo, adj2))
            loss = loss_cal(torch.stack(z1_list), torch.stack(z2_list), torch.stack(zt1_list), torch.stack(zt2_list), tau=config.tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        print(f"    topoids epoch {epoch + 1}/{config.epochs_topoids} loss={total_loss / max(seen, 1):.6f}", flush=True)


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


def train_topoids_scores(
    train_graphs: List[GraphWindow],
    eval_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    train_benign = [g for g in train_graphs if g.label == 0]
    if len(train_benign) < 2:
        raise RuntimeError("TopoIDS needs at least two benign graphs in the train split.")
    model = GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.emb_dim)
    train_topoids_encoder(model=model, graphs=train_benign, config=config, seed=seed, device=device)
    train_emb = compute_gcn_embeddings(model, train_benign, device=device, tag="embed topoids train benign")
    eval_emb = compute_gcn_embeddings(model, eval_graphs, device=device, tag="embed topoids eval")
    center = train_emb.mean(dim=0)
    return torch.norm(eval_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32)


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
        incoming_keys = {str(row["model"]) for row in rows}
        for existing in read_existing_csv_rows(path):
            existing_key = str(existing.get("model", ""))
            if existing_key not in incoming_keys:
                merged_rows.append(existing)
    merged_rows.extend(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def model_runner(
    model_name: str,
    train_graphs: List[GraphWindow],
    val_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "gnn":
        factory = lambda: GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim)
        classifier = train_supervised_graph_model(
            model_name="gnn",
            encoder_factory=factory,
            train_graphs=train_graphs,
            hidden_dim=config.hidden_dim,
            epochs=config.epochs_supervised,
            lr=config.lr_supervised,
            seed=seed,
            device=device,
        )
        return (
            predict_supervised_graph_model(classifier, val_graphs, device),
            predict_supervised_graph_model(classifier, test_graphs, device),
        )

    if model_name == "graphsage":
        factory = lambda: GraphSAGEEncoder(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim)
        classifier = train_supervised_graph_model(
            model_name="graphsage",
            encoder_factory=factory,
            train_graphs=train_graphs,
            hidden_dim=config.hidden_dim,
            epochs=config.epochs_supervised,
            lr=config.lr_supervised,
            seed=seed,
            device=device,
        )
        return (
            predict_supervised_graph_model(classifier, val_graphs, device),
            predict_supervised_graph_model(classifier, test_graphs, device),
        )

    if model_name == "graphcl":
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
        center = train_emb.mean(dim=0)
        val_emb = compute_graphcl_embeddings(encoder, val_graphs, device=device, tag="embed graphcl val")
        test_emb = compute_graphcl_embeddings(encoder, test_graphs, device=device, tag="embed graphcl test")
        return (
            torch.norm(val_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
            torch.norm(test_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
        )

    if model_name == "topoids":
        train_benign = [g for g in train_graphs if g.label == 0]
        if len(train_benign) < 2:
            raise RuntimeError("TopoIDS needs at least two benign graphs in the train split.")
        model = GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.emb_dim)
        train_topoids_encoder(model=model, graphs=train_benign, config=config, seed=seed, device=device)
        train_emb = compute_gcn_embeddings(model, train_benign, device=device, tag="embed topoids train benign")
        center = train_emb.mean(dim=0)
        val_emb = compute_gcn_embeddings(model, val_graphs, device=device, tag="embed topoids val")
        test_emb = compute_gcn_embeddings(model, test_graphs, device=device, tag="embed topoids test")
        return (
            torch.norm(val_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
            torch.norm(test_emb - center, p=2, dim=1).detach().cpu().numpy().astype(np.float32),
        )

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
                print(f"[MODEL] dataset={config.dataset} train_ratio={train_ratio} model={canonical_model_name(model_name)} - {model_description(model_name)}", flush=True)
                val_scores, test_scores = model_runner(
                    model_name=model_name,
                    train_graphs=train_graphs,
                    val_graphs=val_graphs,
                    test_graphs=test_graphs,
                    in_dim=in_dim,
                    config=config,
                    seed=seed,
                    device=device,
                )
                threshold = best_threshold_from_validation(y_val, val_scores)
                metrics = compute_metrics(y_test, test_scores, threshold=threshold)
                metrics["threshold"] = threshold
                per_model_metrics[model_name].append(metrics)

        for model_name, metrics_list in per_model_metrics.items():
            summary = summarize_metrics(metrics_list)
            summary_rows.append({"model": canonical_model_name(model_name), **summary})
            all_details[ratio_key]["models"][canonical_model_name(model_name)] = {"runs": metrics_list, "summary": summary}

    results = {
        "dataset": config.dataset,
        "graph_dir": str(config.graph_dir),
        "device": str(device),
        "num_graphs_loaded": len(base_graphs),
        "train_ratios": list(config.train_ratios),
        "runs_by_train_ratio": all_details,
        "training": {
            "models": [canonical_model_name(model) for model in config.models],
            "epochs_supervised": config.epochs_supervised,
            "epochs_graphcl": config.epochs_graphcl,
            "epochs_topoids": config.epochs_topoids,
            "lr_supervised": config.lr_supervised,
            "lr_graphcl": config.lr_graphcl,
            "lr_topoids": config.lr_topoids,
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
        },
    }

    config.out_json.parent.mkdir(parents=True, exist_ok=True)
    with config.out_json.open("w") as f:
        json.dump(results, f, indent=2)
    write_summary_csv(config.out_csv, summary_rows, append=True)
    print(f"\n[OK] wrote {config.out_json}", flush=True)
    print(f"[OK] wrote {config.out_csv}", flush=True)
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
    parser = argparse.ArgumentParser(description="Train graph-only IDS baselines on existing graph classification .npz files.")
    parser.add_argument("--graph-dir", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-json", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_results_25%.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_summary_25%.csv"))
    parser.add_argument("--train-ratios", default="0.25")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--models",
        default="gnn,graphsage,graphcl,topoids",
        help="Comma-separated graph models: GNN, GraphSAGE, GraphCL, TopoIDS",
    )
    parser.add_argument("--epochs-supervised", type=int, default=1)
    parser.add_argument("--epochs-graphcl", type=int, default=100)
    parser.add_argument("--epochs-topoids", type=int, default=25)
    parser.add_argument("--lr-supervised", type=float, default=1e-3)
    parser.add_argument("--lr-graphcl", type=float, default=1e-3)
    parser.add_argument("--lr-topoids", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--graphcl-layers", type=int, default=2)
    parser.add_argument("--edge-drop", type=float, default=0.001)
    parser.add_argument("--feat-mask", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models_list: List[str] = []
    for raw_model in args.models.split(","):
        if not raw_model.strip():
            continue
        model = normalize_model_name(raw_model)
        if model not in models_list:
            models_list.append(model)
    models = tuple(models_list)
    return ExperimentConfig(
        dataset=args.dataset or infer_dataset_name(args.graph_dir),
        graph_dir=args.graph_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
        epochs_supervised=args.epochs_supervised,
        epochs_graphcl=args.epochs_graphcl,
        epochs_topoids=args.epochs_topoids,
        lr_supervised=args.lr_supervised,
        lr_graphcl=args.lr_graphcl,
        lr_topoids=args.lr_topoids,
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
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
