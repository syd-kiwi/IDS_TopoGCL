#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from ids_graph_data import (
    GraphWindow,
    best_threshold_from_validation,
    build_sparse_a_hat_from_undirected,
    clone_graphs,
    compute_metrics,
    load_dataset_graphs,
    make_score_rows,
    parse_float_tuple,
    parse_int_tuple,
    split_for_all_models,
    standardize_from_train,
    summarize_metrics,
    write_scores_csv,
    write_summary_csv,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATASET_CHOICES = ("streamspot", "grasec", "all")


@dataclass(frozen=True)
class ModelAugmentationConfig:
    edge_drop: float
    feat_mask: float
    tau: float
    batch_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    data_root: Path
    out_json: Path
    out_csv: Path
    out_scores_csv: Path
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
    epochs_graphcl: int
    epochs_infograph: int
    lr_graphcl: float
    lr_infograph: float
    infograph_layers: int
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


def get_model_augmentation_config(model_name: str, config: ExperimentConfig) -> ModelAugmentationConfig:
    if model_name != "graphcl":
        raise ValueError(f"No augmentation config defined for model: {model_name}")
    return ModelAugmentationConfig(
        edge_drop=config.edge_drop,
        feat_mask=config.feat_mask,
        tau=config.tau,
        batch_size=config.batch_size,
    )


def format_config_float(value: float) -> str:
    return f"{value:.3f}" if 0 < abs(value) < 0.01 else f"{value:.2f}"


def log_augmentation_config(model_name: str, aug_config: ModelAugmentationConfig) -> None:
    print(
        f"[CONFIG] model={model_name} edge_drop={format_config_float(aug_config.edge_drop)} "
        f"feat_mask={format_config_float(aug_config.feat_mask)} "
        f"tau={format_config_float(aug_config.tau)} batch_size={aug_config.batch_size}",
        flush=True,
    )


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
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        return float("nan")


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
    if model_name == "infograph":
        return train_infograph_ids(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=in_dim,
            config=config,
            seed=seed,
            device=device,
        )
    raise ValueError(f"Unknown or unsupported CL model: {model_name}")


def dataset_names(selection: str) -> Tuple[str, ...]:
    return ("streamspot", "grasec") if selection == "all" else (selection,)


def run_dataset(dataset: str, config: ExperimentConfig, device: torch.device) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    load_seed = config.seeds[0] if config.seeds else 42
    max_graphs = config.max_graphs if config.max_graphs > 0 else None
    benign_limit = config.benign_limit if config.benign_limit > 0 else None
    mal_limit = config.mal_limit if config.mal_limit > 0 else None
    base_graphs = load_dataset_graphs(dataset, data_root=config.data_root, seed=load_seed, max_nodes=config.max_nodes)
    if max_graphs is not None:
        base_graphs = base_graphs[:max_graphs]
    print(f"[OK] dataset={dataset} data_root={config.data_root}", flush=True)
    print(f"[OK] total graphs loaded: {len(base_graphs)}", flush=True)

    summary_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    all_details: Dict[str, object] = {}
    for train_ratio in config.train_ratios:
        ratio_key = f"{train_ratio:.4g}"
        all_details[ratio_key] = {"models": {}, "split": {}, "labels": {}}
        per_model_metrics: Dict[str, List[Dict[str, float]]] = {model: [] for model in config.models}
        for seed in config.seeds:
            print(f"\n[RUN] dataset={dataset} train_ratio={train_ratio} seed={seed}", flush=True)
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
                print(f"[MODEL] dataset={dataset} train_ratio={train_ratio} model={model_name}", flush=True)
                output = model_runner(model_name, train_graphs, val_graphs, test_graphs, in_dim, config, seed, device)
                if isinstance(output, tuple):
                    val_scores, test_scores = output
                    threshold = best_threshold_from_validation(y_val, val_scores)
                    metrics = compute_metrics(y_test, test_scores, threshold=threshold)
                    metrics["threshold"] = threshold
                    per_model_metrics[model_name].append(metrics)
                    score_rows.extend(make_score_rows(dataset, train_ratio, seed, model_name, y_val, val_scores, y_test, test_scores))
                else:
                    per_model_metrics[model_name].append(output)
        for model_name, metrics_list in per_model_metrics.items():
            summary = summarize_metrics(metrics_list)
            summary_rows.append({"dataset": dataset, "train_ratio": train_ratio, "model": model_name, **summary})
            all_details[ratio_key]["models"][model_name] = {"runs": metrics_list, "summary": summary}
    return {"dataset": dataset, "data_root": str(config.data_root), "num_graphs_loaded": len(base_graphs), "runs_by_train_ratio": all_details}, summary_rows, score_rows


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)
    details_by_dataset: Dict[str, object] = {}
    summary_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    for dataset in dataset_names(config.dataset):
        details, ds_summary_rows, ds_score_rows = run_dataset(dataset, config, device)
        details_by_dataset[dataset] = details
        summary_rows.extend(ds_summary_rows)
        score_rows.extend(ds_score_rows)
    results = {
        "dataset": config.dataset,
        "data_root": str(config.data_root),
        "device": str(device),
        "train_ratios": list(config.train_ratios),
        "datasets": details_by_dataset,
        "training": {
            "models": list(config.models),
            "epochs_graphcl": config.epochs_graphcl,
            "epochs_infograph": config.epochs_infograph,
            "lr_graphcl": config.lr_graphcl,
            "lr_infograph": config.lr_infograph,
            "infograph_layers": config.infograph_layers,
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
    with config.out_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_summary_csv(config.out_csv, summary_rows, append=True)
    write_scores_csv(config.out_scores_csv, score_rows, append=True)
    print(f"\n[OK] wrote {config.out_json}", flush=True)
    print(f"[OK] wrote {config.out_csv}", flush=True)
    print(f"[OK] wrote {config.out_scores_csv}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive IDS baselines on StreamSpot and/or GraSec-IoT graphs.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-json", type=Path, default=Path("results/ids_cl_baselines/metrics.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/ids_cl_baselines/summary.csv"))
    parser.add_argument("--out-scores-csv", type=Path, default=Path("results/ids_cl_baselines/scores.csv"))
    parser.add_argument("--train-ratios", default="0.8")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--models", default="graphcl,infograph", help="Comma-separated CL baseline models: graphcl,infograph")
    parser.add_argument("--epochs", type=int, default=None, help="Shared epoch count for GraphCL and InfoGraph when model-specific values are not supplied.")
    parser.add_argument("--epochs-graphcl", type=int, default=None)
    parser.add_argument("--epochs-infograph", type=int, default=None)
    parser.add_argument("--lr-graphcl", type=float, default=1e-3)
    parser.add_argument("--lr-infograph", type=float, default=1e-3)
    parser.add_argument("--infograph-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--graphcl-layers", type=int, default=3)
    parser.add_argument("--edge-drop", type=float, default=0.001)
    parser.add_argument("--feat-mask", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models = tuple(model.strip().lower() for model in args.models.split(",") if model.strip())
    allowed = {"graphcl", "infograph"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"Unknown or unsupported CL models {sorted(unknown)}; allowed models are {sorted(allowed)}")
    shared_epochs = 10 if args.epochs is None else args.epochs
    return ExperimentConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        out_json=args.out_json,
        out_csv=args.out_csv,
        out_scores_csv=args.out_scores_csv,
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
        epochs_graphcl=shared_epochs if args.epochs_graphcl is None else args.epochs_graphcl,
        epochs_infograph=shared_epochs if args.epochs_infograph is None else args.epochs_infograph,
        lr_graphcl=args.lr_graphcl,
        lr_infograph=args.lr_infograph,
        infograph_layers=args.infograph_layers,
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
