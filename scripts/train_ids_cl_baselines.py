#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

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
DATASET_CHOICES = ("streamspot", "grasec", "wget", "all")


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
    lr_graphcl: float
    hidden_dim: int
    graphcl_layers: int
    edge_drop: float
    feat_mask: float
    tau: float
    batch_size: int
    knn_k: int
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


# =========================================================
# GraphCL encoder, trained only on benign graphs and evaluated as
# benign-only anomaly detection via kNN distances.
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
    effective_batch_size = max(2, batch_size)

    for epoch in range(epochs):
        model.train()
        permutation = indices[torch.randperm(len(train_graphs), generator=rng)]
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_graphs), effective_batch_size):
            batch_indices = permutation[start : start + effective_batch_size].tolist()
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


def knn_distance_scores(reference_embeddings: torch.Tensor, query_embeddings: torch.Tensor, k: int) -> np.ndarray:
    if reference_embeddings.numel() == 0:
        raise RuntimeError("GraphCL kNN scoring needs at least one benign training embedding.")
    if query_embeddings.numel() == 0:
        return np.array([], dtype=np.float32)
    effective_k = min(max(1, k), reference_embeddings.shape[0])
    distances = torch.cdist(query_embeddings, reference_embeddings, p=2)
    knn_distances = torch.topk(distances, k=effective_k, largest=False, dim=1).values
    return knn_distances.mean(dim=1).detach().cpu().numpy().astype(np.float32)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = y_true == 0
    if not np.any(negatives):
        return float("nan")
    return float(np.mean(y_pred[negatives] == 1))



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
    if model_name != "graphcl":
        raise ValueError(f"Unknown or unsupported CL model: {model_name}")

    aug_config = get_model_augmentation_config(model_name, config)
    log_augmentation_config(model_name, aug_config)
    train_benign = [graph for graph in train_graphs if graph.label == 0]
    if len(train_benign) < 2:
        raise RuntimeError("GraphCL contrastive training needs at least two benign graphs in the train split.")

    encoder = train_graphcl_encoder(
        train_graphs=train_benign,
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
    train_emb = compute_graphcl_embeddings(encoder, train_benign, device=device, tag="embed graphcl train benign")
    val_emb = compute_graphcl_embeddings(encoder, val_graphs, device=device, tag="embed graphcl val")
    test_emb = compute_graphcl_embeddings(encoder, test_graphs, device=device, tag="embed graphcl test")
    return (
        knn_distance_scores(train_emb, val_emb, k=config.knn_k),
        knn_distance_scores(train_emb, test_emb, k=config.knn_k),
    )



def summarize_graphcl_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    summary = summarize_metrics(metrics_list)
    for name in ("fpr", "threshold", "val_f1"):
        vals = np.array([m[name] for m in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(vals))
        summary[f"{name}_std"] = float(np.nanstd(vals))
    return summary


def dataset_names(selection: str) -> Tuple[str, ...]:
    return ("streamspot", "grasec", "wget") if selection == "all" else (selection,)


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
                val_scores, test_scores = model_runner(model_name, train_graphs, val_graphs, test_graphs, in_dim, config, seed, device)
                threshold = best_threshold_from_validation(y_val, val_scores)
                val_pred = (val_scores >= threshold).astype(int)
                test_pred = (test_scores >= threshold).astype(int)
                metrics = compute_metrics(y_test, test_scores, threshold=threshold)
                metrics["fpr"] = false_positive_rate(y_test, test_pred)
                metrics["threshold"] = float(threshold)
                metrics["val_f1"] = float(f1_score(y_val, val_pred, zero_division=0))
                per_model_metrics[model_name].append(metrics)
                score_rows.extend(make_score_rows(dataset, train_ratio, seed, model_name, y_val, val_scores, y_test, test_scores))
        for model_name, metrics_list in per_model_metrics.items():
            summary = summarize_graphcl_metrics(metrics_list)
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
            "lr_graphcl": config.lr_graphcl,
            "hidden_dim": config.hidden_dim,
            "graphcl_layers": config.graphcl_layers,
            "edge_drop": config.edge_drop,
            "feat_mask": config.feat_mask,
            "tau": config.tau,
            "batch_size": config.batch_size,
            "knn_k": config.knn_k,
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
    parser.add_argument("--models", default="graphcl", help="Comma-separated CL baseline models; only graphcl is supported.")
    parser.add_argument("--epochs", type=int, default=None, help="GraphCL epoch count when --epochs-graphcl is not supplied.")
    parser.add_argument("--epochs-graphcl", type=int, default=None)
    parser.add_argument("--lr-graphcl", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--graphcl-layers", type=int, default=3)
    parser.add_argument("--edge-drop", type=float, default=0.001)
    parser.add_argument("--feat-mask", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=5, help="Number of benign train embeddings used for GraphCL kNN anomaly scores.")
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models = tuple(model.strip().lower() for model in args.models.split(",") if model.strip())
    allowed = {"graphcl"}
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
        lr_graphcl=args.lr_graphcl,
        hidden_dim=args.hidden_dim,
        graphcl_layers=args.graphcl_layers,
        edge_drop=args.edge_drop,
        feat_mask=args.feat_mask,
        tau=args.tau,
        batch_size=args.batch_size,
        knn_k=args.knn_k,
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
