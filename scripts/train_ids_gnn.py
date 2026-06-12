#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from train_ids_binary import (
    CSV_FIELDNAMES,
    GraphWindow,
    best_threshold_from_validation,
    build_sparse_a_hat_from_undirected,
    clone_graphs,
    compute_metrics,
    infer_dataset_name,
    load_npz_graphs,
    parse_float_tuple,
    parse_int_tuple,
    split_for_all_models,
    standardize_from_train,
    summarize_metrics,
    write_summary_csv,
)


@dataclass(frozen=True)
class SupervisedConfig:
    dataset: str
    graph_dir: Path
    out_json: Path
    out_csv: Path
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
    models: Tuple[str, ...]
    epochs: int
    lr: float
    hidden_dim: int
    benign_limit: int
    mal_limit: int
    max_graphs: int
    max_nodes: int
    standardize: bool


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(adj, self.w1(x)))
        h = torch.sparse.mm(adj, self.w2(h))
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


def run_supervised_model(
    model_name: str,
    train_graphs: List[GraphWindow],
    val_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    config: SupervisedConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    if model_name != "gnn":
        raise ValueError(
            f"Unknown supervised model: {model_name}. train_ids_gnn.py intentionally runs only the base GNN baseline."
        )
    factory = lambda: GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim)

    classifier = train_supervised_graph_model(
        model_name=model_name,
        encoder_factory=factory,
        train_graphs=train_graphs,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        seed=seed,
        device=device,
    )
    y_val = np.array([g.label for g in val_graphs], dtype=np.int64)
    y_test = np.array([g.label for g in test_graphs], dtype=np.int64)
    val_scores = predict_supervised_graph_model(classifier, val_graphs, device)
    test_scores = predict_supervised_graph_model(classifier, test_graphs, device)
    threshold = best_threshold_from_validation(y_val, val_scores)
    metrics = compute_metrics(y_test, test_scores, threshold=threshold)
    metrics["threshold"] = threshold
    return metrics


def run_experiment(config: SupervisedConfig) -> Dict[str, object]:
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
            for model_name in config.models:
                print(f"[MODEL] dataset={config.dataset} train_ratio={train_ratio} model={model_name}", flush=True)
                per_model_metrics[model_name].append(
                    run_supervised_model(model_name, train_graphs, val_graphs, test_graphs, in_dim, config, seed, device)
                )

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
            "epochs": config.epochs,
            "lr": config.lr,
            "hidden_dim": config.hidden_dim,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised base GNN IDS baseline on .npz graph files.")
    parser.add_argument("--graph-dir", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-json", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_supervised_results_25%.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_supervised_summary_25%.csv"))
    parser.add_argument("--train-ratios", default="0.25")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default="gnn", help="Supervised graph model to run. Only gnn is supported in this script.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SupervisedConfig:
    models = tuple(model.strip().lower() for model in args.models.split(",") if model.strip())
    allowed = {"gnn"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; allowed models are {sorted(allowed)}")
    return SupervisedConfig(
        dataset=args.dataset or infer_dataset_name(args.graph_dir),
        graph_dir=args.graph_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
        models=models,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        benign_limit=args.benign_limit,
        mal_limit=args.mal_limit,
        max_graphs=args.max_graphs,
        max_nodes=args.max_nodes,
        standardize=not args.no_standardize,
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
