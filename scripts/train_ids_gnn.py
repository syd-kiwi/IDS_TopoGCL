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
class SupervisedConfig:
    dataset: str
    data_root: Path
    out_json: Path
    out_csv: Path
    out_scores_csv: Path | None
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
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


def predict_supervised_graph_model(classifier: GraphClassifier, eval_graphs: List[GraphWindow], device: torch.device) -> np.ndarray:
    classifier.eval()
    scores: List[float] = []
    with torch.no_grad():
        for graph in eval_graphs:
            logits = classifier(graph, device)
            scores.append(float(torch.softmax(logits, dim=0)[1].item()))
    return np.array(scores, dtype=np.float32)


def run_supervised_model(
    train_graphs: List[GraphWindow],
    val_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    config: SupervisedConfig,
    seed: int,
    device: torch.device,
) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    classifier = train_supervised_graph_model(
        model_name="gnn",
        encoder_factory=lambda: GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.hidden_dim),
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
    return metrics, val_scores, test_scores


def dataset_names(selection: str) -> Tuple[str, ...]:
    return ("streamspot", "grasec", "wget") if selection == "all" else (selection,)


def run_dataset(dataset: str, config: SupervisedConfig, device: torch.device) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
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
        metrics_list: List[Dict[str, float]] = []
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
            metrics, val_scores, test_scores = run_supervised_model(
                train_graphs, val_graphs, test_graphs, train_graphs[0].x.shape[1], config, seed, device
            )
            metrics_list.append(metrics)
            y_val = np.array([g.label for g in val_graphs], dtype=np.int64)
            y_test = np.array([g.label for g in test_graphs], dtype=np.int64)
            score_rows.extend(make_score_rows(dataset, train_ratio, seed, "gnn", y_val, val_scores, y_test, test_scores))
        summary = summarize_metrics(metrics_list)
        summary_rows.append({"dataset": dataset, "train_ratio": train_ratio, "model": "gnn", **summary})
        all_details[ratio_key]["models"]["gnn"] = {"runs": metrics_list, "summary": summary}

    details = {
        "dataset": dataset,
        "data_root": str(config.data_root),
        "num_graphs_loaded": len(base_graphs),
        "runs_by_train_ratio": all_details,
    }
    return details, summary_rows, score_rows


def run_experiment(config: SupervisedConfig) -> Dict[str, object]:
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
            "models": ["gnn"],
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
    with config.out_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_summary_csv(config.out_csv, summary_rows, append=True)
    if config.out_scores_csv is not None:
        write_scores_csv(config.out_scores_csv, score_rows, append=True)
    print(f"\n[OK] wrote {config.out_json}", flush=True)
    print(f"[OK] wrote {config.out_csv}", flush=True)
    if config.out_scores_csv is not None:
        print(f"[OK] wrote {config.out_scores_csv}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised base GNN IDS baseline on StreamSpot and/or GraSec-IoT graphs.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-json", type=Path, default=Path("results/ids_gnn/metrics.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/ids_gnn/summary.csv"))
    parser.add_argument("--out-scores-csv", type=Path, default=Path("results/ids_gnn/scores.csv"))
    parser.add_argument("--train-ratios", default="0.8")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--benign-limit", type=int, default=0)
    parser.add_argument("--mal-limit", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SupervisedConfig:
    return SupervisedConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        out_json=args.out_json,
        out_csv=args.out_csv,
        out_scores_csv=args.out_scores_csv,
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
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
