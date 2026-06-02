#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# =========================================================
# Hardcoded configuration
# Toggle the commented NF-BoT-IoT paths/ratios when needed.
# =========================================================
# GRAPH_DIR = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph")
GRAPH_DIR = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph")

# OUT_JSON = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_results_05%.json")
# OUT_CSV = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_bot_iot/bot_summary_05%.csv")

OUT_JSON = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_results_15%.json")
OUT_CSV = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/nf_ton_iot/ton_summary_15%.csv")

EPOCHS_GRAPHCL = 25
LR_GRAPHCL = 1e-3
HIDDEN_DIM = 16
NUM_LAYERS = 2
EDGE_DROP = 0.001
FEAT_MASK = 0.05
TAU = 0.2
BATCH_SIZE = 8
GRAPHCL_THRESHOLD = 0.50

# TRAIN_RATIO = 0.60
# VAL_RATIO = 0.15
TRAIN_RATIO = 0.15
VAL_RATIO = 0.15

# Set these to 0 to use all available graphs.
BENIGN_LIMIT = 0
MAL_LIMIT = 0
MAX_GRAPHS = 0

MAX_NODES = 50000
SEEDS = [42]
#SEEDS = [42, 43, 44, 45, 46]
STANDARDIZE = True
MODEL_NAME = "graphcl"

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


# =========================================================
# Data loading/splitting helpers for existing .npz graph files
# =========================================================
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


def standardize_from_train(train_graphs: List[GraphWindow], all_graphs: List[GraphWindow]) -> None:
    xs = torch.cat([g.x for g in train_graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)

    for graph in all_graphs:
        graph.x = torch.nan_to_num((graph.x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def split_for_graphcl(
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

    stratify_y = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 3 else None
    train_idx, temp_idx, _y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=stratify_y,
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_size)
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 and np.min(np.bincount(y_temp)) >= 2 else None
    val_idx, test_idx, _y_val, _y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=stratify_temp,
    )

    return [selected[i] for i in train_idx], [selected[i] for i in val_idx], [selected[i] for i in test_idx]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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
    for name in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
        values = np.array([metrics[name] for metrics in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(values))
        summary[f"{name}_std"] = float(np.nanstd(values))
    return summary


# =========================================================
# Small self-contained GraphCL implementation inspired by
# https://github.com/PyGCL/PyGCL/blob/main/examples/GraphCL.py
# =========================================================
def clone_graphs(graphs: List[GraphWindow]) -> List[GraphWindow]:
    cloned: List[GraphWindow] = []
    for g in graphs:
        cloned.append(
            GraphWindow(
                x=g.x.clone(),
                edges_undirected=g.edges_undirected.clone(),
                num_nodes=g.num_nodes,
                window_start=g.window_start,
                label=g.label,
                file_name=g.file_name,
            )
        )
    return cloned


def build_sparse_adj_from_undirected(
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

    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def augment_graphcl_view(
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
        feature_mask = (torch.rand(x.shape, generator=rng) > feat_mask).float()
        x = x * feature_mask

    adj = build_sparse_adj_from_undirected(graph.num_nodes, edges, add_self_loops=True)
    return x, adj


class GraphCLGINEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            torch.nn.Linear(project_dim, project_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(project_dim, project_dim),
        )
        self.output_dim = project_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        graph_embeddings = []

        for mlp, batch_norm in zip(self.mlps, self.batch_norms):
            h = torch.sparse.mm(adj, h)
            h = mlp(h)
            h = torch.relu(batch_norm(h))
            graph_embeddings.append(h.mean(dim=0))

        return torch.cat(graph_embeddings, dim=0)


def graphcl_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss_12 = torch.nn.functional.cross_entropy(logits, labels)
    loss_21 = torch.nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


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

            z1_list = []
            z2_list = []
            for graph_idx in batch_indices:
                graph = train_graphs[graph_idx]
                x1, adj1 = augment_graphcl_view(graph, edge_drop, feat_mask, rng, identity=True)
                x2, adj2 = augment_graphcl_view(graph, edge_drop, feat_mask, rng, identity=False)

                z1_list.append(model(x1.to(device), adj1.to(device)))
                z2_list.append(model(x2.to(device), adj2.to(device)))

            z1 = model.project(torch.stack(z1_list, dim=0))
            z2 = model.project(torch.stack(z2_list, dim=0))
            loss = graphcl_loss(z1, z2, tau=tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)

        print(f"graphcl epoch {epoch + 1}/{epochs} loss={total_loss / max(seen, 1):.6f}", flush=True)

    return model


def compute_graphcl_embeddings(
    model: GraphCLGINEncoder,
    graphs: List[GraphWindow],
    device: torch.device,
    tag: str,
) -> np.ndarray:
    embeddings: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for idx, graph in enumerate(graphs):
            x, adj = augment_graphcl_view(graph, edge_drop=0.0, feat_mask=0.0, rng=torch.Generator(), identity=True)
            embeddings.append(model(x.to(device), adj.to(device)).detach().cpu())

            if (idx + 1) % max(1, len(graphs) // 10) == 0:
                print(f"{tag}: {idx + 1}/{len(graphs)}", flush=True)

    return torch.stack(embeddings, dim=0).numpy()


def train_graphcl_scores(
    train_graphs: List[GraphWindow],
    test_graphs: List[GraphWindow],
    in_dim: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    encoder = train_graphcl_encoder(
        train_graphs=train_graphs,
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        epochs=EPOCHS_GRAPHCL,
        lr=LR_GRAPHCL,
        edge_drop=EDGE_DROP,
        feat_mask=FEAT_MASK,
        tau=TAU,
        batch_size=BATCH_SIZE,
        seed=seed,
        device=device,
    )
    train_benign = [graph for graph in train_graphs if graph.label == 0]
    if not train_benign:
        raise RuntimeError("GraphCL distance scoring needs at least one benign graph in the train split.")

    x_train = compute_graphcl_embeddings(encoder, train_benign, device=device, tag="embed graphcl train benign")
    x_test = compute_graphcl_embeddings(encoder, test_graphs, device=device, tag="embed graphcl test")

    center = x_train.mean(axis=0, keepdims=True)
    return np.linalg.norm(x_test - center, axis=1).astype(np.float32)


# =========================================================
# Result writers: upsert GraphCL without dropping existing rows.
# =========================================================
def read_existing_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def upsert_csv_summary(path: Path, row: Dict[str, object]) -> None:
    rows = read_existing_csv_rows(path)
    rows = [existing for existing in rows if existing.get("model") != row["model"]]
    rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for existing in rows:
            writer.writerow({field: existing.get(field, "") for field in CSV_FIELDNAMES})


def update_json_results(
    path: Path,
    graph_dir: Path,
    device: torch.device,
    num_graphs_loaded: int,
    split_info: Dict[str, object],
    labels_info: Dict[str, object],
    run_metrics: List[Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    if path.exists():
        with path.open() as f:
            results = json.load(f)
    else:
        results = {}

    results.setdefault("dataset", infer_dataset_name(graph_dir))
    results["graph_dir"] = str(graph_dir)
    results["device"] = str(device)
    results["num_graphs_loaded"] = num_graphs_loaded
    results["split"] = split_info
    results["labels"] = labels_info
    training = results.setdefault("training", {})
    training.update(
        {
            "epochs_graphcl": EPOCHS_GRAPHCL,
            "lr_graphcl": LR_GRAPHCL,
            "graphcl_hidden_dim": HIDDEN_DIM,
            "graphcl_num_layers": NUM_LAYERS,
            "graphcl_edge_drop": EDGE_DROP,
            "graphcl_feat_mask": FEAT_MASK,
            "graphcl_tau": TAU,
            "graphcl_batch_size": BATCH_SIZE,
            "graphcl_threshold": GRAPHCL_THRESHOLD,
            "standardized": STANDARDIZE,
            "seeds": SEEDS,
            "benign_limit": BENIGN_LIMIT,
            "mal_limit": MAL_LIMIT,
            "max_graphs": MAX_GRAPHS,
            "max_nodes": MAX_NODES,
        }
    )
    results.setdefault("models", {})[MODEL_NAME] = {"runs": run_metrics, "summary": summary}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)


def infer_dataset_name(graph_dir: Path) -> str:
    graph_dir_text = str(graph_dir).lower()
    if "bot" in graph_dir_text:
        return "NF-BoT-IoT"
    if "ton" in graph_dir_text:
        return "NF-ToN-IoT"
    return graph_dir.parent.name


def main() -> None:
    max_graphs: Optional[int] = MAX_GRAPHS if MAX_GRAPHS > 0 else None
    benign_limit: Optional[int] = BENIGN_LIMIT if BENIGN_LIMIT > 0 else None
    mal_limit: Optional[int] = MAL_LIMIT if MAL_LIMIT > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}", flush=True)

    base_graphs = load_npz_graphs(graph_dir=GRAPH_DIR, max_graphs=max_graphs, max_nodes=MAX_NODES)
    print(f"[OK] graph dir: {GRAPH_DIR}", flush=True)
    print(f"[OK] total graphs loaded: {len(base_graphs)}", flush=True)

    run_metrics: List[Dict[str, float]] = []
    split_info: Dict[str, object] = {}
    labels_info: Dict[str, object] = {}

    for seed in SEEDS:
        print(f"\n[RUN] seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        graphs = clone_graphs(base_graphs)
        train_graphs, val_graphs, test_graphs = split_for_graphcl(
            graphs=graphs,
            seed=seed,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            benign_limit=benign_limit,
            mal_limit=mal_limit,
        )

        if STANDARDIZE:
            standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)

        if not split_info:
            split_info = {
                "train": len(train_graphs),
                "val": len(val_graphs),
                "test": len(test_graphs),
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
            }
            labels_info = {
                "train_benign": int(sum(g.label == 0 for g in train_graphs)),
                "train_malicious": int(sum(g.label == 1 for g in train_graphs)),
                "test_benign": int(sum(g.label == 0 for g in test_graphs)),
                "test_malicious": int(sum(g.label == 1 for g in test_graphs)),
            }

        y_test = np.array([g.label for g in test_graphs], dtype=np.int64)
        scores = train_graphcl_scores(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            in_dim=train_graphs[0].x.shape[1],
            seed=seed,
            device=device,
        )
        run_metrics.append(compute_metrics(y_test, scores, threshold=GRAPHCL_THRESHOLD))

    summary = summarize_metrics(run_metrics)
    summary_row = {"model": MODEL_NAME, **summary}

    upsert_csv_summary(OUT_CSV, summary_row)
    update_json_results(
        path=OUT_JSON,
        graph_dir=GRAPH_DIR,
        device=device,
        num_graphs_loaded=len(base_graphs),
        split_info=split_info,
        labels_info=labels_info,
        run_metrics=run_metrics,
        summary=summary,
    )

    print(f"\n[OK] upserted {MODEL_NAME} into {OUT_CSV}", flush=True)
    print(f"[OK] updated {OUT_JSON}", flush=True)
    print("\nGraphCL summary:")
    print(summary_row, flush=True)


if __name__ == "__main__":
    main()
