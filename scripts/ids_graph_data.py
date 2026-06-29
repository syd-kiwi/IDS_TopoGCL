from __future__ import annotations

import csv
import glob
import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

STREAMSPOT_NODE_TYPES = list("abcdefgh")
STREAMSPOT_EDGE_TYPES = list("ijklmntquvwyzACDEG")
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
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
    "fpr_mean",
    "fpr_std",
    "threshold_mean",
    "threshold_std",
    "val_f1_mean",
    "val_f1_std",
]
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


def parse_float_tuple(raw: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


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


def compute_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
        np.add.at(deg, edge_index[1], 1.0)
    return deg


def subsample_graph(
    x: np.ndarray, edge_index: np.ndarray, max_nodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if max_nodes <= 0 or n <= max_nodes:
        return x.astype(np.float32), edge_index.astype(np.int64)
    deg = compute_degree(n, edge_index)
    top = np.argsort(-deg)[: max_nodes // 2]
    rest = np.setdiff1d(np.arange(n), top)
    extra = rng.choice(rest, size=max_nodes - len(top), replace=False) if len(rest) else np.array([], dtype=np.int64)
    keep = np.sort(np.concatenate([top, extra]))
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    x2 = x[keep]
    if edge_index.size:
        src, dst = edge_index
        mask = (remap[src] >= 0) & (remap[dst] >= 0)
        ei = np.stack([remap[src[mask]], remap[dst[mask]]]) if np.any(mask) else np.zeros((2, 0), dtype=np.int64)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return x2.astype(np.float32), ei.astype(np.int64)


def _make_graph(x: np.ndarray, edge_index: np.ndarray, label: int, name: str) -> GraphWindow:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    num_nodes = int(x.shape[0])
    return GraphWindow(
        x=torch.tensor(x, dtype=torch.float32),
        edges_undirected=edge_index_to_undirected(edge_index, num_nodes),
        num_nodes=num_nodes,
        window_start=0,
        label=int(label),
        file_name=name,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _label_to_int(value: Any) -> int:
    arr = _to_numpy(value).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty Wget graph label")
    if arr.size > 1 and np.issubdtype(arr.dtype, np.number):
        return int(np.argmax(arr))
    raw = arr[0].item() if hasattr(arr[0], "item") else arr[0]
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"benign", "normal", "clean", "0"}:
            return 0
        if text in {"malicious", "attack", "anomaly", "1"}:
            return 1
    return int(raw)


def _looks_like_graph(value: Any) -> bool:
    return hasattr(value, "edges") and (hasattr(value, "num_nodes") or hasattr(value, "number_of_nodes"))


def _graph_num_nodes(graph: Any) -> int:
    if hasattr(graph, "num_nodes"):
        return int(graph.num_nodes())
    return int(graph.number_of_nodes())


def _graph_edge_index(graph: Any) -> np.ndarray:
    src, dst = graph.edges()
    src_np = _to_numpy(src).astype(np.int64).reshape(-1)
    dst_np = _to_numpy(dst).astype(np.int64).reshape(-1)
    if src_np.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([src_np, dst_np]).astype(np.int64)


def _one_hot(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    _, inverse = np.unique(flat, return_inverse=True)
    out = np.zeros((flat.shape[0], int(inverse.max()) + 1), dtype=np.float32)
    out[np.arange(flat.shape[0]), inverse] = 1.0
    return out


def _wget_node_features(graph: Any, num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    ndata = getattr(graph, "ndata", {})
    x = None
    for key in ("node_type", "ntype", "type", "_TYPE", "label", "node_label"):
        if key in ndata:
            raw = _to_numpy(ndata[key])
            if raw.shape[0] == num_nodes:
                x = raw.astype(np.float32) if raw.ndim == 2 and raw.shape[1] > 1 else _one_hot(raw)
                break
    if x is None:
        for key in ("feat", "features", "x", "attr"):
            if key in ndata:
                raw = _to_numpy(ndata[key])
                if raw.shape[0] == num_nodes and np.issubdtype(raw.dtype, np.number):
                    x = raw.reshape(num_nodes, -1).astype(np.float32)
                    break
    if x is None:
        x = np.ones((num_nodes, 1), dtype=np.float32)
    deg = np.log1p(compute_degree(num_nodes, edge_index)).reshape(-1, 1)
    return np.concatenate([x, deg.astype(np.float32)], axis=1)


def _split_wget_record(record: Any) -> tuple[Any, Any]:
    if isinstance(record, dict):
        graph = next((record[k] for k in ("graph", "g", "dgl_graph") if k in record), None)
        label = next((record[k] for k in ("label", "y", "target") if k in record), None)
        if graph is not None and label is not None:
            return graph, label
    if isinstance(record, (tuple, list)):
        graph = next((item for item in record if _looks_like_graph(item)), None)
        label = next((item for item in record if item is not graph), None)
        if graph is not None and label is not None:
            return graph, label
    graph = record if _looks_like_graph(record) else getattr(record, "graph", None)
    label = next((getattr(record, name) for name in ("label", "y", "target") if hasattr(record, name)), None)
    if graph is not None and label is not None:
        return graph, label
    raise ValueError(f"Could not identify graph and label in Wget record of type {type(record).__name__}")


def _iter_wget_records(raw: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(raw, dict):
        graphs = next((raw[k] for k in ("graphs", "graph", "data") if k in raw), None)
        labels = next((raw[k] for k in ("labels", "label", "ys", "y", "targets") if k in raw), None)
        if graphs is not None and labels is not None and not _looks_like_graph(graphs):
            for graph, label in zip(graphs, labels):
                yield graph, label
            return
    graphs_attr = next((getattr(raw, name) for name in ("graphs", "graph_lists") if hasattr(raw, name)), None)
    labels_attr = next((getattr(raw, name) for name in ("labels", "targets") if hasattr(raw, name)), None)
    if graphs_attr is not None and labels_attr is not None:
        for graph, label in zip(graphs_attr, labels_attr):
            yield graph, label
        return
    if hasattr(raw, "__len__") and hasattr(raw, "__getitem__") and not _looks_like_graph(raw):
        for idx in range(len(raw)):
            yield _split_wget_record(raw[idx])
        return
    yield _split_wget_record(raw)


def load_wget_graphs(data_root: Path, seed: int, max_nodes: int) -> List[GraphWindow]:
    pkl = data_root / "wget" / "graphs.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Missing Wget MAGIC graph dataset: {pkl}. Download MAGIC data/wget/graphs.zip "
            "and unzip it into data/wget/ so data/wget/graphs.pkl exists."
        )
    rng = np.random.default_rng(seed)
    with pkl.open("rb") as handle:
        raw = pickle.load(handle)
    graphs: List[GraphWindow] = []
    for idx, (graph, label) in enumerate(_iter_wget_records(raw)):
        num_nodes = _graph_num_nodes(graph)
        if num_nodes == 0:
            continue
        edge_index = _graph_edge_index(graph)
        x = _wget_node_features(graph, num_nodes, edge_index)
        x, edge_index = subsample_graph(x, edge_index, max_nodes, rng)
        graphs.append(_make_graph(x, edge_index, _label_to_int(label), f"wget_{idx}"))
    return graphs


def load_streamspot_graphs(data_root: Path, seed: int, max_nodes: int) -> List[GraphWindow]:
    tsv = data_root / "streamspot" / "all.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Missing StreamSpot TSV: {tsv}")
    rng = np.random.default_rng(seed)
    graphs: List[GraphWindow] = []
    n_node_types = len(STREAMSPOT_NODE_TYPES)
    current = None
    node_type: Dict[str, int] = {}
    node_id: Dict[str, int] = {}
    edges: List[tuple[int, int]] = []

    def finalize(gid: Optional[int]) -> None:
        if gid is None or not node_id:
            return
        n = len(node_id)
        x = np.zeros((n, n_node_types + 1), dtype=np.float32)
        for nid, idx in node_id.items():
            x[idx, node_type[nid]] = 1.0
        ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        x[:, -1] = np.log1p(compute_degree(n, ei))
        x2, ei2 = subsample_graph(x, ei, max_nodes, rng)
        label = 1 if 300 <= gid <= 399 else 0
        graphs.append(_make_graph(x2, ei2, label, f"streamspot_{gid}"))

    with tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            src, st, dst, dt, et, gid_s = line.rstrip("\n").split("\t")
            gid = int(gid_s)
            if current is None:
                current = gid
            if gid != current:
                finalize(current)
                node_type, node_id, edges = {}, {}, []
                current = gid
            if st not in STREAMSPOT_NODE_TYPES or dt not in STREAMSPOT_NODE_TYPES or et not in STREAMSPOT_EDGE_TYPES:
                continue
            if src not in node_id:
                node_id[src] = len(node_id)
                node_type[src] = STREAMSPOT_NODE_TYPES.index(st)
            if dst not in node_id:
                node_id[dst] = len(node_id)
                node_type[dst] = STREAMSPOT_NODE_TYPES.index(dt)
            edges.append((node_id[src], node_id[dst]))
        finalize(current)
    return graphs


def grasec_label(snap: dict) -> int:
    for nd in snap.get("nodes", []):
        if nd.get("entity") == "connection" and "Label" in nd:
            lbl = nd["Label"]
            idx = lbl.index(1.0) if 1.0 in lbl else 6
            if idx != 6:
                return 1
    return 0


def _load_grasec_files(files: Iterable[Path], seed: int, max_nodes: int) -> List[GraphWindow]:
    rng = np.random.default_rng(seed)
    graphs: List[GraphWindow] = []
    ip_dim, con_dim = 8, 8
    feat_dim = 2 + ip_dim + con_dim + 1
    for fp in sorted(files):
        with fp.open("r", encoding="utf-8") as handle:
            snapshots = json.load(handle)
        for i, snap in enumerate(snapshots):
            nodes = snap["nodes"]
            n = len(nodes)
            if n == 0:
                continue
            id_to_idx = {nd["id"]: j for j, nd in enumerate(nodes)}
            x = np.zeros((n, feat_dim), dtype=np.float32)
            for j, nd in enumerate(nodes):
                ent = nd.get("entity", "")
                if ent == "ip":
                    x[j, 0] = 1.0
                    feats = nd.get("ip_feats", [])
                    x[j, 2 : 2 + min(ip_dim, len(feats))] = feats[:ip_dim]
                else:
                    x[j, 1] = 1.0
                    feats = nd.get("conect_feats", [])
                    off = 2 + ip_dim
                    x[j, off : off + min(con_dim, len(feats))] = feats[:con_dim]
            edges = []
            for lk in snap.get("links", []):
                s, t = id_to_idx.get(lk["source"]), id_to_idx.get(lk["target"])
                if s is not None and t is not None:
                    edges.append((s, t))
            ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
            x[:, -1] = np.log1p(compute_degree(n, ei))
            x2, ei2 = subsample_graph(x, ei, max_nodes, rng)
            graphs.append(_make_graph(x2, ei2, grasec_label(snap), f"grasec_{fp.stem}_{i}"))
    return graphs


def load_grasec_graphs(data_root: Path, seed: int, max_nodes: int) -> List[GraphWindow]:
    base = data_root / "grasec-iot" / "graph_json" / "Graph_JSON"
    if not base.exists():
        raise FileNotFoundError(f"Missing GraSec-IoT Graph_JSON directory: {base}")
    files = [Path(p) for p in glob.glob(str(base / "*" / "data_*.json"))]
    if not files:
        files = sorted(base.glob("data_*.json"))
    if not files:
        raise FileNotFoundError(f"No GraSec-IoT data_*.json files found below {base}")
    return _load_grasec_files(files, seed=seed, max_nodes=max_nodes)


def load_dataset_graphs(dataset: str, data_root: Path, seed: int, max_nodes: int) -> List[GraphWindow]:
    if dataset == "streamspot":
        graphs = load_streamspot_graphs(data_root, seed=seed, max_nodes=max_nodes)
    elif dataset == "grasec":
        graphs = load_grasec_graphs(data_root, seed=seed, max_nodes=max_nodes)
    elif dataset == "wget":
        graphs = load_wget_graphs(data_root, seed=seed, max_nodes=max_nodes)
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected streamspot, grasec, or wget")
    if not graphs:
        raise RuntimeError(f"No graphs loaded for dataset {dataset}")
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
    if not train_graphs:
        return
    train_x = torch.cat([g.x for g in train_graphs], dim=0)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    for graph in all_graphs:
        graph.x = (graph.x - mean) / std


def _limited(graphs: List[GraphWindow], limit: Optional[int], rng: random.Random) -> List[GraphWindow]:
    graphs = list(graphs)
    rng.shuffle(graphs)
    return graphs if limit is None or limit <= 0 else graphs[:limit]


def split_for_all_models(
    graphs: List[GraphWindow],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    benign_limit: Optional[int] = None,
    mal_limit: Optional[int] = None,
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
        raise RuntimeError("Need at least three benign graphs for train/val/test splitting.")
    if len(malicious) < 2:
        raise RuntimeError("Need at least two malicious graphs for train/val/test splitting.")

    all_selected = benign + malicious
    y = np.array([g.label for g in all_selected], dtype=np.int64)
    idx = np.arange(len(all_selected))
    test_size = 1.0 - train_ratio - val_ratio
    if test_size <= 0:
        raise RuntimeError("train_ratio + val_ratio must be less than 1.0")

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
    if not any(g.label == 1 for g in train_graphs):
        raise RuntimeError("Training split has no malicious graphs. Increase data size or adjust seed.")
    return train_graphs, val_graphs, test_graphs

def build_sparse_a_hat_from_undirected(
    num_nodes: int, edges_undirected: torch.Tensor, add_self_loops: bool = True
) -> torch.Tensor:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    if add_self_loops:
        for node in range(num_nodes):
            rows.append(node)
            cols.append(node)
            vals.append(1.0)
    if edges_undirected.numel() > 0:
        for u, v in edges_undirected.t().tolist():
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([1.0, 1.0])
    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    r, c = adj.indices()
    norm_vals = adj.values() / torch.sqrt(deg[r] * deg[c])
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape).coalesce()


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


def best_threshold_from_validation(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.unique(y_score)
    if candidates.size == 0:
        return 0.5
    best_thr = float(candidates[0])
    best_f1 = -1.0
    for thr in candidates:
        pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auroc": safe_auc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
    }


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name in METRIC_NAMES:
        vals = np.array([m[name] for m in metrics_list], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.nanmean(vals))
        summary[f"{name}_std"] = float(np.nanstd(vals))
    return summary


def read_existing_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary_csv(path: Path, rows: List[Dict[str, object]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_csv_rows(path) if append else []
    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {
        (str(r.get("dataset", "")), str(r.get("train_ratio", "")), str(r.get("model", ""))): r for r in existing
    }
    for row in rows:
        merged[(str(row["dataset"]), str(row["train_ratio"]), str(row["model"]))] = row
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in merged.values():
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})
    shutil.move(str(tmp), str(path))


def make_score_rows(
    dataset: str,
    train_ratio: float,
    seed: int,
    model_name: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> List[Dict[str, object]]:
    n = max(len(val_scores), len(test_scores))
    rows: List[Dict[str, object]] = []
    for idx in range(n):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_csv_rows(path) if append else []
    merged: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for row in existing:
        key = (str(row.get("dataset", "")), str(row.get("train_ratio", "")), str(row.get("seed", "")), str(row.get("model", "")), str(row.get("index", "")))
        merged[key] = row
    for row in rows:
        key = (str(row["dataset"]), str(row["train_ratio"]), str(row["seed"]), str(row["model"]), str(row["index"]))
        merged[key] = row
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in merged.values():
            writer.writerow({field: row.get(field, "") for field in SCORE_CSV_FIELDNAMES})
    shutil.move(str(tmp), str(path))
