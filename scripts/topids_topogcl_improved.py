#!/usr/bin/env python3
"""
TopIDS — Improved TopoGCL-style IDS anomaly detection for StreamSpot, GraSec, and Wget.

This script is intentionally based on topids_topogcl_prototype.py, not the old
.npz IDS baseline runner. It keeps the real TopoGCL-style pieces:

  - Graph channel: GIN encoder + graph-view InfoNCE.
  - Topology channel: 0-dimensional extended persistence, vectorized as
    Extended Persistence Landscapes (EPL), followed by an ETL MLP.
  - Benign-only self-supervised training.
  - kNN distance to benign training embeddings for anomaly scoring.
  - Validation F1 threshold tuning and balanced benign/malicious testing.

Datasets expected under data/:
  - StreamSpot: data/streamspot/all.tsv
  - GraSec:     data/grasec-iot/graph_json/Graph_JSON/{train,eval,test}/data_*.json
    - Wget:       data/wget/graphs.pkl

Examples:
  python3 topids_topogcl_improved.py --dataset streamspot --epochs 12 --mode graph_topo
  python3 topids_topogcl_improved.py --dataset grasec --epochs 12 --mode graph_topo
  python3 topids_topogcl_improved.py --dataset all --epochs 12 --seeds 42,43,44,45,46 --mode graph_topo
  python3 topids_topogcl_improved.py --dataset all --mode graph_only
  python3 topids_topogcl_improved.py --dataset all --mode topo_only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F

from ids_graph_data import (
    _graph_edge_index,
    _graph_num_nodes,
    _iter_wget_records,
    _label_to_int,
    _load_pickle_with_dgl_compat,
    _wget_node_features,
)

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"

STREAMSPOT_NODE_TYPES = list("abcdefgh")
STREAMSPOT_EDGE_TYPES = list("ijklmntquvwyzACDEG")

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision", "fpr"]
SUMMARY_FIELDS = [
    "dataset", "mode", "seeds", "filtrations", "alpha", "beta", "hidden", "embed_dim",
    "temperature", "drop_rate", "edge_drop_rate", "mask_rate", "batch_size",
]
for metric in METRIC_NAMES:
    SUMMARY_FIELDS.extend([f"{metric}_mean", f"{metric}_std"])
SUMMARY_FIELDS.extend(["threshold_mean", "threshold_std", "val_f1_mean", "val_f1_std"])

SCORE_FIELDS = [
    "dataset", "seed", "mode", "index", "val_score", "y_val", "test_score", "y_test",
]


@dataclass
class Graph:
    x: np.ndarray
    edge_index: np.ndarray
    label: int
    name: str = ""
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Improved TopoGCL IDS anomaly detection for StreamSpot, GraSec, and Wget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["streamspot", "grasec", "wget", "all"], default="all")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--seed", type=int, default=42, help="Used when --seeds is not provided.")
    p.add_argument("--seeds", default="", help="Comma-separated seeds, e.g., 42,43,44,45,46.")

    p.add_argument("--max-nodes", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=0, help="0 means full-batch over benign training graphs.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)

    p.add_argument("--mode", choices=["graph_only", "topo_only", "graph_topo"], default="graph_topo")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--drop-rate", type=float, default=0.1, help="Node drop rate for graph views.")
    p.add_argument("--edge-drop-rate", type=float, default=0.05, help="Edge drop rate for graph views.")
    p.add_argument("--mask-rate", type=float, default=0.1, help="Feature mask rate for graph views.")
    p.add_argument("--topo-mask-rate", type=float, default=0.10)
    p.add_argument("--topo-noise", type=float, default=0.10)

    p.add_argument("--num-landscapes", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument(
        "--filtrations",
        default="degree,closeness",
        help="Comma-separated: degree,total_degree,in_degree,out_degree,closeness,betweenness.",
    )

    p.add_argument("--knn-score", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--val-mal-ratio",
        type=float,
        default=0.2,
        help="Fraction of malicious graphs reserved for balanced validation threshold tuning.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)

    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "topids_streamspot_grasec")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-scores-csv", type=Path, default=None)
    return p.parse_args()


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.seeds_list = parse_csv_ints(args.seeds) if args.seeds.strip() else [args.seed]
    args.filtration_list = parse_csv_strings(args.filtrations)
    if not args.filtration_list:
        args.filtration_list = ["degree", "closeness"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_json is None:
        args.out_json = args.out_dir / f"topids_{args.dataset}_{args.mode}_results.json"
    if args.out_csv is None:
        args.out_csv = args.out_dir / f"topids_{args.dataset}_{args.mode}_summary.csv"
    if args.out_scores_csv is None:
        args.out_scores_csv = args.out_dir / f"topids_{args.dataset}_{args.mode}_scores.csv"

    if args.mode == "graph_only":
        args.alpha_eff, args.beta_eff = 1.0, 0.0
    elif args.mode == "topo_only":
        args.alpha_eff, args.beta_eff = 0.0, 1.0
    else:
        args.alpha_eff, args.beta_eff = args.alpha, args.beta
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
        np.add.at(deg, edge_index[1], 1.0)
    return deg


def compute_in_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[1], 1.0)
    return deg


def compute_out_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
    return deg


def subsample_graph(x: np.ndarray, edge_index: np.ndarray, max_nodes: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_nodes:
        return x.astype(np.float32), edge_index.astype(np.int64)
    deg = compute_degree(n, edge_index)
    top = np.argsort(-deg)[: max_nodes // 2]
    rest = np.setdiff1d(np.arange(n), top)
    extra = rng.choice(rest, size=max_nodes - len(top), replace=False)
    keep = np.sort(np.concatenate([top, extra]))
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    x2 = x[keep]
    if edge_index.size:
        src, dst = edge_index
        m = (remap[src] >= 0) & (remap[dst] >= 0)
        ei = np.stack([remap[src[m]], remap[dst[m]]])
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return x2.astype(np.float32), ei.astype(np.int64)


def _as_nx_graph(n: int, edge_index: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size:
        g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return g


def closeness_vector(n: int, edge_index: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    c = nx.closeness_centrality(_as_nx_graph(n, edge_index))
    return np.array([c.get(i, 0.0) for i in range(n)], dtype=np.float32)


def betweenness_vector(n: int, edge_index: np.ndarray, seed: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    g = _as_nx_graph(n, edge_index)
    # Exact betweenness can be slow. Use an approximation for larger graphs.
    k = None if n <= 200 else min(64, n)
    b = nx.betweenness_centrality(g, k=k, seed=seed, normalized=True)
    return np.array([b.get(i, 0.0) for i in range(n)], dtype=np.float32)


def get_filtration_values(g: Graph, name: str, seed: int) -> np.ndarray | None:
    name = name.lower()
    n = len(g.x)
    if name in g._cache:
        return g._cache[name]
    try:
        if name in {"degree", "total_degree"}:
            vals = compute_degree(n, g.edge_index)
        elif name == "in_degree":
            vals = compute_in_degree(n, g.edge_index)
        elif name == "out_degree":
            vals = compute_out_degree(n, g.edge_index)
        elif name == "closeness":
            vals = closeness_vector(n, g.edge_index)
        elif name == "betweenness":
            vals = betweenness_vector(n, g.edge_index, seed=seed)
        else:
            print(f"Warning: unknown filtration '{name}' skipped for {g.name}.")
            return None
    except Exception as exc:
        print(f"Warning: filtration '{name}' failed for {g.name}: {exc}; skipped.")
        return None
    vals = np.nan_to_num(vals.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    g._cache[name] = vals
    return vals


def _split_counts(graphs: list[Graph]) -> str:
    benign = sum(g.label == 0 for g in graphs)
    attack = len(graphs) - benign
    return f"total={len(graphs)} benign={benign} malicious={attack}"


def check_split_overlap(name: str, train: list[Graph], val: list[Graph], test: list[Graph]) -> None:
    train_names = {g.name for g in train}
    val_names = {g.name for g in val}
    test_names = {g.name for g in test}
    tv = train_names & val_names
    tt = train_names & test_names
    vt = val_names & test_names
    print(f"{name} split: train[{_split_counts(train)}] | val[{_split_counts(val)}] | test[{_split_counts(test)}]")
    print(f"{name} overlap check: train-val={len(tv)} train-test={len(tt)} val-test={len(vt)}")
    if tv or tt or vt:
        raise RuntimeError(f"{name} split overlap detected.")


def _subsample(graphs: list[Graph], n: int, rng: np.random.Generator) -> list[Graph]:
    if n >= len(graphs):
        return list(graphs)
    idx = rng.choice(len(graphs), size=n, replace=False)
    return [graphs[i] for i in idx]


def _balance_1to1(benign: list[Graph], malicious: list[Graph], rng: np.random.Generator):
    n = min(len(benign), len(malicious))
    if n == 0:
        raise RuntimeError("Need both benign and malicious graphs for 1:1 balancing.")
    return _subsample(benign, n, rng), _subsample(malicious, n, rng)


def split_train_val_test(graphs: list[Graph], seed: int, train_ratio: float, val_ratio: float,
                         val_mal_ratio: float = 0.2):
    benign = [g for g in graphs if g.label == 0]
    malicious = [g for g in graphs if g.label == 1]
    if len(benign) < 3:
        raise RuntimeError("Need at least three benign graphs for train/val/test.")
    if not malicious:
        raise RuntimeError("Need malicious graphs for balanced val/test.")

    train_pool, test_benign_pool = train_test_split(
        benign, train_size=train_ratio, random_state=seed, shuffle=True
    )
    n_val = max(1, int(len(train_pool) * val_ratio))
    train_benign, val_benign_pool = train_test_split(
        train_pool, test_size=n_val, random_state=seed, shuffle=True
    )

    rng = np.random.default_rng(seed)
    mal = list(malicious)
    rng.shuffle(mal)
    n_val_mal = max(1, min(len(mal), int(round(len(mal) * val_mal_ratio))))
    val_mal_pool = mal[:n_val_mal]
    test_mal_pool = mal[n_val_mal:]

    val_b, val_m = _balance_1to1(val_benign_pool, val_mal_pool, rng)
    test_b, test_m = _balance_1to1(test_benign_pool, test_mal_pool, rng)
    val = val_b + val_m
    test = test_b + test_m
    rng.shuffle(val)
    rng.shuffle(test)
    rng.shuffle(train_benign)
    return train_benign, val, test


def load_streamspot(args, seed: int):
    print("Step 1/4: Loading StreamSpot provenance graphs...")
    tsv = Path(args.data_root).expanduser().resolve() / "streamspot" / "all.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Missing {tsv}.")
    rng = np.random.default_rng(seed)
    n_node_types = len(STREAMSPOT_NODE_TYPES)
    graphs: list[Graph] = []
    current = None
    node_type: dict[str, int] = {}
    node_id: dict[str, int] = {}
    edges: list[tuple[int, int]] = []

    def finalize(gid: int | None):
        if gid is None or not node_id:
            return
        n = len(node_id)
        x = np.zeros((n, n_node_types + 1), dtype=np.float32)
        for nid, idx in node_id.items():
            x[idx, node_type[nid]] = 1.0
        ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        deg = compute_degree(n, ei)
        x[:, -1] = np.log1p(deg)
        x2, ei2 = subsample_graph(x, ei, args.max_nodes, rng)
        label = 1 if 300 <= gid <= 399 else 0
        graphs.append(Graph(x=x2, edge_index=ei2, label=label, name=f"streamspot_{gid}"))

    with open(tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            src, st, dst, dt, et, gid_s = parts
            gid = int(gid_s)
            if current is None:
                current = gid
            if gid != current:
                finalize(current)
                node_type, node_id, edges = {}, {}, []
                current = gid
            if st not in STREAMSPOT_NODE_TYPES or dt not in STREAMSPOT_NODE_TYPES:
                continue
            if et not in STREAMSPOT_EDGE_TYPES:
                continue
            if src not in node_id:
                node_id[src] = len(node_id)
                node_type[src] = STREAMSPOT_NODE_TYPES.index(st)
            if dst not in node_id:
                node_id[dst] = len(node_id)
                node_type[dst] = STREAMSPOT_NODE_TYPES.index(dt)
            edges.append((node_id[src], node_id[dst]))
        finalize(current)

    train, val, test = split_train_val_test(
        graphs, seed, args.train_ratio, args.val_ratio, args.val_mal_ratio
    )
    check_split_overlap("StreamSpot", train, val, test)
    return train, val, test


def grasec_label(snap: dict) -> int:
    for nd in snap.get("nodes", []):
        if nd.get("entity") == "connection" and "Label" in nd:
            lbl = nd["Label"]
            idx = lbl.index(1.0) if 1.0 in lbl else 6
            if idx != 6:
                return 1
    return 0


def _load_grasec_files(files: list[Path], args, seed: int) -> list[Graph]:
    rng = np.random.default_rng(seed)
    graphs: list[Graph] = []
    IP_DIM, CON_DIM = 8, 8
    feat_dim = 2 + IP_DIM + CON_DIM + 1
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            snapshots = json.load(f)
        for i, snap in enumerate(snapshots):
            nodes = snap.get("nodes", [])
            n = len(nodes)
            if n == 0:
                continue
            id_to_idx = {nd["id"]: j for j, nd in enumerate(nodes) if "id" in nd}
            x = np.zeros((n, feat_dim), dtype=np.float32)
            for j, nd in enumerate(nodes):
                ent = nd.get("entity", "")
                if ent == "ip":
                    x[j, 0] = 1.0
                    feats = nd.get("ip_feats", [])
                    x[j, 2:2 + min(IP_DIM, len(feats))] = feats[:IP_DIM]
                else:
                    x[j, 1] = 1.0
                    feats = nd.get("conect_feats", [])
                    off = 2 + IP_DIM
                    x[j, off:off + min(CON_DIM, len(feats))] = feats[:CON_DIM]
            ei = []
            for lk in snap.get("links", []):
                s, t = id_to_idx.get(lk.get("source")), id_to_idx.get(lk.get("target"))
                if s is not None and t is not None:
                    ei.append((s, t))
            ei_arr = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
            x[:, -1] = np.log1p(compute_degree(n, ei_arr))
            x2, ei2 = subsample_graph(x, ei_arr, args.max_nodes, rng)
            graphs.append(Graph(x=x2, edge_index=ei2, label=grasec_label(snap), name=f"grasec_{fp.stem}_{i}"))
    return graphs


def load_grasec(args, seed: int):
    print("Step 1/4: Loading GraSec-IoT graph snapshots...")
    base = args.data_root / "grasec-iot" / "graph_json" / "Graph_JSON"
    if not base.exists():
        raise FileNotFoundError(f"Missing GraSec Graph_JSON at {base}.")

    official = {s: sorted((base / s).glob("data_*.json")) for s in ("train", "eval", "test")}
    if all(official[s] for s in official):
        train_all = _load_grasec_files(official["train"], args, seed)
        val_all = _load_grasec_files(official["eval"], args, seed)
        test_all = _load_grasec_files(official["test"], args, seed)
        train = [g for g in train_all if g.label == 0]
        val_benign_pool = [g for g in val_all if g.label == 0]
        test_benign_pool = [g for g in test_all if g.label == 0]
        malicious = [g for g in test_all if g.label == 1]
        if train and val_benign_pool and test_benign_pool and malicious:
            rng = np.random.default_rng(seed)
            mal = list(malicious)
            rng.shuffle(mal)
            n_val_mal = max(1, min(len(mal), int(round(len(mal) * args.val_mal_ratio))))
            val_mal_pool = mal[:n_val_mal]
            test_mal_pool = mal[n_val_mal:]
            val_b, val_m = _balance_1to1(val_benign_pool, val_mal_pool, rng)
            test_b, test_m = _balance_1to1(test_benign_pool, test_mal_pool, rng)
            val = val_b + val_m
            test = test_b + test_m
            rng.shuffle(val)
            rng.shuffle(test)
            check_split_overlap("GraSec official", train, val, test)
            return train, val, test
        print("Warning: GraSec official split lacks coverage; falling back to random benign split.")

    all_files = [Path(p) for p in glob.glob(str(base / "*" / "data_*.json"))]
    graphs = _load_grasec_files(all_files, args, seed)
    train, val, test = split_train_val_test(
        graphs, seed, args.train_ratio, args.val_ratio, args.val_mal_ratio
    )
    check_split_overlap("GraSec fallback", train, val, test)
    return train, val, test


def load_wget(args, seed: int):
    print("Step 1/4: Loading Wget MAGIC DGL graphs...")
    pkl = args.data_root / "wget" / "graphs.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Missing Wget MAGIC graph dataset: {pkl}. Download MAGIC data/wget/graphs.zip "
            "and unzip it into data/wget/ so data/wget/graphs.pkl exists."
        )
    rng = np.random.default_rng(seed)
    raw = _load_pickle_with_dgl_compat(pkl)
    graphs: list[Graph] = []
    for idx, (graph, label) in enumerate(_iter_wget_records(raw)):
        n = _graph_num_nodes(graph)
        if n == 0:
            continue
        edge_index = _graph_edge_index(graph)
        x = _wget_node_features(graph, n, edge_index)
        x, edge_index = subsample_graph(x, edge_index, args.max_nodes, rng)
        graphs.append(Graph(x=x, edge_index=edge_index, label=_label_to_int(label), name=f"wget_{idx}"))
    print(f"Wget loaded: {_split_counts(graphs)}")
    train, val, test = split_train_val_test(graphs, seed, args.train_ratio, args.val_ratio, args.val_mal_ratio)
    check_split_overlap("Wget", train, val, test)
    return train, val, test


LOADERS = {"streamspot": load_streamspot, "grasec": load_grasec, "wget": load_wget}


def augment(g: Graph, drop_rate: float, mask_rate: float, edge_drop_rate: float,
            rng: np.random.Generator) -> Graph:
    n = len(g.x)
    if n <= 2 or drop_rate <= 0:
        keep = np.ones(n, dtype=bool)
    else:
        keep = rng.random(n) > drop_rate
        if keep.sum() < 2:
            deg = compute_degree(n, g.edge_index)
            keep[:] = False
            keep[np.argsort(-deg)[:2]] = True
    kept = np.where(keep)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    x = g.x[kept].copy()

    if x.shape[1] > 0 and mask_rate > 0:
        cols = rng.random(x.shape[1]) < mask_rate
        x[:, cols] = 0.0

    if g.edge_index.size:
        src, dst = g.edge_index
        m = (remap[src] >= 0) & (remap[dst] >= 0)
        ei = np.stack([remap[src[m]], remap[dst[m]]]).astype(np.int64) if np.any(m) else np.zeros((2, 0), dtype=np.int64)
        if ei.size and edge_drop_rate > 0 and ei.shape[1] > 1:
            keep_e = rng.random(ei.shape[1]) > edge_drop_rate
            if keep_e.sum() == 0:
                keep_e[rng.integers(0, ei.shape[1])] = True
            ei = ei[:, keep_e]
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return Graph(x=x.astype(np.float32), edge_index=ei, label=g.label, name=g.name)


def _zero_dim_pairs(n: int, adj: list[list[int]], fvals: np.ndarray, ascending: bool):
    order = np.argsort(fvals if ascending else -fvals, kind="stable")
    parent = list(range(n))
    birth = [0.0] * n
    active = np.zeros(n, dtype=bool)
    pairs = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for v in order:
        v = int(v)
        parent[v] = v
        birth[v] = float(fvals[v])
        active[v] = True
        for u in adj[v]:
            if not active[u] or u == v:
                continue
            rv, ru = find(v), find(u)
            if rv == ru:
                continue
            if ascending:
                die, sur = (rv, ru) if birth[rv] >= birth[ru] else (ru, rv)
            else:
                die, sur = (rv, ru) if birth[rv] <= birth[ru] else (ru, rv)
            b, d = birth[die], float(fvals[v])
            if b != d:
                pairs.append((b, d))
            parent[die] = sur
            birth[sur] = min(birth[sur], birth[die]) if ascending else max(birth[sur], birth[die])
    return pairs


def extended_persistence(n: int, edge_index: np.ndarray, fvals: np.ndarray):
    if n == 0:
        return [], []
    adj: list[list[int]] = [[] for _ in range(n)]
    if edge_index.size:
        for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if s != t:
                adj[s].append(t)
                adj[t].append(s)
    plus = _zero_dim_pairs(n, adj, fvals, ascending=True)
    minus = _zero_dim_pairs(n, adj, fvals, ascending=False)
    fmin, fmax = float(fvals.min()), float(fvals.max())
    if fmax > fmin:
        plus.append((fmin, fmax))
    return plus, minus


def landscape_vector(pairs, num_landscapes: int, tmin: float, tmax: float, num_samples: int):
    grid = np.linspace(tmin, tmax, num_samples)
    if not pairs or tmax <= tmin:
        return np.zeros(num_landscapes * num_samples, dtype=np.float32)
    tents = np.zeros((len(pairs), num_samples), dtype=np.float32)
    for i, (b, d) in enumerate(pairs):
        lo, hi = (b, d) if b <= d else (d, b)
        tents[i] = np.maximum(0.0, np.minimum(grid - lo, hi - grid))
    tents.sort(axis=0)
    out = []
    for k in range(num_landscapes):
        if k < tents.shape[0]:
            out.append(tents[-(k + 1)])
        else:
            out.append(np.zeros(num_samples, dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def epl_features(g: Graph, args) -> np.ndarray:
    n = len(g.x)
    blocks = []
    for filtration in args.filtration_list:
        fvals = get_filtration_values(g, filtration, seed=args.current_seed)
        if fvals is None:
            continue
        if n == 0:
            blocks.append(np.zeros(args.num_landscapes * args.num_samples * 2, dtype=np.float32))
            continue
        tmin, tmax = float(fvals.min()), float(fvals.max())
        plus, minus = extended_persistence(n, g.edge_index, fvals)
        v_plus = landscape_vector(plus, args.num_landscapes, tmin, tmax, args.num_samples)
        v_minus = landscape_vector(minus, args.num_landscapes, tmin, tmax, args.num_samples)
        blocks.append(np.concatenate([v_plus, v_minus]))
    if not blocks:
        # Safe fallback if all requested filtrations failed.
        fallback_dim = args.num_landscapes * args.num_samples * 2
        blocks.append(np.zeros(fallback_dim, dtype=np.float32))
    return np.concatenate(blocks).astype(np.float32)


def batch_graphs(graphs: list[Graph]):
    xs, eis, batch, offset = [], [], [], 0
    for gi, g in enumerate(graphs):
        n = len(g.x)
        xs.append(g.x)
        if g.edge_index.size:
            eis.append(g.edge_index + offset)
        batch.append(np.full(n, gi, dtype=np.int64))
        offset += n
    x = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    if eis:
        ei = np.concatenate(eis, axis=1)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    if ei.shape[1]:
        sym = np.concatenate([ei, ei[::-1]], axis=1)
        idx = torch.tensor(sym, dtype=torch.long)
        vals = torch.ones(idx.shape[1], dtype=torch.float32)
        adj = torch.sparse_coo_tensor(idx, vals, (offset, offset)).coalesce()
    else:
        adj = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (offset, offset)).coalesce()
    batch_t = torch.tensor(np.concatenate(batch), dtype=torch.long)
    return x, adj, batch_t, len(graphs)


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed_dim: int, num_layers: int = 2):
        super().__init__()
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.mlps = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.mlps.append(nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
            d = hidden
        self.readout_dim = 3 * hidden
        self.proj = nn.Sequential(nn.Linear(self.readout_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))

    def forward(self, x, adj, batch, num_graphs):
        h = x
        for eps, mlp in zip(self.eps, self.mlps):
            neigh = torch.sparse.mm(adj, h)
            h = mlp((1.0 + eps) * h + neigh)
            h = F.relu(h)
        dim = h.shape[1]
        sum_pool = torch.zeros(num_graphs, dim, device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(num_graphs, 1, device=h.device).index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device))
        mean_pool = sum_pool / counts.clamp(min=1.0)
        max_pool = torch.full((num_graphs, dim), -1e9, device=h.device)
        max_pool = max_pool.scatter_reduce(0, batch.unsqueeze(1).expand(-1, dim), h, reduce="amax")
        max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
        readout = torch.cat([sum_pool, mean_pool, max_pool], dim=1)
        z = self.proj(readout)
        return readout, z


class ETL(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, xi):
        return self.net(xi)


def info_nce(z1, z2, temperature: float):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def topo_matrix(graphs, args):
    return np.stack([epl_features(g, args) for g in graphs])


def _topo_augment(xi: torch.Tensor, args) -> torch.Tensor:
    mask = (torch.rand_like(xi) > args.topo_mask_rate).float()
    return xi * mask + args.topo_noise * torch.randn_like(xi)


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    idx = rng.permutation(n)
    if batch_size <= 0 or batch_size >= n:
        yield idx
        return
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def train_topogcl(train_graphs, args):
    if len(train_graphs) < 2:
        raise RuntimeError("TopoGCL contrastive training needs at least two benign training graphs.")
    print(f"Step 2/4: Training improved TopoGCL on benign graphs | mode={args.mode}")
    print(f"Progress: selected filtrations={args.filtration_list}")
    rng = np.random.default_rng(args.current_seed)
    in_dim = train_graphs[0].x.shape[1]

    gin = GINEncoder(in_dim, args.hidden, args.embed_dim) if args.mode != "topo_only" else None

    print("Progress: precomputing EPL topology features for train graphs...")
    xi_raw = torch.tensor(topo_matrix(train_graphs, args), dtype=torch.float32)
    topo_mean = xi_raw.mean(dim=0, keepdim=True)
    topo_std = xi_raw.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    xi_base = (xi_raw - topo_mean) / topo_std
    etl = ETL(xi_base.shape[1], args.embed_dim) if args.mode != "graph_only" else None

    params = []
    if gin is not None:
        params.extend(gin.parameters())
    if etl is not None:
        params.extend(etl.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    if gin is not None:
        gin.train()
    if etl is not None:
        etl.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        seen = 0
        for idx in _batch_indices(len(train_graphs), args.batch_size, rng):
            if len(idx) < 2:
                continue
            batch_graphs_raw = [train_graphs[i] for i in idx]
            losses = []
            if gin is not None:
                v1 = [augment(g, args.drop_rate, args.mask_rate, args.edge_drop_rate, rng) for g in batch_graphs_raw]
                v2 = [augment(g, args.drop_rate, args.mask_rate, args.edge_drop_rate, rng) for g in batch_graphs_raw]
                x1, a1, b1, ng1 = batch_graphs(v1)
                x2, a2, b2, ng2 = batch_graphs(v2)
                _, zH1 = gin(x1, a1, b1, ng1)
                _, zH2 = gin(x2, a2, b2, ng2)
                losses.append(args.alpha_eff * info_nce(zH1, zH2, args.temperature))
            if etl is not None:
                xi_batch = xi_base[torch.tensor(idx, dtype=torch.long)]
                xi1 = _topo_augment(xi_batch, args)
                xi2 = _topo_augment(xi_batch, args)
                zZ1, zZ2 = etl(xi1), etl(xi2)
                losses.append(args.beta_eff * info_nce(zZ1, zZ2, args.temperature))
            loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"Progress: epoch {epoch}/{args.epochs}, loss={total_loss / max(seen, 1):.4f}")
    if gin is not None:
        gin.eval()
    if etl is not None:
        etl.eval()
    return gin, etl, topo_mean, topo_std


@torch.no_grad()
def embed_graphs(graphs, gin, etl, topo_mean, topo_std, args, chunk: int = 256):
    parts = []
    H, Z = None, None
    if gin is not None:
        H_parts = []
        for i in range(0, len(graphs), chunk):
            sub = graphs[i:i + chunk]
            x, a, b, ng = batch_graphs(sub)
            _, z_graph = gin(x, a, b, ng)
            H_parts.append(F.normalize(z_graph, dim=1).numpy())
        H = np.concatenate(H_parts)
        if args.mode in {"graph_only", "graph_topo"}:
            parts.append(H)
    if etl is not None:
        Z_parts = []
        for i in range(0, len(graphs), chunk):
            sub = graphs[i:i + chunk]
            xi = torch.tensor(topo_matrix(sub, args), dtype=torch.float32)
            xi = (xi - topo_mean) / topo_std
            Z_parts.append(F.normalize(etl(xi), dim=1).numpy())
        Z = np.concatenate(Z_parts)
        if args.mode in {"topo_only", "graph_topo"}:
            parts.append(Z)
    if not parts:
        raise RuntimeError("No embeddings generated. Check --mode.")
    return np.concatenate(parts, axis=1), H, Z


def knn_scores(train_emb, query_emb, k):
    k_eff = min(k, len(train_emb))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(train_emb)
    dist, _ = nn.kneighbors(query_emb)
    return dist.mean(axis=1)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    if len(candidates) > 1:
        mids = (candidates[:-1] + candidates[1:]) / 2.0
        candidates = np.concatenate([candidates, mids])
    best_f1, best_thr = 0.0, float(scores.max()) + 1.0
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f >= best_f1:
            best_f1, best_thr = f, float(thr)
    return best_thr, best_f1


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, scores))


def safe_avg_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, scores))


def evaluate_topids(train_graphs, val_graphs, test_graphs, gin, etl, topo_mean, topo_std, args):
    print("Step 3/4: Scoring graphs (kNN distance to benign train embeddings)...")
    fit_emb, _, _ = embed_graphs(train_graphs, gin, etl, topo_mean, topo_std, args)
    val_emb, _, _ = embed_graphs(val_graphs, gin, etl, topo_mean, topo_std, args)
    test_emb, _, _ = embed_graphs(test_graphs, gin, etl, topo_mean, topo_std, args)

    y_val = np.array([g.label for g in val_graphs], dtype=int)
    y_test = np.array([g.label for g in test_graphs], dtype=int)
    val_s = knn_scores(fit_emb, val_emb, args.knn_score)
    test_s = knn_scores(fit_emb, test_emb, args.knn_score)

    thr, val_f1 = best_f1_threshold(y_val, val_s)
    print(f"Progress: val F1-optimal threshold={thr:.6f} (val F1={val_f1:.4f})")
    pred = (test_s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "val_f1": float(val_f1),
        "roc_auc": safe_roc_auc(y_test, test_s),
        "avg_precision": safe_avg_precision(y_test, test_s),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "val_scores": val_s.tolist(),
        "test_scores": test_s.tolist(),
        "y_val": y_val.tolist(),
        "y_test": y_test.tolist(),
    }


def print_result(name: str, seed: int, r: dict) -> None:
    print(f"Step 4/4: Results [{name}] seed={seed} (test 1:1 balanced)")
    print(f"  Threshold:      {r['threshold']:.6f}  (val F1={r['val_f1']:.4f})")
    print(f"  ROC-AUC:        {r['roc_auc']:.4f}")
    print(f"  Avg Precision:  {r['avg_precision']:.4f}")
    print(f"  F1:             {r['f1']:.4f}")
    print(f"  Accuracy:       {r['accuracy']:.4f}")
    print(f"  Precision:      {r['precision']:.4f}")
    print(f"  Recall:         {r['recall']:.4f}")
    print(f"  FPR:            {r['fpr']:.4f}")
    print(f"  Confusion: TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")


def run_dataset_seed(name: str, seed: int, args) -> dict:
    print(f"\n=== Dataset: {name} | seed={seed} ===")
    set_seed(seed)
    args.current_seed = seed
    train_graphs, val_graphs, test_graphs = LOADERS[name](args, seed)
    gin, etl, topo_mean, topo_std = train_topogcl(train_graphs, args)
    result = evaluate_topids(train_graphs, val_graphs, test_graphs, gin, etl, topo_mean, topo_std, args)
    result.update({
        "dataset": name,
        "seed": seed,
        "mode": args.mode,
        "filtrations": args.filtration_list,
        "alpha": args.alpha_eff,
        "beta": args.beta_eff,
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "temperature": args.temperature,
        "drop_rate": args.drop_rate,
        "edge_drop_rate": args.edge_drop_rate,
        "mask_rate": args.mask_rate,
        "batch_size": args.batch_size,
        "split": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
            "train_benign": int(sum(g.label == 0 for g in train_graphs)),
            "train_malicious": int(sum(g.label == 1 for g in train_graphs)),
            "val_benign": int(sum(g.label == 0 for g in val_graphs)),
            "val_malicious": int(sum(g.label == 1 for g in val_graphs)),
            "test_benign": int(sum(g.label == 0 for g in test_graphs)),
            "test_malicious": int(sum(g.label == 1 for g in test_graphs)),
        },
    })
    print_result(name, seed, result)
    return result


def summarize_results(dataset: str, results: list[dict], args) -> dict:
    summary = {
        "dataset": dataset,
        "mode": args.mode,
        "seeds": ",".join(str(s) for s in args.seeds_list),
        "filtrations": ",".join(args.filtration_list),
        "alpha": args.alpha_eff,
        "beta": args.beta_eff,
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "temperature": args.temperature,
        "drop_rate": args.drop_rate,
        "edge_drop_rate": args.edge_drop_rate,
        "mask_rate": args.mask_rate,
        "batch_size": args.batch_size,
    }
    for metric in METRIC_NAMES:
        values = np.array([r[metric] for r in results], dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    for metric in ["threshold", "val_f1"]:
        values = np.array([r[metric] for r in results], dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def write_outputs(all_results: dict[str, list[dict]], summaries: list[dict], args) -> None:
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "mode": args.mode,
            "seeds": args.seeds_list,
            "filtrations": args.filtration_list,
            "alpha": args.alpha_eff,
            "beta": args.beta_eff,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "embed_dim": args.embed_dim,
            "temperature": args.temperature,
            "drop_rate": args.drop_rate,
            "edge_drop_rate": args.edge_drop_rate,
            "mask_rate": args.mask_rate,
            "topo_mask_rate": args.topo_mask_rate,
            "topo_noise": args.topo_noise,
            "num_landscapes": args.num_landscapes,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
        },
        "summaries": summaries,
        "runs": all_results,
    }
    with args.out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    score_rows = []
    for dataset, runs in all_results.items():
        for r in runs:
            max_len = max(len(r["val_scores"]), len(r["test_scores"]), len(r["y_val"]), len(r["y_test"]))
            for i in range(max_len):
                score_rows.append({
                    "dataset": dataset,
                    "seed": r["seed"],
                    "mode": r["mode"],
                    "index": i,
                    "val_score": r["val_scores"][i] if i < len(r["val_scores"]) else "",
                    "y_val": r["y_val"][i] if i < len(r["y_val"]) else "",
                    "test_score": r["test_scores"][i] if i < len(r["test_scores"]) else "",
                    "y_test": r["y_test"][i] if i < len(r["y_test"]) else "",
                })
    with args.out_scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row in score_rows:
            writer.writerow({field: row.get(field, "") for field in SCORE_FIELDS})

    print(f"\n[OK] wrote {args.out_json}")
    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] wrote {args.out_scores_csv}")


def main():
    args = finalize_args(parse_args())
    print("TopIDS improved TopoGCL started.")
    print(f"Config: dataset={args.dataset} mode={args.mode} seeds={args.seeds_list}")
    print(f"Config: alpha={args.alpha_eff} beta={args.beta_eff} filtrations={args.filtration_list}")
    names = list(LOADERS) if args.dataset == "all" else [args.dataset]
    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []
    for name in names:
        runs = [run_dataset_seed(name, seed, args) for seed in args.seeds_list]
        all_results[name] = runs
        summaries.append(summarize_results(name, runs, args))

    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"  {s['dataset']} [{s['mode']}]: "
            f"ROC-AUC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}  "
            f"F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  "
            f"Recall={s['recall_mean']:.4f}±{s['recall_std']:.4f}"
        )
    write_outputs(all_results, summaries, args)
    print("Done.")


if __name__ == "__main__":
    main()
