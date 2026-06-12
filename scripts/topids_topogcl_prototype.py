#!/usr/bin/env python3
"""
TopIDS — TopoGCL-style graph intrusion detection for StreamSpot and GraSec.

Pipeline:
  - Real graph edges from StreamSpot/GraSec loaders only.
  - Train on benign graphs only; validation/test are balanced benign:malicious.
  - Graph channel: GIN encoder with InfoNCE over safe graph augmentations.
  - Topology channel: extended-persistence-landscape (EPL) vectors encoded by ETL.
  - Detection: kNN distance to benign training embeddings; tune threshold on val F1.

Default command preserved:
  python3 topids_topogcl_prototype.py --dataset all --epochs 12
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

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STREAMSPOT_NODE_TYPES = list("abcdefgh")
STREAMSPOT_EDGE_TYPES = list("ijklmntquvwyzACDEG")
VALID_FILTRATIONS = {"degree", "closeness", "in_degree", "out_degree", "total_degree", "betweenness"}
_WARNED: set[str] = set()


@dataclass
class Graph:
    x: np.ndarray
    edge_index: np.ndarray
    label: int
    name: str = ""
    _filtration_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        print(f"Warning: {msg}")
        _WARNED.add(key)


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StreamSpot/GraSec TopoGCL IDS experiment")
    p.add_argument("--dataset", choices=["streamspot", "grasec", "all"], default="all")
    p.add_argument("--seed", type=int, default=42, help="Single seed; ignored when --seeds is supplied")
    p.add_argument("--seeds", default="", help="Comma-separated seeds, e.g. 42,43,44,45,46")
    p.add_argument("--mode", choices=["graph_only", "topo_only", "graph_topo"], default="graph_topo")
    p.add_argument("--max-nodes", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=0, help="0 keeps full-batch training")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5, help="Graph InfoNCE loss weight")
    p.add_argument("--beta", type=float, default=0.5, help="Topology InfoNCE loss weight")
    p.add_argument("--force-streamspot-graph-only", action="store_true",
                   help="Explicitly set beta=0 for StreamSpot compatibility ablations")
    p.add_argument("--drop-rate", type=float, default=0.1, help="Node-drop rate")
    p.add_argument("--mask-rate", type=float, default=0.1, help="Feature-mask rate")
    p.add_argument("--edge-drop-rate", type=float, default=0.1)
    p.add_argument("--filtrations", default="degree,closeness",
                   help="Comma-separated subset of degree,closeness,in_degree,out_degree,total_degree,betweenness")
    p.add_argument("--betweenness-max-nodes", type=int, default=256,
                   help="Skip betweenness above this size unless approximate k is used")
    p.add_argument("--betweenness-k", type=int, default=64,
                   help="Approximate betweenness sample size for large safe graphs; 0 uses exact")
    p.add_argument("--num-landscapes", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--knn-score", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-mal-ratio", type=float, default=0.2,
                   help="Fraction of malicious graphs reserved for balanced val threshold tuning")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--out-dir", default="results/topids_topogcl")
    p.add_argument("--out-json", default="metrics.json")
    p.add_argument("--out-csv", default="summary.csv")
    p.add_argument("--out-scores-csv", default="scores.csv")
    args = p.parse_args()
    args.filtration_list = parse_csv_list(args.filtrations)
    bad = sorted(set(args.filtration_list) - VALID_FILTRATIONS)
    if bad:
        raise ValueError(f"Unknown filtrations {bad}; valid={sorted(VALID_FILTRATIONS)}")
    if not args.filtration_list:
        raise ValueError("At least one filtration is required.")
    args.seed_list = [int(s) for s in parse_csv_list(args.seeds)] if args.seeds else [args.seed]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
        np.add.at(deg, edge_index[1], 1.0)
    return deg


def compute_directed_degrees(n: int, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indeg = np.zeros(n, dtype=np.float32)
    outdeg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(outdeg, edge_index[0], 1.0)
        np.add.at(indeg, edge_index[1], 1.0)
    return indeg, outdeg


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
        ei = np.stack([remap[src[m]], remap[dst[m]]]) if np.any(m) else np.zeros((2, 0), dtype=np.int64)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return x2.astype(np.float32), ei.astype(np.int64)


def closeness_vector(n: int, edge_index: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size:
        g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    c = nx.closeness_centrality(g)
    return np.array([c.get(i, 0.0) for i in range(n)], dtype=np.float32)


def betweenness_vector(n: int, edge_index: np.ndarray, args, graph_name: str) -> np.ndarray | None:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n > args.betweenness_max_nodes:
        warn_once("betweenness_skip_large",
                  f"skipping betweenness for graphs with >{args.betweenness_max_nodes} nodes; using zero block")
        return None
    try:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        if edge_index.size:
            g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        k = None
        if args.betweenness_k and n > args.betweenness_k:
            k = args.betweenness_k
        c = nx.betweenness_centrality(g, k=k, seed=args.seed, normalized=True)
        return np.array([c.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception as exc:  # networkx can fail on pathological inputs; do not crash the run.
        warn_once(f"betweenness_fail_{graph_name}", f"could not compute betweenness for {graph_name}: {exc}; using zero block")
        return None


def filtration_values(g: Graph, name: str, args) -> np.ndarray | None:
    if name in g._filtration_cache:
        return g._filtration_cache[name]
    n, ei = len(g.x), g.edge_index
    if name == "degree":
        vals = compute_degree(n, ei)
    elif name in {"in_degree", "out_degree", "total_degree"}:
        indeg, outdeg = compute_directed_degrees(n, ei)
        vals = indeg if name == "in_degree" else outdeg if name == "out_degree" else indeg + outdeg
    elif name == "closeness":
        vals = closeness_vector(n, ei)
    elif name == "betweenness":
        vals = betweenness_vector(n, ei, args, g.name)
    else:
        vals = None
    if vals is not None:
        g._filtration_cache[name] = vals.astype(np.float32)
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
    print(f"Dataset label counts: benign={len(benign)} malicious={len(malicious)}")
    if len(benign) < 3:
        raise RuntimeError("Need at least three benign graphs for train/val/test.")
    if not malicious:
        raise RuntimeError("Need malicious graphs for balanced val/test.")

    train_pool, test_benign_pool = train_test_split(benign, train_size=train_ratio, random_state=seed, shuffle=True)
    n_val = max(1, int(len(train_pool) * val_ratio))
    train_benign, val_benign_pool = train_test_split(train_pool, test_size=n_val, random_state=seed, shuffle=True)

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


def load_streamspot(args):
    print("Step 1/4: Loading StreamSpot provenance graphs...")
    tsv = DATA / "streamspot" / "all.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Missing {tsv}.")
    rng = np.random.default_rng(args.seed)
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
        x, ei = subsample_graph(x, ei, args.max_nodes, rng)
        label = 1 if 300 <= gid <= 399 else 0
        graphs.append(Graph(x=x, edge_index=ei, label=label, name=f"streamspot_{gid}"))

    with open(tsv, "r", encoding="utf-8") as f:
        for line in f:
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

    print(f"StreamSpot loaded: {_split_counts(graphs)}")
    train, val, test = split_train_val_test(graphs, args.seed, args.train_ratio, args.val_ratio, args.val_mal_ratio)
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


def _load_grasec_files(files: list[Path], args) -> list[Graph]:
    rng = np.random.default_rng(args.seed)
    graphs: list[Graph] = []
    IP_DIM, CON_DIM = 8, 8
    feat_dim = 2 + IP_DIM + CON_DIM + 1
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            snapshots = json.load(f)
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
                    x[j, 2:2 + min(IP_DIM, len(feats))] = feats[:IP_DIM]
                else:
                    x[j, 1] = 1.0
                    feats = nd.get("conect_feats", [])
                    off = 2 + IP_DIM
                    x[j, off:off + min(CON_DIM, len(feats))] = feats[:CON_DIM]
            ei = []
            for lk in snap.get("links", []):
                s, t = id_to_idx.get(lk["source"]), id_to_idx.get(lk["target"])
                if s is not None and t is not None:
                    ei.append((s, t))
            ei = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
            x[:, -1] = np.log1p(compute_degree(n, ei))
            x, ei = subsample_graph(x, ei, args.max_nodes, rng)
            graphs.append(Graph(x=x, edge_index=ei, label=grasec_label(snap), name=f"grasec_{Path(fp).stem}_{i}"))
    return graphs


def load_grasec(args):
    print("Step 1/4: Loading GraSec-IoT graph snapshots...")
    base = DATA / "grasec-iot" / "graph_json" / "Graph_JSON"
    if not base.exists():
        raise FileNotFoundError("Missing GraSec Graph_JSON.")

    official = {s: sorted((base / s).glob("data_*.json")) for s in ("train", "eval", "test")}
    if all(official[s] for s in official):
        train_all = _load_grasec_files(official["train"], args)
        val_all = _load_grasec_files(official["eval"], args)
        test_all = _load_grasec_files(official["test"], args)
        train = [g for g in train_all if g.label == 0]
        val_benign_pool = [g for g in val_all if g.label == 0]
        test_benign_pool = [g for g in test_all if g.label == 0]
        malicious = [g for g in test_all if g.label == 1]
        print(f"GraSec loaded: train[{_split_counts(train_all)}] eval[{_split_counts(val_all)}] test[{_split_counts(test_all)}]")
        print(f"GraSec official label pools: benign_train={len(train)} val_benign={len(val_benign_pool)} "
              f"test_benign={len(test_benign_pool)} malicious={len(malicious)}")
        if train and val_benign_pool and test_benign_pool and malicious:
            rng = np.random.default_rng(args.seed)
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
    graphs = _load_grasec_files(all_files, args)
    print(f"GraSec fallback loaded: {_split_counts(graphs)}")
    train, val, test = split_train_val_test(graphs, args.seed, args.train_ratio, args.val_ratio, args.val_mal_ratio)
    check_split_overlap("GraSec fallback", train, val, test)
    return train, val, test


def augment(g: Graph, drop_rate: float, mask_rate: float, edge_drop_rate: float, rng: np.random.Generator) -> Graph:
    n = len(g.x)
    if n <= 2 or drop_rate <= 0:
        kept = np.arange(n)
    else:
        keep = rng.random(n) > drop_rate
        if keep.sum() < 2:
            keep[:] = False
            keep[rng.choice(n, size=2, replace=False)] = True
        kept = np.where(keep)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    x = g.x[kept].copy()
    if x.shape[1] and mask_rate > 0:
        cols = rng.random(x.shape[1]) < mask_rate
        x[:, cols] = 0.0
    if g.edge_index.size:
        src, dst = g.edge_index
        m = (remap[src] >= 0) & (remap[dst] >= 0)
        if edge_drop_rate > 0 and np.any(m):
            edge_keep = rng.random(m.sum()) > edge_drop_rate
            idx = np.where(m)[0]
            m[idx] = edge_keep
        ei = np.stack([remap[src[m]], remap[dst[m]]]).astype(np.int64) if np.any(m) else np.zeros((2, 0), dtype=np.int64)
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
            if s != t and 0 <= s < n and 0 <= t < n:
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
        out.append(tents[-(k + 1)] if k < tents.shape[0] else np.zeros(num_samples, dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def zero_epl_block(args) -> np.ndarray:
    return np.zeros(args.num_landscapes * args.num_samples * 2, dtype=np.float32)


def epl_features(g: Graph, args) -> np.ndarray:
    n, ei = len(g.x), g.edge_index
    blocks = []
    for filt in args.filtration_list:
        fvals = filtration_values(g, filt, args)
        if fvals is None or n == 0:
            blocks.append(zero_epl_block(args))
            continue
        try:
            tmin, tmax = float(fvals.min()), float(fvals.max())
            plus, minus = extended_persistence(n, ei, fvals)
            v_plus = landscape_vector(plus, args.num_landscapes, tmin, tmax, args.num_samples)
            v_minus = landscape_vector(minus, args.num_landscapes, tmin, tmax, args.num_samples)
            blocks.append(np.concatenate([v_plus, v_minus]))
        except Exception as exc:
            warn_once(f"epl_fail_{filt}_{g.name}", f"could not compute {filt} EPL for {g.name}: {exc}; using zero block")
            blocks.append(zero_epl_block(args))
    return np.concatenate(blocks).astype(np.float32)


def batch_graphs(graphs: list[Graph]):
    xs, eis, batch, offset = [], [], [], 0
    for gi, g in enumerate(graphs):
        n = len(g.x)
        if n == 0:
            continue
        xs.append(g.x)
        if g.edge_index.size:
            eis.append(g.edge_index + offset)
        batch.append(np.full(n, gi, dtype=np.int64))
        offset += n
    if not xs:
        raise RuntimeError("Cannot batch empty graphs only.")
    x = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    ei = np.concatenate(eis, axis=1) if eis else np.zeros((2, 0), dtype=np.int64)
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
        self.proj = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))

    def forward(self, x, adj, batch, num_graphs):
        h = x
        for eps, mlp in zip(self.eps, self.mlps):
            neigh = torch.sparse.mm(adj, h)
            h = F.relu(mlp((1.0 + eps) * h + neigh))
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


def topo_matrix(graphs: list[Graph], args) -> np.ndarray:
    return np.stack([epl_features(g, args) for g in graphs])


def _topo_augment(xi: torch.Tensor) -> torch.Tensor:
    mask = (torch.rand_like(xi) > 0.10).float()
    return xi * mask + 0.10 * torch.randn_like(xi)


def batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = rng.permutation(n)
    if batch_size <= 0 or batch_size >= n:
        yield order
    else:
        for i in range(0, n, batch_size):
            yield order[i:i + batch_size]


def mode_weights(args) -> tuple[float, float]:
    alpha = 0.0 if args.mode == "topo_only" else args.alpha
    beta = 0.0 if args.mode == "graph_only" else args.beta
    if alpha <= 0 and beta <= 0:
        raise ValueError(f"Mode {args.mode} produced zero loss weights; check --alpha/--beta.")
    return alpha, beta


def train_topogcl(train_graphs, args):
    print("Step 2/4: Training TopoGCL on benign graphs (InfoNCE in-batch negatives)...")
    rng = np.random.default_rng(args.seed)
    alpha, beta = mode_weights(args)
    in_dim = train_graphs[0].x.shape[1]
    topo_dim = len(args.filtration_list) * args.num_landscapes * args.num_samples * 2
    gin = GINEncoder(in_dim, args.hidden, args.embed_dim)
    etl = ETL(topo_dim, args.embed_dim)
    params = []
    if alpha > 0:
        params += list(gin.parameters())
    if beta > 0:
        params += list(etl.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    print(f"Progress: precomputing EPL for {len(train_graphs)} train graphs...")
    xi_raw = torch.tensor(topo_matrix(train_graphs, args), dtype=torch.float32)
    topo_mean = xi_raw.mean(dim=0, keepdim=True)
    topo_std = xi_raw.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    xi_base = (xi_raw - topo_mean) / topo_std

    gin.train(); etl.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, steps = 0.0, 0
        for idx in batch_indices(len(train_graphs), args.batch_size, rng):
            if len(idx) < 2:
                continue
            sub = [train_graphs[int(i)] for i in idx]
            loss = torch.tensor(0.0)
            if alpha > 0:
                v1 = [augment(g, args.drop_rate, args.mask_rate, args.edge_drop_rate, rng) for g in sub]
                v2 = [augment(g, args.drop_rate, args.mask_rate, args.edge_drop_rate, rng) for g in sub]
                x1, a1, b1, ng1 = batch_graphs(v1)
                x2, a2, b2, ng2 = batch_graphs(v2)
                _, zH1 = gin(x1, a1, b1, ng1)
                _, zH2 = gin(x2, a2, b2, ng2)
                loss = loss + alpha * info_nce(zH1, zH2, args.temperature)
            if beta > 0:
                xi = xi_base[torch.tensor(idx, dtype=torch.long)]
                zZ1, zZ2 = etl(_topo_augment(xi)), etl(_topo_augment(xi))
                loss = loss + beta * info_nce(zZ1, zZ2, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"Progress: epoch {epoch}/{args.epochs}, loss={avg_loss:.4f}")
    gin.eval(); etl.eval()
    return gin, etl, topo_mean, topo_std


@torch.no_grad()
def embed_graphs(graphs, gin, etl, topo_mean, topo_std, args, chunk: int = 256):
    H_parts, Z_parts = [], []
    for i in range(0, len(graphs), chunk):
        sub = graphs[i:i + chunk]
        x, a, b, ng = batch_graphs(sub)
        readout, z_graph = gin(x, a, b, ng)
        # Use the GIN projection embedding for graph-only scoring so embed_dim controls both channels.
        H_parts.append(F.normalize(z_graph, dim=1).numpy())
        xi = torch.tensor(topo_matrix(sub, args), dtype=torch.float32)
        xi = (xi - topo_mean) / topo_std
        Z_parts.append(F.normalize(etl(xi), dim=1).numpy())
    H = np.concatenate(H_parts)
    Z = np.concatenate(Z_parts)
    if args.mode == "graph_only":
        emb = H
    elif args.mode == "topo_only":
        emb = Z
    else:
        emb = np.concatenate([H, Z], axis=1)
    return emb, H, Z


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


def safe_auc(fn, y_true, scores) -> float:
    try:
        return float(fn(y_true, scores))
    except ValueError:
        return float("nan")


def metric_dict(y_true: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": safe_auc(roc_auc_score, y_true, scores),
        "avg_precision": safe_auc(average_precision_score, y_true, scores),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


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
    pred_val = (val_s >= thr).astype(int)
    pred_test = (test_s >= thr).astype(int)
    result = {"threshold": float(thr), "val_f1": float(val_f1)}
    result.update(metric_dict(y_test, test_s, pred_test))
    result["val_metrics"] = metric_dict(y_val, val_s, pred_val)
    score_rows = []
    for split, graphs, labels, scores, preds in (
        ("val", val_graphs, y_val, val_s, pred_val),
        ("test", test_graphs, y_test, test_s, pred_test),
    ):
        for g, y, s, p in zip(graphs, labels, scores, preds):
            score_rows.append({"split": split, "graph": g.name, "label": int(y), "score": float(s), "prediction": int(p)})
    return result, score_rows


def print_result(name: str, r: dict) -> None:
    print(f"Step 4/4: Results [{name}] (test 1:1 balanced)")
    print(f"  Threshold:      {r['threshold']:.6f}  (val F1={r['val_f1']:.4f})")
    print(f"  ROC-AUC:        {r['roc_auc']:.4f}")
    print(f"  Avg Precision:  {r['avg_precision']:.4f}")
    print(f"  F1:             {r['f1']:.4f}")
    print(f"  Accuracy:       {r['accuracy']:.4f}")
    print(f"  Precision:      {r['precision']:.4f}")
    print(f"  Recall:         {r['recall']:.4f}")
    print(f"  FPR:            {r['fpr']:.4f}")
    print(f"  Confusion: TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")


LOADERS = {"streamspot": load_streamspot, "grasec": load_grasec}


def config_row(dataset: str, args, result: dict) -> dict:
    keys = ["threshold", "val_f1", "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision", "fpr", "tn", "fp", "fn", "tp"]
    row = {
        "dataset": dataset,
        "seed": args.seed,
        "mode": args.mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "temperature": args.temperature,
        "drop_rate": args.drop_rate,
        "mask_rate": args.mask_rate,
        "edge_drop_rate": args.edge_drop_rate,
        "filtrations": ",".join(args.filtration_list),
        "batch_size": args.batch_size,
        "knn_score": args.knn_score,
    }
    row.update({k: result[k] for k in keys})
    return row


def run_dataset(name: str, args) -> tuple[dict, list[dict]]:
    print(f"\n=== Dataset: {name} | seed={args.seed} ===")
    if name == "streamspot" and args.force_streamspot_graph_only:
        print("Config: --force-streamspot-graph-only supplied; setting mode=graph_only and beta=0.0")
        args.mode = "graph_only"
        args.beta = 0.0
    set_seed(args.seed)
    print(f"Model mode: {args.mode} | alpha={args.alpha} beta={args.beta}")
    print(f"Selected filtrations: {','.join(args.filtration_list)}")
    print(f"Augmentations: node_drop={args.drop_rate} feature_mask={args.mask_rate} edge_drop={args.edge_drop_rate}")
    print(f"Batch size: {'full-batch' if args.batch_size == 0 else args.batch_size}")
    train_graphs, val_graphs, test_graphs = LOADERS[name](args)
    gin, etl, topo_mean, topo_std = train_topogcl(train_graphs, args)
    result, score_rows = evaluate_topids(train_graphs, val_graphs, test_graphs, gin, etl, topo_mean, topo_std, args)
    print_result(name, result)
    return result, score_rows


def summarize(rows: list[dict]) -> list[dict]:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision", "fpr", "threshold", "val_f1"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["dataset"], row["mode"], row["alpha"], row["beta"], row["hidden"], row["embed_dim"],
               row["temperature"], row["drop_rate"], row["mask_rate"], row["edge_drop_rate"], row["filtrations"])
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in groups.items():
        base = {
            "dataset": key[0], "mode": key[1], "alpha": key[2], "beta": key[3], "hidden": key[4],
            "embed_dim": key[5], "temperature": key[6], "drop_rate": key[7], "mask_rate": key[8],
            "edge_drop_rate": key[9], "filtrations": key[10], "n_seeds": len(vals),
            "seeds": ",".join(str(v["seed"]) for v in vals),
        }
        for m in metric_names:
            arr = np.array([float(v[m]) for v in vals], dtype=float)
            base[f"{m}_mean"] = float(np.nanmean(arr))
            base[f"{m}_std"] = float(np.nanstd(arr, ddof=0))
        out.append(base)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(args, metric_rows: list[dict], score_rows: list[dict]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / args.out_json
    summary_path = out_dir / args.out_csv
    scores_path = out_dir / args.out_scores_csv
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)
    write_csv(summary_path, summarize(metric_rows))
    write_csv(scores_path, score_rows)
    print(f"Saved per-seed metrics JSON: {json_path}")
    print(f"Saved summary metrics CSV:  {summary_path}")
    print(f"Saved anomaly scores CSV:   {scores_path}")


def main():
    args = parse_args()
    print("TopIDS TopoGCL StreamSpot/GraSec experiment started.")
    names = list(LOADERS) if args.dataset == "all" else [args.dataset]
    metric_rows: list[dict] = []
    all_score_rows: list[dict] = []
    for seed in args.seed_list:
        for name in names:
            run_args = argparse.Namespace(**vars(args))
            run_args.seed = seed
            result, scores = run_dataset(name, run_args)
            metric_rows.append(config_row(name, run_args, result))
            for row in scores:
                row.update({
                    "dataset": name,
                    "seed": seed,
                    "mode": run_args.mode,
                    "alpha": run_args.alpha,
                    "beta": run_args.beta,
                    "hidden": run_args.hidden,
                    "embed_dim": run_args.embed_dim,
                    "temperature": run_args.temperature,
                    "drop_rate": run_args.drop_rate,
                    "mask_rate": run_args.mask_rate,
                    "edge_drop_rate": run_args.edge_drop_rate,
                    "filtrations": ",".join(run_args.filtration_list),
                    "threshold": result["threshold"],
                })
                all_score_rows.append(row)
    save_outputs(args, metric_rows, all_score_rows)
    print("\n=== Summary ===")
    for row in summarize(metric_rows):
        print(f"  {row['dataset']} [{row['mode']} seeds={row['seeds']}]: "
              f"ROC-AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}  "
              f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
              f"Recall={row['recall_mean']:.4f}±{row['recall_std']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

# Example commands:
# Run StreamSpot:
#   python3 topids_topogcl_prototype.py --dataset streamspot --epochs 12 --mode graph_topo
# Run GraSec:
#   python3 topids_topogcl_prototype.py --dataset grasec --epochs 12 --mode graph_topo
# Run both:
#   python3 topids_topogcl_prototype.py --dataset all --epochs 12 --seeds 42,43,44,45,46 --mode graph_topo
# Run ablations:
#   python3 topids_topogcl_prototype.py --dataset all --mode graph_only
#   python3 topids_topogcl_prototype.py --dataset all --mode topo_only
