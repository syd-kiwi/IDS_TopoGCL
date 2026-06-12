#!/usr/bin/env python3
"""
TopIDS — TopoGCL-based graph intrusion detection (end-to-end prototype).

Pipeline (TopoGCL, AAAI 2024):
  - Real graph edges from each dataset (no kNN re-wiring).
  - Graph channel: GIN encoder, readout = concat(sum, mean, max).
    Two contrastive views via node-drop + feature-mask augmentation.
  - Topology channel: 0-dim extended persistence (degree + closeness filtrations),
    vectorized as Extended Persistence Landscapes (EPL), fed to ETL (5-layer MLP).
    EPL computed once per graph; two views use standardized feature-space aug
    (dropout + Gaussian noise).
  - Loss: L = alpha * InfoNCE(H, H') + beta * InfoNCE(Z, Z').
    Negatives: standard in-batch InfoNCE (diagonal positives, all others negative).
  - Train on benign graphs only; val/test balanced 1:1 (benign:malicious).
  - Detection: kNN distance to benign train embeddings; threshold tuned on
    balanced val set to maximize F1, then applied to balanced test set.

Datasets (under data/):
  - streamspot: 600 provenance graphs (gid 300..399 = attack).
  - grasec:     IoT flow-graph snapshots (connection.Label idx 6 = benign).

Run:
  python3 topids_topogcl_prototype.py --dataset all --epochs 12
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

STREAMSPOT_NODE_TYPES = list("abcdefgh")
STREAMSPOT_EDGE_TYPES = list("ijklmntquvwyzACDEG")


@dataclass
class Graph:
    x: np.ndarray
    edge_index: np.ndarray
    label: int
    name: str = ""
    _deg: np.ndarray | None = field(default=None, repr=False)
    _close: np.ndarray | None = field(default=None, repr=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["streamspot", "grasec", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-nodes", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--mask-rate", type=float, default=0.1)
    p.add_argument("--num-landscapes", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--knn-score", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-mal-ratio", type=float, default=0.2,
                   help="Fraction of malicious graphs reserved for balanced val (threshold tuning)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_degree(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
        np.add.at(deg, edge_index[1], 1.0)
    return deg


def subsample_graph(x: np.ndarray, edge_index: np.ndarray, max_nodes: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_nodes:
        return x, edge_index
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


def closeness_vector(n: int, edge_index: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size:
        g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    c = nx.closeness_centrality(g)
    return np.array([c.get(i, 0.0) for i in range(n)], dtype=np.float32)


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
    print(f"{name} split: train[{_split_counts(train)}] | val[{_split_counts(val)}] | "
          f"test[{_split_counts(test)}]")
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

    def finalize(gid: int):
        if not node_id:
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
        graphs, args.seed, args.train_ratio, args.val_ratio, args.val_mal_ratio
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
            graphs.append(Graph(x=x, edge_index=ei, label=grasec_label(snap),
                                name=f"grasec_{Path(fp).stem}_{i}"))
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
    train, val, test = split_train_val_test(
        graphs, args.seed, args.train_ratio, args.val_ratio, args.val_mal_ratio
    )
    check_split_overlap("GraSec fallback", train, val, test)
    return train, val, test


def augment(g: Graph, drop_rate: float, mask_rate: float, rng: np.random.Generator) -> Graph:
    n = len(g.x)
    keep = rng.random(n) > drop_rate
    if keep.sum() < 4:
        keep[:] = True
    kept = np.where(keep)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    x = g.x[kept].copy()
    if x.shape[1] > 1:
        cols = rng.random(x.shape[1]) < mask_rate
        x[:, cols] = 0.0
    if g.edge_index.size:
        src, dst = g.edge_index
        m = (remap[src] >= 0) & (remap[dst] >= 0)
        ei = np.stack([remap[src[m]], remap[dst[m]]]).astype(np.int64)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    close = g._close[kept] if g._close is not None else None
    return Graph(x=x.astype(np.float32), edge_index=ei, label=g.label, name=g.name, _close=close)


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
    ei = g.edge_index
    deg = compute_degree(n, ei)
    if g._close is None:
        g._close = closeness_vector(n, ei)
    close = g._close
    blocks = []
    for fvals in (deg, close):
        if n == 0:
            blocks.append(np.zeros(args.num_landscapes * args.num_samples * 2, dtype=np.float32))
            continue
        tmin, tmax = float(fvals.min()), float(fvals.max())
        plus, minus = extended_persistence(n, ei, fvals)
        v_plus = landscape_vector(plus, args.num_landscapes, tmin, tmax, args.num_samples)
        v_minus = landscape_vector(minus, args.num_landscapes, tmin, tmax, args.num_samples)
        blocks.append(np.concatenate([v_plus, v_minus]))
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
        adj = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long),
                                      torch.zeros(0), (offset, offset)).coalesce()
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
            h = mlp((1.0 + eps) * h + neigh)
            h = F.relu(h)
        dim = h.shape[1]
        sum_pool = torch.zeros(num_graphs, dim).index_add_(0, batch, h)
        counts = torch.zeros(num_graphs, 1).index_add_(0, batch, torch.ones(h.shape[0], 1))
        mean_pool = sum_pool / counts.clamp(min=1.0)
        max_pool = torch.full((num_graphs, dim), -1e9)
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
    """Symmetric InfoNCE: in-batch negatives (standard TopoGCL contrastive loss)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def topo_matrix(graphs, args):
    return np.stack([epl_features(g, args) for g in graphs])


def _topo_augment(xi: torch.Tensor) -> torch.Tensor:
    mask = (torch.rand_like(xi) > 0.10).float()
    return xi * mask + 0.10 * torch.randn_like(xi)


def train_topogcl(train_graphs, args):
    print("Step 2/4: Training TopoGCL on benign graphs (InfoNCE in-batch negatives)...")
    rng = np.random.default_rng(args.seed)
    print(f"Progress: precomputing closeness for {len(train_graphs)} train graphs...")
    for g in train_graphs:
        if g._close is None:
            g._close = closeness_vector(len(g.x), g.edge_index)
    in_dim = train_graphs[0].x.shape[1]
    topo_dim = epl_features(train_graphs[0], args).shape[0]
    gin = GINEncoder(in_dim, args.hidden, args.embed_dim)
    etl = ETL(topo_dim, args.embed_dim)
    opt = torch.optim.Adam(list(gin.parameters()) + list(etl.parameters()), lr=1e-3)

    print("Progress: precomputing EPL for train graphs...")
    xi_raw = torch.tensor(topo_matrix(train_graphs, args), dtype=torch.float32)
    topo_mean = xi_raw.mean(dim=0, keepdim=True)
    topo_std = xi_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    xi_base = (xi_raw - topo_mean) / topo_std

    gin.train(); etl.train()
    for epoch in range(1, args.epochs + 1):
        v1 = [augment(g, args.drop_rate, args.mask_rate, rng) for g in train_graphs]
        v2 = [augment(g, args.drop_rate, args.mask_rate, rng) for g in train_graphs]
        xi1 = _topo_augment(xi_base)
        xi2 = _topo_augment(xi_base)
        x1, a1, b1, ng1 = batch_graphs(v1)
        x2, a2, b2, ng2 = batch_graphs(v2)
        _, zH1 = gin(x1, a1, b1, ng1)
        _, zH2 = gin(x2, a2, b2, ng2)
        zZ1, zZ2 = etl(xi1), etl(xi2)
        loss = args.alpha * info_nce(zH1, zH2, args.temperature) \
            + args.beta * info_nce(zZ1, zZ2, args.temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"Progress: epoch {epoch}/{args.epochs}, loss={loss.item():.4f}")
    gin.eval(); etl.eval()
    return gin, etl, topo_mean, topo_std


@torch.no_grad()
def embed_graphs(graphs, gin, etl, topo_mean, topo_std, args, chunk: int = 256):
    H_parts, Z_parts = [], []
    for i in range(0, len(graphs), chunk):
        sub = graphs[i:i + chunk]
        xi = torch.tensor(topo_matrix(sub, args), dtype=torch.float32)
        xi = (xi - topo_mean) / topo_std
        x, a, b, ng = batch_graphs(sub)
        readout, _ = gin(x, a, b, ng)
        H_parts.append(F.normalize(readout, dim=1).numpy())
        Z_parts.append(F.normalize(etl(xi), dim=1).numpy())
    H = np.concatenate(H_parts)
    Z = np.concatenate(Z_parts)
    emb = np.concatenate([H, Z], axis=1)
    return emb, H, Z


def knn_scores(train_emb, query_emb, k):
    k_eff = min(k, len(train_emb))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(train_emb)
    dist, _ = nn.kneighbors(query_emb)
    return dist.mean(axis=1)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Pick threshold on validation scores that maximizes F1."""
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
        "threshold": thr,
        "val_f1": val_f1,
        "roc_auc": roc_auc_score(y_test, test_s),
        "avg_precision": average_precision_score(y_test, test_s),
        "f1": f1_score(y_test, pred, zero_division=0),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


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

DATASET_CONFIG = {
    "streamspot": {"alpha": 1.0, "beta": 0.0, "knn_score": 5},
    "grasec": {"alpha": 0.5, "beta": 0.5, "knn_score": 5},
}


def apply_dataset_config(name: str, args) -> None:
    cfg = DATASET_CONFIG.get(name, {})
    for k, v in cfg.items():
        setattr(args, k, v)
    if cfg:
        print(f"Config: alpha={args.alpha} beta={args.beta} knn={args.knn_score}")


def run_dataset(name: str, args) -> dict:
    print(f"\n=== Dataset: {name} ===")
    set_seed(args.seed)
    apply_dataset_config(name, args)
    train_graphs, val_graphs, test_graphs = LOADERS[name](args)
    gin, etl, topo_mean, topo_std = train_topogcl(train_graphs, args)
    result = evaluate_topids(train_graphs, val_graphs, test_graphs, gin, etl, topo_mean, topo_std, args)
    print_result(name, result)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    print("TopIDS TopoGCL prototype started.")
    names = list(LOADERS) if args.dataset == "all" else [args.dataset]
    all_results = {}
    for name in names:
        all_results[name] = run_dataset(name, args)
    print("\n=== Summary ===")
    for name, r in all_results.items():
        print(f"  {name}: ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  Recall={r['recall']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
