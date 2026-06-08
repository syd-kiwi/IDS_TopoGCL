#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import train_test_split

from build_ids_graph_classification_dataset import (
    ATTACK_MIN_EDGES,
    BENIGN_MIN_EDGES,
    CHUNK_SIZE,
    DST_IP_CANDIDATES,
    FIXED_NODE_FEATURES,
    MAX_EDGES,
    REQUIRED_EDGE_FEATURES,
    SRC_IP_CANDIDATES,
    TIME_CANDIDATES,
    _numeric,
    apply_edge_limits,
    find_col,
    normalize_chunk,
)
from train_ids_binary import (
    GCN,
    GraphSAGEEncoder,
    build_sparse_a_hat_from_undirected,
    compute_gcn_embeddings,
    compute_graphcl_embeddings,
    edge_index_to_undirected,
    train_graphcl_encoder,
    train_topogcl_encoder,
)

ATTACK_FAMILY_ORDER = ["Benign", "DDoS", "DoS", "Recon", "Mirai", "Spoofing_MITM", "Web", "BruteForce", "Malware"]
FOLDER_TO_FAMILY_EXACT = {
    "Benign_Final": "Benign",
    "VulnerabilityScan": "Recon",
    "DNS_Spoofing": "Spoofing_MITM",
    "MITM-ArpSpoofing": "Spoofing_MITM",
    "SqlInjection": "Web",
    "XSS": "Web",
    "CommandInjection": "Web",
    "Uploading_Attack": "Web",
    "BrowserHijacking": "Web",
    "DictionaryBruteForce": "BruteForce",
    "Backdoor_Malware": "Malware",
}
MODEL_DISPLAY = {"gnn": "GNN", "graphsage": "GraphSAGE", "graphcl": "GraphCL", "topoids": "TopoIDS", "topogcl": "TopoIDS"}
METRIC_NAMES = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
TABULAR_RESERVED_COLUMNS = {"label", "attack", "attacktype", "attack_cat", "class", "sub_label"}
CICIOT2023_AGGREGATE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

CSV_FIELDNAMES = [
    "dataset",
    "model",
    "seed",
    "train_ratio",
    "window_size",
    "num_classes",
    "class_distribution",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "per_class_metrics",
    "confusion_matrix",
]


@dataclass
class GraphWindow:
    x: torch.Tensor
    edges_undirected: torch.Tensor
    num_nodes: int
    window_start: int
    label: int
    family: str
    original_label: str
    file_name: str


@dataclass
class FolderGraphSample:
    graph_id: int
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    label: int
    family: str
    original_label: str
    num_benign_flows: int
    num_attack_flows: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    dataset_root: Path
    graph_cache_dir: Path
    out_json: Path
    out_csv: Path
    train_ratios: Tuple[float, ...]
    val_ratio: float
    seeds: Tuple[int, ...]
    epochs_supervised: int
    epochs_graphcl: int
    epochs_topogcl: int
    lr_supervised: float
    lr_graphcl: float
    lr_topogcl: float
    hidden_dim: int
    emb_dim: int
    graphcl_layers: int
    edge_drop: float
    feat_mask: float
    tau: float
    batch_size: int
    max_graphs: int
    max_nodes: int
    standardize: bool
    models: Tuple[str, ...]
    window_size: int
    rebuild_graphs: bool


def folder_to_family(folder_name: str) -> str:
    if folder_name in FOLDER_TO_FAMILY_EXACT:
        return FOLDER_TO_FAMILY_EXACT[folder_name]
    if folder_name.startswith("DDoS-"):
        return "DDoS"
    if folder_name.startswith("DoS-"):
        return "DoS"
    if folder_name.startswith("Recon-"):
        return "Recon"
    if folder_name.startswith("Mirai-"):
        return "Mirai"
    raise ValueError(f"No attack family mapping for folder {folder_name!r}")


def class_maps(families: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    ordered = [family for family in ATTACK_FAMILY_ORDER if family in set(families)]
    label_to_id = {family: idx for idx, family in enumerate(ordered)}
    return label_to_id, {idx: family for family, idx in label_to_id.items()}


def build_folder_window_graph(
    dfw: pd.DataFrame,
    graph_id: int,
    src_col: str,
    dst_col: str,
    edge_feat_cols: List[str],
    label: int,
    family: str,
    original_label: str,
) -> FolderGraphSample:
    ips = pd.unique(pd.concat([dfw[src_col].astype(str), dfw[dst_col].astype(str)], ignore_index=True))
    ip_to_idx = {ip: i for i, ip in enumerate(ips)}
    n = len(ips)

    src_idx = dfw[src_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    dst_idx = dfw[dst_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.stack([src_idx, dst_idx], axis=0)
    edge_features = dfw[edge_feat_cols].to_numpy(dtype=np.float32)

    in_bytes = _numeric(dfw, "IN_BYTES")
    out_bytes = _numeric(dfw, "OUT_BYTES")
    in_pkts = _numeric(dfw, "IN_PKTS")
    out_pkts = _numeric(dfw, "OUT_PKTS")
    flow_duration = _numeric(dfw, "FLOW_DURATION_MILLISECONDS")

    deg_in = np.zeros(n, dtype=np.float32)
    deg_out = np.zeros(n, dtype=np.float32)
    total_in_bytes = np.zeros(n, dtype=np.float32)
    total_out_bytes = np.zeros(n, dtype=np.float32)
    total_in_packets = np.zeros(n, dtype=np.float32)
    total_out_packets = np.zeros(n, dtype=np.float32)
    mean_flow_duration = np.zeros(n, dtype=np.float32)
    node_flow_count = np.zeros(n, dtype=np.float32)

    np.add.at(deg_out, src_idx, 1.0)
    np.add.at(deg_in, dst_idx, 1.0)
    np.add.at(total_out_bytes, src_idx, out_bytes)
    np.add.at(total_in_bytes, dst_idx, in_bytes)
    np.add.at(total_out_packets, src_idx, out_pkts)
    np.add.at(total_in_packets, dst_idx, in_pkts)
    np.add.at(mean_flow_duration, src_idx, flow_duration)
    np.add.at(mean_flow_duration, dst_idx, flow_duration)
    np.add.at(node_flow_count, src_idx, 1.0)
    np.add.at(node_flow_count, dst_idx, 1.0)
    nonzero = node_flow_count > 0
    mean_flow_duration[nonzero] /= node_flow_count[nonzero]

    node_features = np.concatenate(
        [
            deg_in[:, None],
            deg_out[:, None],
            total_in_bytes[:, None],
            total_out_bytes[:, None],
            total_in_packets[:, None],
            total_out_packets[:, None],
            mean_flow_duration[:, None],
            node_flow_count[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    is_benign = family == "Benign"
    return FolderGraphSample(
        graph_id=graph_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        label=label,
        family=family,
        original_label=original_label,
        num_benign_flows=len(dfw) if is_benign else 0,
        num_attack_flows=0 if is_benign else len(dfw),
    )


def save_multiclass_graph_npz(sample: FolderGraphSample, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / f"graph_{sample.graph_id:06d}_{sample.original_label}.npz",
        node_features=sample.node_features,
        edge_index=sample.edge_index,
        edge_features=sample.edge_features,
        label=np.array([sample.label], dtype=np.int64),
        attack_family=np.array([sample.family], dtype=np.str_),
        original_label=np.array([sample.original_label], dtype=np.str_),
    )


def append_graph_summary(summary_path: Path, sample: FolderGraphSample) -> None:
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["graph_id", "num_nodes", "num_edges", "num_benign_flows", "num_attack_flows", "label", "attack_family", "original_label"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "graph_id": sample.graph_id,
                "num_nodes": sample.node_features.shape[0],
                "num_edges": sample.edge_index.shape[1],
                "num_benign_flows": sample.num_benign_flows,
                "num_attack_flows": sample.num_attack_flows,
                "label": sample.label,
                "attack_family": sample.family,
                "original_label": sample.original_label,
            }
        )


def numeric_feature_columns(columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col.strip().lower() not in TABULAR_RESERVED_COLUMNS]


def is_ciciot2023_aggregate_columns(columns: Sequence[str]) -> bool:
    return set(CICIOT2023_AGGREGATE_COLUMNS).issubset(set(columns))


def build_tabular_feature_graph(
    dfw: pd.DataFrame,
    graph_id: int,
    feature_cols: List[str],
    label: int,
    family: str,
    original_label: str,
) -> FolderGraphSample:
    node_features = dfw[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    num_nodes = int(node_features.shape[0])
    if num_nodes <= 1:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 1), dtype=np.float32)
    else:
        src = np.arange(num_nodes - 1, dtype=np.int64)
        dst = np.arange(1, num_nodes, dtype=np.int64)
        edge_index = np.stack([src, dst], axis=0)
        edge_features = np.ones((num_nodes - 1, 1), dtype=np.float32)
    is_benign = family == "Benign"
    return FolderGraphSample(
        graph_id=graph_id,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        label=label,
        family=family,
        original_label=original_label,
        num_benign_flows=num_nodes if is_benign else 0,
        num_attack_flows=0 if is_benign else num_nodes,
    )


def save_sample_if_usable(sample: FolderGraphSample, out_dir: Path, summary_path: Path, family: str) -> bool:
    min_edges = BENIGN_MIN_EDGES if family == "Benign" else ATTACK_MIN_EDGES
    sample = apply_edge_limits(sample, min_edges, MAX_EDGES)
    if sample is None:
        return False
    if sample.node_features.ndim != 2 or sample.edge_index.shape[0] != 2:
        raise ValueError(
            f"Invalid graph shapes for graph_id={sample.graph_id}: "
            f"node_features={sample.node_features.shape}, edge_index={sample.edge_index.shape}"
        )
    save_multiclass_graph_npz(sample, out_dir)
    append_graph_summary(summary_path, sample)
    return True


def build_graphs_from_csv(
    input_csv: Path,
    folder_name: str,
    family: str,
    class_id: int,
    config: ExperimentConfig,
    out_dir: Path,
    summary_path: Path,
    start_graph_id: int,
) -> Tuple[int, int]:
    header_df = pd.read_csv(input_csv, nrows=1)
    cols = list(header_df.columns)
    if is_ciciot2023_aggregate_columns(cols):
        print(
            f"[INFO] {input_csv} matches the released CICIoT2023 aggregate-feature header; "
            "using folder-labeled tabular window graph construction.",
            flush=True,
        )
        return build_tabular_graphs_from_csv(input_csv, folder_name, family, class_id, config, out_dir, summary_path, start_graph_id)

    time_col = find_col(cols, TIME_CANDIDATES)
    src_col = find_col(cols, SRC_IP_CANDIDATES)
    dst_col = find_col(cols, DST_IP_CANDIDATES)
    missing_edge_cols = [col for col in REQUIRED_EDGE_FEATURES if col not in cols]
    missing_flow_cols = []
    if time_col is None:
        missing_flow_cols.append("time")
    if src_col is None:
        missing_flow_cols.append("src_ip")
    if dst_col is None:
        missing_flow_cols.append("dst_ip")
    missing_flow_cols.extend(missing_edge_cols)
    can_use_flow_pipeline = not missing_flow_cols
    if not can_use_flow_pipeline:
        # The released CICIoT2023 per-attack CSVs are aggregate feature tables
        # (e.g. Header_Length, Protocol Type, Rate, ...), not raw flow/IP
        # tables. Use the folder as the original attack label instead of
        # requiring missing time/IP/label columns.
        print(
            f"[INFO] {input_csv} lacks flow graph columns {missing_flow_cols}; "
            "using CICIoT2023 folder-labeled tabular window graph construction.",
            flush=True,
        )
        return build_tabular_graphs_from_csv(input_csv, folder_name, family, class_id, config, out_dir, summary_path, start_graph_id)

    graph_count = start_graph_id
    total_rows_read = 0
    carry_window_df = pd.DataFrame()
    for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE):
        total_rows_read += len(chunk)
        cleaned_chunk = normalize_chunk(chunk, time_col, src_col, dst_col, REQUIRED_EDGE_FEATURES.copy(), config.window_size)
        if cleaned_chunk.empty and carry_window_df.empty:
            continue
        current_df = pd.concat([carry_window_df, cleaned_chunk], ignore_index=True)
        if current_df.empty:
            continue
        windows = current_df["__window"].to_numpy()
        last_window = windows[-1]
        complete_df = current_df.loc[windows != last_window]
        carry_window_df = current_df.loc[windows == last_window].copy()
        for _, gdf in complete_df.groupby("__window", sort=True):
            sample = build_folder_window_graph(gdf, graph_count, src_col, dst_col, REQUIRED_EDGE_FEATURES.copy(), class_id, family, folder_name)
            if save_sample_if_usable(sample, out_dir, summary_path, family):
                graph_count += 1
            if config.max_graphs > 0 and graph_count >= config.max_graphs:
                return graph_count, total_rows_read
    if not carry_window_df.empty and (config.max_graphs <= 0 or graph_count < config.max_graphs):
        sample = build_folder_window_graph(carry_window_df, graph_count, src_col, dst_col, REQUIRED_EDGE_FEATURES.copy(), class_id, family, folder_name)
        if save_sample_if_usable(sample, out_dir, summary_path, family):
            graph_count += 1
    return graph_count, total_rows_read


def build_tabular_graphs_from_csv(
    input_csv: Path,
    folder_name: str,
    family: str,
    class_id: int,
    config: ExperimentConfig,
    out_dir: Path,
    summary_path: Path,
    start_graph_id: int,
) -> Tuple[int, int]:
    header_df = pd.read_csv(input_csv, nrows=1)
    feature_cols = numeric_feature_columns(list(header_df.columns))
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {input_csv}. Available columns: {list(header_df.columns)}")

    graph_count = start_graph_id
    total_rows_read = 0
    carry_df = pd.DataFrame()
    rows_per_graph = max(2, int(config.window_size))
    for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE):
        total_rows_read += len(chunk)
        chunk = chunk[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        current_df = pd.concat([carry_df, chunk], ignore_index=True)
        while len(current_df) >= rows_per_graph:
            gdf = current_df.iloc[:rows_per_graph].copy()
            current_df = current_df.iloc[rows_per_graph:].reset_index(drop=True)
            sample = build_tabular_feature_graph(gdf, graph_count, feature_cols, class_id, family, folder_name)
            if save_sample_if_usable(sample, out_dir, summary_path, family):
                graph_count += 1
            if config.max_graphs > 0 and graph_count >= config.max_graphs:
                return graph_count, total_rows_read
        carry_df = current_df.reset_index(drop=True)
    if not carry_df.empty and (config.max_graphs <= 0 or graph_count < config.max_graphs):
        sample = build_tabular_feature_graph(carry_df, graph_count, feature_cols, class_id, family, folder_name)
        if save_sample_if_usable(sample, out_dir, summary_path, family):
            graph_count += 1
    return graph_count, total_rows_read


def mapped_dataset_folders(dataset_root: Path, graph_cache_dir: Path) -> List[Tuple[Path, str]]:
    folders: List[Tuple[Path, str]] = []
    cache_resolved = graph_cache_dir.resolve()
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if folder.resolve() == cache_resolved:
            continue
        try:
            folders.append((folder, folder_to_family(folder.name)))
        except ValueError:
            print(f"[WARN] skipping unmapped dataset folder {folder}", flush=True)
    return folders


def build_graph_cache(config: ExperimentConfig, label_to_id: Dict[str, int]) -> None:
    out_dir = config.graph_cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("graph_*.npz"):
        old_file.unlink()
    summary_path = out_dir / "graph_window_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    graph_count = 0
    for folder, family in mapped_dataset_folders(config.dataset_root, config.graph_cache_dir):
        class_id = label_to_id[family]
        csv_paths = sorted(folder.glob("*.csv"))
        if not csv_paths:
            print(f"[WARN] no CSV files found in {folder}", flush=True)
            continue
        for input_csv in csv_paths:
            graph_count, total_rows_read = build_graphs_from_csv(
                input_csv=input_csv,
                folder_name=folder.name,
                family=family,
                class_id=class_id,
                config=config,
                out_dir=out_dir,
                summary_path=summary_path,
                start_graph_id=graph_count,
            )
            print(f"[OK] processed {input_csv} rows={total_rows_read} graphs_so_far={graph_count}", flush=True)
            if config.max_graphs > 0 and graph_count >= config.max_graphs:
                break
        if config.max_graphs > 0 and graph_count >= config.max_graphs:
            break

    with (out_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "dataset_root": str(config.dataset_root),
                "window_size": config.window_size,
                "window_size_units": "seconds for flow/IP CSVs; rows for CICIoT2023 aggregate feature CSVs without time/IP columns",
                "node_feature_columns": FIXED_NODE_FEATURES,
                "edge_feature_columns": REQUIRED_EDGE_FEATURES,
                "class_to_id": label_to_id,
                "num_graphs": graph_count,
            },
            f,
            indent=2,
        )
    print(f"[OK] wrote {graph_count} multiclass graph samples to {out_dir}", flush=True)


def load_npz_graphs(graph_dir: Path, max_graphs: Optional[int], max_nodes: int) -> List[GraphWindow]:
    graphs: List[GraphWindow] = []
    for path in sorted(graph_dir.glob("graph_*.npz")):
        arr = np.load(path, allow_pickle=True)
        node_features = np.nan_to_num(arr["node_features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if node_features.ndim == 1:
            node_features = node_features.reshape(-1, 1)
        num_nodes = int(node_features.shape[0])
        if num_nodes == 0 or num_nodes > max_nodes:
            continue
        edge_index = arr["edge_index"].astype(np.int64)
        family = str(np.array(arr["attack_family"]).reshape(-1)[0])
        original_label = str(np.array(arr["original_label"]).reshape(-1)[0])
        graphs.append(
            GraphWindow(
                x=torch.tensor(node_features, dtype=torch.float32),
                edges_undirected=edge_index_to_undirected(edge_index, num_nodes=num_nodes),
                num_nodes=num_nodes,
                window_start=len(graphs),
                label=int(np.array(arr["label"]).reshape(-1)[0]),
                family=family,
                original_label=original_label,
                file_name=path.name,
            )
        )
        if max_graphs is not None and max_graphs > 0 and len(graphs) >= max_graphs:
            break
    if not graphs:
        raise RuntimeError(f"No usable multiclass graph files loaded from {graph_dir}")
    return graphs


def clone_graphs(graphs: Sequence[GraphWindow]) -> List[GraphWindow]:
    return [GraphWindow(g.x.clone(), g.edges_undirected.clone(), g.num_nodes, g.window_start, g.label, g.family, g.original_label, g.file_name) for g in graphs]


def count_by_class(graphs: Sequence[GraphWindow], id_to_label: Dict[int, str]) -> Dict[str, int]:
    counts = Counter(g.label for g in graphs)
    return {id_to_label[idx]: int(counts.get(idx, 0)) for idx in sorted(id_to_label)}


def print_counts(tag: str, graphs: Sequence[GraphWindow], id_to_label: Dict[int, str]) -> None:
    print(f"[COUNTS] {tag}: {count_by_class(graphs, id_to_label)}", flush=True)


def standardize_from_train(train_graphs: List[GraphWindow], all_graphs: List[GraphWindow]) -> None:
    xs = torch.cat([g.x for g in train_graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)
    for g in all_graphs:
        g.x = torch.nan_to_num((g.x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def split_for_all_models(graphs: List[GraphWindow], seed: int, train_ratio: float, val_ratio: float) -> Tuple[List[GraphWindow], List[GraphWindow], List[GraphWindow]]:
    y = np.array([g.label for g in graphs], dtype=np.int64)
    idx = np.arange(len(graphs))
    test_size = 1.0 - train_ratio - val_ratio
    if test_size <= 0:
        raise RuntimeError("TRAIN_RATIO + VAL_RATIO must be less than 1.0")
    counts = np.bincount(y)
    strat = y if len(np.unique(y)) > 1 and np.min(counts[counts > 0]) >= 3 else None
    train_idx, temp_idx, _, y_temp = train_test_split(idx, y, test_size=(1.0 - train_ratio), random_state=seed, stratify=strat)
    val_fraction_of_temp = val_ratio / (val_ratio + test_size)
    temp_counts = np.bincount(y_temp)
    strat_temp = y_temp if len(np.unique(y_temp)) > 1 and np.min(temp_counts[temp_counts > 0]) >= 2 else None
    val_idx, test_idx, _, _ = train_test_split(temp_idx, y_temp, test_size=(1.0 - val_fraction_of_temp), random_state=seed, stratify=strat_temp)
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx], [graphs[i] for i in test_idx]


class GraphClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, emb_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, graph: GraphWindow, device: torch.device) -> torch.Tensor:
        adj = build_sparse_a_hat_from_undirected(graph.num_nodes, graph.edges_undirected).to(device)
        z = self.encoder(graph.x.to(device), adj)
        return self.head(z)


class FrozenEmbeddingClassifier(torch.nn.Module):
    def __init__(self, emb_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def predict_supervised(classifier: GraphClassifier, graphs: List[GraphWindow], device: torch.device) -> np.ndarray:
    classifier.eval()
    preds: List[int] = []
    with torch.no_grad():
        for graph in graphs:
            preds.append(int(torch.argmax(classifier(graph, device), dim=0).item()))
    return np.array(preds, dtype=np.int64)


def train_supervised(
    model_name: str,
    encoder_factory: Callable[[], torch.nn.Module],
    train_graphs: List[GraphWindow],
    hidden_dim: int,
    num_classes: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> GraphClassifier:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    classifier = GraphClassifier(encoder_factory(), emb_dim=hidden_dim, num_classes=num_classes).to(device)
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


def train_embedding_classifier(
    train_emb: torch.Tensor,
    train_labels: np.ndarray,
    eval_emb: torch.Tensor,
    num_classes: int,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
    model_name: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = FrozenEmbeddingClassifier(train_emb.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.lr_supervised)
    labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    train_emb = train_emb.detach().to(device)
    indices = list(range(train_emb.shape[0]))
    for epoch in range(config.epochs_supervised):
        rng.shuffle(indices)
        total_loss = 0.0
        for idx in indices:
            logits = head(train_emb[idx]).unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(logits, labels[idx].view(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"    {model_name} linear epoch {epoch + 1}/{config.epochs_supervised} loss={total_loss / max(len(indices), 1):.6f}", flush=True)
    head.eval()
    with torch.no_grad():
        return torch.argmax(head(eval_emb.to(device)), dim=1).detach().cpu().numpy().astype(np.int64)


def model_runner(model_name: str, train_graphs: List[GraphWindow], test_graphs: List[GraphWindow], in_dim: int, num_classes: int, config: ExperimentConfig, seed: int, device: torch.device) -> np.ndarray:
    if model_name == "gnn":
        classifier = train_supervised("gnn", lambda: GCN(in_dim, config.hidden_dim, config.hidden_dim), train_graphs, config.hidden_dim, num_classes, config.epochs_supervised, config.lr_supervised, seed, device)
        return predict_supervised(classifier, test_graphs, device)
    if model_name == "graphsage":
        classifier = train_supervised("graphsage", lambda: GraphSAGEEncoder(in_dim, config.hidden_dim, config.hidden_dim), train_graphs, config.hidden_dim, num_classes, config.epochs_supervised, config.lr_supervised, seed, device)
        return predict_supervised(classifier, test_graphs, device)
    if model_name == "graphcl":
        encoder = train_graphcl_encoder(train_graphs, in_dim, config.hidden_dim, config.graphcl_layers, config.epochs_graphcl, config.lr_graphcl, config.edge_drop, config.feat_mask, config.tau, config.batch_size, seed, device)
        train_emb = compute_graphcl_embeddings(encoder, train_graphs, device, tag="embed graphcl train")
        test_emb = compute_graphcl_embeddings(encoder, test_graphs, device, tag="embed graphcl test")
        return train_embedding_classifier(train_emb, np.array([g.label for g in train_graphs]), test_emb, num_classes, config, seed, device, "graphcl")
    if model_name in {"topoids", "topogcl"}:
        model = GCN(in_dim=in_dim, hidden_dim=config.hidden_dim, out_dim=config.emb_dim)
        train_topogcl_encoder(model=model, graphs=train_graphs, config=config, seed=seed, device=device)
        train_emb = compute_gcn_embeddings(model, train_graphs, device, tag="embed topoids train")
        test_emb = compute_gcn_embeddings(model, test_graphs, device, tag="embed topoids test")
        return train_embedding_classifier(train_emb, np.array([g.label for g in train_graphs]), test_emb, num_classes, config, seed, device, "topoids")
    raise ValueError(f"Unknown graph model: {model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, id_to_label: Dict[int, str]) -> Dict[str, object]:
    labels = sorted(id_to_label)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            id_to_label[label]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def write_results_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), (dict, list)) else row.get(field, "") for field in CSV_FIELDNAMES})


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    families = [family for _, family in mapped_dataset_folders(config.dataset_root, config.graph_cache_dir)]
    if not families:
        raise RuntimeError(f"No mapped CICIoT2023 attack folders found under {config.dataset_root}")
    label_to_id, id_to_label = class_maps(families)
    if config.rebuild_graphs or not any(config.graph_cache_dir.glob("graph_*.npz")):
        build_graph_cache(config, label_to_id)
    base_graphs = load_npz_graphs(config.graph_cache_dir, config.max_graphs if config.max_graphs > 0 else None, config.max_nodes)
    num_classes = len(id_to_label)
    print(f"[OK] device: {device}", flush=True)
    print(f"[OK] dataset={config.dataset} classes={id_to_label}", flush=True)
    print_counts("before split", base_graphs, id_to_label)

    csv_rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    for train_ratio in config.train_ratios:
        for seed in config.seeds:
            print(f"\n[RUN] dataset={config.dataset} train_ratio={train_ratio} seed={seed}", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            graphs = clone_graphs(base_graphs)
            train_graphs, val_graphs, test_graphs = split_for_all_models(graphs, seed, train_ratio, config.val_ratio)
            print_counts("train split", train_graphs, id_to_label)
            print_counts("validation split", val_graphs, id_to_label)
            print_counts("test split", test_graphs, id_to_label)
            if config.standardize:
                standardize_from_train(train_graphs, train_graphs + val_graphs + test_graphs)
            in_dim = train_graphs[0].x.shape[1]
            y_test = np.array([g.label for g in test_graphs], dtype=np.int64)
            split_distribution = {"all": count_by_class(base_graphs, id_to_label), "train": count_by_class(train_graphs, id_to_label), "validation": count_by_class(val_graphs, id_to_label), "test": count_by_class(test_graphs, id_to_label)}
            for model_name in config.models:
                print(f"[MODEL] {MODEL_DISPLAY[model_name]}", flush=True)
                y_pred = model_runner(model_name, train_graphs, test_graphs, in_dim, num_classes, config, seed, device)
                metrics = compute_metrics(y_test, y_pred, id_to_label)
                run_record = {
                    "dataset": config.dataset,
                    "model": MODEL_DISPLAY[model_name],
                    "seed": seed,
                    "train_ratio": train_ratio,
                    "window_size": config.window_size,
                    "num_classes": num_classes,
                    "class_distribution": split_distribution,
                    "metrics": metrics,
                }
                runs.append(run_record)
                csv_rows.append({"dataset": config.dataset, "model": MODEL_DISPLAY[model_name], "seed": seed, "train_ratio": train_ratio, "window_size": config.window_size, "num_classes": num_classes, "class_distribution": split_distribution, **{m: metrics[m] for m in METRIC_NAMES}, "per_class_metrics": metrics["per_class"], "confusion_matrix": metrics["confusion_matrix"]})
    results = {
        "dataset": config.dataset,
        "dataset_root": str(config.dataset_root),
        "graph_cache_dir": str(config.graph_cache_dir),
        "device": str(device),
        "window_size": config.window_size,
        "num_classes": num_classes,
        "class_to_id": label_to_id,
        "id_to_class": id_to_label,
        "class_distribution": count_by_class(base_graphs, id_to_label),
        "training": {"models": [MODEL_DISPLAY[m] for m in config.models], "seeds": list(config.seeds), "train_ratios": list(config.train_ratios), "val_ratio": config.val_ratio, "epochs_supervised": config.epochs_supervised, "epochs_graphcl": config.epochs_graphcl, "epochs_topogcl": config.epochs_topogcl, "hidden_dim": config.hidden_dim, "emb_dim": config.emb_dim, "graphcl_layers": config.graphcl_layers, "edge_drop": config.edge_drop, "feat_mask": config.feat_mask, "tau": config.tau, "batch_size": config.batch_size, "standardized": config.standardize},
        "runs": runs,
    }
    config.out_json.parent.mkdir(parents=True, exist_ok=True)
    with config.out_json.open("w") as f:
        json.dump(results, f, indent=2)
    write_results_csv(config.out_csv, csv_rows)
    print(f"[OK] wrote {config.out_json}", flush=True)
    print(f"[OK] wrote {config.out_csv}", flush=True)
    return results


def parse_float_tuple(raw: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CICIoT2023 graph IDS models for multiclass attack-family classification.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/kiwi-pandas/Extreme SSD/network_data/CIC_IoT"))
    parser.add_argument("--dataset", default="CICIoT2023")
    parser.add_argument("--graph-cache-dir", type=Path, default=Path("/media/kiwi-pandas/Extreme SSD/network_data/CIC_IoT/Graph_Multiclass"))
    parser.add_argument("--out-json", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/cic_iot/cic_iot_multiclass_results_10.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/results/cic_iot/cic_iot_multiclass_results_10.csv"))
    parser.add_argument("--train-ratios", default="0.10")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--models", default="gnn,graphsage,graphcl,topoids", help="Comma-separated models: gnn,graphsage,graphcl,topoids")
    parser.add_argument("--window-size", type=int, default=15, help="Window size in seconds; matches the existing graph construction pipeline default.")
    parser.add_argument("--epochs-supervised", type=int, default=1)
    parser.add_argument("--epochs-graphcl", type=int, default=1)
    parser.add_argument("--epochs-topogcl", type=int, default=1)
    parser.add_argument("--lr-supervised", type=float, default=1e-3)
    parser.add_argument("--lr-graphcl", type=float, default=1e-3)
    parser.add_argument("--lr-topogcl", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--graphcl-layers", type=int, default=2)
    parser.add_argument("--edge-drop", type=float, default=0.001)
    parser.add_argument("--feat-mask", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--rebuild-graphs", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    models = tuple(model.strip().lower() for model in args.models.split(",") if model.strip())
    allowed = {"gnn", "graphsage", "graphcl", "topoids", "topogcl"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; allowed models are {sorted(allowed)}")
    normalized_models = tuple("topoids" if model == "topogcl" else model for model in models)
    return ExperimentConfig(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        graph_cache_dir=args.graph_cache_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        train_ratios=parse_float_tuple(args.train_ratios),
        val_ratio=args.val_ratio,
        seeds=parse_int_tuple(args.seeds),
        epochs_supervised=args.epochs_supervised,
        epochs_graphcl=args.epochs_graphcl,
        epochs_topogcl=args.epochs_topogcl,
        lr_supervised=args.lr_supervised,
        lr_graphcl=args.lr_graphcl,
        lr_topogcl=args.lr_topogcl,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        graphcl_layers=args.graphcl_layers,
        edge_drop=args.edge_drop,
        feat_mask=args.feat_mask,
        tau=args.tau,
        batch_size=args.batch_size,
        max_graphs=args.max_graphs,
        max_nodes=args.max_nodes,
        standardize=not args.no_standardize,
        models=normalized_models,
        window_size=args.window_size,
        rebuild_graphs=args.rebuild_graphs,
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
