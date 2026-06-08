#!/usr/bin/env python3
"""Build malicious-ratio multiclass IDS graph windows from foldered NetFlow CSVs.

Each CSV time window becomes one graph saved as the same ``.npz`` format used by
this repository's graph training scripts: ``node_features``, ``edge_index``,
``edge_features``, and ``label`` arrays. Labels are derived from the fraction of
malicious flows inside each window rather than from attack-family folder names.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from build_ids_graph_classification_dataset import (
    ATTACK_MIN_EDGES,
    BENIGN_MIN_EDGES,
    CHUNK_SIZE,
    DST_IP_CANDIDATES,
    FIXED_NODE_FEATURES,
    LABEL_CANDIDATES,
    MAX_EDGES,
    PROTO_CANDIDATES,
    REQUIRED_EDGE_FEATURES,
    SRC_IP_CANDIDATES,
    TIME_CANDIDATES,
    _numeric,
    apply_edge_limits,
    assert_graph_shapes,
    find_col,
    infer_attack,
)

BENIGN_FOLDER_NAME = "Benign_Final"


@dataclass
class RatioGraphSample:
    graph_id: int
    window_id: int
    source_csv: Path
    source_folder: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    graph_label: int
    malicious_ratio: float
    num_benign_flows: int
    num_attack_flows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an IDS NetFlow dataset root with attack folders and "
            "Benign_Final into malicious-ratio multiclass graph windows."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder containing attack folders and Benign_Final.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where graph .npz files and metadata are written.")
    parser.add_argument("--window-size", type=int, default=15, help="Time-window size in seconds. Default: 15.")
    parser.add_argument(
        "--label-mode",
        choices=("3class", "4class"),
        default="3class",
        help="Use 3 malicious-ratio classes by default, or split partial windows into two classes with 4class.",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=0,
        help="Optional maximum number of CSV files to process per top-level class folder. 0 means no limit.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"CSV chunk size. Default: {CHUNK_SIZE}.")
    parser.add_argument("--max-edges", type=int, default=MAX_EDGES, help=f"Maximum edges to keep per graph. Default: {MAX_EDGES}.")
    parser.add_argument(
        "--allow-missing-label",
        action="store_true",
        help="If a CSV lacks a label column, infer all flows as benign in Benign_Final and malicious otherwise.",
    )
    return parser.parse_args()


def label_from_ratio(malicious_ratio: float, label_mode: str) -> int:
    if malicious_ratio <= 0.0:
        return 0
    if malicious_ratio >= 1.0:
        return 2 if label_mode == "3class" else 3
    if label_mode == "3class":
        return 1
    return 1 if malicious_ratio < 0.5 else 2


def class_names(label_mode: str) -> Dict[int, str]:
    if label_mode == "3class":
        return {0: "benign", 1: "partially_malicious", 2: "fully_malicious"}
    return {0: "benign", 1: "lightly_malicious", 2: "half_heavily_malicious", 3: "fully_malicious"}


def dataset_folders(dataset_root: Path, output_dir: Path) -> List[Path]:
    output_resolved = output_dir.resolve()
    folders = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if folder.resolve() == output_resolved:
            continue
        folders.append(folder)
    return folders


def csv_files_for_folder(folder: Path, max_files_per_class: int) -> List[Path]:
    paths = sorted(folder.rglob("*.csv"))
    if max_files_per_class > 0:
        return paths[:max_files_per_class]
    return paths


def normalize_chunk(
    chunk: pd.DataFrame,
    time_col: str,
    src_col: str,
    dst_col: str,
    edge_feat_cols: Sequence[str],
    window_seconds: int,
    label_col: Optional[str],
    folder_is_benign: bool,
) -> pd.DataFrame:
    df = chunk.copy()
    numeric_time = pd.to_numeric(df[time_col], errors="coerce")
    if numeric_time.notna().any():
        median_abs = float(numeric_time.dropna().abs().median())
        unit = "ms"
        if median_abs < 10_000_000_000:
            unit = "s"
        elif median_abs > 10_000_000_000_000_000:
            unit = "ns"
        elif median_abs > 10_000_000_000_000:
            unit = "us"
        df["__time"] = pd.to_datetime(numeric_time, unit=unit, errors="coerce", utc=True)
    else:
        df["__time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    df = df.dropna(subset=["__time", src_col, dst_col]).sort_values("__time").reset_index(drop=True)
    for col in edge_feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if label_col and label_col in df.columns and not folder_is_benign:
        df["__attack"] = df[label_col].map(infer_attack).astype(np.int32)
    else:
        df["__attack"] = np.int32(0 if folder_is_benign else 1)

    epoch = (df["__time"].astype("int64") // 10**9).astype(np.int64)
    df["__window"] = (epoch // window_seconds).astype(np.int64)
    return df


def aggregate_edge_features(dfw: pd.DataFrame, src_col: str, dst_col: str, edge_feat_cols: Sequence[str], ip_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    grouped = dfw.groupby([src_col, dst_col], sort=False, dropna=False)
    edge_pairs: List[Tuple[int, int]] = []
    edge_features: List[np.ndarray] = []
    for (src_ip, dst_ip), group in grouped:
        edge_pairs.append((ip_to_idx[str(src_ip)], ip_to_idx[str(dst_ip)]))
        # Keep the established IDS edge feature width/order, but aggregate all
        # numeric flow values for the src-dst pair into one mean feature vector.
        edge_features.append(group[list(edge_feat_cols)].mean(numeric_only=False).to_numpy(dtype=np.float32))

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(edge_feat_cols)), dtype=np.float32)
    return np.asarray(edge_pairs, dtype=np.int64).T, np.vstack(edge_features).astype(np.float32)


def build_ratio_window_graph(
    dfw: pd.DataFrame,
    graph_id: int,
    src_col: str,
    dst_col: str,
    edge_feat_cols: Sequence[str],
    label_mode: str,
    source_csv: Path,
    source_folder: str,
) -> RatioGraphSample:
    ips = pd.unique(pd.concat([dfw[src_col].astype(str), dfw[dst_col].astype(str)], ignore_index=True))
    ip_to_idx = {ip: i for i, ip in enumerate(ips)}
    num_nodes = len(ips)

    src_idx = dfw[src_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    dst_idx = dfw[dst_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    edge_index, edge_features = aggregate_edge_features(dfw, src_col, dst_col, edge_feat_cols, ip_to_idx)

    in_bytes = _numeric(dfw, "IN_BYTES")
    out_bytes = _numeric(dfw, "OUT_BYTES")
    in_pkts = _numeric(dfw, "IN_PKTS")
    out_pkts = _numeric(dfw, "OUT_PKTS")
    flow_duration = _numeric(dfw, "FLOW_DURATION_MILLISECONDS")

    deg_in = np.zeros(num_nodes, dtype=np.float32)
    deg_out = np.zeros(num_nodes, dtype=np.float32)
    total_in_bytes = np.zeros(num_nodes, dtype=np.float32)
    total_out_bytes = np.zeros(num_nodes, dtype=np.float32)
    total_in_packets = np.zeros(num_nodes, dtype=np.float32)
    total_out_packets = np.zeros(num_nodes, dtype=np.float32)
    mean_flow_duration = np.zeros(num_nodes, dtype=np.float32)
    node_flow_count = np.zeros(num_nodes, dtype=np.float32)

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

    attack_flags = dfw["__attack"].to_numpy(dtype=np.int32)
    num_attack = int(attack_flags.sum())
    num_total = int(len(attack_flags))
    num_benign = num_total - num_attack
    malicious_ratio = float(num_attack / num_total) if num_total else 0.0
    graph_label = label_from_ratio(malicious_ratio, label_mode)

    return RatioGraphSample(
        graph_id=graph_id,
        window_id=int(dfw["__window"].iloc[0]),
        source_csv=source_csv,
        source_folder=source_folder,
        start_time=dfw["__time"].iloc[0],
        end_time=dfw["__time"].iloc[-1],
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        graph_label=graph_label,
        malicious_ratio=malicious_ratio,
        num_benign_flows=num_benign,
        num_attack_flows=num_attack,
    )


def save_graph_npz(sample: RatioGraphSample, out_dir: Path, label_mode: str) -> None:
    label_name = class_names(label_mode)[sample.graph_label]
    np.savez_compressed(
        out_dir / f"graph_{sample.graph_id:06d}.npz",
        node_features=sample.node_features,
        edge_index=sample.edge_index,
        edge_features=sample.edge_features,
        label=np.array([sample.graph_label], dtype=np.int64),
        malicious_ratio=np.array([sample.malicious_ratio], dtype=np.float32),
        num_benign_flows=np.array([sample.num_benign_flows], dtype=np.int64),
        num_attack_flows=np.array([sample.num_attack_flows], dtype=np.int64),
        source_folder=np.array([sample.source_folder], dtype=np.str_),
        attack_family=np.array([label_name], dtype=np.str_),
        original_label=np.array([sample.source_folder], dtype=np.str_),
    )


def append_metadata_row(metadata_path: Path, sample: RatioGraphSample) -> None:
    fieldnames = [
        "graph_id",
        "window_id",
        "class_label",
        "malicious_ratio",
        "num_flows",
        "num_benign_flows",
        "num_attack_flows",
        "num_nodes",
        "num_edges",
        "start_time",
        "end_time",
        "source_folder",
        "source_csv",
    ]
    write_header = not metadata_path.exists()
    with metadata_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "graph_id": sample.graph_id,
                "window_id": sample.window_id,
                "class_label": sample.graph_label,
                "malicious_ratio": f"{sample.malicious_ratio:.8f}",
                "num_flows": sample.num_benign_flows + sample.num_attack_flows,
                "num_benign_flows": sample.num_benign_flows,
                "num_attack_flows": sample.num_attack_flows,
                "num_nodes": sample.node_features.shape[0],
                "num_edges": sample.edge_index.shape[1],
                "start_time": sample.start_time.isoformat(),
                "end_time": sample.end_time.isoformat(),
                "source_folder": sample.source_folder,
                "source_csv": str(sample.source_csv),
            }
        )


def prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_graph in out_dir.glob("graph_*.npz"):
        old_graph.unlink()
    for old_metadata in ("graph_window_metadata.csv", "graph_window_summary.csv", "metadata.json"):
        path = out_dir / old_metadata
        if path.exists():
            path.unlink()


def validate_columns(input_csv: Path, allow_missing_label: bool) -> Tuple[str, str, str, Optional[str], Optional[str], List[str]]:
    header_df = pd.read_csv(input_csv, nrows=1)
    cols = list(header_df.columns)
    time_col = find_col(cols, TIME_CANDIDATES)
    src_col = find_col(cols, SRC_IP_CANDIDATES)
    dst_col = find_col(cols, DST_IP_CANDIDATES)
    proto_col = find_col(cols, PROTO_CANDIDATES)
    label_col = find_col(cols, LABEL_CANDIDATES)
    missing = [name for name, col in [("time", time_col), ("src_ip", src_col), ("dst_ip", dst_col), ("protocol", proto_col)] if col is None]
    if label_col is None and not allow_missing_label:
        missing.append("label")
    if missing:
        raise ValueError(f"Missing required columns {missing} in {input_csv}. Available columns: {cols}")
    missing_edge_cols = [col for col in REQUIRED_EDGE_FEATURES if col not in cols]
    if missing_edge_cols:
        raise ValueError(f"Missing required edge feature columns {missing_edge_cols} in {input_csv}")
    return time_col, src_col, dst_col, proto_col, label_col, REQUIRED_EDGE_FEATURES.copy()  # type: ignore[return-value]


def process_complete_windows(
    complete_df: pd.DataFrame,
    graph_count: int,
    src_col: str,
    dst_col: str,
    edge_feat_cols: Sequence[str],
    label_mode: str,
    source_csv: Path,
    source_folder: str,
    out_dir: Path,
    metadata_path: Path,
    class_counts: Counter,
    max_edges: int,
) -> int:
    for _, gdf in complete_df.groupby("__window", sort=True):
        sample = build_ratio_window_graph(gdf, graph_count, src_col, dst_col, edge_feat_cols, label_mode, source_csv, source_folder)
        min_edges = BENIGN_MIN_EDGES if sample.graph_label == 0 else ATTACK_MIN_EDGES
        sample = apply_edge_limits(sample, min_edges, max_edges)  # type: ignore[arg-type]
        if sample is None:
            continue
        assert_graph_shapes(sample)  # type: ignore[arg-type]
        save_graph_npz(sample, out_dir, label_mode)
        append_metadata_row(metadata_path, sample)
        class_counts[sample.graph_label] += 1
        graph_count += 1
    return graph_count


def process_csv(
    input_csv: Path,
    source_folder: str,
    out_dir: Path,
    metadata_path: Path,
    graph_count: int,
    class_counts: Counter,
    args: argparse.Namespace,
) -> int:
    time_col, src_col, dst_col, _proto_col, label_col, edge_feat_cols = validate_columns(input_csv, args.allow_missing_label)
    folder_is_benign = source_folder == BENIGN_FOLDER_NAME
    carry_window_df = pd.DataFrame()
    total_rows_read = 0

    for chunk in pd.read_csv(input_csv, chunksize=args.chunk_size):
        total_rows_read += len(chunk)
        cleaned_chunk = normalize_chunk(
            chunk,
            time_col,
            src_col,
            dst_col,
            edge_feat_cols,
            args.window_size,
            label_col,
            folder_is_benign,
        )
        if cleaned_chunk.empty and carry_window_df.empty:
            continue
        current_df = pd.concat([carry_window_df, cleaned_chunk], ignore_index=True)
        if current_df.empty:
            continue
        windows = current_df["__window"].to_numpy()
        last_window = windows[-1]
        complete_df = current_df.loc[windows != last_window]
        carry_window_df = current_df.loc[windows == last_window].copy()
        if not complete_df.empty:
            graph_count = process_complete_windows(
                complete_df,
                graph_count,
                src_col,
                dst_col,
                edge_feat_cols,
                args.label_mode,
                input_csv,
                source_folder,
                out_dir,
                metadata_path,
                class_counts,
                args.max_edges,
            )

    if not carry_window_df.empty:
        graph_count = process_complete_windows(
            carry_window_df,
            graph_count,
            src_col,
            dst_col,
            edge_feat_cols,
            args.label_mode,
            input_csv,
            source_folder,
            out_dir,
            metadata_path,
            class_counts,
            args.max_edges,
        )
    print(f"[OK] processed {input_csv} rows={total_rows_read} graphs_so_far={graph_count}", flush=True)
    return graph_count


def write_metadata_json(args: argparse.Namespace, out_dir: Path, graph_count: int, class_counts: Counter, files_by_folder: Dict[str, int]) -> None:
    names = class_names(args.label_mode)
    payload = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(out_dir),
        "window_seconds": args.window_size,
        "label_mode": args.label_mode,
        "class_names": names,
        "class_counts": {str(label): int(class_counts.get(label, 0)) for label in sorted(names)},
        "num_graphs": graph_count,
        "max_files_per_class": args.max_files_per_class,
        "files_processed_by_folder": files_by_folder,
        "node_feature_columns": FIXED_NODE_FEATURES,
        "edge_feature_columns": REQUIRED_EDGE_FEATURES,
        "edge_aggregation": "mean numeric flow features per directed src-dst IP pair",
        "metadata_csv": "graph_window_metadata.csv",
        "summary_csv": "graph_window_summary.csv",
        "benign_folder_name": BENIGN_FOLDER_NAME,
        "benign_min_edges": BENIGN_MIN_EDGES,
        "attack_min_edges": ATTACK_MIN_EDGES,
        "max_edges": args.max_edges,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(payload, f, indent=2)


def print_class_counts(class_counts: Counter, label_mode: str) -> None:
    print("[COUNTS] generated graph classes:", flush=True)
    for label, name in class_names(label_mode).items():
        print(f"  {label} ({name}): {int(class_counts.get(label, 0))}", flush=True)


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if not args.dataset_root.exists() or not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {args.dataset_root}")

    out_dir = args.output_dir
    prepare_output_dir(out_dir)
    metadata_path = out_dir / "graph_window_metadata.csv"
    summary_path = out_dir / "graph_window_summary.csv"
    class_counts: Counter = Counter()
    files_by_folder: Dict[str, int] = defaultdict(int)
    graph_count = 0

    folders = dataset_folders(args.dataset_root, out_dir)
    if not folders:
        raise RuntimeError(f"No class folders found in {args.dataset_root}")

    for folder in folders:
        csv_paths = csv_files_for_folder(folder, args.max_files_per_class)
        if not csv_paths:
            print(f"[WARN] no CSV files found in {folder}", flush=True)
            continue
        for input_csv in csv_paths:
            graph_count = process_csv(input_csv, folder.name, out_dir, metadata_path, graph_count, class_counts, args)
            files_by_folder[folder.name] += 1

    if metadata_path.exists():
        summary_path.write_text(metadata_path.read_text())
    write_metadata_json(args, out_dir, graph_count, class_counts, dict(files_by_folder))
    print_class_counts(class_counts, args.label_mode)
    print(f"[OK] wrote {graph_count} malicious-ratio graph samples to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
