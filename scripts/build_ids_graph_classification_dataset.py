#!/usr/bin/env python3
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "flow_start_milliseconds",
    "flow_start_ms",
    "timestamp",
    "time",
    "ts",
    "flow_start",
    "bidirectional_first_seen_ms",
]
SRC_IP_CANDIDATES = ["ipv4_src_addr", "src_ip", "srcip", "src_ip_addr", "sourceip"]
DST_IP_CANDIDATES = ["ipv4_dst_addr", "dst_ip", "dstip", "dst_ip_addr", "destinationip"]
PROTO_CANDIDATES = ["protocol", "proto", "ip_protocol"]
LABEL_CANDIDATES = [
    "label",
    "attack",
    "attacktype",
    "attack_cat",
    "binary_label",
    "is_attack",
    "class",
]
ATTACK_TYPE_CANDIDATES = ["attack_type", "attacktype", "attack_cat", "category", "traffic_type", "sub_label"]

REQUIRED_EDGE_FEATURES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "PROTOCOL",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "TCP_FLAGS",
]
FIXED_NODE_FEATURES = [
    "in_degree",
    "out_degree",
    "total_in_bytes",
    "total_out_bytes",
    "total_in_packets",
    "total_out_packets",
    "mean_flow_duration",
    "total_flow_count",
]

BENIGN_MIN_EDGES = 1
ATTACK_MIN_EDGES = 5
MAX_EDGES = 3000
MIN_BENIGN_GRAPHS = 100
MIN_ATTACK_GRAPHS = 100
MAX_BENIGN_GRAPHS = 200
MAX_ATTACK_GRAPHS = 200
CHUNK_SIZE = 500


@dataclass
class GraphSample:
    graph_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    graph_label: int
    attack_type: str
    num_benign_flows: int
    num_attack_flows: int


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lc_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lc_map:
            return lc_map[cand]
    return None


def infer_attack(y) -> int:
    if pd.isna(y):
        return 0
    s = str(y).strip().lower()
    if s in {"1", "true", "attack", "malicious", "anomaly", "ddos", "dos", "scan", "bot", "botnet"}:
        return 1
    if s in {"0", "false", "benign", "normal"}:
        return 0
    try:
        return 1 if float(s) > 0 else 0
    except ValueError:
        return 0 if "benign" in s or "normal" in s else 1


def _numeric(dfw: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(dfw[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def build_window_graph(
    dfw: pd.DataFrame,
    graph_id: int,
    src_col: str,
    dst_col: str,
    proto_col: str,
    label_col: str,
    attack_type_col: Optional[str],
    edge_feat_cols: List[str],
) -> GraphSample:
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

    attack_flags = dfw[label_col].map(infer_attack).to_numpy(dtype=np.int32)
    num_attack = int(attack_flags.sum())
    num_benign = int(len(attack_flags) - num_attack)
    graph_label = 1 if num_attack > 0 else 0

    attack_type = "benign"
    if graph_label == 1:
        if attack_type_col and attack_type_col in dfw.columns:
            sub = dfw.loc[attack_flags == 1, attack_type_col].astype(str)
            attack_type = sub.mode().iloc[0] if not sub.empty else "attack"
        else:
            attack_type = "attack"

    return GraphSample(
        graph_id=graph_id,
        start_time=dfw["__time"].iloc[0],
        end_time=dfw["__time"].iloc[-1],
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        graph_label=graph_label,
        attack_type=attack_type,
        num_benign_flows=num_benign,
        num_attack_flows=num_attack,
    )


def assert_graph_shapes(sample: GraphSample) -> None:
    if sample.node_features.shape[1] != 8 or sample.edge_features.shape[1] != 9 or sample.edge_index.shape[0] != 2:
        print(
            f"graph_id={sample.graph_id} "
            f"node_features.shape={sample.node_features.shape} "
            f"edge_features.shape={sample.edge_features.shape} "
            f"edge_index.shape={sample.edge_index.shape}",
            flush=True,
        )
        assert sample.node_features.shape[1] == 8
        assert sample.edge_features.shape[1] == 9
        assert sample.edge_index.shape[0] == 2


def apply_edge_limits(sample: GraphSample, min_edges: int, max_edges: int) -> Optional[GraphSample]:
    num_edges = sample.edge_index.shape[1]
    if num_edges < min_edges:
        return None
    if num_edges > max_edges:
        keep_idx = np.random.choice(num_edges, size=max_edges, replace=False)
        keep_idx.sort()
        sample.edge_index = sample.edge_index[:, keep_idx]
        sample.edge_features = sample.edge_features[keep_idx]
    return sample


def save_graph_npz(sample: GraphSample, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / f"graph_{sample.graph_id:06d}.npz",
        node_features=sample.node_features,
        edge_index=sample.edge_index,
        edge_features=sample.edge_features,
        label=np.array([sample.graph_label], dtype=np.int64),
        attack_type=np.array([sample.attack_type], dtype=np.str_),
    )


def append_summary_row(summary_path: Path, sample: GraphSample) -> None:
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "start_time",
                "end_time",
                "num_nodes",
                "num_edges",
                "num_benign_flows",
                "num_attack_flows",
                "graph_label",
                "attack_type",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "graph_id": sample.graph_id,
                "start_time": sample.start_time.isoformat(),
                "end_time": sample.end_time.isoformat(),
                "num_nodes": sample.node_features.shape[0],
                "num_edges": sample.edge_index.shape[1],
                "num_benign_flows": sample.num_benign_flows,
                "num_attack_flows": sample.num_attack_flows,
                "graph_label": sample.graph_label,
                "attack_type": sample.attack_type,
            }
        )


def normalize_chunk(
    chunk: pd.DataFrame,
    time_col: str,
    src_col: str,
    dst_col: str,
    edge_feat_cols: List[str],
    window_seconds: int,
) -> pd.DataFrame:
    df = chunk.copy()
    df["__time"] = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["__time", src_col, dst_col]).sort_values("__time").reset_index(drop=True)
    for c in edge_feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    epoch = (df["__time"].astype("int64") // 10**9).astype(np.int64)
    df["__window"] = (epoch // window_seconds).astype(np.int64)
    return df


def main() -> None:
    input_csv = "/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/NF-BoT-IoT-v3.csv"
    output_dir = "/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph_15s_balanced"
    window_seconds = 15

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "graph_window_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    header_df = pd.read_csv(input_csv, nrows=1)
    cols = list(header_df.columns)
    time_col = find_col(cols, TIME_CANDIDATES)
    src_col = find_col(cols, SRC_IP_CANDIDATES)
    dst_col = find_col(cols, DST_IP_CANDIDATES)
    label_col = find_col(cols, LABEL_CANDIDATES)
    proto_col = find_col(cols, PROTO_CANDIDATES)
    attack_type_col = find_col(cols, ATTACK_TYPE_CANDIDATES)

    missing = [name for name, c in [("time", time_col), ("src_ip", src_col), ("dst_ip", dst_col), ("label", label_col), ("protocol", proto_col)] if c is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {cols}")

    missing_edge_cols = [c for c in REQUIRED_EDGE_FEATURES if c not in cols]
    if missing_edge_cols:
        raise ValueError(f"Missing required edge feature columns: {missing_edge_cols}")

    edge_feat_cols = REQUIRED_EDGE_FEATURES.copy()
    carry_window_df = pd.DataFrame()
    total_rows_read = 0
    graph_count = 0
    benign_count = 0
    attack_count = 0

    chunk_iter = pd.read_csv(input_csv, chunksize=CHUNK_SIZE)

    try:
        for chunk_idx, chunk in enumerate(chunk_iter, start=1):
            total_rows_read += len(chunk)
            if total_rows_read % 50000 == 0:
                print(
                    f"rows processed={total_rows_read} benign graphs found={benign_count} attack graphs found={attack_count}",
                    flush=True,
                )

            cleaned_chunk = normalize_chunk(chunk, time_col, src_col, dst_col, edge_feat_cols, window_seconds)
            if cleaned_chunk.empty and carry_window_df.empty:
                continue

            current_df = pd.concat([carry_window_df, cleaned_chunk], ignore_index=True)
            if current_df.empty:
                continue

            windows = current_df["__window"].to_numpy()
            last_window = windows[-1]
            complete_mask = windows != last_window
            complete_df = current_df.loc[complete_mask]
            carry_window_df = current_df.loc[~complete_mask].copy()

            if not complete_df.empty:
                for _, gdf in complete_df.groupby("__window", sort=True):
                    sample = build_window_graph(gdf, graph_count, src_col, dst_col, proto_col, label_col, attack_type_col, edge_feat_cols)
                    assert sample.graph_label == (1 if sample.num_attack_flows > 0 else 0)
                    if sample.graph_label == 0 and sample.num_attack_flows != 0:
                        continue
                    if sample.graph_label == 1 and sample.num_attack_flows <= 0:
                        continue

                    min_edges = ATTACK_MIN_EDGES if sample.graph_label == 1 else BENIGN_MIN_EDGES
                    sample = apply_edge_limits(sample, min_edges, MAX_EDGES)
                    if sample is None:
                        continue

                    if sample.graph_label == 0:
                        if benign_count >= MAX_BENIGN_GRAPHS:
                            continue
                    else:
                        if attack_count >= MAX_ATTACK_GRAPHS:
                            continue

                    assert_graph_shapes(sample)
                    save_graph_npz(sample, out_dir)
                    assert sample.edge_index.shape[1] >= min_edges
                    assert sample.edge_index.shape[1] <= MAX_EDGES
                    assert sample.node_features.shape[1] == 8
                    assert sample.edge_features.shape[1] == 9
                    append_summary_row(summary_path, sample)
                    graph_count += 1
                    if sample.graph_label == 0:
                        benign_count += 1
                    else:
                        attack_count += 1

            if (
                benign_count >= MIN_BENIGN_GRAPHS
                and attack_count >= MIN_ATTACK_GRAPHS
                and benign_count <= MAX_BENIGN_GRAPHS
                and attack_count <= MAX_ATTACK_GRAPHS
            ):
                break

        if not carry_window_df.empty:
            sample = build_window_graph(carry_window_df, graph_count, src_col, dst_col, proto_col, label_col, attack_type_col, edge_feat_cols)
            assert sample.graph_label == (1 if sample.num_attack_flows > 0 else 0)
            if (sample.graph_label == 0 and sample.num_attack_flows == 0) or (sample.graph_label == 1 and sample.num_attack_flows > 0):
                min_edges = ATTACK_MIN_EDGES if sample.graph_label == 1 else BENIGN_MIN_EDGES
                sample = apply_edge_limits(sample, min_edges, MAX_EDGES)
                if sample is not None:
                    can_save = (sample.graph_label == 0 and benign_count < MAX_BENIGN_GRAPHS) or (
                        sample.graph_label == 1 and attack_count < MAX_ATTACK_GRAPHS
                    )
                    if can_save:
                        assert_graph_shapes(sample)
                        save_graph_npz(sample, out_dir)
                        assert sample.edge_index.shape[1] >= min_edges
                        assert sample.edge_index.shape[1] <= MAX_EDGES
                        assert sample.node_features.shape[1] == 8
                        assert sample.edge_features.shape[1] == 9
                        append_summary_row(summary_path, sample)
                        graph_count += 1
                        if sample.graph_label == 0:
                            benign_count += 1
                        else:
                            attack_count += 1

    except Exception as exc:
        print(
            f"ERROR while processing chunk {chunk_idx} at total_rows={total_rows_read}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "node_feature_columns": FIXED_NODE_FEATURES,
                "edge_feature_columns": edge_feat_cols,
                "num_graphs": graph_count,
                "input_csv": input_csv,
                "window_seconds": window_seconds,
                "chunk_size": CHUNK_SIZE,
                "min_benign_graphs": MIN_BENIGN_GRAPHS,
                "min_attack_graphs": MIN_ATTACK_GRAPHS,
                "max_benign_graphs": MAX_BENIGN_GRAPHS,
                "max_attack_graphs": MAX_ATTACK_GRAPHS,
                "benign_min_edges": BENIGN_MIN_EDGES,
                "attack_min_edges": ATTACK_MIN_EDGES,
            },
            f,
            indent=2,
        )
    print(f"wrote {graph_count} graph samples to {out_dir} (benign={benign_count}, attack={attack_count})")


if __name__ == "__main__":
    main()
