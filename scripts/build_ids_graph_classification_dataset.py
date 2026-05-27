#!/usr/bin/env python3
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

    proto_vals = _numeric(dfw, proto_col).astype(np.int64)
    unique_proto = sorted({int(v) for v in proto_vals})
    proto_to_col = {p: i for i, p in enumerate(unique_proto)}
    proto_counts = np.zeros((n, len(unique_proto)), dtype=np.float32)
    for edge_i, p in enumerate(proto_vals):
        col_i = proto_to_col[int(p)]
        proto_counts[src_idx[edge_i], col_i] += 1.0
        proto_counts[dst_idx[edge_i], col_i] += 1.0

    node_features = np.concatenate(
        [
            deg_in[:, None],
            deg_out[:, None],
            total_in_bytes[:, None],
            total_out_bytes[:, None],
            total_in_packets[:, None],
            total_out_packets[:, None],
            mean_flow_duration[:, None],
            proto_counts,
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


def main() -> None:
    input_csv = "datasets/NF-BoT-IoT.csv"
    output_dir = "datasets/IDS_GRAPH_BENCHMARK"
    window_seconds = 300
    max_rows = None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, nrows=max_rows)
    cols = list(df.columns)
    time_col = find_col(cols, TIME_CANDIDATES)
    src_col = find_col(cols, SRC_IP_CANDIDATES)
    dst_col = find_col(cols, DST_IP_CANDIDATES)
    label_col = find_col(cols, LABEL_CANDIDATES)
    proto_col = find_col(cols, PROTO_CANDIDATES)
    attack_type_col = find_col(cols, ATTACK_TYPE_CANDIDATES)

    missing = [name for name, c in [("time", time_col), ("src_ip", src_col), ("dst_ip", dst_col), ("label", label_col), ("protocol", proto_col)] if c is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {cols}")

    missing_edge_cols = [c for c in REQUIRED_EDGE_FEATURES if c not in df.columns]
    if missing_edge_cols:
        raise ValueError(f"Missing required edge feature columns: {missing_edge_cols}")

    df["__time"] = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["__time", src_col, dst_col]).sort_values("__time").reset_index(drop=True)

    edge_feat_cols = REQUIRED_EDGE_FEATURES.copy()
    for c in edge_feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    epoch = (df["__time"].astype("int64") // 10**9).astype(np.int64)
    df["__window"] = (epoch // window_seconds).astype(np.int64)

    graphs: List[GraphSample] = []
    summary = []
    for gid, (_, gdf) in enumerate(df.groupby("__window", sort=True)):
        sample = build_window_graph(gdf, gid, src_col, dst_col, proto_col, label_col, attack_type_col, edge_feat_cols)
        graphs.append(sample)
        summary.append(
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

    np.save(out_dir / "graphs.npy", np.array(graphs, dtype=object), allow_pickle=True)
    pd.DataFrame(summary).to_csv(out_dir / "graph_window_summary.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "edge_feature_columns": edge_feat_cols,
                "num_graphs": len(graphs),
                "input_csv": input_csv,
                "window_seconds": window_seconds,
            },
            f,
            indent=2,
        )
    print(f"wrote {len(graphs)} graph samples to {out_dir}")


if __name__ == "__main__":
    main()
