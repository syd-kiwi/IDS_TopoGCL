#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_CANDIDATES = ["timestamp", "time", "ts", "flow_start", "bidirectional_first_seen_ms"]
SRC_IP_CANDIDATES = ["src_ip", "srcip", "src_ip_addr", "ipv4_src_addr", "sourceip"]
DST_IP_CANDIDATES = ["dst_ip", "dstip", "dst_ip_addr", "ipv4_dst_addr", "destinationip"]
PROTO_CANDIDATES = ["protocol", "proto", "ip_protocol"]
LABEL_CANDIDATES = ["label", "attack", "is_attack", "binary_label", "class"]
ATTACK_TYPE_CANDIDATES = ["attack_type", "category", "traffic_type", "sub_label"]


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


def build_window_graph(dfw: pd.DataFrame, graph_id: int, src_col: str, dst_col: str, proto_col: Optional[str], label_col: str, attack_type_col: Optional[str], edge_feat_cols: List[str]) -> GraphSample:
    ips = pd.unique(pd.concat([dfw[src_col].astype(str), dfw[dst_col].astype(str)], ignore_index=True))
    ip_to_idx = {ip: i for i, ip in enumerate(ips)}
    n = len(ips)

    src_idx = dfw[src_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    dst_idx = dfw[dst_col].astype(str).map(ip_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.stack([src_idx, dst_idx], axis=0)

    edge_features = dfw[edge_feat_cols].to_numpy(dtype=np.float32)

    deg_in = np.zeros(n, dtype=np.float32)
    deg_out = np.zeros(n, dtype=np.float32)
    np.add.at(deg_out, src_idx, 1.0)
    np.add.at(deg_in, dst_idx, 1.0)

    bytes_col = next((c for c in ["in_bytes", "tot_bytes", "bytes", "bidirectional_bytes"] if c in dfw.columns), None)
    pkts_col = next((c for c in ["in_pkts", "tot_pkts", "packets", "bidirectional_packets"] if c in dfw.columns), None)
    total_bytes = np.zeros(n, dtype=np.float32)
    total_pkts = np.zeros(n, dtype=np.float32)
    if bytes_col:
        vals = pd.to_numeric(dfw[bytes_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        np.add.at(total_bytes, src_idx, vals)
        np.add.at(total_bytes, dst_idx, vals)
    if pkts_col:
        vals = pd.to_numeric(dfw[pkts_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        np.add.at(total_pkts, src_idx, vals)
        np.add.at(total_pkts, dst_idx, vals)

    proto_counts = np.zeros((n, 3), dtype=np.float32)
    if proto_col:
        p = dfw[proto_col].astype(str).str.lower()
        tcp = p.str.contains("tcp").to_numpy(dtype=np.float32)
        udp = p.str.contains("udp").to_numpy(dtype=np.float32)
        other = (1.0 - np.clip(tcp + udp, 0, 1)).astype(np.float32)
        np.add.at(proto_counts[:, 0], src_idx, tcp)
        np.add.at(proto_counts[:, 1], src_idx, udp)
        np.add.at(proto_counts[:, 2], src_idx, other)

    node_features = np.concatenate([
        deg_in[:, None], deg_out[:, None], total_bytes[:, None], total_pkts[:, None], proto_counts
    ], axis=1).astype(np.float32)

    attack_flags = dfw[label_col].map(infer_attack).to_numpy(dtype=np.int32)
    num_attack = int(attack_flags.sum())
    num_benign = int(len(attack_flags) - num_attack)
    graph_label = 1 if num_attack > 0 else 0

    attack_type = "benign"
    if graph_label == 1 and attack_type_col:
        sub = dfw.loc[attack_flags == 1, attack_type_col].astype(str)
        attack_type = sub.mode().iloc[0] if not sub.empty else "attack"

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
    ap = argparse.ArgumentParser(description="Build IDS graph classification benchmark from NetFlow CSV.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", default="datasets/IDS_GRAPH_BENCHMARK")
    ap.add_argument("--window_seconds", type=int, default=300)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, nrows=args.max_rows)
    cols = list(df.columns)
    time_col = find_col(cols, TIME_CANDIDATES)
    src_col = find_col(cols, SRC_IP_CANDIDATES)
    dst_col = find_col(cols, DST_IP_CANDIDATES)
    label_col = find_col(cols, LABEL_CANDIDATES)
    proto_col = find_col(cols, PROTO_CANDIDATES)
    attack_type_col = find_col(cols, ATTACK_TYPE_CANDIDATES)

    missing = [name for name, c in [("time", time_col), ("src_ip", src_col), ("dst_ip", dst_col), ("label", label_col)] if c is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {cols}")

    df["__time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["__time", src_col, dst_col]).sort_values("__time").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c not in {time_col, src_col, dst_col, label_col, attack_type_col, "__time"}]
    edge_feat_cols = []
    for c in numeric_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().any():
            df[c] = v.fillna(0.0)
            edge_feat_cols.append(c)

    if not edge_feat_cols:
        raise ValueError("No numeric NetFlow feature columns found for edge features.")

    epoch = (df["__time"].view("int64") // 10**9).astype(np.int64)
    df["__window"] = (epoch // args.window_seconds).astype(np.int64)

    graphs: List[GraphSample] = []
    summary = []
    for gid, (_, gdf) in enumerate(df.groupby("__window", sort=True)):
        sample = build_window_graph(gdf, gid, src_col, dst_col, proto_col, label_col, attack_type_col, edge_feat_cols)
        graphs.append(sample)
        summary.append({
            "graph_id": sample.graph_id,
            "start_time": sample.start_time.isoformat(),
            "end_time": sample.end_time.isoformat(),
            "num_nodes": sample.node_features.shape[0],
            "num_edges": sample.edge_index.shape[1],
            "num_benign_flows": sample.num_benign_flows,
            "num_attack_flows": sample.num_attack_flows,
            "graph_label": sample.graph_label,
            "attack_type": sample.attack_type,
        })

    np.save(out_dir / "graphs.npy", np.array(graphs, dtype=object), allow_pickle=True)
    pd.DataFrame(summary).to_csv(out_dir / "graph_summary.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"edge_feature_columns": edge_feat_cols, "num_graphs": len(graphs)}, f, indent=2)
    print(f"wrote {len(graphs)} graph samples to {out_dir}")


if __name__ == "__main__":
    main()
