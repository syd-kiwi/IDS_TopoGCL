#!/usr/bin/env python3
"""
Make a moderate sized LANL flows subset.

What it does
1) Finds a flows file in a directory (txt, csv, gz supported)
2) First pass: counts events per host and finds the min timestamp
3) Selects hosts from low, mid, high volume buckets
4) Second pass: filters rows by time slice and selected hosts
5) Writes subset csv.gz plus metadata

Assumptions
- The first column is time (int)
- There are at least two columns for source and destination hosts
  (we try to detect which ones)
"""

import argparse
import numpy as np
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def find_data_file(data_dir: Path) -> Path:
    candidates = []
    for pat in ["*.csv", "*.txt", "*.log", "*.csv.gz", "*.txt.gz", "*.log.gz", "*.gz"]:
        candidates.extend(sorted(data_dir.glob(pat)))
    if not candidates:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    # Prefer non header files
    for c in candidates:
        if "header" not in c.name.lower():
            return c
    return candidates[0]


def load_header_names(data_dir: Path) -> Optional[List[str]]:
    # Common LANL pattern
    for name in ["flows_header.txt", "flow_header.txt", "netflow_header.txt", "header.txt"]:
        p = data_dir / name
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            # header could be comma separated or whitespace separated
            if "," in txt:
                return [x.strip() for x in txt.split(",") if x.strip()]
            return [x.strip() for x in txt.split() if x.strip()]
    return None


def detect_src_dst_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    # If header exists, try common names
    lowered = {c: str(c).lower() for c in cols}
    preferred_src = ["src", "source", "srccomp", "srcip", "sourcecomputer", "source_host", "scomputer"]
    preferred_dst = ["dst", "dest", "destination", "dstcomp", "dstip", "computerresolved", "dest_host", "dcomputer"]

    src_col = None
    dst_col = None

    for c in cols:
        if lowered[c] in preferred_src:
            src_col = c
            break
    for c in cols:
        if lowered[c] in preferred_dst:
            dst_col = c
            break

    # Fallback: assume col1 and col2 after time
    if src_col is None or dst_col is None:
        # time is column 0
        if len(cols) < 3:
            raise ValueError("Not enough columns to detect source and destination")
        src_col = cols[1]
        dst_col = cols[2]

    return src_col, dst_col


def read_in_chunks(path: Path, names: Optional[List[str]], chunksize: int) -> pd.io.parsers.TextFileReader:
    # Auto detect delimiter: LANL files are often comma separated
    # Use python engine with sep=None for robust detection, then lock on comma if it works
    # But sep=None can be slower. We will assume comma first, fallback to whitespace.
    compression = "gzip" if path.suffix == ".gz" else None

    try:
        return pd.read_csv(
            path,
            header=None if names else "infer",
            names=names,
            sep=",",
            compression=compression,
            chunksize=chunksize,
            low_memory=True,
        )
    except Exception:
        return pd.read_csv(
            path,
            header=None if names else "infer",
            names=names,
            sep=r"\s+",
            compression=compression,
            chunksize=chunksize,
            low_memory=True,
            engine="python",
        )


def pick_hosts_from_buckets(host_counts: pd.Series, per_bucket: int, seed: int) -> List[str]:
    # host_counts index are host ids, values are counts
    host_counts = host_counts.sort_values()
    n = len(host_counts)
    if n == 0:
        return []

    # Define bucket ranges by quantiles
    q1 = int(0.33 * n)
    q2 = int(0.66 * n)

    low = host_counts.iloc[:q1] if q1 > 0 else host_counts.iloc[: max(1, n // 3)]
    mid = host_counts.iloc[q1:q2] if q2 > q1 else host_counts.iloc[:0]
    high = host_counts.iloc[q2:] if q2 < n else host_counts.iloc[max(0, n - max(1, n // 3)) :]

    rng = np.random.RandomState(seed)


    def sample_index(s: pd.Series) -> List[str]:
        if len(s) == 0:
            return []
        k = min(per_bucket, len(s))
        return list(rng.choice(s.index.to_numpy(), size=k, replace=False))

    selected = sample_index(low) + sample_index(mid) + sample_index(high)
    # Make unique while preserving order
    seen = set()
    out = []
    for h in selected:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_dir", required=True, help="Path to LANL FLOWS folder")
    ap.add_argument("--out_dir", default="subset_out", help="Output folder")
    ap.add_argument("--days", type=int, default=7, help="Number of days from start time")
    ap.add_argument("--per_bucket", type=int, default=20, help="Hosts per volume bucket (low mid high)")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per chunk")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    flows_dir = Path(args.flows_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = find_data_file(flows_dir)
    header_names = load_header_names(flows_dir)

    print(f"[+] Using file: {data_file}")
    if header_names:
        print(f"[+] Using header names from file, columns: {len(header_names)}")
    else:
        print("[+] No header file found, will infer columns")

    # -------- Pass 1: find min time and host counts --------
    min_time = None
    total_rows = 0

    src_counts: Dict[str, int] = {}
    dst_counts: Dict[str, int] = {}

    reader = read_in_chunks(data_file, header_names, args.chunksize)
    src_col = None
    dst_col = None
    time_col = None

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Detect columns on first chunk
        if i == 0:
            cols = list(chunk.columns)
            time_col = cols[0]
            src_col, dst_col = detect_src_dst_columns(chunk)
            print(f"[+] Detected time col: {time_col}, src col: {src_col}, dst col: {dst_col}")

        # time
        t = pd.to_numeric(chunk[time_col], errors="coerce")
        tmin = int(t.min()) if t.notna().any() else None
        if tmin is not None:
            min_time = tmin if min_time is None else min(min_time, tmin)

        # host counts
        s = chunk[src_col].astype(str)
        d = chunk[dst_col].astype(str)

        for val, cnt in s.value_counts().items():
            src_counts[val] = src_counts.get(val, 0) + int(cnt)
        for val, cnt in d.value_counts().items():
            dst_counts[val] = dst_counts.get(val, 0) + int(cnt)

        if (i + 1) % 5 == 0:
            print(f"[+] Pass1 processed chunks: {i+1}, rows: {total_rows:,}")

    if min_time is None:
        raise RuntimeError("Could not determine min_time from file")

    # Combine src and dst counts
    all_hosts = set(src_counts.keys()) | set(dst_counts.keys())
    host_total = {h: src_counts.get(h, 0) + dst_counts.get(h, 0) for h in all_hosts}
    host_counts = pd.Series(host_total).sort_values(ascending=False)

    print(f"[+] Found hosts: {len(host_counts):,}")
    print(f"[+] min_time: {min_time}")

    # Time slice
    seconds = args.days * 24 * 60 * 60
    max_time = min_time + seconds
    print(f"[+] Time slice: [{min_time}, {max_time})  days: {args.days}")

    # Select hosts
    selected_hosts = pick_hosts_from_buckets(host_counts, args.per_bucket, args.seed)
    selected_set = set(selected_hosts)
    print(f"[+] Selected hosts: {len(selected_hosts)}")

    # Save host list
    (out_dir / "selected_hosts.txt").write_text("\n".join(selected_hosts), encoding="utf-8")

    # -------- Pass 2: filter rows --------
    out_path = out_dir / f"lanl_flows_subset_{args.days}d_{len(selected_hosts)}hosts.csv.gz"
    kept = 0

    reader2 = read_in_chunks(data_file, header_names, args.chunksize)

    # Write header once
    first_write = True

    for j, chunk in enumerate(reader2):
        # Ensure same column names
        cols = list(chunk.columns)
        time_col = cols[0] if time_col is None else time_col

        # Filter time
        t = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk = chunk[(t >= min_time) & (t < max_time)]
        if chunk.empty:
            continue

        # Detect src dst if needed
        if src_col is None or dst_col is None:
            src_col, dst_col = detect_src_dst_columns(chunk)

        s = chunk[src_col].astype(str)
        d = chunk[dst_col].astype(str)
        chunk = chunk[s.isin(selected_set) | d.isin(selected_set)]
        if chunk.empty:
            continue

        kept += len(chunk)

        # Write
        chunk.to_csv(
            out_path,
            mode="wt" if first_write else "at",
            index=False,
            header=first_write,
            compression="gzip",
        )
        first_write = False

        if (j + 1) % 5 == 0:
            print(f"[+] Pass2 processed chunks: {j+1}, kept rows: {kept:,}")

    meta = {
        "input_file": str(data_file),
        "flows_dir": str(flows_dir),
        "days": args.days,
        "min_time": int(min_time),
        "max_time": int(max_time),
        "per_bucket": args.per_bucket,
        "selected_hosts": len(selected_hosts),
        "kept_rows": int(kept),
        "seed": args.seed,
        "output_file": str(out_path),
    }
    (out_dir / "subset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[+] Wrote: {out_path}")
    print(f"[+] Kept rows: {kept:,}")
    print(f"[+] Meta: {out_dir / 'subset_meta.json'}")


if __name__ == "__main__":
    main()
