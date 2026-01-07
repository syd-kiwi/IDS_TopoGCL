import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_header(header_path: Path) -> List[str]:
    txt = header_path.read_text(encoding="utf-8", errors="ignore").strip()
    cols = [c.strip() for c in txt.split(",") if c.strip()]
    if len(cols) >= 3:
        return cols[:3]
    return ["time", "source", "resolved"]


def pick_hosts(host_counts: pd.Series, per_bucket: int, seed: int) -> List[str]:
    host_counts = host_counts.sort_values()
    n = len(host_counts)
    if n == 0:
        return []
    q1 = int(0.33 * n)
    q2 = int(0.66 * n)

    low = host_counts.iloc[:q1] if q1 > 0 else host_counts.iloc[: max(1, n // 3)]
    mid = host_counts.iloc[q1:q2] if q2 > q1 else host_counts.iloc[:0]
    high = host_counts.iloc[q2:] if q2 < n else host_counts.iloc[max(0, n - max(1, n // 3)) :]

    rng = np.random.RandomState(seed)

    def sample(s: pd.Series) -> List[str]:
        if len(s) == 0:
            return []
        k = min(per_bucket, len(s))
        return list(rng.choice(s.index.to_numpy(), size=k, replace=False))

    chosen = sample(low) + sample(mid) + sample(high)
    seen, out = set(), []
    for h in chosen:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns_dir", required=True, help="Folder with dns.txt and dns_header.txt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--per_bucket", type=int, default=20)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dns_dir = Path(args.dns_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = dns_dir / "dns.txt"
    header_path = dns_dir / "dns_header.txt"

    cols = read_header(header_path)
    names = ["time", "source", "resolved"]

    # Pass 1: min_time and host counts
    min_time = None
    total_rows = 0
    src_counts: Dict[str, int] = {}
    dst_counts: Dict[str, int] = {}

    for chunk in pd.read_csv(
        data_path,
        header=None,
        names=cols,
        sep=",",
        chunksize=args.chunksize,
        engine="python",
        low_memory=True,
    ):
        total_rows += len(chunk)

        chunk.columns = names
        t = pd.to_numeric(chunk["time"], errors="coerce")
        if t.notna().any():
            cmin = int(t.min())
            min_time = cmin if min_time is None else min(min_time, cmin)

        s = chunk["source"].astype(str).str.strip()
        r = chunk["resolved"].astype(str).str.strip()

        for v, c in s.value_counts().items():
            src_counts[v] = src_counts.get(v, 0) + int(c)
        for v, c in r.value_counts().items():
            dst_counts[v] = dst_counts.get(v, 0) + int(c)

    if min_time is None:
        raise RuntimeError("Could not find min_time in dns.txt")

    all_hosts = set(src_counts) | set(dst_counts)
    host_total = {h: src_counts.get(h, 0) + dst_counts.get(h, 0) for h in all_hosts}
    host_counts = pd.Series(host_total)

    selected = pick_hosts(host_counts, args.per_bucket, args.seed)
    selected_set = set(selected)

    max_time = min_time + args.days * 24 * 60 * 60

    # Pass 2: write subset
    out_file = out_dir / f"lanl_dns_subset_{args.days}d_{len(selected)}hosts.csv.gz"
    first = True
    kept = 0

    for chunk in pd.read_csv(
        data_path,
        header=None,
        names=cols,
        sep=",",
        chunksize=args.chunksize,
        engine="python",
        low_memory=True,
    ):
        chunk.columns = names
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time"])
        chunk["time"] = chunk["time"].astype(np.int64)

        chunk = chunk[(chunk["time"] >= min_time) & (chunk["time"] < max_time)]
        if chunk.empty:
            continue

        chunk["source"] = chunk["source"].astype(str).str.strip()
        chunk["resolved"] = chunk["resolved"].astype(str).str.strip()

        chunk = chunk[chunk["source"].isin(selected_set) | chunk["resolved"].isin(selected_set)]
        if chunk.empty:
            continue

        kept += len(chunk)
        chunk.to_csv(out_file, mode="wt" if first else "at", index=False, header=first, compression="gzip")
        first = False

    (out_dir / "selected_hosts.txt").write_text("\n".join(selected), encoding="utf-8")
    meta = {
        "input": str(data_path),
        "days": args.days,
        "min_time": int(min_time),
        "max_time": int(max_time),
        "per_bucket": args.per_bucket,
        "selected_hosts": len(selected),
        "kept_rows": int(kept),
        "total_rows_seen": int(total_rows),
        "output": str(out_file),
    }
    (out_dir / "subset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] wrote {out_file}")
    print(f"[+] kept rows {kept:,}")


if __name__ == "__main__":
    main()
