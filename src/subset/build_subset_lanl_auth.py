import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_header(header_path: Path) -> List[str]:
    txt = header_path.read_text(encoding="utf-8", errors="ignore").strip()
    cols = [c.strip() for c in txt.split(",") if c.strip()]
    return cols


def pick_hosts_from_buckets(host_counts: pd.Series, per_bucket: int, seed: int) -> List[str]:
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

    def sample_index(s: pd.Series) -> List[str]:
        if len(s) == 0:
            return []
        k = min(per_bucket, len(s))
        return list(rng.choice(s.index.to_numpy(), size=k, replace=False))

    selected = sample_index(low) + sample_index(mid) + sample_index(high)

    # unique while preserving order
    seen = set()
    out = []
    for h in selected:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth_dir", required=True, help="Folder containing auth.txt and auth_header.txt")
    ap.add_argument("--out_dir", required=True, help="Output folder for subset")
    ap.add_argument("--days", type=int, default=7, help="Number of days from min timestamp")
    ap.add_argument("--per_bucket", type=int, default=20, help="Hosts per bucket (low/mid/high)")
    ap.add_argument("--chunksize", type=int, default=300_000, help="Rows per chunk")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    auth_dir = Path(args.auth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = auth_dir / "auth.txt"
    header_path = auth_dir / "auth_header.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    cols: Optional[List[str]] = None
    if header_path.exists():
        cols = read_header(header_path)
        if len(cols) == 0:
            cols = None

    # Column names based on known header
    # If header file exists, these should match exactly.
    # If not, we assume the CSV column order matches the header shown.
    time_col = "time"
    src_comp_col = "source computer"
    dst_comp_col = "destination computer"

    # ---------------- Pass 1: min_time + host counts ----------------
    min_time = None
    total_rows = 0
    host_counts_dict: Dict[str, int] = {}

    reader = pd.read_csv(
        data_path,
        header=None if cols else "infer",
        names=cols,
        sep=",",
        chunksize=args.chunksize,
        engine="python",
        low_memory=True,
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # If no header file, normalize expected columns by position
        if cols is None and i == 0:
            # Force our expected names on the first 9 columns
            expected = [
                "time",
                "source user@domain",
                "destination user@domain",
                "source computer",
                "destination computer",
                "authentication type",
                "logon type",
                "authentication orientation",
                "success/failure",
            ]
            # Only rename if the chunk has enough columns
            if len(chunk.columns) >= len(expected):
                chunk.columns = expected + list(chunk.columns[len(expected):])

        # Time
        t = pd.to_numeric(chunk[time_col], errors="coerce")
        if t.notna().any():
            cmin = int(t.min())
            min_time = cmin if min_time is None else min(min_time, cmin)

        # Host counts from BOTH computers
        s = chunk[src_comp_col].astype(str)
        d = chunk[dst_comp_col].astype(str)

        for v, c in s.value_counts().items():
            host_counts_dict[v] = host_counts_dict.get(v, 0) + int(c)
        for v, c in d.value_counts().items():
            host_counts_dict[v] = host_counts_dict.get(v, 0) + int(c)

        if (i + 1) % 5 == 0:
            print(f"[+] Pass1 chunks {i+1} rows {total_rows:,}  min_time={min_time}")

    if min_time is None:
        raise RuntimeError("Could not determine min_time from auth.txt")

    host_counts = pd.Series(host_counts_dict)
    selected_hosts = pick_hosts_from_buckets(host_counts, args.per_bucket, args.seed)
    selected_set = set(selected_hosts)

    max_time = min_time + args.days * 24 * 60 * 60

    print(f"[+] Found hosts: {len(host_counts):,}")
    print(f"[+] Selected hosts: {len(selected_hosts):,}")
    print(f"[+] Time slice: [{min_time}, {max_time}) days={args.days}")

    # Save host list now
    (out_dir / "selected_hosts.txt").write_text("\n".join(selected_hosts), encoding="utf-8")

    # ---------------- Pass 2: filter + write ----------------
    out_file = out_dir / f"lanl_auth_subset_{args.days}d_{len(selected_hosts)}hosts.csv.gz"
    kept = 0
    first = True

    reader2 = pd.read_csv(
        data_path,
        header=None if cols else "infer",
        names=cols,
        sep=",",
        chunksize=args.chunksize,
        engine="python",
        low_memory=True,
    )

    for j, chunk in enumerate(reader2):
        # If no header file, normalize expected columns by position again
        if cols is None and j == 0:
            expected = [
                "time",
                "source user@domain",
                "destination user@domain",
                "source computer",
                "destination computer",
                "authentication type",
                "logon type",
                "authentication orientation",
                "success/failure",
            ]
            if len(chunk.columns) >= len(expected):
                chunk.columns = expected + list(chunk.columns[len(expected):])

        chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk = chunk.dropna(subset=[time_col])
        if chunk.empty:
            continue

        chunk[time_col] = chunk[time_col].astype(np.int64)
        chunk = chunk[(chunk[time_col] >= min_time) & (chunk[time_col] < max_time)]
        if chunk.empty:
            continue

        s = chunk[src_comp_col].astype(str)
        d = chunk[dst_comp_col].astype(str)

        chunk = chunk[s.isin(selected_set) | d.isin(selected_set)]
        if chunk.empty:
            continue

        kept += len(chunk)

        chunk.to_csv(
            out_file,
            mode="wt" if first else "at",
            index=False,
            header=first,
            compression="gzip",
        )
        first = False

        if (j + 1) % 5 == 0:
            print(f"[+] Pass2 chunks {j+1} kept {kept:,}")

    meta = {
        "input": str(data_path),
        "auth_dir": str(auth_dir),
        "days": args.days,
        "min_time": int(min_time),
        "max_time": int(max_time),
        "per_bucket": args.per_bucket,
        "selected_hosts": int(len(selected_hosts)),
        "kept_rows": int(kept),
        "total_rows_seen": int(total_rows),
        "seed": args.seed,
        "output": str(out_file),
        "host_cols_used": [src_comp_col, dst_comp_col],
    }
    (out_dir / "subset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[+] Wrote: {out_file}")
    print(f"[+] Kept rows: {kept:,}")
    print(f"[+] Meta: {out_dir / 'subset_meta.json'}")


if __name__ == "__main__":
    main()