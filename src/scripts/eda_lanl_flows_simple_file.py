from pathlib import Path
import argparse
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_file", required=True, help="Path to flows file (txt or csv)")
    ap.add_argument("--out", default="plots/lanl_flows_simple", help="Output folder")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--bin_seconds", type=int, default=3600)
    args = ap.parse_args()

    flows_path = Path(args.flows_file)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    src_counts = Counter()
    dst_counts = Counter()
    bins = Counter()

    min_time = None
    total_rows = 0

    reader = pd.read_csv(
        flows_path,
        header=None,
        #usecols=[0, 1, 2],
        usecols=[0, 2, 4],
        names=["time", "src", "dst"],
        sep=",",
        chunksize=args.chunksize,
        low_memory=True,
    )

    for chunk in reader:
        total_rows += len(chunk)

        t = pd.to_numeric(chunk["time"], errors="coerce").dropna().astype(np.int64)
        s = chunk["src"].astype(str).str.strip()
        d = chunk["dst"].astype(str).str.strip()

        if len(t):
            if min_time is None:
                min_time = int(t.min())
            rel = t - min_time
            b = (rel // args.bin_seconds).to_numpy()
            uniq, cnts = np.unique(b, return_counts=True)
            for bi, ci in zip(uniq, cnts):
                bins[int(bi)] += int(ci)

        src_counts.update(s.to_list())
        dst_counts.update(d.to_list())

        if total_rows % (args.chunksize * 5) == 0:
            print(f"[+] processed rows: {total_rows:,}")

    print(f"[+] total rows: {total_rows:,}")
    print(f"[+] unique src: {len(src_counts):,}")
    print(f"[+] unique dst: {len(dst_counts):,}")

    x = np.array(sorted(bins.keys()), dtype=np.int64)
    y = np.array([bins[i] for i in x], dtype=np.int64)

    plt.figure()
    plt.plot(x, y)
    plt.title("LANL flows per hour")
    plt.xlabel("hours since start")
    plt.ylabel("rows")
    save_plot(outdir / "01_flows_per_hour.png")

    top_src = src_counts.most_common(args.topk)
    labels = [a for a, _ in top_src][::-1]
    vals = [b for _, b in top_src][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals)
    plt.title(f"Top {args.topk} source hosts")
    plt.xlabel("rows")
    plt.ylabel("source")
    save_plot(outdir / "02_top_sources.png")

    top_dst = dst_counts.most_common(args.topk)
    labels = [a for a, _ in top_dst][::-1]
    vals = [b for _, b in top_dst][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals)
    plt.title(f"Top {args.topk} destination hosts")
    plt.xlabel("rows")
    plt.ylabel("destination")
    save_plot(outdir / "03_top_destinations.png")

    print(f"[+] saved plots to: {outdir}")


if __name__ == "__main__":
    main()

