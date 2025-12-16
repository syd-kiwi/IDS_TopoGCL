from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def read_header(header_path: Path):
    txt = header_path.read_text(encoding="utf-8").strip()
    # header may be comma separated on one line
    cols = [c.strip() for c in txt.split(",") if c.strip()]
    if len(cols) < 3:
        # fallback
        cols = ["time", "source", "resolved"]
    # normalize to safe names
    cols = cols[:3]
    cols = ["time", "source", "resolved"]
    return cols

def coerce_time(df: pd.DataFrame):
    # Try numeric first
    df["time_raw"] = df["time"]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    if df["time"].notna().mean() > 0.9:
        # likely epoch seconds or a counter
        # if values look like epoch seconds, convert
        t = df["time"].dropna()
        if t.min() > 1e9 and t.max() < 3e10:
            df["time_dt"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        else:
            df["time_dt"] = pd.NaT
    else:
        # try datetime parse
        df["time_dt"] = pd.to_datetime(df["time_raw"], errors="coerce")

    return df

def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/dns.txt")
    ap.add_argument("--header", default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/dns_header.txt")
    ap.add_argument("--out", default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data/dns_plots")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()


    data_path = Path(args.data)
    header_path = Path(args.header)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = read_header(header_path)

    df = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        sep=",",
        engine="python"
    )

    df["source"] = df["source"].astype(str).str.strip()
    df["resolved"] = df["resolved"].astype(str).str.strip()
    df = coerce_time(df)

    print(df.head())
    print("rows:", len(df))
    print("unique source:", df["source"].nunique())
    print("unique resolved:", df["resolved"].nunique())

    # 1) Top sources
    top_src = df["source"].value_counts().head(args.topk)
    plt.figure()
    top_src.sort_values().plot(kind="barh")
    plt.title(f"Top {args.topk} source computers")
    plt.xlabel("count")
    save_plot(outdir / "01_top_sources.png")

    # 2) Top resolved
    top_res = df["resolved"].value_counts().head(args.topk)
    plt.figure()
    top_res.sort_values().plot(kind="barh")
    plt.title(f"Top {args.topk} resolved computers")
    plt.xlabel("count")
    save_plot(outdir / "02_top_resolved.png")

    # 3) Top pairs
    top_pairs = df.groupby(["source", "resolved"]).size().sort_values(ascending=False).head(args.topk)
    plt.figure()
    top_pairs.iloc[::-1].plot(kind="barh")
    plt.title(f"Top {args.topk} source to resolved pairs")
    plt.xlabel("count")
    save_plot(outdir / "03_top_pairs.png")

    # 4) Fan out per source: how many unique resolved per source
    fanout = df.groupby("source")["resolved"].nunique().sort_values(ascending=False).head(args.topk)
    plt.figure()
    fanout.sort_values().plot(kind="barh")
    plt.title(f"Top {args.topk} sources by unique resolved count")
    plt.xlabel("unique resolved")
    save_plot(outdir / "04_fanout_unique_resolved.png")

    # 5) Fan in per resolved: how many unique sources query it
    fanin = df.groupby("resolved")["source"].nunique().sort_values(ascending=False).head(args.topk)
    plt.figure()
    fanin.sort_values().plot(kind="barh")
    plt.title(f"Top {args.topk} resolved by unique source count")
    plt.xlabel("unique sources")
    save_plot(outdir / "05_fanin_unique_sources.png")

    # 6) Time series: events per bucket (only if we have a real datetime)
    if df["time_dt"].notna().mean() > 0.5:
        ts = df.set_index("time_dt").sort_index()

        for rule, name in [("1min", "minute"), ("1H", "hour")]:
            counts = ts["source"].resample(rule).size()
            plt.figure()
            counts.plot()
            plt.title(f"DNS events per {name}")
            plt.xlabel("time")
            plt.ylabel("events")
            save_plot(outdir / f"06_events_per_{name}.png")

        # 7) New resolved over time (first seen)
        first_seen = ts.reset_index().groupby("resolved")["time_dt"].min().sort_values()
        new_per_hour = first_seen.dt.floor("1H").value_counts().sort_index()
        plt.figure()
        new_per_hour.plot()
        plt.title("New resolved computers first seen per hour")
        plt.xlabel("time")
        plt.ylabel("new resolved")
        save_plot(outdir / "07_new_resolved_per_hour.png")
    else:
        # If time is numeric, still do a simple volume over time by binning
        t = pd.to_numeric(df["time"], errors="coerce").dropna()
        if len(t) > 0:
            df2 = df[df["time"].notna()].copy()
            df2["time_bin"] = pd.cut(df2["time"], bins=50)
            counts = df2.groupby("time_bin").size()
            plt.figure()
            counts.plot(kind="bar")
            plt.title("DNS events over binned time")
            plt.xlabel("time bins")
            plt.ylabel("events")
            save_plot(outdir / "06_events_binned_time.png")

    print(f"Saved plots in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
