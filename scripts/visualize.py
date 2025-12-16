from pathlib import Path
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_header(header_path: Path):
    txt = header_path.read_text(encoding="utf-8").strip()
    cols = [c.strip() for c in txt.split(",") if c.strip()]
    if len(cols) >= 3:
        return cols[:3]
    return ["time", "source", "resolved"]


def coerce_time(df: pd.DataFrame):
    df["time_raw"] = df["time"]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    if df["time"].notna().mean() > 0.9:
        t = df["time"].dropna()
        if len(t) and t.min() > 1e9 and t.max() < 3e10:
            df["time_dt"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        else:
            df["time_dt"] = pd.NaT
    else:
        df["time_dt"] = pd.to_datetime(df["time_raw"], errors="coerce")

    return df


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/kiwi-pandas/Documents/IDS_TopoGCL/data")
    ap.add_argument("--data", default="dns.txt")
    ap.add_argument("--header", default="dns_header.txt")
    ap.add_argument("--out", default="/home/kiwi-pandas/Documents/IDS_TopoGCL/plots")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / args.data
    header_path = data_dir / args.header
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = read_header(header_path)

    df = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        sep=",",
        engine="python",
    )

    # Normalize expected column names
    df.columns = ["time", "source", "resolved"]

    df["source"] = df["source"].astype(str).str.strip()
    df["resolved"] = df["resolved"].astype(str).str.strip()
    df = coerce_time(df)

    print(df.head())
    print("rows:", len(df))
    print("unique source:", df["source"].nunique())
    print("unique resolved:", df["resolved"].nunique())

    # 1) Time distribution
    plt.figure()
    sns.histplot(df["time"].dropna(), bins=50)
    plt.title("Distribution of time")
    plt.xlabel("time")
    plt.ylabel("count")
    save_plot(outdir / "01_time_hist.png")

    # 2) Events per time
    if df["time_dt"].notna().mean() > 0.5:
        ts = df[["time_dt"]].dropna().copy()
        ts = ts.sort_values("time_dt")
        per_hour = ts.set_index("time_dt").resample("1H").size().reset_index(name="count")

        plt.figure()
        sns.lineplot(data=per_hour, x="time_dt", y="count")
        plt.title("DNS events per hour")
        plt.xlabel("time")
        plt.ylabel("events")
        save_plot(outdir / "02_events_per_hour.png")
    else:
        tmp = df[df["time"].notna()].copy()
        if len(tmp):
            counts_by_t = tmp.groupby("time").size().reset_index(name="count").sort_values("time")

            plt.figure()
            sns.lineplot(data=counts_by_t, x="time", y="count")
            plt.title("DNS events per time")
            plt.xlabel("time")
            plt.ylabel("events")
            save_plot(outdir / "02_events_per_time.png")

    # 3) Top sources
    top_src = df["source"].value_counts().head(args.topk).reset_index()
    top_src.columns = ["source", "count"]

    plt.figure()
    sns.barplot(data=top_src.sort_values("count"), x="count", y="source")
    plt.title(f"Top {args.topk} source computers")
    plt.xlabel("count")
    plt.ylabel("source")
    save_plot(outdir / "03_top_sources.png")

    # 4) Top resolved
    top_res = df["resolved"].value_counts().head(args.topk).reset_index()
    top_res.columns = ["resolved", "count"]

    plt.figure()
    sns.barplot(data=top_res.sort_values("count"), x="count", y="resolved")
    plt.title(f"Top {args.topk} resolved computers")
    plt.xlabel("count")
    plt.ylabel("resolved")
    save_plot(outdir / "04_top_resolved.png")

    # 5) Top pairs
    pairs = (
        df.groupby(["source", "resolved"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .head(args.topk)
    )
    pairs["pair"] = pairs["source"] + " to " + pairs["resolved"]

    plt.figure()
    sns.barplot(data=pairs.sort_values("count"), x="count", y="pair")
    plt.title(f"Top {args.topk} source to resolved pairs")
    plt.xlabel("count")
    plt.ylabel("pair")
    save_plot(outdir / "05_top_pairs.png")

    # 6) Fan out per source
    fanout = (
        df.groupby("source")["resolved"]
          .nunique()
          .sort_values(ascending=False)
          .head(args.topk)
          .reset_index()
    )
    fanout.columns = ["source", "unique_resolved"]

    plt.figure()
    sns.barplot(data=fanout.sort_values("unique_resolved"), x="unique_resolved", y="source")
    plt.title(f"Top {args.topk} sources by unique resolved")
    plt.xlabel("unique resolved")
    plt.ylabel("source")
    save_plot(outdir / "06_fanout_unique_resolved.png")

    # 7) Fan in per resolved
    fanin = (
        df.groupby("resolved")["source"]
          .nunique()
          .sort_values(ascending=False)
          .head(args.topk)
          .reset_index()
    )
    fanin.columns = ["resolved", "unique_sources"]

    plt.figure()
    sns.barplot(data=fanin.sort_values("unique_sources"), x="unique_sources", y="resolved")
    plt.title(f"Top {args.topk} resolved by unique sources")
    plt.xlabel("unique sources")
    plt.ylabel("resolved")
    save_plot(outdir / "07_fanin_unique_sources.png")

    print(f"Saved plots in: {outdir}")


if __name__ == "__main__":
    main()

