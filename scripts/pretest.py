from pathlib import Path
import numpy as np
import pandas as pd

base = Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph")

if not base.exists():
    print("DEBUG folder not found. Available benchmark folders:")
    for p in Path("datasets").glob("IDS_GRAPH_BENCHMARK*"):
        print(p)
    raise SystemExit

files = sorted(base.rglob("*.npz"))
print("Graph files:", len(files))

for f in files[:3]:
    print("\nFILE:", f)
    data = np.load(f, allow_pickle=True)
    print("Keys:", data.files)
    for k in data.files:
        arr = data[k]
        print(k, arr.shape, arr.dtype)

summary = base / "graph_window_summary.csv"
if summary.exists():
    df = pd.read_csv(summary)
    print("\nSummary rows:", len(df))
    print(df.head())

    print("\nGraph labels:")
    print(df["graph_label"].value_counts())

    print("\nNode count stats:")
    print(df["num_nodes"].describe())

    print("\nEdge count stats:")
    print(df["num_edges"].describe())
else:
    print("\nNo graph_window_summary.csv found")