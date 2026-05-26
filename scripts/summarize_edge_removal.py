#!/usr/bin/env python3
import csv, glob, json, os

rows = []
for p in glob.glob("results/edge_removal/*.json"):
    with open(p) as f:
        d = json.load(f)
    rows.append({
        "dataset": d.get("dataset", os.path.basename(p).split("_")[0]),
        "method": d.get("method", "unknown"),
        "edge_drop_rate": d.get("edge_drop_rate", d.get("corruption_rate", 0.0)),
        "auroc": d.get("metrics", {}).get("auroc"),
        "f1": d.get("metrics", {}).get("f1"),
        "accuracy": d.get("metrics", {}).get("accuracy"),
    })

rows = sorted(rows, key=lambda r: (r["dataset"], float(r["edge_drop_rate"]), r["method"]))
os.makedirs("results", exist_ok=True)
with open("results/edge_removal_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "method", "edge_drop_rate", "auroc", "f1", "accuracy"])
    w.writeheader(); w.writerows(rows)

# gap table (TopoGCL - baseline) on AUROC
by = {}
for r in rows:
    key = (r["dataset"], r["edge_drop_rate"])
    by.setdefault(key, {})[r["method"]] = r

gaps = []
for (ds, rate), methods in sorted(by.items()):
    t = methods.get("topogcl")
    g = methods.get("gnn")
    s = methods.get("svm")
    if not t:
        continue
    ta = t.get("auroc")
    gap_g = (ta - g.get("auroc")) if (g and ta is not None and g.get("auroc") is not None) else None
    gap_s = (ta - s.get("auroc")) if (s and ta is not None and s.get("auroc") is not None) else None
    gaps.append({"dataset": ds, "edge_drop_rate": rate, "gap_topogcl_minus_gnn": gap_g, "gap_topogcl_minus_svm": gap_s})

with open("results/edge_removal_gaps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "edge_drop_rate", "gap_topogcl_minus_gnn", "gap_topogcl_minus_svm"])
    w.writeheader(); w.writerows(gaps)

print("wrote results/edge_removal_summary.csv and results/edge_removal_gaps.csv")
