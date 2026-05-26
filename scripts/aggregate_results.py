import glob
import json
import os

import pandas as pd


SCENARIOS = ["clean", "low_volume", "missing_structure", "interference"]
METRICS = ["accuracy", "precision", "recall", "f1", "fpr", "auroc"]


rows = []
json_paths = sorted(set(glob.glob("results/**/*.json", recursive=True) + glob.glob("results/fast_lanl/*.json")))
for p in json_paths:
    with open(p) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    ss = d.get("score_stats", {})
    btr = ss.get("benign_train", {}) if isinstance(ss, dict) else {}
    bte = ss.get("benign_test", {}) if isinstance(ss, dict) else {}
    mal = ss.get("malicious", {}) if isinstance(ss, dict) else {}
    seed = d.get("training", {}).get("random_seed")
    if seed is None:
        seed = d.get("random_seed")
    rows.append(
        {
            "path": p,
            "dataset": d.get("dataset"),
            "method": d.get("method"),
            "seed": seed,
            "train_scenario": d.get("train_scenario"),
            "test_scenario": d.get("test_scenario"),
            "train_degradation_rate": d.get("train_degradation_rate"),
            "test_degradation_rate": d.get("test_degradation_rate"),
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "fpr": m.get("fpr"),
            "auroc": m.get("auroc"),
            "benign_train_min": btr.get("min"),
            "benign_train_mean": btr.get("mean"),
            "benign_train_max": btr.get("max"),
            "benign_test_min": bte.get("min"),
            "benign_test_mean": bte.get("mean"),
            "benign_test_max": bte.get("max"),
            "malicious_min": mal.get("min"),
            "malicious_mean": mal.get("mean"),
            "malicious_max": mal.get("max"),
        }
    )

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/summary_metrics.csv", index=False)
for sc in SCENARIOS:
    df[df["test_scenario"] == sc].to_csv(f"results/table_{sc}.csv", index=False)

if not df.empty:
    grouped = (
        df.groupby(
            [
                "dataset",
                "method",
                "train_scenario",
                "test_scenario",
                "train_degradation_rate",
                "test_degradation_rate",
            ],
            dropna=False,
        )[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
else:
    grouped = pd.DataFrame()

grouped.to_csv("results/summary_metrics_mean_std.csv", index=False)
print("wrote aggregation tables")
