#!/usr/bin/env python3
"""
Plot IDS model results after training.

This script replaces the old .npz graph-image visualizer. It is now a small
post-training plotting utility for StreamSpot/GraSec experiments.

It reads the CSV files produced by the training scripts and saves one separate
plot per model:
  1. A metric summary plot from *_summary*.csv files.
  2. A benign-vs-malicious score distribution plot from *_scores*.csv files.

Expected summary CSV columns:
  dataset, train_ratio, model, accuracy_mean, accuracy_std, precision_mean,
  precision_std, recall_mean, recall_std, f1_mean, f1_std, auroc_mean,
  auroc_std, auprc_mean, auprc_std

Expected scores CSV columns:
  dataset, train_ratio, seed, model, index, val_score, y_val, test_score, y_test

Examples:
  python3 visualize_graphs.py --results-dir results --out-dir results/plots

  python3 visualize_graphs.py \
    --summary-csv results/streamspot/cl_summary.csv results/streamspot/gnn_summary.csv \
    --scores-csv results/streamspot/cl_scores.csv \
    --out-dir results/plots
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")
LABEL_NAMES = {0: "benign", 1: "malicious"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one separate post-training result figure per IDS model."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to search recursively for summary/scores CSVs when explicit CSVs are not provided.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        nargs="*",
        default=None,
        help="One or more summary CSV files. If omitted, searches --results-dir for *summary*.csv.",
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        nargs="*",
        default=None,
        help="One or more scores CSV files. If omitted, searches --results-dir for *scores*.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--metrics",
        default="accuracy,precision,recall,f1,auroc,auprc",
        help="Comma-separated metric names to show in summary plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--no-auto-discover",
        action="store_true",
        help="Do not search --results-dir when explicit CSV paths are missing.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: object) -> int | None:
    as_float = safe_float(value)
    if as_float is None:
        return None
    return int(as_float)


def read_csv_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] missing CSV: {path}", flush=True)
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_csv"] = str(path)
                rows.append(dict(row))
        print(f"[OK] loaded {path}", flush=True)
    return rows


def discover_csvs(results_dir: Path, kind: str) -> List[Path]:
    if not results_dir.exists():
        return []
    if kind == "summary":
        paths = sorted(results_dir.rglob("*summary*.csv"))
    elif kind == "scores":
        paths = sorted(results_dir.rglob("*scores*.csv"))
    else:
        raise ValueError(f"unknown CSV kind: {kind}")
    return [p for p in paths if p.is_file()]


def slugify(text: object) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown"


def model_display_name(model: str) -> str:
    names = {
        "gnn": "Base GNN",
        "graphcl": "GraphCL",
        "infograph": "InfoGraph",
        "rgcl": "RGCL",
        "topogcl": "TopoGCL",
        "ids_topogcl": "IDS-TopoGCL",
        "improved": "IDS-TopoGCL",
    }
    return names.get(model.lower(), model)


def dataset_display_name(dataset: str) -> str:
    if not dataset:
        return "dataset"
    return dataset


def group_summary_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "unknown")
        dataset = row.get("dataset", "dataset")
        train_ratio = row.get("train_ratio", "")
        grouped[(dataset, train_ratio, model)].append(row)
    return grouped


def plot_metric_summary(
    summary_rows: List[Dict[str, str]],
    out_dir: Path,
    metrics: Sequence[str],
    dpi: int,
) -> List[Path]:
    if not summary_rows:
        return []

    out_paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_summary_rows(summary_rows)

    for (dataset, train_ratio, model), rows in sorted(grouped.items()):
        # Usually one row per dataset/train_ratio/model. If there are duplicates from
        # multiple CSVs, average them so the plot remains stable.
        means = []
        stds = []
        labels = []
        for metric in metrics:
            mean_values = [safe_float(row.get(f"{metric}_mean")) for row in rows]
            std_values = [safe_float(row.get(f"{metric}_std")) for row in rows]
            mean_values = [v for v in mean_values if v is not None]
            std_values = [v for v in std_values if v is not None]
            if not mean_values:
                continue
            labels.append(metric.upper() if metric in {"f1", "auroc", "auprc"} else metric.title())
            means.append(float(np.mean(mean_values)))
            stds.append(float(np.mean(std_values)) if std_values else 0.0)

        if not means:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(
            f"{model_display_name(model)} on {dataset_display_name(dataset)}"
            + (f" (train ratio={train_ratio})" if train_ratio != "" else "")
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"summary_{slugify(dataset)}_{slugify(train_ratio)}_{slugify(model)}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        out_paths.append(out_path)
        print(f"[OK] saved {out_path}", flush=True)

    return out_paths


def group_score_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "unknown")
        dataset = row.get("dataset", "dataset")
        train_ratio = row.get("train_ratio", "")
        grouped[(dataset, train_ratio, model)].append(row)
    return grouped


def collect_scores(rows: Sequence[Dict[str, str]], split: str) -> Tuple[np.ndarray, np.ndarray]:
    score_key = f"{split}_score"
    label_key = f"y_{split}"
    scores: List[float] = []
    labels: List[int] = []
    for row in rows:
        score = safe_float(row.get(score_key))
        label = safe_int(row.get(label_key))
        if score is None or label is None:
            continue
        scores.append(score)
        labels.append(label)
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64)


def plot_score_distributions(score_rows: List[Dict[str, str]], out_dir: Path, dpi: int) -> List[Path]:
    if not score_rows:
        return []

    out_paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_score_rows(score_rows)

    for (dataset, train_ratio, model), rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False

        for split, linestyle_label in (("val", "validation"), ("test", "test")):
            scores, labels = collect_scores(rows, split)
            if scores.size == 0:
                continue
            for label_value in sorted(set(labels.tolist())):
                label_scores = scores[labels == label_value]
                if label_scores.size == 0:
                    continue
                ax.hist(
                    label_scores,
                    bins=30,
                    alpha=0.45 if split == "test" else 0.25,
                    density=True,
                    label=f"{linestyle_label} {LABEL_NAMES.get(label_value, f'label {label_value}')}",
                )
                plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Anomaly score / model score")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Score distribution: {model_display_name(model)} on {dataset_display_name(dataset)}"
            + (f" (train ratio={train_ratio})" if train_ratio != "" else "")
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"scores_{slugify(dataset)}_{slugify(train_ratio)}_{slugify(model)}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        out_paths.append(out_path)
        print(f"[OK] saved {out_path}", flush=True)

    return out_paths


def parse_metrics(raw: str) -> List[str]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = [metric for metric in requested if metric in SUMMARY_METRICS]
    invalid = sorted(set(requested) - set(valid))
    if invalid:
        print(f"[WARN] ignoring unknown metrics: {', '.join(invalid)}", flush=True)
    return valid or list(SUMMARY_METRICS)


def main() -> None:
    args = parse_args()
    metrics = parse_metrics(args.metrics)

    summary_paths = list(args.summary_csv or [])
    scores_paths = list(args.scores_csv or [])

    if not args.no_auto_discover:
        if not summary_paths:
            summary_paths = discover_csvs(args.results_dir, "summary")
        if not scores_paths:
            scores_paths = discover_csvs(args.results_dir, "scores")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] output directory: {args.out_dir}", flush=True)
    print(f"[INFO] summary CSVs: {len(summary_paths)}", flush=True)
    print(f"[INFO] scores CSVs: {len(scores_paths)}", flush=True)

    summary_rows = read_csv_rows(summary_paths)
    score_rows = read_csv_rows(scores_paths)

    saved_summary = plot_metric_summary(summary_rows, args.out_dir, metrics, args.dpi)
    saved_scores = plot_score_distributions(score_rows, args.out_dir, args.dpi)

    print("\nDone.", flush=True)
    print(f"Saved summary plots: {len(saved_summary)}", flush=True)
    print(f"Saved score plots: {len(saved_scores)}", flush=True)


if __name__ == "__main__":
    main()
