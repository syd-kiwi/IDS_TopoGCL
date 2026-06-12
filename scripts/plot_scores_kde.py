#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "dataset",
    "train_ratio",
    "seed",
    "model",
    "index",
    "val_score",
    "y_val",
    "test_score",
    "y_test",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot KDE distribution of test_score from bit_out_scores.csv by y_test label."
    )
    parser.add_argument(
        "--input-csv",
        default="results/bit_out_scores.csv",
        help="Path to bit_out_scores.csv",
    )
    parser.add_argument(
        "--output-png",
        default="results/bit_out_scores_kde.png",
        help="Path where the KDE plot PNG should be saved",
    )
    parser.add_argument("--dataset", default=None, help="Optional dataset filter")
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional train_ratio filter")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter")
    parser.add_argument("--model", default=None, help="Optional model filter")
    parser.add_argument(
        "--title",
        default="KDE Distribution of Test Scores by y_test",
        help="Plot title",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.dataset is not None:
        df = df[df["dataset"] == args.dataset]

    if args.train_ratio is not None:
        df = df[df["train_ratio"] == args.train_ratio]

    if args.seed is not None:
        df = df[df["seed"] == args.seed]

    if args.model is not None:
        df = df[df["model"] == args.model]

    df = df[["test_score", "y_test"]].dropna()

    if df.empty:
        raise ValueError("No rows left to plot after filtering.")

    df["y_test"] = df["y_test"].astype(str)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x="test_score",
        hue="y_test",
        multiple="stack",
    )

    plt.title(args.title)
    plt.xlabel("Test score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Rows plotted: {len(df)}")
    print(f"Saved KDE plot to: {output_png}")


if __name__ == "__main__":
    main()