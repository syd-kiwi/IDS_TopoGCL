#!/usr/bin/env python3
"""Randomly sample a fraction of lines from a LANL auth.txt file."""

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample a random subset of rows from LANL auth.txt")
    p.add_argument("input_path", type=Path, help="Path to source auth.txt file")
    p.add_argument("output_path", type=Path, help="Path to write sampled file")
    p.add_argument("--fraction", type=float, default=0.30, help="Fraction of rows to keep (default: 0.30)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    p.add_argument("--has-header", action="store_true", help="Treat first line as header and always keep it")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    rng = random.Random(args.seed)

    with args.input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    header_lines = 1 if args.has_header and total_lines > 0 else 0
    data_lines = total_lines - header_lines
    sample_size = int(data_lines * args.fraction)
    sample_size = max(1, sample_size) if data_lines > 0 else 0

    chosen = set(rng.sample(range(data_lines), sample_size)) if sample_size > 0 else set()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.input_path.open("r", encoding="utf-8") as src, args.output_path.open("w", encoding="utf-8") as dst:
        if header_lines:
            dst.write(src.readline())

        for idx, line in enumerate(src):
            if idx in chosen:
                dst.write(line)

    print(
        f"Wrote {sample_size} / {data_lines} data lines "
        f"({(sample_size / data_lines * 100) if data_lines else 0:.2f}%) to {args.output_path}"
    )


if __name__ == "__main__":
    main()
