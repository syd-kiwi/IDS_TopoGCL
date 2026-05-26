#!/usr/bin/env python3
"""Write a random 30% sample from the LANL auth file."""

import random

INPUT_PATH = "/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/LANL/auth.txt"
OUTPUT_PATH = "/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/LANL/auth_30pct.txt"
FRACTION = 0.30
SEED = 42
HAS_HEADER = True


def main() -> None:
    random.seed(SEED)
    kept = 0
    total = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as src, open(OUTPUT_PATH, "w", encoding="utf-8") as dst:
        if HAS_HEADER:
            header = src.readline()
            if header:
                dst.write(header)

        for line in src:
            total += 1
            if random.random() < FRACTION:
                dst.write(line)
                kept += 1

    pct = (kept / total * 100) if total else 0.0
    print(f"Wrote {kept}/{total} lines ({pct:.2f}%) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
