#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def host_to_int(x) -> int:
    s = str(x).strip()
    if s.startswith("C") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    return abs(hash(s)) % 10_000_000


def write_no_header(df: pd.DataFrame, out_csv: Path, first: bool):
    df.to_csv(out_csv, mode="w" if first else "a", index=False, header=False)


def detect_kind(cols_lower):
    if "packet count" in cols_lower and "byte count" in cols_lower and "duration" in cols_lower:
        return "flows"
    if "resolved" in cols_lower and "source" in cols_lower:
        return "dns"
    if "authentication type" in cols_lower and "source computer" in cols_lower and "destination computer" in cols_lower:
        return "auth"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--kind", default="auto", choices=["auto", "flows", "dns", "auth"])
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = True
    kind = None

    for chunk in pd.read_csv(in_path, chunksize=args.chunksize):
        if kind is None:
            cols_lower = {c.lower(): c for c in chunk.columns}
            if args.kind == "auto":
                kind = detect_kind(set(cols_lower.keys()))
            else:
                kind = args.kind

            if kind == "unknown":
                raise ValueError(f"Could not detect kind from columns: {list(chunk.columns)}")

            print(f"[+] Detected kind: {kind}")

        if kind == "flows":
            out = pd.DataFrame()
            out["t"] = pd.to_numeric(chunk["time"], errors="coerce").fillna(0).astype("int64")
            out["c1"] = pd.to_numeric(chunk["duration"], errors="coerce").fillna(0).astype("int64")
            out["src"] = chunk["source computer"].map(host_to_int).astype("int64")
            out["srcp"] = pd.to_numeric(chunk["source port"], errors="coerce").fillna(0).astype("int64")
            out["dst"] = chunk["destination computer"].map(host_to_int).astype("int64")
            out["dstp"] = pd.to_numeric(chunk["destination port"], errors="coerce").fillna(0).astype("int64")
            out["proto"] = pd.to_numeric(chunk["protocol"], errors="coerce").fillna(0).astype("int64")
            out["pkt"] = pd.to_numeric(chunk["packet count"], errors="coerce").fillna(0).astype("int64")
            out["byt"] = pd.to_numeric(chunk["byte count"], errors="coerce").fillna(0).astype("int64")
            write_no_header(out, out_path, first)

        elif kind == "dns":
            out = pd.DataFrame()
            out["t"] = pd.to_numeric(chunk["time"], errors="coerce").fillna(0).astype("int64")
            out["c1"] = 0
            out["src"] = chunk["source"].map(host_to_int).astype("int64")
            out["srcp"] = 0
            out["dst"] = chunk["resolved"].map(host_to_int).astype("int64")
            out["dstp"] = 0
            out["proto"] = 1000
            out["pkt"] = 1
            out["byt"] = 0
            write_no_header(out, out_path, first)

        elif kind == "auth":
            out = pd.DataFrame()
            out["t"] = pd.to_numeric(chunk["time"], errors="coerce").fillna(0).astype("int64")
            out["c1"] = 0
            out["src"] = chunk["source computer"].map(host_to_int).astype("int64")
            out["srcp"] = 0
            out["dst"] = chunk["destination computer"].map(host_to_int).astype("int64")
            out["dstp"] = 0
            out["proto"] = 2000
            out["pkt"] = 1
            out["byt"] = 0
            write_no_header(out, out_path, first)

        first = False

    print(f"[+] Wrote {out_path}")


if __name__ == "__main__":
    main()
