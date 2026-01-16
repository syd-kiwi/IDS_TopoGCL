#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from bisect import bisect_left

def split_csv(line: str):
    return [p.strip() for p in line.strip().split(",")]

def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def norm(s: str) -> str:
    return s.strip()

class CatMap:
    def __init__(self):
        self.m = {}
        self.next_id = 0
    def get(self, key: str) -> int:
        k = norm(key)
        if k not in self.m:
            self.m[k] = self.next_id
            self.next_id += 1
        return self.m[k]

def build_red_index(red_path: Path):
    """
    (user, src_comp, dst_comp) -> sorted times
    """
    idx = {}
    with red_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("time,"):
                continue
            parts = split_csv(line)
            if len(parts) < 4:
                continue
            t = safe_int(parts[0])
            user = norm(parts[1])
            src_comp = norm(parts[2])
            dst_comp = norm(parts[3])
            key = (user, src_comp, dst_comp)
            idx.setdefault(key, []).append(t)
    for k in idx:
        idx[k].sort()
    return idx

def has_time_within(sorted_times, t: int, tol: int) -> bool:
    if not sorted_times:
        return False
    i = bisect_left(sorted_times, t)
    if i < len(sorted_times) and abs(sorted_times[i] - t) <= tol:
        return True
    if i > 0 and abs(sorted_times[i - 1] - t) <= tol:
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", required=True, help="LANL AUTH file (csv with header)")
    ap.add_argument("--red", required=True, help="LANL REDTEAM file (csv with header)")
    ap.add_argument("--out_auth", required=True, help="Output baseline-formatted benign file")
    ap.add_argument("--out_red", required=True, help="Output baseline-formatted malicious file")
    ap.add_argument("--time_tol", type=int, default=0, help="seconds tolerance for time matching")
    ap.add_argument("--out_maps_dir", type=str, default="", help="Optional dir to write mapping csv files")
    args = ap.parse_args()

    auth_path = Path(args.auth)
    red_path = Path(args.red)
    out_auth = Path(args.out_auth)
    out_red = Path(args.out_red)

    out_auth.parent.mkdir(parents=True, exist_ok=True)
    out_red.parent.mkdir(parents=True, exist_ok=True)

    red_idx = build_red_index(red_path)
    print(f"[OK] redteam keys: {len(red_idx)}")

    # Shared maps so IDs are consistent between benign and malicious files
    user_ids = CatMap()
    comp_ids = CatMap()
    auth_type_ids = CatMap()
    logon_type_ids = CatMap()
    orient_ids = CatMap()
    success_ids = CatMap()

    benign_rows = 0
    mal_rows = 0

    with auth_path.open("r", encoding="utf-8", errors="ignore") as f_in, \
         out_auth.open("w", newline="", encoding="utf-8") as f_ben, \
         out_red.open("w", newline="", encoding="utf-8") as f_mal:

        w_ben = csv.writer(f_ben)
        w_mal = csv.writer(f_mal)

        for line in f_in:
            line = line.strip()
            if not line or line.lower().startswith("time,"):
                continue
            parts = split_csv(line)
            if len(parts) < 9:
                continue

            t = safe_int(parts[0])

            src_user = norm(parts[1])
            dst_user = norm(parts[2])
            src_comp = norm(parts[3])
            dst_comp = norm(parts[4])

            auth_type = norm(parts[5])
            logon_type = norm(parts[6])
            orient = norm(parts[7])
            success = norm(parts[8])

            # Map into baseline 9-int schema:
            # time, c1, c2, c3, c4, c5, c6, c7, c8
            c1 = user_ids.get(src_user)
            c2 = comp_ids.get(src_comp)
            c3 = auth_type_ids.get(auth_type)
            c4 = comp_ids.get(dst_comp)
            c5 = user_ids.get(dst_user)
            c6 = logon_type_ids.get(logon_type)
            c7 = orient_ids.get(orient)
            c8 = success_ids.get(success)

            row = [t, c1, c2, c3, c4, c5, c6, c7, c8]
            w_ben.writerow(row)
            benign_rows += 1

            if benign_rows % 1_000_000 == 0:
                print(f"[PROGRESS] auth lines written: {benign_rows:,}   mal so far: {mal_rows:,}")

            # Decide if this auth row belongs in malicious file
            key = (src_user, src_comp, dst_comp)
            if has_time_within(red_idx.get(key, []), t, args.time_tol):
                w_mal.writerow(row)
                mal_rows += 1

    print(f"[OK] wrote benign rows: {benign_rows} -> {out_auth}")
    print(f"[OK] wrote malicious rows: {mal_rows} -> {out_red}")
    print(f"[OK] time tolerance: {args.time_tol}s")

    # Optional: dump maps for debugging
    if args.out_maps_dir:
        md = Path(args.out_maps_dir)
        md.mkdir(parents=True, exist_ok=True)

        def dump_map(path: Path, m: CatMap, prefix: str):
            with path.open("w", encoding="utf-8") as f:
                for k, v in sorted(m.m.items(), key=lambda kv: kv[1]):
                    f.write(f"{v},{prefix}{k}\n")

        dump_map(md / "users.csv", user_ids, "user:")
        dump_map(md / "computers.csv", comp_ids, "comp:")
        dump_map(md / "auth_type.csv", auth_type_ids, "auth_type:")
        dump_map(md / "logon_type.csv", logon_type_ids, "logon_type:")
        dump_map(md / "orient.csv", orient_ids, "orient:")
        dump_map(md / "success.csv", success_ids, "success:")

        print(f"[OK] wrote maps -> {md}")

if __name__ == "__main__":
    main()
