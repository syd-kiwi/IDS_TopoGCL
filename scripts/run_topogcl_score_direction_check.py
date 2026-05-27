#!/usr/bin/env python3
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

DATASET = "OPTC"
LABEL_RATIOS = [0.10, 0.50, 1.00]
OUTPUT_DIR = "results/limited_labels/OPTC_score_direction"
THRESHOLD_METHOD = "best_validation_f1"
AUTH_PATH = "datasets/OPTC/auth_optc.txt"
RED_PATH = "datasets/OPTC/redteam_optc.txt"
WINDOW_SIZE = 300
SEED = 42


@dataclass
class GraphWindow:
    x: torch.Tensor
    a_hat: torch.Tensor
    num_nodes: int


def parse_lines_to_windows(file_path: str, window_size: int = 300) -> Generator[List[Tuple[int, int, int, int, int, int, int, int, int]], None, None]:
    current_start = None
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                t = int(parts[0])
                r = (t, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]))
            except ValueError:
                continue
            if current_start is None:
                current_start = t
            if t >= current_start + window_size:
                if rows:
                    yield rows
                rows = []
                current_start = (t // window_size) * window_size
            rows.append(r)
    if rows:
        yield rows


def build_graph(rows):
    if not rows:
        return GraphWindow(x=torch.zeros(1, 15), a_hat=torch.eye(1), num_nodes=1)
    node_ids: Dict[int, int] = {}
    for _, s, _, _, d, *_ in rows:
        if s not in node_ids:
            node_ids[s] = len(node_ids)
        if d not in node_ids:
            node_ids[d] = len(node_ids)
    n = len(node_ids)
    deg_in = torch.zeros(n)
    deg_out = torch.zeros(n)
    a = torch.zeros((n, n))
    out_sums = torch.zeros((n, 6))
    in_sums = torch.zeros((n, 6))
    out_cnt = torch.zeros(n)
    in_cnt = torch.zeros(n)
    for (_, s, c2, c3, d_id, c5, c6, c7, c8) in rows:
        u = node_ids[s]
        v = node_ids[d_id]
        a[u, v] += 1.0
        a[v, u] += 1.0
        deg_out[u] += 1.0
        deg_in[v] += 1.0
        feat = torch.tensor([float(c2), float(c3), float(c5), float(c6), float(c7), float(c8)])
        out_sums[u] += feat
        in_sums[v] += feat
        out_cnt[u] += 1.0
        in_cnt[v] += 1.0
    deg = deg_in + deg_out
    out_mean = out_sums / out_cnt.clamp_min(1.0).unsqueeze(1)
    in_mean = in_sums / in_cnt.clamp_min(1.0).unsqueeze(1)
    x = torch.cat([deg_in.unsqueeze(1), deg_out.unsqueeze(1), deg.unsqueeze(1), out_mean, in_mean], dim=1)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1.0)
    a = a + torch.eye(n)
    d = a.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp_min(1.0), -0.5)
    a_hat = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    return GraphWindow(x=x, a_hat=a_hat, num_nodes=n)


def load_graphs(path: str) -> List[GraphWindow]:
    return [build_graph(r) for r in parse_lines_to_windows(path, WINDOW_SIZE)]


class GCN(torch.nn.Module):
    def __init__(self, in_dim=15, hidden=32, out_dim=16):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden)
        self.w2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, a_hat):
        h = torch.relu(a_hat @ self.w1(x))
        h = a_hat @ self.w2(h)
        return h.mean(dim=0)


def embed_graphs(model: GCN, graphs: Sequence[GraphWindow]) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([model(g.x, g.a_hat) for g in graphs], dim=0)


def choose_threshold(y_true, scores):
    cand = np.unique(scores)
    best_thr, best_f1 = float(cand[0]), -1.0
    for thr in cand:
        pred = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def recall_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def evaluate(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fpr = fp / max(tn + fp, 1)
    return {
        "accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_positive_rate": float(fpr), "true_positive_rate": float(recall),
        "AUROC": float(roc_auc_score(y_true, scores)), "AUPRC": float(average_precision_score(y_true, scores)),
        "recall_at_1_percent_fpr": recall_at_fpr(y_true, scores, 0.01),
        "recall_at_5_percent_fpr": recall_at_fpr(y_true, scores, 0.05),
        "recall_at_10_percent_fpr": recall_at_fpr(y_true, scores, 0.10),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benign = load_graphs(AUTH_PATH)
    malicious = load_graphs(RED_PATH)

    b_train_end, b_val_end = int(0.6 * len(benign)), int(0.8 * len(benign))
    m_train_end, m_val_end = int(0.6 * len(malicious)), int(0.8 * len(malicious))
    b_train, b_val, b_test = benign[:b_train_end], benign[b_train_end:b_val_end], benign[b_val_end:]
    m_train, m_val, m_test = malicious[:m_train_end], malicious[m_train_end:m_val_end], malicious[m_val_end:]

    rng = np.random.default_rng(SEED)
    idx_b = np.arange(len(b_train)); rng.shuffle(idx_b)
    idx_m = np.arange(len(m_train)); rng.shuffle(idx_m)

    summary_rows = []
    for ratio in LABEL_RATIOS:
        nb = max(1, int(math.floor(ratio * len(b_train))))
        nm = max(1, int(math.floor(ratio * len(m_train))))
        b_sel, m_sel = idx_b[:nb], idx_m[:nm]

        train_graphs = [b_train[i] for i in b_sel] + [m_train[i] for i in m_sel]
        y_train = np.array([0] * len(b_sel) + [1] * len(m_sel))
        val_graphs = b_val + m_val
        y_val = np.array([0] * len(b_val) + [1] * len(m_val))
        test_graphs = b_test + m_test
        y_test = np.array([0] * len(b_test) + [1] * len(m_test))

        enc = GCN(); pre_epochs = 30
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
        unlabeled = b_train + m_train
        for _ in range(pre_epochs):
            for g in unlabeled:
                mask = (torch.rand_like(g.x) > 0.2).float()
                z1 = enc(g.x * mask, g.a_hat)
                z2 = enc(g.x, g.a_hat)
                loss = 1 - torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        clf = torch.nn.Linear(16, 1)
        opt2 = torch.optim.Adam(clf.parameters(), lr=1e-3)
        for _ in range(20):
            for g, y in zip(train_graphs, y_train):
                z = enc(g.x, g.a_hat).detach()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(z), torch.tensor([float(y)]))
                opt2.zero_grad(); loss.backward(); opt2.step()

        with torch.no_grad():
            val_scores = torch.sigmoid(clf(embed_graphs(enc, val_graphs))).squeeze(1).numpy()
            test_scores = torch.sigmoid(clf(embed_graphs(enc, test_graphs))).squeeze(1).numpy()

        val_scores_neg = -val_scores
        test_scores_neg = -test_scores
        auroc_original = float(roc_auc_score(y_val, val_scores))
        auroc_negative = float(roc_auc_score(y_val, val_scores_neg))
        auprc_original = float(average_precision_score(y_val, val_scores))
        auprc_negative = float(average_precision_score(y_val, val_scores_neg))

        if auroc_negative > auroc_original:
            chosen_direction = "negative"
            chosen_val_scores = val_scores_neg
            chosen_test_scores = test_scores_neg
        else:
            chosen_direction = "original"
            chosen_val_scores = val_scores
            chosen_test_scores = test_scores

        thr = choose_threshold(y_val, chosen_val_scores)
        metrics = evaluate(y_test, chosen_test_scores, thr)
        res = {
            "dataset": DATASET,
            "model": "topogcl",
            "label_ratio": ratio,
            "labeled_benign_training_windows": int(nb),
            "labeled_malicious_training_windows": int(nm),
            "validation_benign_windows": len(b_val),
            "validation_malicious_windows": len(m_val),
            "test_benign_windows": len(b_test),
            "test_malicious_windows": len(m_test),
            "threshold_selection_method": THRESHOLD_METHOD,
            "threshold": float(thr),
            "pretraining_epochs": pre_epochs,
            "downstream_training_epochs": 20,
            "contrastive_temperature": 0.2,
            "corruption_type": "node_feature_mask",
            "corruption_rate": 0.2,
            "used_unlabeled_graphs_during_pretraining": True,
            "auroc_original": auroc_original,
            "auroc_negative": auroc_negative,
            "auprc_original": auprc_original,
            "auprc_negative": auprc_negative,
            "chosen_score_direction": chosen_direction,
            **metrics,
        }

        out = os.path.join(OUTPUT_DIR, f"topogcl_label_{ratio:.2f}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        summary_rows.append(res)
        print(f"[topogcl] label_ratio={ratio:.2f} chosen_direction={chosen_direction} val_auroc_orig={auroc_original:.4f} val_auroc_neg={auroc_negative:.4f}")

    fields = [
        "dataset", "model", "label_ratio", "chosen_score_direction",
        "auroc_original", "auroc_negative", "auprc_original", "auprc_negative",
        "AUROC", "AUPRC", "f1", "recall_at_5_percent_fpr", "accuracy", "precision", "recall"
    ]
    with open(os.path.join(OUTPUT_DIR, "limited_label_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in fields})


if __name__ == "__main__":
    main()
