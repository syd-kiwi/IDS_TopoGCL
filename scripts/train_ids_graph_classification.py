#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def graph_vector(g):
    x = g.node_features
    return np.concatenate([x.mean(axis=0), x.std(axis=0), np.array([x.shape[0], g.edge_index.shape[1]], dtype=np.float32)])


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden)
        self.w2 = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        n = x.size(0)
        a = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        a[edge_index[0], edge_index[1]] = 1.0
        a = a + torch.eye(n, device=x.device)
        d = a.sum(1)
        d_inv = torch.pow(d.clamp_min(1), -0.5)
        a_hat = d_inv[:, None] * a * d_inv[None, :]
        h = torch.relu(a_hat @ self.w1(x))
        h = a_hat @ self.w2(h)
        g = h.mean(0)
        return self.head(g)


def eval_binary(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/IDS_GRAPH_BENCHMARK")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_json", default="results/ids_graph_classification_metrics.json")
    args = ap.parse_args()

    graphs = np.load(Path(args.dataset_dir) / "graphs.npy", allow_pickle=True)
    X = np.stack([graph_vector(g) for g in graphs], axis=0)
    y = np.array([int(g.graph_label) for g in graphs], dtype=np.int64)

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, graphs, test_size=args.test_size, random_state=args.seed, stratify=y if len(set(y)) > 1 else None)

    svm = SVC(kernel="rbf", probability=True, random_state=args.seed)
    svm.fit(Xtr, ytr)
    svm_prob = svm.predict_proba(Xte)[:, 1]
    svm_metrics = eval_binary(yte, svm_prob)

    model = GNN(in_dim=gtr[0].node_features.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(args.epochs):
        model.train()
        for g, yy in zip(gtr, ytr):
            x = torch.tensor(g.node_features, dtype=torch.float32)
            ei = torch.tensor(g.edge_index, dtype=torch.long)
            logits = model(x, ei).unsqueeze(0)
            target = torch.tensor([yy], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()

    def predict_scores(gs):
        model.eval(); out = []
        with torch.no_grad():
            for g in gs:
                x = torch.tensor(g.node_features, dtype=torch.float32)
                ei = torch.tensor(g.edge_index, dtype=torch.long)
                p = torch.softmax(model(x, ei), dim=0)[1].item()
                out.append(p)
        return np.array(out)

    gnn_prob = predict_scores(gte)
    gnn_metrics = eval_binary(yte, gnn_prob)

    # TopoGCL placeholder: same graph-level vector pipeline with linear proxy.
    topogcl = SVC(kernel="linear", probability=True, random_state=args.seed)
    topogcl.fit(Xtr, ytr)
    topogcl_prob = topogcl.predict_proba(Xte)[:, 1]
    topogcl_metrics = eval_binary(yte, topogcl_prob)

    out = {"svm": svm_metrics, "gnn": gnn_metrics, "topogcl": topogcl_metrics, "n_train": int(len(ytr)), "n_test": int(len(yte))}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
