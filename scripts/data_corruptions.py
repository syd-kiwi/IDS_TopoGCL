from typing import Dict, Optional, Tuple

import torch


def _num_to_drop(total: int, rate: float) -> int:
    rate = float(max(0.0, min(1.0, rate)))
    return int(round(total * rate))


def drop_edges(edge_index: torch.Tensor, rate: float, rng: Optional[torch.Generator] = None):
    if edge_index.numel() == 0:
        return edge_index, torch.empty(0, dtype=torch.bool), {"edges_before": 0, "edges_after": 0}
    m = edge_index.shape[1]
    k = _num_to_drop(m, rate)
    if k <= 0:
        keep_mask = torch.ones(m, dtype=torch.bool)
        return edge_index, keep_mask, {"edges_before": m, "edges_after": m}
    perm = torch.randperm(m, generator=rng)
    keep_mask = torch.ones(m, dtype=torch.bool)
    keep_mask[perm[:k]] = False
    kept = edge_index[:, keep_mask]
    return kept, keep_mask, {"edges_before": m, "edges_after": int(kept.shape[1])}


def drop_nodes(x: torch.Tensor, rate: float, mode: str = "zero", fill_value: float = 0.0, rng: Optional[torch.Generator] = None):
    n = x.shape[0]
    k = _num_to_drop(n, rate)
    if k <= 0:
        return x, {"masked_fraction": 0.0}
    perm = torch.randperm(n, generator=rng)
    idx = perm[:k]
    out = x.clone()
    if mode == "mean":
        out[idx] = x.mean(dim=0, keepdim=True)
    else:
        out[idx] = fill_value
    return out, {"masked_fraction": float(k / max(n, 1))}


def mask_node_features(x: torch.Tensor, rate: float, mode: str = "zero", fill_value: float = 0.0, rng: Optional[torch.Generator] = None):
    if x.numel() == 0:
        return x, {"masked_fraction": 0.0}
    out = x.clone()
    if mode == "column":
        d = out.shape[1]
        k = _num_to_drop(d, rate)
        if k > 0:
            idx = torch.randperm(d, generator=rng)[:k]
            out[:, idx] = fill_value
        frac = float(k / max(d, 1))
    elif mode == "node":
        return drop_nodes(out, rate=rate, mode="zero", fill_value=fill_value, rng=rng)
    else:
        mask = torch.rand(out.shape, generator=rng) < max(0.0, min(1.0, rate))
        out[mask] = fill_value
        frac = float(mask.float().mean().item())
    return out, {"masked_fraction": frac}


def drop_raw_events(rows, rate: float, mode: str = "uniform", rng: Optional[torch.Generator] = None):
    total = len(rows)
    if total == 0:
        return rows, {"events_before": 0, "events_after": 0}
    k = _num_to_drop(total, rate)
    if k <= 0:
        return rows, {"events_before": total, "events_after": total}
    if mode == "head":
        out = rows[k:]
    elif mode == "tail":
        out = rows[: total - k]
    else:
        keep = torch.ones(total, dtype=torch.bool)
        drop_idx = torch.randperm(total, generator=rng)[:k]
        keep[drop_idx] = False
        out = [r for i, r in enumerate(rows) if bool(keep[i])]
    return out, {"events_before": total, "events_after": len(out)}
