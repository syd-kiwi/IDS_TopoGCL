from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

RawEvent = Tuple[int, int, int, int, int, int, int, int, int]


def _clamp_rate(rate: float) -> float:
    if not 0.0 <= rate <= 0.9:
        raise ValueError(f"degradation rate must be in [0.0, 0.9], got {rate}")
    return float(rate)


def mask_node_features(x: torch.Tensor, rate: float, mode: str = "element", fill_value: float = 0.0, rng: Optional[torch.Generator] = None):
    rate = _clamp_rate(rate)
    if rate <= 0.0:
        return x.clone(), {"masked_entries": 0, "total_entries": int(x.numel()), "masked_fraction": 0.0}
    out = x.clone()
    if mode == "element":
        keep = torch.rand(out.shape, generator=rng, device=out.device) > rate
        out = torch.where(keep, out, torch.full_like(out, fill_value))
        masked_entries = int((~keep).sum().item())
    elif mode == "dimension":
        num_dims = out.shape[1] if out.dim() > 1 else 1
        keep_dims = torch.rand(num_dims, generator=rng, device=out.device) > rate
        out = torch.where(keep_dims.unsqueeze(0), out, torch.full_like(out, fill_value)) if out.dim() > 1 else torch.where(keep_dims, out, torch.full_like(out, fill_value))
        masked_entries = int((~keep_dims).sum().item()) * (int(out.shape[0]) if out.dim() > 1 else 1)
    else:
        raise ValueError(f"unknown node mask mode: {mode}")
    total_entries = int(out.numel())
    return out, {"masked_entries": masked_entries, "total_entries": total_entries, "masked_fraction": masked_entries / max(total_entries, 1)}


def inject_feature_noise(x: torch.Tensor, noise_std: float = 0.1, rng: Optional[torch.Generator] = None):
    noise = torch.randn(x.shape, generator=rng, device=x.device) * noise_std
    return x + noise, {"noise_std": float(noise_std)}


def perturb_features(x: torch.Tensor, rate: float, rng: Optional[torch.Generator] = None):
    out = x.clone()
    if rate <= 0:
        return out, {"perturbed_fraction": 0.0}
    mask = torch.rand(out.shape, generator=rng, device=out.device) < rate
    perm = torch.randperm(out.numel(), generator=rng, device=out.device)
    shuffled = out.reshape(-1)[perm].reshape_as(out)
    out[mask] = shuffled[mask]
    return out, {"perturbed_fraction": float(mask.float().mean().item())}


def drop_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, rate: float = 0.0, rng: Optional[torch.Generator] = None):
    rate = _clamp_rate(rate)
    if rate <= 0.0 or edge_index.numel() == 0:
        e = int(edge_index.shape[1] if edge_index.dim() == 2 else 0)
        return edge_index.clone(), edge_attr.clone() if edge_attr is not None else None, {"edges_before": e, "edges_after": e}
    keep = torch.rand(edge_index.shape[1], generator=rng, device=edge_index.device) > rate
    out_edges = edge_index[:, keep]
    out_attr = edge_attr[keep] if edge_attr is not None else None
    return out_edges, out_attr, {"edges_before": int(edge_index.shape[1]), "edges_after": int(out_edges.shape[1])}


def drop_nodes(x: torch.Tensor, edge_index: torch.Tensor, rate: float = 0.0, rng: Optional[torch.Generator] = None):
    rate = _clamp_rate(rate)
    n = x.shape[0]
    if rate <= 0.0 or n == 0:
        return x.clone(), edge_index.clone(), {"nodes_before": n, "nodes_after": n}
    keep = torch.rand(n, generator=rng, device=x.device) > rate
    if keep.sum() == 0:
        keep[torch.randint(0, n, (1,), generator=rng)] = True
    idx = torch.full((n,), -1, dtype=torch.long, device=x.device)
    idx[keep] = torch.arange(int(keep.sum().item()), device=x.device)
    edges_keep = keep[edge_index[0]] & keep[edge_index[1]] if edge_index.numel() > 0 else torch.zeros((0,), dtype=torch.bool, device=x.device)
    out_edges = idx[edge_index[:, edges_keep]] if edge_index.numel() > 0 else edge_index.clone()
    return x[keep], out_edges, {"nodes_before": n, "nodes_after": int(keep.sum().item())}


def drop_raw_events(events: Sequence[RawEvent], rate: float = 0.0, mode: str = "random", window_size: Optional[int] = None, rng: Optional[torch.Generator] = None):
    rate = _clamp_rate(rate)
    events_list = list(events)
    n = len(events_list)
    if rate <= 0.0 or n == 0:
        return events_list, {"events_before": n, "events_after": n}
    if mode == "random":
        keep = torch.rand(n, generator=rng) > rate
        out = [row for row, k in zip(events_list, keep.tolist()) if k]
    else:
        drop_len = window_size if window_size is not None else max(1, int(round(n * rate)))
        start = int(torch.randint(0, max(n - drop_len + 1, 1), (1,), generator=rng).item())
        out = events_list[:start] + events_list[start + drop_len :]
    return out, {"events_before": n, "events_after": len(out)}


def subsample_windows(graphs: Sequence, rate: float = 0.0, rng: Optional[torch.Generator] = None):
    rate = _clamp_rate(rate)
    items = list(graphs)
    n = len(items)
    if rate <= 0.0 or n == 0:
        return items, {"windows_before": n, "windows_after": n}
    keep = torch.rand(n, generator=rng) > rate
    out = [g for g, k in zip(items, keep.tolist()) if k]
    return out, {"windows_before": n, "windows_after": len(out)}


def delay_events_or_windows(graphs: Sequence, delay_steps: int = 1):
    items = list(graphs)
    d = max(0, int(delay_steps))
    if d <= 0 or len(items) <= 1:
        return items, {"delay_steps": d}
    return [items[max(0, i - d)] for i in range(len(items))], {"delay_steps": d}
