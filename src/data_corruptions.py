from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


RawEvent = Tuple[int, int, int, int, int, int, int, int, int]


def mask_node_features(
    x: torch.Tensor,
    rate: float,
    mode: str = "element",
    fill_value: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if rate <= 0.0:
        return x.clone(), {"masked_entries": 0, "total_entries": int(x.numel()), "masked_fraction": 0.0}
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"mask rate must be in [0, 1], got {rate}")
    if mode not in {"element", "dimension"}:
        raise ValueError(f"unknown node mask mode: {mode}")

    out = x.clone()
    total_entries = int(out.numel())
    masked_entries = 0

    if mode == "element":
        keep = torch.rand(out.shape, generator=rng, device=out.device) > rate
        out = torch.where(keep, out, torch.full_like(out, fill_value))
        masked_entries = int((~keep).sum().item())
    else:
        num_dims = out.shape[1] if out.dim() > 1 else 1
        keep_dims = torch.rand(num_dims, generator=rng, device=out.device) > rate
        if out.dim() == 1:
            out = torch.where(keep_dims, out, torch.full_like(out, fill_value))
            masked_entries = int((~keep_dims).sum().item())
            total_entries = int(out.shape[0])
        else:
            out = torch.where(keep_dims.unsqueeze(0), out, torch.full_like(out, fill_value))
            masked_entries = int((~keep_dims).sum().item()) * int(out.shape[0])

    frac = float(masked_entries / max(total_entries, 1))
    return out, {"masked_entries": masked_entries, "total_entries": total_entries, "masked_fraction": frac}


def drop_edges(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    rate: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
    if rate <= 0.0 or edge_index.numel() == 0:
        return edge_index.clone(), edge_attr.clone() if edge_attr is not None else None, {
            "edges_before": int(edge_index.shape[1] if edge_index.dim() == 2 else 0),
            "edges_after": int(edge_index.shape[1] if edge_index.dim() == 2 else 0),
        }
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"edge drop rate must be in [0, 1], got {rate}")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape [2, E], got {tuple(edge_index.shape)}")

    num_edges = edge_index.shape[1]
    keep = torch.rand(num_edges, generator=rng, device=edge_index.device) > rate
    out_edges = edge_index[:, keep]
    out_attr = edge_attr[keep] if edge_attr is not None else None
    return out_edges, out_attr, {"edges_before": int(num_edges), "edges_after": int(out_edges.shape[1])}


def drop_raw_events(
    events: Sequence[RawEvent],
    rate: float = 0.0,
    mode: str = "random",
    window_size: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[List[RawEvent], Dict[str, int]]:
    events_list = list(events)
    n = len(events_list)
    if rate <= 0.0 or n == 0:
        return events_list, {"events_before": n, "events_after": n}
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"event drop rate must be in [0, 1], got {rate}")
    if mode not in {"random", "window"}:
        raise ValueError(f"unknown temporal drop mode: {mode}")

    if mode == "random":
        keep = torch.rand(n, generator=rng) > rate
        dropped = [row for row, k in zip(events_list, keep.tolist()) if k]
    else:
        drop_len = window_size if window_size is not None else max(1, int(round(n * rate)))
        drop_len = max(1, min(int(drop_len), n))
        start_max = max(n - drop_len, 0)
        start = int(torch.randint(0, start_max + 1, (1,), generator=rng).item())
        dropped = events_list[:start] + events_list[start + drop_len :]

        # if window_size is explicitly given, also match drop-rate target by random filtering remainder
        if window_size is not None and rate > 0:
            target_keep = max(int(round(n * (1.0 - rate))), 0)
            if len(dropped) > target_keep:
                perm = torch.randperm(len(dropped), generator=rng)
                keep_idx = set(perm[:target_keep].tolist())
                dropped = [row for idx, row in enumerate(dropped) if idx in keep_idx]

    return dropped, {"events_before": n, "events_after": len(dropped)}
