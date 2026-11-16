# Saved experiment results

These directories preserve historical experimental outputs. Do not edit numerical values by hand. Write new runs to `outputs/` at the repository root.

## TRACE results used in the manuscript

| Dataset | Summary CSV | Seeds |
| --- | --- | --- |
| StreamSpot and GraSec-IoT | [Joint graph/topology summary](topids_streamspot_grasec/topids_all_graph_topo_summary.csv) | 42, 43, 44 |
| Unicorn Wget | [Joint graph/topology summary](topids_wget_trace/topids_wget_graph_topo_summary.csv) | 42–46 |

The corresponding `*_results.json` files contain configurations and per-seed results; `*_scores.csv` files contain validation/test scores and labels.

`topids_streamspot_grasec/` also includes `graph_only` and `topo_only` ablations. The directory name is historical; the joint improved model is TRACE.

## Baseline and historical outputs

| Location | Contents |
| --- | --- |
| `results/ids_gnn/` (nested under this directory) | Archived StreamSpot/GraSec-IoT supervised GNN runs |
| `results/ids_cl_baselines/` (nested under this directory) | Archived StreamSpot/GraSec-IoT contrastive runs, including GraphCL and historical InfoGraph |
| `ids_gnn/wget_*` | Unicorn Wget supervised GNN outputs |
| `ids_cl_baselines/wget_*` | Unicorn Wget GraphCL outputs |
| `topids_topogcl/` and `results/topids_topogcl/` | Earlier prototype runs |
| `plots/` | Saved score plots |
| `../scripts/results/` | Additional historical baseline outputs |

The nested `results/results/` and `scripts/results/` locations are retained to preserve existing provenance and references. They are not interchangeable duplicates of the top-level outputs. Check JSON configuration, seeds, split counts, and model names before comparing experiments. Earlier results may not match the current scripts or manuscript.
