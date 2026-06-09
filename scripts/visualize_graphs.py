#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


DATASETS = {
    "NF-BoT-IoT": {
        "graph_dir": Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph"),
        "image_dir": Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph_Images"),
    },
    "NF-ToN-IoT": {
        "graph_dir": Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph"),
        "image_dir": Path("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-ToN-IoT/Graph_Images"),
    },
}


def load_npz_as_networkx(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    edge_index = data["edge_index"]
    node_features = data["node_features"]
    label = int(data["label"][0]) if "label" in data else -1

    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    for src, dst in edge_index.T:
        G.add_edge(int(src), int(dst))

    return G, label, num_nodes, num_edges


def label_to_name(label):
    if label == 0:
        return "benign"
    elif label == 1:
        return "malicious"
    else:
        return f"label_{label}"


def draw_graph(npz_path, image_dir):
    G, label, num_nodes, num_edges = load_npz_as_networkx(npz_path)

    label_name = label_to_name(label)

    label_image_dir = image_dir / label_name
    label_image_dir.mkdir(parents=True, exist_ok=True)

    out_path = label_image_dir / f"{npz_path.stem}.png"

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    show_labels = num_nodes <= 40

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=show_labels,
        node_size=500 if show_labels else 80,
        font_size=8,
        arrows=True,
        arrowsize=12,
        width=1.2,
    )

    plt.title(
        f"{npz_path.name} | {label_name} | nodes={num_nodes} | edges={num_edges}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path, label, label_name, num_nodes, num_edges


def visualize_dataset(dataset_name, graph_dir, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(graph_dir.glob("*.npz"))

    if not graph_files:
        print(f"No .npz files found in: {graph_dir}")
        return

    summary_path = image_dir / "graph_visualization_summary.csv"

    print(f"\nDataset: {dataset_name}")
    print(f"Graph directory: {graph_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Found {len(graph_files)} graph files")

    with open(summary_path, "w") as f:
        f.write("dataset,file,label,label_name,num_nodes,num_edges,image\n")

        for i, npz_path in enumerate(graph_files, start=1):
            try:
                out_path, label, label_name, num_nodes, num_edges = draw_graph(
                    npz_path=npz_path,
                    image_dir=image_dir,
                )

                f.write(
                    f"{dataset_name},{npz_path.name},{label},{label_name},"
                    f"{num_nodes},{num_edges},{out_path}\n"
                )

                print(
                    f"[{i}/{len(graph_files)}] {npz_path.name} "
                    f"| {label_name} | nodes={num_nodes} | edges={num_edges}"
                )

            except Exception as e:
                print(f"[{i}/{len(graph_files)}] ERROR {npz_path.name}: {e}")

    print(f"Summary saved to: {summary_path}")


def main():
    for dataset_name, paths in DATASETS.items():
        visualize_dataset(
            dataset_name=dataset_name,
            graph_dir=paths["graph_dir"],
            image_dir=paths["image_dir"],
        )

    print("\nDone.")


if __name__ == "__main__":
    main()