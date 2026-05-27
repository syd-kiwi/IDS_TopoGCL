from pathlib import Path
import pandas as pd

df = pd.read_csv("/home/kiwi-pandas/Documents/IDS_TopoGCL/datasets/NF-BoT-IoT/Graph/graph_window_summary.csv")

print("Graphs:", len(df))
print("\nLabels:")
print(df["graph_label"].value_counts())

print("\nNodes:")
print(df["num_nodes"].describe())

print("\nEdges:")
print(df["num_edges"].describe())

print("\nAttack flows:")
print(df["num_attack_flows"].describe())

print("\nBenign flows:")
print(df["num_benign_flows"].describe())