import glob
import json
import os
import pandas as pd

rows = []
for path in sorted(glob.glob("results/edge_removal/*.json")):
    name = os.path.basename(path).replace('.json', '')
    dataset, method, _, rate = name.split('_', 3)
    with open(path) as f:
        d = json.load(f)
    m = d.get('metrics', {})
    rows.append({
        'dataset': dataset,
        'method': method,
        'edge_drop_rate': float(rate.replace('edge_', '')) if rate.startswith('edge_') else float(rate),
        'accuracy': m.get('accuracy'),
        'precision': m.get('precision'),
        'recall': m.get('recall'),
        'f1': m.get('f1'),
        'fpr': m.get('fpr'),
        'auroc': m.get('auroc'),
    })

df = pd.DataFrame(rows).sort_values(['dataset','edge_drop_rate','method'])
os.makedirs('results', exist_ok=True)
df.to_csv('results/edge_removal_summary.csv', index=False)

pivot = df.pivot_table(index=['dataset','edge_drop_rate'], columns='method', values='auroc').reset_index()
if 'topogcl' in pivot.columns and 'gnn' in pivot.columns:
    pivot['gap_topogcl_minus_gnn'] = pivot['topogcl'] - pivot['gnn']
if 'topogcl' in pivot.columns and 'svm' in pivot.columns:
    pivot['gap_topogcl_minus_svm'] = pivot['topogcl'] - pivot['svm']
pivot.to_csv('results/edge_removal_gaps.csv', index=False)
print('wrote results/edge_removal_summary.csv and results/edge_removal_gaps.csv')
