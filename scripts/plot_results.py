import os,pandas as pd,matplotlib.pyplot as plt
os.makedirs('figures',exist_ok=True)
df=pd.read_csv('results/summary_metrics.csv') if os.path.exists('results/summary_metrics.csv') else pd.DataFrame()
if df.empty: raise SystemExit('No summary_metrics.csv found. Run aggregate_results.py first.')
for sc in ['clean','low_volume','missing_structure','interference']:
    sub=df[df['test_scenario']==sc]
    for metric in (['auroc','f1'] if sc!='clean' else ['auroc']):
        plt.figure(figsize=(8,4))
        for (ds,m),g in sub.groupby(['dataset','method']):
            g=g.sort_values('test_degradation_rate')
            plt.plot(g['test_degradation_rate'],g[metric],label=f'{ds}-{m}')
        plt.title(f'{sc} {metric.upper()}')
        plt.legend(); plt.xlabel('test degradation rate'); plt.ylabel(metric)
        fn=f'figures/{sc}_{metric}.png' if sc!='clean' else 'figures/clean_comparison.png'
        plt.tight_layout(); plt.savefig(fn); plt.close()
