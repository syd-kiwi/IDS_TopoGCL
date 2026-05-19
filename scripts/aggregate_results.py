import json,glob,os
import pandas as pd
rows=[]
for p in glob.glob('results/**/*.json',recursive=True):
    with open(p) as f:
        d=json.load(f)
    m=d.get('metrics',{})
    rows.append({
        'path':p,'dataset':d.get('dataset'),'method':d.get('method'),'train_scenario':d.get('train_scenario'),
        'test_scenario':d.get('test_scenario'),'train_degradation_rate':d.get('train_degradation_rate'),'test_degradation_rate':d.get('test_degradation_rate'),
        'accuracy':m.get('accuracy'),'precision':m.get('precision'),'recall':m.get('recall'),'f1':m.get('f1'),'fpr':m.get('fpr'),'auroc':m.get('auroc')
    })
df=pd.DataFrame(rows)
os.makedirs('results',exist_ok=True)
df.to_csv('results/summary_metrics.csv',index=False)
for sc in ['clean','low_volume','missing_structure','interference']:
    df[df['test_scenario']==sc].to_csv(f'results/table_{sc}.csv',index=False)
print('wrote aggregation tables')
