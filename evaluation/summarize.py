"""Aggregate perturbations within seeds, then report sample SD across seeds."""
from pathlib import Path
import pandas as pd
METRICS=['pearson_delta','mse','mmd']

def summarize(folder):
    folder=Path(folder)
    files=sorted(folder.glob('*_seed[012].csv'))
    if not files:raise FileNotFoundError(f'No per-seed CSV files in {folder}')
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if df.duplicated(['dataset','method','seed','perturbation']).any():raise ValueError('Duplicate metric rows')
    if df[METRICS].isna().any().any():raise ValueError('Missing metrics')
    for _,group in df.groupby(['dataset','method']):
        if set(group.seed)!={0,1,2}:raise ValueError('Final summary requires seeds 0, 1, 2')
        sets=[set(x.perturbation) for _,x in group.groupby('seed')]
        if any(x!=sets[0] for x in sets):raise ValueError('Perturbation sets differ across seeds')
    df.to_csv(folder/'per_perturbation.csv',index=False)
    per=df.groupby(['dataset','method','seed'])[METRICS].mean().reset_index()
    per.to_csv(folder/'per_seed.csv',index=False)
    summary=per.groupby(['dataset','method'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(folder/'summary.csv',index=False)
    return summary
