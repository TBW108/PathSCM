"""Original seen evaluation: frozen targets, original sampling and training PCA."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data.dataset import load_generative_data
from evaluation.metrics import mmd
from evaluation.predict import generator_for
from evaluation.protocol import DATASETS
ROOT=Path(__file__).resolve().parents[1]
METHODS={'PathSCM':'Ours','GEARS':'GEARS','scGen':'scGen','discrepancy-VAE':'discrepancy-VAE','SENA':'SENA'}

def corr_rows(a,b):
    a,b=a-a.mean(-1,keepdim=True),b-b.mean(-1,keepdim=True)
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)

def effect_metrics(predicted,truth):
    return corr_rows(predicted,truth),((predicted-truth)**2).mean(-1)

def evaluate(ds,method,seed,device='cuda',limit=None):
    cfg=DATASETS[ds]
    data=load_generative_data(str(ROOT/cfg['data']))
    ref=load_generative_data(str(ROOT/cfg['reference']))
    assert np.array_equal(data.genes,ref.genes)
    spec=json.loads((ROOT/f'configs/splits/{ds}_seen.json').read_text())
    conditions=spec['evaluation_conditions']
    if limit:conditions=conditions[:limit]
    X=torch.as_tensor(ref.X,device=device)
    train,test=ref.split=='train',ref.split=='test'
    controls=X[torch.as_tensor(np.flatnonzero(train&(ref.condition==0)),device=device)]
    cm=controls.mean(0);index={c:i for i,c in enumerate(ref.conditions)};local={c:i for i,c in enumerate(data.conditions)}
    pool=X[torch.as_tensor(np.random.default_rng(0).choice(np.flatnonzero(train),50000,replace=False),device=device)]
    center=pool.mean(0);basis=torch.linalg.svd(pool-center,full_matrices=False)[2][:50].T
    pcs=lambda x:(x-center)@basis
    model=generator_for(cfg[METHODS[method]][seed],data,device,controls)
    if method=='GEARS':
        with np.load(ROOT/cfg['GEARS'][seed][4:]) as z:
            seen=set(z['genes'][z['set']=='seen'])
            if not set(conditions)<=seen:raise ValueError('GEARS split does not match frozen evaluation conditions')
    rows=[]
    with torch.no_grad():
        base=model.sample_population(model.condition_input(0),1000,seed=10000)['mean'].float().mean(0)
        for c in conditions:
            rr=np.flatnonzero(test&(ref.condition==index[c]));assert len(rr)>=10
            real=X[torch.as_tensor(rr,device=device)]
            truth=real.mean(0)-cm;t=local[c]
            effect=model.sample_population(model.condition_input(t),1000,seed=10000+t)['mean'].float().mean(0)-base
            sample=model.sample_population(model.condition_input(t),min(256,len(rr)),seed=t)['sample'].float()
            pearson,mse=effect_metrics(effect,truth)
            rows.append(dict(dataset=ds,method=method,seed=seed,perturbation=c,pearson_delta=float(pearson),mse=float(mse),mmd=float(mmd(pcs(sample),pcs(real[:256])))))
    folder=ROOT/'results'/('smoke' if limit else 'seen');folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'{ds}_{method}_seed{seed}.csv';pd.DataFrame(rows).to_csv(dest,index=False);print(dest)
    return rows
