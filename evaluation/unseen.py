"""Audited strict unseen protocol, without plotting or background queues."""
import argparse

import csv

import hashlib

import json

import os

from pathlib import Path

import pickle

import re

import shutil

import subprocess

import sys

import time

import traceback

import numpy as np

import torch

import anndata as ad

from evaluation.protocol import DATASETS

from data.dataset import load_generative_data, gene_split_masks

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT/'results/unseen'

PY = Path(sys.executable)

GPY = Path(os.environ.get('PATHSCM_GEARS_PYTHON',str(ROOT/'.venv-gears/bin/python')))

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n'); tmp.replace(path)

def data_path(ds): return OUT/'data'/f'{ds}_unseen_k64.h5ad'

def split_path(ds): return OUT/f'{ds}_strict_gene_split.json'

def run_name(ds, seed): return f'unseen20261101_{ds}_seed{seed}'

def prediction_path(ds, seed): return OUT/'gears'/f'{ds}_seed{seed}.npz'

def command(args, log, status):
    log.parent.mkdir(parents=True, exist_ok=True)
    dump(OUT/'status.json', dict(status='running', pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                job=status, command=list(map(str,args)), log=str(log)))
    print('START', status, flush=True)
    with log.open('w') as f:
        p = subprocess.Popen(list(map(str,args)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        dump(OUT/'active_job.json', dict(pid=p.pid,job=status,log=str(log)))
        code = p.wait()
    if code: raise RuntimeError(f'{status} exited {code}; see {log}')
    print('DONE', status, flush=True)

def prepare():
    (OUT/'data').mkdir(parents=True, exist_ok=True)
    for source in (ROOT/'configs/splits/unseen').glob('*gene_split.json'):
        destination = OUT/source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
    with (ROOT/'baselines/gears_data/gene2go_all.pkl').open('rb') as f: go = pickle.load(f)
    with (ROOT/'baselines/gears_data/essential_all_data_pert_genes.pkl').open('rb') as f: essential = pickle.load(f)
    allowed = set(go) & set(essential)
    full = dict(k562='replogle_2022_k562_essential.h5ad',rpe1='rpe1_full.h5ad',hepg2='hepg2_full.h5ad')
    summary = {}
    for ds in DATASETS:
        spec = json.loads((OUT/f'{ds}_gene_split.json').read_text())
        spec['excluded_not_in_gears_prior'] = {}
        for key in ('seen','heldout_tune','heldout_test'):
            spec['excluded_not_in_gears_prior'][key] = sorted(set(spec[key])-allowed)
            spec[key] = sorted(set(spec[key]) & allowed)
        spec['status'] = 'frozen prior to training; common GEARS prior coverage'
        if split_path(ds).exists():
            assert json.loads(split_path(ds).read_text()) == spec, 'Refuse to change frozen split'
        else: dump(split_path(ds), spec)
        if not data_path(ds).exists():
            cmd = [PY, ROOT/'data/_protocol.py', '--full', ROOT/'datasets/raw'/full[ds],
                   '--output',data_path(ds),'--gene-split',split_path(ds),'--n-hvg','2000','--genes','hvg',
                   '--min-cells','50','--min-pathway-genes','10','--max-pathway-genes','0',
                   '--max-pathway-jaccard','0.5','--n-pathways','64','--exclude-top-level',
                   ROOT/'configs/pathways/ReactomePathwaysRelation.txt']
            if ds=='rpe1': cmd += ['--log-normalized']
            command(cmd,OUT/'logs'/f'prepare_{ds}.log',f'prepare/{ds}')
        d = ad.read_h5ad(data_path(ds),backed='r')
        masks = gene_split_masks(d.uns['conditions'],split_path(ds))
        labels=d.obs.condition_index.to_numpy();train=d.obs.split.astype(str).to_numpy()=='train'
        scaling=d.obs.used_for_scaling.to_numpy()
        assert np.array_equal(scaling, train & ((labels==0)|masks[0][labels]))
        assert d.uns['gene_split_sha256']==hashlib.sha256(split_path(ds).read_bytes()).hexdigest()
        assert len(d.uns['node_names'])==64 and d.shape[1]==2000
        # Training statistics check without ever fitting on held-out response rows.
        x=np.asarray(d.X[np.flatnonzero(scaling)],np.float32)
        assert np.max(np.abs(x.mean(0,dtype=np.float64)))<1e-5
        assert np.max(np.abs(x.std(0,dtype=np.float64)-1))<1e-4
        summary[ds]=dict(n_cells=d.n_obs,n_genes=d.n_vars,n_pathways=64,
                        gene_counts=dict(zip(['seen','tune','test'],[int(m.sum()) for m in masks])),
                        n_scaling_cells=int(scaling.sum()),excluded_no_support=list(map(str,d.uns['excluded_conditions'])),
                        data_sha256=hashlib.sha256(data_path(ds).read_bytes()).hexdigest(),
                        split_sha256=d.uns['gene_split_sha256'])
        d.file.close();del x
    dump(OUT/'prepared_data_validation.json', summary)
    return summary

def pathscm_command(ds, seed, device, smoke=False):
    cp=ROOT/'checkpoints/pathscm'/DATASETS[ds]['Ours'][seed]/'response.pth'
    a=(torch.load(cp,map_location='cpu',weights_only=False)['args'].copy() if cp.exists()
       else json.loads((ROOT/'configs'/f'{ds}.json').read_text()))
    a.update(data=str(data_path(ds)),run_name=run_name(ds,seed),seed=seed,device=device,
             holdout_genes=0.,holdout_folds=0,gene_split=str(split_path(ds)))
    if smoke: a.update(run_name=run_name(ds,seed)+'_smoke',graph_steps=2,validate_every=1,conditions_per_step=2,cells=16)
    cmd=[PY,ROOT/'pathscm/train.py']
    for k,v in a.items():
        if v is None or v is False:continue
        cmd += ['--'+k.replace('_','-')]
        if v is not True:cmd += [str(v)]
    return cmd

def gears_command(ds,seed,device,smoke=False):
    return [GPY,ROOT/'baselines/gears.py','--data',data_path(ds),'--gene-split',split_path(ds),
            '--cell-type',ds,'--name',f'unseen20261101_{ds}','--output',prediction_path(ds,seed),
            '--epochs','20','--batch-size','256','--hidden-size','64','--lr','0.001','--seed',str(seed),'--device',device]

def evaluate(ds,seed,device):
    from pathscm.transfer import UnseenResponseGenerator
    from baselines.adapters.mean import MeanShiftGenerator
    from evaluation.metrics import mmd
    from sklearn.decomposition import PCA
    d=load_generative_data(str(data_path(ds)));seen,tune,test=gene_split_masks(d.conditions,split_path(ds))
    pred=np.load(prediction_path(ds,seed));assert np.array_equal(pred['var_names'],d.genes)
    supported=set(pred['genes'][pred['set']=='heldout_test'])
    counts=np.bincount(d.condition[d.split=='test'],minlength=len(d.conditions))
    ts=[int(t) for t in np.flatnonzero(test) if counts[t]>=10 and d.conditions[t] in supported]
    if not ts:raise ValueError('No common test conditions')
    ctrl_idx=np.flatnonzero((d.split=='train')&(d.condition==0))
    controls=torch.as_tensor(d.X[ctrl_idx],device=device);control_mean=controls.mean(0)
    checkpoint=ROOT/'checkpoints/pathscm'/run_name(ds,seed)
    model=UnseenResponseGenerator(checkpoint,device,d.node_mask,controls,seen,tune|test)
    gears=MeanShiftGenerator(str(prediction_path(ds,seed)),d,device,controls)
    cache=OUT/'data'/f'{ds}_evaluation_pca.npz'
    if not cache.exists():
        rr=np.flatnonzero((d.split=='train')&((d.condition==0)|seen[d.condition]))
        rr=np.random.default_rng(0).choice(rr,min(20000,len(rr)),replace=False)
        pca=PCA(n_components=50,svd_solver='randomized',random_state=0).fit(d.X[rr])
        np.savez(cache,mean=pca.mean_,basis=pca.components_.T)
    z=np.load(cache);center=torch.as_tensor(z['mean'],device=device);basis=torch.as_tensor(z['basis'],device=device)
    pcs=lambda x:(x-center)@basis
    def corr(a,b):
        a=a-a.mean();b=b-b.mean();return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))
    rows=[];effects={};truth=[]
    with torch.no_grad():
        base=model.sample_population(model.condition_input(0),2048,seed=10000)['mean'].mean(0)
        effects={k:[] for k in ['PathSCM','GEARS']}
        for t in ts:
            rr=np.flatnonzero((d.split=='test')&(d.condition==t));real=torch.as_tensor(d.X[rr],device=device)
            R=real.mean(0)-control_mean;de=R.abs().topk(20).indices;truth.append(R.cpu().numpy())
            take=np.random.default_rng(t).choice(len(rr),min(len(rr),256),replace=False);real_small=real[take]
            for method,m in [('PathSCM',model),('GEARS',gears)]:
                E=(model.sample_population(model.condition_input(t),2048,seed=10000)['mean'].mean(0)-base
                   if method=='PathSCM' else gears.effect[d.conditions[t]])
                sample=m.sample_population(m.condition_input(t),len(take),seed=t)['sample'].float()
                rows.append(dict(dataset=ds,seed=seed,method=method,gene=d.conditions[t],n_test_cells=len(rr),
                    pearson_delta=corr(E,R),mse=float(((E-R)**2).mean()),
                    pearson_de20=corr(E[de],R[de]),mse_de20=float(((E[de]-R[de])**2).mean()),
                    mmd=float(mmd(pcs(sample),pcs(real_small)))))
                effects[method].append(E.cpu().numpy())
        common_truth=np.stack(truth)
        for method in effects:
            es=np.stack(effects[method]);es-=es.mean(0);rs=common_truth-common_truth.mean(0)
            for row,e,r in zip([x for x in rows if x['method']==method],es,rs):
                row['specific']=corr(torch.from_numpy(e),torch.from_numpy(r))
    (OUT/'results').mkdir(exist_ok=True)
    dest=OUT/'results'/f'{ds}_seed{seed}.csv'
    with dest.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(dest.with_suffix('.npz'),genes=np.array([d.conditions[t] for t in ts]),
                        truth=common_truth,PathSCM=np.stack(effects['PathSCM']),GEARS=np.stack(effects['GEARS']))
    dump(dest.with_suffix('.json'),dict(dataset=ds,seed=seed,n_common=len(ts),
         zero_jaccard_targets=[d.conditions[t] for t,ok in zip(model.transfer_target_indices,model.transfer_covered.cpu().numpy()) if not ok],
         de20='Top absolute ground-truth test mean effect, used only for evaluation',
         distribution='GEARS uses train-control plus native mean effect; not its own generative model'))
    plot()
