"""Check prepared identities, masks, scaling and splits against original data."""
import hashlib,json
from pathlib import Path
import anndata as ad
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def fingerprint(a):
    def strings(v):return hashlib.sha256('\n'.join(map(str,v)).encode()).hexdigest()
    def array(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    return dict(shape=list(a.shape),genes=strings(a.var_names),cells=strings(a.obs_names),split=strings(a.obs['split'].astype(str)),conditions=strings(a.uns['conditions']),nodes=strings(a.uns['node_names']),mask=array(np.asarray(a.varm['node_mask'],dtype=np.float32)),support=array(np.asarray(a.uns['support'],dtype=np.float32)),mean=array(a.var['train_mean'].to_numpy(dtype=np.float32)),sd=array(a.var['train_sd'].to_numpy(dtype=np.float32)))

def verify(ds):
    expected=json.loads((ROOT/f'configs/splits/{ds}_seen.json').read_text())
    for name in ['reference','data']:
        a=ad.read_h5ad(ROOT/expected[name]['path'],backed='r')
        actual=fingerprint(a);a.file.close()
        if actual!=expected[name]['fingerprint']:raise ValueError(f'{ds} {name}: preparation differs from original experiment')
    print(ds,'prepared identities, splits, masks and scaling match original experiment')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['k562','rpe1','hepg2']);verify(p.parse_args().dataset)
