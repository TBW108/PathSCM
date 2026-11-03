"""Two CPU optimizer steps on synthetic cells; temporary weights are removed."""
import json,sys,tempfile,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch,anndata as ad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pathscm.model import ResponseGenerator

def main():
    rng=np.random.default_rng(7);G,K,C=64,4,4
    cond=np.repeat(np.arange(C),672);split=np.tile(['train']*640+['val']*16+['test']*16,C)
    mask=np.zeros((G,K),np.float32)
    for k in range(K):mask[k*16:(k+1)*16,k]=1
    support=np.zeros((C,K),np.float32);support[1:, :3]=np.eye(3)
    x=rng.normal(size=(len(cond),G)).astype(np.float32)+cond[:,None].astype(np.float32)*.05
    a=ad.AnnData(x,obs=pd.DataFrame({'condition_index':cond,'split':split},index=[f'cell{i}' for i in range(len(cond))]),var=pd.DataFrame({'train_mean':np.zeros(G,np.float32),'train_sd':np.ones(G,np.float32)},index=[f'G{i}' for i in range(G)]))
    a.varm['node_mask']=mask;a.uns.update(conditions=np.array(['control','G0','G16','G32']),node_names=np.array([f'P{k}' for k in range(K)]),support=support)
    with tempfile.TemporaryDirectory(prefix='pathscm-smoke-') as tmp:
        tmp=Path(tmp);a.write_h5ad(tmp/'toy.h5ad')
        config=json.loads((ROOT/'configs/k562.json').read_text());config.update(data=str(tmp/'toy.h5ad'),run_name=str(tmp/'model'),device='cpu',graph_steps=2,validate_every=1,conditions_per_step=2,cells=16)
        cmd=[sys.executable,'-m','pathscm.train']
        for k,v in config.items():
            if v is None or v is False:continue
            cmd+=['--'+k.replace('_','-')]
            if v is not True:cmd+=[str(v)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        model=ResponseGenerator(tmp/'model','cpu',mask,torch.tensor(x[(cond==0)&(split=='train')]))
        y=model.sample_population(model.condition_input(1),8,seed=0)['sample']
        assert y.shape==(8,G) and torch.isfinite(y).all()
        print('PASS: two training steps, checkpoint reload, and finite predictions')
if __name__=='__main__':main()
