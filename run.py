"""Explicit, foreground commands for training and scoring current benchmarks."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=['PathSCM','GEARS','scGen','discrepancy-VAE','SENA']

def train_command(ds,method,seed,device,protocol='seen'):
    from evaluation.protocol import DATASETS
    cfg=DATASETS[ds];data=cfg['data'];reference=cfg['reference']
    if protocol=='unseen':
        from evaluation import unseen
        return list(map(str,unseen.pathscm_command(ds,seed,device) if method=='PathSCM' else unseen.gears_command(ds,seed,device)))
    if method=='PathSCM':
        a=json.loads((ROOT/f'configs/{ds}.json').read_text());a.update(seed=seed,device=device,run_name=cfg['Ours'][seed],data=data)
        cmd=[sys.executable,'-m','pathscm.train']
        for k,v in a.items():
            if v is None or v is False:continue
            cmd+=['--'+k.replace('_','-')]
            if v is not True:cmd+=[str(v)]
        return cmd
    if method=='GEARS':
        return [os.environ.get('PATHSCM_GEARS_PYTHON',str(ROOT/'.venv-gears/bin/python')),'-m','baselines.gears','--data',reference,'--name',f'{ds}_generative_v3','--output',cfg['GEARS'][seed][4:],'--seed',str(seed),'--device',device,'--batch-size','256','--epochs','20','--lr','0.001','--hidden-size','64']
    if method=='scGen':
        return [os.environ.get('PATHSCM_SCGEN_PYTHON',str(ROOT/'.venv-scgen/bin/python')),'-m','baselines.scgen','--data',data,'--reference',reference,'--output',cfg['scGen'][seed][6:],'--seed',str(seed),'--batch-size','32','--max-epochs','100','--patience','25']
    return [sys.executable,'-m','baselines.discrepancy_vae' if method=='discrepancy-VAE' else 'baselines.sena','--data',data,'--out',str(ROOT/'checkpoints/crl'/cfg[method][seed]),'--seed',str(seed),'--device',device,'--latdim','64']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','evaluate','summarize','prepare-unseen']);p.add_argument('--dataset',choices=['k562','rpe1','hepg2'],default='k562');p.add_argument('--method',choices=METHODS+['all'],default='all');p.add_argument('--seeds',type=int,nargs='+',choices=[0,1,2],default=[0,1,2]);p.add_argument('--device',default='cuda');p.add_argument('--protocol',choices=['seen','unseen'],default='seen');p.add_argument('--dry-run',action='store_true');p.add_argument('--limit',type=int,help='Seen evaluation only: first N frozen targets, writes results/smoke');a=p.parse_args()
    if a.limit is not None and (a.limit<=0 or a.action!='evaluate' or a.protocol!='seen'):p.error('--limit is positive and only supported for seen evaluation')
    if a.action=='summarize':
        from evaluation.summarize import summarize
        if a.protocol=='unseen':
            import pandas as pd
            folder=ROOT/'results/unseen/results'
            for f in folder.glob('*_seed[012].csv'):
                df=pd.read_csv(f)
                if 'gene' in df:df.rename(columns={'gene':'perturbation'}).to_csv(f,index=False)
            summarize(folder)
        else:summarize(ROOT/'results/seen')
        return
    if a.action=='prepare-unseen':
        from evaluation.unseen import prepare
        prepare();return
    methods=METHODS if a.method=='all' else [a.method]
    if a.protocol=='unseen':
        if a.method=='all':methods=['PathSCM','GEARS']
        if any(m not in ['PathSCM','GEARS'] for m in methods):p.error('Unseen protocol supports PathSCM and GEARS only')
    if a.action=='evaluate':
        if a.dry_run:p.error('--dry-run is for training commands; use --limit for a small seen evaluation')
        if a.protocol=='unseen':
            from evaluation.unseen import evaluate
            for seed in a.seeds:evaluate(a.dataset,seed,a.device)
        else:
            from evaluation.seen import evaluate
            for seed in a.seeds:
                for method in methods:evaluate(a.dataset,method,seed,a.device,a.limit)
        return
    from evaluation.protocol import DATASETS
    for seed in a.seeds:
        for method in methods:
            cmd=train_command(a.dataset,method,seed,a.device,a.protocol);print(shlex.join(cmd),flush=True)
            if a.dry_run:continue
            cfg=DATASETS[a.dataset]
            if a.protocol=='unseen':
                from evaluation.unseen import run_name,prediction_path
                destination=ROOT/'checkpoints/pathscm'/run_name(a.dataset,seed) if method=='PathSCM' else prediction_path(a.dataset,seed)
            elif method=='PathSCM':destination=ROOT/'checkpoints/pathscm'/cfg['Ours'][seed]
            elif method in ['GEARS','scGen']:destination=ROOT/cfg[method][seed].split(':',1)[1]
            else:destination=ROOT/'checkpoints/crl'/cfg[method][seed]
            if destination.exists():raise FileExistsError(f'Refusing to overwrite {destination}')
            subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
