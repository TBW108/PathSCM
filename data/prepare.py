"""Reproduce full-node preparation followed by the original top-64 subset."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['k562','rpe1','hepg2'],required=True);p.add_argument('--input',required=True);p.add_argument('--converted',action='store_true');a=p.parse_args()
    raw=ROOT/'datasets/raw';out=ROOT/'datasets/processed';raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    full=Path(a.input).resolve()
    reference=out/f'{a.dataset}_generative_v3.h5ad';subset=out/f'{a.dataset}_generative_v3_k64.h5ad'
    if reference.exists() or subset.exists():raise FileExistsError('Prepared files already exist; refusing overwrite')
    def run(args):subprocess.run([sys.executable,'-m',*map(str,args)],cwd=ROOT,check=True)
    if a.dataset!='k562' and not a.converted:
        full=raw/f'{a.dataset}_full.h5ad'
        if full.exists():raise FileExistsError(full)
        run(['data._external','rpe1' if a.dataset=='rpe1' else 'nadig','--input',Path(a.input).resolve(),'--output',full])
    cmd=['data._protocol','--full',full,'--gmt',ROOT/'configs/pathways/ReactomePathways.gmt','--output',reference,'--n-hvg','2000','--genes','hvg','--min-cells','50','--min-pathway-genes','10','--max-pathway-genes','0','--max-pathway-jaccard','0.5','--n-pathways','0','--exclude-top-level',ROOT/'configs/pathways/ReactomePathwaysRelation.txt','--seed','0','--val-groups','8','--test-groups','8']
    if a.dataset=='rpe1':cmd+=['--log-normalized']
    run(cmd);run(['data._subset','--data',reference,'--nodes','64','--output',subset])
    from data.verify import verify
    verify(a.dataset)

if __name__=='__main__':main()
