"""Fetch pinned official baseline source trees; never modify an existing checkout."""
import argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPOS={
 'GEARS':('https://github.com/snap-stanford/GEARS.git','f374e43e197b295016d80395d7a54ddb81cc6769','baselines/GEARS'),
 'discrepancy-VAE':('https://github.com/uhlerlab/discrepancy_vae.git','4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a','baselines/sota_src/discrepancy_vae'),
 'SENA':('https://github.com/ML4BM-Lab/SENA.git','d5511aa31982f42995d55610f590f5891b3d3da8','baselines/sota_src/SENA')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=list(REPOS)+['all'],default='all');a=p.parse_args()
 for method in REPOS if a.method=='all' else [a.method]:
  url,revision,relative=REPOS[method];dest=ROOT/relative
  if dest.exists():
   actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
   if actual!=revision:raise ValueError(f'{dest}: expected {revision}, found {actual}')
   continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['git','clone',url,str(dest)],check=True)
  subprocess.run(['git','-C',str(dest),'checkout','--detach',revision],check=True)
if __name__=='__main__':main()
