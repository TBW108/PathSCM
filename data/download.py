"""Download original inputs. Existing files are never overwritten."""
import argparse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'k562':('https://zenodo.org/records/7416068/files/ReplogleWeissman2022_K562_essential.h5ad?download=1','replogle_2022_k562_essential.h5ad'),
 'rpe1':('https://dataverse.harvard.edu/api/access/datafile/7458694','replogle_rpe1_essential.zip'),
 'hepg2':('https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main/GSE264667_hepg2_raw_singlecell_01.h5ad','hepg2_raw_singlecell_01.h5ad')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=SOURCES);a=p.parse_args();url,name=SOURCES[a.dataset]
 dest=ROOT/'datasets/raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 part=dest.with_suffix(dest.suffix+'.download');urllib.request.urlretrieve(url,part);part.replace(dest);print(dest)
if __name__=='__main__':main()
