"""Bring another Perturb-seq screen into the input format of data_protocol.py (obs: perturbation, batch, guide_id).

    rpe1   Replogle et al. 2022 RPE1 essential screen, GEARS-processed (log-normalized, 5,000 genes; obs 'condition'
           'GENE+ctrl' / 'ctrl'); the gem group is the suffix of the cell barcode. Run data_protocol.py with
           --log-normalized.
    nadig  Nadig et al. 2025 essential screens (GEO GSE264667, *_raw_singlecell_01.h5ad: raw counts, Replogle-lab
           obs with gene / gem_group / sgID_AB). Only cells with a single targeting guide or a non-targeting guide.

    python data/_external.py rpe1 --input <h5ad> --output data/rpe1_full.h5ad
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["rpe1", "nadig"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = ad.read_h5ad(args.input)
    if args.format == "rpe1":
        condition = data.obs["condition"].astype(str).to_numpy()
        perturbation = np.where(condition == "ctrl", "control", np.char.replace(condition.astype(str), "+ctrl", ""))
        batch = np.array([int(b.rsplit("-", 1)[-1]) for b in data.obs_names])
        genes = data.var["gene_name"].astype(str).to_numpy()
        guide = condition
    else:
        obs = data.obs
        gene = obs["gene"].astype(str).to_numpy()
        perturbation = np.where(np.isin(gene, ["non-targeting", "nan", ""]), "control", gene)
        batch = obs["gem_group"].astype(int).to_numpy()
        guide = obs["sgID_AB"].astype(str).to_numpy() if "sgID_AB" in obs else gene
        genes = (data.var["gene_name"] if "gene_name" in data.var else data.var_names.to_series()).astype(str).to_numpy()
        keep = np.char.find(perturbation.astype(str), "_") < 0      # single-gene perturbations and controls only
        data, perturbation, batch, guide = data[keep], perturbation[keep], batch[keep], guide[keep]
    out = ad.AnnData(X=data.X, obs=pd.DataFrame({"perturbation": perturbation, "batch": batch, "guide_id": guide},
                                                 index=data.obs_names.astype(str)))
    out.var_names = genes
    out.var_names_make_unique()
    out.write_h5ad(args.output)
    print(out, "conditions", len(np.unique(perturbation)), "control cells", int((perturbation == "control").sum()),
          "gem groups", len(np.unique(batch)))


if __name__ == "__main__":
    main()
