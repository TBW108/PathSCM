"""Keep the K pathway nodes with the most perturbed members (design information only, no expression) and the
perturbations whose target lies in at least one kept node; write a data file in the same format.

    python data/_subset.py --data <h5ad> --nodes 64 --output <h5ad>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "datasets/processed/k562_generative_v3.h5ad"))
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = ad.read_h5ad(args.data)
    support = np.asarray(data.uns["support"]) > 0                       # [conditions, nodes], row 0 = control
    perturbed = support[1:].sum(0)
    kept = np.sort(np.argsort(-perturbed, kind="stable")[:args.nodes])
    new_support = support[:, kept]
    keep_condition = new_support.any(1)
    keep_condition[0] = True
    old_to_new = -np.ones(len(support), dtype=np.int64)
    old_to_new[keep_condition] = np.arange(keep_condition.sum())
    cells = keep_condition[data.obs["condition_index"].to_numpy()]
    out = data[cells].copy()
    out.obs["condition_index"] = old_to_new[out.obs["condition_index"].to_numpy()]
    out.varm["node_mask"] = np.asarray(data.varm["node_mask"])[:, kept]
    out.uns["support"] = new_support[keep_condition].astype(np.int8)
    out.uns["conditions"] = np.asarray(data.uns["conditions"])[keep_condition]
    out.uns["node_names"] = np.asarray(data.uns["node_names"])[kept]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output)
    print(f"nodes {len(kept)}, conditions {int(keep_condition.sum())} (of {len(support)}), cells {int(cells.sum())}, "
          f"perturbed members per node {int(perturbed[kept].min())}-{int(perturbed[kept].max())}, "
          f"genes in a node {int((out.varm['node_mask'] > 0).any(1).sum())}")


if __name__ == "__main__":
    main()
