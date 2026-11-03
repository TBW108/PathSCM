"""Data protocol of the generative pathway SCM (specification, section 2).

Cells are split by gem group first. Everything that depends on expression
(highly variable genes, standardization) is fitted on the training cells
only. Pathway nodes are chosen by how many perturbed genes they contain,
which uses the experimental design, not expression.

Conditions are control and every single-gene knockdown with enough cells
whose target gene is listed by the database in at least one node. The
candidate direct support of a condition is the set of nodes that list its
target gene: an approximation of where the knockdown acts directly, stated
as a modelling assumption.

Output: an AnnData with standardized log-normalized expression of the
selected genes, the split of every cell, the node membership of every gene,
the support of every condition, and the parameters needed to undo the
standardization; plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pathways import pathway_membership, reactome_pathway_mask  # noqa: E402


def split_gem_groups(groups: np.ndarray, n_val: int, n_test: int, seed: int) -> dict[str, list[int]]:
    """Whole gem groups go to validation and test; the rest are training."""
    order = np.random.default_rng(seed).permutation(np.sort(np.unique(groups)))
    return {"val": sorted(order[:n_val].tolist()), "test": sorted(order[n_val:n_val + n_test].tolist()),
            "train": sorted(order[n_val + n_test:].tolist())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", default=str(ROOT / "data/replogle_2022_k562_essential.h5ad"))
    parser.add_argument("--gmt", default=str(ROOT / "configs/pathways/ReactomePathways.gmt"))
    parser.add_argument("--output", default=str(ROOT / "datasets/processed/k562_generative.h5ad"))
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--min-pathway-genes", type=int, default=15)
    parser.add_argument("--max-pathway-genes", type=int, default=150)
    parser.add_argument("--max-pathway-jaccard", type=float, default=0.5)
    parser.add_argument("--n-pathways", type=int, default=64, help="0 keeps every node")
    parser.add_argument("--exclude-top-level", default=None,
                        help="ReactomePathwaysRelation.txt: drop the top-level human pathways (no parent), e.g. Signal Transduction")
    parser.add_argument("--genes", choices=["hvg+targets", "hvg"], default="hvg+targets",
                        help="hvg: only the highly variable genes are measured; membership of targets still uses the full gene sets")
    parser.add_argument("--val-groups", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-normalized", action="store_true",
                        help="X is already log1p(counts per 10,000) (e.g. GEARS-processed data): skip the normalization")
    parser.add_argument('--gene-split', default=None, help='Name-based gene split; fit HVG/scaling only on seen train cells')
    args = parser.parse_args()

    full = sc.read_h5ad(args.full)
    labels = full.obs["perturbation"].astype(str).to_numpy()
    groups = full.obs["batch"].astype(int).to_numpy()
    is_control = labels == "control"
    gene_spec = json.loads(Path(args.gene_split).read_text()) if args.gene_split else None

    # split by gem group, before any fitting
    split_groups = split_gem_groups(groups, args.val_groups, args.test_groups, args.seed)
    split = np.full(len(labels), "train", dtype=object)
    split[np.isin(groups, split_groups["val"])] = "val"
    split[np.isin(groups, split_groups["test"])] = "test"

    # per-cell normalization (no fitted parameters): counts per 10,000, then log1p
    X = np.asarray(full.X.toarray() if hasattr(full.X, "toarray") else full.X, dtype=np.float32)
    if not args.log_normalized:
        totals = X.sum(axis=1, keepdims=True)
        for start in range(0, len(X), 20000):
            block = X[start:start + 20000]
            np.log1p(block / totals[start:start + 20000] * 1e4, out=block)
    genes_all = full.var_names.to_numpy().astype(str)

    # conditions with enough cells
    counts = pd.Series(labels[~is_control]).value_counts()
    targets = np.array(sorted(counts[counts >= args.min_cells].index))
    if gene_spec:
        universe = set().union(*(gene_spec[k] for k in ('seen','heldout_tune','heldout_test')))
        targets = targets[np.isin(targets, list(universe))]

    # highly variable genes on training cells only
    train_cells = split == "train"
    if gene_spec:
        train_cells &= is_control | np.isin(labels, gene_spec['seen'])
    train_view = ad.AnnData(X[train_cells])
    train_view.var_names = genes_all
    sc.pp.highly_variable_genes(train_view, n_top_genes=args.n_hvg, flavor="seurat")
    hvg = set(genes_all[train_view.var["highly_variable"].to_numpy()])
    del train_view
    measured_targets = {t for t in targets if t in set(genes_all)}
    candidates = np.array(sorted(hvg | measured_targets)) if args.genes == "hvg+targets" else np.array(sorted(hvg))

    # pathway nodes on the candidate genes, then the nodes with the most perturbed genes
    exclude = None
    if args.exclude_top_level:
        relation = [line.split("\t") for line in Path(args.exclude_top_level).read_text().splitlines() if line.startswith("R-HSA")]
        exclude = {p for p, _ in relation} - {c for _, c in relation}
    node_mask, node_names = reactome_pathway_mask(
        candidates, args.gmt, args.min_pathway_genes, args.max_pathway_genes, args.max_pathway_jaccard, exclude)
    target_membership = pathway_membership(targets, node_names, args.gmt)          # [targets, nodes]
    perturbed_members = target_membership.sum(axis=0)
    order = sorted(range(len(node_names)), key=lambda k: (-perturbed_members[k], node_names[k]))
    kept = sorted(order[:args.n_pathways] if args.n_pathways > 0 else order)
    node_mask, node_names = node_mask[:, kept], [node_names[k] for k in kept]
    support = target_membership[:, kept]
    has_support = support.sum(axis=1) > 0
    excluded = targets[~has_support].tolist()
    targets, support = targets[has_support], support[has_support]

    # cells: all control cells and the cells of the kept conditions
    conditions = ["control"] + targets.tolist()
    index_of = {c: i for i, c in enumerate(conditions)}
    keep_cells = is_control | np.isin(labels, targets)
    position = {g: i for i, g in enumerate(genes_all)}
    columns = np.array([position[g] for g in candidates])
    expression = X[np.ix_(np.flatnonzero(keep_cells), columns)]
    del X

    # standardization fitted on training cells only
    cell_split = split[keep_cells]
    train_rows = cell_split == "train"
    if gene_spec:
        train_rows &= is_control[keep_cells] | np.isin(labels[keep_cells], gene_spec['seen'])
    mean = expression[train_rows].mean(axis=0, dtype=np.float64)
    sd = expression[train_rows].std(axis=0, dtype=np.float64)
    sd[sd < 1e-6] = 1.0
    expression = ((expression - mean) / sd).astype(np.float32)

    obs = pd.DataFrame({
        "condition": labels[keep_cells],
        "condition_index": [index_of[c] for c in labels[keep_cells]],
        "gem_group": groups[keep_cells],
        "split": cell_split,
        "guide_id": full.obs["guide_id"].astype(str).to_numpy()[keep_cells],
    }, index=full.obs_names[keep_cells])
    var = pd.DataFrame({"train_mean": mean, "train_sd": sd, "in_node": node_mask.sum(axis=1) > 0,
                        "highly_variable": [g in hvg for g in candidates]}, index=candidates)
    out = ad.AnnData(X=expression, obs=obs, var=var)
    out.varm["node_mask"] = node_mask.astype(np.float32)
    support_matrix = np.vstack([np.zeros((1, len(node_names))), support]).astype(np.int8)    # control row is zero
    out.uns["node_names"] = np.array(node_names, dtype=object)
    out.uns["conditions"] = np.array(conditions, dtype=object)
    out.uns["support"] = support_matrix
    out.uns["perturbed_members_per_node"] = perturbed_members[kept].astype(np.int64)
    out.uns["split_groups"] = {k: np.array(v) for k, v in split_groups.items()}
    out.uns["excluded_conditions"] = np.array(excluded, dtype=object)
    if gene_spec:
        out.uns['gene_split_sha256'] = hashlib.sha256(Path(args.gene_split).read_bytes()).hexdigest()
        out.obs['used_for_scaling'] = train_rows
        out.uns['hvg_fit_cells'] = int(train_cells.sum())
        out.uns['scaling_fit_cells'] = int(train_rows.sum())
    out.uns["parameters"] = {k: v for k, v in vars(args).items() if k not in ("full", "output", "gmt")}
    out.write_h5ad(args.output)

    size = node_mask.sum(axis=0)
    shared = node_mask.T @ node_mask
    jaccard = shared / (size[:, None] + size[None, :] - shared)
    per_split = pd.crosstab(obs["condition"], obs["split"])
    summary = {
        "cells": int(len(obs)), "control_cells": int((obs["condition"] == "control").sum()),
        "cells_per_split": obs["split"].value_counts().to_dict(),
        "genes": int(len(candidates)), "genes_in_a_node": int(var["in_node"].sum()),
        "genes_without_node": [g for g, inside in zip(candidates, var["in_node"]) if not inside],
        "nodes": len(node_names), "node_size_median": float(np.median(size)), "node_size_range": [int(size.min()), int(size.max())],
        "node_jaccard_max": float(np.max(jaccard - np.eye(len(size)))),
        "conditions_with_support": int(len(targets)), "conditions_without_support": len(excluded),
        "support_size_median": float(np.median(support.sum(axis=1))),
        "perturbed_members_per_node_range": [int(perturbed_members[kept].min()), int(perturbed_members[kept].max())],
        "cells_per_condition_median": per_split.drop(index="control").median().to_dict(),
        "conditions_with_fewer_than_10_test_cells": int((per_split.drop(index="control").get("test", 0) < 10).sum()),
        "split_groups": split_groups,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: v for k, v in summary.items() if k != "genes_without_node"}, indent=2, default=str))


if __name__ == "__main__":
    main()
