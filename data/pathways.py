"""Pathway database utilities used by current data preparation and benchmarks."""

from __future__ import annotations

import numpy as np


def reactome_pathway_mask(
    var_names: np.ndarray, gmt_path: str, min_genes: int, max_genes: int, max_jaccard: float,
    exclude_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Pathway nodes: GMT gene sets with min_genes..max_genes measured genes.

    The database is hierarchical, so many gene sets are near copies of each
    other. Going from the smallest set up, a set is kept only if its Jaccard
    index with every set already kept is at most ``max_jaccard``.
    """
    position = {gene: index for index, gene in enumerate(var_names)}
    columns, names, seen = [], [], set()
    for name, identifier, genes in read_gmt(gmt_path):
        if exclude_ids and identifier in exclude_ids:
            continue
        members = tuple(sorted({position[gene] for gene in genes if gene in position}))
        if min_genes <= len(members) and (max_genes <= 0 or len(members) <= max_genes) and members not in seen:
            seen.add(members)
            column = np.zeros(len(var_names), dtype=np.float32)
            column[list(members)] = 1.0
            columns.append(column)
            names.append(name)
    if not columns:
        raise ValueError(f"No pathway in {gmt_path} has {min_genes}..{max_genes} measured genes")
    mask = np.stack(columns, axis=1)
    size = mask.sum(axis=0)
    shared = mask.T @ mask
    jaccard = shared / (size[:, None] + size[None, :] - shared)
    kept: list[int] = []
    for candidate in np.argsort(size, kind="stable"):
        if not kept or jaccard[candidate, kept].max() <= max_jaccard:
            kept.append(int(candidate))
    kept.sort()
    return mask[:, kept], [names[index] for index in kept]


def read_gmt(gmt_path: str) -> list[tuple[str, str, list[str]]]:
    """(name, identifier, genes) of every gene set in a GMT file."""
    with open(gmt_path) as handle:
        fields = (line.rstrip("\n").split("\t") for line in handle)
        return sorted((f[0], f[1], f[2:]) for f in fields if len(f) >= 4)


def pathway_membership(genes: np.ndarray, pathway_names: list[str], gmt_path: str) -> np.ndarray:
    """[genes, pathways] 1 where the database lists the gene in the pathway."""
    gene_sets = {name: set(members) for name, _, members in read_gmt(gmt_path)}
    return np.array([[gene in gene_sets[name] for name in pathway_names] for gene in genes], dtype=np.float32)
