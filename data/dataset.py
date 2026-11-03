"""Loading the generative dataset, split guards and condition-balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch
import json
from pathlib import Path


@dataclass
class GenerativeData:
    X: np.ndarray                 # [cells, genes] standardized log-normalized expression
    condition: np.ndarray         # [cells] condition index; 0 is control
    split: np.ndarray             # [cells] train / val / test (whole gem groups)
    cell_ids: np.ndarray          # [cells]
    genes: np.ndarray             # [genes]
    node_names: list[str]         # [pathways]
    node_mask: np.ndarray         # [genes, pathways] membership of measured genes
    conditions: list[str]         # [conditions]; conditions[0] == "control"
    support: np.ndarray           # [conditions, pathways] candidate direct support; row 0 is zero
    train_mean: np.ndarray        # [genes] to undo the standardization
    train_sd: np.ndarray          # [genes]
    gene_split_sha256: str | None = None

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def load_generative_data(path: str) -> GenerativeData:
    data = ad.read_h5ad(path)
    return GenerativeData(
        X=np.asarray(data.X, dtype=np.float32), condition=data.obs["condition_index"].to_numpy().astype(np.int64),
        split=data.obs["split"].astype(str).to_numpy(), cell_ids=data.obs_names.to_numpy().astype(str),
        genes=data.var_names.to_numpy().astype(str), node_names=[str(n) for n in data.uns["node_names"]],
        node_mask=np.asarray(data.varm["node_mask"], dtype=np.float32), conditions=[str(c) for c in data.uns["conditions"]],
        support=np.asarray(data.uns["support"], dtype=np.float32), train_mean=data.var["train_mean"].to_numpy(),
        train_sd=data.var["train_sd"].to_numpy(),
        gene_split_sha256=data.uns.get("gene_split_sha256"),
    )


def gene_split_masks(conditions, path):
    """Name-based, disjoint seen/tune/test split; never resample after filtering."""
    spec = json.loads(Path(path).read_text())
    groups = [set(spec[k]) for k in ('seen', 'heldout_tune', 'heldout_test')]
    if any('control' in g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Gene split must be disjoint and exclude control')
    unknown = set(conditions) - {'control'} - set.union(*groups)
    if unknown:
        raise ValueError(f'Conditions absent from gene split: {sorted(unknown)[:10]}')
    return tuple(np.isin(conditions, list(g)) for g in groups)


def assert_training_only(split: np.ndarray, indices: np.ndarray, what: str) -> None:
    """Every cell used to fit ``what`` must belong to the training split."""
    wrong = np.asarray(split)[np.asarray(indices)] != "train"
    if wrong.any():
        raise ValueError(f"{what}: {int(wrong.sum())} cells outside the training split")


class ConditionBalancedSampler:
    """Draw a condition uniformly among those present in ``rows``, then one of its cells, for every sample.

    With ``control_weight`` w, control (condition 0) is drawn with probability w and the perturbations share 1 - w
    equally; ``None`` treats control as one condition among all.
    """

    def __init__(self, condition: torch.Tensor, rows: torch.Tensor, generator: torch.Generator, control_weight: float | None = None):
        order = rows[torch.argsort(condition[rows], stable=True)]
        present, counts = torch.unique_consecutive(condition[order], return_counts=True)
        self.order, self.present, self.counts, self.generator = order, present, counts, generator
        self.starts = torch.cumsum(counts, 0) - counts
        self.control_weight = control_weight

    def __call__(self, n: int) -> torch.Tensor:
        device = self.order.device
        if self.control_weight is None:
            which = torch.randint(len(self.present), (n,), device=device, generator=self.generator)
        else:                                                   # present[0] is control (condition 0 sorts first)
            perturbation = 1 + torch.randint(len(self.present) - 1, (n,), device=device, generator=self.generator)
            is_control = torch.rand(n, device=device, generator=self.generator) < self.control_weight
            which = torch.where(is_control, torch.zeros_like(perturbation), perturbation)
        offset = (torch.rand(n, device=device, generator=self.generator) * self.counts[which]).long()
        return self.order[self.starts[which] + offset]


def balanced_mean(values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """Mean over conditions of the mean over each condition's cells."""
    present, inverse = torch.unique(condition, return_inverse=True)
    sums = torch.zeros(len(present), dtype=values.dtype, device=values.device).index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=len(present)).to(values.dtype)
    return (sums / counts).mean()


def held_out_conditions(n_conditions: int, fraction: float, seed: int = 0) -> np.ndarray:
    """Perturbations whose cells are never used for fitting (unseen target genes), as a mask over conditions.

    A fixed random share ``fraction`` of the perturbations (control is never held out), drawn with ``seed``;
    the same fraction and seed always give the same genes.
    """
    held = np.zeros(n_conditions, dtype=bool)
    if fraction > 0:
        rng = np.random.default_rng(seed)
        perturbations = np.arange(1, n_conditions)
        held[rng.choice(perturbations, int(round(fraction * len(perturbations))), replace=False)] = True
    return held


def split_held_out(held: np.ndarray, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Halve the held-out perturbations into genes for tuning and genes for the final test (masks over conditions)."""
    index = np.flatnonzero(held)
    order = np.random.default_rng(seed).permutation(index)
    tune = np.zeros_like(held)
    tune[order[: len(order) // 2]] = True
    return tune, held & ~tune


def weighted_condition_mean(values: torch.Tensor, condition: torch.Tensor, control_weight: float | None) -> torch.Tensor:
    """Condition-balanced mean; with ``control_weight`` w, w * control mean + (1 - w) * mean over perturbations."""
    if control_weight is None:
        return balanced_mean(values, condition)
    control = condition == 0
    return control_weight * values[control].mean() + (1 - control_weight) * balanced_mean(values[~control], condition[~control])
