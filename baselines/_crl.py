"""Train discrepancy-VAE or SENA-discrepancy-VAE (official code, unmodified) on our data, then export what we evaluate.

Sources (cloned into baselines/sota_src, used as is):
    discrepancy-VAE        uhlerlab/discrepancy_vae  @ 4451fdbc  (src/train.py: train, src/model.py: CMVAE)
    SENA-discrepancy-VAE   ML4BM-Lab/SENA            @ d5511aa3  (src/sena_discrepancy_vae/train.py: train, model.py: CMVAE)
Each method is trained with its own training loop and the default options of its repository (discrepancy-VAE:
src/run.py; SENA: Options in src/sena_discrepancy_vae/main.py), except the latent dimension (64, the number of our
modules). Only the data provision is ours: the training cells of our split, each perturbed cell paired with a random
training control cell (as their SCDataset does), batches of one perturbation (their SCDATA_sampler), the perturbation
encoded one-hot. SENA's gene sets are our 64 Reactome pathways. Expression is given on the log1p scale (the scale of
their Norman data): X * train_sd + train_mean.

Their causal layer is u = z (I - G_u)^-1 with G_u = triu(G, 1), i.e. u_j = z_j + sum_i u_i G_ij (edge i -> j), and a
perturbation adds bc_t * s_t to z (bc_t: softmax target over latents, s_t: strength). In our convention
(z = A z + a_t + eps, A[k, l] for l -> k) this is A = G_u^T and a_t = bc_t * s_t. Exported (export.npz):
    A [K, K]; inputs [conditions, K] (row 0 = control = 0) at the final softmax temperature;
    states [training cells, K]: u of the encoder mean; state_rows: their row indices in the data;
    control_sd [K]: standard deviation of u over the training control cells.

    python baselines/_crl.py --method dvae --data <h5ad> --out checkpoints/crl/<name> [--cell-half 0|1] [--seed 0]
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import importlib
import json
import logging
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset import load_generative_data  # noqa: E402

SOURCES = {"dvae": ROOT / "baselines/sota_src/discrepancy_vae/src",
           "sena": ROOT / "baselines/sota_src/SENA/src/sena_discrepancy_vae"}

# repository defaults (discrepancy-VAE: src/run.py; SENA: Options in main.py); latdim set to the number of modules
DEFAULTS = {
    "dvae": dict(batch_size=32, mode="train", lr=1e-3, epochs=100, grad_clip=False, mxAlpha=10, mxBeta=2, mxTemp=5,
                 lmbda=1e-3, MMD_sigma=1000, kernel_num=10, matched_IO=False),
    "sena": dict(name="ours", model="sena", dataset_name="ours", batch_size=32, sena_lambda=0, lr=1e-3, epochs=100,
                 grad_clip=False, mxAlpha=1.0, mxBeta=1.0, mxTemp=100.0, lmbda=0.1, MMD_sigma=200.0, kernel_num=10,
                 matched_IO=False, log=False, mlflow_port=5678),
}


def import_official(method: str):
    """train() and the model module of a repository, with its logging services (wandb, mlflow) stubbed out:
    logging is off in every run, but the modules are imported at the top of their train.py."""
    for name in ("wandb", "mlflow"):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)
    for name in ("train", "model", "utils", "dataset"):          # both repositories use these top-level names
        sys.modules.pop(name, None)
    sys.path.insert(0, str(SOURCES[method]))
    train = importlib.import_module("train")
    utils = importlib.import_module("utils")
    sys.path.pop(0)
    return train.train, utils.SCDATA_sampler


class PairedCells(torch.utils.data.Dataset):
    """Perturbed training cells, each with a random training control cell; batched fetching (__getitems__)."""

    def __init__(self, X: torch.Tensor, perturbed: np.ndarray, controls: np.ndarray, condition: np.ndarray,
                 used: np.ndarray, rng: np.random.Generator, device: str):
        self.X = X
        self.y_rows = torch.as_tensor(perturbed, device=device)
        self.x_rows = torch.as_tensor(rng.choice(controls, len(perturbed), replace=True), device=device)
        position = np.full(condition.max() + 1, -1)
        position[used] = np.arange(len(used))
        self.code = torch.as_tensor(position[condition[perturbed]], device=device)  # one-hot over the used perturbations
        self.c_dim = len(used)
        self.ptb_names = condition[perturbed]

    def __len__(self) -> int:
        return len(self.y_rows)

    def __getitems__(self, items):
        i = torch.as_tensor(items, device=self.y_rows.device)
        c = torch.nn.functional.one_hot(self.code[i], self.c_dim).double()
        return self.X[self.x_rows[i]].double(), self.X[self.y_rows[i]].double(), c

    def __getitem__(self, item):
        return self.__getitems__([item])


def training_rows(data, cell_half: int | None) -> dict[int, np.ndarray]:
    """Training cells of every condition; with cell_half, the same disjoint halves as train_response.py --cell-half."""
    rows = {t: np.flatnonzero((data.split == "train") & (data.condition == t)) for t in range(len(data.conditions))}
    if cell_half is not None:
        halves = np.random.default_rng(12345)
        rows = {t: np.sort(halves.permutation(r)[cell_half * (len(r) // 2):(cell_half + 1) * (len(r) // 2)])
                for t, r in rows.items()}
    return rows


@torch.no_grad()
def export(model, X: torch.Tensor, rows: dict[int, np.ndarray], n_conditions: int, used: np.ndarray, temp: float,
           out: Path) -> None:
    model.eval()
    K = model.z_dim
    G_u = torch.triu(model.G, diagonal=1)
    inverse = torch.inverse(torch.eye(K, dtype=G_u.dtype, device=G_u.device) - G_u)
    state_rows = np.concatenate([rows[t] for t in range(n_conditions)])
    states = []
    for i in range(0, len(state_rows), 8192):
        mu, _ = model.encode(X[torch.as_tensor(state_rows[i:i + 8192], device=X.device)].double())
        states.append((mu @ inverse).cpu().numpy())
    states = np.concatenate(states)
    controls = np.isin(state_rows, rows[0])
    bc, s = model.c_encode(torch.eye(len(used), dtype=torch.float64, device=X.device), temp)
    inputs, targets, strength = np.zeros((n_conditions, K)), np.zeros((n_conditions, K)), np.zeros(n_conditions)
    inputs[used], targets[used], strength[used] = (bc * s[:, None]).cpu().numpy(), bc.cpu().numpy(), s.cpu().numpy()
    np.savez(out / "export.npz", A=G_u.T.cpu().numpy(), inputs=inputs, targets=targets, strength=strength, used=used,
             states=states.astype(np.float32), state_rows=state_rows, control_sd=states[controls].std(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(SOURCES), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latdim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None, help="default: the repository default (100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-half", type=int, choices=[0, 1], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--one-per-module", action="store_true",
                        help="train on one perturbation per module only (the single-module target with most cells)")
    parser.add_argument("--set", nargs="*", default=[], metavar="OPTION=VALUE",
                        help="override a repository option (hyperparameter search), e.g. lr=1e-4 mxTemp=1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = load_generative_data(args.data)
    C = len(data.conditions)
    X = torch.as_tensor(data.X * data.train_sd[None, :].astype(np.float32) + data.train_mean[None, :].astype(np.float32),
                        device=args.device)
    rows = training_rows(data, args.cell_half)
    if args.one_per_module:     # their regime: about one perturbation per latent; per module, the single-module target
        single = data.support.sum(1) == 1                      # with the most training cells
        used = np.array(sorted({max(np.flatnonzero(single & (data.support[:, k] > 0)), key=lambda t: len(rows[t]))
                                for k in range(data.support.shape[1]) if (single & (data.support[:, k] > 0)).any()}))
    else:
        used = np.arange(1, C)
    perturbed = np.concatenate([rows[t] for t in used])
    train, sampler = import_official(args.method)
    opts = Namespace(**DEFAULTS[args.method])
    if args.epochs is not None:
        opts.epochs = args.epochs
    for item in args.set:
        key, value = item.split("=")
        setattr(opts, key, type(getattr(opts, key))(float(value)))
    dataset = PairedCells(X, perturbed, rows[0], data.condition, used, np.random.default_rng(args.seed), args.device)
    opts.dim, opts.cdim, opts.latdim, opts.seed = X.shape[1], dataset.c_dim, args.latdim, args.seed
    loader = DataLoader(dataset, batch_sampler=sampler(dataset, opts.batch_size, dataset.ptb_names), num_workers=0,
                        collate_fn=lambda batch: batch)
    (out / "config.json").write_text(json.dumps({**vars(opts), "data": args.data, "cell_half": args.cell_half,
                                                 "perturbations": int(len(used)),
                                                 "training_cells": int(len(perturbed) + len(rows[0]))}, indent=1))
    if not args.export_only:
        start = time.time()
        if args.method == "dvae":
            train(loader, opts, args.device, str(out), log=False)
        else:
            handler = Namespace(gos=list(data.node_names),
                                rel_dict={g: list(np.flatnonzero(data.node_mask[g] > 0)) for g in range(X.shape[1])})
            train(dataloader=loader, opts=opts, device=args.device, savedir=str(out), logger=logging, data_handler=handler)
        (out / "time.txt").write_text(f"{time.time() - start:.0f} s\n")
    model = torch.load(out / "best_model.pt", map_location=args.device, weights_only=False)
    export(model, X, rows, C, used, float(opts.mxTemp), out)
    print("exported", out / "export.npz", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
