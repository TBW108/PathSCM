"""Train scGen (official code) on the training cells of our split and export what baselines/adapters/scgen.py needs
to generate the SEEN perturbations of the network study.

Source: theislab/scgen, the released package scgen 2.1.0 (PyPI; with scvi-tools 0.18) in the conda env
perturbsb-scgen. The repository head (2.1.1, cloned in baselines/sota_src/scgen @ d79e1f0) needs scvi-tools >= 0.20
and differs from 2.1.0 only in that compatibility, not in the model. Model and training use the scGen defaults
(SCGEN(adata): n_latent 100, n_hidden 800, 2 layers, dropout 0.2; train as in the scGen tutorial: up to 100 epochs,
batch size 32, early stopping with patience 25 on its own 10% validation cells).

scGen is fitted on every training cell (all perturbations and controls; the condition is the scGen "batch", one cell
type). A perturbation t is predicted with SCGEN.predict(ctrl_key="ctrl", stim_key=t): delta_t = mean latent of t's
training cells minus mean latent of as many random training control cells (its balanced arithmetic), then
decoder(latent of a control cell + delta_t). Expression is given on the log1p scale (X * train_sd + train_mean).
Exported (<output>.npz):
    conditions [P]; delta [P, n_latent] (row order of conditions); control_latent [training controls of --reference,
    n_latent] in their row order; decoder.* weights; check_* a few decoded cells of predict() to verify the
    re-implemented decoder.

    <perturbsb-scgen python> baselines/scgen.py --data <k64 h5ad> --reference <h5ad> --output <npz> [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import anndata as ad
import numpy as np
import scgen
import scvi
import torch


def log1p_cells(data: ad.AnnData, rows: np.ndarray) -> np.ndarray:
    sd = data.var["train_sd"].to_numpy(np.float32)
    mean = data.var["train_mean"].to_numpy(np.float32)
    return np.asarray(data.X[rows], dtype=np.float32) * sd[None] + mean[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True, help="the h5ad whose training control cells the benchmark starts from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed

    data = ad.read_h5ad(args.data)
    conditions = [str(c) for c in data.uns["conditions"]]
    train_rows = np.flatnonzero(data.obs["split"].astype(str).to_numpy() == "train")
    index = data.obs["condition_index"].to_numpy()[train_rows]
    adata = ad.AnnData(log1p_cells(data, train_rows))
    adata.var_names = data.var_names.astype(str)
    adata.obs["condition"] = np.where(index == 0, "ctrl", np.array(conditions, dtype=object)[index])
    adata.obs["cell_type"] = "cell"
    scgen.SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata)
    start = time.time()
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, early_stopping=True,
                early_stopping_patience=args.patience, use_gpu=True)
    minutes = (time.time() - start) / 60
    epochs = len(model.history["elbo_train"])
    print(f"trained {epochs} epochs in {minutes:.1f} min", flush=True)
    model.module.eval()

    one = adata[adata.obs["condition"] == "ctrl"][:5].copy()
    delta, check = [], {}
    for c in conditions[1:]:
        predicted, d = model.predict(ctrl_key="ctrl", stim_key=c, adata_to_predict=one)
        delta.append(np.asarray(d, dtype=np.float32))
        if len(check) < 3:
            check[c] = np.asarray(predicted.X, dtype=np.float32)

    reference = ad.read_h5ad(args.reference)
    obs = reference.obs
    control_rows = np.flatnonzero((obs["split"].astype(str).to_numpy() == "train") & (obs["condition_index"].to_numpy() == 0))
    controls = ad.AnnData(log1p_cells(reference, control_rows))
    controls.var_names = reference.var_names.astype(str)
    controls.obs["condition"], controls.obs["cell_type"] = "ctrl", "cell"
    assert list(controls.var_names) == list(adata.var_names)
    control_latent = model.get_latent_representation(controls).astype(np.float32)

    decoder = {f"decoder.{k}": v.detach().cpu().numpy() for k, v in model.module.decoder.state_dict().items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, conditions=np.array(conditions[1:]), delta=np.stack(delta), control_latent=control_latent,
             check_conditions=np.array(list(check)), check_cells=np.stack(list(check.values())),
             check_latent=model.get_latent_representation(one).astype(np.float32), **decoder)
    report = {"scgen": scgen.__version__, "seed": args.seed, "batch_size": args.batch_size, "epochs": epochs,
              "minutes": round(minutes, 1), "training_cells": int(adata.n_obs), "control_cells": int(len(control_rows)),
              "elbo_validation_last": float(model.history["elbo_validation"].iloc[-1, 0])}
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
