"""Run the official GEARS implementation on the generative v3 data and gene split (plan of 2026-10-20, section 9).

GEARS source: baselines/GEARS (snap-stanford/GEARS, commit in baselines/README.md); nothing in the package is modified.
Same cells, genes and target split as the response model (pathscm/train.py, --holdout-genes 0.2):
- GEARS train: training gem groups, control + the fitted perturbations (the 80% of target genes the model fits);
- GEARS val (its epoch selection): the held-out genes for tuning, cells of the training and validation gem groups;
- GEARS test: the held-out genes for the final test, cells of the test gem groups (only GEARS' own bookkeeping).
GEARS is trained on log-normalized expression (X * train_sd + train_mean of the v3 file), the scale it expects.
Predictions: GEARS.predict (mean over 300 training control cells) for every seen perturbation and held-out gene,
saved as log-normalized means; generative/score_mean_effects.py scores them against the test gem groups.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np
import scanpy as sc
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines/GEARS"))
sys.path.insert(0, str(ROOT))

from gears import GEARS, PertData  # noqa: E402
from data.dataset import held_out_conditions, load_generative_data, split_held_out, gene_split_masks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "datasets/processed/k562_generative_v3.h5ad"))
    parser.add_argument("--workdir", default=str(ROOT / "baselines/gears_data"))
    parser.add_argument("--name", default="k562_generative_v3")
    parser.add_argument("--output", default=str(ROOT / "checkpoints/baselines/gears_v3_predictions.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--gene-split', default=None)
    parser.add_argument('--cell-type', default='K562')
    parser.add_argument('--smoke-test', action='store_true', help='One official-loss forward/backward; do not save predictions')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_generative_data(args.data)
    held = held_out_conditions(len(data.conditions), 0.2, 0)
    tune, final = split_held_out(held)
    if args.gene_split:
        assert data.gene_split_sha256 == hashlib.sha256(Path(args.gene_split).read_bytes()).hexdigest()
        _, tune, final = gene_split_masks(data.conditions, args.gene_split)
        held = tune | final
    split, condition = data.split, data.condition
    keep = (((split == "train") & ~held[condition])                        # control + fitted perturbations
            | (np.isin(split, ["train", "val"]) & tune[condition])
            | ((split == "test") & final[condition]))
    log1p = data.X[keep] * data.train_sd[None, :].astype(np.float32) + data.train_mean[None, :].astype(np.float32)
    names = np.array(data.conditions)
    adata = sc.AnnData(sparse.csr_matrix(log1p.astype(np.float32)))
    adata.obs_names = data.cell_ids[keep]
    adata.var_names = data.genes
    adata.var["gene_name"] = data.genes
    adata.obs["condition"] = np.where(condition[keep] == 0, "ctrl", np.char.add(names[condition[keep]], "+ctrl"))
    adata.obs["cell_type"] = args.cell_type

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(workdir))
    dataset_dir = workdir / args.name
    if not (dataset_dir / "data_pyg/cell_graphs.pkl").exists():
        pert_data.new_data_process(dataset_name=args.name, adata=adata)
    pert_data.load(data_path=str(dataset_dir))               # GEARS drops perturbations absent from its GO graph

    kept = set(pert_data.adata.obs["condition"].astype(str).unique())
    groups = {"train": ~held, "val": tune, "test": final}
    split_dict, dropped = {}, {}
    for name, mask in groups.items():
        wanted = [f"{names[t]}+ctrl" for t in range(1, len(names)) if mask[t]]
        split_dict[name] = [c for c in wanted if c in kept] + (["ctrl"] if name == "train" else [])
        dropped[name] = [c.split("+")[0] for c in wanted if c not in kept]
    split_path = dataset_dir / "generative_v3_split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(split_dict, handle)
    print({k: len(v) for k, v in split_dict.items()}, "dropped (not in GEARS GO graph):", {k: len(v) for k, v in dropped.items()})

    pert_data.prepare_split(split="custom", split_dict_path=str(split_path))
    pert_data.get_dataloader(batch_size=args.batch_size)
    if args.gene_split:
        # No final-test evaluation while fitting; the separate benchmark scores it after export.
        pert_data.dataloader.pop('test_loader', None)
    model = GEARS(pert_data, device=args.device)
    model.model_initialize(hidden_size=args.hidden_size)
    if args.smoke_test:
        from gears.utils import loss_fct
        model.model.train()
        batch = next(iter(pert_data.dataloader['train_loader'])).to(args.device)
        prediction = model.model(batch)
        loss = loss_fct(prediction, batch.y, batch.pert, ctrl=model.ctrl_expression,
                        dict_filter=model.dict_filter, direction_lambda=model.config['direction_lambda'])
        loss.backward()
        assert torch.isfinite(loss)
        print('GEARS_SMOKE_OK', float(loss.detach()), 'pred_shape', tuple(prediction.shape), flush=True)
        return
    try:
        model.train(epochs=args.epochs, lr=args.lr)
    except ZeroDivisionError as error:
        # GEARS' optional per-perturbation test analysis (inference.deeper_analysis) divides by the number of genes with
        # non-zero spread across cells, which is zero for one test perturbation here; the best model (chosen on the
        # validation genes) is already set when it runs, and the analysis is not used here.
        print("GEARS test analysis skipped:", repr(error))
        assert getattr(model, "best_model", None) is not None
    model.save_model(str(dataset_dir / f"model_{Path(args.output).stem}"))

    genes, which = [], []
    for name, mask in (("seen", ~held), ("heldout_tune", tune), ("heldout_test", final)):
        for t in range(1, len(names)):
            if mask[t] and names[t] in model.pert_list:
                genes.append(names[t]); which.append(name)
    np.random.seed(args.seed)
    prediction = model.predict([[g] for g in genes])
    control = np.asarray(pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].X.mean(axis=0)).ravel()
    np.savez(args.output, genes=np.array(genes), set=np.array(which),
             pred_mean_log1p=np.stack([np.asarray(prediction[g]).ravel() for g in genes]),
             control_mean_log1p=control, var_names=np.asarray(pert_data.adata.var["gene_name"]).astype(str))
    Path(args.output).with_suffix(".json").write_text(json.dumps(
        {"args": vars(args), "n_conditions": {k: len(v) for k, v in split_dict.items()}, "dropped_not_in_go_graph": dropped},
        indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
