"""Joint response-model training and supported supplementary variants.

For the current model, use run.py train with configs/*.json.
Those checkpoint-derived settings select the shared encoder, masked pathway
readout, SCM state likelihood with sparse A, and energy-distance generation
loss. Training starts jointly (no warm-start or zero-graph stages). Low-level
CLI defaults and optional branches also support supplementary configurations;
they should not be mistaken for the current experiment configuration.

Only eligible training cells enter gradient updates. Named gene/condition
splits govern held-out effect transfer and validation when supplied. The
prediction branch starts from control cells, without target-cell embeddings.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import assert_training_only, held_out_conditions, load_generative_data, gene_split_masks  # noqa: E402
from pathscm.scm import LinearPathwaySCM  # noqa: E402
from pathscm.model import (  # noqa: E402
    ExogenousEncoder, FullGeneEncoder, GeneStates, JointLoadings, SharedEncoder, LinearDecoder, MaskedPathwayDecoder, ObservationNoise, PathwayLoadings,
    PathwayScores, PathwayStates, ResidualDecoder,
    gaussian_nll_per_gene, scm_state_nll,
)
from evaluation.metrics import mmd  # noqa: E402
from pathscm.model import energy_distance, sinkhorn_divergence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "datasets/processed/k562_generative_v3.h5ad"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-genes", type=float, default=0.2, help="share of perturbations never fitted (0: use all)")
    parser.add_argument("--holdout-folds", type=int, default=0,
                        help="with --holdout-fold k: hold out fold k of the perturbations split into this many folds instead")
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument('--gene-split', default=None, help='Strict named unseen split; validate using Jaccard a/c transfer')
    parser.add_argument("--encoder-hidden", type=int, default=64)
    parser.add_argument("--embedding", type=int, default=16)
    parser.add_argument("--encoder-depth", type=int, default=1)
    parser.add_argument("--states", choices=["learned", "pathway_ae", "joint", "joint_local", "shared", "score", "pc1"],
                        default="learned",
                        help="pathway states: learned per-pathway encoder (trained through the decoder); pathway_ae: per-pathway "
                             "MLP trained only to reconstruct its own members through linear loadings; fixed equal-weight score "
                             "of the members; or fixed score along the first principal axis of the members' shifts")
    parser.add_argument("--lambda-path", type=float, default=1.0, help="weight of the pathway reconstruction (pathway_ae, joint)")
    parser.add_argument("--scm-conditions", type=int, default=0,
                        help="extra perturbations per step for the structural fit only (64 cells each; their states enter the mean-shift or likelihood "
                             "term, besides the step's own perturbations; 0: only those)")
    parser.add_argument("--noise-rank", type=int, default=0,
                        help="likelihood fit: exogenous noise covariance diag(sigma^2) + L L^T with L [K, r] (shared cell-state factors)")
    parser.add_argument("--generation-trains-offset", action="store_true",
                        help="with --scm-only-from-states: the expression losses may still move delta_t (not A, w, gamma)")
    parser.add_argument("--exogenous", action="store_true",
                        help="(with --states shared) encode every cell to its exogenous noise eps ~ q(eps | x), "
                             "z = (I - A)^{-1} (eps + a_t); A, a_t, encoder and decoder learn from the ELBO of all cells, "
                             "the pathway read-out and the generation loss; the SCM itself only has lambda_A ||A||_1")
    parser.add_argument("--decoder-u-twice", action="store_true",
                        help="the decoder reads [z, u0, u0 + c_t]: the start cell's own uncaptured embedding and its "
                             "perturbation-shifted copy")
    parser.add_argument("--loadings-unmasked", action="store_true",
                        help="ablation: the linear read-out of the genes from z uses every gene for every dimension (no "
                             "pathway mask), so no dimension is tied to a pathway")
    parser.add_argument("--exo-kl-per", choices=["gene", "module"], default="gene",
                        help="--exogenous: KL divided by the number of genes (the ELBO per gene) or of modules (the scale "
                             "of the likelihood term L_SCM)")
    parser.add_argument("--scm-only-from-states", action="store_true",
                        help="A and the inputs are fitted only to the encoded pathway states, not to the expression losses")
    parser.add_argument("--scm-loss", choices=["means", "likelihood"], default="means",
                        help="how the SCM is fitted to the encoded states: mean shifts per condition (L_latent), or the "
                             "Gaussian likelihood of every cell's state, z = (I - A)^{-1} (b + a_t + eps), eps ~ N(0, diag sigma^2)")
    parser.add_argument("--inputs", choices=["embedding", "sensitivity", "sensitivity_offset"], default="embedding",
                        help="embedding: a_t = S_t * a_max tanh(MLP(e_t)) + s_t g; sensitivity: a_t = S_t * w + g (w, g shared); "
                             "sensitivity_offset: a_t = S_t * (w + delta_t) + g, delta_t of fitted perturbations (L2)")
    parser.add_argument("--lambda-offset", type=float, default=1e-2, help="L2 on delta_t (sensitivity_offset)")
    parser.add_argument("--output", choices=["absolute", "difference"], default="absolute",
                        help="difference: generated cell = control cell + D(z0 + shift) - D(z0) (no added noise)")
    parser.add_argument("--lambda-rec", type=float, default=1.0, help="weight of the reconstruction term after warm start")
    parser.add_argument("--dim-u", type=int, default=32, help="0: no u (pathway states only)")
    parser.add_argument("--uncaptured-dim", type=int, default=0,
                        help="joint read-out only: an uncaptured embedding u of this size, read from what the pathway read-out "
                             "leaves unexplained, kept uncorrelated with z, shifted by a per-perturbation c_t (replaces --dim-u)")
    parser.add_argument("--lambda-orth", type=float, default=1.0, help="weight of the u-z decorrelation penalty")
    parser.add_argument("--no-uncaptured-shift", action="store_true",
                        help="no per-perturbation shift c_t of u (u then only carries the control cell's own state)")
    parser.add_argument("--no-common-response", action="store_true",
                        help="no common response gamma: a_t = S_t * (w + delta_t)")
    parser.add_argument("--no-base", action="store_true",
                        help="no constant b in the structural equation: the likelihood uses states centred on the control "
                             "cells of the step, so control cells have z = 0 on average")
    parser.add_argument("--decoder", choices=["residual", "masked", "linear"], default="residual",
                        help="masked: a gene is decoded only from the pathways it belongs to (plus u)")
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=3)
    parser.add_argument("--conditions-per-step", type=int, default=4)
    parser.add_argument("--cells", type=int, default=128, help="control and real cells per condition and step")
    parser.add_argument("--warm-steps", type=int, default=2000)
    parser.add_argument("--zero-graph-steps", type=int, default=3000)
    parser.add_argument("--graph-steps", type=int, default=8000)
    parser.add_argument("--no-graph", action="store_true", help="A = 0 throughout (ablation)")
    parser.add_argument("--no-scm", action="store_true",
                        help="ablation: no structural model at all; the generator gets no module shift (Delta_t = 0), "
                             "a perturbation acts only through c_t; L_SCM and |A|_1 are not used")
    parser.add_argument("--cell-fraction", type=float, default=1.0,
                        help="keep only this fraction of the training cells of every perturbation (data-efficiency ablation)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gen-loss", choices=["legacy", "mmd", "ot", "energy"], default="legacy",
                        help="legacy: reconstruction + lambda_mean * mean + lambda_dist * MMD; mmd / ot / energy: one "
                             "distribution loss between generated and real cells (MMD on training-cell PCs / Sinkhorn "
                             "divergence on all genes / unbiased energy distance on all genes), for the perturbations of "
                             "the step and for the control cells, and nothing else")
    parser.add_argument("--ot-eps", type=float, default=0.5, help="entropic blur of the Sinkhorn divergence (cost per gene)")
    parser.add_argument("--keep-rec", action="store_true", help="with --gen-loss mmd/ot/energy: keep the cell-wise reconstruction")
    parser.add_argument("--shared-start", action="store_true",
                        help="one set of --cells control cells per step is the generation starting point of every condition")
    parser.add_argument("--no-gen-noise", action="store_true", help="generated cells are the decoder output (no tau noise)")
    parser.add_argument("--ot-iterations", type=int, default=30)
    parser.add_argument("--lambda-mean", type=float, default=10.0)
    parser.add_argument("--lambda-dist", type=float, default=1.0)
    parser.add_argument("--lambda-A", type=float, default=1e-3)
    parser.add_argument("--lambda-local", type=float, default=1e-3, help="L1 on inputs outside the target's pathways")
    parser.add_argument("--lambda-latent", type=float, default=0.0,
                        help="weight of the latent consistency term (predicted vs encoded mean state shift)")
    parser.add_argument("--support-only", action="store_true", help="hard support: inputs only on the target's pathways")
    parser.add_argument("--embedding-source", choices=["free", "membership", "hybrid"], default="hybrid")
    parser.add_argument("--radius", type=float, default=0.9)
    parser.add_argument("--amplitude", type=float, default=3.0, help="a_max: inputs on the support are a_max tanh(.)")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--cell-half", type=int, choices=[0, 1], default=None,
                        help="fit on one of two disjoint halves of the training cells (reproducibility of A across data)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.uncaptured_dim > 0:
        assert args.states in ("joint", "joint_local", "shared"), "the uncaptured embedding needs the joint pathway read-out"
        args.dim_u = args.uncaptured_dim
    if args.exogenous:
        assert args.states == "shared" and args.shared_start and args.gen_loss != "legacy" and not args.scm_only_from_states
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data = load_generative_data(args.data)
    X = torch.as_tensor(data.X, device=device)
    C = len(data.conditions)
    if args.gene_split:
        split_digest = hashlib.sha256(Path(args.gene_split).read_bytes()).hexdigest()
        if data.gene_split_sha256 != split_digest:
            raise ValueError('Unseen training requires data preprocessed with the identical gene split')
        seen_mask, tune_mask, test_mask = gene_split_masks(data.conditions, args.gene_split)
        held = tune_mask | test_mask
    elif args.holdout_folds > 0:                 # cross-validation over perturbations: fold k is held out
        order = np.random.default_rng(0).permutation(np.arange(1, C))
        held = np.zeros(C, dtype=bool)
        held[np.array_split(order, args.holdout_folds)[args.holdout_fold]] = True
    else:
        held = held_out_conditions(C, args.holdout_genes, 0)
    cond, split = data.condition, data.split

    def rows(which: str, t: int) -> np.ndarray:
        return np.flatnonzero((split == which) & (cond == t))
    train_by_condition = {t: rows("train", t) for t in range(C) if not held[t]}
    if args.cell_half is not None:            # one of two disjoint halves of the training cells of every condition
        halves = np.random.default_rng(12345)
        train_by_condition = {t: np.sort(halves.permutation(r)[args.cell_half * (len(r) // 2):(args.cell_half + 1) * (len(r) // 2)])
                              for t, r in train_by_condition.items()}
    if args.cell_fraction < 1:                # fewer training cells per perturbation (controls are kept)
        thin = np.random.default_rng(54321)
        train_by_condition = {t: (r if t == 0 else np.sort(thin.permutation(r)[:max(2, int(round(args.cell_fraction * len(r))))]))
                              for t, r in train_by_condition.items()}
    fitted = np.array([t for t in range(1, C) if not held[t] and len(train_by_condition[t]) > 0])
    controls = train_by_condition[0]
    assert_training_only(split, np.concatenate([train_by_condition[t] for t in [0, *fitted]]), "response model")
    val_conditions = np.array([t for t in fitted if len(rows("val", t)) >= 10])
    def validation_rows(t):
        if args.gene_split and t > 0:
            return np.flatnonzero(np.isin(split, ['train','val']) & (cond == t))
        return rows('val', t)
    if args.gene_split:
        val_conditions = np.array([t for t in np.flatnonzero(tune_mask) if len(validation_rows(t)) >= 10])
        if len(val_conditions) < 2:
            raise ValueError('Need at least two unseen tuning conditions')
    val_conditions = rng.choice(val_conditions, min(300, len(val_conditions)), replace=False)

    # frozen PCA of training cells for the distribution loss
    train_rows = np.sort(np.concatenate([train_by_condition[t] for t in [0, *fitted]]))
    pool = X[torch.as_tensor(rng.choice(train_rows, min(50000, len(train_rows)), replace=False), device=device)]
    center = pool.mean(dim=0)
    basis = torch.linalg.svd(pool - center, full_matrices=False)[2][:50].T
    to_pcs = lambda x: (x - center) @ basis
    del pool

    def distribution_loss(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Per-condition distance between generated [c, n, G] and real [c, m, G] cells (--gen-loss mmd / ot)."""
        if args.gen_loss == "ot":
            return sinkhorn_divergence(generated, real, args.ot_eps, args.ot_iterations)
        if args.gen_loss == "energy":
            return energy_distance(generated, real)
        return torch.stack([mmd(to_pcs(generated[j]), to_pcs(real[j])) for j in range(len(generated))])

    node_mask = torch.as_tensor(data.node_mask, device=device)
    K, G = node_mask.shape[1], X.shape[1]
    if args.states in ("score", "pc1"):
        shifts = None
        if args.states == "pc1":                  # training cells of the fitted perturbations only
            control_mean = X[torch.as_tensor(controls, device=device)].mean(dim=0)
            shifts = torch.stack([X[torch.as_tensor(train_by_condition[t], device=device)].mean(dim=0) - control_mean
                                  for t in fitted])
        states = PathwayScores(node_mask).to(device).fit_(X[torch.as_tensor(controls, device=device)], shifts)
    elif args.states == "joint":              # all-gene MLP; the masked joint read-out fixes which pathway each dimension is
        states = GeneStates(G, K).to(device)
    elif args.states == "shared":             # one all-gene MLP that outputs both z and the uncaptured embedding u
        assert args.uncaptured_dim > 0, "--states shared needs --uncaptured-dim"
        states = (ExogenousEncoder if args.exogenous else SharedEncoder)(G, K, args.uncaptured_dim).to(device)
    else:
        states = PathwayStates(node_mask, args.encoder_hidden, args.embedding, args.encoder_depth).to(device)
    loadings = (PathwayLoadings(node_mask).to(device) if args.states == "pathway_ae"
                else JointLoadings(torch.ones_like(node_mask) if args.loadings_unmasked else node_mask).to(device)
                if args.states in ("joint", "joint_local", "shared") else None)
    u_dim = max(args.dim_u, 1)
    u_encoder = FullGeneEncoder(G, u_dim).to(device)
    decoder = (MaskedPathwayDecoder(node_mask, u_dim) if args.decoder == "masked"
               else LinearDecoder(K, G) if args.decoder == "linear"
               else ResidualDecoder(K, u_dim, G, args.decoder_width, args.decoder_blocks,
                                    u_twice=args.decoder_u_twice)).to(device)
    difference = args.output == "difference"

    def respond(x0, z0, u, dz, u0=None):
        """Generated means from control cells x0 (states z0, u0) under the state shift dz and uncaptured state u
        (= u0 + c_t); with --decoder-u-twice the decoder also reads u0 itself."""
        out = decoder(z0 + dz, u, u0) if args.decoder_u_twice else decoder(z0 + dz, u)
        return x0 + out - decoder(z0, u0 if u0 is not None else u) if difference else out
    noise = ObservationNoise(G).to(device)
    scm = LinearPathwaySCM(torch.as_tensor(data.support), args.radius, args.amplitude, learn_graph=not args.no_graph,
                           input_mode=args.inputs, embedding_source=args.embedding_source,
                           generic_input=not args.no_common_response,
                           generic_scale=args.inputs == "embedding", dense_inputs=not args.support_only, off_support_weight=1.0,
                           noise_rank=args.noise_rank,
                           sources=torch.as_tensor(~held), penalized=torch.as_tensor(~held)).to(device)
    support = torch.as_tensor(data.support, device=device, dtype=torch.float64)
    u_scale = 1.0 if args.dim_u > 0 else 0.0                               # dim_u = 0: u is a constant zero

    uncaptured_shift = nn.Parameter(torch.zeros(C, max(args.uncaptured_dim, 1), device=device))   # c_t; row 0 = control
    if args.gene_split:
        from pathscm.scm import jaccard_kernel
        source_mask = np.zeros(C, dtype=bool); source_mask[fitted] = True
        transfer_weight = jaccard_kernel(torch.as_tensor(data.support, device=device),
                                        torch.as_tensor(source_mask, device=device))

    def encode(x):
        if args.exogenous:                   # a cell read as unperturbed: z = (I - A)^{-1} mu_eps(x)
            mu, u = states.both(x)
            return shift(mu.double(), use_graph=not args.no_graph and stage == 3).float(), u
        if args.states == "shared":          # z and u from the same network
            return states.both(x)
        z = states(x)
        if args.uncaptured_dim > 0:          # u reads only what the (fixed) pathway read-out leaves unexplained
            with torch.no_grad():
                residual = x - loadings.reconstruct(z)
            return z, u_encoder(residual)
        return z, u_encoder(x) * u_scale

    def shift(inputs: torch.Tensor, use_graph: bool, iterations: int | None = 2) -> torch.Tensor:
        """[conditions, K] mean latent shift (I - A)^{-1} a for the given input rows."""
        if not use_graph:
            return inputs
        A = scm.graph(iterations=iterations)
        B = torch.eye(K, dtype=torch.float64, device=device) - A
        return torch.linalg.solve(B, inputs.T).T

    def exogenous_step(x0, xr, chosen, stage):
        """--exogenous: every cell of the step (start controls and the real cells of each condition) is encoded to its
        exogenous noise eps_i ~ q(eps | x_i) and u_i; its module state is z_i = (I - A)^{-1} (eps_i + a_{t_i}).
        L = L_path(x_i, z_i) + L_perp(u, z) + ELBO + L_gen + lambda_A ||A||_1, nothing detached; ELBO per gene:
        Gaussian NLL of x_i under D(z_i, u_i) + KL(q(eps_i | x_i) || N(0, diag sigma^2)) / G. Generation from the start
        cells: D((I - A)^{-1} (eps_0 + a_t), u_0 + c_t)."""
        xs = x0[0]                                                       # the shared start cells [n, G]
        n = len(xs)
        index = torch.as_tensor(chosen, device=device)
        cells = torch.cat([xs, xr.reshape(-1, G)])
        cell_condition = torch.cat([torch.zeros(n, dtype=torch.long, device=device), index.repeat_interleave(xr.shape[1])])
        mu, logvar, u = states.posterior(cells)
        eps = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        use_graph = not args.no_graph and stage == 3
        inputs = scm.inputs()                                            # [conditions, K]; row 0 (control) = 0
        z = shift(eps.double() + inputs[cell_condition], use_graph).float()
        loss_path = loadings.loss(cells, z)
        loss_rec = gaussian_nll_per_gene(cells, decoder(z, u), noise()).mean()
        sigma = scm.sigma().float()
        kl = (torch.log(sigma) - 0.5 * logvar + (logvar.exp() + mu ** 2) / (2 * sigma ** 2) - 0.5).sum(1).mean()
        uc, zc = u - u.mean(0), z.detach() - z.detach().mean(0)
        corr = (uc.T @ zc) / (uc.norm(dim=0)[:, None] * zc.norm(dim=0)[None, :]).clamp_min(1e-6)
        z0, u0 = z[:n], u[:n]                                            # start cells: z0 = (I - A)^{-1} eps_0
        dz = shift(inputs[index], use_graph).float()                     # (I - A)^{-1} a_t
        perturbed = torch.as_tensor(chosen > 0, device=device, dtype=torch.float32)[:, None, None]
        u_gen = u0[None] + perturbed * uncaptured_shift[index][:, None, :]
        pred = decoder(z0[None] + dz[:, None, :], u_gen)
        loss_dist = energy_distance(pred, xr).mean()
        loss = (args.lambda_path * loss_path + args.lambda_orth * (corr ** 2).mean() + loss_rec + kl / (G if args.exo_kl_per == "gene" else K) + loss_dist)
        if use_graph:
            loss = loss + args.lambda_A * scm.graph(iterations=2).abs().sum()
        return loss, {"step": step, "stage": stage, "rec": float(loss_rec), "path": float(loss_path), "kl": float(kl),
                      "dist": float(loss_dist)}

    parameters = [*states.parameters(), *u_encoder.parameters(), *decoder.parameters(), *noise.parameters(),
                  *(loadings.parameters() if loadings is not None else []),
                  *([uncaptured_shift] if args.uncaptured_dim > 0 else []),
                  *[p for p in scm.parameters() if p.requires_grad]]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    @torch.no_grad()
    def validate() -> dict[str, float]:
        for m in (states, u_encoder, decoder):
            m.eval()
        x0 = X[torch.as_tensor(rng.choice(controls, 512, replace=False), device=device)]
        z0, u0 = encode(x0)
        base = x0.mean(dim=0) if difference else decoder(z0, u0).mean(dim=0)
        val_inputs, val_c = scm.inputs(), uncaptured_shift
        if args.gene_split:
            val_inputs = transfer_weight[:, fitted] @ val_inputs[fitted]
            val_c = transfer_weight[:, fitted].float() @ uncaptured_shift[fitted]
        dz = shift(val_inputs[val_conditions], use_graph=not args.no_graph and stage == 3, iterations=None).float()
        if args.no_scm:
            dz = torch.zeros_like(dz)
        mse, effects, reals = [], [], []
        control_mean = X[torch.as_tensor(controls, device=device)].mean(dim=0)
        for j, t in enumerate(val_conditions):
            u_t = u0 + val_c[t] if args.uncaptured_dim > 0 and not args.no_uncaptured_shift else u0
            pred = respond(x0, z0, u_t, dz[j], u0).mean(dim=0)
            real = X[torch.as_tensor(validation_rows(t), device=device)].mean(dim=0)
            mse.append(float(((pred - real) ** 2).mean()))
            effects.append(pred - base); reals.append(real - control_mean)
        E, R = torch.stack(effects), torch.stack(reals)
        E, R = E - E.mean(0), R - R.mean(0)
        E, R = E - E.mean(1, keepdim=True), R - R.mean(1, keepdim=True)
        specific = float(torch.median((E * R).sum(1) / (E.norm(dim=1) * R.norm(dim=1)).clamp_min(1e-9)))
        result = {"val_mean_mse": float(np.mean(mse)), "val_specific_corr": specific}
        if args.gen_loss != "legacy":     # the training loss on validation cells (control and 100 perturbations)
            tau = noise()
            gen = []
            for j, t in enumerate([0, *val_conditions[:100]]):
                real = X[torch.as_tensor(validation_rows(t), device=device)]
                if t == 0:
                    real = real[torch.as_tensor(rng.choice(len(real), min(len(real), 512), replace=False), device=device)]
                u_t = u0 + val_c[t] if (t > 0 and args.uncaptured_dim > 0 and not args.no_uncaptured_shift) else u0
                pred = respond(x0, z0, u_t, dz[j - 1] if t > 0 else 0 * dz[0], u0)
                sample = pred if args.no_gen_noise else pred + tau * torch.randn_like(pred)
                gen.append(float(distribution_loss(sample[None], real[None])[0]))
            result["val_gen_loss"] = float(np.mean(gen))
        for m in (states, u_encoder, decoder):
            m.train()
        return result

    history, best, best_state, start = [], float("inf"), None, time.time()
    total = args.warm_steps + args.zero_graph_steps + args.graph_steps
    for step in range(1, total + 1):
        stage = 1 if step <= args.warm_steps else 2 if step <= args.warm_steps + args.zero_graph_steps else 3
        chosen = rng.choice(fitted, args.conditions_per_step, replace=False)
        if args.gen_loss != "legacy":             # the control cells are generated too (zero shift, c_0 = 0)
            chosen = np.concatenate([[0], chosen])
        if args.shared_start:                     # one set of control cells is the starting point of every condition
            x0 = X[torch.as_tensor(rng.choice(controls, (1, args.cells)), device=device)].expand(len(chosen), -1, -1)
        else:
            x0 = X[torch.as_tensor(rng.choice(controls, (len(chosen), args.cells)), device=device)]      # [c, n, G]
        xr = X[torch.as_tensor(np.stack([rng.choice(train_by_condition[t], args.cells) for t in chosen]), device=device)]
        if args.exogenous:
            loss, record = exogenous_step(x0, xr, chosen, stage)
        else:
            cells = torch.cat([(x0[:1] if args.shared_start else x0).reshape(-1, G), xr.reshape(-1, G)])
            extra = rng.choice(fitted, args.scm_conditions, replace=False) if args.scm_conditions > 0 else np.array([], dtype=int)
            n_main = len(cells)
            if len(extra):                        # cells of extra perturbations, encoded with the same batch statistics
                cells = torch.cat([cells, X[torch.as_tensor(np.stack([rng.choice(train_by_condition[t], 64) for t in extra]),
                                                             device=device)].reshape(-1, G)])
            z_all, u_all = encode(cells)
            loss_path = torch.zeros((), device=device)
            if loadings is not None:              # the pathway encoder learns only to reconstruct its own pathway's genes
                loss_path = loadings.loss(cells, z_all)
                z_all = z_all.detach()
            z_extra = z_all[n_main:].reshape(len(extra), 64, K)
            cells, z_all, u_all = cells[:n_main], z_all[:n_main], u_all[:n_main]
            tau = noise()
            loss_rec = gaussian_nll_per_gene(cells, decoder(z_all, u_all), tau).mean()
            if args.gen_loss == "legacy":
                loss = (loss_rec if stage == 1 else args.lambda_rec * loss_rec) + args.lambda_path * loss_path
            else:                                     # no cell-wise reconstruction: the generator learns only from the
                loss = args.lambda_path * loss_path   # distribution loss below
                if args.keep_rec:
                    loss = loss + args.lambda_rec * loss_rec
            if args.uncaptured_dim > 0:          # u carries nothing the pathway states already carry: no correlation with z
                uc, zc = u_all - u_all.mean(0), z_all.detach() - z_all.detach().mean(0)
                corr = (uc.T @ zc) / (uc.norm(dim=0)[:, None] * zc.norm(dim=0)[None, :]).clamp_min(1e-6)
                loss = loss + args.lambda_orth * (corr ** 2).mean()
            record = {"step": step, "stage": stage, "rec": float(loss_rec), "path": float(loss_path)}
            if stage >= 2:
                n0 = (1 if args.shared_start else len(chosen)) * args.cells
                z0 = z_all[:n0].reshape(-1, args.cells, K).expand(len(chosen), -1, -1)
                u0 = u_all[:n0].reshape(-1, args.cells, u_all.shape[-1]).expand(len(chosen), -1, -1)
                inputs = scm.inputs()
                dz = shift(inputs[torch.as_tensor(chosen, device=device)], use_graph=not args.no_graph and stage == 3).float()
                # with --scm-only-from-states the structural model is fitted only to the encoded states (L_latent); the
                # generator uses its shift as given and does not move A or the inputs
                gen_dz = dz.detach() if args.scm_only_from_states else dz
                if args.no_scm:
                    gen_dz = torch.zeros_like(dz)
                if args.scm_only_from_states and args.generation_trains_offset and args.inputs == "sensitivity_offset":
                    # the expression losses may move the perturbation-specific deviation delta_t, nothing else of the SCM:
                    # same shift as dz, with A, w and gamma held fixed
                    index = torch.as_tensor(chosen, device=device)
                    own = support[index] * scm.input_offset[index - 1]
                    gen_dz = (dz.detach().double() + shift(own - own.detach(), use_graph=not args.no_graph and stage == 3)).float()
                perturbed = torch.as_tensor(chosen > 0, device=device, dtype=torch.float32)[:, None, None]    # c_0 = 0
                u_gen = (u0 + perturbed * uncaptured_shift[torch.as_tensor(chosen, device=device)][:, None, :]
                         if args.uncaptured_dim > 0 and not args.no_uncaptured_shift else u0)
                pred = respond(x0, z0, u_gen, gen_dz[:, None, :], u0)                                               # [c, n, G]
                loss_mean = ((pred.mean(dim=1) - xr.mean(dim=1)) ** 2).mean()
                sample = pred if (difference or args.no_gen_noise) else pred + tau * torch.randn_like(pred)
                loss_dist = (distribution_loss(sample, xr).mean() if args.gen_loss != "legacy" else
                             torch.stack([mmd(to_pcs(sample[j]), to_pcs(xr[j])) for j in range(len(chosen))]).mean())
                index = torch.as_tensor(chosen, device=device)
                specific = (inputs[index] - scm.generic_strength()[index - 1, None] * scm.generic   # without the generic response
                            if scm.generic_input else inputs[index])
                off = ((1 - support[index]) * specific).abs().sum(1).mean()
                zr = z_all[n0:].reshape(len(chosen), args.cells, K)
                if args.scm_loss == "means":
                    shift_pred, shift_real = dz, (zr.mean(dim=1) - z0.mean(dim=1)).detach()
                    if len(extra):                   # more perturbations per step for the structural fit (less noisy A)
                        shift_pred = torch.cat([dz, shift(inputs[torch.as_tensor(extra, device=device)],
                                                          use_graph=not args.no_graph and stage == 3).float()])
                        shift_real = torch.cat([shift_real, (z_extra.mean(dim=1) - z0.reshape(-1, K).mean(dim=0)).detach()])
                    loss_latent = ((shift_pred - shift_real) ** 2).mean()
                else:                                 # Gaussian likelihood of every cell's (fixed) state under the SCM
                    A_now = scm.graph(iterations=2) if (stage == 3 and not args.no_graph) else torch.zeros(K, K, dtype=torch.float64, device=device)
                    cell_inputs = torch.cat([torch.zeros(n0, K, dtype=torch.float64, device=device),
                                             inputs[torch.as_tensor(chosen, device=device)].repeat_interleave(args.cells, dim=0)])
                    cell_states = z_all.detach().double()
                    if len(extra):                    # cells of the extra perturbations enter the likelihood as well
                        cell_inputs = torch.cat([cell_inputs, inputs[torch.as_tensor(extra, device=device)].repeat_interleave(64, dim=0)])
                        cell_states = torch.cat([cell_states, z_extra.reshape(-1, K).detach().double()])
                    base = scm.base
                    if args.no_base:                  # states measured from the control cells of this step; no constant term
                        cell_states = cell_states - cell_states[:n0].mean(dim=0)
                        base = torch.zeros_like(scm.base)
                    loss_latent = scm_state_nll(cell_states, cell_inputs, A_now, base, scm.sigma(),
                                                scm.noise_factors if args.noise_rank > 0 else None) / K
                if args.gen_loss == "legacy":
                    loss = (loss + args.lambda_mean * loss_mean + args.lambda_dist * loss_dist + args.lambda_local * off
                            + args.lambda_latent * loss_latent)
                else:                                 # one expression loss; the structural model only its likelihood
                    loss = loss + loss_dist + (0.0 if args.no_scm else args.lambda_latent) * loss_latent
                if args.inputs == "sensitivity_offset" and args.lambda_offset > 0:
                    own = index[index > 0]
                    loss = loss + args.lambda_offset * (support[own] * scm.input_offset[own - 1] ** 2).sum(1).mean()
                if stage == 3 and not args.no_graph and not args.no_scm:
                    loss = loss + args.lambda_A * scm.graph(iterations=2).abs().sum()
                record.update({"mean": float(loss_mean), "dist": float(loss_dist), "latent": float(loss_latent)})
        optimizer.zero_grad()
        loss.backward()
        if stage == 1:                                                    # inputs and graph are not trained yet
            for p in scm.parameters():
                p.grad = None
        if stage == 2 and scm.learn_graph:
            scm.raw_graph.grad = None
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if step % args.validate_every == 0:
            record.update(validate())
            history.append(record)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}),
                  f"({time.time() - start:.0f}s)", flush=True)
            criterion = record["val_gen_loss"] if args.gen_loss != "legacy" else record["val_mean_mse"]
            if stage >= 2 and criterion < best:
                best = criterion
                best_state = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                              for name, m in (("states", states), ("u_encoder", u_encoder), ("decoder", decoder),
                                              ("noise", noise), ("scm", scm))}
                if loadings is not None:
                    best_state["loadings"] = {k: v.detach().clone() for k, v in loadings.state_dict().items()}
                if args.uncaptured_dim > 0:
                    best_state["uncaptured_shift"] = uncaptured_shift.detach().clone().cpu()
                best_state["step"], best_state["stage"] = step, stage

    run_dir = ROOT / "checkpoints/pathscm" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.scm_only_from_states:
        # the pathway representation and the structural model are fitted only to genes and pathway states, so they are
        # kept at the end of training; early stopping on the expression error selects only the generator
        best_state["scm"] = {k: v.detach().clone() for k, v in scm.state_dict().items()}
        best_state["states"] = {k: v.detach().clone() for k, v in states.state_dict().items()}
        if loadings is not None:
            best_state["loadings"] = {k: v.detach().clone() for k, v in loadings.state_dict().items()}
    scm_final = {k: v.detach().clone() for k, v in scm.state_dict().items()}
    scm.load_state_dict(best_state["scm"])
    # direction of every pathway state: correlation with the mean expression of its members on training control cells
    # (the sign of z_k is a free gauge; reports multiply node k by sign(corr_k), |corr_k| < 0.1 means not anchored)
    states.load_state_dict(best_state["states"])
    states.eval()
    with torch.no_grad():
        xc = X[torch.as_tensor(controls, device=device)]
        zc = torch.cat([states(xc[i:i + 4096]) for i in range(0, len(xc), 4096)])
        member = xc @ (node_mask / node_mask.sum(dim=0, keepdim=True))
        a_, b_ = zc - zc.mean(0), member - member.mean(0)
        anchor = ((a_ * b_).sum(0) / (a_.norm(dim=0) * b_.norm(dim=0)).clamp_min(1e-12)).cpu()
        if loadings is not None:              # direction of z_k: the loadings of its members sum to a positive number
            loadings.load_state_dict(best_state["loadings"])
            anchor = loadings.orientation().cpu() * anchor.abs()
    scm_state = {"scm": scm.state_dict(), "args": {"radius": args.radius, "amplitude": args.amplitude, "sigma_floor": 0.05},
                 "support": data.support, "node_names": data.node_names, "conditions": data.conditions,
                 "learn_graph": not args.no_graph, "input_mode": args.inputs, "embedding_source": args.embedding_source,
                 "generic_input": not args.no_common_response, "generic_scale": args.inputs == "embedding",
                 "dense_inputs": not args.support_only,
                 "noise_rank": args.noise_rank,
                 "held_out_conditions": np.flatnonzero(held).tolist()}
    if args.gene_split:
        scm_state.update(fitted_conditions=fitted.tolist(), gene_split_sha256=split_digest,
                         validation_conditions=val_conditions.tolist(), unseen_rule='jaccard_a_and_c')
    torch.save({"states": best_state["states"], "u_encoder": best_state["u_encoder"], "decoder": best_state["decoder"],
                "noise": best_state["noise"], "scm_state": scm_state, "args": vars(args),
                "use_graph": not args.no_graph and best_state["stage"] == 3, "best_step": best_state["step"],
                "anchor_correlation": anchor, "loadings": best_state.get("loadings"),
                "uncaptured_shift": best_state.get("uncaptured_shift")},
               run_dir / "response.pth")
    torch.save(scm_state, run_dir / "scm_learned.pth")
    if args.exogenous:                        # also the parameters of the last step (A is trained by every loss here)
        final_dir = ROOT / "checkpoints/pathscm" / f"{args.run_name}_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"states": states.state_dict(), "u_encoder": u_encoder.state_dict(), "decoder": decoder.state_dict(),
                    "noise": noise.state_dict(), "scm_state": {**scm_state, "scm": scm_final}, "args": vars(args),
                    "use_graph": not args.no_graph, "best_step": step, "loadings": loadings.state_dict(),
                    "uncaptured_shift": uncaptured_shift.detach().clone().cpu()}, final_dir / "response.pth")
        torch.save({**scm_state, "scm": scm_final}, final_dir / "scm_learned.pth")                     # for evaluate.py / graph tools
    with torch.no_grad():
        np.save(run_dir / "A_learned.npy", scm.graph().cpu().numpy())
    (run_dir / "response_history.json").write_text(json.dumps(history, indent=1))
    print("best step", best_state["step"], "stage", best_state["stage"], "validation criterion", round(best, 5))


if __name__ == "__main__":
    main()
