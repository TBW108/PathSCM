"""scGen runs (baselines/scgen.py, checkpoints/baselines/scgen_*.npz) in our generator interface, so bench_datasets.py
and bench_umap.py score them like the other methods.

A perturbed population is generated the way SCGEN.predict generates it: training control cells are encoded (posterior
mean, exported by run_scgen.py), the perturbation's latent arithmetic delta_t is added and the scGen decoder (FCLayers:
Linear, BatchNorm(eps 1e-3), LeakyReLU, twice; then Linear; eval mode) maps back to log1p expression, which is
standardized like the data. scGen has no observation noise in predict, so the decoded cells are both mean and sample.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def scgen_decoder(weights: dict[str, np.ndarray]) -> nn.Module:
    layers, i = [], 0
    while f"decoder.decoder.fc_layers.Layer {i}.0.weight" in weights:
        prefix = f"decoder.decoder.fc_layers.Layer {i}"
        linear = nn.Linear(*weights[f"{prefix}.0.weight"].shape[::-1])
        norm = nn.BatchNorm1d(linear.out_features, momentum=0.01, eps=0.001)
        linear.load_state_dict({k: torch.as_tensor(weights[f"{prefix}.0.{k}"]) for k in ("weight", "bias")})
        norm.load_state_dict({k: torch.as_tensor(weights[f"{prefix}.1.{k}"])
                              for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")})
        layers += [linear, norm, nn.LeakyReLU()]
        i += 1
    out = nn.Linear(*weights["decoder.linear_out.weight"].shape[::-1])
    out.load_state_dict({k: torch.as_tensor(weights[f"decoder.linear_out.{k}"]) for k in ("weight", "bias")})
    return nn.Sequential(*layers, out).eval()


class ScGenGenerator:
    def __init__(self, path: str, data, device: str, controls: torch.Tensor):
        run = np.load(path)
        if len(run["control_latent"]) != len(controls):
            raise ValueError(f"{path}: {len(run['control_latent'])} encoded control cells, the benchmark uses {len(controls)}")
        self.decoder = scgen_decoder({k: run[k] for k in run.files if k.startswith("decoder.")}).to(device)
        self.latent = torch.as_tensor(run["control_latent"], device=device)
        self.delta = {str(c): torch.as_tensor(d, device=device) for c, d in zip(run["conditions"], run["delta"])}
        self.mean = torch.as_tensor(data.train_mean, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(data.train_sd, dtype=torch.float32, device=device)
        self.conditions, self.device = list(data.conditions), device

    def condition_input(self, condition: int) -> int:
        return condition

    @torch.no_grad()
    def sample_population(self, condition: int, n_cells: int, seed: int = 0) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = self.latent[torch.randint(len(self.latent), (n_cells,), device=self.device, generator=generator)]
        if condition > 0:
            z = z + self.delta[self.conditions[condition]]
        cells = (self.decoder(z) - self.mean) / self.sd
        return {"mean": cells, "sample": cells}
