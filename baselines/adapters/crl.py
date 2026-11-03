"""discrepancy-VAE / SENA-discrepancy-VAE runs (baselines/_crl.py, checkpoints/crl/<run>) in our generator interface, so
compare_seen.py and landing.py score them exactly like our runs.

A perturbed population is generated the way these models generate it: from training control cells x_0, a latent z is
drawn from the encoder posterior, the perturbation adds bc_t * s_t (at the final softmax temperature) and
y = decode((z + bc_t s_t)(I - G_u)^-1). The models work on the log1p scale; their output is standardized like the data.
They have no observation noise, so the decoded cells are both the mean and the sample.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CRL_DIR = ROOT / "checkpoints/crl"
SOURCES = {"dvae": ROOT / "baselines/sota_src/discrepancy_vae/src",
           "sena": ROOT / "baselines/sota_src/SENA/src/sena_discrepancy_vae"}


def is_crl_run(run: str) -> bool:
    return (CRL_DIR / run / "best_model.pt").exists()


def load_official_model(run_dir: Path, device: str):
    """The saved model is a pickled CMVAE of the module 'model' of its repository; import that module first."""
    method = "sena" if "sena_lambda" in json.loads((run_dir / "config.json").read_text()) else "dvae"
    for name in ("model", "utils", "dataset", "train"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(SOURCES[method]))
    importlib.import_module("model")
    sys.path.pop(0)
    return torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False).eval()


class CRLGenerator:
    def __init__(self, run: str, data, device: str, controls: torch.Tensor):
        run_dir = CRL_DIR / run
        self.model = load_official_model(run_dir, device)
        self.temp = float(json.loads((run_dir / "config.json").read_text())["mxTemp"])
        self.device = device
        self.mean = torch.as_tensor(data.train_mean, dtype=torch.float64, device=device)
        self.sd = torch.as_tensor(data.train_sd, dtype=torch.float64, device=device)
        self.control = controls
        K = self.model.z_dim
        G_u = torch.triu(self.model.G, diagonal=1)
        self.inverse = torch.inverse(torch.eye(K, dtype=G_u.dtype, device=device) - G_u)
        self.n_perturbations = self.model.c_dim

    def condition_input(self, condition: int) -> int:
        return condition

    @torch.no_grad()
    def sample_population(self, condition: int, n_cells: int, seed: int = 0) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x0 = self.control[torch.randint(len(self.control), (n_cells,), device=self.device, generator=generator)]
        mu, var = self.model.encode(x0.double() * self.sd + self.mean)
        z = mu + var.sqrt() * torch.randn(mu.shape, dtype=mu.dtype, device=self.device, generator=generator)
        if condition > 0:
            c = torch.zeros(1, self.n_perturbations, dtype=torch.float64, device=self.device)
            c[0, condition - 1] = 1
            bc, s = self.model.c_encode(c, self.temp)
            z = z + bc * s[:, None]
        y = (self.model.decode(z @ self.inverse) - self.mean) / self.sd
        return {"latent": z, "mean": y.float(), "sample": y.float()}
