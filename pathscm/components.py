"""Pathway-constrained VAE: encoder, observation model and fixed latent coordinates (specification 4.1, 4.2, 5).

Every pathway k has its own small MLP that reads only the member genes of k,
and two heads for the mean and variance of one scalar state:

    h_ik = E_k(x_i[G_k]),   mu_ik = r_k^mu(h_ik),   v_ik = softplus(r_k^v(h_ik)) + v_min,
    q(z_i | x_i) = N(mu_i, diag(v_i)).

The pathway states are never mixed before the SCM. A shared decoder maps the
K states to the mean of every gene, p(x | z) = N(D(z), tau^2 I).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PathwayEncoder(nn.Module):
    """K masked MLPs (members -> hidden -> embedding) with separate mean and variance heads."""

    def __init__(self, node_mask: torch.Tensor, hidden: int = 64, embedding: int = 16, variance_floor: float = 1e-4,
                 depth: int = 1):
        super().__init__()
        member = (node_mask.float().T > 0).float()                                  # [pathways, genes]
        n_pathways, n_genes = member.shape
        self.n_pathways, self.hidden, self.embedding, self.variance_floor = n_pathways, hidden, embedding, variance_floor
        self.register_buffer("member", member)
        device = member.device
        fan_in = member.sum(dim=1).clamp_min(1.0)
        self.first_weight = nn.Parameter(torch.randn(n_pathways, hidden, n_genes, device=device) / fan_in.sqrt()[:, None, None])
        self.first_bias = nn.Parameter(torch.zeros(n_pathways, hidden, device=device))
        # depth - 1 further hidden layers per pathway (still no mixing between pathways)
        self.extra_weight = nn.ParameterList([nn.Parameter(torch.randn(n_pathways, hidden, hidden, device=device) / math.sqrt(hidden))
                                              for _ in range(depth - 1)])
        self.extra_bias = nn.ParameterList([nn.Parameter(torch.zeros(n_pathways, hidden, device=device)) for _ in range(depth - 1)])
        self.second_weight = nn.Parameter(torch.randn(n_pathways, embedding, hidden, device=device) / math.sqrt(hidden))
        self.second_bias = nn.Parameter(torch.zeros(n_pathways, embedding, device=device))
        self.mean_weight = nn.Parameter(torch.randn(n_pathways, embedding, device=device) / math.sqrt(embedding))
        self.mean_bias = nn.Parameter(torch.zeros(n_pathways, device=device))
        self.variance_weight = nn.Parameter(torch.zeros(n_pathways, embedding, device=device))
        self.variance_bias = nn.Parameter(torch.full((n_pathways,), math.log(math.expm1(1.0)), device=device))  # v = 1 at the start

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """h [cells, pathways, embedding]; pathway k reads only its member genes."""
        weight = (self.first_weight * self.member[:, None, :]).reshape(-1, x.shape[-1])
        hidden = F.silu(x @ weight.T + self.first_bias.reshape(-1)).reshape(len(x), self.n_pathways, self.hidden)
        for w, b in zip(self.extra_weight, self.extra_bias):
            hidden = F.silu(torch.einsum("nkh,kjh->nkj", hidden, w) + b)
        return F.silu(torch.einsum("nkh,keh->nke", hidden, self.second_weight) + self.second_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        mean = (h * self.mean_weight).sum(dim=-1) + self.mean_bias
        variance = F.softplus((h * self.variance_weight).sum(dim=-1) + self.variance_bias) + self.variance_floor
        return mean, variance


class ExpressionDecoder(nn.Module):
    """Mean of every gene given the K pathway states: K -> hidden -> G, no output activation."""

    def __init__(self, n_pathways: int, n_genes: int, hidden: int = 128, depth: int = 1):
        super().__init__()
        layers = [nn.Linear(n_pathways, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.network = nn.Sequential(*layers, nn.Linear(hidden, n_genes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """-log N(x | mean, diag(tau^2)), summed over genes, one value per cell; tau is a scalar or one value per gene."""
    log_tau = math.log(tau) if isinstance(tau, (int, float)) else torch.log(tau)
    return (0.5 * ((x - mean) / tau) ** 2 + log_tau + 0.5 * math.log(2 * math.pi)).sum(dim=-1)


def standard_normal_kl(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """KL(N(mean, diag(variance)) || N(0, I)), summed over pathways, one value per cell."""
    return 0.5 * (mean ** 2 + variance - torch.log(variance) - 1.0).sum(dim=-1)


class LatentCoordinates(nn.Module):
    """Fixed coordinates of the pathway states, set on training control cells (specification 5).

        c_k = E[mu_k],  d_k^2 = Var(mu_k) + E[v_k],  z_bar_k = s_k (z_k - c_k) / d_k,  v_bar_k = v_k / d_k^2,

    with s_k = +1 or -1 chosen so that z_bar_k rises with the mean expression of
    the members of k. Nodes whose posterior mean barely correlates with that
    score (|corr| < 0.1) are flagged as not anchored in direction.
    """

    def __init__(self, n_pathways: int):
        super().__init__()
        self.register_buffer("center", torch.zeros(n_pathways))
        self.register_buffer("scale", torch.ones(n_pathways))
        self.register_buffer("sign", torch.ones(n_pathways))
        self.register_buffer("anchor_correlation", torch.zeros(n_pathways))
        self.register_buffer("small_scale", torch.zeros(n_pathways, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, mean: torch.Tensor, variance: torch.Tensor, member_score: torch.Tensor, scale_floor: float = 1e-3) -> None:
        """``mean``, ``variance``: posteriors of training control cells; ``member_score``: their mean member expression per node."""
        self.center.copy_(mean.mean(dim=0))
        scale = (mean.var(dim=0) + variance.mean(dim=0)).sqrt()
        self.small_scale.copy_(scale < scale_floor)
        self.scale.copy_(scale.clamp_min(scale_floor))
        a, b = mean - mean.mean(dim=0), member_score - member_score.mean(dim=0)
        correlation = (a * b).sum(dim=0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)
        self.anchor_correlation.copy_(correlation)
        self.sign.copy_(torch.where(correlation < 0, -1.0, 1.0))

    @property
    def anchored(self) -> torch.Tensor:
        return self.anchor_correlation.abs() >= 0.1

    def to_fixed(self, mean: torch.Tensor, variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sign * (mean - self.center) / self.scale, variance / self.scale ** 2

    def from_fixed(self, z_bar: torch.Tensor) -> torch.Tensor:
        return self.center + self.scale * self.sign * z_bar


class CalibratedDecoder(nn.Module):
    """D_bar(z_bar) = D(c + d * s * z_bar): the frozen decoder in the fixed coordinates (output unchanged)."""

    def __init__(self, decoder: ExpressionDecoder, coordinates: LatentCoordinates):
        super().__init__()
        self.decoder, self.coordinates = decoder, coordinates

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.coordinates.from_fixed(z_bar))
