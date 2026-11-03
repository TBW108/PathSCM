"""Pathway SCM trained on responses from control cells (plan of 2026-10-20, CausCell-style split into z and u).

Every cell is encoded into
    z = pathway states: one scalar per pathway, each read only from that pathway's genes; either a fixed score
        (PathwayScores: equal-weight mean of the standardized member genes) or learned (PathwayEncoder, standardized),
    u = an embedding of the whole expression vector, for variation not assigned to the pathways (FullGeneEncoder).
A decoder D(z, u) gives the mean expression; the observation noise is one standard deviation per gene.

A perturbation g changes only the pathway states, through the structural model:
    z_g = z_0 + (I - A)^{-1} a_g,      u_g = u_0,      x_g ~ N(D(z_g, u_0), diag(tau^2)),
starting from control cells x_0 (their z_0, u_0 are kept jointly). a_g comes from LinearPathwaySCM.inputs().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pathscm.scm import scm_from_checkpoint
from pathscm.components import PathwayEncoder


class FullGeneEncoder(nn.Module):
    """u = E_u(x): all genes -> hidden -> hidden -> d_u, deterministic; never sees the perturbation."""

    def __init__(self, n_genes: int, dim: int = 32, hidden: tuple[int, int] = (512, 128)):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_genes, hidden[0]), nn.SiLU(), nn.Linear(hidden[0], hidden[1]), nn.SiLU(),
                                     nn.Linear(hidden[1], dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LinearDecoder(nn.Module):
    """D(z) = W z + c: gene g changes by W_gk per unit of pathway state k (u is ignored)."""

    def __init__(self, n_pathways: int, n_genes: int):
        super().__init__()
        self.linear = nn.Linear(n_pathways, n_genes)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class ResidualDecoder(nn.Module):
    """D(z, u): [z, u] -> width, then residual blocks into each of which a projection of z is added again.
    With u_twice the input is [z, u0, u]: the start cell's own uncaptured embedding u0 next to the shifted u = u0 + c_t
    (u0 = u when not given, e.g. for the control condition)."""

    def __init__(self, n_pathways: int, dim_u: int, n_genes: int, width: int = 512, blocks: int = 3, u_twice: bool = False):
        super().__init__()
        self.u_twice = u_twice
        self.input = nn.Linear(n_pathways + dim_u * (2 if u_twice else 1), width)
        self.z_projection = nn.ModuleList([nn.Linear(n_pathways, width) for _ in range(blocks)])
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
                                     for _ in range(blocks)])
        self.output = nn.Linear(width, n_genes)

    def forward(self, z: torch.Tensor, u: torch.Tensor, u0: torch.Tensor | None = None) -> torch.Tensor:
        if self.u_twice:
            u = torch.cat([u if u0 is None else u0.expand_as(u), u], dim=-1)
        h = self.input(torch.cat([z, u], dim=-1))
        for block, projection in zip(self.blocks, self.z_projection):
            h = h + block(h + projection(z))
        return self.output(F.silu(h))


class MaskedPathwayDecoder(nn.Module):
    """Pathway-local decoder (as in VEGA / expiMap): a gene is read out only from the pathways it belongs to, plus u.

        f_k = phi_k(z_k) in R^m                       (a small MLP per pathway, on the scalar state of pathway k)
        x_g = c_g + sum_{k in P(g)} w_gk . f_k + v_g . psi(u)

    P(g) is the set of pathways containing gene g; genes in no pathway read all pathways. A perturbation that moves
    only the target's pathways therefore changes the genes of other pathways only through the graph A.
    """

    def __init__(self, node_mask: torch.Tensor, dim_u: int, features: int = 8, hidden: int = 16, background: int = 16):
        super().__init__()
        member = (node_mask > 0).float()                                                   # [G, K]
        member[member.sum(dim=1) == 0] = 1.0                                               # genes in no pathway read all
        G, K = member.shape
        device = member.device
        self.K, self.features = K, features
        self.register_buffer("readout_mask", member.repeat_interleave(features, dim=1))   # [G, K * m]
        self.phi_w1 = nn.Parameter(torch.randn(K, hidden, device=device))
        self.phi_b1 = nn.Parameter(torch.zeros(K, hidden, device=device))
        self.phi_w2 = nn.Parameter(torch.randn(K, features, hidden, device=device) / hidden ** 0.5)
        self.phi_b2 = nn.Parameter(torch.zeros(K, features, device=device))
        self.readout = nn.Parameter(torch.randn(G, K * features, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(G, device=device))
        self.background = nn.Sequential(nn.Linear(dim_u, 64), nn.SiLU(), nn.Linear(64, background)).to(device)
        self.background_readout = nn.Linear(background, G, bias=False).to(device)

    def pathway_features(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(z[..., None] * self.phi_w1 + self.phi_b1)                              # [..., K, hidden]
        return F.silu(torch.einsum("...kh,kfh->...kf", h, self.phi_w2) + self.phi_b2)    # [..., K, m]

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        f = self.pathway_features(z).flatten(-2)
        return f @ (self.readout * self.readout_mask).T + self.bias + self.background_readout(self.background(u))


class PathwayStates(nn.Module):
    """z: pathway encoder means, standardized with running statistics (batch statistics while training)."""

    def __init__(self, node_mask: torch.Tensor, hidden: int = 64, embedding: int = 16, depth: int = 1, momentum: float = 0.01):
        super().__init__()
        self.encoder = PathwayEncoder(node_mask, hidden, embedding, 1e-4, depth)
        K = node_mask.shape[1]
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(K, device=node_mask.device))
        self.register_buffer("running_var", torch.ones(K, device=node_mask.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.encoder(x)
        if self.training:
            center, var = mean.mean(dim=0), mean.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(center.detach(), self.momentum)
                self.running_var.lerp_(var.detach(), self.momentum)
        else:
            center, var = self.running_mean, self.running_var
        return (mean - center) / (var + 1e-5).sqrt()


class GeneStates(nn.Module):
    """z: one number per pathway from an MLP over all genes (G -> 512 -> 256 -> K), standardized like PathwayStates.
    Which pathway a dimension stands for is fixed by the masked read-out (JointLoadings), not by the encoder."""

    def __init__(self, n_genes: int, n_pathways: int, momentum: float = 0.01):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_genes, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, n_pathways))
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(n_pathways))
        self.register_buffer("running_var", torch.ones(n_pathways))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.network(x)
        if self.training:
            center, var = h.mean(dim=0), h.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(center.detach(), self.momentum)
                self.running_var.lerp_(var.detach(), self.momentum)
        else:
            center, var = self.running_mean, self.running_var
        return (h - center) / (var + 1e-5).sqrt()


class SharedEncoder(nn.Module):
    """One MLP over all genes with two outputs: the K pathway activities z and the uncaptured embedding u.

    trunk: G -> 512 -> 256 (SiLU); z = standardized linear head (K, one per pathway; which pathway a dimension stands for
    is fixed by the masked joint read-out, JointLoadings); u = linear head (dim_u). z is standardized like PathwayStates
    (batch statistics while training, running statistics afterwards)."""

    def __init__(self, n_genes: int, n_pathways: int, dim_u: int, momentum: float = 0.01):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_genes, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU())
        self.pathway_head = nn.Linear(256, n_pathways)
        self.uncaptured_head = nn.Linear(256, dim_u)
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(n_pathways))
        self.register_buffer("running_var", torch.ones(n_pathways))

    def both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        raw = self.pathway_head(h)
        if self.training:
            center, var = raw.mean(dim=0), raw.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(center.detach(), self.momentum)
                self.running_var.lerp_(var.detach(), self.momentum)
        else:
            center, var = self.running_mean, self.running_var
        return (raw - center) / (var + 1e-5).sqrt(), self.uncaptured_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.both(x)[0]

    def uncaptured(self, x: torch.Tensor) -> torch.Tensor:
        return self.both(x)[1]


class ExogenousEncoder(SharedEncoder):
    """SharedEncoder whose pathway head is the posterior mean of the exogenous noise eps (standardized the same way), plus
    a head for its log variance: q(eps | x) = N(mu(x), diag exp(logvar(x))). Used with --exogenous, where the module
    states are generated by the SCM, z = (I - A)^{-1} (eps + a_t)."""

    def __init__(self, n_genes: int, n_pathways: int, dim_u: int, momentum: float = 0.01):
        super().__init__(n_genes, n_pathways, dim_u, momentum)
        self.logvar_head = nn.Linear(256, n_pathways)

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, u = self.both(x)
        return mu, self.logvar_head(self.trunk(x)).clamp(-10.0, 4.0), u


class ExogenousStates(nn.Module):
    """Module states of a --exogenous run for a cell read as unperturbed: z = (I - A)^{-1} mu_eps(x)."""

    def __init__(self, encoder: ExogenousEncoder, graph: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("B", torch.eye(len(graph), dtype=torch.float64, device=graph.device) - graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.B, self.encoder(x).double().T).T.float()

    def uncaptured(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.uncaptured(x)


class JointLoadings(nn.Module):
    """Masked linear read-out of all genes from all pathway states: x_hat_g = sum_k M_gk U_gk z_k + c_g.

    A gene is explained jointly by every pathway it belongs to; genes in no pathway keep only c_g. Column U[:, k]
    holds the loadings of pathway k's members, so dimension k means "activity of pathway k"."""

    def __init__(self, node_mask: torch.Tensor):
        super().__init__()
        member = (node_mask > 0).float()
        self.register_buffer("member", member)                                            # [G, K]
        self.loading = nn.Parameter(0.1 * member.clone())
        self.offset = nn.Parameter(torch.zeros(member.shape[0], device=member.device))

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return z @ (self.loading * self.member).T + self.offset

    def loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return ((x - self.reconstruct(z)) ** 2).mean()

    @torch.no_grad()
    def orientation(self) -> torch.Tensor:
        return torch.where((self.loading * self.member).sum(0) < 0, -1.0, 1.0)


class PathwayLoadings(nn.Module):
    """Linear read-out of every pathway from its own state: x_hat[g, k] = U[g, k] z_k + C[g, k] for the members g of k.

    loss(x, z) = mean over cells and over (pathway, member) pairs of (x_g - U_gk z_k - C_gk)^2. It trains a pathway
    encoder so that z_k is the one number that best reconstructs the genes of pathway k, and nothing else; U[:, k] are
    the gene loadings of pathway k. Computed from sums over genes, without the [cells, genes, pathways] tensor."""

    def __init__(self, node_mask: torch.Tensor):
        super().__init__()
        member = (node_mask > 0).float()
        self.register_buffer("member", member)                                            # [G, K]
        self.loading = nn.Parameter(0.1 * member.clone())
        self.offset = nn.Parameter(torch.zeros_like(member))

    def loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        U, C = self.loading * self.member, self.offset * self.member
        total = ((x ** 2) @ self.member).sum(1) - 2 * (z * (x @ U)).sum(1) - 2 * (x @ C).sum(1) \
            + (z ** 2 * (U ** 2).sum(0)).sum(1) + 2 * (z * (U * C).sum(0)).sum(1) + (C ** 2).sum()
        return (total / self.member.sum()).mean()

    @torch.no_grad()
    def orientation(self) -> torch.Tensor:
        """+1 / -1 per pathway: the sign that makes the loadings of its members sum to a positive number."""
        return torch.where((self.loading * self.member).sum(0) < 0, -1.0, 1.0)


class PathwayScores(nn.Module):
    """Fixed pathway states (no parameters): z_k = mean over the members g of pathway k of (x_g - m_g) / s_g, divided by the
    standard deviation of that mean; m, s and the standard deviation come from training control cells (fit_).
    The same function of the cell in every run, so z_k means the same thing across seeds."""

    def __init__(self, node_mask: torch.Tensor):
        super().__init__()
        member = (node_mask > 0).float()                                                   # [G, K]
        G, K = member.shape
        self.register_buffer("weight", member / member.sum(dim=0, keepdim=True).clamp_min(1.0))
        self.register_buffer("gene_mean", torch.zeros(G, device=member.device))
        self.register_buffer("gene_scale", torch.ones(G, device=member.device))
        self.register_buffer("score_scale", torch.ones(K, device=member.device))

    @torch.no_grad()
    def fit_(self, controls: torch.Tensor, shifts: torch.Tensor | None = None) -> "PathwayScores":
        """shifts [perturbations, G] (optional): mean expression shifts of the fitted perturbations. Given, the member
        weights of pathway k become its first principal axis of these shifts (sign: positive weight sum, sum |w| = 1)
        instead of equal weights: the direction along which perturbations move the pathway."""
        self.gene_mean.copy_(controls.mean(dim=0))
        self.gene_scale.copy_(controls.std(dim=0).clamp_min(1e-3))
        if shifts is not None:
            standardized = shifts / self.gene_scale
            member = self.weight > 0
            for k in range(member.shape[1]):
                genes = torch.nonzero(member[:, k]).flatten()
                block = standardized[:, genes] - standardized[:, genes].mean(dim=0)
                w = torch.linalg.svd(block, full_matrices=False)[2][0]
                w = w * (1.0 if w.sum() >= 0 else -1.0)
                self.weight[:, k] = 0.0
                self.weight[genes, k] = w / w.abs().sum()
        self.score_scale.copy_((((controls - self.gene_mean) / self.gene_scale) @ self.weight).std(dim=0).clamp_min(1e-3))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (((x - self.gene_mean) / self.gene_scale) @ self.weight) / self.score_scale


class ObservationNoise(nn.Module):
    def __init__(self, n_genes: int, floor: float = 0.1):
        super().__init__()
        self.floor = floor
        self.raw = nn.Parameter(torch.zeros(n_genes))

    def forward(self) -> torch.Tensor:
        return self.floor + F.softplus(self.raw)


def scm_state_nll(z: torch.Tensor, inputs: torch.Tensor, A: torch.Tensor, base: torch.Tensor, sigma: torch.Tensor,
                  factors: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over cells of -log p(z_i) - (K/2) log(2 pi) under z = A z + b + a_i + eps, eps ~ N(0, diag(sigma^2) + F F^T),
    i.e. z_i ~ N((I - A)^{-1} (b + a_i), (I - A)^{-1} Sigma (I - A)^{-T}); z [n, K] states, inputs [n, K] the a_i."""
    B = torch.eye(A.shape[0], dtype=A.dtype, device=A.device) - A
    eps = z @ B.T - base - inputs
    if factors is None:
        quadratic, log_det_sigma = ((eps / sigma) ** 2).sum(1), 2 * torch.log(sigma).sum()
    else:
        chol = torch.linalg.cholesky(torch.diag(sigma ** 2) + factors @ factors.T)
        quadratic = (torch.linalg.solve_triangular(chol, eps.T, upper=False) ** 2).sum(0)
        log_det_sigma = 2 * torch.log(torch.diagonal(chol)).sum()
    return 0.5 * quadratic.mean() + 0.5 * log_det_sigma - torch.linalg.slogdet(B)[1]


def gaussian_nll_per_gene(x: torch.Tensor, mean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Mean over genes of -log N(x | mean, tau^2), one value per cell."""
    return (0.5 * ((x - mean) / tau) ** 2 + torch.log(tau) + 0.9189385332046727).mean(dim=-1)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased energy distance between cell sets x [..., n, G] and y [..., m, G] (leading dimensions are batched):
    2 E|X - Y| - E|X - X'| - E|Y - Y'|, distances as root mean squared difference over genes. It is the MMD with the
    distance kernel; it is zero only when the two distributions are equal, and the unbiased form (no self-pairs) has
    its expectation minimised by the true distribution, so it pushes neither towards over- nor under-dispersion."""
    def mean_distance(a: torch.Tensor, b: torch.Tensor, same: bool) -> torch.Tensor:
        squared = (a.square().sum(-1)[..., :, None] + b.square().sum(-1)[..., None, :] - 2 * a @ b.transpose(-1, -2))
        distance = (squared.clamp_min(1e-9) / a.shape[-1]).sqrt()
        if not same:
            return distance.mean((-2, -1))
        n = a.shape[-2]
        off = ~torch.eye(n, dtype=torch.bool, device=a.device)
        return (distance * off).sum((-2, -1)) / (n * (n - 1))
    return 2 * mean_distance(x, y, False) - mean_distance(x, x, True) - mean_distance(y, y, True)


def sinkhorn_divergence(x: torch.Tensor, y: torch.Tensor, eps: float = 0.5, iterations: int = 30) -> torch.Tensor:
    """Debiased entropic optimal-transport distance between equally weighted cell sets x [..., n, G] and y [..., m, G]
    (leading dimensions are batched): S(x, y) = OT(x, y) - (OT(x, x) + OT(y, y)) / 2, cost = mean squared difference
    over genes. As eps -> 0 it tends to the squared Wasserstein-2 distance; as eps grows, to the squared distance
    between the two means. The dual potentials are found without gradient; one last update with gradient gives the
    gradient of the transport cost (envelope theorem)."""
    def transport(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cost = torch.cdist(a, b).square() / a.shape[-1]
        log_a, log_b = -float(np.log(a.shape[-2])), -float(np.log(b.shape[-2]))
        with torch.no_grad():
            f, g = torch.zeros(cost.shape[:-1], device=a.device), torch.zeros(cost.shape[:-2] + cost.shape[-1:], device=a.device)
            for _ in range(iterations):
                f = -eps * torch.logsumexp(log_b + (g[..., None, :] - cost) / eps, dim=-1)
                g = -eps * torch.logsumexp(log_a + (f[..., :, None] - cost) / eps, dim=-2)
        f = -eps * torch.logsumexp(log_b + (g[..., None, :] - cost) / eps, dim=-1)
        g = -eps * torch.logsumexp(log_a + (f.detach()[..., :, None] - cost) / eps, dim=-2)
        return f.mean(-1) + g.mean(-1)
    return transport(x, y) - 0.5 * transport(x, x) - 0.5 * transport(y, y)


class ResponseGenerator:
    """Generation from a trained response model for current benchmarks and interpretation."""

    def __init__(self, run_dir: str | Path, device: str, node_mask: np.ndarray, control_cells: torch.Tensor):
        state = torch.load(Path(run_dir) / "response.pth", map_location=device, weights_only=False)
        args = state["args"]
        mask = torch.as_tensor(node_mask, device=device)
        if args.get("states", "learned") in ("score", "pc1"):
            self.states = PathwayScores(mask).to(device)
        elif args.get("states") == "joint":
            self.states = GeneStates(mask.shape[0], mask.shape[1]).to(device)
        elif args.get("states") == "shared":
            self.states = (ExogenousEncoder if args.get("exogenous") else SharedEncoder)(mask.shape[0], mask.shape[1],
                                                                                          args["dim_u"]).to(device)
        else:
            self.states = PathwayStates(mask, args["encoder_hidden"], args["embedding"], args["encoder_depth"]).to(device)
        self.states.load_state_dict(state["states"])
        self.u_scale = 1.0 if args["dim_u"] > 0 else 0.0                  # dim_u = 0: no u (a constant zero)
        self.u_encoder = FullGeneEncoder(mask.shape[0], max(args["dim_u"], 1)).to(device)
        self.u_encoder.load_state_dict(state["u_encoder"])
        # uncaptured embedding: u is read from what the pathway read-out leaves unexplained, and each perturbation
        # moves it by its own c_t (zero for control and for perturbations never fitted)
        self.uncaptured = args.get("uncaptured_dim", 0) > 0
        if self.uncaptured:
            self.loadings = JointLoadings(mask).to(device)
            self.loadings.load_state_dict(state["loadings"])
            self.uncaptured_shift = state["uncaptured_shift"].to(device)
        self.current_condition = 0
        if args.get("decoder", "residual") == "masked":
            self.decoder = MaskedPathwayDecoder(mask, max(args["dim_u"], 1)).to(device)
        elif args.get("decoder") == "linear":
            self.decoder = LinearDecoder(mask.shape[1], mask.shape[0]).to(device)
        else:
            self.decoder = ResidualDecoder(mask.shape[1], max(args["dim_u"], 1), mask.shape[0], args["decoder_width"],
                                           args["decoder_blocks"], u_twice=bool(args.get("decoder_u_twice"))).to(device)
        self.decoder.load_state_dict(state["decoder"])
        self.noise = ObservationNoise(mask.shape[0]).to(device)
        self.noise.load_state_dict(state["noise"])
        self.scm = scm_from_checkpoint(state["scm_state"]).to(device)
        for module in (self.states, self.u_encoder, self.decoder, self.noise):
            module.eval()
        self.use_graph = state.get("use_graph", True)
        if args.get("exogenous"):               # states of a cell = (I - A)^{-1} mu_eps(x); generation adds (I - A)^{-1} a_t
            with torch.no_grad():
                graph = self.scm.graph() if self.use_graph else torch.zeros(mask.shape[1], mask.shape[1], dtype=torch.float64,
                                                                            device=device)
            self.states = ExogenousStates(self.states, graph).eval()
        self.difference = args.get("output", "absolute") == "difference"
        self.device, self.control = device, control_cells
        self.tau = self.noise().detach()
        self.no_noise = bool(args.get("no_gen_noise", False))
        self.no_scm = bool(args.get("no_scm", False))            # ablation: no module shift at all

    @torch.no_grad()
    def condition_input(self, condition: int) -> torch.Tensor:
        self.current_condition = condition              # sample_population adds this condition's uncaptured shift
        return self.scm.inputs()[condition]

    def encode_u(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if isinstance(self.states, (SharedEncoder, ExogenousStates)):
            return self.states.uncaptured(x)
        if self.uncaptured:
            return self.u_encoder(x - self.loadings.reconstruct(z))
        return self.u_encoder(x) * self.u_scale

    @torch.no_grad()
    def shift(self, input_vector: torch.Tensor) -> torch.Tensor:
        if self.no_scm:
            return torch.zeros_like(input_vector, dtype=torch.float64)
        if not self.use_graph:
            return input_vector
        B = torch.eye(len(input_vector), dtype=torch.float64, device=input_vector.device) - self.scm.graph()
        return torch.linalg.solve(B, input_vector.to(torch.float64))

    @torch.no_grad()
    def sample_population(self, input_vector: torch.Tensor, n_cells: int, seed: int = 0) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        index = torch.randint(len(self.control), (n_cells,), device=self.device, generator=generator)
        x0 = self.control[index]
        z0 = self.states(x0)
        u0 = self.encode_u(x0, z0)
        z = z0 + self.shift(input_vector).float()
        u = u0 + self.uncaptured_shift[self.current_condition] if self.uncaptured else u0
        if self.difference:                     # the control cell keeps its own expression; only the change is decoded
            mean = x0 + self.decoder(z, u) - self.decoder(z0, u0)
            return {"latent": z, "mean": mean, "sample": mean}
        mean = self.decoder(z, u, u0) if getattr(self.decoder, "u_twice", False) else self.decoder(z, u)
        if self.no_noise:                       # trained with --no-gen-noise: the decoder output is the cell
            return {"latent": z, "mean": mean, "sample": mean}
        sample = mean + self.tau * torch.randn(mean.shape, device=self.device, generator=generator)
        return {"latent": z, "mean": mean, "sample": sample}
