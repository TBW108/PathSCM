"""Linear Gaussian pathway SCM with feedback (specification 4.3-4.5, 6.2, 7).

    z = A z + b + a_t + eps,  eps ~ N(0, diag(sigma^2))
    p_A(z | t) = N(T (b + a_t), T Sigma T^T),  T = (I - A)^{-1}

A[k, l] is the effect of pathway l (source) on pathway k (receiver). Cells are
row vectors, so a column-vector expression B z becomes z @ B.T in code.
Computations run in float64.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LinearPathwaySCM(nn.Module):
    """Shared graph A, base input b and exogenous noise; one input a_t per condition (row 0 is control, a_0 = 0).

    A = tanh(R) * (1 - I) / max(1, ||.||_2 / r): zero diagonal, ||A||_2 <= r < 1,
    positive and negative edges and feedback loops allowed.
    a_t = S_t * a_max tanh(raw_t): only nodes in the candidate support S_t can receive an input.

    How a_t is obtained (``input_mode``):

    - ``free``: a_t = S_t * a_max tanh(rho_t), one free vector rho_t per condition;
    - ``shared``: a_t = S_t * a_max tanh(alpha), one vector alpha shared by all conditions;
    - ``kernel``: a_t = S_t * a_max tanh(alpha + sum_t' W[t, t'] c_t'), where W[t, t'] is the
      Jaccard index of the supports of t and t', normalized over the source conditions t'
      (the training perturbations, t' != t), and c_t' is a free prototype input of the source t'.
      A new perturbation only needs its support S_t: its weights follow from S_t.
    - ``embedding``: a_t = S_t * a_max tanh(MLP(e_t)), one free embedding e_t per condition and one
      shared MLP (embedding -> 64 -> pathways). A new perturbation gets e_t from the training
      embeddings, weighted by the Jaccard index of the supports (``embed_new_perturbations_``).
    - ``sensitivity``: a_t = S_t * w (+ gamma with ``generic_input``), w a free vector shared by all
      conditions (w_k: the direct change of pathway k when any of its members is knocked down), no bound.
    - ``sensitivity_offset``: a_t = S_t * (w + delta_t) (+ gamma); delta_t a free deviation of every condition,
      zero at the start, so it stays zero for conditions never fitted (new genes get S_t * w + gamma).

    ``penalized`` marks the conditions whose inputs enter the L1 penalty (the training ones).

    ``fixed_gauge`` fixes b = 0 and sigma = 1. With a jointly trained encoder and
    decoder the location and scale of every node are free (they move into the
    encoder heads and the first decoder layer), so this choice loses nothing;
    it is used by joint training, where the coordinates cannot be fixed after
    an encoder-only stage.
    """

    def __init__(self, support: torch.Tensor, radius: float = 0.9, amplitude: float = 3.0, sigma_floor: float = 0.05,
                 learn_graph: bool = True, edge_support: torch.Tensor | None = None, init_scale: float = 1e-3,
                 fixed_gauge: bool = False, input_mode: str = "free", sources: torch.Tensor | None = None,
                 penalized: torch.Tensor | None = None, embedding_dim: int = 16, generic_input: bool = False,
                 generic_scale: bool = False, noise_rank: int = 0, embedding_source: str = "free", dense_inputs: bool = False,
                 off_support_weight: float = 1.0, noise_after: bool = False):
        super().__init__()
        if not 0.0 < radius < 1.0:
            raise ValueError("radius must be in (0, 1)")
        support = support.to(torch.float64)
        if torch.any(support[0] != 0):
            raise ValueError("row 0 of the support is control and must be empty")
        n_conditions, n_pathways = support.shape
        self.n_pathways, self.radius, self.amplitude, self.sigma_floor = n_pathways, radius, amplitude, sigma_floor
        self.learn_graph, self.fixed_gauge = learn_graph, fixed_gauge
        # dense_inputs: a perturbation may act on every pathway, not only on the pathways of its target
        self.dense_inputs = dense_inputs
        # in the L1 penalty, inputs outside the target's pathways weigh off_support_weight times more (dense inputs only)
        self.off_support_weight = off_support_weight
        # noise_after: z = T (b + a_t) + eta, eta ~ N(0, Sigma); the graph moves only the means, cell-to-cell
        # variation does not pass through A. Default (False): z = T (b + a_t + eps), covariance T Sigma T^T.
        self.noise_after = noise_after
        self.register_buffer("support", support)
        allowed = 1.0 - torch.eye(n_pathways, dtype=torch.float64)
        if edge_support is not None:
            allowed = allowed * edge_support.to(torch.float64)
        self.register_buffer("edge_mask", allowed)
        # power-iteration vectors for the spectral norm (warm-started across steps; an SVD of a large K is slow on GPUs)
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("power_u", nn.functional.normalize(torch.randn(n_pathways, dtype=torch.float64, generator=generator), dim=0))
        self.register_buffer("power_v", nn.functional.normalize(torch.randn(n_pathways, dtype=torch.float64, generator=generator), dim=0))
        self.raw_graph = nn.Parameter(init_scale * torch.randn(n_pathways, n_pathways, dtype=torch.float64),
                                      requires_grad=learn_graph)
        self.base = nn.Parameter(torch.zeros(n_pathways, dtype=torch.float64), requires_grad=not fixed_gauge)
        if input_mode not in ("free", "shared", "kernel", "embedding", "sensitivity", "sensitivity_offset"):
            raise ValueError(f"unknown input_mode {input_mode}")
        self.input_mode = input_mode
        if input_mode == "free":
            self.raw_strength = nn.Parameter(init_scale * torch.randn(n_conditions - 1, n_pathways, dtype=torch.float64))
        elif input_mode == "embedding":
            # where the embedding e_t of a perturbation comes from:
            #   free:       one free vector per perturbation (unseen genes: Jaccard-weighted mean after training)
            #   kernel:     e_t = sum_{t' != t} W[t, t'] c_t', W the Jaccard kernel over the fitted perturbations
            #   membership: e_t = MLP(S_t), a function of the pathway membership only
            #   hybrid:     e_t = MLP(S_t) + delta_t, delta_t a free offset of every fitted perturbation (zero for unseen genes)
            self.embedding_source = embedding_source
            if embedding_source == "free":
                self.embedding = nn.Parameter(0.1 * torch.randn(n_conditions, embedding_dim, dtype=torch.float64))
            elif embedding_source == "kernel":
                source = torch.ones(n_conditions, dtype=torch.bool) if sources is None else sources.to(torch.bool).clone()
                source[0] = False
                self.register_buffer("embedding_kernel", jaccard_kernel(support, source))
                self.prototype_embedding = nn.Parameter(0.1 * torch.randn(n_conditions, embedding_dim, dtype=torch.float64))
            elif embedding_source in ("membership", "hybrid"):
                if embedding_source == "hybrid":
                    self.embedding_offset = nn.Parameter(torch.zeros(n_conditions, embedding_dim, dtype=torch.float64))
                self.membership_network = nn.Sequential(nn.Linear(n_pathways, 64), nn.SiLU(),
                                                        nn.Linear(64, embedding_dim)).to(torch.float64)
            else:
                raise ValueError(f"unknown embedding_source {embedding_source}")
            # with generic_scale the network has one more output: the strength s_t = 2 sigmoid(.) of the generic response
            outputs = n_pathways + (1 if generic_scale else 0)
            self.input_network = nn.Sequential(nn.Linear(embedding_dim, 64), nn.SiLU(), nn.Linear(64, outputs)).to(torch.float64)
        else:
            self.input_base = nn.Parameter(init_scale * torch.randn(n_pathways, dtype=torch.float64))
        if input_mode == "sensitivity_offset":
            self.input_offset = nn.Parameter(torch.zeros(n_conditions - 1, n_pathways, dtype=torch.float64))
        if input_mode == "kernel":
            source = torch.ones(n_conditions, dtype=torch.bool) if sources is None else sources.to(torch.bool).clone()
            source[0] = False
            self.register_buffer("kernel_weight", jaccard_kernel(support, source))
            self.prototype = nn.Parameter(init_scale * torch.randn(n_conditions, n_pathways, dtype=torch.float64))
        # exogenous noise Sigma = diag(sigma^2) + L L^T: L [K, r] are common factors of the pathways (r = 0: diagonal)
        self.noise_rank = noise_rank
        if noise_rank > 0:
            self.noise_factors = nn.Parameter(0.1 * torch.randn(n_pathways, noise_rank, dtype=torch.float64))
        # generic perturbation response g: added to the input of every perturbation (not control), on every pathway
        self.generic_input, self.generic_scale = generic_input, generic_scale and generic_input and input_mode == "embedding"
        if generic_input:
            self.generic = nn.Parameter(init_scale * torch.randn(n_pathways, dtype=torch.float64))
        mask = torch.ones(n_conditions, dtype=torch.bool) if penalized is None else penalized.to(torch.bool).clone()
        mask[0] = False
        self.register_buffer("penalized", mask)
        self.raw_sigma = nn.Parameter(torch.full((n_pathways,), math.log(math.expm1(1.0 - sigma_floor)), dtype=torch.float64),
                                      requires_grad=not fixed_gauge)

    # parameters -------------------------------------------------------------------------------
    def spectral_norm(self, M: torch.Tensor, iterations: int) -> torch.Tensor:
        """Largest singular value u^T M v, with u, v from warm-started power iteration (no gradient through u, v)."""
        with torch.no_grad():
            u, v, W = self.power_u, self.power_v, M.detach()
            for _ in range(iterations):
                v = nn.functional.normalize(W.T @ u, dim=0)
                u = nn.functional.normalize(W @ v, dim=0)
            self.power_u.copy_(u)
            self.power_v.copy_(v)
        return u @ M @ v

    def graph(self, iterations: int | None = None) -> torch.Tensor:
        """A. With ``iterations`` the spectral norm comes from warm-started power iteration (training steps);
        without, it is exact (computed on the CPU, where an SVD of this size is fast)."""
        if not self.learn_graph:
            return torch.zeros_like(self.edge_mask)
        raw = torch.tanh(self.raw_graph) * self.edge_mask
        if iterations is None:
            norm = torch.linalg.matrix_norm(raw.cpu(), ord=2).to(raw.device)
        else:
            norm = self.spectral_norm(raw, iterations)
        return raw / torch.clamp(norm / self.radius, min=1.0)

    def input_drive(self) -> torch.Tensor:
        """Pre-activation of the inputs, [conditions - 1, pathways] (a_t = S_t * a_max tanh(drive_t))."""
        if self.input_mode == "free":
            return self.raw_strength
        if self.input_mode == "embedding":
            return self.input_network(self.embeddings()[1:])[:, :self.n_pathways]
        drive = self.input_base.expand(self.support.shape[0] - 1, -1)
        if self.input_mode == "kernel":
            drive = drive + (self.kernel_weight @ self.prototype)[1:]
        return drive

    def inputs(self) -> torch.Tensor:
        """[conditions, pathways]; the control row is zero and has no parameter."""
        mask = torch.ones_like(self.support[1:]) if self.dense_inputs else self.support[1:]
        if self.input_mode == "sensitivity":
            strength = mask * self.input_base
        elif self.input_mode == "sensitivity_offset":
            strength = mask * (self.input_base + self.input_offset)
        else:
            strength = mask * self.amplitude * torch.tanh(self.input_drive())
        if self.generic_input:
            strength = strength + self.generic_strength()[:, None] * self.generic
        return torch.cat([torch.zeros_like(strength[:1]), strength])

    def embeddings(self) -> torch.Tensor:
        """e_t of every condition, [conditions, embedding_dim] (row 0, control, is never used)."""
        if self.embedding_source == "free":
            return self.embedding
        if self.embedding_source == "kernel":
            return self.embedding_kernel @ self.prototype_embedding
        e = self.membership_network(self.support)
        return e + self.embedding_offset if self.embedding_source == "hybrid" else e

    @torch.no_grad()
    def embed_new_perturbations_(self, new: torch.Tensor) -> None:
        """e_t for perturbations never fitted: the Jaccard-weighted mean of the fitted embeddings
        (the mean of all fitted embeddings when no fitted perturbation shares a pathway)."""
        fitted = self.penalized.clone()
        weight = jaccard_kernel(self.support, fitted)
        mean = self.embedding[fitted].mean(dim=0)
        for t in torch.nonzero(new.to(torch.bool)).flatten().tolist():
            self.embedding[t] = weight[t] @ self.embedding if weight[t].sum() > 0 else mean

    def generic_strength(self) -> torch.Tensor:
        """s_t of every perturbation, [conditions - 1]; 1 without per-perturbation strengths."""
        if self.generic_scale:
            return 2 * torch.sigmoid(self.input_network(self.embeddings()[1:])[:, self.n_pathways])
        return torch.ones(self.support.shape[0] - 1, dtype=torch.float64, device=self.support.device)

    def input_penalty(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mean L1 norm of the inputs of the fitted perturbations, without the generic response g."""
        if self.generic_input:
            inputs = inputs - torch.cat([torch.zeros_like(self.generic[None]), self.generic_strength()[:, None] * self.generic])
        magnitude = inputs.abs()
        if self.dense_inputs and self.off_support_weight != 1.0:
            on = self.support
            magnitude = magnitude * (on + self.off_support_weight * (1 - on))
        return magnitude[self.penalized].sum(dim=1).mean()

    def sigma(self) -> torch.Tensor:
        if self.fixed_gauge:
            return torch.ones_like(self.raw_sigma)
        return self.sigma_floor + F.softplus(self.raw_sigma)

    @torch.no_grad()
    def flip_signs_(self, sign: torch.Tensor) -> None:
        """Re-express the model after z_k -> s_k z_k (s_k = +1 or -1): A -> D A D, a -> D a, b -> D b, sigma unchanged.

        tanh is odd and the spectral norm is invariant under D, so flipping the raw parameters flips A and a exactly.
        """
        s = sign.to(self.raw_graph)
        self.raw_graph.mul_(s[:, None] * s[None, :])
        self.power_u.mul_(s)                                    # singular vectors of D M D are D u, D v
        self.power_v.mul_(s)
        if self.input_mode == "free":
            self.raw_strength.mul_(s)
        elif self.input_mode == "embedding":
            self.input_network[-1].weight[:self.n_pathways].mul_(s[:, None])     # the strength output is not a node
            self.input_network[-1].bias[:self.n_pathways].mul_(s)
        else:
            self.input_base.mul_(s)
        if self.input_mode == "kernel":
            self.prototype.mul_(s)
        if self.input_mode == "sensitivity_offset":
            self.input_offset.mul_(s)
        self.base.mul_(s)
        if self.noise_rank > 0:
            self.noise_factors.mul_(s[:, None])
        if self.generic_input:
            self.generic.mul_(s)

    def structure(self) -> tuple[torch.Tensor, torch.Tensor]:
        """B = I - A and log det B (det B > 0 under the norm bound; a non-positive sign is an error)."""
        B = torch.eye(self.n_pathways, dtype=torch.float64, device=self.edge_mask.device) - self.graph()
        sign, logdet = torch.linalg.slogdet(B)
        if sign <= 0:
            raise FloatingPointError("det(I - A) is not positive")
        return B, logdet

    def noise_covariance(self) -> torch.Tensor:
        Sigma = torch.diag(self.sigma() ** 2)
        if self.noise_rank > 0:
            Sigma = Sigma + self.noise_factors @ self.noise_factors.T
        return Sigma

    def _kl_from(self, mean, variance, condition, B, logdet, inputs) -> torch.Tensor:
        """KL(N(mu, diag v) || N(T d, T Sigma T^T)) per cell, d = b + a_t, precision P = B^T Sigma^{-1} B:

            KL = 1/2 [ v . diag(P) + r^T Sigma^{-1} r - K + log det Sigma - 2 log det B - sum log v ],  r = B mu - d.
        With diagonal Sigma this is the formula of section 6.2 of the specification.
        """
        if self.noise_after:
            means = torch.linalg.solve(B, (self.base + inputs).T).T                         # T (b + a_t) for every condition
            residual = mean - means[condition]
            chol = torch.linalg.cholesky(self.noise_covariance())
            whitened = torch.linalg.solve_triangular(chol, residual.T, upper=False)
            inverse_diag = (torch.linalg.solve_triangular(chol, torch.eye(self.n_pathways, dtype=chol.dtype, device=chol.device),
                                                          upper=False) ** 2).sum(dim=0)     # diag(Sigma^{-1})
            logdet_Sigma = 2 * torch.log(torch.diagonal(chol)).sum()
            return 0.5 * (variance @ inverse_diag + (whitened ** 2).sum(dim=0) - self.n_pathways + logdet_Sigma
                          - torch.log(variance).sum(dim=-1))
        residual = mean @ B.T - (self.base + inputs[condition])
        if self.noise_rank == 0:
            sigma2 = self.sigma() ** 2
            return 0.5 * ((residual ** 2 + variance @ (B ** 2).T) / sigma2 + torch.log(sigma2) - torch.log(variance) - 1.0).sum(dim=-1) - logdet
        chol = torch.linalg.cholesky(self.noise_covariance())
        whitened = torch.linalg.solve_triangular(chol, residual.T, upper=False)          # L_c^{-1} r
        WB = torch.linalg.solve_triangular(chol, B, upper=False)                          # L_c^{-1} B
        diag_P = (WB ** 2).sum(dim=0)
        logdet_Sigma = 2 * torch.log(torch.diagonal(chol)).sum()
        return 0.5 * (variance @ diag_P + (whitened ** 2).sum(dim=0) - self.n_pathways + logdet_Sigma
                      - torch.log(variance).sum(dim=-1)) - logdet

    # probability model ------------------------------------------------------------------------------
    def kl(self, mean: torch.Tensor, variance: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """KL(N(mean_i, diag(variance_i)) || p_A(z | t_i)) for every cell, analytically:

            r_i = B mu_i - d_t,  w_ik = sum_j B_kj^2 v_ij,
            KL_i = 1/2 sum_k [(r_ik^2 + w_ik) / sigma_k^2 + log sigma_k^2 - log v_ik - 1] - log det B.
        """
        mean, variance = mean.to(torch.float64), variance.to(torch.float64)
        B, logdet = self.structure()
        return self._kl_from(mean, variance, condition, B, logdet, self.inputs())

    def objective(self, mean: torch.Tensor, variance: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Mean KL of a batch, ||A||_1 and mean_t ||a_t||_1, computing A and the inputs once."""
        mean, variance = mean.to(torch.float64), variance.to(torch.float64)
        A = self.graph(iterations=2)
        B = torch.eye(self.n_pathways, dtype=torch.float64, device=A.device) - A
        sign, logdet = torch.linalg.slogdet(B)
        if sign <= 0:
            raise FloatingPointError("det(I - A) is not positive")
        inputs = self.inputs()
        kl = self._kl_from(mean, variance, condition, B, logdet, inputs)
        return kl.mean(), A.abs().sum(), self.input_penalty(inputs)

    def penalties(self) -> tuple[torch.Tensor, torch.Tensor]:
        """||A||_1 and the mean over non-control conditions of ||a_t||_1."""
        return self.graph().abs().sum(), self.input_penalty(self.inputs())

    def moments(self, condition: int | None = None, input_vector: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean T (b + a) and covariance T Sigma T^T of the latent state for a condition or a given input."""
        B, _ = self.structure()
        a = self.inputs()[condition] if input_vector is None else input_vector.to(torch.float64)
        T = torch.linalg.solve(B, torch.eye(self.n_pathways, dtype=torch.float64, device=B.device))
        covariance = self.noise_covariance() if self.noise_after else T @ self.noise_covariance() @ T.T
        return T @ (self.base + a), covariance

    # generation ----------------------------------------------------------------------------------------
    def sample(self, input_vector: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """z = solve(I - A, b + a + eps) for n draws of eps, [n, pathways]."""
        B, _ = self.structure()
        eps = torch.randn(n, self.n_pathways, dtype=torch.float64, device=B.device, generator=generator) * self.sigma()
        if self.noise_rank > 0:
            eps = eps + torch.randn(n, self.noise_rank, dtype=torch.float64, device=B.device, generator=generator) @ self.noise_factors.T
        if self.noise_after:
            return torch.linalg.solve(B, (self.base + input_vector.to(torch.float64)).reshape(-1, 1)).T + eps
        rhs = self.base + input_vector.to(torch.float64) + eps
        return torch.linalg.solve(B, rhs.T).T

    def from_control(self, z0: torch.Tensor, input_vector: torch.Tensor) -> torch.Tensor:
        """Keep the exogenous state eps = (I - A) z0 - b of each cell and add the input: z0 + solve(I - A, a)."""
        B, _ = self.structure()
        return z0.to(torch.float64) + torch.linalg.solve(B, input_vector.to(torch.float64).reshape(-1, 1)).T

    def total_response_operator(self) -> torch.Tensor:
        """T = (I - A)^{-1}; T[k, l] is the total change of k per unit input to l."""
        B, _ = self.structure()
        return torch.linalg.solve(B, torch.eye(self.n_pathways, dtype=torch.float64, device=B.device))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        A = self.graph()
        B, logdet = self.structure()
        singular = torch.linalg.svdvals(B)
        within = self.support[1:] > 0
        strength = torch.tanh(self.input_drive())[within]
        return {
            "A_spectral_norm": float(torch.linalg.matrix_norm(A, ord=2)),
            "A_edges_abs_gt_0.01": int((A.abs() > 0.01).sum()),
            "B_min_singular_value": float(singular.min()), "B_condition_number": float(singular.max() / singular.min()),
            "logdet_B": float(logdet),
            "input_saturated_share": float((strength.abs() > 0.95).double().mean()) if strength.numel() else 0.0,
            "sigma_at_floor_share": float((self.sigma() < self.sigma_floor + 1e-3).double().mean()),
            "sigma_median": float(self.sigma().median()),
        }


def jaccard_kernel(support: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """W[t, t'] = J(S_t, S_t') / sum_{t'' in sources, t'' != t} J(S_t, S_t''), zero for t' outside the sources and for t' = t.

    J is the Jaccard index of the two supports (shared pathways over pathways in either). A row with no
    source sharing a pathway is zero; control (row 0) is zero.
    """
    S = support.to(torch.float64)
    shared = S @ S.T
    size = S.sum(dim=1)
    union = size[:, None] + size[None, :] - shared
    J = torch.where(union > 0, shared / union.clamp_min(1.0), torch.zeros_like(shared))
    J = J * sources.to(torch.float64)[None, :]
    J.fill_diagonal_(0.0)
    J[0] = 0.0
    total = J.sum(dim=1, keepdim=True)
    return torch.where(total > 0, J / total.clamp_min(1e-12), torch.zeros_like(J))


def scm_from_checkpoint(state: dict) -> LinearPathwaySCM:
    """Rebuild a saved SCM (any input mode) and load its parameters."""
    args, params = state["args"], state["scm"]
    mode = state.get("input_mode", "free")
    params = dict(params)
    if "penalized" not in params:                                   # checkpoints written before the input modes
        params["penalized"] = torch.ones(len(state["support"]), dtype=torch.bool)
        params["penalized"][0] = False
    if "power_u" not in params:                                     # checkpoints from before the power iteration
        raw = torch.tanh(params["raw_graph"]) * params["edge_mask"]
        U, _, Vh = torch.linalg.svd(raw)
        params["power_u"], params["power_v"] = U[:, 0].clone(), Vh[0].clone()
    scm = LinearPathwaySCM(torch.as_tensor(state["support"]), args["radius"], args["amplitude"], args["sigma_floor"],
                           state["learn_graph"], fixed_gauge=state.get("fixed_gauge", False), input_mode=mode,
                           generic_input=state.get("generic_input", False), generic_scale=state.get("generic_scale", False),
                           noise_rank=state.get("noise_rank", 0), embedding_source=state.get("embedding_source", "free"),
                           dense_inputs=state.get("dense_inputs", False), off_support_weight=state.get("off_support_weight", 1.0),
                           noise_after=state.get("noise_after", False))
    scm.load_state_dict(params)
    return scm
