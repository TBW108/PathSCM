"""Implementation tests of the generative pathway SCM (specification, section 10).

They check formulas and generation semantics, not biological validity.
"""

import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import ConditionBalancedSampler, assert_training_only  # noqa: E402
from pathscm.scm import LinearPathwaySCM  # noqa: E402
from pathscm.components import CalibratedDecoder, ExpressionDecoder, LatentCoordinates, PathwayEncoder  # noqa: E402


def random_scm(K: int = 5, C: int = 4, seed: int = 0, learn_graph: bool = True) -> LinearPathwaySCM:
    torch.manual_seed(seed)
    support = (torch.rand(C, K) < 0.5).double()
    support[0] = 0
    scm = LinearPathwaySCM(support, learn_graph=learn_graph, init_scale=1.0)
    with torch.no_grad():
        scm.base.normal_()
        scm.raw_sigma.normal_()
    return scm


def gaussian_logpdf(z, mean, cov):
    diff = z - mean
    return -0.5 * (diff @ torch.linalg.solve(cov, diff) + torch.logdet(cov) + len(z) * math.log(2 * math.pi))


def test_without_graph_the_state_is_the_input_and_the_kl_is_the_diagonal_one():
    scm = random_scm(learn_graph=False)
    a = scm.inputs()[2]
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    z = scm.sample(a, 6, g1)
    eps = torch.randn(6, scm.n_pathways, dtype=torch.float64, generator=g2) * scm.sigma()
    assert torch.allclose(z, scm.base + a + eps)
    mean, variance = torch.randn(3, scm.n_pathways, dtype=torch.float64), torch.rand(3, scm.n_pathways, dtype=torch.float64) + 0.1
    target, s2 = scm.base + a, scm.sigma() ** 2
    expected = 0.5 * ((mean - target) ** 2 / s2 + variance / s2 + torch.log(s2) - torch.log(variance) - 1).sum(dim=1)
    assert torch.allclose(scm.kl(mean, variance, torch.full((3,), 2)), expected)


def test_two_node_direction():
    scm = LinearPathwaySCM(torch.zeros(1, 2, dtype=torch.float64), init_scale=0.0)
    w, d = 0.5, 1.3
    with torch.no_grad():
        scm.raw_graph.zero_()
        scm.raw_graph[1, 0] = math.atanh(w)                  # A[1, 0]: pathway 0 acts on pathway 1
    change = scm.from_control(torch.zeros(1, 2, dtype=torch.float64), torch.tensor([d, 0.0]))
    assert torch.allclose(change, torch.tensor([[d, w * d]], dtype=torch.float64))


def test_zero_input_keeps_the_cell():
    scm = random_scm()
    z0 = torch.randn(4, scm.n_pathways, dtype=torch.float64)
    assert torch.allclose(scm.from_control(z0, torch.zeros(scm.n_pathways)), z0)


def test_analytic_kl_equals_the_full_gaussian_kl():
    for seed in range(4):
        scm = random_scm(seed=seed)
        assert torch.linalg.matrix_norm(scm.graph(), ord=2) <= 0.9 + 1e-9
        for t in range(scm.support.shape[0]):
            m, C = scm.moments(t)
            mean = torch.randn(scm.n_pathways, dtype=torch.float64)
            variance = torch.rand(scm.n_pathways, dtype=torch.float64) + 0.05
            Ci = torch.linalg.inv(C)
            full = 0.5 * (torch.trace(Ci @ torch.diag(variance)) + (m - mean) @ Ci @ (m - mean)
                          - scm.n_pathways + torch.logdet(C) - torch.log(variance).sum())
            assert torch.allclose(scm.kl(mean[None], variance[None], torch.tensor([t]))[0], full, atol=1e-8)


def test_residual_density_plus_logdet_is_the_gaussian_density():
    scm = random_scm(seed=3)
    B, logdet = scm.structure()
    for t in range(scm.support.shape[0]):
        m, C = scm.moments(t)
        z = torch.randn(scm.n_pathways, dtype=torch.float64)
        residual = B @ z - (scm.base + scm.inputs()[t])
        s = scm.sigma()
        residual_density = (-0.5 * (residual / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum() + logdet
        assert torch.allclose(residual_density, gaussian_logpdf(z, m, C), atol=1e-8)


def test_generated_moments_match_the_model():
    scm = random_scm(seed=5)
    m, C = scm.moments(1)
    z = scm.sample(scm.inputs()[1], 400_000, torch.Generator().manual_seed(0))
    assert torch.allclose(z.mean(dim=0), m, atol=0.02 * float(C.diag().max().sqrt()) + 0.01)
    assert torch.allclose(torch.cov(z.T), C, atol=0.03 * float(C.abs().max()))


def test_fixed_coordinates_leave_the_decoder_output_unchanged():
    torch.manual_seed(0)
    mask = (torch.rand(12, 3) < 0.5).float()
    mask[:3, :] = 1.0
    decoder = ExpressionDecoder(3, 12, hidden=8)
    coordinates = LatentCoordinates(3)
    mean, variance = torch.randn(50, 3) * 2 + 1, torch.rand(50, 3) + 0.1
    coordinates.fit(mean, variance, member_score=-mean + 0.1 * torch.randn(50, 3))      # every state points the other way
    assert torch.all(coordinates.sign == -1)
    mean_bar, variance_bar = coordinates.to_fixed(mean, variance)
    assert torch.allclose(CalibratedDecoder(decoder, coordinates)(mean_bar), decoder(mean), atol=1e-5)
    assert torch.allclose(variance_bar, variance / coordinates.scale ** 2)
    assert torch.allclose(mean_bar.mean(dim=0), torch.zeros(3), atol=1e-5)
    encoder = PathwayEncoder(mask, hidden=4, embedding=3)
    x = torch.randn(5, 12)
    changed = x.clone()
    changed[:, ~mask[:, 0].bool()] += 3.0                     # genes outside pathway 0
    assert torch.allclose(encoder(x)[0][:, 0], encoder(changed)[0][:, 0])


def test_gradients_are_finite_and_match_finite_differences():
    scm = random_scm(seed=7)
    mean, variance = torch.randn(6, scm.n_pathways, dtype=torch.float64), torch.rand(6, scm.n_pathways, dtype=torch.float64) + 0.1
    condition = torch.tensor([0, 1, 2, 3, 1, 2])
    loss = lambda: scm.kl(mean, variance, condition).mean() + 0.01 * sum(scm.penalties())
    loss().backward()
    for parameter in (scm.raw_graph, scm.raw_strength, scm.raw_sigma, scm.base):
        assert torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        for parameter, index in ((scm.raw_graph, (1, 0)), (scm.raw_sigma, (2,)), (scm.base, (0,))):
            h = 1e-6
            parameter[index] += h
            up = loss()
            parameter[index] -= 2 * h
            down = loss()
            parameter[index] += h
            assert abs(float((up - down) / (2 * h)) - float(parameter.grad[index])) < 1e-5


def test_training_cells_only():
    split = np.array(["train", "val", "train", "test", "train"])
    assert_training_only(split, np.array([0, 2, 4]), "VAE")
    try:
        assert_training_only(split, np.array([0, 1]), "SCM")
        raise AssertionError("a validation cell was accepted")
    except ValueError:
        pass
    condition = torch.tensor([0, 1, 1, 2, 2])
    rows = torch.as_tensor(np.flatnonzero(split == "train"))
    drawn = ConditionBalancedSampler(condition, rows, torch.Generator().manual_seed(0))(1000)
    assert set(drawn.tolist()) <= set(rows.tolist())
    counts = torch.bincount(condition[drawn], minlength=3).double()
    assert torch.all((counts / counts.sum() - 1 / 3).abs() < 0.06)       # conditions, not cells, are equally likely


def test_fixed_gauge_prior_reduces_to_the_stage_one_prior():
    """b = 0 and sigma = 1 are fixed and untrained; with A = 0 and a = 0 the KL is the KL to N(0, I)."""
    support = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.float64)
    scm = LinearPathwaySCM(support, learn_graph=False, fixed_gauge=True, init_scale=0.0)
    assert not scm.base.requires_grad and not scm.raw_sigma.requires_grad
    assert torch.all(scm.sigma() == 1) and torch.all(scm.base == 0)
    mean, variance = torch.randn(4, 3, dtype=torch.float64), torch.rand(4, 3, dtype=torch.float64) + 0.1
    expected = 0.5 * (mean ** 2 + variance - torch.log(variance) - 1).sum(dim=1)       # standard_normal_kl
    assert torch.allclose(scm.kl(mean, variance, torch.tensor([0, 1, 0, 1])), expected)


def test_flipping_node_signs_leaves_the_kl_unchanged():
    for fixed_gauge in (False, True):
        scm = random_scm(seed=11)
        scm.fixed_gauge = fixed_gauge
        mean, variance = torch.randn(7, scm.n_pathways, dtype=torch.float64), torch.rand(7, scm.n_pathways, dtype=torch.float64) + 0.1
        condition = torch.tensor([0, 1, 2, 3, 1, 2, 0])
        before = scm.kl(mean, variance, condition)
        A = scm.graph().clone()
        sign = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
        scm.flip_signs_(sign)
        assert torch.allclose(scm.graph(), sign[:, None] * A * sign[None, :])
        assert torch.allclose(scm.kl(sign * mean, variance, condition), before)


def test_jaccard_kernel_inputs_depend_only_on_the_support():
    from pathscm.scm import jaccard_kernel
    support = torch.tensor([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=torch.float64)
    sources = torch.tensor([False, True, True, True, False])          # condition 4 is held out
    W = jaccard_kernel(support, sources)
    assert torch.all(W[0] == 0) and torch.all(W[:, 4] == 0) and torch.all(W.diagonal() == 0)
    assert torch.allclose(W[1], torch.tensor([0, 0, 0.5 / (0.5 + 1 / 3), (1 / 3) / (0.5 + 1 / 3), 0], dtype=torch.float64))
    scm = LinearPathwaySCM(support, input_mode="kernel", sources=sources, penalized=sources, init_scale=1.0)
    a = scm.inputs()
    total = 1 + 0.5 + 1 / 3                                              # a held-out perturbation uses every source
    assert torch.allclose(W[4], torch.tensor([0, 1 / total, 0.5 / total, (1 / 3) / total, 0], dtype=torch.float64))
    assert torch.all(a[0] == 0)
    assert torch.all(a[2][1:] == 0)                                       # only the support receives an input
    for mode in ("shared", "kernel", "embedding"):
        scm = LinearPathwaySCM(support, input_mode=mode, sources=sources, init_scale=1.0)
        mean, variance = torch.randn(5, 3, dtype=torch.float64), torch.rand(5, 3, dtype=torch.float64) + 0.1
        condition = torch.arange(5)
        before, sign = scm.kl(mean, variance, condition), torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64)
        scm.flip_signs_(sign)
        assert torch.allclose(scm.kl(sign * mean, variance, condition), before)


def test_unseen_perturbations_take_the_jaccard_weighted_embedding():
    support = torch.tensor([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=torch.float64)
    fitted = torch.tensor([False, True, True, True, False])
    scm = LinearPathwaySCM(support, input_mode="embedding", penalized=fitted, embedding_dim=4)
    scm.embed_new_perturbations_(~fitted & (torch.arange(5) > 0))
    total = 1 + 0.5 + 1 / 3
    expected = (scm.embedding[1] + 0.5 * scm.embedding[2] + scm.embedding[3] / 3) / total
    assert torch.allclose(scm.embedding[4], expected)
    assert torch.all(scm.inputs()[4][2] == 0) and torch.all(scm.inputs().abs() < scm.amplitude)


def test_low_rank_noise_kl_equals_the_full_gaussian_kl():
    torch.manual_seed(3)
    support = (torch.rand(4, 5) < 0.5).double(); support[0] = 0
    scm = LinearPathwaySCM(support, init_scale=1.0, noise_rank=2)
    with torch.no_grad():
        scm.base.normal_(); scm.raw_sigma.normal_()
    for t in range(4):
        m, C = scm.moments(t)
        mean, variance = torch.randn(5, dtype=torch.float64), torch.rand(5, dtype=torch.float64) + 0.05
        Ci = torch.linalg.inv(C)
        full = 0.5 * (torch.trace(Ci @ torch.diag(variance)) + (m - mean) @ Ci @ (m - mean) - 5 + torch.logdet(C) - torch.log(variance).sum())
        assert torch.allclose(scm.kl(mean[None], variance[None], torch.tensor([t]))[0], full, atol=1e-8)
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    mean, variance, condition = torch.randn(4, 5, dtype=torch.float64), torch.rand(4, 5, dtype=torch.float64) + 0.1, torch.arange(4)
    before = scm.kl(mean, variance, condition); scm.flip_signs_(sign)
    assert torch.allclose(scm.kl(sign * mean, variance, condition), before)


def test_noise_after_propagation_kl_equals_the_full_gaussian_kl():
    torch.manual_seed(4)
    support = (torch.rand(4, 5) < 0.5).double(); support[0] = 0
    for rank in (0, 2):
        scm = LinearPathwaySCM(support, init_scale=1.0, noise_rank=rank, noise_after=True)
        with torch.no_grad():
            scm.base.normal_(); scm.raw_sigma.normal_()
        for t in range(4):
            m, C = scm.moments(t)
            assert torch.allclose(C, scm.noise_covariance())                      # the graph does not shape the covariance
            mean, variance = torch.randn(5, dtype=torch.float64), torch.rand(5, dtype=torch.float64) + 0.05
            Ci = torch.linalg.inv(C)
            full = 0.5 * (torch.trace(Ci @ torch.diag(variance)) + (m - mean) @ Ci @ (m - mean) - 5 + torch.logdet(C) - torch.log(variance).sum())
            assert torch.allclose(scm.kl(mean[None], variance[None], torch.tensor([t]))[0], full, atol=1e-8)
        sign = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
        mean, variance, condition = torch.randn(4, 5, dtype=torch.float64), torch.rand(4, 5, dtype=torch.float64) + 0.1, torch.arange(4)
        before = scm.kl(mean, variance, condition); scm.flip_signs_(sign)
        assert torch.allclose(scm.kl(sign * mean, variance, condition), before)


def test_pathway_scores_are_fixed_member_means_in_control_units():
    from pathscm.model import PathwayScores
    torch.manual_seed(5)
    mask = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float32)          # 4 genes, 2 pathways
    controls = torch.randn(500, 4) * torch.tensor([1.0, 2.0, 0.5, 3.0]) + torch.tensor([0.0, 1.0, -1.0, 2.0])
    scores = PathwayScores(mask).fit_(controls)
    assert not list(scores.parameters())                                                   # nothing to train
    z = scores(controls)
    assert torch.allclose(z.mean(0), torch.zeros(2), atol=1e-5) and torch.allclose(z.std(0), torch.ones(2), atol=1e-4)
    x = torch.randn(3, 4)
    standardized = (x - controls.mean(0)) / controls.std(0)
    by_hand = torch.stack([standardized[:, :2].mean(1), standardized[:, 1:3].mean(1)], dim=1) / scores.score_scale
    assert torch.allclose(scores(x), by_hand, atol=1e-5)
    x[:, 3] += 10.0                                                                        # a gene in no pathway
    assert torch.allclose(scores(x), by_hand, atol=1e-5)


def test_pathway_loading_loss_equals_the_direct_sum():
    from pathscm.model import PathwayLoadings
    torch.manual_seed(6)
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)   # 5 genes, 3 pathways
    loadings = PathwayLoadings(mask)
    with torch.no_grad():
        loadings.loading.normal_(); loadings.offset.normal_()
    x, z = torch.randn(7, 5), torch.randn(7, 3)
    U, C = loadings.loading * mask, loadings.offset * mask
    direct = (mask * (x[:, :, None] - U[None] * z[:, None, :] - C[None]) ** 2).sum((1, 2)).mean() / mask.sum()
    assert torch.allclose(loadings.loss(x, z), direct, atol=1e-5)


def test_sensitivity_inputs_are_shared_and_on_the_support():
    support = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float64)
    scm = LinearPathwaySCM(support, input_mode="sensitivity", generic_input=True)
    with torch.no_grad():
        scm.input_base.copy_(torch.tensor([0.5, -2.0, 4.0], dtype=torch.float64))
        scm.generic.copy_(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
    expected = support * scm.input_base + torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected[0] = 0
    assert torch.allclose(scm.inputs(), expected)                                   # a_t = S_t * w + gamma, unbounded


def test_scm_state_likelihood_is_the_gaussian_density_of_the_scm():
    from pathscm.model import scm_state_nll
    torch.manual_seed(7)
    K, n = 4, 6
    A = 0.3 * torch.randn(K, K, dtype=torch.float64); A.fill_diagonal_(0)
    base, sigma = torch.randn(K, dtype=torch.float64), torch.rand(K, dtype=torch.float64) + 0.5
    z, a = torch.randn(n, K, dtype=torch.float64), torch.randn(n, K, dtype=torch.float64)
    T = torch.linalg.inv(torch.eye(K, dtype=torch.float64) - A)
    for factors in (None, 0.3 * torch.randn(K, 2, dtype=torch.float64)):
        Sigma = torch.diag(sigma ** 2) + (factors @ factors.T if factors is not None else 0)
        dist = [torch.distributions.MultivariateNormal(T @ (base + a[i]), T @ Sigma @ T.T) for i in range(n)]
        expected = -torch.stack([d.log_prob(z[i]) for i, d in enumerate(dist)]).mean() - 0.5 * K * math.log(2 * math.pi)
        assert torch.allclose(scm_state_nll(z, a, A, base, sigma, factors), expected, atol=1e-8)


def test_joint_loadings_explain_a_gene_by_all_its_pathways():
    from pathscm.model import JointLoadings
    mask = torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.float32)                # gene 0 in both, gene 2 in none
    loadings = JointLoadings(mask)
    with torch.no_grad():
        loadings.loading.copy_(torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]))
        loadings.offset.copy_(torch.tensor([0.5, 0.0, 1.0]))
    z = torch.tensor([[1.0, -1.0]])
    expected = torch.tensor([[2.0 - 3.0 + 0.5, 5.0, 1.0]])                             # masked entries do not count
    x = torch.zeros(1, 3)
    assert torch.allclose(loadings.loss(x, z), (expected ** 2).mean())
    assert torch.equal(loadings.orientation(), torch.tensor([1.0, 1.0]))
