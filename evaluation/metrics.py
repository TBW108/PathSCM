"""Distribution metrics shared by the current model and benchmarks."""

import torch


def mmd(x: torch.Tensor, y: torch.Tensor, scales: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased squared MMD between point clouds x [..., n, d] and y [..., m, d], Gaussian kernels.

    The kernel is a sum of Gaussians whose bandwidths are ``scales`` times the
    median squared distance between x and y (no gradient through the bandwidth).
    """
    between, within_x, within_y = torch.cdist(x, y).square(), torch.cdist(x, x).square(), torch.cdist(y, y).square()
    bandwidth = between.detach().flatten(-2).median(dim=-1).values[..., None, None].clamp_min(1e-6)

    def kernel(distance: torch.Tensor) -> torch.Tensor:
        return sum(torch.exp(-distance / (scale * bandwidth)) for scale in scales)

    return kernel(within_x).mean(dim=(-2, -1)) + kernel(within_y).mean(dim=(-2, -1)) - 2 * kernel(between).mean(dim=(-2, -1))
