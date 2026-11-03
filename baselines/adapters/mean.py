"""Methods that predict only the mean expression of a perturbation (GEARS; npz files from
baselines/gears.py) as cell generators, so that distribution metrics and UMAPs can
include them: a perturbed population is the training control cells shifted by the predicted mean effect
(predicted mean - control mean, standardized like the data). The control condition returns the control cells.
"""

from __future__ import annotations

import numpy as np
import torch


class MeanShiftGenerator:
    def __init__(self, path: str, data, device: str, controls: torch.Tensor):
        pred = np.load(path)
        effect = (pred["pred_mean_log1p"] - pred["control_mean_log1p"][None]) / data.train_sd[None]
        self.effect = {str(g): torch.as_tensor(e, dtype=torch.float32, device=device) for g, e in zip(pred["genes"], effect)}
        self.conditions, self.control, self.device = list(data.conditions), controls, device

    def condition_input(self, condition: int) -> int:
        return condition

    @torch.no_grad()
    def sample_population(self, condition: int, n_cells: int, seed: int = 0) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        cells = self.control[torch.randint(len(self.control), (n_cells,), device=self.device, generator=generator)]
        if condition > 0:
            cells = cells + self.effect[self.conditions[condition]]
        return {"mean": cells, "sample": cells}
