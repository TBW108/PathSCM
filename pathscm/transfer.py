"""Jaccard transfer of fitted a_t AND c_t, without unseen expression access.

This is an inference adapter, not a way to turn a seen checkpoint into an
unseen experiment. The caller must supply the genes actually used for fitting;
the adapter checks held-out targets against checkpoint metadata.
"""
from pathlib import Path
import numpy as np
import torch
from pathscm.scm import jaccard_kernel
from pathscm.model import ResponseGenerator


def transfer_effects(support, fitted, targets, a, c):
    """Normalized Jaccard on binary pathway memberships, then W @ a and W @ c.

    Only fitted rows are read. No top-k, expression similarity, target remasking,
    sign alignment across seeds, or response-dependent weight tuning is applied.
    Zero-overlap targets receive zero shifts and are explicitly flagged.
    """
    S = torch.as_tensor(support, device=a.device)
    fitted = torch.as_tensor(fitted, dtype=torch.bool, device=a.device).clone()
    targets = torch.as_tensor(targets, dtype=torch.bool, device=a.device)
    n = len(S)
    if S.ndim != 2 or fitted.shape != (n,) or targets.shape != (n,):
        raise ValueError('Invalid support or condition masks')
    if not torch.all((S == 0) | (S == 1)):
        raise ValueError('Jaccard requires binary pathway membership')
    if a.shape != S.shape or c.ndim != 2 or len(c) != n:
        raise ValueError('a/c dimensions must match conditions and pathways')
    if c.device != a.device:
        raise ValueError('a and c must use the same device')
    if fitted[0] or targets[0] or torch.any(fitted & targets):
        raise ValueError('Control cannot be a source/target; fitted and targets must be disjoint')
    if not fitted.any():
        raise ValueError('At least one fitted perturbation is required')
    W = jaccard_kernel(S, fitted)[targets][:, fitted]
    # Slice first: never multiply zero weights by untrained/NaN target rows.
    a_new = W.to(a.dtype) @ a[fitted]
    c_new = W.to(c.dtype) @ c[fitted]
    return a_new, c_new, W, W.sum(1) > 0


class UnseenResponseGenerator(ResponseGenerator):
    """Same trained encoder/A/decoder, with joint a/c transfer for held-out genes."""
    def __init__(self, run_dir, device, node_mask, control_cells, fitted, targets):
        state = torch.load(Path(run_dir)/'response.pth', map_location='cpu', weights_only=False)
        n = len(state['scm_state']['conditions'])
        fitted = np.asarray(fitted, dtype=bool)
        targets = np.asarray(targets, dtype=bool)
        if fitted.shape != (n,) or targets.shape != (n,):
            raise ValueError('Condition order must match the checkpoint')
        held = np.zeros(n, dtype=bool)
        held[state['scm_state'].get('held_out_conditions', [])] = True
        if not targets.any() or np.any(targets & ~held) or np.any(fitted & held):
            raise ValueError('Unseen prediction requires a genuinely gene-held-out checkpoint')
        actual_sources = state['scm_state'].get('fitted_conditions')
        if actual_sources is not None and not np.array_equal(np.flatnonzero(fitted), np.sort(actual_sources)):
            raise ValueError('Source mask must match the perturbations actually fitted in this checkpoint')
        super().__init__(run_dir, device, node_mask, control_cells)
        if not self.uncaptured:
            raise ValueError('This experiment transfers both a_t and c_t; the model must have c_t')
        self.scm.eval()
        with torch.no_grad():
            self.transferred_inputs = self.scm.inputs().detach().clone()
            a_new, c_new, W, covered = transfer_effects(self.scm.support, fitted, targets,
                                                      self.transferred_inputs, self.uncaptured_shift)
            target_mask = torch.as_tensor(targets, device=device)
            self.transferred_inputs[target_mask] = a_new
            self.uncaptured_shift = self.uncaptured_shift.clone()
            self.uncaptured_shift[target_mask] = c_new
        self.transfer_weights, self.transfer_covered = W, covered
        self.transfer_source_indices = np.flatnonzero(fitted)
        self.transfer_target_indices = np.flatnonzero(targets)

    @torch.no_grad()
    def condition_input(self, condition):
        self.current_condition = condition
        return self.transferred_inputs[condition]
