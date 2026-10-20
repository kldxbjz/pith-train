"""Stable grouped-row indices, independent of the CUDA scatter implementation."""

import torch


def stable_group_order(
    expert_idxs: torch.Tensor, actual_counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original rows sorted stably by expert and unpadded group starts.

    ``actual_counts`` is the integer histogram of ``expert_idxs``. Both inputs
    stay on their current device. For sorted position ``p`` with expert ``e``,
    ``p - starts[e]`` is its unique, stable expert-local ordinal.

    Stability is relative to this call's incoming row order. It cannot repair
    nondeterministic token ordering introduced by an earlier EP dispatch.
    """
    order = torch.argsort(expert_idxs, stable=True)
    starts = actual_counts.cumsum(dim=0) - actual_counts
    return order, starts
