"""Independent stable-sort oracles for the fused row-placement candidate."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required")


@pytest.mark.parametrize("rows,groups", [(0, 4), (1, 1), (255, 4), (257, 8), (1025, 4), (4099, 16)])
def test_canonical_positions_match_independent_stable_sort(rows, groups):
    from pithtrain.operators.canonical_group_positions import ROW_BLOCK, canonical_group_positions

    generator = torch.Generator(device="cuda").manual_seed(1731)
    ids = torch.randint(max(1, groups - 1), (rows,), generator=generator, device="cuda")
    ranks, prefixes = canonical_group_positions(ids, groups)
    if not rows:
        assert ranks.numel() == prefixes.numel() == 0
        return
    position = ranks + prefixes[ids, torch.arange(rows, device="cuda") // ROW_BLOCK]
    order = torch.argsort(ids, stable=True)
    counts = torch.bincount(ids, minlength=groups)
    expected = torch.empty_like(ids)
    expected[order] = torch.arange(rows, device="cuda") - (counts.cumsum(0) - counts)[ids[order]]
    assert torch.equal(position, expected)


@pytest.mark.parametrize("rows,groups", [(1, 4), (257, 8), (1025, 4)])
def test_default_scatter_matches_stable_values_padding_and_backward(rows, groups):
    from pithtrain.operators.token_scatter import scatter_for_grouped_gemm

    generator = torch.Generator(device="cuda").manual_seed(37)
    ids = torch.randint(groups - 1, (rows,), device="cuda", generator=generator)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    x.requires_grad_()
    expected = scatter_for_grouped_gemm(x, ids, groups, deterministic=True)
    actual = scatter_for_grouped_gemm(x, ids, groups, deterministic=False)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    dy = torch.randn_like(actual[0])
    actual[0].backward(dy)
    assert torch.equal(x.grad, dy[actual[1]])


@pytest.mark.parametrize("rows,topk,experts", [(1, 2, 4), (257, 4, 8), (1025, 2, 4)])
def test_independent_dispatch_preserves_routes_and_original_slot_order(rows, topk, experts):
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch
    from pithtrain.operators.stable_scatter import stable_dedup_prepare_dispatch

    generator = torch.Generator(device="cuda").manual_seed(1731)
    ids = torch.randint(experts, (rows, topk), device="cuda", generator=generator)
    reference = stable_dedup_prepare_dispatch(ids, experts, 2, experts // 2)
    first = None
    for _ in range(4):
        actual = fused_dedup_prepare_dispatch(ids, experts, 2, experts // 2, independent_slots=True)
        for field in (0, 1, 3, 5):
            assert torch.equal(actual[field], reference[field])
        assert torch.equal(actual[3], torch.argsort(ids.reshape(-1), stable=True))
        assert actual[2].numel() == actual[4].numel() == 0
        if first is not None:
            assert torch.equal(actual[3], first)
        first = actual[3]
