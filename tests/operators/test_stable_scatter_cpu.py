"""CPU contracts for the production stable-index helper; no CUDA execution."""

import pytest
import torch

from pithtrain.operators.stable_scatter import stable_group_order


@pytest.mark.parametrize("rows", [0, 1, 64, 129, 513])
@pytest.mark.parametrize("experts", [1, 4, 16])
def test_stable_order_and_padded_positions(rows, experts):
    generator = torch.Generator().manual_seed(rows + experts)
    ids = torch.randint(experts, (rows,), generator=generator)
    counts = torch.bincount(ids, minlength=experts)
    order, starts = stable_group_order(ids, counts)
    expected = sorted(range(rows), key=lambda index: (int(ids[index]), index))
    assert order.tolist() == expected
    assert starts.tolist() == [int(counts[:index].sum()) for index in range(experts)]
    padded = ((counts + 127) // 128) * 128
    padded_starts = padded.cumsum(0) - padded
    positions = torch.arange(rows) - starts[ids[order]] + padded_starts[ids[order]]
    assert positions.unique().numel() == rows
    if rows:
        assert positions.min() >= 0
        assert positions.max() < padded.sum()
    for expert in range(experts):
        selected = order[ids[order] == expert]
        assert selected.tolist() == torch.where(ids == expert)[0].tolist()


def test_fullgraph_cpu_inductor_helper():
    ids = torch.tensor([3, 1, 3, 0, 1, 0, 3], dtype=torch.int64)
    counts = torch.tensor([2, 2, 0, 3], dtype=torch.int64)
    compiled = torch.compile(stable_group_order, backend="inductor", fullgraph=True)
    expected = stable_group_order(ids, counts)
    actual = compiled(ids, counts)
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
