"""Stable expert-local row ordinals using block sorts and integer prefixes."""

import torch
import triton
import triton.language as tl

ROW_BLOCK = 256


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _block_ranks(
    ids,
    local_rank,
    hist,
    rows,
    NUM_GROUPS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    row = block * BLOCK + lane
    expert = tl.load(ids + row, row < rows, other=NUM_GROUPS).to(tl.int64)
    # Unique keys preserve the incoming row order within every expert.
    key = tl.where(
        row < rows, expert * (rows + 1) + row, tl.full((), NUM_GROUPS, tl.int64) * (rows + 1) + rows
    )
    key = tl.sort(key, descending=False)
    expert = key // (rows + 1)
    original = key % (rows + 1)
    previous = tl.gather(expert, tl.maximum(lane - 1, 0), axis=0)
    start = tl.where((lane == 0) | (expert != previous), lane, 0)
    start = tl.associative_scan(start, 0, _maximum)
    tl.store(local_rank + original, lane - start, expert < NUM_GROUPS)
    counts = tl.histogram(expert.to(tl.int32), BINS)
    group = tl.arange(0, BINS)
    tl.store(hist + group * NUM_BLOCKS + block, counts.to(tl.int64), group < NUM_GROUPS)


@triton.jit
def _block_prefix(hist, prefix, NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    block = tl.arange(0, BLOCK)
    count = tl.load(hist + expert * NUM_BLOCKS + block, block < NUM_BLOCKS, other=0)
    before = tl.cumsum(count, 0) - count
    tl.store(prefix + expert * NUM_BLOCKS + block, before, block < NUM_BLOCKS)


def canonical_group_positions(expert_ids: torch.Tensor, num_groups: int):
    """Return local ranks and per-block prefixes; no floating-point operations.

    Expert-local ordinal for row i is ranks[i] + prefixes[expert[i], i // ROW_BLOCK].
    Inputs have the same valid expert-ID contract as the routing/scatter operators.
    """
    rows = expert_ids.numel()
    blocks = triton.cdiv(rows, ROW_BLOCK)
    ranks = torch.empty((rows,), device=expert_ids.device, dtype=torch.int64)
    hist = torch.empty((num_groups, blocks), device=expert_ids.device, dtype=torch.int64)
    prefix = torch.empty_like(hist)
    if rows:
        _block_ranks[(blocks,)](
            expert_ids,
            ranks,
            hist,
            rows,
            num_groups,
            blocks,
            triton.next_power_of_2(num_groups + 1),
            ROW_BLOCK,
        )
        _block_prefix[(num_groups,)](
            hist,
            prefix,
            blocks,
            triton.next_power_of_2(blocks),
            num_warps=8 if blocks >= 2048 else 4,
        )
    return ranks, prefix


@triton.jit
def scatter_canonical_order(
    ids, ranks, prefix, starts, order, rows, NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ids + row, row < rows, other=0)
    position = tl.load(ranks + row, row < rows, other=0)
    position += tl.load(prefix + expert * NUM_BLOCKS + row // BLOCK, row < rows, other=0)
    start = tl.load(starts + expert, row < rows, other=0)
    tl.store(order + start + position, row, row < rows)
