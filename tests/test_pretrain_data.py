"""Check the real shuffled pretraining batch path under torchrun, without building a model.

Run after preparing examples/tokenize_corpus/dclm-qwen3-omni-smoke:
    torchrun --standalone --nproc-per-node=1 tests/test_pretrain_data.py
    torchrun --standalone --nproc-per-node=2 tests/test_pretrain_data.py --cp 2
"""

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("workspace/datasets/dclm-omni-smoke"))
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    # Keep pytest collection CPU-safe; this integration check requires a CUDA runtime.
    from pithtrain.contexts import distributed
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.operators.cp_sequence import zigzag_spans
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, get_global_batch, setup_dataset

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset / "tokens"
    for name, value in manifest["pretraining"].items():
        setattr(cfg.training, name, value)
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.timeout = timedelta(seconds=60)
    setup_distributed(cfg)
    t, d = cfg.training, distributed
    assert t.global_batch_size % (t.micro_batch_size * d.dp_size) == 0
    assert t.sequence_length % (2 * d.cp_size) == 0

    # Read the actual on-disk stream independently of MemmapDataset/get_chunk.
    expected_inputs, expected_labels = [], []
    paths = sorted(cfg.dataset.rglob("*.bin"))
    assert [path.name for path in paths] == [shard["file"] for shard in manifest["shards"]]
    for path, shard in zip(paths, manifest["shards"], strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"], str(path)
        with path.open("rb") as stream:
            tokens, ends = np.load(stream), np.load(stream)
        assert tokens.dtype == np.uint32
        assert len(ends) == shard["documents"] and ends[-1] == len(tokens)
        assert np.all(tokens[ends.astype(np.int64) - 1] == manifest["tokenizer"]["eos_token_id"])
        assert tokens.max() < manifest["model_vocab_size"]
        count = shard["samples"] * t.sequence_length
        expected_inputs.append(tokens[:count].astype(np.int64).reshape(-1, t.sequence_length))
        expected_labels.append(
            tokens[1 : count + 1].astype(np.int64).reshape(-1, t.sequence_length)
        )
    inputs = torch.from_numpy(np.concatenate(expected_inputs))
    labels = torch.from_numpy(np.concatenate(expected_labels))

    dataset = setup_dataset(cfg)
    indices = torch.from_numpy(np.array(dataset.indices, dtype=np.int64))
    assert len(dataset) == manifest["samples"] == len(inputs)
    assert torch.equal(indices.sort().values, torch.arange(len(dataset))), "Invalid shuffle"
    digests = [None] * d.world_size
    torch.distributed.all_gather_object(
        digests, hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    )
    assert len(set(digests)) == 1, "Ranks disagree on the shuffled sample order"
    # Reinitializing with the same seed must give the same global sample order.
    torch.distributed.barrier()
    dataset = setup_dataset(cfg)
    assert np.array_equal(dataset.indices, indices.numpy()), "Shuffle is not reproducible"
    front, back = zigzag_spans(d.cp_rank, d.cp_size, t.sequence_length)
    positions = torch.tensor([*front, *back])
    for step in range(t.max_steps):
        batch = get_global_batch(cfg, dataset, step, d.device)
        assert len(batch) == t.global_batch_size // d.dp_size // t.micro_batch_size
        # Global micro-batches alternate between data ranks. EP/PP do not select samples.
        selected = indices[step * t.global_batch_size : (step + 1) * t.global_batch_size]
        selected = selected.reshape(-1, d.dp_size, t.micro_batch_size)[:, d.dp_rank].flatten()
        expected_x = inputs[selected][:, positions]
        expected_y = labels[selected][:, positions]
        for micro in batch:
            (x,), (y,) = micro.model_inputs, micro.objective_inputs
            assert x.shape == y.shape == (t.micro_batch_size, t.sequence_length // d.cp_size)
            assert x.dtype == y.dtype == torch.long and x.device == y.device == d.device
            assert micro.cu_seqlens is None
        actual_x = torch.cat([micro.model_inputs[0] for micro in batch]).cpu()
        actual_y = torch.cat([micro.objective_inputs[0] for micro in batch]).cpu()
        torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
    torch.distributed.barrier()
    if d.rank == 0:
        print(
            json.dumps(
                dict(
                    result="PASSED",
                    pp=d.pp_size,
                    dp=d.dp_size,
                    cp=d.cp_size,
                    ep=d.ep_size,
                    steps=t.max_steps,
                    samples_per_step=t.global_batch_size,
                    sequence_length=t.sequence_length,
                    dataset_samples=len(dataset),
                    vocab_size=manifest["model_vocab_size"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
