# Omni text data smoke recipe

Prepare a small, repeatable corpus for the first Thinker text training runs. This
uses PithTrain's existing `Worker` / `Writer` and pretraining loader. It does not
construct a model, download model weights, or require native Omni support.

## Prepare on CPU

From the repository root, with its environment activated:

```bash
python examples/tokenize_corpus/dclm-qwen3-omni-smoke/script.py
```

The recipe streams the first 128 documents from one pinned
[DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
file, requesting only its first 4 MiB of compressed bytes. The normalized text
has a fixed SHA-256 in `config.json`; a changed download or cached input fails.
Only the `text` field is retained, in source order. DCLM's dataset card specifies
CC BY 4.0; see the source card for provenance and terms. Generated corpus files
remain under ignored `workspace/`, not in Git.

The Omni tokenizer is pinned independently. Its configuration, vocabulary and
merges are downloaded, but no model weights. Each document gets the tokenizer's
`<|im_end|>` (151645), following the existing `Worker.encode` behavior. This is
an explicit smoke-test delimiter choice, not a decision about the final
pretraining corpus. No chat template, assistant-only mask or document attention
mask is applied. Samples can cross document boundaries.

Outputs under `workspace/datasets/dclm-omni-smoke/`:

- `raw/train.jsonl`: 128 original document texts (about 575 KiB).
- `tokens/train-000.bin`, `tokens/train-001.bin`: 64 documents per shard,
  stored as `uint32` token IDs followed by document-end offsets.
- `manifest.json`: source/tokenizer revisions, hashes, package versions,
  vocabulary bound, token/sample counts, and the data fields for `PretrainLMCfg`.

Use `--output PATH` to prepare elsewhere. Reruns verify the cached text and
regenerate both shards in deterministic document order. Once the tokenizer is
cached, preparation also runs with `HF_HUB_OFFLINE=1`.

## Read actual training batches on GPUs

Run in an allocated GPU environment with the full PithTrain runtime:

```bash
torchrun --standalone --nproc-per-node=1 tests/test_pretrain_data.py
torchrun --standalone --nproc-per-node=2 tests/test_pretrain_data.py --ep 2
torchrun --standalone --nproc-per-node=2 tests/test_pretrain_data.py --cp 2
torchrun --standalone --nproc-per-node=4 tests/test_pretrain_data.py --cp 2 --ep 2
torchrun --standalone --nproc-per-node=4 tests/test_pretrain_data.py --pp 2 --cp 2 --ep 2
```

For a different output directory, pass `--dataset PATH`. On Slurm, dispatch these
commands with `srun -W 0` inside your allocation. Run one check at a time because
the existing loader writes shuffle metadata alongside the shards.

The check calls the real `setup_distributed`, `setup_dataset` (including GPU
shuffle), and `get_global_batch`. It verifies source hashes, EOS and token bounds,
repeatable shuffle shared by all ranks, DP sample assignment, CP slices, PP/EP sample reuse, microbatch
shape/dtype/device, and exact next-token labels against the saved token streams.
It prints `PASSED` only after every rank has completed four steps. This executes
data loading only; it does not run DualPipeV, model forward/backward or loss.

## Reuse for the first model training run

`config.json` fixes sequence length 128, microbatch size 1, global batch size 8,
four steps and seed 1234. The manifest records these same fields under
`pretraining`. Set `PretrainLMCfg.dataset` to the output's `tokens/` directory and
copy those fields into `cfg.training`. The preparation script requires at least
`max_steps * global_batch_size` complete samples, as `setup_dataset` does.

The native model implementation and model/optimizer/checkpoint configuration
still belong to the model integration task. A tiny model consuming these real
tokens must keep the full **152064-entry vocabulary** while shrinking layers and
hidden dimensions. The existing 256-entry HF comparison fixture uses synthetic IDs, so it
can keep using the existing synthetic data for basic model correctness and
training smoke. It is not required to consume this real-text corpus. To test
the real Omni tokenizer-to-model path, use a small architecture with the
152064-entry vocabulary; preserve the tokenizer's IDs unchanged.

For image/text, audio/transcript and video/text inputs, use the
[multimodal data recipe](../../prepare_omni_data/qwen3-omni-smoke/README.md).
That reader keeps media, grids, lengths and timing attached to each sample;
this token-only recipe does not provide multimodal inputs.
