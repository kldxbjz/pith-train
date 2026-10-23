# Qwen3-Omni native model and training acceptance

This branch consolidates the native model/training work on the reviewed prepared-data
and text-model foundations (#100 and #101). RL/GRPO and rollout orchestration are
tracked separately in the PithRL migration. The target is Qwen3-Omni, with Transformers
5.17.0 and released model
configuration revision `26291f793822fb6be9555850f06dfe95f2d7e695`.

## Implementation

- Native Thinker text/MoE/mRoPE, vision/DeepStack, audio, video and aligned media inputs.
- Talker and its residual-code predictor, differentiable Code2Wav, bounded speech generation.
- SFT and speech losses/readers, model-only initialization, full checkpoint conversion,
  optimizer/data/RNG recovery and checkpoint-to-independent-generation loading.
- Normal dispatch is fused; stable token layout and deterministic attention are explicit
  options through `TrainingCfg.deterministic` / `strict_resume`. The layout is captured
  when the model is constructed and bound into its checkpoint numerical contract.
- The text decoder retains #101's compiled stage1 normalization and stage5. Media decoder
  stage1, and audio stage5, retain the eager boundaries used by their existing numerical
  evidence. Image stage5 retains its fullgraph DeepStack wrapper. Routing stays outside
  the text stage1 compile boundary; CE, FP32 master gradients and expert-order BF16
  accumulation fixes are preserved in both dispatch modes.

## Evidence and remaining gates

Historical results below stay bound to their original source. They are not claims that
the assembled branch has been rerun on CUDA. Assembly CPU receipts and exact source/file
equivalence are recorded with the new PR; future runs must bind the new commit.

| Capability | Existing evidence | Current limitation |
|---|---|---|
| Data/reader and text foundations | #100/#101 core validation and supported topology gates | New assembly inheritance is checked by source mapping, not a new full GPU run. |
| Released payload conversion | Run49: six CPU gates, 28,010 keys and 15 shards, bitwise round-trip | This is conversion, not full-size training. |
| Released Thinker inference | Run97: four comparisons, 48 layers / 235 tokens, NSE 0 | No released-model optimizer/update acceptance. |
| Released Talker and Code2Wav | Run77 component outputs, NSE 0 | Does not certify the combined speech training path. |
| Sharded waveform training | Run98: 108 parameters, four updates, fresh-process recovery | Only the recorded tiny fixture and supported DP path. |
| Checkpoint-to-generation | Run98: 351 checkpoint parameters / 702 slices; four text tokens, eight codec frames, 16 codebooks, 122 waveform values | Bounded fixture; not full speech EOS/quality coverage. |
| Talker pipeline optimizer parity (G2) | Run132: all 87 pre-clip and post-clip gradients passed; 85/87 updates passed, **two updates failed** | Three-step continuation and fresh-process restore remain blocked at step 0. |
| Combined speech training parity (G3) | Run106: 243 parameters/rank; **two gradient and 11 update failures** | Second scan and fresh-process continuation remain blocked; original thresholds unchanged. |
| Released full training/recovery (G1) | **Not run** | Candidate admission blocked by an OS-denied local install-log inspection; no instance created. |

Run100 is source `67ccad1e40d73c90ccb014d7c77744720e9325e8`. Run106 is source `e3ff7d01f3ea5aa9086902ce5099790a94e70e2e`; the 52 speech dependency files were
byte-equal to `1c80c59377043d1274861e9cf8ab1e174666c63f`. Source equality preserves
the failures as well as the passes. Existing per-parameter reports remain authoritative.

Run132 used source `0f1f6796730cf70143877695b3f12634624567fd`, the tiny first-code-only
Talker trunk/head, BF16 compute with FP32 master/gradient/Adam state, and PP2/EP2/CP1
on four H100 GPUs. The remaining input-norm update NSE values are 0.0151594 and
0.0112954 against the unchanged 0.01 threshold. All original per-parameter reports
are identical to the uninstrumented Run129. Complete operand captures reproduced
the local norm VJPs, FP32 accumulation and Adam deltas exactly; the incoming
cotangents already differ. This diagnoses the failing comparison, not a pass.

The later main integration retains both media context and upstream inference-record
cleanup. Its 20 affected CPU checks and all eight pre-commit checks passed. It has
not been rerun on CUDA; the Run132 result remains bound to the earlier source.

## Supported boundaries

Native media uses BF16 activations and FP32 master/gradient/Adam state, CP1 and explicit
unpacked/unpadded media records. Only documented/tested PP/EP combinations are claimed;
the full composite speech training path does not support arbitrary CP/EP/PP layouts.
Speech training checkpoints can initialize a separate unsharded generation model.
Native FSDP generation and in-flight KV-cache/stream-state checkpoint restoration are
not implemented. Released 30B end-to-end training has not passed. Caller-provided speech
codes/data are required; this PR does not add an RVQ tokenizer or a reward model.

Do not turn failed normalized-error checks into passes by changing denominators,
freezing parameters, forcing routing, or citing a small aggregate loss difference.
The original fast-path strict noise ratios are retained as reproducibility diagnostics;
ordinary numerical correctness and strict repeatability remain distinct contracts.
