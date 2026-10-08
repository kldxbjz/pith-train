# Native Omni 32-step noise regression

This adds the missing regression entrance. It does not modify the frozen d237e4c
package or job 155302, and CPU evidence is not a GPU numerical pass.

The acceptance keeps the existing tiny native Omni Thinker text model (four
layers, four experts, vocabulary 152064), the same verified real PR100 prepared
text bundle, sequence length 32, global batch 8 and microbatch 1. It uses native
DualPipeV/FSDP, FA4 and AdamW, BF16, real routing, global-batch load balancing at
1e-3, seed 1234, and linear warmup from 1e-6 to 1e-5 over all 32 updates. There
is no forced-balanced routing, SDPA replacement, synthetic training data, loader
rewrite, checkpoint resume, full-30B numerical claim or audio/video execution.

For each WORLD/PP/EP profile 1/1/1, 2/1/1, 2/2/1, 2/1/2 and 4/2/2, execute
base0, base1 and feat0 **sequentially in fresh torchrun processes**. The two base
arms use the same explicit clean baseline checkout; feat0 uses the explicit
clean candidate checkout. A shared driver and model/data/settings apply to all
arms. No branch switching is performed. Baseline and candidate must have
different commits and production-file fingerprints; self-comparison cannot be
silently labeled a feature regression.

## Baseline meaning

`--base-repo` and `--feature-repo` are mandatory. The CPU compatibility check
uses a0b14f7 (the existing PR100/Omni converter merge, before distributed strict
resume) as the proposed baseline and d237e4c as candidate. This compares the
normal native training path across that integration range, with checkpoint
saving disabled as in the project's correctness template. It does not test
strict restore or replace the separate checkpoint acceptance. A different
baseline changes the scope and needs a fresh CPU plan. The final GPU baseline
must be explicitly selected before execution; no PR101-only comparison is
claimed by the proposed integrated baseline.

## Gates

Each arm must contain exactly 32 ordered, unique completed-step log entries
with finite cross-entropy, load-balance loss, gradient norm and the exact warmup
schedule. Every rank must attest 32 native optimizer/scheduler updates and 256
consumed samples. Missing/extra ranks, stale commits, model/data/config changes,
different starting weights, different real batches or disagreement between
live metrics and retained logs reject the experiment before noise comparison.

The comparator imports the unchanged repository compare.py mean-delta and
threshold definitions. Both losses use the 32-step mean absolute base0-vs-base1
envelope and base0-vs-feat0 signal. Ratio <3 is PASS, 3<=ratio<5 is INVESTIGATE,
and ratio>=5 is FAIL. Zero envelope requires zero signal; otherwise it is FAIL.
Step-1 forward difference is additionally reported without a separate gate.
INVESTIGATE exits 2 and is never summarized as PASS; any base2 follow-up requires
separate scheduling, rather than an automatic fourth arm. No log truncation or
lower step-count/tolerance override is available. Ratios use the production
logger's existing precision, exactly as the project comparator does.

## CPU prepare and separately authorized GPU execution

Use one pinned Python/runtime and the already verified offline processor cache.
Evidence must be a new directory outside both source trees and the data bundle.
Production training imports query CUDA capability in existing operators. CPU
prepare therefore checks the actual config fields with AST plus the shared
configuration builder, and executes only the CPU-safe real bundle/model checks.
It does not mock CUDA or claim the native training import/execution passed on CPU.

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 "$python" \
  "$harness/tests/acceptance/omni_noise_regression.py" prepare \
  --base-repo "$base" --feature-repo "$candidate" \
  --dataset "$verified_data" --output "$new_evidence"
```

Only after a new bounded GPU allocation is authorized:

```bash
export PITHTRAIN_RUN_OMNI_GPU_TESTS=1
srun --jobid="$SLURM_JOB_ID" -N1 -n1 -c32 --gres=gpu:4 -W 0 \
  --open-mode=truncate -o "$new_step_log" \
  bash "$harness/tests/acceptance/run_omni_noise_h100.sh" \
  "$python" "$new_evidence/plan.json"
```

Proposed resource ceiling: one node, 4 H100, 32 CPUs, 128 GiB host RAM, one hour,
at most 4 additional GPU-hours, no requeue. Fifteen arms x32 updates =480 native
updates in total across five meshes. Each arm has a 600-second cap and all arms
share 3300 seconds, followed by at most 15 seconds for child-process shutdown;
the outer wrapper has a 3330-second timeout plus 15-second kill bound. These
are ceilings, not measured runtime guarantees. On failure, timeout or a noise
INVESTIGATE verdict, later arms/profiles remain NOT RUN. Reserve 10 GiB local scratch for logs/configs/JIT caches (a proposal, not a
measured disk peak); no model/optimizer checkpoint payload is saved.
Do not increase allocation, reduce steps, retry arms or submit another job
automatically to make this budget fit.

A future transfer only needs the incremental source bundle and its manifest;
reuse existing verified real data and seven offline processor/config files.
The exact bytes/hash are recorded in the local delivery manifest. No upload
or new GPU submission is part of the current implementation task.

Do not request user PR review until all required tests actually pass. Queued,
NOT RUN, skipped or INVESTIGATE results do not qualify; this local CPU delivery
does not mark any PR ready, push or merge.
