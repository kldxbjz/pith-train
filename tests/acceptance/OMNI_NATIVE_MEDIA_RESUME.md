# Native media staged training and strict resume

These are acceptance drivers, not stored GPU success claims. Use the same clean
commit, pinned dependencies, H100 hardware and prepared input bundle for every
arm. Keep the process timeout at 180 seconds and run jobs sequentially under the
coordinator's shared budget.

The audio source uses two real LibriSpeech recordings with the pinned Omni
processor and a reduced native encoder/text model retaining the real vocabulary
and mel geometry. The staged comparison keeps the existing thresholds:
cross entropy rtol=atol=1e-3 and every parameter gradient NSE < 1e-2. It includes
eight nonuniform microbatches, a separate all-zero DP rank scenario, globally
zero-target rejection, and four production `train_step` updates using the real
reader. Every audio projection, encoder layer, and text component must have
finite, nonzero gradients and updates.

| PP | EP | DP | CP | torchrun processes |
| --- | --- | --- | --- | --- |
| 1 | 1 | 2 | 1 | 2 |
| 1 | 2 | 2 | 1 | 2 |
| 2 | 1 | 2 | 1 | 4 |
| 2 | 2 | 2 | 1 | 4 |

First run CPU fixture preflight into its own fresh directory:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=2 \
  timeout 180 python tests/acceptance/omni_audio_training.py \
  --mode preflight --dataset DATA --output CPU_PREFLIGHT \
  --pp 2 --ep 2 --dp 2 --scenario rank-zero
```

Then run each requested topology and both scenarios in separate fresh GPU jobs.
Use a different output directory for every job:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  timeout --signal=TERM --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=4 tests/acceptance/omni_audio_training.py \
  --mode staged --dataset DATA --output NEW_STAGED_OUTPUT \
  --pp 2 --ep 2 --dp 2 --scenario rank-zero
```

For full native checkpoint acceptance, prepare a separate directory on CPU.
Preparation copies and verifies the bundle, records source identity and exact
sample hashes, and saves the reduced composite model config:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=2 \
  timeout 180 python tests/acceptance/omni_media_resume.py \
  --mode prepare --dataset DATA --output RESUME_OUTPUT \
  --world-size 4 --pp 2 --ep 2
```

The producer and restore commands must be separate Python/torchrun processes,
with the same shared output storage and source/input identity:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  timeout --signal=TERM --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=4 tests/acceptance/omni_media_resume.py \
  --mode produce --dataset DATA --output RESUME_OUTPUT \
  --world-size 4 --pp 2 --ep 2

PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  timeout --signal=TERM --kill-after=10s 180s \
  torchrun --standalone --nproc-per-node=4 tests/acceptance/omni_media_resume.py \
  --mode restore --dataset DATA --output RESUME_OUTPUT \
  --world-size 4 --pp 2 --ep 2
```

The producer saves strict rank-local shards after two production updates,
captures complete runtime state, and performs the third continuous update.
The restore process starts with empty optimizer state, deliberately clears
model weights, changes RNG, then uses production strict loading. It requires
exact model/optimizer/scheduler/Python/NumPy/CPU/CUDA RNG/data restoration,
identical next input IDs, labels, media tensors, layout/context and sample IDs,
identical third objective sums, and exact third-update state. Missing media
towers, gradients or component updates cannot pass.

The existing five corruption cases are available in fresh processes:
`--mode negative --negative-case missing-rank|missing-cuda-rng|shape|topology|optimizer-order`.
They corrupt only copies beneath the evidence directory and require all-rank
failure atomicity.

The runner exposes `MediaSpec` and `execute(args, spec)` for a combined native
consumer. Supply its actual source/config loader, parameter-name mapper,
component groups, training flag and modality tuple. Set
`required_components=("audio_tower.", "visual.")` to require both towers in
the globally collected checkpoint keys. The default audio CLI requires only
`audio_tower.`. Every required component must be exercised in each observed
global training step.

`--restore-root` can validate a checkpoint rebuilt through an authorized
complete-Thinker HF bridge while still comparing against the original producer
boundary. The full-Thinker converter requires both towers; an audio-only model
is validated by direct strict save/restore here, not claimed as a complete
Thinker conversion.

A completed CPU prepare, a parsed CLI, or the presence of a driver does not
establish GPU correctness. Retain per-rank JSON, stdout, input hashes, exact
commit and actual hardware/dependency versions with the final acceptance result.
