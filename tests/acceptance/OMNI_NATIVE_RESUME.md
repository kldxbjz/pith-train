# Native Omni update and fresh-process strict resume

This is a **new, unsharded native-model acceptance entrance**, prepared on CPU.
Its GPU numerical gates have not been executed. It does not establish
DualPipeV, PP/EP/FSDP updates, distributed strict resume, or 32-step regression.
Do not add it to the already frozen first Flame package without its owner
explicitly scheduling a later revision. No upload or submission is performed here.

## Source integration

Independent worktree `/home/kldxbjz/code/pith-train-week3-native-resume`, branch
`feat/omni-native-resume`. Merge base commit for this runner is
`a0b14f7abbf4f84c7a9fbb76eb5a759a63f65010`, whose two parents are PR100
`24529b3101dd339782d2dc97b4ffff06685a60e7` and strict-resume
`fe288fd0a69b5f94ad0d7f3d7ed8f4a42a16ed95` (includes converter ace46ec).
Original worktrees/branches are unchanged. Merge resolution keeps PR100's
`model_class_for_config`, registers Omni, and resolves official top-level
Omni / Thinker / text configs to the implemented text decoder. This does not
register or instantiate media encoders.

PR100 `training_data.py`, `dataset.py`, and `checkpoint.py` are unchanged.
The strict helper is byte-identical to fe288fd and real-loader acceptance
179b39e. That earlier acceptance remains supporting loader evidence; its CPU
Linear holder and machine-specific guards are not used by this entrance.

## Runnable modes

From the clean committed integration repo, with the appropriate existing
Python environment (Torch, Transformers 5.17, and PR100 data dependencies):

```bash
# CPU only: verifies real bundle checksums, offline processor/config cache and
# token bounds/capacity; copies data and records source/config fingerprints.
CUDA_VISIBLE_DEVICES= PYTHONPATH=. HF_HUB_OFFLINE=1 \
  python tests/acceptance/omni_native_resume.py --mode preflight \
  --dataset /path/to/omni-training-small --output /new/evidence
```

Preflight never constructs `ConcatDataset` or initializes CUDA. Subsequent
GPU modes call production `pretrain_lm.setup_dataset`, which constructs the
**unchanged real** `OmniPretrainData` →
`DensePretrainData` → `ConcatDataset`, including CUDA shuffle and barrier.
Every mode rechecks data and source fingerprints. Cache writes occur only in
the copied corpus. Input labels are already shifted by PR100 and are not
shifted again.
The recipe's pinned processor/config assets must already be available in the
HF cache. Preflight requests them with `local_files_only=True`; no weights or
other files are downloaded. Public resolver normalization also covers the
configuration passed by production `setup_dataset`.

Only the sole coordinated GPU owner may run this later, in its allocation:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 bash tests/acceptance/run_omni_native_resume.sh \
  /venv/bin/python /path/to/omni-training-small /another/new/evidence
```

The script uses separate processes for `preflight`, `update`, `produce`,
`restore`, and `negative`, stops at the first failed mode, caps each at 180 s,
and does not submit or upload. A failed output is retained and cannot be
overwritten. Reserve one supported GPU; no Slurm/GPU scheduling is embedded.
Before execution record the clean native FA4 environment provenance. Never
load the historical 5080 SDPA adapter or local FA4 fixes.

## Model and criteria

- Existing tiny text config: V152064, hidden128, four layers/four experts.
  Native phase=-1, actual native FA4/MLP kernels, BF16. HF uses eager attention
  and FP32 rotary buffers. Initial HF tensors load through the converter's
  canonical mapping and public checkpoint localization with strict keys.
- Real text S32, GB4, MB1, seed1234; AdamW lr=1e-4, weight decay0.1,
  `foreach=False`; StepLR gamma0.9. These are fixed acceptance settings, not
  a new recommended production schedule. No forced-balanced routing.
- `update`: loss/backward over the complete real batch with PR100 global-token
  normalization; original logits `<1e-3`, CE rtol/atol1e-3, all gradients
  present/finite and normalized squared error `<1e-2`. Compare actual optimizer
  deltas for **every named parameter** against independently stepped HF with
  `<1e-2`; prove at least one weight changed and all optimizer counters advanced.
  No zip truncation, gradient-name skips, or absolute-weight error camouflage.
- `produce`: native model runs two real updates, commits each loader step,
  saves using `save_strict_checkpoint` (public `CheckpointState` mapping),
  observes live state **after saving**, and continues the third update.
- `restore`: a distinct process constructs a fresh native model/optimizer and
  real loader, zeroes model weights and changes RNG seeds, then loads with
  `load_strict_checkpoint`. Boundary weights, optimizer, scheduler, data cursor
  and Python/NumPy/CPU/CUDA RNG must match exactly. Third-step batch/indices,
  loss, full post-update state and RNG probes must match the continuous arm
  with zero tolerance. The RNG probes audit the generators after real updates;
  they do not replace model inputs or gradients.
- `negative`: on the real restored native model, reject 11 corrupt snapshots
  (missing/unexpected/shape/dtype model, missing optimizer/scheduler/data/CUDA
  RNG, wrong fingerprint/cursor, reordered optimizer membership) without
  changing live state. Existing 40 CPU strict tests continue to cover actual
  DCP metadata corruption; the new mode is additional CUDA-state validation.

Native FA4 may be nondeterministic. If exact continuation fails, retain that
failure: checkpoint-boundary exactness stays mandatory; do not relax the next
update criterion automatically. The repository's base0/base1/feature 32-step
noise-envelope procedure is separate work and is not implemented by this
three-step runner.

## CPU checks and remaining implementation

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. HF_HUB_OFFLINE=1 python -m pytest \
  tests/test_omni_model_registration.py tests/test_omni_native_resume_contract.py \
  tests/test_qwen3_omni_checkpoint.py tests/test_strict_checkpoint.py -q
```

New CPU checks cover official nested configs/registration, GPU opt-in ordering,
the explicit distributed limitation, key/shape/dtype/nonfinite rejection,
optimizer-delta sensitivity and exact detached snapshot comparison. They do
not execute native kernels or mock a loader to claim native acceptance.

Production `setup_training` still builds DualPipeV/FSDP; this entrance builds
the actual full native model with phase=-1 so it can use the implemented
single-process strict checkpoint contract. Production task strict-resume
wiring and true distributed strict resume remain **unimplemented**. Required
future work includes collective metadata preflight, rank-local RNG/shard
validation, synchronized failure atomicity and actual PP/EP/FSDP optimizer
update/restart comparisons. Existing guards are retained; removing a guard
alone is not an implementation.
