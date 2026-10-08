# Week4 acceptance matrix — executed evidence versus prepared GPU gates

This supplement corrects any ambiguity in the initial summary. **No native
Thinker image logits, image CE loss, decoder gradients, GPU optimizer step,
DualPipeV image execution, FSDP image execution or PP+EP DeepStack execution has
been run.** HF comparisons executed so far cover native vision and geometry.
CPU vision gradients/updates must not be described as complete image training.

## Test accounting

Original frozen source `26e75e5d677e7e4d95ff01e39f4607e516f7f002`:
**269 passed / 0 failures / 0 errors / 0 skipped** (187.993 seconds in JUnit).
Of these, **40 newly added tests** are 35 in `test_qwen3_omni_vision.py` and 5
in `test_omni_image_contract.py`. **229 are existing regressions**. The 18
existing setup tests have fixture updates for the new configuration flag and
the actual model resolver; their acceptance assertions were retained.

The follow-up adds **12 CPU tests** in `test_omni_image_acceptance_evidence.py`:
per-component evidence checks and their negative cases, actual CPU optimizer
hook observation, global all-zero rejection on four Gloo ranks, and DP1/2/4
fixture ordering. The complete current inventory is **281 tests: 52 new + 229
existing**. The final delivery JUnit, commit and TRANSFER.json bind the result;
this inventory is not itself a claim that a queued run passed.

| Original file | Cases |
| --- | ---: |
| test_qwen3_omni_vision | 35 new |
| test_omni_image_contract | 5 new |
| test_omni_model_registration | 6 |
| test_omni_training | 5 |
| test_qwen3_omni_data | 12 |
| test_training_data | 13 |
| test_training_sequence_length | 18 |
| test_qwen3_omni_checkpoint | 67 |
| test_strict_checkpoint | 40 |
| test_strict_omni_converter | 47 |
| test_distributed_strict_checkpoint | 6 |
| test_omni_native_resume_contract | 15 |
| **Original total** | **269** |
| Follow-up: test_omni_image_acceptance_evidence | **12 new** |

## Acceptance matrix

| Requirement | Actually executed | Still pending / prepared gate |
| --- | --- | --- |
| Native vision output versus independent HF | **CPU PASS**: last vision hidden state, main projected output and every DeepStack projected output. Multiple grids/activations in synthetic tiny cases; main/DeepStack output parity on two real processor batches. | H100 BF16 behavior and full released encoder dimensions/weights are not verified. |
| Vision gradients and optimizer updates | **CPU PASS**: all 57 named visual parameters, native/HF parameter membership and shape, finite gradient parity and updates. Synthetic tests also compare pixel-input gradients. Real-image audit uses a projected-feature scalar probe, not text CE. | Image CE through the full native model is NOT RUN. |
| Real image/token/target alignment | **CPU PASS**: coco-57870 and coco-384029, actual PR100 dataset/collator, fixed processor revision; image/pixel hashes, grids, placeholders, feature counts, positions, text and shifted labels recorded. Targets are 10 and 12. | Larger resolution, more samples, pretrained/full-size execution are not covered. |
| Native multimodal position IDs | **CPU PASS**: exact T/H/W comparison against HF, multiple image runs, padding geometry helper and malformed runs. Production native image mode explicitly rejects padding/packing. | Their consumption by native GPU decoder attention is NOT RUN. |
| Full image+text logits and CE versus HF | **NOT RUN.** No CPU test claims this. | `--mode reference`: native vision + native Thinker versus independent HF vision + HF text; real processor tensors, identical tiny BF16 weights, logits and externally shifted CE, four steps. |
| Every declared training module's gradients | **Only vision PASS on CPU**; no native decoder gradient comparison executed. | Reference compares every parameter key, gradient presence/finite values and normalized error <1e-2. Staged gate compares each owned shard to the HF global-batch gradient. No name-based parameter skips. |
| Encoder/projection/decoder update witnesses | **Vision only PASS** on CPU scalar probe. Additional CPU hook tests validate evidence machinery using small ordinary parameters. | Follow-up requires nonzero gradients and changes for patch/position embeddings, every vision block, main merger, each DeepStack merger, text embeddings/norm/head and every decoder block. Four real task-loop updates are observed using optimizer pre/post hooks. GPU NOT RUN. |
| Effective-label global normalization, no PP duplication | **CPU 4-rank Gloo PASS**, using the real `global_target_count`/`global_loss_mean` functions with a small Linear+CE model: per DP/CP group one rank has 0 valid labels and one has 3; two PP replicas must both count 3, not 6. Gradients and SGD update match an independent CE reference. | Native CE/FSDP plain-sum reduction under WORLD1/DP2/PP2/EP2/PP2EP2 remains NOT RUN. |
| All ranks globally have zero labels | **Not present in the original 269.** Follow-up adds and executes a four-rank Gloo test: both PP-stage groups reject identically via the actual target-count function. | GPU all-zero rejection is prepared in every staged profile but NOT RUN. |
| Two different microbatches cannot share features | **CPU PASS**: two live native vision graphs, different images, reverse-order backward equals independent execution; differentiable feature-lane helper matches explicit injection and gradients. | Full decoder and overlapped scheduler isolation NOT RUN. Follow-up fixes the test ordering so every DP rank receives alternating images at DP1/2/4 instead of a single repeated image per rank. |
| DeepStack at specified layers | **CPU PASS for vision tap indexes and pure injection/gradient math.** Registered image layer structure inspected. | Actual native `forward_stage1/3/5`, PP crossing, EP dispatch and DeepStack gradients/updates are NOT RUN. |
| Bad sample consistent on all ranks | **CPU 4-rank Gloo PASS**: actual OmniPretrainData, PP+DP groups; only one reader gets a NaN pixel; all four ranks reject with the same error and no durable cursor advancement. | This is not NCCL/process-death recovery evidence. |
| Text regression | **CPU PASS** for data, config/registration, setup, checkpoint/converter and strict resume control paths. | Existing native text HF GPU reference is a required first CUDA gate and remains NOT RUN on this branch. |
| Image checkpoint/HF conversion | **No claim.** Existing text checkpoint CPU regressions passed. | Image checkpoint/converter acceptance, mixed-modality DP and full-model acceptance are outside this first image milestone. |

## GPU package contents and order

The package has real executable tests, not recorded GPU results:

1. `run_week4_h100.sh`: explicit opt-in + Slurm allocation; clean tree and H100
   checks; offline CPU/no-skip gates; existing text GPU reference; image HF
   reference; the staged ladder; mandatory commit-matched JSON reports.
2. `omni_image_training.py --mode cpu`: real-image alignment and independent
   HF/native visual audit. `--mode reference`: four image CE steps, logits,
   losses, every gradient and each component's update witness.
3. `omni_image_staged.py` (through `--mode staged`): initial fixed real-image
   global batch, all owned gradients versus independent HF, target-count and
   zero-target checks, then four actual production `pretrain_lm.train_step`
   calls using the unchanged PR100 reader. Each real optimizer step has
   per-component nonzero-gradient/update evidence.
4. `omni_image_evidence.py`: per-component evidence and observer helpers.
   CPU tests prove dead/frozen/missing/nonfinite components cannot yield PASS.
5. All required source/config/test files in the Git bundle; complete existing
   checksummed development assets and seven processor/config files in the
   asset archive. 81 asset files total. No model weights, checkpoints, venv,
   credentials, uploads, job submission or FA4 change.

| Profile | GPUs / ranks | PP | EP | attention DP | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| native reference | 1 | none | 1 | 1 | NOT RUN |
| w1-pp1-ep1 | 1 | 1 | 1 | 1 | NOT RUN |
| w2-pp1-ep1 | 2 | 1 | 1 | 2 | NOT RUN |
| w2-pp2-ep1 | 2 | 2 | 1 | 1 | NOT RUN |
| w2-pp1-ep2 | 2 | 1 | 2 | 2 | NOT RUN |
| w4-pp2-ep2 | 4 | 2 | 2 | 2 | NOT RUN |

All profiles have CP1. Global batch is 8 and microbatch size is 1. Tiny weights
use the real 152064 vocabulary, 4 text layers/128 hidden, and 3 vision layers/64
hidden. Vision layers 0 and 2 project to decoder layers 0 and 1. Activation
transport width is 384 (128 decoder + two 128-wide feature lanes), deliberately
trading bandwidth for explicit per-microbatch activation/gradient ownership.

Resource proposal remains **one node / 4 H100 / 45 minutes = at most 3 allocated
GPU-hours**, 32 CPU, 128 GiB RAM, 32 GiB scratch. Global script budget is 40
minutes; GPU gate caps are 120/180 seconds with 15 seconds kill grace. No GPU
execution or upload is authorized by packaging this proposal.

The v1 package and its original 269-test logs are preserved. The follow-up v2
delivery contains the final full commit SHA, exact file hashes/sizes, latest
JUnit, this matrix and an isolated bundle/asset reconstruction audit. Use the
v2 README and TRANSFER.json for the final review paths and source identity.
