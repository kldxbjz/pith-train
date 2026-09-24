# Qwen3-Omni multimodal data

Prepare actual image/text, audio/transcript and video/text pairs, then read them
through our local manifest reader and the official Omni processor. This produces
the media tensors that native Thinker will need, rather than a token-only `.bin`.
No model weights or GPUs are needed for data preparation.

## Run the small corpus

Install PithTrain's `omni-data` extra in the development environment, then run:

```bash
python examples/prepare_omni_data/qwen3-omni-smoke/script.py
```

The pinned sources in `config.json` download about 20 MB in total:

| Input | Paired text | Source |
| --- | --- | --- |
| Image of two cats | A caption written after inspecting the image | `hf-internal-testing/fixtures_image_utils`, `cats.jpg` |
| Two speech recordings | Original LibriSpeech transcripts | `hf-internal-testing/librispeech_asr_dummy`, IDs fixed in config |
| Ten-second video without audio | A caption written after inspecting its frames | `raushan-testing-hf/videos-test`, Big Buck Bunny clip |
| Text-only control | Reuses the image caption | No additional corpus |

Each source has a fixed revision and SHA-256. Image/video captions are integration
annotations, not claimed to be original dataset labels. Source cards and original
asset terms remain applicable; downloaded media is not committed to this repo.
These five examples are integration data, not a production training corpus or a
model-quality evaluation set. Larger local media/text collections use the same
reader, which now indexes JSONL shards by byte offset. For source-labeled
train/validation splits, scalable limits, mixture sampling and GCS publication,
use the [training recipe](../qwen3-omni-training/README.md).

Outputs go under ignored `workspace/datasets/omni-multimodal-smoke/`:

- `media/`: actual image, audio and video files.
- `samples.jsonl`: text/media pairings in a fixed order.
- `provenance.json`: original sources, annotation provenance and file hashes.
- `batch-report.json`: actual tensor shapes, decoded text targets, audio lengths,
  video timestamps and a mixed-batch report (up to eight samples). It records that native
  training was not executed.

Rerun with `HF_HUB_OFFLINE=1` after the source files and processor are cached.
Use `--output PATH` to choose a working directory. On Orchard, large persistent
media belongs in GCS and the active working set on an allocated compute node's
`/tmp`; avoid copying corpora to `/home` or shared `/project` for every experiment.

## Use your own image/audio/video pairs

There is no chat template or assistant-only SFT schema. The manifest is JSONL:

```json
{"id":"image-1","text":"A description of the picture.","media":[{"type":"image","path":"media/image.jpg"}]}
{"id":"audio-1","text":"The recording's actual transcript.","media":[{"type":"audio","path":"media/audio.wav"}]}
{"id":"video-1","text":"A description of the clip.","media":[{"type":"video","path":"media/video.mp4"}]}
```

Paths are relative to the manifest directory. Multiple media items in a sample
are allowed and retain their order. `media: []` denotes a text-only record.

```bash
python examples/prepare_omni_data/qwen3-omni-smoke/script.py \
  --manifest /path/to/local/samples.jsonl --output workspace/omni-data-check
```

This uses the supplied data without downloading the example corpus. Batch limits
are in the recipe config. The loader fails on excessive duration/expanded length
instead of truncating text tokens while leaving media features misaligned.

## What the model receives

`Qwen3OmniDataset` and `Qwen3OmniCollator` live in
`pithtrain/modules/qwen3_omni_data.py`. The collator returns `Qwen3OmniBatch`:

- `model_inputs`: `input_ids`, `attention_mask`, and present media fields:
  `pixel_values`/`image_grid_thw`, `input_features`/`feature_attention_mask`,
  `pixel_values_videos`/`video_grid_thw`/`video_second_per_grid`.
- `labels`: already shifted once, aligned with `input_ids` for external CE.
- `sample_ids` and `media_info`: original sample order, audio sample counts,
  and sampled video times. Flattened media tensors follow sample/media order.
Video is resampled onto a uniform timeline by holding the preceding frame;
original frame timestamps are also retained. Its sampling rate therefore
matches the processor's position-timing metadata, including low-FPS sources.

Each sequence is a document boundary (tokenizer EOS), media placeholders, paired
text, and EOS. HF's processor expands the placeholders. Loss targets are the
paired text and final EOS; media wrappers/placeholders and padding are `-100`.
The initial boundary allows a text-only sample's first text token to be predicted.
The default delimiter follows the pinned tokenizer; this is an explicit initial
data contract, not a claim about the original model's full pretraining recipe.

For HF comparison, call the model with `**batch.model_inputs` **without** passing
these shifted labels to HF, then calculate CE against `batch.labels` externally.
The eventual training integration must normalize by the actual non-ignored target
count. Media grids, masks and timing are retained for model-side position IDs;
the collator does not pretend to implement Omni's MRoPE or encoders.

## Scope and validation

Tests exercise our pairing, resampling, media order, next-token labels, mixed
padding and rejection of invalid/truncated media. They compare batching with
individual samples, using the actual cached processor, not a mock HF model:

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/test_qwen3_omni_data.py tests/test_dataset.py -q
```

The current output is not yet a DualPipeV `Microbatch`. Native Omni's media input
and position handling, variable-target loss normalization, PP/CP routing, encoder
gradients and optimizer steps are model-integration work. Data readiness can be
validated now; it is separate from claiming that multimodal training runs.

Synchronized audio-in-video and audio output/Talker are not implemented here.
Videos with audio tracks fail explicitly so their audio cannot silently be lost.
Independent image/audio/video samples, and their mixed batches, are supported.
