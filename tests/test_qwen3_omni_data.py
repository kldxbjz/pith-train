"""Data-boundary tests for our Omni reader/collator, without model weights or GPUs."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("av", reason="Install the omni-data extra")
pytest.importorskip("librosa", reason="Install the omni-data extra")
pytest.importorskip("torchvision", reason="Install the omni-data extra")

import av
import soundfile as sf
from PIL import Image
from transformers import AutoConfig, Qwen3OmniMoeProcessor

from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset


@pytest.fixture(scope="module")
def processor_config():
    kwargs = dict(revision="26291f793822fb6be9555850f06dfe95f2d7e695", local_files_only=True)
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, **kwargs)
        config = AutoConfig.from_pretrained(model_id, **kwargs).thinker_config
    except OSError:
        pytest.skip("Run the Omni data preparation recipe to cache the small processor files")
    return processor, config


@pytest.fixture
def manifest(tmp_path):
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(tmp_path / "image.png")
    waveform = np.sin(2 * np.pi * 440 * np.arange(9600) / 8000).astype(np.float32) * 0.1
    sf.write(tmp_path / "audio.wav", waveform, 8000)
    with av.open(str(tmp_path / "video.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for color in [(0, 255, 0), (0, 0, 255)]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), color=color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows = [
        dict(id="text", text="A text example.", media=[]),
        dict(id="image", text="A red square.", media=[dict(type="image", path="image.png")]),
        dict(id="audio", text="A short tone.", media=[dict(type="audio", path="audio.wav")]),
        dict(
            id="video", text="Green changes to blue.", media=[dict(type="video", path="video.mp4")]
        ),
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_mixed_media_keeps_samples_targets_and_feature_counts(manifest, processor_config):
    processor, config = processor_config
    dataset = Qwen3OmniDataset(manifest)
    collate = Qwen3OmniCollator(processor, config)
    samples = [dataset[i] for i in [3, 1, 0, 2]]
    batch = collate(samples)
    assert batch.sample_ids == ("video", "image", "text", "audio")
    assert batch.model_inputs["input_ids"].shape == batch.labels.shape
    assert (
        batch.model_inputs["pixel_values"].shape[0]
        == batch.model_inputs["image_grid_thw"].prod().item()
    )
    assert (
        batch.model_inputs["pixel_values_videos"].shape[0]
        == batch.model_inputs["video_grid_thw"].prod().item()
    )
    for row, sample in enumerate(samples):
        single = collate([sample])
        length = single.labels.shape[1]
        torch.testing.assert_close(
            batch.model_inputs["input_ids"][row, :length], single.model_inputs["input_ids"][0]
        )
        torch.testing.assert_close(batch.labels[row, :length], single.labels[0])
        assert (batch.labels[row, length:] == -100).all()
        actual = batch.labels[row][batch.labels[row] != -100].tolist()
        expected = processor.tokenizer.encode(
            sample["text"] + processor.tokenizer.eos_token, add_special_tokens=False
        )
        assert actual == expected, sample["id"]
        for token_id in collate.media_token_ids:
            assert token_id not in actual
    # The 8 kHz input is resampled, not falsely declared to be 16 kHz.
    assert batch.media_info[3][0]["sampling_rate"] == 16000
    assert batch.media_info[3][0]["samples"] == 19200
    assert batch.media_info[0][0]["timestamps"] == [0.0, 0.5]
    assert batch.model_inputs["video_second_per_grid"].tolist() == [1.0]


def test_video_resampling_matches_processor_timing(manifest, processor_config):
    processor, config = processor_config
    sample = Qwen3OmniDataset(manifest)[3]
    batch = Qwen3OmniCollator(processor, config, video_fps=4)([sample])
    info = batch.media_info[0][0]
    assert info["timestamps"] == [0.0, 0.25, 0.5, 0.75]
    assert info["source_timestamps"] == [0.0, 0.0, 0.5, 0.5]
    assert batch.model_inputs["video_second_per_grid"].tolist() == [0.5]


def test_multiple_media_preserve_manifest_order(manifest, processor_config):
    processor, config = processor_config
    dataset = Qwen3OmniDataset(manifest)
    sample = dict(
        id="mixed",
        text="A tone and a red square.",
        media=[dataset[2]["media"][0], dataset[1]["media"][0]],
    )
    batch = Qwen3OmniCollator(processor, config)([sample])
    ids = batch.model_inputs["input_ids"][0].tolist()
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert ids.index(audio_id) < ids.index(image_id)
    assert [item["type"] for item in batch.media_info[0]] == ["audio", "image"]


def test_rejects_truncating_expanded_media(manifest, processor_config):
    processor, config = processor_config
    sample = Qwen3OmniDataset(manifest)[1]
    with pytest.raises(ValueError, match="do not truncate media tokens"):
        Qwen3OmniCollator(processor, config, max_length=2)([sample])


@pytest.mark.parametrize(
    "media",
    [
        [dict(type="image", path="missing.png")],
        [dict(type="audio_video", path="clip.mp4")],
    ],
)
def test_rejects_unusable_media_at_manifest_read(tmp_path, media):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(dict(id="bad", text="Text.", media=media)))
    with pytest.raises(ValueError, match="bad:"):
        Qwen3OmniDataset(path)


def test_rejects_reserved_placeholder_in_paired_text(manifest, processor_config):
    processor, config = processor_config
    sample = dict(Qwen3OmniDataset(manifest)[0], text="Unexpected <|image_pad|> token")
    with pytest.raises(ValueError, match="reserved media tokens"):
        Qwen3OmniCollator(processor, config)([sample])
