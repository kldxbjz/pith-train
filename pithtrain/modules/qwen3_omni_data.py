"""Read local media/text pairs and build Qwen3-Omni Thinker pretraining inputs.

The batch keeps HF's media field names. It is not yet a DualPipeV Microbatch:
native Omni must consume these fields and construct multimodal positions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import Dataset


class Qwen3OmniDataset(Dataset):
    """JSONL records: id, text, and an ordered media list of {type, path}.

    Paths are relative to the manifest. This small map-style reader deliberately
    leaves token-only corpora on MemmapDataset and does not introduce chat roles.
    """

    def __init__(self, manifest: str | Path):
        manifest = Path(manifest).resolve()
        self.root = manifest.parent
        self.samples = []
        seen = set()
        for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = sample.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"Line {line_number}: missing or duplicate sample id")
            if not isinstance(sample.get("text"), str) or not sample["text"].strip():
                raise ValueError(f"{sample_id}: expected nonempty paired text")
            if not isinstance(sample.get("media"), list):
                raise ValueError(f"{sample_id}: media must be a list (empty for text-only)")
            media = []
            for item in sample["media"]:
                if set(item) != {"type", "path"} or item["type"] not in {"image", "audio", "video"}:
                    raise ValueError(f"{sample_id}: expected image/audio/video with a local path")
                path = (self.root / item["path"]).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    raise ValueError(
                        f"{sample_id}: media is missing or outside the manifest directory"
                    )
                media.append(dict(type=item["type"], path=path))
            seen.add(sample_id)
            self.samples.append(dict(id=sample_id, text=sample["text"], media=media))
        if not self.samples:
            raise ValueError("The manifest has no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@dataclass
class Qwen3OmniBatch:
    """Labels are ALREADY shifted for PithTrain's external next-token CE.

    For HF comparison, pass model_inputs without labels and calculate CE against
    this labels tensor externally; passing it as HF labels would shift twice.
    media_info is grouped by sample, in the manifest's original media order.
    """

    sample_ids: tuple[str, ...]
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    media_info: tuple[list[dict], ...]


class Qwen3OmniCollator:
    """Media-prefix text prediction, without an instruction/chat template.

    Prefix a document boundary (the tokenizer's EOS), then predict all paired
    text tokens and EOS. Ignore padding and media wrapper/placeholder targets.
    Media tensors stay attached, never flattened into .bin.
    Video currently means visual frames only; clips with audio are rejected so
    an audio track cannot silently disappear from an Omni training sample.
    """

    def __init__(
        self,
        processor,
        thinker_config,
        *,
        max_length=2048,
        min_pixels=4096,
        max_pixels=65536,
        video_fps=2.0,
        max_video_frames=32,
        max_audio_seconds=30.0,
    ):
        if max_length < 2 or video_fps <= 0 or max_video_frames < 2:
            raise ValueError("Invalid sequence length or video sampling limits")
        if min_pixels <= 0 or max_pixels < min_pixels or max_audio_seconds <= 0:
            raise ValueError("Invalid media size/duration limits")
        self.processor = processor
        self.config = thinker_config
        self.max_length = max_length
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.video_fps, self.max_video_frames = video_fps, max_video_frames
        self.max_audio_seconds = max_audio_seconds
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.media_tokens = [
            getattr(processor, name)
            for name in (
                "image_token",
                "audio_token",
                "video_token",
                "vision_bos_token",
                "vision_eos_token",
                "audio_bos_token",
                "audio_eos_token",
            )
        ]
        self.media_token_ids = processor.tokenizer.convert_tokens_to_ids(self.media_tokens)

    def _audio(self, path):
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            if len(stream) / rate > self.max_audio_seconds:
                raise ValueError(f"Audio exceeds {self.max_audio_seconds}s: {path}")
            audio = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        if rate != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=self.sampling_rate)
        return audio, dict(type="audio", sampling_rate=self.sampling_rate, samples=len(audio))

    def _video(self, path):
        frames, timestamps, source_timestamps = [], [], []

        def append(frame, source_time):
            if len(frames) == self.max_video_frames:
                raise ValueError(f"Video exceeds {self.max_video_frames} sampled frames: {path}")
            timestamps.append(len(frames) / self.video_fps)
            source_timestamps.append(source_time)
            frames.append(frame.to_ndarray(format="rgb24"))

        with av.open(str(path)) as container:
            if container.streams.audio:
                raise ValueError(
                    f"Video has an audio track: {path}. Synchronized audio/video is not supported yet."
                )
            start, previous, previous_time = None, None, None
            for frame in container.decode(video=0):
                if frame.time is None:
                    raise ValueError(f"Video frame has no timestamp: {path}")
                if start is None:
                    start = float(frame.time)
                timestamp = float(frame.time) - start
                if previous is not None:
                    if timestamp <= previous_time:
                        raise ValueError(f"Video timestamps are not increasing: {path}")
                    # Hold the previous frame on a uniform timeline. This also
                    # handles source FPS below the requested sampling rate.
                    while len(frames) / self.video_fps < timestamp - 1e-6:
                        append(previous, previous_time)
                previous, previous_time = frame, timestamp
            if previous is not None:
                duration = float(previous.duration * previous.time_base)
                if duration <= 0:
                    rate = container.streams.video[0].average_rate
                    if rate is None or rate <= 0:
                        raise ValueError(f"Video's last frame has no duration: {path}")
                    duration = 1 / float(rate)
                end = previous_time + duration
                while len(frames) / self.video_fps < end - 1e-6:
                    append(previous, previous_time)
        if len(frames) < 2:
            raise ValueError(f"Video needs at least two sampled frames: {path}")
        return np.stack(frames), dict(
            type="video",
            fps=self.video_fps,
            timestamps=timestamps,
            source_timestamps=source_timestamps,
        )

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        processor = self.processor
        texts, images, audio, videos, media_info = [], [], [], [], []
        for sample in samples:
            if any(token in sample["text"] for token in self.media_tokens):
                raise ValueError(f"{sample['id']}: paired text contains reserved media tokens")
            # A boundary also gives the first text-only token a preceding input.
            prefix, info = [processor.tokenizer.eos_token], []
            for item in sample["media"]:
                kind, path = item["type"], item["path"]
                if kind == "image":
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        images.append(image)
                        info.append(dict(type=kind, width=image.width, height=image.height))
                    prefix.append(
                        processor.vision_bos_token
                        + processor.image_token
                        + processor.vision_eos_token
                    )
                elif kind == "audio":
                    waveform, details = self._audio(path)
                    audio.append(waveform)
                    info.append(details)
                    prefix.append(
                        processor.audio_bos_token
                        + processor.audio_token
                        + processor.audio_eos_token
                    )
                else:
                    video, details = self._video(path)
                    videos.append(video)
                    info.append(details)
                    prefix.append(
                        processor.vision_bos_token
                        + processor.video_token
                        + processor.vision_eos_token
                    )
            texts.append("".join(prefix) + sample["text"] + processor.tokenizer.eos_token)
            media_info.append(info)
        inputs = dict(
            processor(
                text=texts,
                images=images or None,
                audio=audio or None,
                videos=videos or None,
                text_kwargs=dict(padding=True, padding_side="right", add_special_tokens=False),
                images_kwargs=dict(min_pixels=self.min_pixels, max_pixels=self.max_pixels),
                audio_kwargs=dict(
                    sampling_rate=self.sampling_rate, n_window=self.config.audio_config.n_window
                ),
                videos_kwargs=dict(
                    fps=self.video_fps,
                    do_sample_frames=False,
                    cap_pixels_per_frame=True,
                    size=dict(shortest_edge=self.min_pixels, longest_edge=self.max_pixels),
                    position_id_per_seconds=self.config.position_id_per_seconds,
                    use_audio_in_video=False,
                ),
                return_tensors="pt",
            )
        )
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if ids.shape[1] - 1 > self.max_length:
            raise ValueError(
                f"Expanded media/text length {ids.shape[1] - 1} exceeds {self.max_length}; do not truncate media tokens"
            )
        if ids.min() < 0 or ids.max() >= self.config.text_config.vocab_size:
            raise ValueError("Processor token IDs exceed the model vocabulary")
        labels = ids[:, 1:].clone()
        valid = mask[:, 1:].bool() & mask[:, :-1].bool()
        for token_id in self.media_token_ids:
            valid &= labels != token_id
        labels.masked_fill_(~valid, -100)
        if not (labels != -100).any(dim=1).all():
            raise ValueError("Each sample must have a text target")
        inputs["input_ids"] = ids[:, :-1].contiguous()
        inputs["attention_mask"] = mask[:, :-1].contiguous()
        return Qwen3OmniBatch(
            tuple(sample["id"] for sample in samples),
            inputs,
            labels.contiguous(),
            tuple(media_info),
        )
