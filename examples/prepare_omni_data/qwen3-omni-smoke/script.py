"""Prepare actual image/audio/video-text pairs and exercise the Omni data loader."""

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoConfig, Qwen3OmniMoeProcessor

from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output: Path, config: dict) -> Path:
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    samples, provenance = [], []
    for source in config["sources"]:
        cached = Path(
            hf_hub_download(
                source["repo_id"], source["file"], revision=source["revision"], repo_type="dataset"
            )
        )
        if sha256(cached) != source["sha256"]:
            raise ValueError(f"Source hash changed: {source['repo_id']}/{source['file']}")
        kind = source["kind"]
        if kind == "audio":
            rows = {row["id"]: row for row in pq.read_table(cached).to_pylist()}
            for sample_id in source["sample_ids"]:
                row = rows[sample_id]
                dest = media / f"{sample_id}.flac"
                dest.write_bytes(row["audio"]["bytes"])
                samples.append(
                    dict(
                        id=sample_id,
                        text=row["text"],
                        media=[dict(type=kind, path=str(dest.relative_to(output)))],
                    )
                )
        else:
            dest = media / cached.name
            shutil.copyfile(cached, dest)
            samples.append(
                dict(
                    id=kind + "-0",
                    text=source["text"],
                    media=[dict(type=kind, path=str(dest.relative_to(output)))],
                )
            )
        provenance.append(source)
    # Reuse a real caption as a text-only control; do not create a second text corpus.
    samples.insert(0, dict(id="text-control", text=samples[0]["text"], media=[]))
    manifest = output / "samples.jsonl"
    manifest.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples)
    )
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                purpose="Small integration corpus; not a training-quality or evaluation dataset",
                sources=provenance,
                files={
                    str(path.relative_to(output)): sha256(path) for path in sorted(media.iterdir())
                },
                manifest_sha256=sha256(manifest),
            ),
            indent=2,
        )
        + "\n"
    )
    return manifest


def check(manifest: Path, config: dict, snapshot: str, output: Path) -> dict:
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot, local_files_only=True)
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True).thinker_config
    dataset = Qwen3OmniDataset(manifest)
    collator = Qwen3OmniCollator(processor, model_config, **config["batch"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    reports = []
    for batch in loader:
        targets = batch.labels[batch.labels != -100].tolist()
        reports.append(
            dict(
                sample_ids=batch.sample_ids,
                tensors={
                    name: dict(shape=list(value.shape), dtype=str(value.dtype))
                    for name, value in batch.model_inputs.items()
                },
                text_targets=len(targets),
                decoded_targets=processor.tokenizer.decode(targets, skip_special_tokens=False),
                media_info=batch.media_info,
            )
        )
    # This covers different modalities and lengths in one padded CPU batch too.
    mixed = collator([dataset[i] for i in range(min(8, len(dataset)))])
    report = dict(
        model_id=config["model_id"],
        revision=config["revision"],
        versions={
            name: version(name)
            for name in (
                "transformers",
                "torch",
                "torchvision",
                "av",
                "librosa",
                "pillow",
                "soundfile",
            )
        },
        manifest=str(manifest),
        manifest_sha256=sha256(manifest),
        label_contract="Already shifted once; only text and EOS targets, media/padding=-100",
        samples=reports,
        mixed_sample_ids=mixed.sample_ids,
        mixed_shape=list(mixed.model_inputs["input_ids"].shape),
        mixed_target_counts=(mixed.labels != -100).sum(-1).tolist(),
        native_training_executed=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/datasets/omni-multimodal-smoke")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Use existing local media/text pairs instead of downloading the sample corpus",
    )
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    snapshot = snapshot_download(
        config["model_id"],
        revision=config["revision"],
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ],
    )
    manifest = args.manifest if args.manifest is not None else prepare(args.output, config)
    report = check(manifest, config, snapshot, args.output)
    print(
        json.dumps(
            dict(
                samples=len(report["samples"]),
                mixed_shape=report["mixed_shape"],
                target_counts=report["mixed_target_counts"],
                report=str(args.output / "batch-report.json"),
            ),
            indent=2,
        )
    )
