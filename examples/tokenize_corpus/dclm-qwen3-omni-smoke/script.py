"""Prepare a pinned, small DCLM corpus with the Omni Thinker text tokenizer."""

import argparse
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import zstandard as zstd
from huggingface_hub import hf_hub_url, snapshot_download

from pithtrain.tasks.tokenize_corpus import Worker, Writer, read_file


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict:
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    source, tokenizer_config = config["dataset"], config["tokenizer"]
    raw = output / "raw" / "train.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        url = hf_hub_url(
            source["repo_id"], source["filename"], repo_type="dataset", revision=source["revision"]
        )
        # Only a compressed prefix is needed; never download the complete 134 MB shard.
        request = Request(url, headers={"Range": "bytes=0-4194303"})
        rows = []
        with urlopen(request, timeout=45) as response:
            with zstd.ZstdDecompressor().stream_reader(response) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                    for _ in range(source["num_documents"]):
                        text = json.loads(next(stream))["text"]
                        assert isinstance(text, str) and text.strip(), "Empty/non-text document"
                        rows.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        content = "".join(rows).encode("utf-8")
        assert hashlib.sha256(content).hexdigest() == source["text_sha256"], "Source changed"
        raw.write_bytes(content)
    assert sha256(raw) == source["text_sha256"], "Cached text does not match the pinned subset"

    snapshot = Path(
        snapshot_download(
            tokenizer_config["repo_id"],
            revision=tokenizer_config["revision"],
            allow_patterns=["config.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        )
    )
    model_config = json.loads((snapshot / "config.json").read_text())
    vocab_size = model_config["thinker_config"]["text_config"]["vocab_size"]
    Worker(str(snapshot))
    assert Worker.tokenizer.vocab_size > 150_000, "Incomplete Omni tokenizer vocabulary"
    assert Worker.tokenizer.eos_token == tokenizer_config["eos_token"]
    assert Worker.tokenizer.eos_token_id == tokenizer_config["eos_token_id"]
    assert max(Worker.tokenizer.get_vocab().values()) < vocab_size

    documents = list(read_file(raw))
    assert len(documents) == source["num_documents"]
    token_dir = output / "tokens"
    token_dir.mkdir(exist_ok=True)
    shard_size = source["documents_per_shard"]
    shard_count = (len(documents) + shard_size - 1) // shard_size
    expected_paths = {token_dir / f"train-{i:03d}.bin" for i in range(shard_count)}
    assert not (set(token_dir.rglob("*.bin")) - expected_paths), "Unexpected shards in output"
    shard_reports = []
    for shard_id, start in enumerate(range(0, len(documents), shard_size)):
        path = token_dir / f"train-{shard_id:03d}.bin"
        writer = Writer(path)
        # Keep document order fixed; multiprocessing's imap_unordered is unnecessary here.
        for text in documents[start : start + shard_size]:
            tokens, _ = Worker.encode(text)
            assert len(tokens) > 1 and tokens[-1] == tokenizer_config["eos_token_id"]
            assert tokens.dtype == np.uint32 and tokens.max() < vocab_size
            writer.append(tokens)
        writer.flush()
        with path.open("rb") as stream:
            tokens, ends = np.load(stream), np.load(stream)
        shard_reports.append(
            dict(
                file=path.name,
                sha256=sha256(path),
                documents=len(ends),
                tokens=len(tokens),
                samples=(len(tokens) - 1) // config["pretraining"]["sequence_length"],
                min_token_id=int(tokens.min()),
                max_token_id=int(tokens.max()),
            )
        )
    samples = sum(shard["samples"] for shard in shard_reports)
    required = config["pretraining"]["max_steps"] * config["pretraining"]["global_batch_size"]
    assert samples >= required, f"Corpus has {samples} samples; the smoke run requires {required}"
    report = dict(
        **config,
        model_vocab_size=vocab_size,
        storage_dtype="uint32",
        samples=samples,
        shards=shard_reports,
        versions={name: version(name) for name in ("transformers", "tokenizers", "numpy")},
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("workspace/datasets/dclm-omni-smoke"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
