"""Offline HF reference for the Qwen3-Omni Thinker text decoder and LM head.

Run the tiny CPU/FP32 forward and backward reference:
    python -m tools.qwen3_omni_reference --config tiny --output workspace/omni-reference
Inspect the full text model's configuration and shapes without allocating weights:
    python -m tools.qwen3_omni_reference --config full --describe

Both presets use the same HF model construction. The JSON files contain the tiny
and full Thinker text configurations; full preserves the checkpoint's dimensions.
RoPE uses Transformers' normalized rope_parameters format. Caching and router
outputs are disabled for this next-token CE reference. Full execution is deferred;
--describe constructs meta tensors only, with no forward/backward or weight load.
This does not exercise PithTrain, media encoders, DeepStack, or the Talker.
No model weights, tokenizer, dataset, or network connection are required.
Validated with Transformers 5.17.0 and PyTorch 2.13.0.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CONFIG_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def load_config(name: str = "tiny") -> Qwen3OmniMoeTextConfig:
    """Read a local text-config preset; no Hub access or model allocation."""
    if name not in ("tiny", "full"):
        raise ValueError(f"Unknown reference config: {name}")
    path = Path(__file__).with_name("qwen3_omni_configs") / f"{name}.json"
    config = Qwen3OmniMoeTextConfig(**json.loads(path.read_text()))
    config._attn_implementation = "eager"
    return config


def build_reference(
    config: Qwen3OmniMoeTextConfig, seed: int = 0, *, device: str = "cpu"
) -> nn.ModuleDict:
    """Construct the decoder/head on CPU, or on meta for shape-only inspection."""
    # Do not change the caller's RNG state or depend on a default GPU device.
    with torch.random.fork_rng(devices=[]), torch.device(device):
        torch.manual_seed(seed)
        decoder = Qwen3OmniMoeThinkerTextModel(config).float()
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        nn.init.normal_(head.weight, std=config.initializer_range)
    return nn.ModuleDict({"model": decoder, "lm_head": head})


def make_tokens(config: Qwen3OmniMoeTextConfig, seed: int = 0) -> torch.Tensor:
    """Two sequences with 16 inputs plus one final next-token target each."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return torch.randint(0, config.vocab_size, (2, 17), generator=generator, device="cpu")


def forward_logits(model: nn.ModuleDict, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model["model"](input_ids=input_ids, use_cache=False).last_hidden_state
    return model["lm_head"](hidden)


def run_reference(config: Qwen3OmniMoeTextConfig, seed: int = 0) -> dict:
    model = build_reference(config, seed)
    model.train()
    tokens = make_tokens(config, seed)
    # Match MemmapDataset: targets are shifted once, before the objective.
    input_ids, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    logits = forward_logits(model, input_ids)
    # Pure next-token CE; no auxiliary router loss in this initial comparison.
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()

    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite reference output or loss")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
        if not torch.count_nonzero(parameter.grad):
            raise RuntimeError(f"Entire parameter has zero gradient: {name}")
        gradients[name] = parameter.grad.detach().clone()

    return {
        "metadata": {
            "model_id": MODEL_ID,
            "config_revision": CONFIG_REVISION,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "seed": seed,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "loss": "next-token cross-entropy, mean over targets, no router auxiliary loss",
            "scope": "HF Thinker text decoder and LM head only; no PithTrain or media path",
        },
        "config": model["model"].config.to_dict(),
        "input_ids": input_ids,
        "labels": labels,
        "logits": logits.detach(),
        "loss": loss.detach(),
        "state_dict": {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        },
        "gradients": gradients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--describe", action="store_true", help="Print config/shapes on meta only")
    parser.add_argument("--output", type=Path, help="Output directory for the tiny reference")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.describe:
        if args.config == "full":
            parser.error("full execution is not implemented; use --config full --describe")
        if args.output is None:
            parser.error("--output is required when running the tiny reference")
    torch.set_num_threads(1)
    config = load_config(args.config)
    if args.describe:
        model = build_reference(config, args.seed, device="meta")
        description = {
            "model_id": MODEL_ID,
            "config_revision": CONFIG_REVISION,
            "preset": args.config,
            "scope": "Thinker text decoder and LM head only; shapes, not execution",
            "config": config.to_dict(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_shapes": {name: list(p.shape) for name, p in model.named_parameters()},
        }
        print(json.dumps(description, indent=2))
        return
    result = run_reference(config, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output / "reference.pt")
    summary = {
        **result["metadata"],
        "loss_value": result["loss"].item(),
        "logits_shape": list(result["logits"].shape),
        "target_tokens": result["labels"].numel(),
        "parameter_count": sum(t.numel() for t in result["state_dict"].values()),
        "parameter_shapes": {name: list(t.shape) for name, t in result["state_dict"].items()},
        "gradient_norms": {name: grad.norm().item() for name, grad in result["gradients"].items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "loss": summary["loss_value"], "status": "PASS"}))


if __name__ == "__main__":
    main()
