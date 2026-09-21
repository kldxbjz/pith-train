"""Offline CPU/FP32 reference for the Qwen3-Omni Thinker text training path.

Run: python -m tools.qwen3_omni_reference --output workspace/omni-reference

Uses Transformers' actual Thinker decoder plus its untied, bias-free LM head.
This does not exercise PithTrain, media encoders, DeepStack, or the Talker.
The tiny config preserves GQA, Q/K normalization, routed SwiGLU experts,
head_dim=128, and interleaved MRoPE sections [24, 20, 20] from the checkpoint
below. Width, depth, vocabulary, expert count/top-k, and context are reduced.
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


def build_reference(seed: int = 0) -> nn.ModuleDict:
    """Construct the HF text decoder and head without allocating the media encoders."""
    config = Qwen3OmniMoeTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        moe_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        use_qk_norm=True,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "mrope_section": [24, 20, 20],
            "interleaved": True,
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "eager"
    # Do not change the caller's RNG state or depend on a default GPU device.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(seed)
        decoder = Qwen3OmniMoeThinkerTextModel(config).float()
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        nn.init.normal_(head.weight, std=config.initializer_range)
    return nn.ModuleDict({"model": decoder, "lm_head": head})


def make_tokens(seed: int = 0) -> torch.Tensor:
    """Two sequences with 16 inputs plus one final next-token target each."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return torch.randint(0, 256, (2, 17), generator=generator, device="cpu")


def forward_logits(model: nn.ModuleDict, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model["model"](input_ids=input_ids, use_cache=False).last_hidden_state
    return model["lm_head"](hidden)


def run_reference(seed: int = 0) -> dict:
    model = build_reference(seed)
    model.train()
    tokens = make_tokens(seed)
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_reference(args.seed)
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
