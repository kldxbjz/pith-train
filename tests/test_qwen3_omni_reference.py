"""Offline HF baseline for the Qwen3-Omni Thinker text decoder and LM head.

Run: python -m pytest tests/test_qwen3_omni_reference.py -q -rs

Three active checks cover tiny CPU/FP32 forward/backward, next-token label shifting,
and full-config shapes on meta. Inputs and weights are synthetic; no Hub access,
released weights, tokenizer or dataset is required. Both configs are text-only.
The skipped test_native_omni_matches_hf outlines how to reuse this baseline once
the native Omni model and CUDA reference support exist. Distributed validation
belongs in the shared tests/test_dualpipev.py harness.
Validated with Transformers 5.17.0 and PyTorch 2.13.0.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CONFIG_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def load_config(name: str = "tiny") -> Qwen3OmniMoeTextConfig:
    """Read only the text config; apply execution choices in the reference itself."""
    if name not in ("tiny", "full"):
        raise ValueError(f"Unknown reference config: {name}")
    path = Path(__file__).parent / "configs" / "qwen3_omni_text" / f"{name}.json"
    return Qwen3OmniMoeTextConfig(**json.loads(path.read_text()))


def build_reference(
    config: Qwen3OmniMoeTextConfig, seed: int = 0, *, device: str = "cpu"
) -> nn.ModuleDict:
    """Construct the decoder/head on CPU, or on meta for shape-only inspection."""
    # TODO: Add CUDA/dtype support to construction, input placement and run_reference
    # metadata, preserving CUDA RNG state. Compare GPU FP32/BF16 with the CPU baseline
    # using identical weights/inputs and dtype-appropriate tolerances.
    if device not in ("cpu", "meta"):
        raise ValueError("The reference currently supports only cpu execution or meta inspection")
    # HF construction sets runtime fields; keep the caller's architecture config reusable.
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    # Do not change the caller's RNG state or depend on a default GPU device.
    with torch.random.fork_rng(devices=[]), torch.device(device):
        torch.manual_seed(seed)
        decoder = Qwen3OmniMoeThinkerTextModel(config).float()
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        nn.init.normal_(head.weight, std=config.initializer_range)
        if config.tie_word_embeddings:
            head.weight = decoder.embed_tokens.weight
    return nn.ModuleDict({"model": decoder, "lm_head": head})


def make_tokens(
    config: Qwen3OmniMoeTextConfig,
    seed: int = 0,
    *,
    batch_size: int = 2,
    sequence_length: int = 16,
) -> torch.Tensor:
    """Synthetic text batch with one extra token for next-token targets."""
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return torch.randint(
        0, config.vocab_size, (batch_size, sequence_length + 1), generator=generator, device="cpu"
    )


def forward_logits(model: nn.ModuleDict, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model["model"](
        input_ids=input_ids, use_cache=False, output_router_logits=False, return_dict=True
    ).last_hidden_state
    return model["lm_head"](hidden)


def run_reference(config: Qwen3OmniMoeTextConfig, tokens: torch.Tensor, seed: int = 0) -> dict:
    """Run CPU/FP32 next-token CE on a caller-provided [batch, sequence + 1] text batch."""
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape [batch >= 1, sequence + 1 >= 2]")
    if tokens.device.type != "cpu" or tokens.dtype != torch.long:
        raise ValueError("tokens must be a CPU torch.long tensor")
    model = build_reference(config, seed)
    model.train()
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
            "use_cache": False,
            "output_router_logits": False,
            "initialization": "random",
            "config_scope": "thinker_config.text_config",
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


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_tiny_reference_forward_backward():
    """Run the baseline on a caller-provided batch without changing its inputs/config."""
    config = load_config("tiny")
    config._attn_implementation = "sdpa"
    config.output_router_logits = True
    original_config = copy.deepcopy(config.to_dict())
    tokens = make_tokens(config, seed=17, batch_size=3, sequence_length=7)
    original_tokens = tokens.clone()

    # run_reference checks that logits/loss and every parameter gradient are finite,
    # and that no entire parameter has a missing or all-zero gradient.
    result = run_reference(config, tokens)

    torch.testing.assert_close(result["input_ids"], original_tokens[:, :-1])
    torch.testing.assert_close(result["labels"], original_tokens[:, 1:])
    torch.testing.assert_close(tokens, original_tokens)
    assert result["logits"].shape == (3, 7, config.vocab_size)
    assert config.to_dict() == original_config
    assert config._attn_implementation == "sdpa"


def test_hf_label_shift_matches_pretraining():
    """HF shifts inside its loss; PithTrain's dataset already shifts the targets."""
    config = load_config("tiny")
    model = build_reference(config)
    tokens = make_tokens(config)
    full_logits = forward_logits(model, tokens)
    hf_loss = ForCausalLMLoss(full_logits, tokens, vocab_size=config.vocab_size)

    # An independently executed, shorter causal forward must have the same prefix.
    logits = forward_logits(model, tokens[:, :-1])
    torch.testing.assert_close(logits, full_logits[:, :-1], rtol=1e-5, atol=1e-6)
    pretraining_loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(pretraining_loss, hf_loss, rtol=1e-6, atol=1e-6)

    parameters = dict(model.named_parameters())
    hf_gradients = torch.autograd.grad(hf_loss, tuple(parameters.values()))
    pretraining_gradients = torch.autograd.grad(pretraining_loss, tuple(parameters.values()))
    for name, expected, actual in zip(parameters, hf_gradients, pretraining_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7, msg=name)


def test_full_model_shapes_on_meta():
    """Exercise the full HF constructor without allocating full-size weights."""
    # TODO: Add an opt-in full-size GPU forward/backward test when the execution path
    # and hardware are ready. Keep this meta test for fast, allocation-free shape checks.
    model = build_reference(load_config("full"), device="meta")
    assert all(t.is_meta for t in model.parameters())
    assert all(t.is_meta for t in model.buffers())
    assert len(model["model"].layers) == 48
    parameters = dict(model.named_parameters())
    assert parameters["model.embed_tokens.weight"].shape == (152064, 2048)
    assert parameters["lm_head.weight"].shape == (152064, 2048)
    assert parameters["model.layers.0.self_attn.q_proj.weight"].shape == (4096, 2048)
    assert parameters["model.layers.0.mlp.experts.gate_up_proj"].shape == (128, 1536, 2048)
    assert parameters["model.layers.0.mlp.experts.down_proj"].shape == (128, 2048, 768)


@pytest.mark.skip(reason="TODO: native Omni and CUDA reference support are not implemented")
def test_native_omni_matches_hf():
    """Replace this placeholder with a single-GPU HF-vs-native correctness test."""
    # 1. Extend the helpers above for CUDA/dtype. Reuse load_config("tiny") and
    #    make_tokens(); run_reference supplies inputs, labels, weights and HF results.
    # 2. Set up the native backend with PP=EP=CP=1 and build the whole model (phase=-1).
    #    Copy/map the HF state_dict, including expert layouts; matching seeds is not enough.
    #    Both models must use the same weights/dtype, positions and loss settings:
    #    no cache, no router auxiliary loss, labels shifted once, mean next-token CE.
    # 3. Run native.reference_forward on the reference input_ids, calculate CE with
    #    the reference labels, and backward. Recompute HF results at the chosen dtype.
    # 4. Compare with dtype-appropriate tolerances (comparison sketch):
    #    torch.testing.assert_close(native_logits, expected["logits"], rtol=rtol, atol=atol)
    #    torch.testing.assert_close(native_loss, expected["loss"], rtol=rtol, atol=atol)
    #    Map native gradients back to HF names/layouts and require complete key coverage:
    #    assert native_gradients.keys() == expected["gradients"].keys()
    #    for name, expected_grad in expected["gradients"].items():
    #        torch.testing.assert_close(native_gradients[name], expected_grad,
    #                                   rtol=rtol, atol=atol, msg=name)
    # 5. Add Omni to tests/test_dualpipev.py and tests/test_dualpipev.sh for separate
    #    native-reference-vs-pipeline checks across PP/EP/CP layouts.
    raise NotImplementedError("Implement native Omni construction and weight/gradient mapping")
