"""Offline HF checks for the Qwen3-Omni Thinker text decoder and LM head.

Run: python -m pytest tests/test_qwen3_omni_reference.py

The tiny fixture runs random-weight CPU/FP32 forward and backward; full is built
on meta for shape checks only. Both fixtures describe text model size, not media
support. RoPE uses normalized rope_parameters; runtime choices (eager attention,
no cache/router outputs) are applied by the helpers, leaving the fixtures intact.
Replay artifacts live in pytest's tmp_path. No Hub access, released weights,
tokenizer or dataset is required. Native PithTrain and DualPipeV comparisons will
be added when the native Omni model is implemented.
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


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("vocab_size", [17, 256])
def test_hf_label_shift_matches_pretraining(seed, vocab_size):
    """HF shifts inside its loss; PithTrain's dataset already shifts the targets."""
    config = load_config("tiny")
    config.vocab_size = vocab_size
    model = build_reference(config, seed)
    tokens = make_tokens(config, seed)
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


def test_reference_artifact_can_be_replayed(tmp_path):
    config = load_config()
    result = run_reference(config, make_tokens(config), seed=0)
    path = tmp_path / "reference.pt"
    torch.save(result, path)
    saved = torch.load(path, weights_only=True)

    # A different random initialization must reproduce the saved reference after loading.
    model = build_reference(load_config(), seed=1)
    model.load_state_dict(saved["state_dict"], strict=True)
    logits = forward_logits(model, saved["input_ids"])
    loss = F.cross_entropy(logits.flatten(0, 1), saved["labels"].flatten())
    loss.backward()
    torch.testing.assert_close(logits, saved["logits"], rtol=0, atol=0)
    torch.testing.assert_close(loss, saved["loss"], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, saved["gradients"][name], rtol=0, atol=0)


def test_reference_is_deterministic():
    config = load_config()
    tokens = make_tokens(config)
    first = run_reference(config, tokens, seed=0)
    second = run_reference(config, tokens, seed=0)
    for key in ("input_ids", "labels", "logits", "loss"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in ("state_dict", "gradients"):
        for name in first[key]:
            torch.testing.assert_close(first[key][name], second[key][name], rtol=0, atol=0)


def test_reference_uses_caller_batch_without_changing_config():
    config = load_config()
    assert config.use_cache
    config._attn_implementation = "sdpa"
    config.output_router_logits = True
    original_config = config.to_dict()
    tokens = make_tokens(config, seed=17, batch_size=3, sequence_length=7)
    original_tokens = tokens.clone()

    result = run_reference(config, tokens)

    torch.testing.assert_close(result["input_ids"], original_tokens[:, :-1])
    torch.testing.assert_close(result["labels"], original_tokens[:, 1:])
    torch.testing.assert_close(tokens, original_tokens)
    assert result["logits"].shape == (3, 7, config.vocab_size)
    assert config.to_dict() == original_config
    assert config._attn_implementation == "sdpa"
    assert result["metadata"]["attention"] == "eager"
    assert not result["metadata"]["use_cache"]
    assert not result["metadata"]["output_router_logits"]


def test_reference_honors_tied_embeddings():
    config = load_config()
    config.tie_word_embeddings = True
    model = build_reference(config)
    assert model["lm_head"].weight is model["model"].embed_tokens.weight


@pytest.mark.parametrize(
    "preset, layers, vocab, hidden, query_width, experts, expert_width",
    [("tiny", 2, 256, 128, 256, 4, 64), ("full", 48, 152064, 2048, 4096, 128, 768)],
)
def test_preset_model_shapes_on_meta(
    preset, layers, vocab, hidden, query_width, experts, expert_width
):
    """Exercise both actual HF constructors without allocating full-size weights."""
    model = build_reference(load_config(preset), device="meta")
    assert all(t.is_meta for t in model.parameters())
    assert all(t.is_meta for t in model.buffers())
    assert len(model["model"].layers) == layers
    parameters = dict(model.named_parameters())
    assert parameters["model.embed_tokens.weight"].shape == (vocab, hidden)
    assert parameters["lm_head.weight"].shape == (vocab, hidden)
    assert parameters["model.layers.0.self_attn.q_proj.weight"].shape == (query_width, hidden)
    assert parameters["model.layers.0.mlp.experts.gate_up_proj"].shape == (
        experts,
        2 * expert_width,
        hidden,
    )
    assert parameters["model.layers.0.mlp.experts.down_proj"].shape == (
        experts,
        hidden,
        expert_width,
    )
