"""Check the offline Omni reference before using it to validate a PithTrain port."""

import pytest
import torch
import torch.nn.functional as F
from transformers.loss.loss_utils import ForCausalLMLoss

from tools.qwen3_omni_reference import (
    build_reference,
    forward_logits,
    load_config,
    make_tokens,
    run_reference,
)


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
    result = run_reference(load_config(), seed=0)
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
    first = run_reference(load_config(), seed=0)
    second = run_reference(load_config(), seed=0)
    for key in ("input_ids", "labels", "logits", "loss"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in ("state_dict", "gradients"):
        for name in first[key]:
            torch.testing.assert_close(first[key][name], second[key][name], rtol=0, atol=0)


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
