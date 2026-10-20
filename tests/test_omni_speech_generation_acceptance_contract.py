"""Generation-gate contracts and real CPU HF codec controls, not native GPU proof."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.acceptance import omni_speech_generation as gate


class MethodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("rotary", torch.tensor([1.0, 0.1]))

    def generate(self, value):
        assert self.linear.weight.dtype == torch.bfloat16
        assert self.rotary.dtype == torch.float32
        return self.linear(value)


def test_functional_generate_dispatch_preserves_master_and_rotary_buffers():
    model = MethodModel()
    before = gate.state_hash(model)
    ids = [id(p) for p in model.parameters()]
    result = gate.compute(
        model, lambda module, value: module.generate(value), torch.ones(1, 2, dtype=torch.bfloat16)
    )
    assert result.dtype == torch.bfloat16
    assert gate.state_hash(model) == before
    assert [id(p) for p in model.parameters()] == ids
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_control_restores_rng_modes_and_mutable_cache_even_after_exception():
    from pithtrain.tasks.omni_speech import _rng_state

    model = MethodModel()
    model.linear.eval()
    model.rope_deltas = torch.tensor([3])
    rng, modes = _rng_state(), [m.training for m in model.modules()]
    with pytest.raises(RuntimeError, match="control stopped"):
        with gate.preserve_generation_state(model) as proof:
            model.rope_deltas = torch.tensor([9])
            torch.rand(13)
            raise RuntimeError("control stopped")
    gate.chain.exact(_rng_state(), rng)
    assert [m.training for m in model.modules()] == modes
    assert model.rope_deltas.tolist() == [3]
    assert all(proof.values())


def result():
    speech = SimpleNamespace(
        codes=torch.zeros(1, 2, 16, dtype=torch.long),
        code_lengths=torch.tensor([2]),
        reached_eos=torch.tensor([False]),
        waveforms=[torch.ones(1, 10)],
    )
    return SimpleNamespace(
        sequences=torch.tensor([[1, 2, 3, 4]]),
        text_reached_eos=False,
        speech=speech,
        logits=[torch.tensor([[1.0, 2.0]])],
    )


@pytest.mark.parametrize(
    "damage", ["text", "code", "length", "eos", "wave", "nan", "logits", "missing"]
)
def test_comparison_rejects_each_real_output_axis(damage):
    actual, expected = result(), result()
    rows = [expected.logits[0].clone()]
    if damage == "text":
        actual.sequences[0, -1] = 7
    elif damage == "code":
        actual.speech.codes[0, 0, 15] = 1
    elif damage == "length":
        actual.speech.code_lengths[0] = 1
    elif damage == "eos":
        actual.speech.reached_eos[0] = True
    elif damage == "wave":
        actual.speech.waveforms[0].zero_()
    elif damage == "nan":
        actual.speech.waveforms[0][0, 0] = float("nan")
    elif damage == "logits":
        rows[0].neg_()
    elif damage == "missing":
        actual.speech = None
    with pytest.raises(AssertionError):
        gate.compare_generation(actual, rows, expected, 2)


def codec_pair():
    from tests.test_omni_speech_integration import condition, pair
    from tests.test_qwen3_omni_code2wav import paired, tiny_config

    talker, hf_talker = pair()
    codec, hf_codec = paired(tiny_config(num_quantizers=16, codebook_size=23))

    class Components(nn.Module):
        def __init__(self, talker, codec):
            super().__init__()
            self.talker, self.code2wav = talker, codec
            self.config = SimpleNamespace(talker_config=talker.config)

    torch.manual_seed(71)
    prepared = condition(talker, 1)
    values = dict(
        inputs_embeds=prepared.inputs_embeds,
        talker_input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        trailing_text_hidden=prepared.trailing_text_hidden,
        tts_pad_embed=prepared.tts_pad_embed,
    )
    values = {
        name: value.detach().to(torch.bfloat16) if value.is_floating_point() else value
        for name, value in values.items()
    }
    return Components(talker, codec), Components(hf_talker, hf_codec), values


def test_actual_hf_cached_ragged_codec_and_waveform_control_cpu():
    torch.set_num_threads(1)
    native, hf, prepared = codec_pair()
    assert "forward" not in native.talker.__dict__
    with torch.no_grad(), gate.preserve_generation_state(native, hf) as preserved:
        report = gate.ragged_control(native, hf, prepared, {"position_kwargs": {}})
    assert "forward" not in native.talker.__dict__
    assert report["scope"] == "controlled_codec_eos_boundary"
    assert report["code_lengths"] == [0, 2]
    assert report["reached_eos"] == [True, True]
    assert report["waveforms"][0]["samples"] == 0
    assert report["waveforms"][1]["samples"] > 0
    assert all(wave["nse"] < 1e-3 for wave in report["waveforms"])
    assert all(preserved.values())


def test_unforced_generation_and_control_paths_are_separate_and_bounded():
    path = Path(gate.__file__)
    tree = ast.parse(path.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    native_source = ast.unparse(functions["native_generate"])
    assert "model.generate" in native_source and "original(*args, **kwargs)" in native_source
    assert "logits_processor" not in native_source and "controlled" not in native_source
    hf_source = ast.unparse(functions["hf_codec"])
    assert "model.talker.generate" in hf_source and "model.code2wav.chunked_decode" in hf_source
    validate = ast.unparse(functions["validate"])
    assert "for sample in samples" in validate and "set(cases) != set(MEDIA)" in validate
    assert "previous.logits[0].argmax" in validate
    assert (gate.TEXT_TOKENS, gate.CODEC_FRAMES, gate.OUTPUT_NSE) == (2, 3, 1e-3)
    assert gate.MEDIA == ("text", "image", "audio", "video", "mixed")
    assert "source_identity(args.expected_commit)" in ast.unparse(functions["main"])
    assert "runtime_precondition(args)" in ast.unparse(functions["main"])


def test_summary_is_complete_and_compact_without_raw_tensor_payloads():
    actual = result()
    report = gate.compare_generation(actual, actual.logits, copy.deepcopy(actual), 2)
    assert report["status"] == "PASS" and report["code_groups"] == 16
    assert report["generated_tokens"] == [[3, 4]]
    import json

    encoded = json.dumps(report, allow_nan=False).encode()
    assert len(encoded) < 4096
    assert report["thinker_logits_nse"] == [0.0]


@pytest.mark.parametrize("existing", [False, True])
def test_hf_position_metadata_restored_on_exception(existing):
    owner = SimpleNamespace()
    prior = SimpleNamespace(original=True)
    if existing:
        owner.audio_config = prior
    hf = SimpleNamespace(talker=SimpleNamespace(config=owner))
    config = SimpleNamespace(thinker_config=SimpleNamespace(audio_config={"official": 7}))
    with pytest.raises(RuntimeError, match="stop"):
        with gate.hf_position_metadata(hf, config):
            assert owner.audio_config == {"official": 7}
            assert owner.audio_config is not config.thinker_config.audio_config
            raise RuntimeError("stop")
    assert hasattr(owner, "audio_config") is existing
    if existing:
        assert owner.audio_config is prior


@pytest.mark.parametrize("fail", [False, True])
def test_native_logit_observer_restores_method_without_instance_override(fail):
    class Thinker(nn.Module):
        def forward_epilog(self, value):
            return value

    class ProtocolModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.thinker = Thinker()
            self.value = torch.ones(1, 2, 3)

        def generate(self, *args, **kwargs):
            observed = self.thinker.forward_epilog(self.value)
            assert observed is self.value
            if fail:
                raise RuntimeError("generation stopped")
            return observed

    model = ProtocolModel()
    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "model_context": {},
        "position_kwargs": {},
    }
    if fail:
        with pytest.raises(RuntimeError, match="generation stopped"):
            gate.native_generate(model, batch)
    else:
        output, rows = gate.native_generate(model, batch)
        assert output is model.value
        assert torch.equal(rows[0], model.value[:, -1])
    assert "forward_epilog" not in model.thinker.__dict__


def test_preservation_removes_generation_cache_created_during_control():
    model = MethodModel()
    assert not hasattr(model, "rope_deltas")
    with gate.preserve_generation_state(model) as proof:
        model.rope_deltas = torch.tensor([3])
    assert not hasattr(model, "rope_deltas")
    assert all(proof.values())
