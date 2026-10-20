"""Bounded native Thinker -> 16-group Talker -> Code2Wav GPU generation parity.

Uses the source-bound tiny speech_chain fixture, official HF 5.17 component
forwards/helpers and cached Talker.generate as an independent reference. Five
unforced greedy media cases are distinct from explicit EOS/ragged controls.
No downloads, released-weight quality, sampling-quality or FSDP generation claim.
"""

import argparse
import copy
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.func import functional_call

from tests.acceptance import omni_speech_chain as chain

SCHEMA = "qwen3_omni_generation_acceptance_v1"
TEXT_TOKENS = 2
CODEC_FRAMES = 3
OUTPUT_NSE = chain.OUTPUT_NSE
MEDIA = chain.MEDIA


class MethodCall(nn.Module):
    """Functional-call dispatch reaches methods without converting RoPE buffers."""

    def __init__(self, model, method):
        super().__init__()
        self.model, self.method = model, method

    def forward(self, *args, **kwargs):
        return self.method(self.model, *args, **kwargs)


def compute(model, method, *args, **kwargs):
    wrapper = MethodCall(model, method)
    parameters = {name: p.to(torch.bfloat16) for name, p in wrapper.named_parameters()}
    return functional_call(
        wrapper, (parameters, dict(wrapper.named_buffers())), args, kwargs, strict=True
    )


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


@contextmanager
def preserve_generation_state(*models):
    from pithtrain.tasks.omni_speech import _rng_state, _set_rng

    rng = _rng_state()
    modes = [(module, module.training) for model in models for module in model.modules()]
    mutable = [
        (module, name, hasattr(module, name), copy.deepcopy(getattr(module, name, None)))
        for module, _ in modes
        for name in ("rope_deltas", "generation_config")
    ]
    before = [state_hash(model) for model in models]
    identities = [[(name, id(p)) for name, p in model.named_parameters()] for model in models]
    proof = {}
    try:
        for model in models:
            model.eval()
        yield proof
    finally:
        for module, name, existed, value in mutable:
            if existed:
                setattr(module, name, value)
            elif hasattr(module, name):
                delattr(module, name)
        for module, mode in modes:
            module.training = mode
        _set_rng(rng)
        proof.update(
            weights_unchanged=before == [state_hash(model) for model in models],
            parameter_identities_unchanged=identities
            == [[(name, id(p)) for name, p in model.named_parameters()] for model in models],
            modes_restored=all(module.training == mode for module, mode in modes),
            rng_restored=True,
        )
        chain.exact(_rng_state(), rng, "generation-rng")
        if not all(proof.values()):
            raise AssertionError("Generation/control changed model state or parameter identities")


def prompt_sample(sample):
    end = sample["prompt_lengths"][0]
    ids = sample["input_ids"][:, :end].clone()
    context = dict(sample["model_context"], input_ids=ids, attention_mask=torch.ones_like(ids))
    context.pop("position_ids", None)
    return dict(
        input_ids=ids,
        model_context=context,
        prompt_lengths=[end],
        position_kwargs=sample["position_kwargs"],
    )


def hf_features(model, sequence, context):
    from tests.acceptance.omni_multimodal_training import positions
    from tests.acceptance.omni_reference_attention import reference_attention_kwargs

    inputs = dict(context, input_ids=sequence, attention_mask=torch.ones_like(sequence))
    inputs.update(
        position_ids=positions(model.config.thinker_config, inputs),
        use_cache=False,
        output_hidden_states=True,
        **reference_attention_kwargs(model.thinker),
    )
    return model.thinker(**inputs)


def hf_condition(model, sequence, output, batch):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as HFComposite,
    )

    config = model.config
    embed, hidden = (
        output.hidden_states[0],
        output.hidden_states[config.talker_config.accept_hidden_layer],
    )
    specials = sequence.new_tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]]
    )
    bos, eos, pad = model.talker.text_projection(
        model.thinker.get_input_embeddings()(specials)
    ).chunk(3, 1)
    multimodal = sum(
        (sequence == getattr(config.thinker_config, name)).int()
        for name in ("image_token_id", "video_token_id", "audio_token_id")
    ).bool()
    start = batch["prompt_lengths"][0] - 3
    user = HFComposite._get_talker_user_parts(model, 0, start, multimodal, hidden, embed)
    assistant, assistant_ids, tail = HFComposite._get_talker_assistant_parts(
        model,
        start,
        sequence.shape[1],
        config.talker_config.speaker_id["ethan"],
        embed,
        pad,
        bos,
        eos,
    )
    ids = torch.cat((sequence[:, :start], assistant_ids), 1)
    return dict(
        inputs_embeds=torch.cat((user, assistant), 1),
        talker_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        trailing_text_hidden=tail,
        tts_pad_embed=pad,
    )


def position_kwargs(batch):
    names = {"audio_seqlens": "audio_feature_lengths", "second_per_grids": "video_second_per_grid"}
    return {names.get(key, key): value for key, value in batch["position_kwargs"].items()}


def hf_codec(model, condition, batch, *, max_frames=CODEC_FRAMES, logits_processor=None):
    options = dict(
        condition,
        max_new_tokens=max_frames + 1,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=model.config.talker_config.codec_eos_token_id,
        pad_token_id=model.config.talker_config.codec_pad_id,
        suppress_tokens=[
            i
            for i in range(
                model.config.talker_config.code_predictor_config.vocab_size,
                model.config.talker_config.text_config.vocab_size,
            )
            if i != model.config.talker_config.codec_eos_token_id
        ],
        output_hidden_states=True,
        return_dict_in_generate=True,
        **position_kwargs(batch),
    )
    if logits_processor is not None:
        options["logits_processor"] = logits_processor
    generated = model.talker.generate(**options)
    frames = [hidden[-1] for hidden in generated.hidden_states if hidden[-1] is not None]
    rows = condition["inputs_embeds"].shape[0]
    codes = (
        torch.stack(frames, 1) if frames else condition["talker_input_ids"].new_empty((rows, 0, 16))
    )
    first = generated.sequences[:, :max_frames]
    eos_mask = first == model.config.talker_config.codec_eos_token_id
    reached = eos_mask.any(-1)
    positions = torch.arange(first.shape[1], device=first.device)[None].expand_as(first)
    lengths = torch.where(eos_mask, positions, codes.shape[1]).min(-1).values
    if int(lengths.max()) > max_frames:
        raise AssertionError("Official generation exceeded the reviewed codec cap")
    valid = torch.arange(codes.shape[1], device=codes.device)[None] < lengths[:, None]
    codes = codes.masked_fill(~valid[..., None], 0)
    waves = []
    for row, length in enumerate(lengths.tolist()):
        waves.append(
            model.code2wav.chunked_decode(
                codes[row : row + 1, :length].transpose(1, 2), chunk_size=300, left_context_size=25
            )[0].float()
            if length
            else condition["inputs_embeds"].new_empty((1, 0), dtype=torch.float32)
        )
    return SimpleNamespace(codes=codes, code_lengths=lengths, reached_eos=reached, waveforms=waves)


def hf_generate(model, batch, *, eos_id=None):
    eos_id = model.config.im_end_token_id if eos_id is None else eos_id
    sequence = batch["input_ids"].clone()
    prompt_length, reached, logits = sequence.shape[1], False, []
    for _ in range(TEXT_TOKENS):
        output = hf_features(model, sequence, batch["model_context"])
        logits.append(output.logits[:, -1].float().clone())
        token = output.logits[:, -1].argmax(-1)
        if int(token[0]) == eos_id:
            reached = True
            break
        sequence = torch.cat((sequence, token[:, None]), 1)
    speech, condition = None, None
    if sequence.shape[1] > prompt_length:
        output = hf_features(model, sequence, batch["model_context"])
        logits.append(output.logits[:, -1].float().clone())
        condition = hf_condition(model, sequence, output, batch)
        speech = hf_codec(model, condition, batch)
    if reached:
        sequence = torch.cat((sequence, token[:, None]), 1)
    return SimpleNamespace(
        sequences=sequence,
        text_reached_eos=reached,
        speech=speech,
        logits=logits,
        condition=condition,
    )


def native_generate(model, batch, *, eos_id=None):
    rows, original = [], model.thinker.forward_epilog
    had_override = "forward_epilog" in model.thinker.__dict__
    prior_override = model.thinker.__dict__.get("forward_epilog")

    def observe(*args, **kwargs):
        logits = original(*args, **kwargs)
        rows.append(logits[:, -1].float().clone())
        return logits

    # A read-only observer: returns the exact original tensor and is always removed.
    model.thinker.forward_epilog = observe
    try:
        output = model.generate(
            batch["input_ids"],
            batch["model_context"],
            max_text_tokens=TEXT_TOKENS,
            max_codec_frames=CODEC_FRAMES,
            text_eos_token_id=eos_id,
            do_sample=False,
            position_kwargs=batch["position_kwargs"],
        )
    finally:
        if had_override:
            model.thinker.forward_epilog = prior_override
        else:
            del model.thinker.forward_epilog
    return output, rows


def compare_speech(actual, expected):
    if (actual is None) != (expected is None):
        raise AssertionError("Speech presence differs")
    report = dict(
        speech_presence_exact=True,
        code_groups=16,
        code_exact=True,
        code_lengths_exact=True,
        reached_eos_exact=True,
        code_lengths=[],
        reached_eos=[],
        native_codes=[],
        hf_codes=[],
        waveforms=[],
    )
    if actual is None:
        return report
    chain.exact(actual.code_lengths, expected.code_lengths, "code-lengths")
    chain.exact(actual.reached_eos, expected.reached_eos, "codec-eos")
    if actual.codes.shape[-1] != 16 or expected.codes.shape[-1] != 16:
        raise AssertionError("Generation must exercise all sixteen codebooks")
    for row, length in enumerate(actual.code_lengths.tolist()):
        chain.exact(actual.codes[row, :length], expected.codes[row, :length], "all-codebooks")
        if actual.codes[row, length:].ne(0).any():
            raise AssertionError("Finished row retained codec tokens after its EOS")
        left, right = actual.waveforms[row], expected.waveforms[row]
        error = chain.nse(left, right)
        if error >= OUTPUT_NSE or left.shape != right.shape or left.ndim != 2 or left.shape[0] != 1:
            raise AssertionError("Waveform shape/finite/numerical parity failed")
        if (length == 0) != (left.numel() == 0):
            raise AssertionError("Empty codec rows must produce exactly empty waveforms")
        report["waveforms"].append(
            dict(
                row=row,
                codec_frames=length,
                samples=left.shape[-1],
                shape=list(left.shape),
                finite=True,
                nse=error,
                native_sha256=chain.tensor_tree(left)["sha256"],
                hf_sha256=chain.tensor_tree(right)["sha256"],
            )
        )
    report.update(
        code_lengths=actual.code_lengths.tolist(),
        reached_eos=actual.reached_eos.tolist(),
        native_codes=actual.codes.tolist(),
        hf_codes=expected.codes.tolist(),
    )
    return report


def compare_generation(actual, rows, expected, prompt_length):
    chain.exact(actual.sequences, expected.sequences, "generated-text")
    if actual.text_reached_eos != expected.text_reached_eos or len(rows) != len(expected.logits):
        raise AssertionError("Text EOS or forward observation count differs")
    errors = [chain.nse(left, right) for left, right in zip(rows, expected.logits, strict=True)]
    if not errors or any(error >= OUTPUT_NSE for error in errors):
        raise AssertionError("Native/HF Thinker generation logits differ")
    return dict(
        status="PASS",
        scope="unforced_generation",
        text_exact=True,
        text_eos_exact=True,
        generated_tokens=actual.sequences[:, prompt_length:].tolist(),
        native_generated_tokens=actual.sequences[:, prompt_length:].tolist(),
        hf_generated_tokens=expected.sequences[:, prompt_length:].tolist(),
        text_reached_eos=actual.text_reached_eos,
        thinker_logits_nse=errors,
        **compare_speech(actual.speech, expected.speech),
    )


def ragged_control(native, hf, condition, batch):
    from transformers import LogitsProcessor, LogitsProcessorList

    from pithtrain.models.qwen3_omni_speech import TalkerCondition, generate_speech_from_condition

    eos = native.config.talker_config.codec_eos_token_id
    repeated = {key: value.expand(2, *value.shape[1:]).clone() for key, value in condition.items()}
    # This control is text-only; no media grids are duplicated or invented.
    positions, _ = hf.talker.get_rope_index(
        repeated["talker_input_ids"], attention_mask=repeated["attention_mask"]
    )
    prepared = TalkerCondition(
        repeated["inputs_embeds"],
        repeated["talker_input_ids"],
        repeated["attention_mask"],
        positions,
        repeated["trailing_text_hidden"],
        repeated["tts_pad_embed"],
    )
    original, counter = native.talker.forward, [0]
    had_override = "forward" in native.talker.__dict__
    prior_override = native.talker.__dict__.get("forward")

    def controlled(*args, **kwargs):
        output = original(*args, **kwargs)
        logits = output.logits.clone()
        logits[:, -1] = -torch.inf
        logits[0, -1, eos] = 0
        logits[1, -1, eos if counter[0] == 2 else 3] = 0
        counter[0] += 1
        return type(output)(logits, output.last_hidden_state)

    class ForcedEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            result = torch.full_like(scores, -torch.inf)
            result[0, eos] = 0
            result[1, eos if input_ids.shape[1] == 2 else 3] = 0
            return result

    native.talker.forward = controlled
    try:
        actual = compute(
            native,
            lambda model: generate_speech_from_condition(
                model.talker, model.code2wav, prepared, max_frames=4, do_sample=False
            ),
        )
    finally:
        if had_override:
            native.talker.forward = prior_override
        else:
            del native.talker.forward
    expected = compute(
        hf,
        lambda model: hf_codec(
            model,
            repeated,
            batch,
            max_frames=4,
            logits_processor=LogitsProcessorList([ForcedEOS()]),
        ),
    )
    if actual.code_lengths.tolist() != [0, 2] or actual.reached_eos.tolist() != [True, True]:
        raise AssertionError("Controlled individual EOS did not produce ragged [0,2] lengths")
    return dict(
        status="PASS",
        scope="controlled_codec_eos_boundary",
        expected_lengths=[0, 2],
        **compare_speech(actual, expected),
    )


@contextmanager
def hf_position_metadata(hf, config):
    owner = hf.talker.config
    existed = hasattr(owner, "audio_config")
    previous = getattr(owner, "audio_config", None)
    owner.audio_config = copy.deepcopy(config.thinker_config.audio_config)
    try:
        yield
    finally:
        if existed:
            owner.audio_config = previous
        else:
            del owner.audio_config


@torch.no_grad()
def validate(args, config, samples, native, hf):
    native, hf = native.model, hf.model
    # Official HF Talker position helper requires this official Thinker metadata.
    # No model expression, weight, logit or native result is substituted.
    cases, boundaries, text_context = {}, {}, None
    with preserve_generation_state(native, hf) as preservation, hf_position_metadata(hf, config):
        for sample in samples:
            batch = prompt_sample(chain.move(sample, args.device))
            actual, rows = compute(native, native_generate, batch)
            expected = compute(hf, hf_generate, batch)
            report = compare_generation(actual, rows, expected, batch["input_ids"].shape[1])
            report["media_kind"] = sample["media_kind"]
            cases[sample["media_kind"]] = report
            chain.write_json(args.output / f"generation-{sample['media_kind']}.json", report)
            if sample["media_kind"] == "text":
                text_context = batch, expected
        if set(cases) != set(MEDIA) or not any(
            w["samples"] > 0 for c in cases.values() for w in c["waveforms"]
        ):
            raise AssertionError(
                "All media and at least one actual nonempty end-to-end waveform are required"
            )
        batch, previous = text_context
        # Use the independent HF result dynamically, never a hard-coded seed token.
        eos_id = int(previous.logits[0].argmax(-1)[0])
        actual, rows = compute(native, native_generate, batch, eos_id=eos_id)
        expected = compute(hf, hf_generate, batch, eos_id=eos_id)
        boundary = compare_generation(actual, rows, expected, batch["input_ids"].shape[1])
        if (
            not actual.text_reached_eos
            or actual.speech is not None
            or actual.sequences.shape[1] != batch["input_ids"].shape[1] + 1
        ):
            raise AssertionError("Immediate Thinker EOS must produce no speech")
        boundary.update(
            scope="explicit_text_eos_boundary", eos_id=eos_id, immediate_eos_no_speech=True
        )
        boundaries["eos"] = boundary
        chain.write_json(args.output / "generation-eos.json", boundary)
        if previous.condition is None:
            raise AssertionError(
                "Text fixture produced no condition for the separate ragged control"
            )
        boundaries["ragged"] = ragged_control(native, hf, previous.condition, batch)
        chain.write_json(args.output / "generation-ragged.json", boundaries["ragged"])
    return dict(
        status="PASS",
        cases=cases,
        boundaries=boundaries,
        **preservation,
        native_parameter_count=len(list(native.named_parameters())),
        reference="official HF 5.17 forwards/conditioning helpers/cached Talker.generate/Code2Wav",
        precision="FP32 model leaves/rotary buffers; functional BF16 compute parameters",
        nonempty_waveform_cases=sum(
            any(w["samples"] > 0 for w in c["waveforms"]) for c in cases.values()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema=SCHEMA,
        status="NOT_RUN",
        gpu_acceptance="NOT_RUN",
        seed=chain.SEED,
        limits=dict(text_tokens=TEXT_TOKENS, codec_frames=CODEC_FRAMES),
        thresholds=dict(output_nse=OUTPUT_NSE),
        full_size="NOT_RUN",
        fsdp_generation="UNSUPPORTED",
    )
    started, phase = time.monotonic(), "source_precondition"
    try:
        result["source"] = chain.source_identity(args.expected_commit)
        result["runtime"] = chain.runtime_precondition(args)
        config, samples, _ = chain.load_fixture(args.prepared, args.expected_commit)
        if (
            config.talker_config.num_code_groups != 16
            or config.code2wav_config.num_quantizers != 16
        ):
            raise chain.PreconditionError("All sixteen codebooks are mandatory")
        result["fixture_sha256"] = chain.sha(args.prepared / "fixture.json")
        phase = "model_setup"
        native, hf, attention = chain.build_pair(config, args.device)
        result["reference_attention"] = attention
        phase = "generation"
        result.update(validate(args, config, samples, native, hf))
        result["gpu_acceptance"] = "PASS_THIS_GATE_ONLY"
    except Exception as error:
        result.update(chain.failure_report(phase, error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        chain.write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
