"""CPU-only file roundtrips; no native CUDA kernel or full-model numeric claims."""

import copy
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFTextModel,
)

from pithtrain.contexts import distributed
from pithtrain.modules.checkpoint import to_canonical_model, to_localized_model
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import (
    SIDECAR,
    ConversionError,
    canonical_to_hf,
    export_config,
    hf_to_canonical,
    load_canonical,
    parameter_shapes,
    text_config,
)

LOGGER = logging.getLogger(__name__)
CONFIGS = Path(__file__).parent / "configs/qwen3_omni_text"


def make_cfg(**kwargs):
    cfg = ConvertCheckpointCfg()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def config():
    raw = json.loads((CONFIGS / "tiny.json").read_text())
    # Asymmetric dimensions reveal accidental transposes; small vocab keeps all
    # CPU/disk tests cheap. This is synthetic data, not the real-token recipe.
    raw.update(
        vocab_size=37,
        hidden_size=24,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=10,
        num_experts=3,
        num_experts_per_tok=2,
    )
    raw["rope_parameters"]["mrope_section"] = [2, 1, 1]
    return text_config(raw)


def model(config, dtype=torch.float32):
    cfg = copy.deepcopy(config)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(123)
        result = nn.ModuleDict(
            {
                "model": HFTextModel(cfg),
                "lm_head": nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False),
            }
        )
        # HF raw expert Parameters need deterministic nontrivial initialization.
        for name, param in result.named_parameters():
            if param.ndim > 1:
                nn.init.normal_(param, std=0.05)
            elif name.endswith("bias"):
                nn.init.zeros_(param)
            else:
                nn.init.ones_(param)
    return result.to(dtype=dtype).eval()


def synthetic_hf(config, prefix="thinker.", layout="fused", dtype=torch.float32):
    fused = {
        prefix + key: value.detach().clone()
        for key, value in model(config, dtype).state_dict().items()
    }
    assert {key: tuple(value.shape) for key, value in fused.items()} == parameter_shapes(
        config, "fused", prefix
    )
    if layout == "fused":
        return fused
    # Independent source fixture: mimic released individual Linear modules,
    # not the converter's inverse (which could hide paired mapping mistakes).
    result = {}
    for key, value in fused.items():
        if key.endswith(".experts.gate_up_proj"):
            root = key.removesuffix("gate_up_proj")
            for i in range(config.num_experts):
                result[f"{root}{i}.gate_proj.weight"] = value[
                    i, : config.moe_intermediate_size
                ].clone()
                result[f"{root}{i}.up_proj.weight"] = value[
                    i, config.moe_intermediate_size :
                ].clone()
        elif key.endswith(".experts.down_proj"):
            root = key.removesuffix("down_proj")
            for i in range(config.num_experts):
                result[f"{root}{i}.down_proj.weight"] = value[i].clone()
        else:
            result[key] = value
    return result


def write_hf(path, config, state, sharded=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    if not sharded:
        save_file(state, path / "model.safetensors")
        return
    parts = [dict(list(state.items())[i::2]) for i in range(2)]
    # gate and up need not share a shard.
    mapping = {}
    for i, part in enumerate(parts):
        filename = f"model-{i}.safetensors"
        save_file(part, path / filename)
        mapping.update({key: filename for key in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))


def read_export(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    result = {}
    for filename in set(index["weight_map"].values()):
        result.update(load_file(path / filename))
    return result


@pytest.mark.parametrize("prefix", ["thinker.", ""])
@pytest.mark.parametrize("layout", ["split", "fused"])
@pytest.mark.parametrize("variant", ["moe", "dense-bias", "bf16"])
def test_cpu_file_roundtrip(config, tmp_path, monkeypatch, prefix, layout, variant):
    if variant == "dense-bias":
        config.mlp_only_layers, config.attention_bias = [0], True
        config.norm_topk_prob = False
    dtype = torch.bfloat16 if variant == "bf16" else torch.float32
    original = synthetic_hf(config, prefix, layout, dtype)
    src, disk, out = (tmp_path / name for name in ("hf", "dcp", "out"))
    write_hf(src, config, original, sharded=layout == "split")
    cfg = make_cfg(load_path=src, save_path=disk, omni_prefix=prefix)
    hf2dcp(cfg, LOGGER)
    canonical = load_canonical(disk, config)
    # Exercise checkpoint.py's actual stack/unstack path. HF text modules have
    # exactly native's weight names/shapes; only add native's expert count and
    # LM head. No replacement native forward or CUDA kernel is claimed here.
    runtime = model(config, dtype)
    body = runtime["model"]
    body.add_module("lm_head", runtime["lm_head"])
    for layer in body.layers:
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts_per_rank = config.num_experts
    monkeypatch.setattr(distributed, "ep_rank", 0, raising=False)
    monkeypatch.setattr(distributed, "ep_size", 1, raising=False)
    local = nn.ModuleDict({"module": nn.ModuleList([body])})
    localized = to_localized_model(canonical, local)
    local.load_state_dict(localized, strict=True)
    repacked = to_canonical_model(local.state_dict(), local)
    assert canonical.keys() == repacked.keys()
    for key in canonical:
        torch.testing.assert_close(canonical[key], repacked[key], rtol=0, atol=0)
    cfg.load_path, cfg.save_path = disk, out
    cfg.omni_expert_layout, cfg.max_shard_size = layout, 4000
    dcp2hf(cfg, LOGGER)
    exported = read_export(out)
    assert original.keys() == exported.keys()
    for key in original:
        assert original[key].dtype == exported[key].dtype
        torch.testing.assert_close(original[key], exported[key], rtol=0, atol=0)
    assert json.loads((out / SIDECAR).read_text())["scope"] == "thinker_text_only"
    # Feed the reloaded HF fused weights into a real eager HF prediction.
    before = model(config, dtype)
    after = model(text_config(json.loads((out / "config.json").read_text())), dtype)
    rebuilt, _ = hf_to_canonical(exported, config, prefix=prefix)
    after.load_state_dict(canonical_to_hf(rebuilt, config, prefix=""), strict=True)
    tokens = torch.tensor([[1, 3, 2, 11, 5], [7, 4, 8, 9, 6]])
    with torch.no_grad():
        logits_before = before["lm_head"](
            before["model"](input_ids=tokens, use_cache=False).last_hidden_state
        )
        logits_after = after["lm_head"](
            after["model"](input_ids=tokens, use_cache=False).last_hidden_state
        )
    torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)
    assert torch.equal(logits_after.argmax(-1), logits_before.argmax(-1))


def test_native_dcp_requires_explicit_config_and_preserves_values(config, tmp_path):
    state = synthetic_hf(config)
    canon, _ = hf_to_canonical(state, config)
    disk = tmp_path / "native"
    # Native checkpoint model payload plus unrelated optimizer/scheduler fields.
    dcp.save(
        {"app": {"model": canon, "optimizer": {"step": 3}, "scheduler": [3]}},
        checkpoint_id=disk,
        no_dist=True,
    )
    with pytest.raises(ConversionError, match="requires omni_text_config"):
        dcp2hf(make_cfg(load_path=disk, save_path=tmp_path / "unidentified"), LOGGER)
    assert not (tmp_path / "unidentified").exists()
    path = tmp_path / "text.json"
    path.write_text(json.dumps(config.to_dict()))
    cfg = make_cfg(load_path=disk, save_path=tmp_path / "out", omni_text_config=path)
    dcp2hf(cfg, LOGGER)
    exported = read_export(cfg.save_path)
    for key in state:
        torch.testing.assert_close(exported[key], state[key], rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["split", "fused"])
@pytest.mark.parametrize(
    "fault", ["missing", "unexpected", "shape", "prefix", "mixed", "expert-id", "integer"]
)
def test_hf_rejects_invalid_state(config, layout, fault):
    state = synthetic_hf(config, layout=layout)
    key = "thinker.model.layers.0.mlp.experts." + (
        "gate_up_proj" if layout == "fused" else "0.gate_proj.weight"
    )
    if fault == "missing":
        del state[key]
    elif fault == "unexpected":
        state["thinker.model.typo.weight"] = torch.ones(1)
    elif fault == "shape":
        state[key] = state[key].transpose(-1, -2)
    elif fault == "prefix":
        state["model.thinker." + key] = state.pop(key)
    elif fault == "mixed":
        other = synthetic_hf(config, layout="split" if layout == "fused" else "fused")
        state.update(other)
    elif fault == "expert-id":
        state["thinker.model.layers.0.mlp.experts.99.down_proj.weight"] = torch.ones(24, 12)
    else:
        state[key] = state[key].to(torch.int32)
    with pytest.raises(ConversionError):
        hf_to_canonical(state, config)


def test_reports_all_errors_and_no_silent_media(config, tmp_path):
    state = synthetic_hf(config)
    del state["thinker.lm_head.weight"]
    state["thinker.model.typo.weight"] = torch.ones(1)
    state["thinker.model.norm.weight"] = torch.ones(13)
    with pytest.raises(ConversionError) as error:
        hf_to_canonical(state, config)
    for text in (
        "missing=",
        "lm_head.weight",
        "unexpected=",
        "typo.weight",
        "shape_mismatches=",
        "norm.weight",
    ):
        assert text in str(error.value)
    state = synthetic_hf(config)
    state["thinker.visual.deepstack_merger_list.0.weight"] = torch.ones(2)
    state["talker.model.embed_tokens.weight"] = torch.ones(2)
    with pytest.raises(ConversionError, match="explicit omni_text_only=True"):
        hf_to_canonical(state, config)
    canonical, excluded = hf_to_canonical(state, config, text_only=True)
    assert len(excluded) == 2
    assert len(canonical) == len(parameter_shapes(config))
    src = tmp_path / "hf"
    write_hf(src, config, state)
    cfg = make_cfg(load_path=src, save_path=tmp_path / "dcp", omni_text_only=True)
    hf2dcp(cfg, LOGGER)
    assert json.loads((cfg.save_path / SIDECAR).read_text())["excluded_keys"] == excluded
    state["thinker.model.deepstack_unknown.weight"] = torch.ones(2)
    with pytest.raises(ConversionError, match="unexpected"):
        hf_to_canonical(state, config, text_only=True)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "no-head"])
def test_dcp_rejects_invalid_metadata_before_export(config, tmp_path, fault):
    canon, _ = hf_to_canonical(synthetic_hf(config), config)
    key = "layers.0.mlp.experts.1.gate_up_proj"
    if fault == "missing":
        del canon[key]
    elif fault == "unexpected":
        canon["garbage"] = torch.ones(2)
    elif fault == "shape":
        canon[key] = canon[key][:-1]
    else:
        del canon["lm_head.weight"]
    with pytest.raises(ConversionError):
        canonical_to_hf(canon, config)
    disk = tmp_path / "invalid"
    dcp.save({"app": {"model": canon}}, checkpoint_id=disk, no_dist=True)
    with pytest.raises(ConversionError):
        load_canonical(disk, config)


@pytest.mark.parametrize(
    "fault", ["missing-index", "wrong-shard", "extra-header", "duplicate", "bad-shape"]
)
def test_sharded_input_fails_before_output(config, tmp_path, fault):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write_hf(src, config, synthetic_hf(config, layout="split"), sharded=True)
    path = src / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    key = next(iter(index["weight_map"]))
    filename = index["weight_map"][key]
    other = "model-1.safetensors" if filename == "model-0.safetensors" else "model-0.safetensors"
    if fault == "missing-index":
        del index["weight_map"][key]
    elif fault == "wrong-shard":
        index["weight_map"][key] = other
    elif fault == "duplicate":
        values = load_file(src / other)
        values[key] = load_file(src / filename)[key]
        save_file(values, src / other)
    else:
        values = load_file(src / filename)
        values[key if fault == "bad-shape" else "unindexed.weight"] = torch.ones(2)
        save_file(values, src / filename)
    path.write_text(json.dumps(index))
    with pytest.raises(ConversionError):
        hf2dcp(make_cfg(load_path=src, save_path=dst), LOGGER)
    assert not dst.exists()


def test_dtype_and_config_guards(config, tmp_path):
    split = synthetic_hf(config, layout="split")
    split["thinker.model.layers.0.mlp.experts.0.up_proj.weight"] = split[
        "thinker.model.layers.0.mlp.experts.0.up_proj.weight"
    ].double()
    with pytest.raises(ConversionError, match="Mixed gate/up"):
        hf_to_canonical(split, config)
    canon, _ = hf_to_canonical(synthetic_hf(config), config)
    canon["layers.0.mlp.experts.0.down_proj"] = canon["layers.0.mlp.experts.0.down_proj"].double()
    with pytest.raises(ConversionError, match="Mixed expert"):
        canonical_to_hf(canon, config)
    raw = config.to_dict()
    del raw["hidden_size"]
    with pytest.raises(ConversionError, match="Missing config"):
        text_config(raw)
    src = tmp_path / "hf"
    write_hf(src, config, synthetic_hf(config))
    cfg = make_cfg(load_path=src, save_path=tmp_path / "dcp")
    hf2dcp(cfg, LOGGER)
    with pytest.raises(ConversionError, match="empty"):
        hf2dcp(cfg, LOGGER)
    raw = config.to_dict()
    raw["norm_topk_prob"] = False
    override = tmp_path / "override.json"
    override.write_text(json.dumps(raw))
    with pytest.raises(ConversionError, match="disagrees"):
        export_config(cfg.save_path, override)


@pytest.mark.parametrize("dense", [False, True])
def test_full_meta_schema_and_bidirectional_shapes(dense):
    config = text_config(json.loads((CONFIGS / "full.json").read_text()))
    if dense:
        config.num_experts = 0
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with torch.device("meta"):
        body = HFTextModel(config)
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        wrapper = nn.ModuleDict({"model": body, "lm_head": head})
    state = {"thinker." + key: value for key, value in wrapper.state_dict().items()}
    assert {key: tuple(value.shape) for key, value in state.items()} == parameter_shapes(
        config, "fused"
    )
    canon, _ = hf_to_canonical(state, config)
    split = canonical_to_hf(canon, config, layout="split")
    restored, _ = hf_to_canonical(split, config)
    assert {key: tuple(value.shape) for key, value in restored.items()} == parameter_shapes(config)
    assert all(value.device.type == "meta" for value in restored.values())
    if not dense:
        assert len(canon) == 12723
        assert len(split) == 18867
        assert len(state) == 531


def test_existing_qwen_generic_file_roundtrip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "qwen3_moe"}')
    state = {
        "model.layers.0.weight": torch.arange(6).reshape(2, 3).float(),
        "lm_head.weight": torch.ones(3, 2),
    }
    save_file(state, src / "weights.safetensors")
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in state}})
    )
    cfg = make_cfg(load_path=src, save_path=tmp_path / "dcp")
    hf2dcp(cfg, LOGGER)
    cfg.load_path, cfg.save_path = cfg.save_path, tmp_path / "out"
    dcp2hf(cfg, LOGGER)
    for key, value in read_export(cfg.save_path).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_no_cuda_context_used():
    assert not torch.cuda.is_initialized()


def test_cli_dense_native_requires_config(tmp_path):
    script = Path(__file__).parents[1] / "examples/convert_checkpoint/qwen3-omni-text/script.py"
    result = subprocess.run(
        [sys.executable, str(script), "dcp2hf", str(tmp_path), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires --text-config" in result.stderr
    assert not (tmp_path / "out").exists()
