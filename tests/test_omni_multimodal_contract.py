"""Exercise real selection/entrypoint ASTs without importing GPU-only operators."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from pithtrain.modules.model_config import resolve_text_model_config
from tests.test_data_config import task_config
from tests.test_qwen3_omni_multimodal import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "flag", ("native_vision", "native_audio", "native_video", "native_multimodal")
)
def test_all_native_modes_have_explicit_selection_and_no_ambiguous_combination(flag):
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "model_class_for_config"
    )
    classes = {
        "native_vision": ("Qwen3OmniMoeThinkerImageModel", object()),
        "native_audio": ("Qwen3OmniMoeThinkerAudioModel", object()),
        "native_video": ("Qwen3OmniMoeThinkerVideoModel", object()),
        "native_multimodal": ("Qwen3OmniMoeThinkerMultimodalModel", object()),
    }
    text = object()
    namespace = {name: value for name, value in classes.values()}
    namespace.update(
        Qwen3OmniMoeThinkerTextModel=text, resolve_text_model_config=resolve_text_model_config
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "registration", "exec"), namespace)
    choose = namespace[function.name]
    assert choose(config()) is text
    assert choose(config(), **{flag: True}) is classes[flag][1]
    other = "native_audio" if flag == "native_multimodal" else "native_multimodal"
    with pytest.raises(ValueError, match="Select one"):
        choose(config(), **{flag: True, other: True})


def test_actual_dataset_entry_forwards_every_native_selection_flag(tmp_path):
    observed = []

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def validate_model(self, *args):
            pass

    def choose(source, **flags):
        observed.append(flags)
        return object()

    task = task_config(
        OmniPretrainData=Provider,
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: object()),
        model_class_for_config=choose,
    )
    cfg = task.PretrainLMCfg()
    cfg.data.dataset, cfg.data.format = tmp_path, "prepared_bundle"
    cfg.training.model, cfg.training.native_multimodal = "composite", True
    task.setup_dataset(cfg)
    assert observed == [
        dict(native_vision=False, native_audio=False, native_video=False, native_multimodal=True)
    ]


def test_unified_production_has_no_transformers_execution_dependency():
    for filename in ("qwen3_omni_multimodal.py", "qwen3_omni_multimodal_inputs.py"):
        tree = ast.parse((ROOT / "pithtrain/models" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers")
