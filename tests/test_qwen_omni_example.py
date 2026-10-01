"""Qwen example CLI wiring and real CPU LR schedules; no GPU/model training."""

import ast
import builtins
import copy
import math
import re
import runpy
import shlex
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tests.test_data_config import ROOT, example_modules

SCRIPT = ROOT / "examples/pretrain_lm/qwen3-omni/script.py"


def scheduler_builders():
    # Execute the actual builder ASTs without importing the GPU model registry.
    path = ROOT / "pithtrain/modules/training.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in {"make_wsd_scheduler", "make_constant_scheduler"}
    ]
    assert len(nodes) == 2
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    namespace = dict(
        math=math, LambdaLR=torch.optim.lr_scheduler.LambdaLR, training=SimpleNamespace()
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def run_example(monkeypatch, tmp_path, arguments, *, muon_builder=None):
    calls = example_modules(monkeypatch)
    builders = scheduler_builders()
    training_module = sys.modules["pithtrain.modules.training"]
    if muon_builder is not None:
        monkeypatch.setattr(training_module, "make_muon_optimizer", muon_builder)
    for name in ("make_wsd_scheduler", "make_constant_scheduler"):
        monkeypatch.setattr(training_module, name, builders[name])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dataset",
            str(tmp_path / "bundle"),
            "--model",
            "model-config",
            "--checkpoint",
            str(tmp_path / "checkpoints"),
            *arguments,
        ],
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert len(calls) == 1 and not torch.cuda.is_initialized()
    return calls[0], training_module, builders["training"]


def cpu_schedule(cfg, runtime, count=1):
    optimizers = tuple(
        torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=cfg.training.lr)
        for _ in range(count)
    )
    runtime.optimizers = optimizers
    return optimizers, cfg.training.scheduler(cfg.training)


def advance(optimizers, schedulers):
    for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
        optimizer.step()
        scheduler.step()
    return [optimizer.param_groups[0]["lr"] for optimizer in optimizers]


def test_default_training_recipe_is_explicit(monkeypatch, tmp_path):
    cfg, module, _ = run_example(monkeypatch, tmp_path, [])
    tc = cfg.training
    assert tc.optimizer.func is module.make_adamw_optimizer
    assert tc.optimizer.keywords == {"weight_decay": 0.1}
    assert isinstance(tc.scheduler, partial) and tc.scheduler.func is module.make_wsd_scheduler
    assert tc.scheduler.keywords == dict(
        start_lr=1e-5, warmup_ratio=0.03, final_lr=1e-5, decay_ratio=0.1, decay_shape="cosine"
    )
    assert (tc.lr, tc.max_steps, tc.global_batch_size, tc.save_interval) == (1e-4, 4096, 1024, 256)
    assert (tc.moe_load_balance_type, tc.moe_load_balance_coef) == ("global-batch", 1e-3)


def test_optimizer_and_load_balance_overrides_reach_builder(monkeypatch, tmp_path):
    observed = []

    def muon_spy(config, *, weight_decay):
        observed.append((config, weight_decay))
        return ("muon", "adamw")

    cfg, module, _ = run_example(
        monkeypatch,
        tmp_path,
        [
            "--optimizer",
            "muon",
            "--weight-decay",
            "0.05",
            "--lr",
            "3e-4",
            "--moe-load-balance-type",
            "sequence",
            "--moe-load-balance-coef",
            "0.02",
            "--steps",
            "8192",
            "--global-batch-size",
            "512",
            "--save-interval",
            "128",
        ],
        muon_builder=muon_spy,
    )
    assert cfg.training.optimizer.func is module.make_muon_optimizer
    # Selection/argument transport only; the real GPU optimizer is not executed.
    assert cfg.training.optimizer(cfg.training) == ("muon", "adamw")
    assert observed == [(cfg.training, 0.05)]
    assert (cfg.training.moe_load_balance_type, cfg.training.moe_load_balance_coef) == (
        "sequence",
        0.02,
    )
    assert (cfg.training.max_steps, cfg.training.global_batch_size, cfg.training.save_interval) == (
        8192,
        512,
        128,
    )


@pytest.mark.parametrize(
    "shape,tail",
    [
        ("linear", [0.0775, 0.055, 0.0325, 0.01]),
        ("cosine", [0.08681980515339464, 0.055, 0.023180194846605363, 0.01]),
    ],
)
def test_wsd_cli_controls_real_lr_curve_for_each_optimizer(monkeypatch, tmp_path, shape, tail):
    cfg, _, runtime = run_example(
        monkeypatch,
        tmp_path,
        [
            "--steps",
            "10",
            "--lr",
            "0.1",
            "--start-lr",
            "0.02",
            "--final-lr",
            "0.01",
            "--warmup-ratio",
            "0.2",
            "--decay-ratio",
            "0.4",
            "--decay-shape",
            shape,
        ],
    )
    optimizers, schedulers = cpu_schedule(cfg, runtime, count=2)
    actual = [[optimizer.param_groups[0]["lr"] for optimizer in optimizers]]
    actual.extend(advance(optimizers, schedulers) for _ in range(10))
    expected = [0.02, 0.06, 0.1, 0.1, 0.1, 0.1, 0.1, *tail]
    for rates, rate in zip(actual, expected, strict=True):
        assert rates == pytest.approx([rate, rate], abs=1e-12)


def test_wsd_state_resume_preserves_remaining_lr_sequence(monkeypatch, tmp_path):
    cfg, _, runtime = run_example(
        monkeypatch,
        tmp_path,
        [
            "--steps",
            "20",
            "--warmup-ratio",
            "0.2",
            "--decay-ratio",
            "0.3",
        ],
    )
    optimizers, schedulers = cpu_schedule(cfg, runtime)
    for _ in range(7):
        advance(optimizers, schedulers)
    saved_optimizer = copy.deepcopy(optimizers[0].state_dict())
    saved_scheduler = copy.deepcopy(schedulers[0].state_dict())
    expected = [advance(optimizers, schedulers) for _ in range(13)]
    restored_optimizers, restored_schedulers = cpu_schedule(cfg, runtime)
    restored_optimizers[0].load_state_dict(saved_optimizer)
    restored_schedulers[0].load_state_dict(saved_scheduler)
    actual = [advance(restored_optimizers, restored_schedulers) for _ in range(13)]
    assert actual == expected


def test_constant_smoke_options_keep_fixed_lr(monkeypatch, tmp_path):
    cfg, module, runtime = run_example(
        monkeypatch,
        tmp_path,
        [
            "--scheduler",
            "constant",
            "--optimizer",
            "adamw",
            "--lr",
            "1e-4",
            "--moe-load-balance-coef",
            "0",
            "--steps",
            "4",
            "--global-batch-size",
            "8",
            "--save-interval",
            "2",
        ],
    )
    assert cfg.training.scheduler is module.make_constant_scheduler
    assert cfg.training.moe_load_balance_coef == 0
    optimizers, schedulers = cpu_schedule(cfg, runtime)
    assert optimizers[0].param_groups[0]["lr"] == 1e-4
    assert [advance(optimizers, schedulers) for _ in range(4)] == [[1e-4]] * 4


def test_wsd_can_explicitly_disable_both_phases(monkeypatch, tmp_path):
    cfg, _, runtime = run_example(
        monkeypatch,
        tmp_path,
        [
            "--steps",
            "1",
            "--warmup-ratio",
            "0",
            "--decay-ratio",
            "0",
        ],
    )
    optimizers, schedulers = cpu_schedule(cfg, runtime)
    assert advance(optimizers, schedulers) == [cfg.training.lr]


@pytest.mark.parametrize(
    "arguments,match",
    [
        (["--lr", "0"], "--lr"),
        (["--lr", "nan"], "--lr"),
        (["--weight-decay", "-1"], "--weight-decay"),
        (["--weight-decay", "inf"], "--weight-decay"),
        (["--moe-load-balance-coef", "-0.1"], "--moe-load-balance-coef"),
        (["--moe-load-balance-coef", "nan"], "--moe-load-balance-coef"),
        (["--start-lr", "1"], "--start-lr"),
        (["--final-lr", "-1"], "--final-lr"),
        (["--warmup-ratio", "-0.1"], "--warmup-ratio"),
        (["--decay-ratio", "1.1"], "--decay-ratio"),
        (["--decay-ratio", "nan"], "--decay-ratio"),
        (["--warmup-ratio", "0.8", "--decay-ratio", "0.3"], "sum to at most 1"),
        (["--steps", "4"], "rounds to zero"),
        (["--steps", "3", "--warmup-ratio", "0.5", "--decay-ratio", "0.5"], "Rounded WSD"),
        (["--scheduler", "constant", "--warmup-ratio", "0.2"], "WSD options require"),
    ],
)
def test_invalid_training_options_fail_before_gpu_imports(
    monkeypatch, tmp_path, capsys, arguments, match
):
    real_import = builtins.__import__

    def no_training_import(name, *args, **kwargs):
        assert not name.startswith("pithtrain"), "Validate training options before GPU imports"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_training_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dataset",
            str(tmp_path),
            "--model",
            "model-config",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == 2
    assert match in capsys.readouterr().err


@pytest.mark.parametrize("index", [0, 1], ids=["smoke", "training"])
def test_documented_example_commands_build_expected_config(monkeypatch, tmp_path, index):
    readme = (SCRIPT.parent / "README.md").read_text()
    commands = [
        shlex.split(block.replace("\\\n", " "))
        for block in re.findall(r"```bash\n(.*?)```", readme, re.S)
        if "examples/pretrain_lm/qwen3-omni/script.py" in block
    ]
    assert len(commands) == 2
    command = commands[index]
    start = command.index("examples/pretrain_lm/qwen3-omni/script.py") + 1
    cfg, module, _ = run_example(monkeypatch, tmp_path, command[start:])
    if index == 0:
        assert cfg.training.scheduler is module.make_constant_scheduler
        assert cfg.training.max_steps == 4 and cfg.training.moe_load_balance_coef == 0
    else:
        assert cfg.training.optimizer.func is module.make_muon_optimizer
        assert cfg.training.scheduler.func is module.make_wsd_scheduler
        assert cfg.training.max_steps == 4096 and cfg.training.global_batch_size == 1024
        assert cfg.training.moe_load_balance_coef == 1e-3
        assert cfg.distributed.pipeline_parallel_size == cfg.distributed.expert_parallel_size == 2
