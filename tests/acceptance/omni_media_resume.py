"""Strict fresh-process resume gates reusable by native multimodal consumers.

The default CLI consumer is audio. A combined consumer can call execute(args,
MediaSpec(...)) with its real source/config/group functions. All native model
parameters, optimizers, schedulers, RNG and data cursor remain in strict snapshots.
"""

import argparse
import inspect
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from pithtrain.tasks.prepare_omni_data import verify_bundle
from tests.acceptance.omni_distributed_resume import negative_case, snapshot
from tests.acceptance.omni_image_evidence import OptimizerStepEvidence
from tests.acceptance.omni_native_resume import equal, sha
from tests.test_omni_training_acceptance import digest

REPO = Path(__file__).resolve().parents[2]
PROCESS_NONCE = uuid.uuid4().hex


@dataclass(frozen=True)
class MediaSpec:
    name: str
    modalities: tuple[str, ...]
    training_flag: str
    required_components: tuple[str, ...]
    source_inputs: object
    training_groups: object
    hf_name: object
    sequence_length: int = 2048

    def contract(self):
        assert self.modalities and self.required_components
        return dict(
            name=self.name,
            modalities=list(self.modalities),
            training_flag=self.training_flag,
            required_components=list(self.required_components),
            sequence_length=self.sequence_length,
            sources={
                str(Path(inspect.getsourcefile(function)).relative_to(REPO)): sha(
                    inspect.getsourcefile(function)
                )
                for function in (self.source_inputs, self.training_groups, self.hf_name)
            },
        )


def audio_spec():
    from tests.acceptance.omni_audio_training import hf_name, source_inputs, training_groups

    return MediaSpec(
        "audio",
        ("audio",),
        "native_audio",
        ("audio_tower.",),
        source_inputs,
        training_groups,
        hf_name,
    )


def input_identity(root, modalities):
    bundle = verify_bundle(root)
    assert set(modalities) <= set(bundle["manifests"]["train"]), "Requested modality is absent"
    return dict(bundle=sha(root / "bundle.json"), checksums=sha(root / "checksums.sha256"))


def identity(args, spec):
    return dict(
        protocol="strict_multimodal_save_then_runtime_state_v1",
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        driver=sha(Path(__file__)),
        consumer=spec.contract(),
        input=input_identity(args.dataset, spec.modalities),
        pp=args.pp,
        ep=args.ep,
        world_size=args.world_size,
        dp=args.world_size // args.pp,
        cp=1,
        sequence_length=spec.sequence_length,
        global_batch_size=8,
        micro_batch_size=1,
        steps=3,
    )


def prepare(args, spec):
    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), "Use a fresh evidence directory"
    assert not args.output.is_relative_to(args.dataset) and not args.dataset.is_relative_to(
        args.output
    )
    metadata = identity(args, spec)
    _, samples, batches, config = spec.source_inputs(args.dataset)
    assert samples and batches
    assert all((batch.labels != -100).any() for batch in batches)
    args.output.mkdir(parents=True)
    shutil.copytree(args.dataset, args.output / "data")
    config.save_pretrained(args.output / "model")
    assert input_identity(args.output / "data", spec.modalities) == metadata["input"]
    metadata["sample_inputs"] = [
        digest(
            dict(model_inputs=batch.model_inputs, labels=batch.labels, sample_ids=batch.sample_ids)
        )
        for batch in batches
    ]
    (args.output / "prepare.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("CPU_PREPARE_PASS; CUDA_INITIALIZED=False; GPU_ACCEPTANCE=NOT_RUN")


class ObservedMediaData:
    """Hash every tensor returned by the unchanged production source."""

    def __init__(self, source):
        self.source, self.batches = source, {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_batch(self, step, device):
        batches = self.source.get_batch(step, device)
        self.batches[step] = digest(
            [
                dict(
                    model_inputs=mb.model_inputs,
                    objective_inputs=mb.objective_inputs,
                    cu_seqlens=mb.cu_seqlens,
                    sample_ids=mb.sample_ids,
                    model_context=mb.model_context,
                    media_inputs=mb.media_inputs,
                )
                for mb in batches
            ]
        )
        return batches


class ObjectiveObservation:
    """Observe exact detached objective sums; preserve the production objective."""

    def __init__(self, objective):
        self.objective, self.outputs = objective, []

    def __call__(self, model_outputs, objective_inputs):
        loss, detached = self.objective(model_outputs, objective_inputs)
        self.outputs.append(digest(dict(loss=detached, labels=objective_inputs)))
        return loss, detached


def require_components(keys, prefixes):
    assert keys, "Empty model state"
    for prefix in prefixes:
        assert any(key.startswith(prefix) for key in keys), (
            f"Missing checkpoint component: {prefix}"
        )


def run_gpu(args, spec):
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU execution is opt-in"
    assert int(os.environ.get("WORLD_SIZE", "1")) == args.world_size
    assert torch.cuda.is_available(), "Requires CUDA"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert "H100" in torch.cuda.get_device_name(), "Requires H100"
    prepared = json.loads((args.output / "prepare.json").read_text())
    assert {key: value for key, value in prepared.items() if key != "sample_inputs"} == identity(
        args, spec
    )
    assert input_identity(args.output / "data", spec.modalities) == prepared["input"]
    subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=True)
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.checkpoint import find_checkpoint, load_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import _phase, resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler, setup_training
    from pithtrain.tasks import pretrain_lm

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.timeout = timedelta(seconds=90)
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.output / "data",
        "prepared_bundle",
        spec.modalities,
    )
    cfg.data.num_workers = 0
    t = cfg.training
    t.model = args.output / "model"
    setattr(t, spec.training_flag, True)
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-5,
        3,
        spec.sequence_length,
        8,
        1,
    )
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.strict_resume, t.save_interval = True, 2
    t.save_location = args.output / (
        "checkpoints" if args.mode == "produce" else "restored-checkpoints"
    )
    setup_logging(cfg)
    setup_distributed(cfg)
    data = ObservedMediaData(pretrain_lm.setup_dataset(cfg))
    setup_training(cfg)
    rank = distributed.rank
    label = args.mode if args.mode != "negative" else "negative-" + args.negative_case
    report_path = args.output / f"{label}-rank-{rank:05d}.json"
    assert not report_path.exists(), "Retain previous evidence; use a fresh output"
    expected_path = args.output / f"continuous-rank-{rank:05d}.pt"
    report = dict(
        status="RUNNING",
        mode=args.mode,
        rank=rank,
        source=prepared,
        process_nonce=PROCESS_NONCE,
        pid=os.getpid(),
        hardware=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
    )
    actual = {
        spec.hf_name(name): parameter
        for module in training.model.module
        for name, parameter in module.named_parameters()
    }
    observer = OptimizerStepEvidence(
        actual, spec.training_groups(config_from_file(t.model)), reduce=dist.all_reduce
    )
    assert len(training.optimizers) == 1
    optimizer = training.optimizers[0]
    pre_hook, post_hook = (
        optimizer.register_step_pre_hook(observer.before_step),
        optimizer.register_step_post_hook(observer.after_step),
    )

    def observed_step(step):
        observation = ObjectiveObservation(pretrain_lm.objective)
        with patch.object(pretrain_lm, "objective", observation):
            pretrain_lm.train_step(cfg, data, step)
        return observation.outputs

    def complete_snapshot():
        state = snapshot(data)
        local_keys = sorted(state["model"])
        keys = [None] * args.world_size
        dist.all_gather_object(keys, local_keys)
        require_components(set().union(*(set(part) for part in keys)), spec.required_components)
        assert set(state) == {"model", "optimizer", "scheduler", "data", "rng"}
        assert set(state["rng"]) == {"torch_cpu", "python", "numpy", "cuda"}
        return state

    try:
        if args.mode == "negative":
            negative_case(args, cfg, data)
            report["all_ranks_rejected"] = True
        elif args.mode == "produce":
            for step in range(2):
                observed_step(step)
            assert find_checkpoint(t.save_location) == 2
            expected = dict(
                boundary=complete_snapshot(), producer_nonce=PROCESS_NONCE, producer_pid=os.getpid()
            )
            objectives = observed_step(2)
            expected.update(
                after=complete_snapshot(),
                next_batch=data.batches[2],
                third_objectives=objectives,
                third_module_updates=observer.steps[-1],
            )
            torch.save(expected, expected_path)
            assert len(observer.steps) == 3
        else:
            expected = torch.load(expected_path, weights_only=True)
            assert expected["producer_nonce"] != PROCESS_NONCE, (
                "Use a fresh Python/torchrun process"
            )
            assert all(not opt.state for opt in training.optimizers)
            with torch.no_grad():
                for parameter in training.model.parameters():
                    parameter.zero_()
            torch.manual_seed(9917 + rank)
            load_checkpoint(
                args.restore_root or args.output / "checkpoints",
                2,
                data_state=data.checkpoint_state,
                strict=True,
                resume_config=resume_contract(cfg),
            )
            boundary = complete_snapshot()
            _phase(
                "exact complete boundary",
                lambda: equal(boundary, expected["boundary"], f"rank-{rank}-boundary"),
            )
            objectives = observed_step(2)
            after = complete_snapshot()
            _phase(
                "exact next media input",
                lambda: equal(data.batches[2], expected["next_batch"], f"rank-{rank}-input"),
            )
            _phase(
                "exact third objective",
                lambda: equal(objectives, expected["third_objectives"], f"rank-{rank}-objective"),
            )
            _phase(
                "exact third state", lambda: equal(after, expected["after"], f"rank-{rank}-after")
            )
            assert len(observer.steps) == 1
            report["exact_restore"] = True
            report["exact_third_step"] = True
        assert data.consumed_samples == (0 if args.mode == "negative" else 24)
        report.update(
            status="PASSED",
            consumed_samples=data.consumed_samples,
            optimizer_step_evidence=observer.steps,
            required_components=spec.required_components,
            converted_checkpoint_root=str(args.restore_root) if args.restore_root else None,
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        pre_hook.remove()
        post_hook.remove()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()


def config_from_file(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    return getattr(config, "thinker_config", config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "produce", "restore", "negative"), required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--pp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--negative-case",
        choices=("missing-rank", "missing-cuda-rng", "shape", "topology", "optimizer-order"),
    )
    args = parser.parse_args(argv)
    if (args.mode == "negative") != (args.negative_case is not None):
        parser.error("--negative-case is required exactly for negative mode")
    if args.restore_root is not None and args.mode not in {"restore", "negative"}:
        parser.error("--restore-root requires restore/negative mode")
    if args.world_size < 1 or args.world_size % args.pp or (args.world_size // args.pp) % args.ep:
        parser.error("Invalid CP1 PP/EP world topology")
    dp = args.world_size // args.pp
    if dp not in (1, 2) or 8 // dp < 2 * args.pp:
        parser.error("Require DP1/2 and at least 2*PP local microbatches")
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if args.restore_root is not None:
        args.restore_root = args.restore_root.resolve()
    return args


def execute(args, spec):
    prepare(args, spec) if args.mode == "prepare" else run_gpu(args, spec)


if __name__ == "__main__":
    execute(parse_args(), audio_spec())
