"""GPU acceptance runner for the Omni data boundary (not an Omni encoder).

Run each arm in a NEW torchrun process. Reports retain full precision metrics,
input hashes and exact checkpoint-state hashes. --legacy also runs on the PR's
base archive so data/normalization changes are compared against unchanged code.
"""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch


def digest(value):
    """Hash every state entry, including shape/dtype, without storing a second checkpoint."""
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        value = value.detach().cpu().contiguous()
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    if isinstance(value, dict):
        return {str(key): digest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=1)
    args = parser.parse_args()

    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models.qwen3_moe import Qwen3MoeModel
    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler, setup_training
    from pithtrain.pipeline.execution import model_forward
    from pithtrain.tasks import pretrain_lm

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.dataset = args.dataset / "tokens/train" if args.legacy else args.dataset
    if not args.legacy:
        from pithtrain.modules.training_data import OmniDataCfg

        cfg.omni_data = OmniDataCfg()
        cfg.omni_data.stage = "video" if args.media else "text"
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.timeout = timedelta(minutes=3)
    t = cfg.training
    t.model = args.output / "model"
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length = 1e-5, args.steps, args.sequence_length
    t.global_batch_size, t.micro_batch_size = 8, 1
    t.fp8, t.moe_load_balance_coef = False, 0.01
    t.moe_load_balance_type = "global-batch"
    t.save_location = args.output / "checkpoints"
    setup_logging(cfg)
    setup_distributed(cfg)
    if distributed.rank == 0:
        base = (
            Path(pretrain_lm.__file__).parents[2] / "examples/pretrain_lm/qwen3-30b-a3b/config.json"
        )
        config = json.loads(base.read_text())
        config.update(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=max(4, 2 * args.pp),
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            vocab_size=152064,
        )
        t.model.mkdir(parents=True, exist_ok=True)
        (t.model / "config.json").write_text(json.dumps(config))
        args.report.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    # A test-only consumer accepts REAL processor outputs. Its scalar injection
    # exercises device transport and context association; it is not a vision/audio
    # encoder and cannot certify Omni semantics. No production capability is changed.
    seen, normal_calls, posemb_calls = Counter(), Counter(), Counter()
    view_checks = []

    def audit_method(method):
        def checked(self, *a, **kw):
            from torch.utils._pytree import tree_leaves

            outputs = method(self, *a, **kw)
            for output in tree_leaves(outputs):
                if (
                    not isinstance(output, torch.Tensor)
                    or not output.requires_grad
                    or output._base is None
                ):
                    continue
                record = dict(tensor=output, version=output._version, backward=False)
                view_checks.append(record)

                def before_backward(gradient, record=record):
                    tensor = record.pop("tensor")
                    assert tensor._version == record["version"], (
                        "A decoder view was modified in place"
                    )
                    record["backward"] = True
                    return gradient

                output.register_hook(before_backward)
            return outputs

        return checked

    original_prolog = Qwen3MoeModel.forward_prolog
    original_posemb = Qwen3MoeModel.forward_posemb

    def forward(self, inputs, cu_seqlens=None, model_context=None):
        normal_calls[self.stage_index] += 1
        return model_forward(self, inputs, self.chunk_record, cu_seqlens, model_context)

    def prolog(self, inputs, model_context=None):
        torch.testing.assert_close(inputs, model_context["input_ids"], rtol=0, atol=0)
        hidden = original_prolog(self, inputs)
        for name in ("pixel_values", "input_features", "pixel_values_videos"):
            if name in model_context:
                feature = model_context[name]
                assert feature.device == hidden.device and torch.isfinite(feature).all(), name
                assert feature.numel() > 0, name
                hidden = hidden + feature.float().mean().to(hidden.dtype) * 0.01
        return hidden

    def posemb(self, length, cu_seqlens=None, model_context=None):
        assert model_context["input_ids"].shape[1] == length
        for name, value in model_context.items():
            assert value.device == distributed.device, name
            if value.is_floating_point():
                assert value.dtype == torch.float32, (name, value.dtype)
        posemb_calls[self.stage_index] += 1
        return original_posemb(self, length, cu_seqlens)

    with ExitStack() as stack:
        if not args.legacy:
            from pithtrain.models.qwen3_moe import Qwen3MoeDecoderLayer

            # Validate the condition in FSDP's view warning instead of suppressing
            # it: every view's hook must run and its version must stay unchanged.
            for method in ("forward_stage1", "forward_stage3", "forward_stage5"):
                stack.enter_context(
                    patch.object(
                        Qwen3MoeDecoderLayer,
                        method,
                        audit_method(getattr(Qwen3MoeDecoderLayer, method)),
                    )
                )
        if args.media:
            stack.enter_context(
                patch.object(
                    Qwen3MoeModel,
                    "input_modalities",
                    {"text", "image", "audio", "video"},
                    create=True,
                )
            )
            for name, method in (
                ("forward", forward),
                ("forward_prolog", prolog),
                ("forward_posemb", posemb),
            ):
                stack.enter_context(patch.object(Qwen3MoeModel, name, method))
        data = pretrain_lm.setup_dataset(cfg)
        setup_training(cfg)

        def runtime_state():
            return digest(
                dict(
                    weights=dict(training.model.named_parameters()),
                    optimizers=[opt.state_dict() for opt in training.optimizers],
                    schedulers=[s.state_dict() for s in training.schedulers],
                    cuda_rng=torch.cuda.get_rng_state(),
                    data=None if args.legacy else data.state_dict(),
                )
            )

        start = 0
        expected = None
        if args.restore is not None:
            assert not args.legacy and args.expected is not None
            expected = json.loads((args.expected / f"rank{distributed.rank}.json").read_text())
            load_checkpoint(args.restore, args.checkpoint_step, data_state=data)
            assert runtime_state() == expected["checkpoint_state"], (
                "Fresh-process restored state differs"
            )
            start = args.checkpoint_step

        rows, batches = [], []
        original_batch = pretrain_lm.get_global_batch

        def get_batch(*a, **kw):
            result = original_batch(*a, **kw)
            batch = digest(
                [
                    (
                        mb.model_inputs,
                        mb.objective_inputs,
                        mb.cu_seqlens,
                        getattr(mb, "model_context", None),
                        getattr(mb, "sample_ids", ()),
                    )
                    for mb in result
                ]
            )
            batches.append(batch)
            if expected is not None:
                assert batch == expected["batches"][start + len(batches) - 1], (
                    "Restart changed next inputs/media"
                )
            for mb in result:
                context = getattr(mb, "model_context", None) or {}
                kind = "text"
                for field, modality in (
                    ("pixel_values", "image"),
                    ("input_features", "audio"),
                    ("pixel_values_videos", "video"),
                ):
                    if field in context:
                        kind = modality
                seen[kind] += 1
            return result

        def capture(metrics):
            row = {key: float(value) for key, value in metrics.items() if key.startswith("train/")}
            assert all(torch.isfinite(torch.tensor(v)) for v in row.values()), row
            assert row["train/gradient-norm"] > 0, row
            rows.append(row)

        stack.enter_context(patch.object(pretrain_lm, "get_global_batch", get_batch))
        stack.enter_context(patch.object(pretrain_lm, "activate_wandb", lambda _: None))
        stack.enter_context(patch.object(logging, "wandb", object()))
        stack.enter_context(patch.object(pretrain_lm.wandb, "log", capture))
        checkpoint_state = None
        initial = digest(dict(training.model.named_parameters()))
        for step in range(start, args.steps):
            pretrain_lm.train_step(cfg, data, step)
            if not args.legacy and args.restore is None and step + 1 == args.checkpoint_step:
                save_checkpoint(t.save_location, step + 1, data_state=data)
                checkpoint_state = runtime_state()
        final = digest(dict(training.model.named_parameters()))
        assert initial != final, "Training did not update weights"
        for parameter in training.model.parameters():
            assert torch.isfinite(parameter.to_local()).all()
        if not args.legacy:
            assert data.consumed_samples == args.steps * t.global_batch_size
        if args.media:
            all_seen = [None] * distributed.world_size
            torch.distributed.all_gather_object(all_seen, dict(seen))
            assert set().union(*(item.keys() for item in all_seen)) == {
                "text",
                "image",
                "audio",
                "video",
            }
            assert normal_calls and posemb_calls
            if args.pp > 1:
                assert sum(posemb_calls.values()) > sum(normal_calls.values()), (
                    "No overlap context calls"
                )
        assert all(record["backward"] for record in view_checks), (
            "A decoder view lost its backward hook"
        )
        report = dict(
            audited_view_hooks=len(view_checks),
            source_file=str(Path(pretrain_lm.__file__).resolve()),
            source_sha256=hashlib.sha256(Path(pretrain_lm.__file__).read_bytes()).hexdigest(),
            result="PASSED",
            start=start,
            steps=args.steps,
            rows=rows,
            batches=batches,
            checkpoint_state=checkpoint_state,
            exact_restore=expected is not None,
            modalities=dict(seen),
            normal_calls=dict(normal_calls),
            posemb_calls=dict(posemb_calls),
            config=cfg.training.to_json_dict(),
        )
        (args.report / f"rank{distributed.rank}.json").write_text(json.dumps(report, indent=2))
    torch.distributed.barrier()
    if distributed.rank == 0:
        print(
            json.dumps(
                dict(
                    result="PASSED",
                    report=str(args.report),
                    fresh_process_restore=expected is not None,
                )
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
