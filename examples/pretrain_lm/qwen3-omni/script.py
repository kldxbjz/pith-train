"""Configure Qwen3-Omni data training through the shared pretrain_lm task."""

import argparse
import json
import math
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset root for the selected format"
    )
    parser.add_argument("--model", type=Path, required=True, help="Native model config directory")
    parser.add_argument(
        "--modalities", nargs="+", choices=("text", "image", "audio", "video"), default=["text"]
    )
    parser.add_argument(
        "--data-format", choices=("token_bin", "prepared_bundle"), default="prepared_bundle"
    )
    parser.add_argument(
        "--sampling-weights", type=json.loads, help='JSON object, e.g. {"text":1,"image":1}'
    )
    parser.add_argument(
        "--epoch-samples", type=int, help="Media draws per epoch; multiple of global batch"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=("wsd", "constant"), default="wsd")
    parser.add_argument("--start-lr", type=float, help="WSD starting LR (default: 1e-5)")
    parser.add_argument("--warmup-ratio", type=float, help="WSD warmup fraction (default: 0.03)")
    parser.add_argument("--final-lr", type=float, help="WSD final LR (default: 1e-5)")
    parser.add_argument("--decay-ratio", type=float, help="WSD decay fraction (default: 0.1)")
    parser.add_argument(
        "--decay-shape", choices=("cosine", "linear"), help="WSD decay (default: cosine)"
    )
    parser.add_argument(
        "--moe-load-balance-type",
        choices=("micro-batch", "global-batch", "sequence"),
        default="global-batch",
    )
    parser.add_argument("--moe-load-balance-coef", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-interval", type=int, default=256)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    if (
        min(
            args.sequence_length,
            args.global_batch_size,
            args.micro_batch_size,
            args.steps,
            args.save_interval,
            args.pp,
            args.cp,
            args.ep,
        )
        <= 0
    ):
        parser.error("Lengths, batch sizes, steps and parallel degrees must be positive")
    if args.sampling_weights is not None and not isinstance(args.sampling_weights, dict):
        parser.error("--sampling-weights must be a JSON object")

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    for name in ("weight_decay", "moe_load_balance_coef"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    scheduler_kwargs = dict(
        start_lr=args.start_lr,
        warmup_ratio=args.warmup_ratio,
        final_lr=args.final_lr,
        decay_ratio=args.decay_ratio,
        decay_shape=args.decay_shape,
    )
    if args.scheduler == "constant":
        if any(value is not None for value in scheduler_kwargs.values()):
            parser.error("WSD options require --scheduler wsd")
    else:
        defaults = dict(
            start_lr=1e-5, warmup_ratio=0.03, final_lr=1e-5, decay_ratio=0.1, decay_shape="cosine"
        )
        scheduler_kwargs = {
            name: defaults[name] if value is None else value
            for name, value in scheduler_kwargs.items()
        }
        for name in ("start_lr", "final_lr"):
            value = scheduler_kwargs[name]
            if not math.isfinite(value) or not 0 <= value <= args.lr:
                parser.error(f"--{name.replace('_', '-')} must be finite and between 0 and --lr")
        for name in ("warmup_ratio", "decay_ratio"):
            value = scheduler_kwargs[name]
            if not math.isfinite(value) or not 0 <= value <= 1:
                parser.error(f"--{name.replace('_', '-')} must be finite and between 0 and 1")
            if value > 0 and round(value * args.steps) == 0:
                parser.error(
                    f"--{name.replace('_', '-')} rounds to zero steps; use 0 to disable it"
                )
        if scheduler_kwargs["warmup_ratio"] + scheduler_kwargs["decay_ratio"] > 1:
            parser.error("WSD warmup and decay fractions must sum to at most 1")
        if (
            sum(
                round(scheduler_kwargs[name] * args.steps)
                for name in ("warmup_ratio", "decay_ratio")
            )
            > args.steps
        ):
            parser.error("Rounded WSD warmup and decay steps exceed --steps")

    # Keep --help usable on a machine without the CUDA training dependencies.
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        make_muon_optimizer,
        make_wsd_scheduler,
    )
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.data.dataset = args.dataset
    cfg.data.format = args.data_format
    cfg.data.modalities = tuple(args.modalities)
    cfg.data.sampling_weights = args.sampling_weights
    cfg.data.epoch_samples = args.epoch_samples
    cfg.data.num_workers = args.num_workers
    cfg.data.validate()
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    training = cfg.training
    training.model = args.model
    optimizer = {"adamw": make_adamw_optimizer, "muon": make_muon_optimizer}[args.optimizer]
    training.optimizer = partial(optimizer, weight_decay=args.weight_decay)
    training.scheduler = (
        partial(make_wsd_scheduler, **scheduler_kwargs)
        if args.scheduler == "wsd"
        else make_constant_scheduler
    )
    training.lr = args.lr
    training.seed = args.seed
    training.max_steps = args.steps
    training.micro_batch_size = args.micro_batch_size
    training.global_batch_size = args.global_batch_size
    training.sequence_length = args.sequence_length
    training.fp8 = False
    training.moe_load_balance_type = args.moe_load_balance_type
    training.moe_load_balance_coef = args.moe_load_balance_coef
    training.save_interval = args.save_interval
    training.save_location = args.checkpoint
    launch(cfg)


if __name__ == "__main__":
    main()
