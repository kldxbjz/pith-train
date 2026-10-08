"""H100-only DualPipeV/FSDP image gradient gate; invoked by omni_image_training."""

import copy
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import global_loss_mean, global_target_count
from tests.acceptance.omni_image_evidence import (
    OptimizerStepEvidence,
    fixture_image_index,
    module_update_evidence,
    training_groups,
)
from tests.acceptance.omni_image_training import (
    HFReference,
    commit,
    error_ratio,
    hf_name,
    source_inputs,
)


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def expected_shard(value, parameter, name, distributed):
    if ".experts." in name:
        count = value.shape[0] // distributed.ep_size
        value = value.narrow(0, distributed.ep_rank * count, count)
    if isinstance(parameter, DTensor):
        value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
    return value


def run(args):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks import pretrain_lm

    world = int(os.environ["WORLD_SIZE"])
    assert world % args.pp == 0 and (world // args.pp) % args.ep == 0
    _, _, batches, config = source_inputs(args.dataset)
    rank = int(os.environ["RANK"])
    # Before the first barrier every rank checks that evidence is fresh.
    assert not args.output.exists(), "Use a fresh shared output"
    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.timeout = timedelta(seconds=90)
    t = cfg.training
    t.model = args.output / "model"
    t.native_vision, t.fp8, t.moe_load_balance_coef = True, False, 0.0
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-4,
        4,
        512,
        8,
        1,
    )
    t.save_interval = None
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.dataset,
        "prepared_bundle",
        ("image",),
    )
    cfg.data.num_workers = 0
    setup_logging(cfg)
    setup_distributed(cfg)
    if rank == 0:
        args.output.mkdir(parents=True)
        config.save_pretrained(t.model)
    dist.barrier()
    data = pretrain_lm.setup_dataset(cfg)
    setup_training(cfg)
    reference = HFReference(config, torch.bfloat16).to(distributed.device)
    expected = dict(reference.named_parameters())
    actual = {}
    for module in training.model.module:
        for name, parameter in module.named_parameters():
            key = hf_name(name)
            assert key not in actual and key in expected, key
            actual[key] = parameter
            with torch.no_grad():
                parameter.copy_(expected_shard(expected[key].detach(), parameter, key, distributed))
    owned_keys = [None] * world
    dist.all_gather_object(owned_keys, sorted(actual))
    assert set().union(*(set(keys) for keys in owned_keys)) == set(expected)

    # Fixed real-image batches isolate staged math and expose nonuniform/zero
    # local targets. DP coordinates use separate fixture entries; PP replicas
    # see identical shared layout but only stage 0 sees pixels.
    global_batches = []
    for index in range(8):
        batch = copy.deepcopy(batches[fixture_image_index(index, distributed.dp_size)])
        batch.labels = batch.labels.clone()
        if index == 0:
            batch.labels.fill_(-100)
        elif index % 3 == 0:
            batch.labels[:, ::2] = -100
        global_batches.append(batch)
    local_batches = global_batches[distributed.dp_rank :: distributed.dp_size]
    microbatches = []
    for batch in local_batches:
        inputs = {k: v.to(distributed.device) for k, v in batch.model_inputs.items()}
        pixels = inputs.pop("pixel_values")
        microbatches.append(
            Microbatch(
                model_inputs=(inputs["input_ids"],),
                cu_seqlens=None,
                objective_inputs=(batch.labels.to(distributed.device),),
                model_context=inputs,
                media_inputs={"pixel_values": pixels} if distributed.pp_rank == 0 else None,
                sample_ids=batch.sample_ids,
            )
        )
    group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
    count = global_target_count(microbatches, group)
    assert int(count) == sum(int((b.labels != -100).sum()) for b in global_batches)
    # Independent HF processes all eight global samples. No PP or DP replicas
    # are counted in its denominator.
    hf_loss = 0
    for batch in global_batches:
        inputs = {k: v.to(distributed.device) for k, v in batch.model_inputs.items()}
        logits = reference(inputs)
        labels = batch.labels.to(distributed.device)
        loss = (
            F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum") / count
        )
        loss.backward()
        hf_loss += loss.detach()
    outputs = training.model.step(microbatches, pretrain_lm.objective)
    if distributed.pp_rank == 0:
        loss = global_loss_mean(torch.stack(outputs).sum(), count, group)
        torch.testing.assert_close(loss, hf_loss, rtol=1e-3, atol=1e-3)
    errors = {}
    for key, parameter in actual.items():
        assert parameter.grad is not None and expected[key].grad is not None, key
        parameter.grad.div_(count)
        target = expected_shard(expected[key].grad, parameter, key, distributed)
        errors[key] = error_ratio(local_tensor(parameter.grad), local_tensor(target))
        assert errors[key] < 1e-2, (key, errors[key])
    before = {n: local_tensor(p).detach().clone() for n, p in actual.items()}
    for optimizer in training.optimizers:
        optimizer.step()
    updated = [n for n, p in actual.items() if not torch.equal(before[n], local_tensor(p))]
    groups = training_groups(config)
    modules = module_update_evidence(actual, before, groups, reduce=dist.all_reduce)
    for optimizer in training.optimizers:
        optimizer.zero_grad(set_to_none=True)
    zero = [
        Microbatch(
            model_inputs=mb.model_inputs,
            cu_seqlens=None,
            objective_inputs=(torch.full_like(mb.objective_inputs[0], -100),),
        )
        for mb in microbatches
    ]
    try:
        global_target_count(zero, group)
    except ValueError as error:
        zero_error = str(error)
    else:
        raise AssertionError("All-zero targets accepted")
    all_errors = [None] * world
    dist.all_gather_object(all_errors, zero_error)
    assert len(set(all_errors)) == 1
    # Observe every real task optimizer step, after backward/gradient scaling,
    # without replacing train_step, model execution or optimizer behavior.
    assert len(training.optimizers) == 1
    observer = OptimizerStepEvidence(actual, groups, reduce=dist.all_reduce)
    optimizer = training.optimizers[0]
    pre_hook = optimizer.register_step_pre_hook(observer.before_step)
    post_hook = optimizer.register_step_post_hook(observer.after_step)
    try:
        for step in range(4):
            pretrain_lm.train_step(cfg, data, step)
    finally:
        pre_hook.remove()
        post_hook.remove()
    assert len(observer.steps) == 4
    assert data.consumed_samples == 32
    report = dict(
        status="PASSED",
        commit=commit(),
        rank=rank,
        pp=args.pp,
        ep=args.ep,
        target_count=int(count),
        gradient_errors=errors,
        updated_parameters=updated,
        training_modules=modules,
        real_reader_module_updates=observer.steps,
        real_reader_steps=4,
        consumed_samples=32,
        zero_target_rejection=zero_error,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.output / f"rank-{rank:05d}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
