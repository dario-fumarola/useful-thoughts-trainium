"""One ordinary-causal production driver; long runs require real qualification.

--qualify is a bounded plumbing/timing path. It must never be labeled a
scientific result. Use xmp for partitioned runs on the installed Neuron stack.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from weekend_thinking.artifacts import (atomic_json, canonical_hash, sha256_file,
                                        source_fingerprint, utc_now)
from weekend_thinking.checkpoints import capture_cpu_rng, restore_cpu_rng
from weekend_thinking.core import Format, loo_advantages
from weekend_thinking.data import load_records
from weekend_thinking.experiment import validate_qualification
from weekend_thinking.order import seed_order, distributed_batch_indices
from weekend_thinking.pretrained import load_base
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.runtime_policy import (MAX_CHECKPOINT_INTERVAL_SECONDS, checkpoint_due,
                                             require_runtime_policy, runtime_policy)


def validate_fixed_recipe(proposal, seed):
    """Fail rather than silently ignore a changed experiment's settings."""
    thought, opt = proposal['thought_arm'], proposal['optimization']
    expected_thought = dict(algorithm='one_fresh_signed_group_relative_policy_update',
        group_size=4, max_thought_tokens=1024,
        reward_horizon=1, temperature=.7, normalization='sequence',
        group_std_normalization=False, negative_advantage_scaling=1.,
        best_of_selection_for_training=False, reader_gradient=False,
        ntp_auxiliary_weight=0., reference_kl_weight=0.)
    expected_opt = dict(parameter_dtype='fp32', compute_dtype='fp32', learning_rate=1e-6,
        schedule='constant', betas=[.9,.95], weight_decay=.1,
        gradient_clip=1., micro_prefixes_per_rank=1)
    for actual, expected, label in [(thought,expected_thought,'thought_arm'),
                                     (opt,expected_opt,'optimization')]:
        for name,value in expected.items():
            if actual.get(name) != value:
                raise ValueError(f'Unsupported frozen setting: {label}.{name}')
    if proposal['model']['id'] != 'Qwen/Qwen3-1.7B-Base':
        raise ValueError('This driver must separately qualify a model fallback')
    if seed not in opt['seeds']:
        raise ValueError('Training seed is absent from frozen plan')


def cpu_state(value):
    """Bound transfer staging for large embeddings, without a BytesIO buffer."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.device.type == 'cpu':
            return value
        flat = value.reshape(-1)
        cpu = torch.empty(flat.shape, dtype=value.dtype)
        for start in range(0, flat.numel(), 8*1024*1024):
            cpu[start:start+8*1024*1024].copy_(flat[start:start+8*1024*1024].cpu())
        return cpu.reshape(value.shape)
    if isinstance(value, dict):
        return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def stream_save(path, value):
    temp = path.with_name('.'+path.name+'.incomplete')
    if path.exists():
        raise FileExistsError(path)
    with temp.open('xb') as f:
        torch.save(cpu_state(value), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save_checkpoint(output, model, optimizer, step, state, rank, world, xm, device,
                    *, checkpoint_root=None):
    # Call only at a settled update/load boundary, with no pending random IR.
    # Checkpoint observation (including the qualification readback) must not
    # change the next draw, even if an export or runtime barrier advances RNG.
    rng = dict(cpu=capture_cpu_rng(), device=int(xm.get_rng_state(device)))
    directory = (output if checkpoint_root is None else checkpoint_root)/f'step-{step:07d}'
    directory.mkdir(parents=True, exist_ok=True)
    xm.rendezvous(f'checkpoint-{step}-start')
    if rank == 0:
        # Tied weights use a single CPU tensor storage in the saved dictionary.
        weights = cpu_state(model.state_dict())
        if model.config.tie_word_embeddings:
            weights['lm_head.weight'] = weights['model.embed_tokens.weight']
        stream_save(directory/'model.pt', {'model': weights, 'step': step,
                                         'run_id': output.name, 'spec_hash': state['spec_hash']})
        del weights
    stream_save(directory/f'rank-{rank:05d}.pt',
                dict(optimizer=optimizer.state_dict(), rng=rng, step=step,
                     rank=rank, world_size=world, **state))
    xm.rendezvous(f'checkpoint-{step}-written')
    if rank == 0:
        paths = [directory/'model.pt'] + [directory/f'rank-{r:05d}.pt' for r in range(world)]
        files = {p.name:dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in paths}
        atomic_json(directory/'COMPLETE.json', dict(schema='neuron-v1', step=step,
            run_id=output.name, world_size=world, spec_hash=state['spec_hash'],
            files=files, created_utc=utc_now(), metadata=state))
    xm.rendezvous(f'checkpoint-{step}-complete')
    restore_cpu_rng(rng['cpu'])
    xm.set_rng_state(rng['device'], device)
    return directory


def restore_rng_after_sync(rng, xm, device, sync):
    """Finish device loads before resetting the continuation's exact RNG state.

    XLA MarkStep advances the seed even without a random operation. No sync may
    follow this reset before the resumed program's ordinary continuation.
    """
    sync(wait=True)
    restore_cpu_rng(rng['cpu'])
    xm.set_rng_state(int(rng['device']), device)


def verify_checkpoint(directory, expected_hash, world):
    manifest = json.loads((directory/'COMPLETE.json').read_text())
    if manifest.get('schema') != 'neuron-v1' or not isinstance(manifest.get('step'), int) or manifest['step'] < 0:
        raise ValueError('Invalid checkpoint schema/step')
    if manifest['spec_hash'] != expected_hash or manifest['world_size'] != world:
        raise ValueError('Resume spec/topology mismatch')
    expected = {'model.pt'} | {f'rank-{r:05d}.pt' for r in range(world)}
    if set(manifest['files']) != expected:
        raise ValueError('Checkpoint file set mismatch')
    for name, entry in manifest['files'].items():
        path = directory/name
        if path.is_symlink() or path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError(f'Checkpoint integrity mismatch: {name}')
    return manifest


def validate_resume_state(manifest, weights, saved, *, run_id, spec_hash, world,
                          rank, order_hash, arm, seed, batch, accumulation,
                          precision, execution_policy=None):
    """Reject mixed steps, ranks and C/R lineages before mutating a model.

    The frozen experiment and record order are intentionally shared by paired
    C/R arms. Their hashes therefore cannot identify an arm or its optimizer.
    """
    step = manifest['step']
    if manifest['run_id'] != run_id or manifest['spec_hash'] != spec_hash or manifest['world_size'] != world:
        raise ValueError('Resume manifest belongs to a different run/spec/topology')
    for key, expected in dict(step=step, run_id=run_id, spec_hash=spec_hash).items():
        if weights.get(key) != expected:
            raise ValueError(f'Resume model identity mismatch: {key}')
    expected_shard = dict(step=step, rank=rank, world_size=world,
                          spec_hash=spec_hash, order_hash=order_hash, arm=arm,
                          seed=seed, global_batch=batch, accumulation=accumulation,
                          parameter_dtype='fp32', compute_dtype=precision)
    for key, expected in expected_shard.items():
        if saved.get(key) != expected:
            raise ValueError(f'Resume shard identity mismatch: {key}')
    if execution_policy is not None:
        require_runtime_policy(saved.get('runtime_policy'), execution_policy)
    optimizer = saved.get('optimizer', {})
    if not optimizer.get('sharded_master_weights') or not optimizer.get('base_state'):
        raise ValueError('Resume lacks FP32 masters or initialized optimizer moments')
    counters = saved.get('counters', {})
    if counters.get('source_records') != step * batch or any(
            not isinstance(counters.get(key), int) or counters[key] < 0
            for key in ('source_records', 'source_tokens', 'sampled_actions')):
        raise ValueError('Resume counters do not match the completed step')


def validate_resume_run(output, *, arm, seed, spec_hash, execution_policy=None):
    run = json.loads((output/'run.json').read_text())
    for key, expected in dict(run_id=output.name, arm=arm, seed=seed,
                              spec_hash=spec_hash).items():
        if run.get(key) != expected:
            raise ValueError(f'Resume output identity mismatch: {key}')
    if execution_policy is not None:
        require_runtime_policy(run.get('runtime_policy'), execution_policy)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['vanilla','thought'], required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--qualification', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--qualify', action='store_true')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--records', type=Path)
    p.add_argument('--format', type=Path)
    p.add_argument('--max-steps', type=int, default=1)
    p.add_argument('--global-prefixes', type=int, default=128)
    p.add_argument('--diagnostic-cap', type=int, default=1024)
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--processes', type=int, default=1)
    p.add_argument('--checkpoint-interval-seconds', type=int,
                   default=MAX_CHECKPOINT_INTERVAL_SECONDS)
    a = p.parse_args()
    if not 1 <= a.checkpoint_interval_seconds <= MAX_CHECKPOINT_INTERVAL_SECONDS:
        p.error('Checkpoint interval must be 1..43200 seconds')
    if a.qualify:
        if not 1 <= a.max_steps <= 32 or not 1 <= a.global_prefixes <= 128:
            p.error('Qualification bounded to 32 updates and 128 prefixes/update')
        if any(v is None for v in (a.model_dir,a.records,a.format)):
            p.error('Qualification needs explicit model/records/format')
        if not 1 <= a.diagnostic_cap <= 1024:
            p.error('Diagnostic cap must be 1..1024')
    elif a.frozen is None or a.qualification is None:
        p.error('Scientific runs need a frozen experiment and hardware receipt')
    if a.processes < 1 or a.global_prefixes % a.processes:
        p.error('Global batch must be divisible by processes')
    if 'TORCHELASTIC_RUN_ID' in os.environ:
        p.error('Use xmp with --processes; torchrun overrides core partitioning here')
    return a


def worker(_index, a):
    import torch.distributed as dist
    import torch_xla
    import torch_xla.amp
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
    from train.runtime_compat import configure_runtime, restore_syncfree_device_state
    from torch_xla.debug import metrics
    compat = configure_runtime()
    execution_policy = runtime_policy()
    if not hasattr(torch, 'xla'):
        torch._register_device_module('xla', torch_xla)
    device = torch_xla.device()
    rank, world = xr.global_ordinal(), xr.world_size()
    if world != a.processes:
        raise RuntimeError(f'Runtime world {world} differs from requested {a.processes}')
    if not dist.is_initialized():
        dist.init_process_group('xla', init_method='xla://')
    if a.qualify:
        model_dir, records_path, fmt_path = a.model_dir, a.records, a.format
        steps, batch, cap = a.max_steps, a.global_prefixes, a.diagnostic_cap
        spec = dict(stage='qualification', arm=a.arm, seed=a.seed, steps=steps,
            global_prefixes=batch, cap=cap, precision=a.precision, world_size=world,
            model_manifest_sha256=sha256_file(model_dir/'ASSET_MANIFEST.json'),
            records_sha256=sha256_file(records_path), format_sha256=sha256_file(fmt_path),
            source=source_fingerprint(Path.cwd()), learning_rate=1e-6,
            betas=[.9,.95], eps=1e-8, weight_decay=.1,
            runtime_policy=execution_policy,
            checkpoint_interval_seconds=a.checkpoint_interval_seconds)
        deadline = None
    else:
        frozen = json.loads(a.frozen.read_text())
        receipt = json.loads(a.qualification.read_text())
        validate_fixed_recipe(frozen['proposal'], a.seed)
        require_runtime_policy(frozen['site'].get('runtime_policy_by_arm', {}).get(a.arm),
                               execution_policy)
        if frozen['site'].get('checkpoint_interval_seconds') != a.checkpoint_interval_seconds:
            raise ValueError('Checkpoint interval differs from frozen plan')
        if rank == 0:
            validate_qualification(Path.cwd(), frozen, receipt)
        xm.rendezvous('qualified')
        model_dir = Path(frozen['assets']['model_dir'])
        records_path = Path(frozen['assets']['records_file'])
        fmt_path = Path(frozen['assets']['format_file'])
        steps = frozen['terminal_steps']
        batch = frozen['proposal']['optimization']['global_unique_prefixes_per_update']
        cap = frozen['proposal']['thought_arm']['max_thought_tokens']
        if (world != frozen['site']['world_size_per_arm'] or
                a.precision != frozen['proposal']['optimization']['compute_dtype']):
            raise ValueError('Production topology/precision differs from frozen plan')
        spec = frozen
        deadline = datetime.fromisoformat(frozen['deadline_utc'])
    spec_hash = canonical_hash(spec) if a.qualify else frozen['spec_hash']
    if batch % world:
        raise ValueError('Batch does not divide the world')
    accumulation = batch//world
    if rank == 0:
        if a.resume is None:
            a.output.mkdir(parents=True, exist_ok=False)
            atomic_json(a.output/'run.json', dict(spec=spec, spec_hash=spec_hash,
                run_id=a.output.name, arm=a.arm, seed=a.seed,
                started_utc=utc_now(), compatibility=compat,
                runtime_policy=execution_policy))
        elif not a.output.exists():
            raise ValueError('Resume requires the original run output')
        else:
            validate_resume_run(a.output, arm=a.arm, seed=a.seed, spec_hash=spec_hash,
                                execution_policy=execution_policy)
    xm.rendezvous('output-ready')
    records = load_records(records_path, split='train')
    order = seed_order(records, a.seed)
    order_hash = canonical_hash([records[i].id for i in order])
    if steps * batch > len(order):
        raise ValueError('Insufficient fresh records; cycling is forbidden')
    random.seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    torch.manual_seed(a.seed)
    model, tok = load_base(model_dir)
    model.to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError('Released embedding tie lost')
    fmt = Format.from_dict(json.loads(fmt_path.read_text())['format'])
    valid = set(tok.get_vocab().values())
    banned = tuple(i for i in range(model.config.vocab_size) if i not in valid)
    amp = nullcontext if a.precision == 'fp32' else lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)
    backend = NeuronCausalBackend(model, amp=amp, sync=lambda: torch_xla.sync(), banned_ids=banned)
    # The recipe is constant LR. A Python float also matches the installed
    # syncfree kernel's SupportsFloat contract and avoids device scalar reads.
    lr = 1e-6
    optimizer = ZeroRedundancyOptimizer(model.parameters(), AdamW,
        optimizer_dtype=torch.float32, grad_clipping=True, max_norm=1.,
        save_master_weights=True, lr=lr, betas=(.9,.95), eps=1e-8, weight_decay=.1)
    found_inf = torch.zeros((), device=device)
    start_step, counters = 0, dict(source_records=0, source_tokens=0, sampled_actions=0)
    xm.set_rng_state(a.seed + 100003*rank, device)
    if a.resume is not None:
        if rank == 0:
            manifest = verify_checkpoint(a.resume, spec_hash, world)
        xm.rendezvous('resume-integrity-verified')
        saved = torch.load(a.resume/f'rank-{rank:05d}.pt', weights_only=True)
        weights = torch.load(a.resume/'model.pt', weights_only=True)
        if rank != 0:
            manifest = json.loads((a.resume/'COMPLETE.json').read_text())
        validate_resume_state(manifest, weights, saved, run_id=a.output.name,
            spec_hash=spec_hash, world=world, rank=rank, order_hash=order_hash,
            arm=a.arm, seed=a.seed, batch=batch, accumulation=accumulation,
            precision=a.precision, execution_policy=execution_policy)
        model.load_state_dict(weights['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_syncfree_device_state(optimizer)
        start_step, counters = saved['step'], saved['counters']
        restore_rng_after_sync(saved['rng'], xm, device, torch_xla.sync)
    else:
        torch_xla.sync(wait=True)
    def checkpoint_state():
        return dict(spec_hash=spec_hash, order_hash=order_hash, counters=counters.copy(),
                    arm=a.arm, seed=a.seed, global_batch=batch, accumulation=accumulation,
                    banned_ids=list(banned), parameter_dtype='fp32', compute_dtype=a.precision,
                    runtime_policy=execution_policy)
    if a.qualify and a.resume is not None:
        # Read back ACTUAL restored device state before the next sampling call.
        # Merely comparing cloned input files would not verify the restore.
        save_checkpoint(a.output, model, optimizer, start_step, checkpoint_state(),
                        rank, world, xm, device,
                        checkpoint_root=a.output/'resume-restored')
    def parameter_sample():
        return torch.cat([p.detach().reshape(-1)[:32] for p in model.parameters()]).cpu()
    last_checkpoint_time = time.monotonic()
    for step in range(start_step, steps):
        started = time.monotonic()
        if rank == 0:
            print(f'{utc_now()} rank0 {a.arm} update {step+1}/{steps}', flush=True)
        optimizer.zero_grad(set_to_none=True)
        before = parameter_sample()
        replay_error = 0.0
        replay_audit_performed = False
        local = dict(loss=0., clean=0., thought=0., gain=0., actions=0., capped=0.,
                     abs_advantage=0., mean_signed_length=0., padded_decode_slots=0.)
        for micro, index in enumerate(distributed_batch_indices(step, rank, world, accumulation)):
            record = records[order[index]]
            prefix = torch.tensor([record.prefix], device=device)
            future = torch.tensor([record.future], device=device)
            if a.arm == 'vanilla':
                loss = backend.vanilla_loss(prefix, future)/accumulation
                if not np.isfinite(float(loss.detach().cpu())):
                    raise RuntimeError('Nonfinite continuation loss before backward')
                loss.backward()
                torch_xla.sync(wait=True)
                local['loss'] += float(loss.detach().cpu())
            else:
                rollout = backend.sample(prefix, fmt=fmt, group_size=4, max_actions=cap,
                    temperature=.7, policy_version=step)
                thought = backend.score(prefix, future, rollout, fmt=fmt, horizon=1)
                clean = backend.clean_score(prefix, future)
                gain = thought - clean[:,None]
                advantage = loo_advantages(gain)
                # Device materialization detaches the reward before any replay.
                torch_xla.sync(wait=True)
                if not bool(torch.isfinite(thought).all().cpu()) or not bool(torch.isfinite(clean).all().cpu()):
                    raise RuntimeError('Nonfinite reward before policy replay')
                if micro == 0 and (a.qualify or step % 64 == 0):
                    replay_audit_performed = True
                    # Recheck after real updates on rank-distinct real data.
                    # The qualification timing includes this extra audit work.
                    with torch.no_grad():
                        replay_lp = backend.replay(prefix, rollout, fmt=fmt, temperature=.7)
                        error = torch.where(rollout.action_mask,
                            (replay_lp-rollout.behavior_logps).abs(), 0.).max()
                        replay_error = float(error.cpu())
                    tolerance = .001 if a.precision == 'fp32' else .2
                    if not np.isfinite(replay_error) or replay_error > tolerance:
                        raise RuntimeError(f'Post-update cache/replay mismatch: {replay_error}')
                for g in range(4):
                    loss = backend.replay_one_loss(prefix, rollout.actions[:,g],
                        rollout.action_mask[:,g], advantage[0,g], fmt=fmt,
                        temperature=.7, divisor=4*accumulation)
                    if not np.isfinite(float(loss.detach().cpu())):
                        raise RuntimeError('Nonfinite policy loss before backward')
                    loss.backward()
                    torch_xla.sync(wait=True)
                    local['loss'] += float(loss.detach().cpu())
                lengths = rollout.action_mask.sum(-1)
                local['padded_decode_slots'] += 4 * rollout.decode_iterations
                stops = rollout.actions.eq(fmt.stop_id)
                for stop in fmt.additional_stop_ids:
                    stops = stops | rollout.actions.eq(stop)
                for key,value in [('clean',clean.mean()),('thought',thought.mean()),
                    ('gain',gain.mean()),('actions',lengths.float().sum()),
                    ('capped',(~stops.any(-1)).float().mean()),
                    ('abs_advantage',advantage.abs().mean()),
                    ('mean_signed_length',(advantage*lengths).mean())]:
                    local[key] += float(value.cpu())
                if a.qualify:
                    # Persist AFTER this micro's backwards and synchronization.
                    # Diagnostic transfers must not change program residency
                    # before the memory-sensitive backward being qualified.
                    tape_dir = a.output/'qualification-tapes'/f'step-{step+1:07d}'
                    tape_dir.mkdir(parents=True, exist_ok=True)
                    stream_save(tape_dir/f'rank-{rank:05d}-micro-{micro:03d}.pt',
                        dict(schema='qualification-rollout-v1', run_id=a.output.name,
                             spec_hash=spec_hash, order_hash=order_hash, arm=a.arm,
                             seed=a.seed, step=step+1, rank=rank, world_size=world,
                             micro=micro, global_batch=batch, accumulation=accumulation,
                             record_id=record.id, ordered_record_index=index,
                             policy_version=rollout.policy_version, temperature=.7,
                             decode_iterations=rollout.decode_iterations,
                             actions=rollout.actions, action_mask=rollout.action_mask,
                             behavior_logps=rollout.behavior_logps,
                             clean=clean, thought=thought, advantages=advantage,
                             lengths=lengths))
                if rank == 0 and micro == 0:
                    host_actions = rollout.actions.cpu()
                    host_mask = rollout.action_mask.cpu()
                    sample = dict(step=step+1, record_id=record.id, split=record.split,
                        stream='training_diagnostic', ordered_record_index=index,
                        policy_version=step, temperature=.7,
                        decode_iterations=rollout.decode_iterations,
                        prefix=tok.decode(record.prefix), future=tok.decode(record.future[:1]),
                        actions=host_actions.tolist(), action_mask=host_mask.tolist(),
                        text=[tok.decode(host_actions[0,g][host_mask[0,g]].tolist(),
                                         skip_special_tokens=False) for g in range(4)],
                        clean=clean.cpu().tolist(), thought=thought.cpu().tolist(),
                        advantages=advantage.cpu().tolist())
                    atomic_json(a.output/f'train-trace-{step+1:07d}.json',sample)
        # This is the single reduction AND global clipping owner.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                              if p.grad is not None]).all()
        bad = xm.all_reduce(xm.REDUCE_MAX, (~finite).float())
        if bool(bad.cpu()):
            raise RuntimeError('Nonfinite gradient detected before optimizer update')
        optimizer.step(found_inf=found_inf)
        torch_xla.sync(wait=True)
        norm = float(optimizer.grad_norm.cpu())
        if not np.isfinite(norm):
            raise RuntimeError('Nonfinite reduced gradient; no scientific continuation')
        after = parameter_sample()
        replicas = xm.all_gather(after.to(device)).reshape(world, -1).cpu()
        if not torch.equal(replicas, replicas[0:1].expand_as(replicas)):
            raise RuntimeError('Rank replicas disagree on fixed parameter sample')
        delta = after-before
        sampled_update = dict(sampled_elements=delta.numel(),
            sampled_update_l2=float(delta.norm()),
            sampled_changed_fraction=float(delta.ne(0).float().mean()),
            sampled_replicas_bitwise_equal=True)
        if not bool(torch.isfinite(after).all()) or not bool(delta.ne(0).any()):
            raise RuntimeError('FP32 parameters nonfinite or no sampled values changed')
        # Metrics only: this does not touch gradients or double-reduce them.
        keys = sorted(local)
        totals = xm.all_reduce(xm.REDUCE_SUM, torch.tensor([local[k] for k in keys], device=device))
        values = dict(zip(keys, totals.cpu().tolist()))
        max_replay_error = float(xm.all_reduce(xm.REDUCE_MAX,
            torch.tensor(replay_error, device=device)).cpu())
        counts = values.copy()
        values['loss'] /= world
        for key in ('clean','thought','gain','capped','abs_advantage','mean_signed_length'):
            values[key] /= batch
        if a.arm == 'vanilla':
            # No thought reader or reward was evaluated for C. Zero would
            # falsely represent a measured likelihood or utility.
            for key in ('clean','thought','gain','capped','abs_advantage','mean_signed_length'):
                values[key] = None
        counters['source_records'] += batch
        counters['source_tokens'] += batch * (len(record.prefix)+1)
        counters['sampled_actions'] += int(counts['actions'])
        elapsed = time.monotonic()-started
        log = dict(step=step+1, arm=a.arm, elapsed_seconds=elapsed, gradient_norm=norm,
                   metrics=values,
                   max_replay_logp_error=max_replay_error if replay_audit_performed else None,
                   replay_audit_performed=replay_audit_performed,
                   sampled_parameter_update=sampled_update,
                   counters=counters.copy(), time_utc=utc_now(),
                   status='qualification' if a.qualify else 'science')
        if rank == 0:
            with (a.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(log, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps(log), flush=True)
        # Decisions are synchronized so every rank checkpoints/stops together.
        should_stop = ((a.stop_file is not None and a.stop_file.exists()) or
                       (deadline is not None and datetime.now(timezone.utc) >= deadline))
        decision = torch.tensor(int(should_stop), device=device)
        should_stop = bool(xm.all_reduce(xm.REDUCE_MAX, decision).cpu())
        save_due = checkpoint_due(step=step+1, total_steps=steps,
            elapsed_seconds=time.monotonic()-last_checkpoint_time,
            interval_seconds=a.checkpoint_interval_seconds,
            resume_probe=a.qualify and a.arm == 'thought')
        save_due = bool(xm.all_reduce(xm.REDUCE_MAX, torch.tensor(int(save_due), device=device)).cpu())
        if save_due or should_stop:
            if shutil.disk_usage(a.output).free < 35*1024**3:
                raise RuntimeError('Less than 35 GiB free before checkpoint; preserve previous checkpoint')
            save_checkpoint(a.output, model, optimizer, step+1,
                            checkpoint_state(), rank, world, xm, device)
            last_checkpoint_time = time.monotonic()
            if rank == 0:
                (a.output/f'xla-metrics-{step+1:07d}.txt').write_text(metrics.metrics_report())
        if should_stop:
            break
    dist.destroy_process_group()


def main():
    a = parse_args()
    if a.processes > 1:
        if os.environ.get('NEURONCORE_NUM_DEVICES') != str(a.processes):
            raise ValueError('Set NEURONCORE_NUM_DEVICES to requested process count')
        if not os.environ.get('NEURON_RT_ROOT_COMM_ID'):
            raise ValueError('Set a dedicated NEURON_RT_ROOT_COMM_ID endpoint')
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(worker, args=(a,), nprocs=None)
    else:
        worker(0,a)


if __name__ == '__main__':
    main()
