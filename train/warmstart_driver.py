"""Short disclosed TRAIN-label warmstart/control diagnostic, independently from B0.

No sampling, resume, candidate selection, or claim of deployed-policy capability.
Each rank consumes one example; the sole ZeRO step owns mean reduction/clipping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("warmstart", "raw_control")
REQUIRED_SOURCES = {
    "train/warmstart_driver.py", "train/warmstart_data.py", "train/masked_causal_ce.py",
    "train/neuron_backend.py", "train/neuron_driver.py", "train/runtime_compat.py",
    "weekend_thinking/pretrained.py", "weekend_thinking/artifacts.py",
    "weekend_thinking/core.py", "weekend_thinking/runtime_policy.py",
}


def order_index(position, count, seed):
    if type(position) is not int or position < 0 or type(count) is not int or count < 1:
        raise ValueError("Require a nonnegative cursor and nonempty dataset")
    epoch, offset = divmod(position, count)
    indices = list(range(count))
    random.Random(seed + epoch).shuffle(indices)
    return indices[offset]


def stop_due(stop_file, deadline, now=None):
    return stop_file.exists() or (now or datetime.now(timezone.utc)) >= deadline


def validate_source_files(files):
    bound = set()
    for name, digest in files.items():
        path = (ROOT / name).resolve()
        if sha256_file(path) != digest:
            raise ValueError(f"Changed source {name}")
        bound.add(str(path.relative_to(ROOT)))
    if not REQUIRED_SOURCES <= bound:
        raise ValueError("Protocol omits a required training/packing/loss source")


def validate_protocol(p, processes):
    from weekend_thinking.runtime_policy import POLICY_ENVIRONMENT
    expected = dict(seed=20261018, world_size=8, steps=64, lr=1e-5,
                    betas=[.9, .95], eps=1e-8, weight_decay=.1, clip=1.,
                    note_weight=1., future_weight=1.)
    if processes != 8:
        raise ValueError("This new diagnostic requires exactly eight processes")
    for name, value in expected.items():
        if isinstance(p.get(name), bool) or p.get(name) != value:
            raise ValueError(f"Unsupported frozen setting: {name}")
    for name in ("seed", "world_size", "steps"):
        if type(p[name]) is not int:
            raise ValueError(f"Expected integer {name}")
    if p.get("global_batch", 8) != 8 or p.get("accumulation", 1) != 1 or p.get("precision", "fp32") != "fp32":
        raise ValueError("Require FP32/global8/one example per rank without accumulation")
    policy = p.get("runtime_policy", {})
    if (policy.get("schema") != 1 or set(policy.get("environment", {})) != set(POLICY_ENVIRONMENT)
            or policy["environment"].get("XLA_COMPILATION_CACHE_SIZE") != "2"):
        raise ValueError("The first short diagnostic uses the measured LRU2 policy")
    for name in ("data_sha256", "dataset_sha256", "format_sha256", "model_manifest_sha256"):
        if not isinstance(p.get(name), str) or len(p[name]) != 64:
            raise ValueError(f"Missing exact {name}")


def load_inputs(p):
    from train.evaluate import asset_provenance
    from train.warmstart_data import pack_example
    from weekend_thinking.core import Format

    for key, digest in (("data_file", "data_sha256"), ("dataset_file", "dataset_sha256"),
                        ("format_file", "format_sha256")):
        if sha256_file(p[key]) != p[digest]:
            raise ValueError(f"Changed {key}")
    validate_source_files(p["source_files"])
    model = asset_provenance(Path(p["model_dir"]))
    if model.get("model_id") != "Qwen/Qwen3-1.7B-Base" or model["manifest_sha256"] != p["model_manifest_sha256"]:
        raise ValueError("Expected the exact released B0 assets")
    rows = [json.loads(line) for line in Path(p["data_file"]).read_text().splitlines() if line.strip()]
    dataset = json.loads(Path(p["dataset_file"]).read_text())
    ids = [r["item_id"] for r in rows]
    if (not rows or dataset["split"] != "train" or dataset["items"] != len(rows)
            or dataset["examples_sha256"] != p["data_sha256"] or len(set(ids)) != len(ids)
            or dataset["item_ids_sha256"] != canonical_hash(ids)):
        raise ValueError("Dataset metadata/unique TRAIN examples mismatch")
    fmt = Format.from_dict(json.loads(Path(p["format_file"]).read_text())["format"])
    vocab = json.loads((Path(p["model_dir"]) / "config.json").read_text())["vocab_size"]
    for row in rows:
        for arm in ARMS:
            packed = pack_example(row, fmt, arm)
            if bool((packed["input_ids"] >= vocab).any()):
                raise ValueError("Packed input token outside released vocabulary")
    return rows, fmt, model


def worker(_index, args, p, rows, fmt, assets, protocol_hash):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa: F401
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
    from train.runtime_compat import configure_runtime
    from train.neuron_backend import NeuronCausalBackend
    from train.neuron_driver import save_checkpoint
    from train.masked_causal_ce import masked_causal_ce
    from train.warmstart_data import pack_example
    from weekend_thinking.pretrained import load_base
    from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy

    configure_runtime()
    if not hasattr(torch, 'xla'):
        torch._register_device_module('xla', torch_xla)
    require_runtime_policy(p["runtime_policy"], runtime_policy())
    device = torch_xla.device(); rank, world = xr.global_ordinal(), xr.world_size()
    if world != args.processes:
        raise ValueError("Actual runtime world differs from declared eight ranks")
    dist.init_process_group('xla', init_method='xla://')
    random.seed(p["seed"]+rank); np.random.seed(p["seed"]+rank); torch.manual_seed(p["seed"])
    model, _ = load_base(Path(p["model_dir"])); model.to(device); model.tie_weights()
    if (model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or any(t.dtype != torch.float32 for t in model.parameters())):
        raise RuntimeError("Released FP32 embedding/head tie was lost")
    backend = NeuronCausalBackend(model, sync=lambda:torch_xla.sync(wait=True))
    optimizer = ZeroRedundancyOptimizer(model.parameters(), AdamW, optimizer_dtype=torch.float32,
        grad_clipping=True, max_norm=p["clip"], save_master_weights=True, lr=p["lr"],
        betas=tuple(p["betas"]), eps=p["eps"], weight_decay=p["weight_decay"])
    found_inf = torch.zeros((), device=device)
    order = [order_index(i, len(rows), p["seed"]) for i in range(world*p["steps"])]
    order_hash = canonical_hash([rows[i]["item_id"] for i in order]); spec_hash = canonical_hash(p)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output/"run.json", dict(started_utc=utc_now(), arm=args.arm, seed=p["seed"],
            protocol_sha256=protocol_hash, spec_hash=spec_hash, order_hash=order_hash, model=assets,
            dataset_items=len(rows), planned_steps=p["steps"], world_size=world, compute="fp32",
            runtime_policy=runtime_policy(), limitations="Future-selected TRAIN supervision; no deployed-policy benefit claim."))
    xm.rendezvous('warmstart-output-ready'); torch_xla.sync(wait=True)
    xm.set_rng_state(p["seed"]+100003*rank, device)
    counters = dict(source_records=0, source_tokens=0, sampled_actions=0, note_close_labels=0, future_labels=0)
    deadline = require_utc_deadline(args.deadline_utc)
    def synchronized_stop():
        requested = torch.tensor(int(stop_due(args.stop_file, deadline)), device=device)
        return bool(xm.all_reduce(xm.REDUCE_MAX, requested).cpu())
    # Asset/model/device setup may outlast the deadline or an owner's STOP.
    # Every rank must take this decision before any optimizer update is begun.
    if synchronized_stop():
        if rank == 0:
            atomic_json(args.output/"completion.json", dict(completed_utc=utc_now(),
                arm=args.arm, actual_steps=0, planned_steps=p["steps"],
                stopped_after_setup=True, counters=counters, protocol_sha256=protocol_hash))
        dist.destroy_process_group()
        return
    def sample_parameters():
        return torch.cat([t.detach().reshape(-1)[:32] for t in model.parameters()]).cpu()
    for step in range(p["steps"]):
        started = time.monotonic(); optimizer.zero_grad(set_to_none=True); before = sample_parameters()
        row = rows[order[step*world+rank]]; packed = pack_example(row, fmt, args.arm)
        ce = masked_causal_ce(backend, **packed, note_weight=p["note_weight"], future_weight=p["future_weight"])
        losses = torch.stack([ce.loss.detach(), ce.note_close_loss.detach(), ce.future_loss.detach()]).cpu().tolist()
        if not all(math.isfinite(x) for x in losses):
            raise RuntimeError("Nonfinite masked CE before backward")
        ce.loss.backward(); torch_xla.sync(wait=True)
        finite = torch.stack([torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None]).all()
        if bool(xm.all_reduce(xm.REDUCE_MAX, (~finite).float()).cpu()):
            raise RuntimeError("Nonfinite gradient before sole ZeRO reduction owner")
        optimizer.step(found_inf=found_inf); torch_xla.sync(wait=True)
        norm = float(optimizer.grad_norm.cpu()); after = sample_parameters(); delta = after-before
        replicas = xm.all_gather(after.to(device)).reshape(world, -1).cpu()
        if (not math.isfinite(norm) or norm <= 0 or not bool(torch.isfinite(after).all()) or not bool(delta.ne(0).any())
                or not torch.equal(replicas, replicas[:1].expand_as(replicas))):
            raise RuntimeError("Nonfinite/unchanged sampled parameters or replica mismatch")
        local = losses+[int(packed["note_close_mask"].sum()), int(packed["future_mask"].sum())]
        totals = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(local, device=device)).cpu().tolist()
        counters["source_records"] += world; counters["source_tokens"] += world*528
        counters["note_close_labels"] += int(totals[3]); counters["future_labels"] += int(totals[4])
        passes, partial = divmod(counters["source_records"], len(rows))
        metric = dict(step=step+1, time_utc=utc_now(), arm=args.arm, elapsed_seconds=time.monotonic()-started,
            loss=totals[0]/world, note_close_loss=totals[1]/world, future_loss=totals[2]/world,
            gradient_norm=norm, counters=counters.copy(), complete_ordered_passes=passes,
            next_pass_records=partial, ordered_item_ids=[rows[i]["item_id"] for i in order[step*world:(step+1)*world]],
            sampled_update_l2=float(delta.norm()), sampled_changed_fraction=float(delta.ne(0).float().mean()),
            sampled_replicas_bitwise_equal=True)
        if rank == 0:
            with (args.output/"metrics.jsonl").open('a') as f:
                f.write(json.dumps(metric, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps(metric), flush=True)
        stop = synchronized_stop()
        if step+1 in (16, p["steps"]) or stop:
            if shutil.disk_usage(args.output).free < 35*1024**3:
                raise RuntimeError("Less than 35 GiB free before complete checkpoint")
            save_checkpoint(args.output, model, optimizer, step+1,
                dict(spec_hash=spec_hash, order_hash=order_hash, counters=counters.copy(), arm=args.arm,
                     seed=p["seed"], global_batch=world, accumulation=1, protocol_sha256=protocol_hash,
                     data_sha256=p["data_sha256"], dataset_sha256=p["dataset_sha256"],
                     parameter_dtype="fp32", compute_dtype="fp32", runtime_policy=runtime_policy(),
                     complete_ordered_passes=passes, next_pass_records=partial,
                     stopped_early=stop and step+1<p["steps"]),
                rank, world, xm, device)
        if stop:
            break
    dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--processes', type=int, choices=[8], default=8)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args(); p = json.loads(args.protocol.read_text()); validate_protocol(p, args.processes)
    if (args.output.exists() or args.stop_file.is_symlink()
            or (args.stop_file.exists() and not args.stop_file.is_file())
            or require_utc_deadline(args.deadline_utc) <= datetime.now(timezone.utc)):
        raise ValueError("Require fresh output, ordinary STOP path and future finite deadline")
    if (os.environ.get('NEURONCORE_NUM_DEVICES') != str(args.processes)
            or any(not os.environ.get(k) for k in ('NEURON_RT_VISIBLE_CORES', 'NEURON_RT_ROOT_COMM_ID', 'MASTER_ADDR', 'MASTER_PORT'))):
        raise ValueError("Set exact eight-device visibility and explicit dedicated host/Neuron endpoints")
    rows, fmt, assets = load_inputs(p)
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(args,p,rows,fmt,assets,sha256_file(args.protocol)), nprocs=None)


if __name__ == '__main__':
    main()
