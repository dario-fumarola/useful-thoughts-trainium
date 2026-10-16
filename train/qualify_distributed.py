"""Bounded real-XLA ZeRO mean-update and fresh-process resume qualification.

This tiny FP32 model qualifies optimizer semantics, not Qwen memory or speed.
Run both phases with NEURONCORE_NUM_DEVICES=2 and a dedicated runtime port.
The second phase must be a new invocation; COMPLETE is published after all shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORLD_SIZE = 2
ACCUMULATION = 2
MICRO_BATCH = 3
MAX_NORM = 0.07
OPTIMIZER = dict(lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
ATOL = 4e-6
RTOL = 4e-5
RESUME_ATOL = 1e-7
RESUME_RTOL = 1e-6


def build_model() -> nn.Module:
    # Odd leading dimensions exercise padding in ZeRO's shard/all-gather path.
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))


def to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def flat_weights(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])


def compare_tree(actual: Any, expected: Any, *, exact: bool = False,
                 atol: float = ATOL, rtol: float = RTOL, path: str = "state") -> float:
    """Fail on missing optimizer/RNG fields as well as differing tensor values."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError("Expected a tensor")
        if exact:
            if not torch.equal(actual, expected):
                raise AssertionError(f"Resume tensor is not bitwise identical: {path}")
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: f"{path}: {m}")
        return float((actual - expected).abs().max()) if actual.numel() else 0.0
    if isinstance(expected, np.ndarray):
        if not np.array_equal(actual, expected):
            raise AssertionError("NumPy RNG state changed after resume")
        return 0.0
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("State keys differ")
        return max((compare_tree(actual[k], v, exact=exact, atol=atol, rtol=rtol, path=f"{path}.{k}")
                    for k, v in expected.items()), default=0.0)
    if isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError("State lengths differ")
        return max((compare_tree(a, e, exact=exact, atol=atol, rtol=rtol, path=f"{path}[{i}]")
                    for i, (a, e) in enumerate(zip(actual, expected))), default=0.0)
    if actual != expected:
        raise AssertionError(f"State differs: {actual!r} != {expected!r}")
    return 0.0


def tensor_error_metrics(actual: Any, expected: Any) -> dict[str, float | int]:
    """Summarize FP32 differences relative to the separately declared tolerance."""
    metrics = dict(max_abs=0.0, max_tolerance_fraction=0.0, changed_values=0)

    def visit(a, e):
        if isinstance(e, torch.Tensor) and e.numel():
            delta = (a.to(torch.float64) - e.to(torch.float64)).abs()
            allowance = RESUME_ATOL + RESUME_RTOL * e.to(torch.float64).abs()
            metrics["max_abs"] = max(metrics["max_abs"], float(delta.max()))
            metrics["max_tolerance_fraction"] = max(metrics["max_tolerance_fraction"], float((delta / allowance).max()))
            metrics["changed_values"] += int((a != e).sum())
        elif isinstance(e, dict):
            for key, item in e.items():
                visit(a[key], item)
        elif isinstance(e, (tuple, list)):
            for item_a, item_e in zip(a, e):
                visit(item_a, item_e)

    visit(actual, expected)
    return metrics


def atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(to_cpu(value), temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe(steps: int, expected_cores: list[str]) -> dict[str, Any]:
    return dict(world_size=WORLD_SIZE, steps=steps, checkpoint_step=steps // 2,
                accumulation=ACCUMULATION, micro_batch=MICRO_BATCH,
                max_norm=MAX_NORM, optimizer=OPTIMIZER, expected_cores=expected_cores,
                dtype="float32", model="Linear(5,7)-Tanh-Linear(7,3)",
                gradient_reducer_and_clipper="ZeroRedundancyOptimizer.step")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("uninterrupted", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--expected-cores", default="4,5")
    args = parser.parse_args()
    if args.steps < 4 or args.steps > 8 or args.steps % 2:
        parser.error("Qualification requires an even hard step limit between 4 and 8")
    expected_cores = args.expected_cores.split(",")
    if len(expected_cores) != WORLD_SIZE or len(set(expected_cores)) != WORLD_SIZE:
        parser.error("Exactly two distinct expected core IDs are required")
    if "TORCHELASTIC_RUN_ID" in os.environ:
        parser.error("Use xmp launch: this installed torchrun initializer overrides visible cores")
    if os.environ.get("NEURONCORE_NUM_DEVICES") != str(WORLD_SIZE):
        parser.error("Set NEURONCORE_NUM_DEVICES=2")
    if not os.environ.get("NEURON_RT_ROOT_COMM_ID"):
        parser.error("Set a dedicated NEURON_RT_ROOT_COMM_ID=127.0.0.1:<port>")

    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(args,), nprocs=None)


def worker(_index: int, args: argparse.Namespace) -> None:
    expected_cores = args.expected_cores.split(",")
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa: F401
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

    from train.runtime_compat import configure_runtime, restore_syncfree_device_state
    compiler_receipt = configure_runtime()

    device = torch_xla.device()
    rank = xr.global_ordinal()
    if xr.global_device_count() != WORLD_SIZE:
        raise RuntimeError(f"Expected 2 global devices, found {xr.global_device_count()}")
    visible = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if visible != expected_cores[rank]:
        raise RuntimeError(f"Rank {rank} expected core {expected_cores[rank]}, got {visible}")
    dist.init_process_group("xla", init_method="xla://")

    output = args.output.resolve()
    checkpoint = output / "checkpoint"
    phase_file = output / f"{args.phase}.rank{rank}.json"
    if phase_file.exists():
        raise FileExistsError(f"Refusing to overwrite qualification evidence: {phase_file}")
    if args.phase == "uninterrupted":
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            checkpoint.mkdir()
        xm.rendezvous("output-created")
    elif not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError("Checkpoint has no COMPLETE marker")

    torch.manual_seed(1729)
    cpu_model = build_model()
    model = build_model()
    model.load_state_dict(cpu_model.state_dict())
    model.to(device=device, dtype=torch.float32)
    cpu_optimizer = torch.optim.AdamW(cpu_model.parameters(), **OPTIMIZER, foreach=False)
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(), optimizer_class=AdamW, optimizer_dtype=torch.float32,
        grad_clipping=True, max_norm=MAX_NORM, save_master_weights=True,
        higher_cc_precision=True, **OPTIMIZER, foreach=False,
    )
    xm.mark_step()
    xm.wait_device_ops()
    torch.manual_seed(9000 + rank)
    np.random.seed(11000 + rank)
    random.seed(12000 + rank)
    xm.set_rng_state(13000 + rank, device=device)

    def rng_state() -> dict[str, Any]:
        return dict(torch=torch.get_rng_state().clone(), numpy=np.random.get_state(),
                    python=random.getstate(), xla=xm.get_rng_state(device=device))

    def state(step: int) -> dict[str, Any]:
        xm.mark_step()
        xm.wait_device_ops()
        return to_cpu(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           cpu_model=cpu_model.state_dict(), cpu_optimizer=cpu_optimizer.state_dict(),
                           rng=rng_state(), rank=rank, recipe=recipe(args.steps, expected_cores)))

    start_step = 0
    if args.phase == "resume":
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if manifest["recipe"] != json.loads(json.dumps(recipe(args.steps, expected_cores))):
            raise RuntimeError("Checkpoint topology or recipe differs")
        for name, digest in manifest["shards"].items():
            if sha256_file(checkpoint / name) != digest:
                raise RuntimeError(f"Checkpoint hash mismatch: {name}")
        saved = torch.load(checkpoint / f"rank{rank}.pt", weights_only=False)
        if saved["rank"] != rank or "sharded_master_weights" not in saved["optimizer"]:
            raise RuntimeError("Missing rank-local FP32 master weights")
        model.load_state_dict({k: v.to(device) for k, v in saved["model"].items()})
        optimizer.load_state_dict(saved["optimizer"])
        restore_syncfree_device_state(optimizer)
        optimizer.zero_grad(set_to_none=True)
        cpu_model.load_state_dict(saved["cpu_model"])
        cpu_optimizer.load_state_dict(saved["cpu_optimizer"])
        torch.set_rng_state(saved["rng"]["torch"])
        np.random.set_state(saved["rng"]["numpy"])
        random.setstate(saved["rng"]["python"])
        xm.set_rng_state(saved["rng"]["xla"], device=device)
        start_step = saved["step"]
        # Verify the optimizer loader restored every shard, moment and master.
        compare_tree(to_cpu(optimizer.state_dict()), saved["optimizer"], exact=True)
        compare_tree(rng_state(), saved["rng"], exact=True)
        print(json.dumps(dict(event="resume_loaded", rank=rank, step=start_step)), flush=True)

    initial = flat_weights(model)
    trace = []
    gradient_metrics = []
    for step in range(start_step, args.steps):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        cpu_optimizer.zero_grad(set_to_none=True)
        # Every RNG contributes to a subsequent update, so resume tests all four.
        host_x = torch.randn(ACCUMULATION * MICRO_BATCH, 5)
        host_x += float(np.random.uniform(-0.2, 0.2)) + random.uniform(-0.2, 0.2)
        host_x += rank * 0.3
        host_y = torch.randn(ACCUMULATION * MICRO_BATCH, 3) + rank * 0.2
        x = host_x.to(device) + torch.rand(host_x.shape, device=device) * 0.05
        y = host_y.to(device)
        # Detached data gathering supplies the independent concatenated CPU oracle.
        all_x = xm.all_gather(x.detach(), dim=0).cpu()
        all_y = xm.all_gather(y.detach(), dim=0).cpu()
        data_hash = tensor_hash(x.detach().cpu())
        before = flat_weights(model)
        for micro in range(ACCUMULATION):
            lo, hi = micro * MICRO_BATCH, (micro + 1) * MICRO_BATCH
            (F.mse_loss(model(x[lo:hi]), y[lo:hi]) / ACCUMULATION).backward()
            xm.mark_step()
        local_gradients = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu()
        atomic_save(local_gradients, output / f"{args.phase}.gradients.step{step + 1}.rank{rank}.pt")
        if args.phase == "resume":
            expected_gradients = torch.load(output / f"uninterrupted.gradients.step{step + 1}.rank{rank}.pt", weights_only=True)
            compare_tree(local_gradients, expected_gradients, atol=RESUME_ATOL, rtol=RESUME_RTOL)
            gradient_metrics.append(tensor_error_metrics(local_gradients, expected_gradients))
        cpu_loss = F.mse_loss(cpu_model(all_x), all_y)
        cpu_loss.backward()
        cpu_norm = torch.nn.utils.clip_grad_norm_(cpu_model.parameters(), MAX_NORM)
        cpu_optimizer.step()
        # No xm.optimizer_step or manual gradient reduction: ZeRO owns both.
        optimizer.step(found_inf=torch.zeros((), device=device))
        xm.mark_step()
        xm.wait_device_ops()
        current = flat_weights(model)
        reference = flat_weights(cpu_model)
        norm = float(optimizer.grad_norm.cpu())
        torch.testing.assert_close(torch.tensor(norm), cpu_norm, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(current, reference, atol=ATOL, rtol=RTOL)
        changed = float((current != before).float().mean())
        if not torch.isfinite(current).all() or changed == 0:
            raise AssertionError("Optimizer did not produce finite changed parameters")
        # Both ranks must hold the same full updated weights.
        peers = xm.all_gather(torch.cat([p.detach().flatten() for p in model.parameters()]), dim=0).cpu()
        if not torch.equal(peers[:current.numel()], peers[current.numel():]):
            raise AssertionError("Replicas disagree after ZeRO all-gather")
        entry = dict(step=step + 1, rank=rank, data_sha256=data_hash,
                     local_gradient_sha256=tensor_hash(local_gradients),
                     max_abs_cpu_error=float((current - reference).abs().max()),
                     grad_norm=norm, cpu_grad_norm=float(cpu_norm), clipped=norm > MAX_NORM,
                     update_l2=float((current - before).norm()), changed_fraction=changed,
                     weights_sha256=tensor_hash(current), seconds=time.monotonic() - begin)
        trace.append(entry)
        print(json.dumps(dict(event="step", **entry)), flush=True)
        if args.phase == "uninterrupted" and step + 1 == args.steps // 2:
            atomic_save(state(step + 1), checkpoint / f"rank{rank}.pt")
            xm.rendezvous("checkpoint-shards-written")
            if rank == 0:
                shards = {f"rank{r}.pt": sha256_file(checkpoint / f"rank{r}.pt") for r in range(WORLD_SIZE)}
                (checkpoint / "manifest.json").write_text(json.dumps(dict(recipe=recipe(args.steps, expected_cores), shards=shards), indent=2) + "\n")
                (checkpoint / "COMPLETE").write_text("complete\n")
            xm.rendezvous("checkpoint-complete")

    final_state = state(args.steps)
    resume_error = None
    resume_bitwise = None
    component_errors = None
    if args.phase == "uninterrupted":
        atomic_save(final_state, output / f"final.rank{rank}.pt")
    else:
        expected = torch.load(output / f"final.rank{rank}.pt", weights_only=False)
        atomic_save(final_state, output / f"resumed.final.rank{rank}.pt")
        compare_tree(final_state["rng"], expected["rng"], exact=True)
        resume_error = compare_tree(final_state, expected, atol=RESUME_ATOL, rtol=RESUME_RTOL)
        resume_bitwise = resume_error == 0.0
        component_errors = {key: tensor_error_metrics(final_state[key], expected[key])
                            for key in ("model", "optimizer", "cpu_model", "cpu_optimizer")}
        previous = json.loads((output / f"uninterrupted.rank{rank}.json").read_text())["trace"]
        for actual, wanted in zip(trace, previous[start_step:]):
            for field in ("data_sha256", "grad_norm"):
                if actual[field] != wanted[field]:
                    raise AssertionError(f"Fresh-process continuation differs in {field}")
    result = dict(status="PASS", scope="tiny_fp32_distributed_optimizer_only", phase=args.phase,
                  rank=rank, pid=os.getpid(), visible_core=visible, device=str(device),
                  torch_version=torch.__version__, xla_version=torch_xla.__version__,
                  master_weights_saved=True, rng_seed_rank=13000 + rank,
                  compiler_compatibility=compiler_receipt,
                  resume_max_abs_state_error=resume_error, resume_bitwise=resume_bitwise,
                  resume_component_errors=component_errors, resume_gradient_errors=gradient_metrics,
                  total_update_l2=float((flat_weights(model) - initial).norm()), trace=trace)
    phase_file.write_text(json.dumps(result, indent=2) + "\n")
    xm.rendezvous(f"{args.phase}-finished")
    if rank == 0:
        other = json.loads((output / f"{args.phase}.rank1.json").read_text())
        if [e["data_sha256"] for e in trace] == [e["data_sha256"] for e in other["trace"]]:
            raise AssertionError("Ranks used identical examples and RNG")
        summary = dict(status="PASS", phase=args.phase, rank_distinct_data=True,
                       fresh_process_resume=args.phase == "resume",
                       resume_rng_exact=args.phase == "resume",
                       resume_max_abs_state_error=resume_error,
                       resume_bitwise=resume_bitwise,
                       resume_tolerance=dict(atol=RESUME_ATOL, rtol=RESUME_RTOL),
                       resume_component_errors=component_errors, resume_gradient_errors=gradient_metrics,
                       all_steps_clipped=all(e["clipped"] for e in trace),
                       max_abs_cpu_error=max(e["max_abs_cpu_error"] for e in trace),
                       recipe=recipe(args.steps, expected_cores),
                       limitation="Tiny FP32 MLP; does not qualify Qwen memory, BF16 compute, sampler or throughput")
        (output / f"{args.phase}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
