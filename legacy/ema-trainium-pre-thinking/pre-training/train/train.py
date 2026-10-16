"""Pretrain Qwen3-0.6B from scratch on AWS Trainium.

    torchrun --nproc_per_node=2 train.py --corpus ../data/corpus

Neuron/XLA notes that drive the structure of this file:

  * Every tensor shape is fixed for the whole run. A new shape means a new
    graph and a fresh compilation, which costs minutes.
  * torch_xla.sync() is called once per *micro*-step, not per optimizer step.
    Syncing per optimizer step would fold all grad-accumulation micro-steps
    into one enormous graph. This way there are exactly two graphs: one
    forward/backward, one optimizer update.
  * .item() forces a graph execution plus a device->host transfer, so loss is
    accumulated in a device tensor and only read at logging boundaries.

Scaling: parallelism is data-parallel only, which is all a 0.6B model needs.
--nproc_per_node=32 on a trn1.32xlarge, or torchrun --nnodes>1, requires no
change to this file. Pass --zero1 to shard optimizer state as you scale out.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.amp            # submodule; `import torch_xla` alone does not bind it
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_backend  # noqa: F401  registers the 'xla' backend

# torch.utils.checkpoint's backward enters torch.random.fork_rng(), which
# resolves getattr(torch, "xla") *before* it honours enabled=False and raises
# if it is missing. With preserve_rng_state=False the module is never actually
# used, so registering it is purely to satisfy that lookup.
if not hasattr(torch, "xla"):
    torch._register_device_module("xla", torch_xla)

from data import Mixture, MixtureBatcher
from evaluate import evaluate, format_eval
from model_init import build_model, param_summary, untie_lm_head, MODEL_ID


def is_master():
    return xr.global_ordinal() == 0


def log(msg):
    if is_master():
        print(msg, flush=True)


def lr_at(step, args):
    """Linear warmup then cosine decay to --min-lr. The standard recipe.
    --lr-horizon (when set) decouples the cosine length from --max-steps so
    a run can stop early yet follow a longer schedule (segment replays)."""
    horizon = getattr(args, "lr_horizon", None) or args.max_steps
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(args.warmup_steps, 1)
    if step >= horizon:
        return args.min_lr
    prog = (step - args.warmup_steps) / max(horizon - args.warmup_steps, 1)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


def build_optimizer(model, args, lr_tensor):
    """No weight decay on biases or normalisation weights -- standard practice;
    decaying a gain term just pulls it toward zero for no benefit."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    groups = [
        dict(params=decay, weight_decay=args.weight_decay),
        dict(params=no_decay, weight_decay=0.0),
    ]
    # torch_xla's syncfree AdamW, not torch.optim.AdamW. Stock AdamW pushes the
    # lr through _get_value(), which calls .item() in eager mode -- that both
    # forces a host sync and turns the lr back into a baked graph constant, so
    # a changing schedule recompiles every step. The syncfree version consumes
    # a device tensor directly.
    import torch_xla.amp.syncfree as syncfree
    adamw_cls = torch.optim.AdamW if args.stock_adamw else syncfree.AdamW

    if args.zero1:
        from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
        # ZeRO-1 shards Adam's m and v across ranks (4.8GB -> 2.4GB on 2 cores)
        # and does its own reduce-scatter + clipping + all-gather inside step(),
        # so gradients must NOT be all-reduced or clipped externally.
        opt = ZeroRedundancyOptimizer(
            groups, adamw_cls, grad_clipping=True, max_norm=args.grad_clip,
            lr=lr_tensor, betas=(args.beta1, args.beta2), eps=args.eps)
    else:
        opt = adamw_cls(groups, lr=lr_tensor, betas=(args.beta1, args.beta2),
                                eps=args.eps)
    return opt, len(decay), len(no_decay)


# ---------------------------------------------------------------------------
# The objective. Kept deliberately small and separate: an RL term added later
# should slot in beside lm_loss without touching the loop below.
# ---------------------------------------------------------------------------
def lm_loss(model, x, y, autocast_ctx, fp32_logits=True, loss_chunks=1):
    """Cross-entropy over the vocabulary, optionally chunked over the sequence.

    Qwen3's vocab is 151936, so the logits for one seq_len=2048 sample are
    2048*151936*2 = 622MB in bf16 -- larger than every activation in the
    28-layer trunk combined, and the single allocation that overflows a
    16GB NeuronCore.

    With loss_chunks>1 the vocab projection is computed on one slice of the
    sequence at a time, each slice wrapped in a checkpoint so its logits are
    freed after use and recomputed during backward. Peak logits memory becomes
    622MB/loss_chunks. Mathematically identical: reduction="sum" over all
    slices divided by the token count is exactly the mean.
    """
    if loss_chunks <= 1:
        with autocast_ctx():
            logits = model(input_ids=x).logits
        if fp32_logits:
            logits = logits.float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))

    with autocast_ctx():
        hidden = model.model(input_ids=x)[0]

    def slice_loss(h, yy):
        with autocast_ctx():
            lg = model.lm_head(h)
        if fp32_logits:
            lg = lg.float()
        return F.cross_entropy(lg.view(-1, lg.shape[-1]), yy.reshape(-1),
                               reduction="sum")

    B, T = x.shape
    step = T // loss_chunks
    total = None
    for i in range(loss_chunks):
        sl = slice(i * step, (i + 1) * step if i < loss_chunks - 1 else T)
        part = torch.utils.checkpoint.checkpoint(
            slice_loss, hidden[:, sl], y[:, sl],
            use_reentrant=True, preserve_rng_state=False)
        total = part if total is None else total + part
    return total / (B * T)


def train_step(model, batcher, step, args, autocast_ctx, device):
    """One optimizer step's worth of forward/backward over grad-accum micro-steps.

    Returns a device-resident scalar; the caller decides when to read it.
    """
    total = torch.zeros((), device=device, dtype=torch.float32)
    for micro in range(args.grad_accum):
        x, y, _ = batcher.batch(step, micro)
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        loss = lm_loss(model, xb, yb, autocast_ctx, bool(args.fp32_logits),
                       args.loss_chunks)
        (loss / args.grad_accum).backward()
        total = total + loss.detach()
        torch_xla.sync()
    return total / args.grad_accum


_D2H_CHUNK = 32 << 20            # elements per transfer slice (~128MB fp32)


def _to_cpu(obj):
    """Recursively copy a state dict to host, big tensors in slices.

    A whole-tensor D2H of the 622MB embedding needs a device-side linear
    staging buffer that a RESUMED process's fragmented pool cannot provide
    (fresh processes fit it -- the original 0-350 run checkpointed fine).
    ~128MB slices materialize, transfer and free one at a time, reusing the
    same small hole."""
    if torch.is_tensor(obj):
        t = obj.detach()
        if t.numel() > _D2H_CHUNK:
            flat = t.reshape(-1)
            parts = []
            for i in range(0, flat.numel(), _D2H_CHUNK):
                parts.append(flat[i:i + _D2H_CHUNK].cpu())
                torch_xla.sync()
            return torch.cat(parts).reshape(t.shape)
        return t.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_checkpoint(model, opt, step, args, tokens_seen, rank, world):
    """Model weights are replicated across ranks, so one copy is written by the
    master. ZeRO-1 optimizer state is NOT replicated -- each rank owns a
    distinct shard of Adam's m and v -- so every rank writes its own shard, or
    resuming would silently restore only 1/world_size of the optimizer state.
    """
    if not args.ckpt_dir:
        return
    os.makedirs(args.ckpt_dir, exist_ok=True)
    base = os.path.join(args.ckpt_dir, f"step_{step:07d}")
    # Per-tensor device->host copies instead of xm.save's bulk sync: the bulk
    # path stages a device-side copy of the largest tensor (the 622MB fp32
    # embedding), which fits a fresh process's pool but not a RESUMED
    # process's (same total, more fragmentation) -- OOMed at the first
    # post-resume checkpoint on 2026-08-28. One tensor at a time reuses the
    # same staging hole. Written to a temp file and renamed so a crash
    # mid-save can never leave a truncated step_*.pt that a later
    # --resume latest would load.
    if is_master():
        log(f"  ckpt {step}: copying model to host ...")
        cpu_sd = _to_cpu(model.state_dict())
        tmp = base + ".pt.tmp"
        torch.save(dict(step=step, tokens_seen=tokens_seen, args=vars(args),
                        world_size=world, zero1=bool(args.zero1),
                        model=cpu_sd), tmp)
        os.replace(tmp, base + ".pt")
        del cpu_sd
        log(f"  ckpt {step}: model written")
    xm.rendezvous(f"ckpt_{step}_model")
    if getattr(args, "save_optim", 1):
        if args.zero1:
            # Serialising every rank's shard at once needs world_size * ~3GB
            # of HOST RAM simultaneously, which the oom-killer will not
            # tolerate on a 30GB box. Stagger so only one rank holds a CPU
            # copy at a time.
            for r in range(world):
                if r == rank:
                    log(f"  ckpt {step}: copying optim shard {rank} ...")
                    tmp = f"{base}.optim_rank{rank}.pt.tmp"
                    torch.save(_to_cpu(opt.state_dict()), tmp)
                    os.replace(tmp, f"{base}.optim_rank{rank}.pt")
                    log(f"  ckpt {step}: optim shard {rank} written")
                xm.rendezvous(f"ckpt_{step}_shard_{r}")
        else:
            if is_master():
                # 2026-10-04 22:07 tick (G136): run 91's step-190 optimizer save died on
                # `AllocBuffer: NRT_RESOURCE ... size=622329856` -- the 622MB fp32 embedding
                # staging buffer this D2H needs -- four minutes after that same step recorded
                # the first POSITIVE rung this programme has produced. The model .pt was
                # already on disk, so nothing the experiment needs was lost; but the raised
                # exception killed all 64 ranks, and the three keepalive retries then
                # crash-looped on the MISSING .optim.pt (`--resume latest` picks the model
                # file and load_checkpoint demands the optimizer beside it) until D1 stopped
                # the run. A failed optimizer save must cost Adam's momentum, never 190 steps
                # of training. So: catch it, say so loudly, and still enter the rendezvous so
                # the other 63 ranks are not left in a broken collective.
                tmp = base + ".optim.pt.tmp"
                try:
                    torch.save(_to_cpu(opt.state_dict()), tmp)
                    os.replace(tmp, base + ".optim.pt")
                except Exception as e:
                    log(f"  ckpt {step}: OPTIMIZER SAVE FAILED "
                        f"({type(e).__name__}: {e}) -- continuing with a MODEL-ONLY "
                        f"checkpoint; a resume from it needs --resume-model-only 1")
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            xm.rendezvous(f"ckpt_{step}_optim")

    if is_master():
        log(f"  checkpoint -> {base}.pt"
            + (" (+ optimizer state)" if getattr(args, "save_optim", 1)
               else " (model only, --save-optim 0)"))
        steps = sorted({f.split(".")[0] for f in os.listdir(args.ckpt_dir)
                        if f.startswith("step_")})
        for old in steps[:-args.keep_checkpoints]:
            for f in os.listdir(args.ckpt_dir):
                if f.startswith(old + "."):
                    os.remove(os.path.join(args.ckpt_dir, f))
            log(f"  pruned old checkpoint {old}")


def load_checkpoint(model, opt, args, device, rank, world):
    path = args.resume
    if path == "latest":
        cands = sorted(f for f in os.listdir(args.ckpt_dir)
                       if f.startswith("step_") and f.endswith(".pt")
                       and ".optim" not in f)
        if not cands:
            log("  --resume latest: no checkpoint found, starting fresh")
            return 0, 0
        path = os.path.join(args.ckpt_dir, cands[-1])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])

    if getattr(args, "resume_model_only", 0):
        # Cold-optimizer resume: skip the optimizer-state restore (and the
        # grad pre-claim it needs), so the process's device-allocation
        # history matches a FRESH run -- Adam states appear lazily at the
        # first opt.step exactly as at step 0. A fresh layout is the only
        # one proven to fit the per-step transients AND checkpoint saves
        # (2026-08-28: warm resumes OOMed at the first post-resume save
        # through three different save implementations). Costs one-time
        # loss of Adam momentum (~50 steps to rebuild).
        step, seen = ck["step"], ck.get("tokens_seen", 0)
        del ck
        gc.collect()
        log(f"  resumed MODEL ONLY from {path} at step {step} "
            f"(cold optimizer; tokens_seen={seen/1e9:.4f}B)")
        return step, seen

    saved_world = ck.get("world_size", 1)
    if ck.get("zero1") and saved_world != world:
        raise SystemExit(
            f"checkpoint was written with ZeRO-1 across {saved_world} ranks but "
            f"this run has {world}. The optimizer shards do not map across a "
            f"different world size; re-run with --nproc_per_node={saved_world}.")

    stem = path[:-3]
    opt_path = (f"{stem}.optim_rank{rank}.pt" if ck.get("zero1")
                else f"{stem}.optim.pt")
    # Claim the gradient buffers BEFORE the optimizer state is restored.
    # A fresh run allocates grads on the first backward, while Adam's states do
    # not exist yet; a resume would otherwise restore optimizer state first and
    # leave no contiguous room for the 622MB fp32 gradient of the tied
    # 151936x1024 embedding. Same total footprint, different order.
    for prm in model.parameters():
        if prm.requires_grad and prm.grad is None:
            prm.grad = torch.zeros_like(prm)
    torch_xla.sync()

    opt_sd = torch.load(opt_path, map_location="cpu", weights_only=False)
    opt.load_state_dict(opt_sd)

    step, seen = ck["step"], ck.get("tokens_seen", 0)
    # Drop the host copies and flush pending device work before training
    # starts. ZeRO's load_state_dict deepcopies the state dict, and the stale
    # buffers stay reachable long enough to eat the headroom the logits
    # allocation needs on the first step after a resume.
    del ck, opt_sd
    gc.collect()
    torch_xla.sync()

    # Now FREE the pre-claimed grad buffers: they were only needed to hold
    # address space while the optimizer state was placed. If they survive
    # into the first backward, that backward traces the grad-ACCUMULATE
    # variant -- a second large NEFF alongside the steady-state grad-CREATE
    # one (zero_grad(set_to_none=True) frees grads every step, so all later
    # steps use CREATE). Loaded NEFFs never unload, and the two variants'
    # combined residency tipped the core over 16GB at resumed step 351
    # (2026-08-28: 15.49/16GB then OOM on the 622MB embedding grad).
    for prm in model.parameters():
        prm.grad = None
    torch_xla.sync()

    log(f"  resumed from {path} at step {step} (tokens_seen={seen/1e9:.4f}B)")
    return step, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus"))
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=5722)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-sequences", type=int, default=32,
                   help="sequences per domain per eval")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--ckpt-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--init-from", default=None,
                   help="load model WEIGHTS from this .pt and start at step 0 "
                        "with a fresh optimizer (think_train semantics; used "
                        "by the eval/ token-matched vanilla control)")
    p.add_argument("--lr-horizon", type=int, default=None,
                   help="cosine horizon for lr_at when it must differ from "
                        "--max-steps (segment replays that stop early)")
    p.add_argument("--resume", default=None, help="path to .pt, or 'latest'")
    p.add_argument("--resume-model-only", type=int, default=0,
                   help="1 = restore weights/step only, fresh Adam. WARNING: "
                        "tested 2026-08-28 and the cold ZeRO-1 start blew the "
                        "model up (loss 5.46 -> 12.1 in one step); use "
                        "--param-dtype bf16 for the warm-resume OOM instead")
    p.add_argument("--param-dtype", choices=["fp32", "bf16"], default="fp32",
                   help="bf16 halves param+grad residency (~4.5GB/core less), "
                        "the proven think_train config; fp32 master shards "
                        "live in the ZeRO-1 optimizer either way")
    p.add_argument("--zero1", action="store_true",
                   help="shard optimizer state across ranks (for scale-out)")
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument("--loss-chunks", type=int, default=1,
                   help="split the vocab projection into N sequence slices to "
                        "cap peak logits memory (needed for seq_len 2048)")
    p.add_argument("--save-optim", type=int, default=1,
                   help="0 = checkpoint model weights only (a crash-resume "
                        "then restarts Adam cold). Escape hatch for resumed "
                        "processes whose fragmented device pool cannot stage "
                        "the optimizer-state transfers.")
    p.add_argument("--stock-adamw", action="store_true",
                   help="use torch.optim.AdamW (recompiles every step on XLA)")
    p.add_argument("--grad-checkpoint", type=int, default=1,
                   help="recompute activations in backward (1=on). Required to "
                        "fit seq_len=2048 in a NeuronCore's 16GB HBM.")
    p.add_argument("--fp32-logits", type=int, default=1,
                   help="upcast logits before cross-entropy (1=on)")
    p.add_argument("--metrics-file", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch_xla.device()

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()

    tokens_per_step = args.micro_bs * args.grad_accum * args.seq_len * world

    log("=" * 72)
    log(f"world_size={world}  micro_bs={args.micro_bs}  grad_accum={args.grad_accum}"
        f"  seq_len={args.seq_len}")
    log(f"tokens/optimizer-step = {tokens_per_step:,}")
    log(f"max_steps={args.max_steps} -> {tokens_per_step*args.max_steps/1e9:.2f}B tokens")
    log("=" * 72)

    model, cfg, n_res = build_model(
        MODEL_ID, seq_len=args.seq_len, seed=args.seed,
        dtype=torch.bfloat16 if args.param_dtype == "bf16" else torch.float32)
    s = param_summary(model, cfg)
    log(f"params: {s['total']/1e6:.1f}M total, {s['non_embedding']/1e6:.1f}M non-embedding"
        f"  (depth-scaled projections: {n_res})")
    if args.grad_checkpoint:
        # A NeuronCore has 16GB of HBM (the 32GB neuron-ls reports is per
        # *device*, shared by its two cores). The full fwd+bwd graph for
        # seq_len=2048 peaks at ~25.7GB without this, and the compiler refuses
        # to build it. Recomputing activations brings it under the limit.
        # use_reentrant=True + preserve_rng_state=False is the combination that
        # works on XLA. The non-reentrant path calls _get_device_module("xla")
        # unconditionally, which resolves getattr(torch, "xla") and raises;
        # the reentrant path only touches it when preserving RNG state.
        # Qwen3 sets attention_dropout=0.0 and has no other dropout, so
        # recomputation is deterministic and the RNG state is not needed.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": True, "preserve_rng_state": False})
        log("gradient checkpointing: ON")
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        untie_lm_head(model)         # ck holds two diverged matrices
        model.load_state_dict(ck["model"], strict=True)
        log(f"init-from {args.init_from} (step {ck.get('step')}, "
            f"{ck.get('tokens_seen', 0)/1e9:.3f}B tokens; fresh optimizer, "
            f"step counter starts at 0)")
        del ck
    model = model.to(device)
    model.train()

    # The learning rate MUST be a device tensor, not a Python float. A float
    # written into param_groups is baked into the optimizer's XLA graph as a
    # constant, so a warmup/cosine schedule produces a *new graph every step*:
    # measured 257s/step recompiling versus 0.68s/step with a stable graph.
    # As a tensor it is a graph input, updated by a host->device copy.
    lr_tensor = torch.tensor(args.lr, dtype=torch.float32, device=device)
    lr_host = torch.empty((), dtype=torch.float32)
    opt, nd, nnd = build_optimizer(model, args, lr_tensor)
    log(f"optimizer: AdamW  decay-groups={nd} tensors, no-decay={nnd} tensors"
        f"  zero1={args.zero1}")

    autocast_ctx = (nullcontext if args.no_autocast
                    else (lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)))

    mix = Mixture(args.corpus, "train")
    if is_master():
        for d, v in mix.summary().items():
            log(f"  data {d:<14} {v['tokens']/1e6:9.1f}M tokens  w={v['weight']:.4f}")
    batcher = MixtureBatcher(mix, args.micro_bs, args.seq_len,
                             seed=args.seed, rank=rank, world_size=world)

    start_step, tokens_seen = 0, 0
    if args.resume:
        start_step, tokens_seen = load_checkpoint(model, opt, args, device, rank, world)

    params = [p_ for g in opt.param_groups for p_ in g["params"]]
    metrics = []
    t_last = time.time()
    log("\nstarting (first step includes compilation) ...")

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        lr_host.fill_(lr)
        lr_tensor.copy_(lr_host)   # data transfer; graph shape/consts unchanged

        t0 = time.time()
        loss_dev = train_step(model, batcher, step, args, autocast_ctx, device)

        if args.zero1:
            # reduce-scatter, clip, step, all-gather -- all inside step()
            opt.step()
            gnorm = None
        else:
            if world > 1:
                xm.reduce_gradients(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch_xla.sync()

        tokens_seen += tokens_per_step
        dt = time.time() - t0

        if step % args.log_every == 0 or step == args.max_steps - 1:
            l = loss_dev.item()          # single sync point per log interval
            g = float("nan") if gnorm is None else float(gnorm)
            tps = tokens_per_step / dt
            log(f"step {step:6d}/{args.max_steps}  loss {l:7.4f}  "
                f"lr {lr:.2e}  gnorm {g:6.3f}  {dt:6.2f}s  "
                f"{tps:9,.0f} tok/s  seen {tokens_seen/1e9:.3f}B")
            metrics.append(dict(step=step, loss=l, lr=lr, grad_norm=g,
                                dt=dt, tokens_per_s=tps, tokens_seen=tokens_seen))

        if args.eval_every and (step + 1) % args.eval_every == 0:
            t_ev = time.time()
            res = evaluate(model, args.corpus, device, args.seq_len, args.micro_bs,
                           args.eval_sequences, autocast_ctx, torch_xla.sync,
                           args.loss_chunks)
            log(f"  eval @ step {step}: {format_eval(res)}  ({time.time()-t_ev:.0f}s)")
            metrics.append(dict(step=step, eval=res))

        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(model, opt, step + 1, args, tokens_seen, rank, world)

        if args.metrics_file and is_master() and metrics:
            with open(args.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

    # The in-loop save already covers max_steps when it lands on the interval;
    # repeating it here would serialise a second full copy while the first is
    # still resident and can trip the host oom-killer.
    if not (args.ckpt_every and args.max_steps % args.ckpt_every == 0):
        save_checkpoint(model, opt, args.max_steps, args, tokens_seen, rank, world)
    log(f"\ndone. {tokens_seen/1e9:.3f}B tokens in "
        f"{time.time()-t_last:.0f}s wall")


if __name__ == "__main__":
    main()
