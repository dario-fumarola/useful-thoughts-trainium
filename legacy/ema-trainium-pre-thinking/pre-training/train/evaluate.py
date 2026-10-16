"""Per-domain held-out evaluation.

Reports loss for each domain separately rather than one aggregate number: with
a 74/16/10 mixture an aggregate is dominated by fineweb-edu and would hide a
domain going wrong.
"""

import math

import torch
import torch.nn.functional as F

from data import Mixture, eval_batches


@torch.no_grad()
def _chunked_ce(model, xb, yb, autocast_ctx, loss_chunks):
    """Same slicing as training's lm_loss, minus the checkpointing -- under
    no_grad nothing is retained for backward, so the slices only need to keep
    peak logits memory below what a NeuronCore can allocate."""
    if loss_chunks <= 1:
        with autocast_ctx():
            logits = model(input_ids=xb).logits
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               yb.reshape(-1))
    with autocast_ctx():
        hidden = model.model(input_ids=xb)[0]
    B, T = xb.shape
    step = T // loss_chunks
    total = None
    for i in range(loss_chunks):
        sl = slice(i * step, (i + 1) * step if i < loss_chunks - 1 else T)
        with autocast_ctx():
            lg = model.lm_head(hidden[:, sl])
        part = F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                               yb[:, sl].reshape(-1), reduction="sum")
        total = part if total is None else total + part
    return total / (B * T)


@torch.no_grad()
def evaluate(model, corpus_dir, device, seq_len, micro_bs, max_sequences,
             autocast_ctx, sync, loss_chunks=1):
    """Returns {domain: {loss, ppl, sequences}} plus an aggregate."""
    model.eval()
    mix = Mixture(corpus_dir, "val")
    out = {}
    for dom in mix.domains:
        tot = torch.zeros((), device=device, dtype=torch.float32)
        n = 0
        for x, y in eval_batches(mix, dom, micro_bs, seq_len, max_sequences):
            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y).to(device)
            tot = tot + _chunked_ce(model, xb, yb, autocast_ctx, loss_chunks)
            n += 1
            sync()
        if n:
            v = (tot / n).item()
            out[dom] = dict(loss=v, ppl=math.exp(min(v, 20)), batches=n,
                            sequences=n * micro_bs)
    model.train()
    if out:
        # Aggregate weighted by the training mixture, so it tracks the loss the
        # training objective actually optimises.
        w = {d: mix.manifest["domains"][d]["weight"] for d in out}
        s = sum(w.values())
        out["_mixture_weighted"] = dict(
            loss=sum(out[d]["loss"] * w[d] for d in out) / s)
    return out


def format_eval(res):
    lines = []
    for d, v in res.items():
        if d.startswith("_"):
            continue
        lines.append(f"{d}={v['loss']:.4f}(ppl {v['ppl']:.1f})")
    if "_mixture_weighted" in res:
        lines.append(f"mix={res['_mixture_weighted']['loss']:.4f}")
    return "  ".join(lines)
