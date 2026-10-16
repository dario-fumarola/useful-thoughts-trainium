"""Sample completions from the vanilla checkpoint on held-out val prefixes.

Static-shape batched generation (no KV cache, one compiled graph reused per
token, same recipe as qwen0.6b-inference.py). Writes a side-by-side report of
model continuation vs the true continuation to runs/VANILLA_GEN_REPORT.md.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.amp

from data import Mixture, eval_batches
from model_init import build_model, untie_lm_head, MODEL_ID


def main():
    p = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--ckpt", default=os.path.join(
        base, "..", "checkpoints", "vanilla_p0", "step_0000350.pt"))
    p.add_argument("--corpus", default=os.path.join(base, "..", "data", "corpus"))
    p.add_argument("--n-prompts", type=int, default=24)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prefix", type=int, default=64)
    p.add_argument("--gen", type=int, default=48)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out-dir", default=os.path.join(base, "..", "runs"))
    args = p.parse_args()

    device = torch_xla.device()
    MAX = args.prefix + args.gen

    print(f"loading {args.ckpt} ...", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg, _ = build_model(MODEL_ID, seq_len=MAX, seed=0)
    untie_lm_head(model)          # ck holds two diverged matrices
    model.load_state_dict(ck["model"], strict=True)
    step_trained, toks = ck.get("step"), ck.get("tokens_seen", 0)
    del ck
    model = model.to(torch.bfloat16).to(device)
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    mixture = Mixture(args.corpus, "val")
    streams = {d: eval_batches(mixture, d, 1, 2 * MAX, max_sequences=128)
               for d in mixture.domains}
    rng = np.random.default_rng(args.seed)

    prompts = []          # (domain, prefix_ids[prefix], true_cont_ids[gen])
    while len(prompts) < args.n_prompts:
        d = mixture.domains[rng.choice(len(mixture.domains), p=mixture.weights)]
        try:
            x, _ = next(streams[d])
        except StopIteration:
            continue
        prompts.append((d, x[0, :args.prefix].copy(),
                        x[0, args.prefix:args.prefix + args.gen].copy()))

    results = []
    t0 = time.time()
    for b0 in range(0, len(prompts), args.batch):
        chunk = prompts[b0:b0 + args.batch]
        B = len(chunk)
        ids = np.zeros((B, MAX), dtype=np.int64)
        for i, (_, pre, _) in enumerate(chunk):
            ids[i, :args.prefix] = pre
        cur = args.prefix
        for step in range(args.gen):
            bt = torch.from_numpy(ids).to(device)
            pos = torch.tensor([cur - 1], device=device)
            with torch.no_grad(), torch_xla.amp.autocast(device, dtype=torch.bfloat16):
                out = model(input_ids=bt, use_cache=False)
            lg = out.logits.index_select(1, pos)[:, 0].float().cpu().numpy()
            lg = lg / args.temp
            for i in range(B):                       # top-k sample per row
                row = lg[i]
                kth = np.partition(row, -args.top_k)[-args.top_k]
                row[row < kth] = -1e9
                row -= row.max()
                pr = np.exp(row); pr /= pr.sum()
                ids[i, cur] = rng.choice(len(pr), p=pr)
            cur += 1
            if step == 0:
                print(f"  first token {time.time()-t0:.0f}s (incl. compile)",
                      flush=True)
        for i, (d, pre, true_cont) in enumerate(chunk):
            results.append(dict(
                domain=d,
                prefix=tok.decode(pre.tolist()),
                model=tok.decode(ids[i, args.prefix:cur].tolist()),
                true=tok.decode(true_cont.tolist()),
            ))
        print(f"  batch {b0//args.batch + 1} done [{time.time()-t0:.0f}s]",
              flush=True)

    md = [f"# Vanilla completions (step {step_trained}, "
          f"{toks/1e6:.0f}M tokens)",
          "",
          f"{len(results)} held-out val prefixes ({args.prefix} tokens), "
          f"{args.gen} generated tokens each, temp {args.temp}, "
          f"top-k {args.top_k}. Val mixture loss at end of training: 5.75 "
          f"(ppl ~315).",
          ""]
    for i, r in enumerate(results, 1):
        md += [f"**{i}.** [{r['domain']}]",
               f"- prefix: `...{r['prefix'][-160:]}`",
               f"- model : `{r['model']}`",
               f"- true  : `{r['true']}`",
               ""]
    path = os.path.join(args.out_dir, "VANILLA_GEN_REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
