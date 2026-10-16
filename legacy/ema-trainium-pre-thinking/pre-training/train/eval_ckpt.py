"""Standard val-mixture eval (same protocol as train.py's in-loop eval that
produced the vanilla 5.75 figure) for any checkpoint: vanilla or Stage-A."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.amp

from evaluate import evaluate, format_eval
from model_init import build_model, untie_lm_head, MODEL_ID


def main():
    p = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus", default=os.path.join(base, "..", "data", "corpus"))
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--eval-sequences", type=int, default=8)
    p.add_argument("--loss-chunks", type=int, default=4)
    args = p.parse_args()

    device = torch_xla.device()
    autocast_ctx = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone, cfg, _ = build_model(MODEL_ID, seq_len=args.seq_len, seed=0)
    untie_lm_head(backbone)          # ck holds two diverged matrices
    # This tool always evaluates the BACKBONE in clean mode, so a thinking
    # checkpoint's mixing head (mix.*) and any "backbone." prefix are stripped
    # rather than reconstructed -- same handling as runs/eval_ckpt_cpu.py.
    # (The old branch imported ThinkModel from think_train.py purely to satisfy
    # a strict load; that module was removed in the 2026-09-06 cleanup, so the
    # branch raised ImportError on every thinking checkpoint.)
    sd = ck["model"]
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
          for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("mix.")}
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(m.startswith("mix.") for m in missing), missing[:5]
    step, toks = ck.get("step"), ck.get("tokens_seen", 0)
    del ck
    backbone = backbone.to(torch.bfloat16).to(device)
    backbone.eval()

    res = evaluate(backbone, args.corpus, device, args.seq_len, 1,
                   args.eval_sequences, autocast_ctx, torch_xla.sync,
                   args.loss_chunks)
    print(f"[{os.path.basename(args.ckpt)} step {step} +{toks/1e6:.1f}M] "
          f"{format_eval(res)}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
