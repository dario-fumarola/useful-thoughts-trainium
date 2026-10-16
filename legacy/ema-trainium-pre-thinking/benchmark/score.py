"""Zero-shot likelihood scoring of a checkpoint, with or without thinking.

No fine-tuning and no task heads: the same frozen weights answer every task,
scored by ranking the log-likelihood of each candidate continuation. This is
the standard base-LM protocol (GPT-3 / Pythia / OLMo / LLaMA all report it).

Two modes, identical in every respect except what sits in the context:

  --mode clean   score  P(choice | context)
  --mode think   the model first writes a thought after the context; that
                 thought is spliced in as <T>...</T> and the choice is then
                 scored as  P(choice | context, <T>thought</T>).

INFERENCE-TIME THINKING, not the training-time layout: one thought, at the
end of the prompt, immediately before the answer. That is the question a
benchmark actually asks ("does thinking before answering help?"), and it is
independent of the anchor geometry a checkpoint happened to train with, so
one script compares checkpoints from different runs.

Per-example gold log-probs are written out so comparisons are PAIRED.
Runs on CPU by default and never touches the Neuron device, so training can
continue alongside:

  USE_TORCH_XLA=0 python benchmark/score.py --ckpt <pt> --mode clean \
      --out benchmark/results/<name>.jsonl
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/ema-trainium/pre-training/train")
from model_init import build_model, untie_lm_head, MODEL_ID   # noqa: E402
import thinking as th                                          # noqa: E402


def load_model(ckpt, seq_len, device):
    model, _, _ = build_model(MODEL_ID, seq_len=seq_len, seed=0)
    untie_lm_head(model)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
          for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("mix.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(m.startswith("mix.") for m in missing), missing[:5]
    return model.float().eval().to(device)


def block_positions(n, B, slot_pos="built"):
    """Positions of the thought block, mirroring think_par.build_positions.

      "built"   <T> at n+1, thought token i at n+2+i  (runs 13-29)
      "aligned" <T> at n+1, thought token i at n+1+i  (runs 30+, --slot-pos
                aligned: the position of the real token the thought stands for)

    The default reproduces the geometry this script used before 2026-09-20.
    A checkpoint scored in the wrong register sees a layout it never trained
    in -- the Section-3 failure, worth 0.85 nats on run 25.
    """
    off = 1 if slot_pos == "built" else 0
    return [n + 1] + [n + 1 + off + i for i in range(B - 1)]


@torch.no_grad()
def think_logprob(model, ctx_ids, block_ids, cont_ids, device,
                  slot_pos="built"):
    """sum log P(cont | ctx, thought), replicating the TRAINING layout exactly.

    think_par assigns clean j -> j, thought t at anchor n -> n+1+t, window j
    -> n+j: thought and continuation deliberately OCCUPY THE SAME POSITIONS,
    held apart by the mask. Two things must both be right or the model sees a
    geometry it never trained on (the Section-3 failure in a new costume):

      positions  ctx 0..n | block n+1.. | window n, n+1, ...
      mask       block sees ctx[0..n] and itself, causally;
                 window sees ctx[0..n-1], the block, and itself, causally --
                 crucially NOT ctx[n], which carries the same position n as
                 window row 0 and would be a duplicate under plain causality.
    """
    n = len(ctx_ids) - 1
    win = [ctx_ids[-1]] + cont_ids          # window row 0 re-emits x_n
    C, B, W = len(ctx_ids), len(block_ids), len(win)
    S = C + B + W
    ids = ctx_ids + block_ids + win
    pos = list(range(C)) + block_positions(n, B, slot_pos) \
        + [n + i for i in range(W)]
    NEG = torch.finfo(torch.float32).min
    m = torch.full((S, S), NEG, dtype=torch.float32)
    ar = torch.arange(S)
    # clean rows: ordinary causal over the context
    m[:C, :C] = torch.where(ar[:C, None] >= ar[None, :C], 0.0, NEG)
    # block rows: all of the context, plus themselves causally
    m[C:C + B, :C] = 0.0
    m[C:C + B, C:C + B] = torch.where(
        ar[:B, None] >= ar[None, :B], 0.0, NEG)
    # window rows: context STRICTLY BEFORE the anchor, the block, themselves
    m[C + B:, :C - 1] = 0.0
    m[C + B:, C:C + B] = 0.0
    m[C + B:, C + B:] = torch.where(ar[:W, None] >= ar[None, :W], 0.0, NEG)
    lg = model(input_ids=torch.tensor([ids], device=device),
               position_ids=torch.tensor([pos], device=device),
               attention_mask=m[None, None].to(device)).logits.float()
    lp = torch.log_softmax(lg[0], -1)
    w0 = C + B
    return float(sum(lp[w0 + i, t] for i, t in enumerate(cont_ids))), len(cont_ids)


@torch.no_grad()
def seq_logprob(model, tok, ctx_ids, cont_ids, device):
    """sum log P(cont | ctx) and the token count, one sequence."""
    ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
    lg = model(input_ids=ids).logits.float()
    lp = torch.log_softmax(lg[0, :-1], -1)
    tgt = ids[0, 1:]
    n = len(cont_ids)
    sel = lp[-n:].gather(1, tgt[-n:, None]).squeeze(1)
    return float(sel.sum()), n


@torch.no_grad()
def write_thought(model, ctx_ids, L, device, temperature=0.0,
                  slot_pos="built"):
    """Greedy (or sampled) L-token thought, stopped at </T>. Returns ids.

    Decoded in the same register the thought will be scored in: with
    slot_pos="built" the explicit positions are identical to the implicit
    0..S-1 this function used before, so the default is behaviour-preserving.
    """
    n = len(ctx_ids) - 1
    ids = torch.tensor([ctx_ids + [th.T_OPEN]], dtype=torch.long, device=device)
    out = []
    for _ in range(L):
        pos = list(range(len(ctx_ids))) + block_positions(
            n, ids.shape[1] - len(ctx_ids), slot_pos)
        lg = model(input_ids=ids,
                   position_ids=torch.tensor([pos], device=device)
                   ).logits[0, -1].float()
        lg[th.EOD] = -1e9
        if temperature > 0:
            nxt = int(torch.multinomial(
                torch.softmax(lg / temperature, -1), 1))
        else:
            nxt = int(lg.argmax())
        if nxt == th.T_CLOSE:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("clean", "think"), default="clean")
    ap.add_argument("--bench", default=os.path.join(here, "data", "*.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--think-len", type=int, default=16)
    ap.add_argument("--think-temp", type=float, default=0.0)
    ap.add_argument("--slot-pos", choices=("built", "aligned"), default="built",
                    help="thought-block register; must match the run's "
                         "--slot-pos (runs 30+ trained 'aligned')")
    ap.add_argument("--max-ctx", type=int, default=768)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dev = torch.device(a.device)
    model = load_model(a.ckpt, a.seq_len, dev)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    files = sorted(glob.glob(a.bench))
    assert files, f"no benchmarks matched {a.bench}"
    fout = open(a.out, "w")
    t0 = time.time()
    for path in files:
        rows = [json.loads(l) for l in open(path)]
        if a.limit:
            rows = rows[:a.limit]
        name = rows[0].get("task", os.path.basename(path).split(".")[0])
        for i, r in enumerate(rows):
            ctx_ids = tok(r["context"], add_special_tokens=False)["input_ids"]
            ctx_ids = ctx_ids[-a.max_ctx:]
            thought, block = [], []
            if a.mode == "think":
                thought = write_thought(model, ctx_ids, a.think_len, dev,
                                        a.think_temp, a.slot_pos)
                block = [th.T_OPEN] + thought + [th.T_CLOSE]

            def score(cont):
                if block:
                    return think_logprob(model, ctx_ids, block, cont, dev,
                                         a.slot_pos)
                return seq_logprob(model, tok, ctx_ids, cont, dev)
            rec = dict(task=name, idx=i, mode=a.mode)
            if a.mode == "think":
                rec["slot_pos"] = a.slot_pos
            if "choices" in r:
                lps, lens = [], []
                for c in r["choices"]:
                    cid = tok(c, add_special_tokens=False)["input_ids"]
                    if not cid:
                        cid = [tok.eos_token_id]
                    lp, n = score(cid)
                    lps.append(lp); lens.append(n)
                lps = np.array(lps); lens = np.array(lens)
                g = int(r["gold"])
                rec.update(gold=g, logprobs=lps.tolist(),
                           gold_logprob=float(lps[g]),
                           pred=int(lps.argmax()),
                           correct=int(lps.argmax() == g),
                           pred_norm=int((lps / lens).argmax()),
                           correct_norm=int((lps / lens).argmax() == g),
                           n_choices=len(lps))
            else:                                   # lambada: final word
                cid = tok(r["target"], add_special_tokens=False)["input_ids"]
                lp, n = score(cid)
                if block:
                    # greedy under the identical corrected layout: score a
                    # 1-token window and read its argmax
                    _n = len(ctx_ids) - 1
                    _C, _B = len(ctx_ids), len(block)
                    _ids = ctx_ids + block + [ctx_ids[-1]]
                    _pos = list(range(_C)) + block_positions(
                        _n, _B, a.slot_pos) + [_n]
                    _S = len(_ids)
                    _NEG = torch.finfo(torch.float32).min
                    _m = torch.full((_S, _S), _NEG, dtype=torch.float32)
                    _ar = torch.arange(_S)
                    _m[:_C, :_C] = torch.where(_ar[:_C, None] >= _ar[None, :_C], 0.0, _NEG)
                    _m[_C:_C + _B, :_C] = 0.0
                    _m[_C:_C + _B, _C:_C + _B] = torch.where(
                        _ar[:_B, None] >= _ar[None, :_B], 0.0, _NEG)
                    _m[_C + _B:, :_C - 1] = 0.0
                    _m[_C + _B:, _C:_C + _B] = 0.0
                    _m[_C + _B:, _C + _B:] = 0.0
                    with torch.no_grad():
                        greedy = int(model(
                            input_ids=torch.tensor([_ids], device=dev),
                            position_ids=torch.tensor([_pos], device=dev),
                            attention_mask=_m[None, None].to(dev)
                        ).logits[0, -1].argmax())
                else:
                    ids = torch.tensor([ctx_ids], dtype=torch.long, device=dev)
                    with torch.no_grad():
                        greedy = int(model(input_ids=ids).logits[0, -1].argmax())
                rec.update(gold=-1, gold_logprob=lp, n_tokens=n,
                           correct=int(greedy == cid[0]),
                           correct_norm=int(greedy == cid[0]))
            if thought:
                rec["thought"] = tok.decode(thought)
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print(f"  {name:14s} {len(rows):4d} done  ({time.time()-t0:6.0f}s)",
              flush=True)
    fout.close()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
