"""CPU equivalence gate for think_par (PARALLEL_THINKING.md, run 13).

Run with:  USE_TORCH_XLA=0 python test_think_par.py
Add RUN_BIG=1 for the full 0.6B-geometry test (slow).

A failure here means the mask/cache/position geometry is wrong and the
parallel path MUST NOT be used for training (the failure mode is silent
wrongness that masquerades as a science result -- the fractional-positions
lesson).

  1. mask invariants: exact allowed-set checks for every row category,
     including the no-cross-anchor-leak and no-self-duplicate guarantees.
  2. buffer/target/position construction: teacher-forcing shift, oracle
     span extraction, EOD validity.
  3. generation equivalence: the KV-cached parallel decode reproduces, step
     by step, the logits of naive per-anchor causal forwards; greedy rows
     match exactly; the run is deterministic under a fixed seed.
  4. topk-transfer sampling == full-vocab sampling, token-exact.
  5. scoring equivalence: score_buffer's with-thought window logp matches a
     naive per-anchor spliced forward; the silence logp matches a plain
     causal forward.
  6. weighted_ce3 == hand-computed three-part loss; gradients flow.
  7. (RUN_BIG=1) generation equivalence at the 0.6B geometry with gradient
     checkpointing enabled and the model in train() mode (eval-flip guard).
"""
import contextlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

assert os.environ.get("USE_TORCH_XLA") == "0", \
    "run with USE_TORCH_XLA=0 (torch_xla must not initialise on CPU)"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thinking as th
import think_par as tp

nullctx = contextlib.nullcontext
nosync = lambda: None
SPACE = 220


def make_tiny(vocab=152000, layers=3):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      intermediate_size=128, max_position_embeddings=4096,
                      use_cache=False, tie_word_embeddings=False)
    torch.manual_seed(7)
    m = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    m.eval()
    return m, cfg


def make_case(T=64, stride=4, k=4, L=5, first=8, seed=3):
    layout = tp.ParLayout(T, stride, k, L, first)
    rng = np.random.default_rng(seed)
    x = rng.integers(10, 5000, size=T + 1).astype(np.int64)
    x[T // 2] = th.EOD                     # a document boundary
    return layout, x[:T], x[1:T + 1]


# ------------------------------------------------------------------ test 1
def test_mask_invariants():
    layout, x, y = make_case()
    m = tp.build_full_mask(layout)[0, 0].numpy()
    T, N, k, L, blk = layout.T, layout.N, layout.k, layout.L, layout.blk
    ok = m == 0.0
    # clean rows: strictly causal, clean only
    for j in range(0, T, 7):
        exp = np.zeros(layout.S, dtype=bool)
        exp[:j + 1] = True
        assert (ok[j] == exp).all(), f"clean row {j}"
    for a in range(N):
        n = int(layout.anchors[a])
        b0, w0 = layout.block_row(a, 0), layout.win_row(a, 0)
        for t in range(blk):
            exp = np.zeros(layout.S, dtype=bool)
            exp[:n + 1] = True                       # clean prefix incl anchor
            exp[b0:b0 + t + 1] = True                # own block causal
            assert (ok[b0 + t] == exp).all(), f"block row a={a} t={t}"
        for j in range(k):
            exp = np.zeros(layout.S, dtype=bool)
            exp[:n] = True                           # clean prefix EXCL anchor
            exp[b0:b0 + blk] = True                  # own block fully
            exp[w0:w0 + j + 1] = True                # own window causal
            assert (ok[w0 + j] == exp).all(), f"win row a={a} j={j}"
    # decode masks: row a at step t sees base <= n_a plus own cols <= t
    for t in range(L):
        dm = tp.build_decode_mask(layout, t)[0, 0].numpy() == 0.0
        for a in range(N):
            exp = np.zeros(layout.C, dtype=bool)
            exp[:int(layout.anchors[a]) + 1] = True
            for j in range(t + 1):
                exp[T + j * N + a] = True
            assert (dm[a] == exp).all(), f"decode t={t} a={a}"
    print("test 1 mask invariants: OK "
          f"(N={N}, S={layout.S}, C={layout.C})")


# ------------------------------------------------------------------ test 2
def test_buffer_targets():
    layout, x, y = make_case()
    orc = tp.oracle_content(layout, x)
    assert orc.shape == (layout.N, layout.L)
    for a in range(layout.N):
        n = int(layout.anchors[a])
        assert (orc[a] == x[n + layout.k + 1:n + layout.k + 1 + layout.L]).all()
    buf = tp.build_buffer(layout, x, orc, SPACE)
    tgt = tp.build_targets(layout, x, y, orc)
    assert (buf[:layout.T] == x).all() and (tgt[:layout.T] == y).all()
    for a in range(layout.N):
        n = int(layout.anchors[a])
        b0, w0 = layout.block_row(a, 0), layout.win_row(a, 0)
        assert buf[b0] == th.T_OPEN
        assert (buf[b0 + 1:b0 + layout.blk] == orc[a]).all()
        # teacher-forcing shift: row t input is content[t-1], target content[t]
        for t in range(layout.L):
            assert tgt[b0 + t] == orc[a][t]
            if t >= 1:
                assert buf[b0 + t] == orc[a][t - 1]
        assert (buf[w0:w0 + layout.k] == x[n:n + layout.k]).all()
        assert (tgt[w0:w0 + layout.k] == x[n + 1:n + 1 + layout.k]).all()
    # positions
    pos = tp.build_positions(layout)[0].numpy()
    assert (pos[:layout.T] == np.arange(layout.T)).all()
    a = layout.N // 2
    n = int(layout.anchors[a])
    assert pos[layout.block_row(a, 0)] == n + 1
    assert pos[layout.block_row(a, layout.L)] == n + 1 + layout.L
    assert pos[layout.win_row(a, 0)] == n
    # validity around the EOD planted at T//2
    vw, vo = tp.validity(layout, x)
    e = layout.T // 2
    for a in range(layout.N):
        n = int(layout.anchors[a])
        in_win = n <= e <= n + layout.k
        in_or = n + layout.k + 1 <= e <= n + layout.k + layout.L
        assert vw[a] == (0.0 if in_win else 1.0), f"vw a={a}"
        assert vo[a] == (0.0 if (in_win or in_or) else 1.0), f"vo a={a}"
    assert vw.sum() < layout.N, "EOD case not exercised"
    print("test 2 buffer/targets/validity: OK")


# ----------------------------------------------------------------- test 2b
def test_stop_and_jitter():
    """Model-controlled length: </T> allowed from min_think on, forces
    padding + no further rng; topk and full paths behave identically on
    synthetic logits; stop_pos/jitter_oracle geometry."""
    layout, x, y = make_case()
    R, N, L, V = 2, layout.N, layout.L, 152000
    args = SimpleNamespace(top_k_thought=50, temp=1.0, freq_pen=1.0,
                           min_think=2)
    rng0 = np.random.default_rng(77)
    base = rng0.standard_normal((R, N, V)).astype(np.float32) * 0.1
    # anchor 0 wants to stop immediately; anchor 1 from t=3 on
    base[:, 0, tp.T_CLOSE] = 10.0
    steps = []
    for t in range(L):
        lg = base.copy()
        if t >= 3:
            lg[:, 1, tp.T_CLOSE] = 10.0
        steps.append(lg)
    K = 60
    z_f = np.zeros((R, N, L), dtype=np.int64)
    z_t = np.zeros((R, N, L), dtype=np.int64)
    stop_f = np.zeros((R, N), dtype=bool)
    stop_t = np.zeros((R, N), dtype=bool)
    rng_f, rng_t = (np.random.default_rng(5), np.random.default_rng(5))
    for t in range(L):
        tp.sample_step_full(steps[t].copy(), t, z_f, stop_f, rng_f, args,
                            1, th.EOD)
        flat = torch.from_numpy(steps[t].copy())
        v, ix = torch.topk(flat, K, dim=-1)
        tp.sample_step_topk(v.numpy(), ix.numpy(), t, z_t, stop_t, rng_t,
                            args, 1, th.EOD)
    assert (z_f == z_t).all(), "stop: topk vs full diverged"
    for r in range(R):
        # min_think ban: no </T> before t=2 despite the huge logit
        assert (z_f[r, 0, :2] != tp.T_CLOSE).all()
        # anchor 0 stops at t=2, padding after
        assert (z_f[r, 0, 2:] == tp.T_CLOSE).all()
        # anchor 1 stops at t=3
        assert (z_f[r, 1, :3] != tp.T_CLOSE).all()
        assert (z_f[r, 1, 3:] == tp.T_CLOSE).all()
    s = tp.stop_pos(z_f[0], L)
    assert s[0] == 2 and s[1] == 3
    assert (s[2:] == L).all() or (z_f[0, 2:] == tp.T_CLOSE).any(-1).any()
    # jittered oracle targets
    orc = tp.oracle_content(layout, x)
    content, lens = tp.jitter_oracle(layout, orc, np.random.default_rng(1))
    assert ((lens >= 3) & (lens <= L - 1)).all()
    for a in range(layout.N):
        assert (content[a, :lens[a]] == orc[a, :lens[a]]).all()
        assert (content[a, lens[a]:] == tp.T_CLOSE).all()
        assert tp.stop_pos(content, L)[a] == lens[a]
    # gate sampling rule + crowding
    lp = np.log(np.array([1e-9, 1e-4, 1e-3, 5e-1] * 3))
    think, q = tp.sample_gate(lp, eps=0.0, thresh=1e-3,
                              rng=np.random.default_rng(2))
    assert q[3] == 1.0 and q[2] == 1.0          # p >= thresh -> certain
    assert abs(q[1] - 0.1) < 1e-9               # proportional below
    assert q[0] < 1e-5
    _, q_eps = tp.sample_gate(lp, eps=0.5, thresh=1e-3,
                              rng=np.random.default_rng(2))
    assert q_eps.min() >= 0.5                   # exploration floor
    tk = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    cw = tp.crowding(tk, win=3)
    assert cw[0] == 0 and cw[3] == 2 and cw[4] == 2 and cw[9] == 0
    print("test 2b stop mechanics + jittered oracle + gate rule: OK")


# ------------------------------------------------------------------ test 3
def naive_step_logits(model, x, layout, z, r, a, t):
    """Causal forward on [x_0..x_{n_a}, <T>, z_0..z_{t-1}]: the reference
    conditional for decode step t of rollout r, anchor a."""
    n = int(layout.anchors[a])
    seq = np.concatenate([x[:n + 1], [th.T_OPEN], z[r, a, :t]])
    with torch.no_grad():
        out = model.model(input_ids=torch.from_numpy(seq[None]),
                          use_cache=False)[0]
        lg = model.lm_head(out[:, -1])
    return lg.float().numpy()[0]


def test_generation_equivalence(atol=2e-4):
    model, cfg = make_tiny()
    layout, x, y = make_case()
    args = SimpleNamespace(top_k_thought=50, temp=1.0, freq_pen=1.0,
                           min_think=2)
    R, spec_row = 2, 1
    gen = tp.ParGenerator(model.model, model.lm_head, layout,
                          cfg.num_hidden_layers, R, torch.device("cpu"),
                          nullctx, nosync, topk_transfer=0)
    collected = []
    z, lpT = gen.generate(torch.from_numpy(x[None]),
                          np.random.default_rng(123),
                          args, spec_row, th.EOD, collect_logits=collected)
    assert len(collected) == layout.L
    # the gate signal: lp_T must equal the plain causal forward's
    # log p(<T>) at every anchor row
    with torch.no_grad():
        hh = model.model(input_ids=torch.from_numpy(x[None]),
                         use_cache=False)[0]
        lg = model.lm_head(hh[0, layout.anchors])
        ref_lpT = torch.log_softmax(lg.float(), -1)[:, tp.T_OPEN].numpy()
    assert np.abs(ref_lpT - lpT).max() < 2e-4, "gate lp_T mismatch"
    worst, bad_argmax = 0.0, 0
    for t in range(layout.L):
        for a in range(0, layout.N, max(1, layout.N // 5)):
            for r in range(R):
                ref = naive_step_logits(model, x, layout, z, r, a, t)
                got = collected[t][r, a]
                worst = max(worst, float(np.abs(ref - got).max()))
                if ref.argmax() != got.argmax():
                    bad_argmax += 1
    assert worst < atol, f"logit mismatch {worst:.2e}"
    assert bad_argmax == 0
    # greedy row: tokens themselves must be the banned-argmax of the ref
    # (with </T> exempt from the ban once t >= min_think, and forced padding
    # after a stop)
    for t in range(layout.L):
        for a in range(0, layout.N, 3):
            if (z[spec_row, a, :t] == tp.T_CLOSE).any():
                assert z[spec_row, a, t] == tp.T_CLOSE
                continue
            ref = naive_step_logits(model, x, layout, z, spec_row, a, t)
            keep = ref[tp.T_CLOSE] if t >= args.min_think else None
            ref[th.EOD:] = -1e9
            if keep is not None:
                ref[tp.T_CLOSE] = keep
            assert ref.argmax() == z[spec_row, a, t], f"greedy t={t} a={a}"
    # determinism
    z2, _ = gen.generate(torch.from_numpy(x[None]),
                         np.random.default_rng(123),
                         args, spec_row, th.EOD, collect_logits=[])
    assert (z == z2).all()
    print(f"test 3 generation equivalence: OK (worst |dlogit| {worst:.2e})")
    return model, cfg, layout, x, y, z


# ------------------------------------------------------------------ test 4
def test_topk_exact(model, cfg, layout, x):
    args = SimpleNamespace(top_k_thought=50, temp=1.0, freq_pen=1.0,
                           min_think=2)
    gen_full = tp.ParGenerator(model.model, model.lm_head, layout,
                               cfg.num_hidden_layers, 2, torch.device("cpu"),
                               nullctx, nosync, topk_transfer=0)
    gen_topk = tp.ParGenerator(model.model, model.lm_head, layout,
                               cfg.num_hidden_layers, 2, torch.device("cpu"),
                               nullctx, nosync, topk_transfer=1)
    zf, lpf = gen_full.generate(torch.from_numpy(x[None]),
                                np.random.default_rng(55), args, 1, th.EOD)
    zt, lpt = gen_topk.generate(torch.from_numpy(x[None]),
                                np.random.default_rng(55), args, 1, th.EOD)
    assert (zf == zt).all(), \
        f"topk path diverged at {np.argwhere(zf != zt)[:4]}"
    assert np.abs(lpf - lpt).max() < 1e-6
    print("test 4 topk-transfer sampling: token-exact OK")


# ------------------------------------------------------------------ test 5
def naive_window_logp(model, x, layout, content, a, gamma):
    """Spliced per-anchor reference: [x_0..x_{n_a}, <T>, z.., x_{n_a}..].
    Causal-in-order EXCEPT window rows never attend the clean copy of
    x_{n_a} (they'd double-count their own first token)."""
    n = int(layout.anchors[a])
    k, L = layout.k, layout.L
    seq = np.concatenate([x[:n + 1], [th.T_OPEN], content[a],
                          x[n:n + k]]).astype(np.int64)
    P = len(seq)
    pos = np.concatenate([np.arange(n + 1), n + 1 + np.arange(L + 1),
                          n + np.arange(k)]).astype(np.float32)
    m = np.triu(np.full((P, P), th.NEG, dtype=np.float32), k=1)
    w0 = n + 1 + L + 1
    m[w0:, n] = th.NEG                     # window rows: no clean-anchor dup
    with torch.no_grad():
        h = model.model(input_ids=torch.from_numpy(seq[None]),
                        position_ids=torch.from_numpy(pos[None]),
                        attention_mask=torch.from_numpy(m[None, None]),
                        use_cache=False)[0]
        lg = model.lm_head(h[0, w0:w0 + k])
        lp = torch.log_softmax(lg.float(), -1)
        tgt = torch.from_numpy(x[n + 1:n + 1 + k].astype(np.int64))
        lpw = lp.gather(1, tgt[:, None])[:, 0].numpy()
    return float((lpw * gamma ** np.arange(k)).sum())


def test_scoring_equivalence(model, cfg, layout, x, y, z, atol=2e-4):
    gamma = 0.9
    content = z[0]                                     # sampled rollout
    buf = tp.build_buffer(layout, x, content, SPACE)
    pos = tp.build_positions(layout)
    mask = tp.build_full_mask(layout)
    wk = torch.from_numpy((gamma ** np.arange(layout.k)).astype(np.float32))
    rows_win = torch.from_numpy(layout.win_rows().reshape(-1))
    rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
    tgt_win = torch.from_numpy(
        np.stack([x[int(n) + 1:int(n) + 1 + layout.k]
                  for n in layout.anchors]).reshape(-1).astype(np.int64))
    lp_win, lp_clean = tp.score_buffer(
        model.model, model.lm_head, buf, pos, mask, layout, wk,
        rows_win, rows_clean, tgt_win, nullctx)
    worst = 0.0
    for a in range(0, layout.N, max(1, layout.N // 6)):
        ref = naive_window_logp(model, x, layout, content, a, gamma)
        worst = max(worst, abs(ref - float(lp_win[a])))
    assert worst < atol, f"win logp mismatch {worst:.2e}"
    # silence baseline == plain causal forward
    with torch.no_grad():
        h = model.model(input_ids=torch.from_numpy(x[None]),
                        use_cache=False)[0]
        lg = model.lm_head(h[0])
        lp = torch.log_softmax(lg.float(), -1)
    worst_c = 0.0
    for a in range(0, layout.N, max(1, layout.N // 6)):
        n = int(layout.anchors[a])
        ref = sum(float(lp[n + j, x[n + 1 + j]]) * gamma ** j
                  for j in range(layout.k))
        worst_c = max(worst_c, abs(ref - float(lp_clean[a])))
    assert worst_c < atol, f"clean logp mismatch {worst_c:.2e}"
    print(f"test 5 scoring equivalence: OK (win {worst:.2e} "
          f"clean {worst_c:.2e})")


# ------------------------------------------------------------------ test 6
def test_weighted_ce3(model, cfg, layout, x, y, z):
    content = z[0]
    buf = tp.build_buffer(layout, x, content, SPACE)
    tgt = tp.build_targets(layout, x, y, content)
    pos = tp.build_positions(layout)
    mask = tp.build_full_mask(layout)
    rng = np.random.default_rng(0)
    w1 = np.zeros(layout.S, dtype=np.float32); w1[:layout.T] = 1.0
    w2 = np.zeros(layout.S, dtype=np.float32)
    w2[layout.win0:] = rng.random(layout.NW).astype(np.float32)
    w3 = np.zeros(layout.S, dtype=np.float32)
    w3[layout.blk0:layout.win0] = \
        rng.standard_normal(layout.NB).astype(np.float32)   # signed (RL)
    for a in range(layout.N):                # last block row: no target
        w3[layout.block_row(a, layout.L)] = 0.0
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": True, "preserve_rng_state": False})
    h = model.model(input_ids=torch.from_numpy(buf[None]),
                    position_ids=pos, attention_mask=mask,
                    use_cache=False)[0]
    t_ = lambda v: torch.from_numpy(v[None])
    n1, n2, n3 = (torch.tensor(float(layout.T)),
                  torch.tensor(7.0), torch.tensor(5.0))
    loss, ce_det = tp.weighted_ce3(h, model.lm_head, t_(tgt), t_(w1), t_(w2),
                                   t_(w3), n1, n2, n3, 4, nullctx)
    with torch.no_grad():
        lg = model.lm_head(h)
        ce_ref = F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                 torch.from_numpy(tgt), reduction="none")
        ref = (ce_ref * t_(w1)[0]).sum() / n1 \
            + (ce_ref * t_(w2)[0]).sum() / n2 \
            + (ce_ref * t_(w3)[0]).sum() / n3
    assert abs(float(loss) - float(ref)) < 1e-3 * abs(float(ref)), \
        f"{float(loss)} vs {float(ref)}"
    assert float((ce_det - ce_ref).abs().max()) < 1e-4
    loss.backward()
    gnorm = model.model.embed_tokens.weight.grad.abs().sum()
    assert torch.isfinite(loss) and float(gnorm) > 0
    model.eval()
    print(f"test 6 weighted_ce3: OK (loss {float(loss):+.4f}, grads flow)")


# ------------------------------------------------------------------ test 7
def test_full_size(atol=1e-4):
    from model_init import build_model
    model, cfg, _ = build_model(seq_len=512, seed=11)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": True, "preserve_rng_state": False})
    model.train()                          # exercises the eval-flip guard
    layout = tp.ParLayout(T=256, stride=8, k=8, L=3, first=32)
    rng = np.random.default_rng(9)
    x = rng.integers(10, 5000, size=256).astype(np.int64)
    args = SimpleNamespace(top_k_thought=50, temp=1.0, freq_pen=1.0,
                           min_think=2)
    gen = tp.ParGenerator(model.model, model.lm_head, layout,
                          cfg.num_hidden_layers, 2, torch.device("cpu"),
                          nullctx, nosync, topk_transfer=0)
    collected = []
    z, _ = gen.generate(torch.from_numpy(x[None]), np.random.default_rng(3),
                        args, 1, th.EOD, collect_logits=collected)
    model.eval()
    worst, bad = 0.0, 0
    for t in range(layout.L):
        for a in (0, layout.N // 2, layout.N - 1):
            for r in range(2):
                ref = naive_step_logits(model, x, layout, z, r, a, t)
                got = collected[t][r, a]
                worst = max(worst, float(np.abs(ref - got).max()))
                if ref.argmax() != got.argmax():
                    bad += 1
    assert bad == 0 and worst < atol, f"full-size mismatch {worst:.2e}"
    print(f"test 7 full 0.6B geometry (+ckpt+train eval-flip): OK "
          f"(worst {worst:.2e})")


if __name__ == "__main__":
    test_mask_invariants()
    test_buffer_targets()
    test_stop_and_jitter()
    model, cfg, layout, x, y, z = test_generation_equivalence()
    test_topk_exact(model, cfg, layout, x)
    test_scoring_equivalence(model, cfg, layout, x, y, z)
    test_weighted_ce3(model, cfg, layout, x, y, z)
    if os.environ.get("RUN_BIG") == "1":
        test_full_size()
    else:
        print("test 7 skipped (set RUN_BIG=1)")
    print("ALL TESTS PASSED")
