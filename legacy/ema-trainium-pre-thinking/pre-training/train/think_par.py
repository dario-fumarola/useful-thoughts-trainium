"""Dense parallel thoughts at fixed-stride anchors (PARALLEL_THINKING.md).

Run-13 geometry: one spliced buffer per micro --

    [ clean t_0..t_{T-1} | blocks: <T>,z_0..z_{L-1} per anchor |
      window copies: t_{n_a}..t_{n_a+k-1} per anchor ]

Anchors sit at a FIXED stride, so every mask and position tensor in this
module is a per-run constant (compiled once, never re-shipped). Windows tile
the sequence (stride == k) and never overlap.

Mask semantics (no cross-anchor visibility anywhere):
  * clean rows: causal over clean only -- bit-identical to vanilla; their CE
    is ntp1 and their target logp is the silence baseline.
  * block row (a,t): clean prefix <= n_a, own block causally (<= t).
  * window row (a,j): clean prefix <= n_a - 1, own block (all rows), own
    window causally (<= j). The prefix comes from CLEAN rows, never from
    other windows -- gains are pure per-anchor counterfactuals and oracle
    content cannot leak across anchors.

Torch-xla-free (tests on CPU); device/autocast/sync are injected.
"""

import numpy as np
import torch
import torch.nn.functional as F

import thinking as th
from think_gen import ScatterCache

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
class ParLayout:
    """Index arithmetic for the dense-anchor spliced buffer and the decode
    cache. All arrays here are per-run constants."""

    def __init__(self, T, stride, k, L, first=64):
        assert stride == k, "windows must tile: stride == k"
        self.T, self.stride, self.k, self.L, self.first = T, stride, k, L, first
        last = T - 1 - (k + L)            # oracle span x[n+k+1 .. n+k+L] fits
        self.N = (last - first) // stride + 1
        self.anchors = (first + stride * np.arange(self.N)).astype(np.int64)
        self.blk = L + 1                  # <T>, z_0 .. z_{L-1}
        self.NB = self.N * self.blk
        self.NW = self.N * k
        self.S = T + self.NB + self.NW
        self.blk0 = T
        self.win0 = T + self.NB
        self.C = T + L * self.N           # decode cache: col t = N entries

    def block_row(self, a, t):
        return self.blk0 + a * self.blk + t

    def win_row(self, a, j):
        return self.win0 + a * self.k + j

    def block_rows(self):
        """[N, blk] row indices of every block."""
        return (self.blk0 + np.arange(self.N)[:, None] * self.blk
                + np.arange(self.blk)[None, :])

    def win_rows(self):
        """[N, k] row indices of every window copy."""
        return (self.win0 + np.arange(self.N)[:, None] * self.k
                + np.arange(self.k)[None, :])

    def cache_col(self, t):
        """Cache offsets of decode column t: [N]."""
        return self.T + t * self.N + np.arange(self.N)


def build_positions(layout, slot_pos="built", writer_delay=0, pos_shift=0):
    """float32 [1, S] int-pos: clean j -> j; window (a,j) -> n_a+j
    (overlapping values, mask keeps them apart).  Thought block:
      "built"   <T> at n+1, thought token i at n+2+i  (runs 13-29)
      "aligned" <T> at n+1, thought token i at n+1+i  -- the position of the
                real token x[n+1+i] that the thought stands for.
    Measured zero-update on run 25 @900 (probe_slotpos.py, 616 anchors), with
    the true continuation teacher-forced into the slot: in-slot next-token
    accuracy 0.270 -> 0.305 (clean-text ceiling 0.413) and the reader's gain
    from identical content +2.30 -> +3.15 nats.  The generator's pos_dec must
    use the same convention (ParGenerator(slot_pos=...))."""
    p = np.zeros(layout.S, dtype=np.float32)
    p[:layout.T] = np.arange(layout.T, dtype=np.float32)
    off = 1.0 if slot_pos == "built" else 0.0
    for a in range(layout.N):
        n = float(layout.anchors[a])
        b0 = layout.block_row(a, 0)
        # writer_delay d (slide run): the thought is written right after x[n-d],
        # so the whole block moves d positions earlier. d = 0: unchanged.
        # pos_shift (run 61, moving span): the thought sits at the positions of
        # the span it stands for, x[n+1+pos_shift+i]. 0 = unchanged.
        p[b0] = n + 1.0 - writer_delay + pos_shift        # <T>
        p[b0 + 1:b0 + layout.blk] = (n + 1.0 - writer_delay + pos_shift + off
                                     + np.arange(layout.blk - 1))
        p[layout.win_row(a, 0):layout.win_row(a, 0) + layout.k] = \
            n + np.arange(layout.k)
    return torch.from_numpy(p[None, :])


def build_full_mask(layout, bridge=0, writer_delay=0):
    """Additive fp32 [1,1,S,S] for the scoring/grad pass (constant).

    bridge > 0 hides the last `bridge` context tokens x[n-bridge .. n) from the
    WINDOW rows only.  The thought block still sees them, so the thought becomes
    the only path by which that text can reach the prediction: a job the writer
    can actually do (compress what it just read) instead of the one it cannot
    (forecast what it has not).  Clean rows are untouched, so the NTP loss and
    the model's ordinary behaviour are unchanged."""
    S, T = layout.S, layout.T
    m = np.full((S, S), NEG, dtype=np.float32)
    m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
    for a in range(layout.N):
        n = int(layout.anchors[a])
        b0 = layout.block_row(a, 0)
        w0 = layout.win_row(a, 0)
        for t in range(layout.blk):
            m[b0 + t, :n + 1 - writer_delay] = 0.0
            m[b0 + t, b0:b0 + t + 1] = 0.0
        for j in range(layout.k):
            m[w0 + j, :n] = 0.0
            if bridge > 0:
                m[w0 + j, max(0, n - bridge):n] = NEG
            m[w0 + j, b0:b0 + layout.blk] = 0.0
            m[w0 + j, w0:w0 + j + 1] = 0.0
    return torch.from_numpy(m[None, None])


def build_prefill_mask(layout):
    """[1,1,T,C]: causal over the base cols, NEG on all thought cols."""
    T, C = layout.T, layout.C
    m = np.full((T, C), NEG, dtype=np.float32)
    m[:, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
    return torch.from_numpy(m[None, None])


def build_decode_mask(layout, t, writer_delay=0):
    """[1,1,N,C] for decode step t: row a attends base <= n_a plus its own
    column entries j <= t (col t holds the step's own just-written K/V)."""
    m = np.full((layout.N, layout.C), NEG, dtype=np.float32)
    for a in range(layout.N):
        m[a, :int(layout.anchors[a]) + 1 - writer_delay] = 0.0
        for j in range(t + 1):
            m[a, layout.T + j * layout.N + a] = 0.0
    return torch.from_numpy(m[None, None])


# --------------------------------------------------------------------------
# host-side content builders
# --------------------------------------------------------------------------
def validity(layout, x_row):
    """valid_win[a]: no EOD in x[n .. n+k] (window incl. anchor token);
    valid_or[a]: additionally no EOD in the oracle span x[n+k+1 .. n+k+L].
    float32 [N] each."""
    vw = np.ones(layout.N, dtype=np.float32)
    vo = np.ones(layout.N, dtype=np.float32)
    for a in range(layout.N):
        n = int(layout.anchors[a])
        if (x_row[n:n + layout.k + 1] == EOD).any():
            vw[a] = 0.0
        if vw[a] == 0.0 or \
                (x_row[n + layout.k + 1:n + layout.k + 1 + layout.L]
                 == EOD).any():
            vo[a] = 0.0
    return vw, vo


def oracle_content(layout, x_row):
    """[N, L] oracle thought = the L tokens beyond the scored window."""
    return np.stack([
        x_row[int(n) + layout.k + 1:int(n) + layout.k + 1 + layout.L]
        for n in layout.anchors]).astype(np.int64)


def build_buffer(layout, x_row, content, filler):
    """int64 [S]: clean = x, block a = [<T>, content[a]], window a =
    x[n_a .. n_a+k-1]. content: [N, L]; invalid anchors should carry
    `filler` rows (weights zero them out of every loss)."""
    buf = np.full(layout.S, filler, dtype=np.int64)
    buf[:layout.T] = x_row
    for a in range(layout.N):
        b0 = layout.block_row(a, 0)
        buf[b0] = T_OPEN
        buf[b0 + 1:b0 + layout.blk] = content[a]
        n = int(layout.anchors[a])
        w0 = layout.win_row(a, 0)
        buf[w0:w0 + layout.k] = x_row[n:n + layout.k]
    return buf


def build_targets(layout, x_row, y_row, content):
    """int64 [S] CE targets: clean j -> y[j]; block (a,t) -> content[a][t]
    for t < L (row L is the last content token, no target -> harmless 0);
    window (a,j) -> x[n+1+j]."""
    tgt = np.zeros(layout.S, dtype=np.int64)
    tgt[:layout.T] = y_row
    for a in range(layout.N):
        b0 = layout.block_row(a, 0)
        tgt[b0:b0 + layout.L] = content[a]
        n = int(layout.anchors[a])
        w0 = layout.win_row(a, 0)
        tgt[w0:w0 + layout.k] = x_row[n + 1:n + 1 + layout.k]
    return tgt


# --------------------------------------------------------------------------
# host-side sampling (top-k transfer): ban -> spec argmax -> top-k cut ->
# freq/prev penalties -> temperature -> sample. Mirrors think_gen semantics,
# plus model-controlled STOP: </T> is the one allowed control token (banned
# before min_think); once emitted, the slot pads with </T> and consumes no
# more randomness. A stopped-thought's length is its content-token count.
# --------------------------------------------------------------------------
def sample_step_topk(vals, idx, t, z, stopped, rng, args, spec_row,
                     ban_from):
    """vals/idx: float/int np [R, N, K] device top-K logits per anchor.
    z: int64 [R, N, L] mutated in place at column t.
    stopped: bool [R, N], mutated (True once </T> has been emitted)."""
    order = np.argsort(idx, axis=-1)          # token-id order: deterministic
    idx = np.take_along_axis(idx, order, -1)  # tie-breaks + CDF inversion
    vals = np.take_along_axis(vals, order, -1)
    R, N, K = vals.shape
    for r in range(R):
        vr = vals[r]
        ir = idx[r]
        ban = ir >= ban_from
        if t >= args.min_think:
            ban &= ir != T_CLOSE            # the stop action stays legal
        vr[ban] = -1e9
        if r == spec_row:
            sel = vr.argmax(-1)
            for a in range(N):
                z[r, a, t] = T_CLOSE if stopped[r, a] else ir[a, sel[a]]
        else:
            if 0 < args.top_k_thought < K:
                kth = np.partition(vr, -args.top_k_thought, -1)[
                    :, -args.top_k_thought][:, None]
                vr[vr < kth] = -1e9
            if t > 0:
                for a in range(N):
                    prev = z[r, a, :t]
                    uniq, cnt = np.unique(prev, return_counts=True)
                    match = ir[a][:, None] == uniq[None, :]
                    vr[a] -= args.freq_pen * (match * cnt[None, :]).sum(-1)
                    # consecutive-repeat ban; --repeat-ban 0 lifts it (E1: a
                    # reversed-digit scratchpad repeats digits ~10% of the time)
                    if getattr(args, "repeat_ban", 1):
                        vr[a][ir[a] == prev[t - 1]] = -1e9
            vr = vr / args.temp
            vr -= vr.max(axis=-1, keepdims=True)
            pr = np.exp(vr)
            pr /= pr.sum(-1, keepdims=True)
            for a in range(N):
                if stopped[r, a]:
                    z[r, a, t] = T_CLOSE     # no rng consumed after stop
                    continue
                j = rng.choice(K, p=pr[a])
                z[r, a, t] = ir[a, j]
        stopped[r] |= z[r, :, t] == T_CLOSE


def sample_step_full(lg, t, z, stopped, rng, args, spec_row, ban_from):
    """Full-vocab reference path (tests): lg float np [R, N, V].

    args.code_size > 0 restricts the thought alphabet to the reserved id block
    [code_base, code_base+code_size) plus the stop token: the writer then signals
    with a learned code instead of words, so it cannot be punished for writing
    the wrong TEXT -- only for sending a message the reader cannot use."""
    R, N, V = lg.shape
    code = int(getattr(args, "code_size", 0) or 0)
    if getattr(args, "force_close", 0) and code == 0 and t == z.shape[-1] - 1:
        # run 61: every text thought is TERMINATED in its last slot (G53: an
        # unterminated block costs ~20 nats; run 60's sampled thoughts never closed)
        z[:, :, t] = T_CLOSE
        stopped[:] = True
        return
    if code > 0 and t == z.shape[-1] - 1:
        # A CODE WORD MUST BE TERMINATED (2026-09-22, G53). The block is
        # [<T>, z_0 .. z_{L-1}] with no structural closing slot, so a code run
        # with min_think >= L can never emit </T> and every thought is an
        # UNTERMINATED block -- measured at -19.6 nats against -0.27 for the
        # same tokens terminated (runs/probes/bridge2_r35_200.json, n=154).
        # That framing penalty swamps any reward signal. With --L 5
        # --min-think 4 the writer therefore emits exactly 4 code symbols and
        # this forces the closing tag into the last slot. Text runs (code == 0)
        # are untouched.
        z[:, :, t] = T_CLOSE
        stopped[:] = True
        return
    for r in range(R):
        lgr = lg[r]
        keep_close = None
        if t >= args.min_think:
            keep_close = lgr[:, T_CLOSE].copy()
        if code > 0:
            lo = int(args.code_base); hi = lo + code
            blk = lgr[:, lo:hi].copy()
            lgr[:] = -1e9
            lgr[:, lo:hi] = blk
        allow = getattr(args, "thought_allow", None)
        if allow is not None:
            # run 56: restricted thought alphabet (e.g. digits + space); </T> is
            # handled by keep_close above/below, so it stays banned before min_think
            blk = lgr[:, allow].copy()
            lgr[:] = -1e9
            lgr[:, allow] = blk
        lgr[:, ban_from:] = -1e9 if code == 0 else lgr[:, ban_from:]
        if keep_close is not None:
            lgr[:, T_CLOSE] = keep_close
        if r == spec_row:
            am = lgr.argmax(-1)
            for a in range(N):
                z[r, a, t] = T_CLOSE if stopped[r, a] else am[a]
        else:
            if args.top_k_thought > 0:
                kth = np.partition(lgr, -args.top_k_thought, -1)[
                    :, -args.top_k_thought][:, None]
                lgr[lgr < kth] = -1e9
            if t > 0:
                for a in range(N):
                    prev = z[r, a, :t]
                    uniq, cnt = np.unique(prev, return_counts=True)
                    lgr[a, uniq] -= args.freq_pen * cnt
                    if getattr(args, "repeat_ban", 1):
                        lgr[a, prev[t - 1]] = -1e9
            lgr = lgr / args.temp
            lgr -= lgr.max(axis=-1, keepdims=True)
            pr = np.exp(lgr)
            pr /= pr.sum(-1, keepdims=True)
            for a in range(N):
                if stopped[r, a]:
                    z[r, a, t] = T_CLOSE
                    continue
                z[r, a, t] = rng.choice(V, p=pr[a])
                # run 57: STOP EXPLORATION. With prob stop_explore per position (sampled rows
                # only, after min_think), end the thought here. Run 56 discovered the right
                # first value from reward alone but never sampled </T>, so its thoughts ran
                # on with junk digits and the reader read the junk as the answer.
                _sp = float(getattr(args, "stop_explore", 0.0) or 0.0)
                if _sp > 0 and t >= args.min_think and rng.random() < _sp:
                    z[r, a, t] = T_CLOSE
        stopped[r] |= z[r, :, t] == T_CLOSE


def stop_pos(zr, L):
    """Content length per anchor: index of the first </T> in [N, L] rows
    (== number of content tokens before the stop), or L if never stopped."""
    hit = zr == T_CLOSE
    any_ = hit.any(-1)
    first = np.where(any_, hit.argmax(-1), L)
    return first.astype(np.int64)


def sample_gate(lp_T, eps, thresh, rng):
    """The WHEN-to-think decision (a learned action, not a heuristic): the
    model thinks at anchor a with probability
        q_a = eps + (1 - eps) * min(1, p(<T>)_a / thresh).
    The model's own <T> logit is the learnable trigger (a full function of
    context, strictly richer than any scalar threshold); `thresh` sets how
    permissive sampling is around it and `eps` is the exploration floor that
    keeps the channel alive while the logit is still uninformed. Returns
    (think bool [N], q float [N])."""
    q = eps + (1.0 - eps) * np.minimum(1.0, np.exp(lp_T) / thresh)
    return rng.random(lp_T.shape[0]) < q, q


def crowding(think, win=8):
    """Frequency discount input: crowd[a] = number of active thoughts among
    the `win` anchors preceding a (the 'too many thoughts in a short period'
    tax rides on this count)."""
    t = think.astype(np.float32)
    c = np.zeros_like(t)
    for a in range(1, len(t)):
        c[a] = t[max(0, a - win):a].sum()
    return c


def jitter_oracle(layout, orc, rng, min_len=3):
    """Length-jittered oracle targets for the imitation channel: content =
    orc[:l] + </T> padding, l ~ U{min_len .. L-1}. Teaches the stop action
    and calibrates its logit; the RL length cost later drives compression.
    Returns (content [N, L], lens [N]) -- lens = content tokens before stop."""
    L = layout.L
    lens = rng.integers(min_len, L, size=layout.N)
    content = np.full((layout.N, L), T_CLOSE, dtype=np.int64)
    for a in range(layout.N):
        content[a, :lens[a]] = orc[a, :lens[a]]
    return content, lens.astype(np.int64)


# --------------------------------------------------------------------------
# KV-cached parallel generation
# --------------------------------------------------------------------------
class ParGenerator:
    """Prefill [1,T] once, then L decode launches of [R,N] tokens (row 0 =
    sampled rollout, row spec_row = greedy speculation). All masks/positions
    are per-run constants held on device."""

    def __init__(self, trunk, lm_head, layout, n_layers, R, device,
                 autocast_ctx, sync_fn, topk_transfer=1, slot_pos="built"):
        self.trunk, self.lm_head = trunk, lm_head
        self.layout, self.R, self.device = layout, R, device
        self.autocast_ctx, self.sync = autocast_ctx, sync_fn
        self.topk_transfer = topk_transfer
        self.pre = ScatterCache(n_layers, layout.C)
        self.dec = ScatterCache(n_layers, layout.C)
        self.pos_base = torch.arange(layout.T, dtype=torch.long, device=device)
        self.m_pre = build_prefill_mask(layout).to(torch.bfloat16).to(device)
        self.m_dec = [build_decode_mask(layout, t).to(torch.bfloat16)
                      .to(device) for t in range(layout.L)]
        self.col_t = [torch.from_numpy(layout.cache_col(t)).to(device)
                      for t in range(layout.L)]
        anc = layout.anchors.astype(np.float32)
        # step t feeds <T> (t=0) or thought token t-1; its position must match
        # build_positions(slot_pos) exactly or thoughts are written in one
        # geometry and read in another.
        _off = 1.0 if slot_pos == "built" else 0.0
        self.pos_dec = [torch.from_numpy(
            (anc + 1.0 + (0.0 if t == 0 else _off + (t - 1)))[None, :]
        ).to(device) for t in range(layout.L)]
        self.pos_pre = torch.from_numpy(
            np.arange(layout.T, dtype=np.float32)[None, :]).to(device)
        self.open_toks = torch.full((R, layout.N), T_OPEN, dtype=torch.long,
                                    device=device)
        self.anchor_rows = torch.from_numpy(layout.anchors).to(device)
        self._slot_off = _off
        self.writer_delay = 0

    def set_writer_delay(self, d, pos_shift=0):
        """Slide run: the writer sees base tokens <= n_a - d and writes at
        positions starting n_a + 1 - d. Updates the decode masks and positions
        IN PLACE (same shapes, so no recompilation). d = 0 is the old behaviour."""
        lo = self.layout
        anc = lo.anchors.astype(np.float32)
        for t in range(lo.L):
            self.m_dec[t].copy_(build_decode_mask(lo, t, d).to(torch.bfloat16)
                                .to(self.m_dec[t].device))
            self.pos_dec[t].copy_(torch.from_numpy(
                (anc + 1.0 - d + pos_shift
                 + (0.0 if t == 0 else self._slot_off + (t - 1)))[None, :]
            ).to(self.pos_dec[t].device))
        self.writer_delay = d

    def generate(self, x_dev, rng, args, spec_row, ban_from,
                 collect_logits=None):
        """x_dev: [1, T] long device tensor (clean tokens). Returns
        (z int64 np [R, N, L], lp_T float np [N]) where lp_T is the model's
        own log p(<T> | clean context) at every anchor row -- the LEARNABLE
        think trigger. collect_logits (tests): appends each step's full
        [R, N, V] logits and forces the full-vocab sampling path."""
        was_training = self.trunk.training
        if was_training:
            # transformers drops past_key_values under gradient checkpointing
            # in train() mode (think_gen lesson, 2026-09-02); no dropout in
            # Qwen3 so this is a pure flag flip.
            self.trunk.eval()
        try:
            return self._generate(x_dev, rng, args, spec_row, ban_from,
                                  collect_logits)
        finally:
            if was_training:
                self.trunk.train()

    def _generate(self, x_dev, rng, args, spec_row, ban_from, collect_logits):
        lo, L, N, R = self.layout, self.layout.L, self.layout.N, self.R
        use_topk = self.topk_transfer > 0 and collect_logits is None
        if use_topk:
            assert args.top_k_thought > 0
            K = min(args.top_k_thought + 16, 4096)
        z = np.zeros((R, N, L), dtype=np.int64)
        stopped = np.zeros((R, N), dtype=bool)

        def topk_chunked(lg):
            """torch.topk on [R, N, V] in <=64-row slices: neuronx-cc hits an
            internal error (NCC_INAS001/ISGV902) compiling topk over ~160+
            rows of the full vocab (run-12 lesson: 80 rows compiled, 160
            failed); chunking keeps every slice at the proven size. Results
            are identical to one big topk."""
            flat = lg.float().reshape(R * N, -1)
            vs, is_ = [], []
            for c in range(0, R * N, 64):
                v_, i_ = torch.topk(flat[c:c + 64], K, dim=-1)
                vs.append(v_)
                is_.append(i_)
            return (torch.cat(vs).reshape(R, N, K),
                    torch.cat(is_).reshape(R, N, K))

        with torch.no_grad():
            self.pre.set_positions(self.pos_base)
            with self.autocast_ctx():
                h0 = self.trunk(input_ids=x_dev, position_ids=self.pos_pre,
                                attention_mask=self.m_pre,
                                past_key_values=self.pre, use_cache=True)[0]
                lgA = self.lm_head(h0[:, self.anchor_rows])   # [1, N, V]
            lp_T = torch.log_softmax(lgA.float(), -1)[0, :, T_OPEN] \
                .cpu().numpy()
            self.dec.broadcast_from(self.pre, R)
            self.sync()
            for t in range(L):
                tok_t = self.open_toks if t == 0 else torch.from_numpy(
                    np.ascontiguousarray(z[:, :, t - 1])).to(self.device)
                self.dec.set_positions(self.col_t[t])
                with self.autocast_ctx():
                    ht = self.trunk(input_ids=tok_t,
                                    position_ids=self.pos_dec[t],
                                    attention_mask=self.m_dec[t],
                                    past_key_values=self.dec,
                                    use_cache=True)[0]      # [R, N, H]
                    lgt = self.lm_head(ht)                  # [R, N, V]
                if use_topk:
                    v, ix = topk_chunked(lgt)
                    v, ix = v.cpu().numpy(), ix.cpu().numpy()
                    self.sync()
                    sample_step_topk(v, ix, t, z, stopped, rng, args,
                                     spec_row, ban_from)
                else:
                    lg_np = lgt.float().cpu().numpy()
                    self.sync()
                    if collect_logits is not None:
                        collect_logits.append(lg_np.copy())
                    sample_step_full(lg_np, t, z, stopped, rng, args,
                                     spec_row, ban_from)
        return z, lp_T


# --------------------------------------------------------------------------
# scoring: per-anchor discounted window logp (with-thought vs silence) from
# one no-grad forward of the spliced buffer
# --------------------------------------------------------------------------
def score_buffer(trunk, lm_head, buf_np, pos_dev, mask_dev, layout, wk_t,
                 rows_win_dev, rows_clean_dev, tgt_win_dev, autocast_ctx):
    """Returns (lp_win [N], lp_clean [N]) float64 np: discounted target logp
    of each anchor's window under the with-thought rows and the clean rows of
    the SAME forward. rows_* / tgt_win are precomputed device tensors for the
    constant layout ([N*k] flattened)."""
    with torch.no_grad():
        bt = torch.from_numpy(buf_np[None]).to(pos_dev.device)
        with autocast_ctx():
            h = trunk(input_ids=bt, position_ids=pos_dev,
                      attention_mask=mask_dev, use_cache=False)[0]
        out = []
        for rows in (rows_win_dev, rows_clean_dev):
            with autocast_ctx():
                lg = lm_head(h[0, rows])
            out.append((torch.log_softmax(lg.float(), -1)
                        .gather(1, tgt_win_dev[:, None])
                        .view(layout.N, layout.k) * wk_t)
                       .sum(-1).cpu().numpy())
    return out[0], out[1]


def clean_rows_for_windows(layout):
    """[N*k] clean-row indices matching win_rows() flattened: clean row
    n_a+j predicts the same target as window row (a,j)."""
    return (layout.anchors[:, None] + np.arange(layout.k)[None, :]) \
        .reshape(-1).astype(np.int64)


# --------------------------------------------------------------------------
# the one grad loss: three weighted CE sums over the same hidden states
# --------------------------------------------------------------------------
def weighted_ce3(h, lm_head, targets, w1, w2, w3, n1, n2, n3, chunks,
                 autocast_ctx):
    """loss = sum(ce*w1)/n1 + sum(ce*w2)/n2 + sum(ce*w3)/n3 with ONE chunked
    (checkpointed) vocab projection. Weights may be signed (REINFORCE rides
    w3 with per-row advantage weights), so the norms are explicit scalar
    device tensors set by the host, never w.sum().

    h [1,S,H]; targets [1,S]; w* [1,S]; n* scalar tensors.
    Returns (loss, ce_detached [S]) -- the detached per-row CE lets the
    caller pull ntp1/reader/imitation stats without extra passes."""
    S = h.shape[1]
    step = S // chunks
    ce_parts = []

    def slice_ce(hh, yy):
        with autocast_ctx():
            lg = lm_head(hh)
        return F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                               yy.reshape(-1), reduction="none")

    total = None
    for i in range(chunks):
        sl = slice(i * step, (i + 1) * step if i < chunks - 1 else S)
        ce = torch.utils.checkpoint.checkpoint(
            slice_ce, h[:, sl], targets[:, sl],
            use_reentrant=True, preserve_rng_state=False)
        ce_parts.append(ce.detach())
        part = (ce * w1[0, sl]).sum() / n1 \
            + (ce * w2[0, sl]).sum() / n2 \
            + (ce * w3[0, sl]).sum() / n3
        total = part if total is None else total + part
    return total, torch.cat(ce_parts)


# --------------------------------------------------------------------------
# gradient sync across ranks (2026-10-02). think_train.py never all-reduced
# gradients (the run-13 rewrite dropped it). A single coalesced all-reduce
# fused into the step graph failed to LOAD on this box (NRT_RESOURCE, "Failed
# to create logical core info"), so: materialise the gradients first, then
# all-reduce them in buckets, each bucket its own small graph.
# --------------------------------------------------------------------------
def sync_gradients(params, world, xm, sync_fn, cap_mb=128):
    """Mean-reduce p.grad across `world` ranks in place, bucket by bucket."""
    sync_fn()
    bucket, size = [], 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        bucket.append(g)
        size += g.numel() * g.element_size()
        if size >= cap_mb * 2 ** 20:
            xm.all_reduce(xm.REDUCE_SUM, bucket, scale=1.0 / world)
            sync_fn()
            bucket, size = [], 0
    if bucket:
        xm.all_reduce(xm.REDUCE_SUM, bucket, scale=1.0 / world)
        sync_fn()
