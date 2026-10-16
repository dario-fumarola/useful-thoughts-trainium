"""KV-cached thought generation + fused rollout scoring (SPEEDUP_THINKING.md).

Torch-xla-free (like thinking.py): device/autocast/sync concerns are injected
by the caller, so everything here unit-tests on CPU (test_think_fast.py).

Why a custom cache: transformers' StaticLayer fills positions sequentially
(an internal cumulative counter). Our decode steps write 8 *scattered* rows
(one per slot: block_start(a)+i), so ScatterCache index-copies at positions
set on the cache before each forward. Reads return the full [B, H, S, D]
tensors; the mask guarantees no stale row is ever attended -- every row a
query can see was written by the prefill or by an earlier decode step of the
same micro-batch.

Semantics (the one deliberate difference from the slow sampler): under the
training mask the anchor row n_a attends the filling block, so a thought
token being sampled sees an *echo* of its own partial thought through the
real stream, recursively per layer. An incremental cache cannot reproduce
that recursion -- every cached block row's deep-layer k/v would be a
function of the current fill -- so fast sampling conditions on the clean
prefix (build_mask(no_echo=True) semantics). Scoring and pass-2 keep the
echo unchanged. The sampler/scorer conditional mismatch this introduces is
of the same order as the fill mismatch the slow sampler already has (module
docstring of think_train.py). Validated by: logit equivalence against a
no-echo slow reference (test_think_fast.py), oracle-ceiling probes, and a
fast-vs-slow A/B at run start.
"""

import numpy as np
import torch

import thinking as th

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:                                    # very old transformers
    Cache, CacheLayerMixin = object, object


class _ScatterLayer(CacheLayerMixin):
    """One layer of ScatterCache: fixed [B, H, S, D] tensors, in-place
    index_copy_ at externally supplied row positions."""

    is_sliding = False
    is_compileable = True

    def __init__(self, holder):
        self.holder = holder                # shared: {"pos": LongTensor, "S": int}
        self.keys = None
        self.values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states, value_states):
        B, H, _, D = key_states.shape
        S = self.holder["S"]
        self.keys = torch.zeros((B, H, S, D), dtype=key_states.dtype,
                                device=key_states.device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        pos = self.holder["pos"]
        self.keys.index_copy_(2, pos, key_states)
        self.values.index_copy_(2, pos, value_states)
        return self.keys, self.values

    def get_seq_length(self):
        return self.holder["S"] if self.is_initialized else 0

    def get_max_length(self):
        return self.holder["S"]

    def get_mask_sizes(self, query_length):
        return self.holder["S"], 0

    def reset(self):
        pass


class ScatterCache(Cache):
    """A Cache whose write positions are cache state, set before each forward
    (transformers 5.x attention passes no positions to update())."""

    def __init__(self, n_layers, S):
        self.holder = {"pos": None, "S": S}
        super().__init__(layers=[_ScatterLayer(self.holder)
                                 for _ in range(n_layers)])

    def set_positions(self, pos):
        self.holder["pos"] = pos

    def broadcast_from(self, src, batch):
        """Initialise every layer as `batch` copies of src's (batch-1) k/v."""
        for ld, ls in zip(self.layers, src.layers):
            if not ld.is_initialized:
                B, H, S, D = ls.keys.shape
                ld.keys = torch.zeros((batch, H, S, D), dtype=ls.keys.dtype,
                                      device=ls.keys.device)
                ld.values = torch.zeros_like(ld.keys)
                ld.is_initialized = True
            ld.keys.copy_(ls.keys.expand_as(ld.keys))
            ld.values.copy_(ls.values.expand_as(ld.values))


def decode_mask_rows(layout, anchors, i):
    """Rows block_start(a)+i of th.build_mask(..., fill=i+1), built directly:
    additive fp32 [1, 1, slots, S]. Verified equal to the build_mask slice in
    test_think_fast.py."""
    S = layout.S
    m = np.full((layout.slots, S), th.NEG, dtype=np.float32)
    for a in range(layout.slots):
        s = layout.block_start(a)
        n = anchors[a]
        if n < 0:
            m[a, s + i] = 0.0
            continue
        m[a, :n + 1] = 0.0
        m[a, s:s + i + 1] = 0.0
    return m[None, None]


def sample_host_step_topk(vals, idx, i, buf2, hist2, layout, rng, args,
                          spec_idx, disallow, ban_from):
    """Top-k variant of sample_host_step: operates on device-side top-K
    (values, indices) instead of full-vocab logits, killing the ~97MB/launch
    device->host transfer and the 152k-wide host shaping that run-11
    profiling showed to be the wall (SPEEDUP_THINKING.md section 8).

    Semantics match the full-vocab path exactly whenever the post-ban
    top-(top_k_thought) set is contained in the device top-K -- guaranteed
    up to the documented epsilon by passing K = top_k_thought + headroom
    (more banned control ids than the headroom inside the device top-K
    would be needed to diverge). Shaping order mirrors the original: ban ->
    spec-argmax -> disallow -> top_k_thought cut -> repeat penalties ->
    temperature -> sample.

    vals/idx: float/int np [R, slots, K]. Mutates buf2/hist2 in place.
    """
    # sort candidates by token id: rng.choice inverts the CDF in candidate
    # order, and the full-vocab path's order IS token-id order -- sorting
    # makes the same uniform draw select the same token (and reproduces
    # argmax's lowest-id tie-break), i.e. the two paths are stream-exact.
    order = np.argsort(idx, axis=-1)
    idx = np.take_along_axis(idx, order, -1)
    vals = np.take_along_axis(vals, order, -1)
    R_roll = vals.shape[0]
    for r in range(R_roll):
        vr = vals[r]                                  # [slots, K]
        ir = idx[r]
        vr[ir >= ban_from] = -1e9
        if r == spec_idx:
            sel = vr.argmax(-1)
            for a in range(layout.slots):
                z = int(ir[a, sel[a]])
                hist2[r, a, i] = z
                buf2[r, layout.block_start(a) + 1 + i] = z
            continue
        if disallow is not None:
            vr[np.isin(ir, disallow)] = -1e9
        if 0 < args.top_k_thought < vr.shape[-1]:
            kth = np.partition(vr, -args.top_k_thought, -1)[
                :, -args.top_k_thought][:, None]
            vr[vr < kth] = -1e9
        if i > 0:
            for a in range(layout.slots):
                prev = hist2[r, a, :i]
                uniq, cnt = np.unique(prev, return_counts=True)
                match = ir[a][:, None] == uniq[None, :]      # [K, U]
                vr[a] -= args.freq_pen * (match * cnt[None, :]).sum(-1)
                vr[a][ir[a] == prev[i - 1]] = -1e9
        vr = vr / (args.temp if r == 0 else (args.temp2 or args.temp))
        vr -= vr.max(axis=-1, keepdims=True)
        pr = np.exp(vr)
        pr /= pr.sum(-1, keepdims=True)
        for a in range(layout.slots):
            j = rng.choice(pr.shape[-1], p=pr[a])
            z = int(ir[a, j])
            hist2[r, a, i] = z
            buf2[r, layout.block_start(a) + 1 + i] = z


def sample_host_step(lg, i, buf2, hist2, layout, rng, args, spec_idx,
                     disallow, ban_from):
    """Host-side shaping + sampling for one generation step, shared verbatim
    between the slow (full re-forward) and fast (KV-cached) samplers so the
    two consume the RNG stream identically.

    lg: float np [R, slots, V] logits at rows block_start(a)+i.
    Mutates buf2 [R, S] and hist2 [R, slots, L] in place.
    """
    R_roll = lg.shape[0]
    for r in range(R_roll):
        lgr = lg[r]
        lgr[:, ban_from:] = -1e9
        if r == spec_idx:
            # greedy self-speculation rollout: the model's plain argmax
            # guess, full vocab, no shaping
            z_all = lgr.argmax(-1)
            for a in range(layout.slots):
                hist2[r, a, i] = z_all[a]
                buf2[r, layout.block_start(a) + 1 + i] = z_all[a]
            continue
        if disallow is not None:
            lgr[:, disallow] = -1e9
        if args.top_k_thought > 0:
            kth = np.partition(lgr, -args.top_k_thought, -1)[
                :, -args.top_k_thought][:, None]
            lgr[lgr < kth] = -1e9
        th.penalize_repeats(lgr, hist2[r], i, args.freq_pen)
        lgr = lgr / (args.temp if r == 0 else (args.temp2 or args.temp))
        lgr -= lgr.max(axis=-1, keepdims=True)
        pr = np.exp(lgr)
        pr /= pr.sum(-1, keepdims=True)
        for a in range(layout.slots):
            z_ai = rng.choice(pr.shape[-1], p=pr[a])
            hist2[r, a, i] = z_ai
            buf2[r, layout.block_start(a) + 1 + i] = z_ai


class FastGenerator:
    """Prefill-once + L-1 incremental decode steps. One instance per run;
    cache storage is reused across micros (every readable row is rewritten
    by each micro's prefill, so no cross-micro state survives)."""

    def __init__(self, trunk, lm_head, layout, n_layers, R, device,
                 autocast_ctx, sync_fn, topk_transfer=0):
        self.trunk, self.lm_head = trunk, lm_head
        self.layout, self.R, self.device = layout, R, device
        self.autocast_ctx, self.sync = autocast_ctx, sync_fn
        self.topk_transfer = topk_transfer   # device top-K headroom incl.
        self.pre = ScatterCache(n_layers, layout.S)
        self.dec = ScatterCache(n_layers, layout.S)
        self.pos_all = torch.arange(layout.S, dtype=torch.long, device=device)
        # decode row indices are layout-only constants: [L][slots]
        gr = th.thought_pred_rows(layout)
        self.rows_t = [torch.from_numpy(gr[:, i].copy()).to(device)
                       for i in range(layout.L)]
        self.gen_rows = gr

    def generate(self, buf2, hist2, anchors, k, pos_ids, rng, args, spec_idx,
                 disallow, ban_from, collect_logits=None):
        """Mirror of the slow sampling loop: fills buf2/hist2 in place.
        buf2: np [R, S]; pos_ids: [1, S] device tensor.
        collect_logits: optional list; each step's [R, slots, V] float np
        logits are appended (equivalence testing only)."""
        L, R = args.L, buf2.shape[0]
        # transformers silently drops past_key_values when gradient
        # checkpointing is enabled on a module in train() mode -- even under
        # no_grad (crashed the first fast resume, 2026-09-02). Generation is
        # no-grad and Qwen3 has no dropout, so eval() here is a pure
        # flag-flip with identical math; restored before returning.
        was_training = self.trunk.training
        if was_training:
            self.trunk.eval()
        try:
            self._generate(buf2, hist2, anchors, k, pos_ids, rng, args,
                           spec_idx, disallow, ban_from, collect_logits)
        finally:
            if was_training:
                self.trunk.train()

    def _generate(self, buf2, hist2, anchors, k, pos_ids, rng, args,
                  spec_idx, disallow, ban_from, collect_logits=None):
        L, R = args.L, buf2.shape[0]
        # collect_logits (test hook) forces the full-vocab path
        use_topk = self.topk_transfer > 0 and collect_logits is None
        if use_topk:
            assert args.top_k_thought > 0, "topk transfer needs top-k shaping"
            K = min(args.top_k_thought + 16, 4096)

        def emit(lg_dev):
            """device logits -> host arrays for the sampler"""
            if use_topk:
                v, ix = torch.topk(lg_dev.float(), K, dim=-1)
                return v.cpu().numpy(), ix.cpu().numpy()
            return lg_dev.float().cpu().numpy(), None

        def sample(host, i):
            if use_topk:
                sample_host_step_topk(host[0], host[1], i, buf2, hist2,
                                      self.layout, rng, args, spec_idx,
                                      disallow, ban_from)
            else:
                if collect_logits is not None:
                    collect_logits.append(host[0].copy())
                sample_host_step(host[0], i, buf2, hist2, self.layout, rng,
                                 args, spec_idx, disallow, ban_from)

        with torch.no_grad():
            # ---- prefill: batch-1 (rollouts are identical before sampling)
            m_pre = th.build_mask(self.layout, anchors, k, fill=1,
                                  no_echo=True) \
                .to(torch.bfloat16).to(self.device)
            bt0 = torch.from_numpy(buf2[0][None]).to(self.device)
            self.pre.set_positions(self.pos_all)
            with self.autocast_ctx():
                h0 = self.trunk(input_ids=bt0, position_ids=pos_ids,
                                attention_mask=m_pre,
                                past_key_values=self.pre, use_cache=True)[0]
                lg0 = self.lm_head(h0[:, self.rows_t[0]])    # [1, slots, V]
            self.dec.broadcast_from(self.pre, R)
            h_a, h_b = emit(lg0)
            h_a = np.repeat(np.ascontiguousarray(h_a), R, axis=0)
            if h_b is not None:
                h_b = np.repeat(np.ascontiguousarray(h_b), R, axis=0)
            self.sync()
            sample((h_a, h_b), 0)
            # ---- decode steps 1..L-1: feed the 8 just-sampled slot tokens
            for i in range(1, L):
                rows_i = self.gen_rows[:, i]                 # np [slots]
                m_i = torch.from_numpy(
                    decode_mask_rows(self.layout, anchors, i)) \
                    .to(torch.bfloat16).to(self.device)      # [1,1,slots,S]
                tok_i = torch.from_numpy(
                    np.ascontiguousarray(buf2[:, rows_i])).to(self.device)
                pos_i = pos_ids[:, rows_i]                   # [1, slots]
                self.dec.set_positions(self.rows_t[i])
                with self.autocast_ctx():
                    hi = self.trunk(input_ids=tok_i, position_ids=pos_i,
                                    attention_mask=m_i,
                                    past_key_values=self.dec,
                                    use_cache=True)[0]       # [R, slots, H]
                    lgi = self.lm_head(hi)                   # [R, slots, V]
                host = emit(lgi)
                self.sync()
                sample(host, i)


class BatchGenerator:
    """Stage 1b: one generation pass for ALL micro-batches of a step.

    The timing regression on run 11 (2026-09-02) showed each decode launch
    costs ~175ms of pure dispatch/transfer/host latency -- as much as the
    full re-forward it replaced -- so the fast path's win was eaten by 192
    launches per step. Batching the B micros' sequences into one decode
    batch of B*R rows cuts that to L-1 launches per STEP.

    Grad semantics are untouched: this class is no-grad only; pass-1/pass-2
    stay per-micro. Each micro keeps its own host RNG and its shaping loop
    runs per micro in micro order, so the drawn thought tokens are identical
    to the serial FastGenerator's, micro by micro.

    Batch layout: row b*R + r  =  (micro b, rollout r).
    """

    def __init__(self, trunk, lm_head, layout, n_layers, B, R, device,
                 autocast_ctx, sync_fn, topk_transfer=0):
        self.trunk, self.lm_head = trunk, lm_head
        self.layout, self.B, self.R = layout, B, R
        self.device = device
        self.autocast_ctx, self.sync = autocast_ctx, sync_fn
        self.topk_transfer = topk_transfer
        self.pre = ScatterCache(n_layers, layout.S)     # batch B
        self.dec = ScatterCache(n_layers, layout.S)     # batch B*R
        self.pos_all = torch.arange(layout.S, dtype=torch.long, device=device)
        gr = th.thought_pred_rows(layout)
        self.rows_t = [torch.from_numpy(gr[:, i].copy()).to(device)
                       for i in range(layout.L)]
        self.gen_rows = gr

    def _broadcast_grouped(self):
        """dec[b*R + r] <- pre[b] for every layer."""
        B, R = self.B, self.R
        for ld, ls in zip(self.dec.layers, self.pre.layers):
            if not ld.is_initialized:
                _, H, S, D = ls.keys.shape
                ld.keys = torch.zeros((B * R, H, S, D), dtype=ls.keys.dtype,
                                      device=ls.keys.device)
                ld.values = torch.zeros_like(ld.keys)
                ld.is_initialized = True
            src_k = ls.keys.unsqueeze(1).expand(B, R, *ls.keys.shape[1:]) \
                .reshape(B * R, *ls.keys.shape[1:])
            src_v = ls.values.unsqueeze(1).expand(B, R, *ls.values.shape[1:]) \
                .reshape(B * R, *ls.values.shape[1:])
            ld.keys.copy_(src_k)
            ld.values.copy_(src_v)

    def generate(self, micros, args, disallow, ban_from, spec_idx):
        """micros: list of B dicts with keys
             buf2   np [R, S]   (mutated in place)
             hist2  np [R, slots, L]  (mutated in place)
             anchors np [slots]
             pos_ids torch [1, S] (host or device; host is fine)
             rng    np.random.Generator (per-micro stream)
             kind   'sample' | 'oracle' | 'neg'  (non-sample micros ride
                    along for shape constancy; their buffers are not touched)
        spec_idx: rollout index of the greedy speculation rollout, or None.
        """
        B = self.B
        assert len(micros) == B
        was_training = self.trunk.training
        if was_training:
            self.trunk.eval()
        try:
            self._run(micros, args, disallow, ban_from, spec_idx)
        finally:
            if was_training:
                self.trunk.train()

    def _run(self, micros, args, disallow, ban_from, spec_idx):
        B, R, L, slots = self.B, self.R, args.L, self.layout.slots
        use_topk = self.topk_transfer > 0
        if use_topk:
            assert args.top_k_thought > 0, "topk transfer needs top-k shaping"
            K = min(args.top_k_thought + 16, 4096)

        def emit(lg_dev):
            if use_topk:
                v, ix = torch.topk(lg_dev.float(), K, dim=-1)
                return v.cpu().numpy(), ix.cpu().numpy()
            return lg_dev.float().cpu().numpy(), None

        def sample(m, sl_a, sl_b, i):
            if use_topk:
                sample_host_step_topk(sl_a, sl_b, i, m["buf2"], m["hist2"],
                                      self.layout, m["rng"], args, spec_idx,
                                      disallow, ban_from)
            else:
                sample_host_step(sl_a, i, m["buf2"], m["hist2"], self.layout,
                                 m["rng"], args, spec_idx, disallow, ban_from)

        with torch.no_grad():
            # ---- prefill: [B, S], one row per micro (rollouts identical)
            m_pre = torch.cat(
                [th.build_mask(self.layout, m["anchors"], args.k, fill=1,
                               no_echo=True) for m in micros], dim=0) \
                .to(torch.bfloat16).to(self.device)          # [B,1,S,S]
            bt0 = torch.from_numpy(
                np.stack([m["buf2"][0] for m in micros])).to(self.device)
            pos_b = torch.cat([m["pos_ids"] for m in micros], dim=0) \
                .to(self.device)                              # [B, S]
            self.pre.set_positions(self.pos_all)
            with self.autocast_ctx():
                h0 = self.trunk(input_ids=bt0, position_ids=pos_b,
                                attention_mask=m_pre,
                                past_key_values=self.pre, use_cache=True)[0]
                lg0 = self.lm_head(h0[:, self.rows_t[0]])     # [B, slots, V]
            self._broadcast_grouped()
            h_a, h_b = emit(lg0)                              # [B, slots, .]
            self.sync()
            pos_dec = pos_b.repeat_interleave(R, dim=0)       # [B*R, S]
            for b, m in enumerate(micros):
                if m["kind"] != "sample":
                    continue
                sl_a = np.repeat(h_a[b][None], R, axis=0)     # fresh copy
                sl_b = np.repeat(h_b[b][None], R, axis=0) \
                    if h_b is not None else None
                sample(m, sl_a, sl_b, 0)
            # ---- decode steps 1..L-1: [B*R, slots] tokens per launch
            for i in range(1, L):
                rows_i = self.gen_rows[:, i]
                m_i = torch.from_numpy(np.concatenate(
                    [decode_mask_rows(self.layout, m["anchors"], i)
                     for m in micros], axis=0)) \
                    .to(torch.bfloat16).to(self.device)       # [B,1,slots,S]
                m_i = m_i.repeat_interleave(R, dim=0)         # [B*R,1,sl,S]
                tok_i = torch.from_numpy(np.ascontiguousarray(
                    np.concatenate([m["buf2"][:, rows_i] for m in micros],
                                   axis=0))).to(self.device)  # [B*R, slots]
                self.dec.set_positions(self.rows_t[i])
                with self.autocast_ctx():
                    hi = self.trunk(input_ids=tok_i,
                                    position_ids=pos_dec[:, rows_i],
                                    attention_mask=m_i,
                                    past_key_values=self.dec,
                                    use_cache=True)[0]
                    lgi = self.lm_head(hi)                    # [B*R, sl, V]
                h_a, h_b = emit(lgi)
                self.sync()
                for b, m in enumerate(micros):
                    if m["kind"] != "sample":
                        continue
                    sl_a = h_a[b * R:(b + 1) * R]
                    sl_b = h_b[b * R:(b + 1) * R] if h_b is not None \
                        else None
                    sample(m, sl_a, sl_b, i)


def score_micros_fused(trunk, lm_head, micros, k, wk_t, autocast_ctx,
                       device, layout, L, want_lpi):
    """Score ALL micros' rollouts in one [B*R, S] no-grad forward.

    micros: as in BatchGenerator.generate, plus per-micro
      rows_np [slots, k] and tgt_np [slots*k] (host targets).
    Returns (lp2 [B, R, slots], lpi [B, R, slots] or None). Rows belonging
    to non-sample micros are computed on placeholder content and must be
    ignored by the caller (shape constancy over kind mix).
    """
    B = len(micros)
    R = micros[0]["buf2"].shape[0]
    slots = layout.slots
    gen_rows = th.thought_pred_rows(layout)
    with torch.no_grad():
        m4 = torch.cat(
            [th.build_mask(layout, m["anchors"], k, fill=layout.block)
             for m in micros], dim=0) \
            .to(torch.bfloat16).to(device).repeat_interleave(R, dim=0)
        bt = torch.from_numpy(np.concatenate(
            [m["buf2"] for m in micros], axis=0)).to(device)   # [B*R, S]
        pos = torch.cat([m["pos_ids"] for m in micros], dim=0) \
            .to(device).repeat_interleave(R, dim=0)
        with autocast_ctx():
            h2 = trunk(input_ids=bt, position_ids=pos, attention_mask=m4,
                       use_cache=False)[0]                     # [B*R, S, H]
        lp2 = np.zeros((B, R, slots), dtype=np.float64)
        lpi = np.zeros((B, R, slots), dtype=np.float64) if want_lpi else None
        tp_sc = torch.from_numpy(gen_rows.reshape(-1)).to(device)
        for b, m in enumerate(micros):
            rows_sc = torch.from_numpy(m["rows_np"].reshape(-1)).to(device)
            tgt_sc = torch.from_numpy(m["tgt_np"]).to(device)
            for r in range(R):          # device-op loop, no extra forwards
                row = h2[b * R + r]
                with autocast_ctx():
                    lg2r = lm_head(row[rows_sc])
                lp2[b, r] = (torch.log_softmax(lg2r.float(), -1)
                             .gather(1, tgt_sc[:, None])
                             .view(slots, k) * wk_t).sum(-1).cpu().numpy()
                if want_lpi:
                    zr = torch.from_numpy(np.stack(
                        [m["buf2"][r, layout.block_start(a) + 1:
                                   layout.block_start(a) + 1 + L]
                         for a in range(slots)]).reshape(-1)).to(device)
                    with autocast_ctx():
                        lgtr = lm_head(row[tp_sc])
                    lpi[b, r] = (torch.log_softmax(lgtr.float(), -1)
                                 .gather(1, zr[:, None])
                                 .view(slots, L).sum(-1).cpu().numpy())
    return lp2, lpi


def score_rollouts_fused(trunk, lm_head, buf2, pos_ids, m4, rows_sc, tgt_sc,
                         wk_t, autocast_ctx, tp_sc=None, zs=None):
    """One batched [R, S] no-grad forward scoring all rollouts at once
    (replaces the per-rollout loop; identical math, batched).

    Returns (lp2 [R, slots, k]-summed np [R, slots], lpi np [R, slots] or
    None): discounted window logp per rollout+slot, and (if tp_sc/zs given)
    each rollout's own-thought logp for the distill floor.
    """
    R = buf2.shape[0]
    slots = zs.shape[1] if zs is not None else None
    with torch.no_grad():
        bt = torch.from_numpy(buf2).to(pos_ids.device)
        with autocast_ctx():
            h2 = trunk(input_ids=bt, position_ids=pos_ids,
                       attention_mask=m4, use_cache=False)[0]  # [R, S, H]
        lp2_np, lpi_np = [], []
        for r in range(R):          # device-op loop, no extra forwards
            with autocast_ctx():
                lg2r = lm_head(h2[r, rows_sc])
            lp2_np.append(
                (torch.log_softmax(lg2r.float(), -1)
                 .gather(1, tgt_sc[:, None])
                 .view(-1, wk_t.shape[0]) * wk_t).sum(-1).cpu().numpy())
            if tp_sc is not None:
                with autocast_ctx():
                    lgtr = lm_head(h2[r, tp_sc])
                lpi_np.append(
                    torch.log_softmax(lgtr.float(), -1)
                    .gather(1, zs[r].reshape(-1)[:, None])
                    .view(slots, -1).sum(-1).cpu().numpy())
    return np.stack(lp2_np), (np.stack(lpi_np) if lpi_np else None)
