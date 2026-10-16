"""Thinking-in-pretraining: splice/mask/position builders, trigger, mixing head.

Implements Stage A of THINKING_DESIGN.md: entropy-triggered, fixed-length
thoughts, single-copy contaminated stream, m=1, myopic credit.

Deliberately torch_xla-free so the whole module unit-tests on CPU: everything
here is plain torch + numpy. Device concerns live in think_train.py.

Layout convention (fixed shape): the spliced buffer physically appends all
thought blocks AFTER the real tokens --

    [ t_0 .. t_{T-1} | block_0 | block_1 | ... | block_{S-1} ]

with block_a = <T>, z_0..z_{L-1}, </T>  (L+2 tokens). Physical order is
irrelevant to the transformer: semantics are carried entirely by the float
position_ids (thoughts sit fractionally inside (n_a, n_a+1)) and the explicit
4D additive attention mask. This keeps every index computable in closed form.

Mask semantics (single-copy, contamination accepted -- design §1.4):
  * real -> real: causal, exactly vanilla.
  * predictor rows of window a (real positions n_a .. n_a+k-1, i.e. the
    positions whose OUTPUTS predict t_{n_a+1}..t_{n_a+k}) additionally attend
    block a (filled part only).
  * thought tokens of block a attend real prefix <= n_a plus earlier tokens of
    their own block (and themselves). Never other blocks, never real > n_a.
  * nothing else attends any block. Dummy blocks attend only themselves.
Indirect contamination (suffix real tokens attending window rows whose hiddens
saw the thought) is intended and unmasked.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9          # additive mask "off" (finite: safe in bf16, kills softmax)

T_OPEN = 151667     # <think>   -- reused as <T>  (verified reserved in Qwen3)
T_CLOSE = 151668    # </think>  -- reused as </T>
EOD = 151643        # <|endoftext|> document separator in our shards


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
class SpliceLayout:
    """All index arithmetic for the fixed-shape spliced buffer."""

    def __init__(self, T, slots, L):
        self.T, self.slots, self.L = T, slots, L
        self.block = L + 2                      # <T> + L content + </T>
        self.S = T + slots * self.block         # spliced length
        self.eps = 1.0 / (L + 3)                # design §4: strict (n, n+1)

    def block_start(self, a):
        return self.T + a * self.block

    def content_slice(self, a):
        s = self.block_start(a)
        return s + 1, s + 1 + self.L            # z_0 .. z_{L-1}


def build_position_ids(layout, anchors):
    """float32 [1, S]; real tokens get integer positions, block a token i gets
    anchors[a] + (i+1)*eps. Dummy blocks (anchor<0) get harmless positions."""
    pos = np.zeros(layout.S, dtype=np.float32)
    pos[:layout.T] = np.arange(layout.T, dtype=np.float32)
    for a in range(layout.slots):
        n = anchors[a] if anchors[a] >= 0 else 0
        s = layout.block_start(a)
        pos[s:s + layout.block] = n + (np.arange(layout.block) + 1) * layout.eps
    return torch.from_numpy(pos[None, :])


def build_position_ids_int(layout, anchors):
    """Integer positions continuing from the anchor (Task 3 diag #2): block a
    token i sits at n_a + 1 + i, overlapping the real suffix positions (the
    mask keeps them separate; RoPE distance to the window shrinks)."""
    pos = np.zeros(layout.S, dtype=np.float32)
    pos[:layout.T] = np.arange(layout.T, dtype=np.float32)
    for a in range(layout.slots):
        n = anchors[a] if anchors[a] >= 0 else 0
        s = layout.block_start(a)
        pos[s:s + layout.block] = n + 1.0 + np.arange(layout.block)
    return torch.from_numpy(pos[None, :])


def build_mask(layout, anchors, k, fill, no_echo=False):
    """Additive fp32 mask [1, 1, S, S]. `fill` = number of visible tokens per
    block (1 = only <T>, block = fully closed); during sampling it grows.
    anchors[a] < 0 marks a dummy slot: isolated (self-attention only).

    no_echo=True drops the window->block attention (predictor rows never see
    the block). Used by the KV-cached sampler (think_gen): thought rows attend
    the anchor row n_a, whose hidden under the training mask incorporates the
    filling block -- an echo of the partial thought through the real stream.
    No incremental cache can reproduce that recursion (every cached block
    row's deep-layer k/v would depend on the current fill), so fast sampling
    conditions on the clean prefix instead. Scoring and pass-2 keep the echo.
    """
    T, S = layout.T, layout.S
    m = np.full((S, S), NEG, dtype=np.float32)

    # real -> real causal (vanilla)
    tri = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
    m[:T, :T] = tri

    for a in range(layout.slots):
        s = layout.block_start(a)
        e = s + layout.block
        n = anchors[a]
        if n < 0:
            # dummy: attends only itself (keeps every softmax row finite)
            for i in range(s, e):
                m[i, i] = 0.0
            continue
        f = s + min(fill, layout.block)
        # thought rows: real prefix <= n, plus own block causally
        for i in range(s, e):
            m[i, :n + 1] = 0.0
            m[i, s:min(i + 1, f)] = 0.0
            m[i, i] = 0.0
        # predictor rows of the window attend the filled part of the block
        if not no_echo:
            m[n:n + k, s:f] = 0.0
    return torch.from_numpy(m[None, None])


def build_spliced_ids(layout, ids_row, anchors, thoughts, pad_id):
    """int64 [1, S]. thoughts: int array [slots, L] (garbage rows ok for dummy
    slots -- they are fully masked and weighted out)."""
    buf = np.full(layout.S, pad_id, dtype=np.int64)
    buf[:layout.T] = ids_row
    for a in range(layout.slots):
        s = layout.block_start(a)
        buf[s] = T_OPEN
        buf[s + 1:s + 1 + layout.L] = thoughts[a]
        buf[s + 1 + layout.L] = T_CLOSE
    return torch.from_numpy(buf[None, :])


# --------------------------------------------------------------------------
# trigger (host-side, design §1.1)
# --------------------------------------------------------------------------
class EntropyTrigger:
    """Running-quantile + absolute-floor entropy trigger. Anchors are the
    top-H positions satisfying gap/EOD/end constraints (greedy by H), not
    first-come -- early in training everything is high-entropy and first-come
    would pile all thoughts at the sequence start."""

    def __init__(self, q=0.0075, tau_abs=2.0, ema=0.05):
        self.q, self.tau_abs, self.ema = q, tau_abs, ema
        self.tau_q = None

    def __call__(self, H, ids_row, k, slots, min_gap, min_pos=0):
        T = len(H)
        batch_q = float(np.quantile(H, 1.0 - self.q))
        self.tau_q = batch_q if self.tau_q is None else \
            (1 - self.ema) * self.tau_q + self.ema * batch_q
        tau = max(self.tau_q, self.tau_abs)

        # candidate n: predicts window t_{n+1}..t_{n+k}; needs room, no EOD
        # inside [n, n+k] (window crossing a document boundary is noise).
        # min_pos: no anchors without real context -- a near-blind pass-1
        # baseline makes the counterfactual gain meaningless (run 10b found
        # its first "eligible" thoughts piled at positions 1-5).
        ok = np.ones(T, dtype=bool)
        ok[:max(1, min_pos)] = False
        ok[T - k - 1:] = False
        eod = np.where(ids_row == EOD)[0]
        for e in eod:
            ok[max(0, e - k):e + 1] = False
            if min_pos > 0:               # fresh document = no context either
                ok[e:e + min_pos] = False
        cand = np.where((H > tau) & ok)[0]

        chosen = []
        for n in cand[np.argsort(-H[cand])]:        # greedy by entropy
            if all(abs(n - c) >= min_gap for c in chosen):
                chosen.append(int(n))
                if len(chosen) == slots:
                    break
        chosen.sort()
        anchors = np.full(slots, -1, dtype=np.int64)
        anchors[:len(chosen)] = chosen
        return anchors, tau


# --------------------------------------------------------------------------
# mixing head (design §2)
# --------------------------------------------------------------------------
class MixingHead(nn.Module):
    def __init__(self, hidden, mix_hidden=64, bias_init=-2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, mix_hidden), nn.Tanh(),
            nn.Linear(mix_hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, bias_init)

    def forward(self, h_base, h_think):
        # base hiddens detached: the gate must not push gradients that degrade
        # the clean pass (design §2)
        return torch.sigmoid(self.net(
            torch.cat([h_base.detach(), h_think], dim=-1)))


# --------------------------------------------------------------------------
# gathered losses over the spliced pass
# --------------------------------------------------------------------------
def window_indices(layout, anchors, k):
    """Predictor rows and their targets for every slot: int64 [slots, k] each,
    plus validity [slots]. Predictor row n_a+j-1 predicts target t_{n_a+j}."""
    slots = layout.slots
    rows = np.zeros((slots, k), dtype=np.int64)
    tgts = np.zeros((slots, k), dtype=np.int64)
    valid = np.zeros(slots, dtype=np.float32)
    for a in range(slots):
        n = anchors[a]
        if n < 0:
            rows[a] = 1; tgts[a] = 2           # in-range dummies, weight 0
            continue
        rows[a] = n + np.arange(k)
        tgts[a] = n + 1 + np.arange(k)
        valid[a] = 1.0
    return rows, tgts, valid


def thought_pred_rows(layout):
    """Rows whose outputs predict z_0..z_{L-1}: [slots, L] -- the <T> position
    predicts z_0, position of z_i predicts z_{i+1}. </T> is forced, no logπ."""
    return np.stack([
        layout.block_start(a) + np.arange(layout.L) for a in range(layout.slots)
    ]).astype(np.int64)


def penalize_repeats(lg, hist, i, freq_pen=1.5):
    """Anti-collapse penalties on host-side sampling logits, in place.

    87/100 autonomous thoughts from the first Stage-A run degenerated into
    repetition loops (" of of of ..."), and those loops carried nearly all of
    the negative reward (-11.4 vs -0.8 nats for non-collapsed thoughts, see
    runs/THINK_INFER_REPORT.md). REINFORCE alone cannot escape the attractor:
    when every slot collapses the same way, the whitened reward is ~0. So the
    constraint is mechanical, at sampling time, on the host logits -- which
    also means it changes NO device graph.

    lg:   [slots, V] float logits (pre-temperature)
    hist: [slots, L] tokens sampled so far in each thought (cols < i valid)
    - the immediately previous token is hard-banned (kills period-1 loops)
    - every token already in the thought pays freq_pen nats per occurrence
      (kills period-k loops without forbidding natural reuse outright)
    """
    if i == 0:
        return lg
    for a in range(lg.shape[0]):
        prev = hist[a, :i]
        uniq, cnt = np.unique(prev, return_counts=True)
        lg[a, uniq] -= freq_pen * cnt
        lg[a, prev[i - 1]] = -1e9
    return lg
