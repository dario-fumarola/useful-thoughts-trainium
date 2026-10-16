"""Synthetic addition sequences for E1 (NEXT-STEPS.md sec. 4): the positive control.

One problem per 64-token block, aligned to the fixed anchor stride so the
compiled geometry is unchanged:

    ... \\n \\n Q : _ a1..aA _+ _ b1..bB _= _ | s1 .. sD \\n \\n ...
                                        ^ anchor n (the ' ' after ' =')
                                          window starts at s1 = x[n+1]

The answer is written most-significant digit first, which one forward pass
handles poorly (carries propagate from the right).  The supervised thought
(scratchpad) is the answer REVERSED, least-significant first, then </T>:
locally computable digit by digit (Lee et al. 2023), and <= 13 digits, so
it fits L = 16.

Token ids are the Qwen3 tokenizer's own encoding of the text (verified
2026-09-26: 'Q: 1 + 2 = 3\\n' -> [48, 25, 220, 16, 488, 220, 17, 284, 220,
18, 198]; every digit is its own token).  Gaps between problems are filled
with the newline token (198) repeated -- non-canonical (the tokenizer would
merge '\\n\\n' into 271) but harmless: the model trains on ids.
"""
import numpy as np

DIG = np.arange(15, 25, dtype=np.int64)      # '0'..'9'
Q, COLON, SP, PLUS, EQ, NL = 48, 25, 220, 488, 284, 198
T_CLOSE = 151668


def _digits(v, width=0):
    return [int(c) for c in (str(v).zfill(width) if width else str(v))]


def problem_tokens(a, b, width=0):
    """'Q: a + b = ' as ids; the last id is the ' ' the anchor sits on.
    width > 0: operands written with exactly `width` digits (leading zeros)."""
    da = _digits(a, width)
    db = _digits(b, width)
    return ([Q, COLON, SP] + [int(DIG[d]) for d in da] + [PLUS, SP]
            + [int(DIG[d]) for d in db] + [EQ, SP])


def answer_tokens(a, b, width=0):
    """width > 0: the sum is zero-padded to width+1 digits, so every answer has
    the same length and reading the reversed scratchpad is a FIXED permutation."""
    return [int(DIG[d]) for d in _digits(a + b, width + 1 if width else 0)]


def draw_operand(rng, dmin, dmax):
    nd = int(rng.integers(dmin, dmax + 1))
    first = int(rng.integers(1, 10))
    rest = rng.integers(0, 10, size=nd - 1)
    return int(str(first) + "".join(str(int(d)) for d in rest))


def make_seq(rng, T=1024, first=64, stride=64, n_anchor=14, L=16,
             dmin=6, dmax=12, width=0):
    """One synthetic training/eval sequence.

    Returns dict(x=int64[T+1], scr=int64[n_anchor, L] (reversed answer then
    </T> padding), slen=int64[n_anchor] (#digits), ans=list of answer-id lists,
    ops=list of (a, b)).  Problems sit at anchors first + stride*m for
    m = 0..n_anchor (one extra, unanchored, so the last anchor's window holds
    a following problem exactly like every other window).
    """
    x = np.full(T + 1, NL, dtype=np.int64)
    scr = np.full((n_anchor, L), T_CLOSE, dtype=np.int64)
    slen = np.zeros(n_anchor, dtype=np.int64)
    ans, ops = [], []
    prev_end = 0                      # first index free for the next problem
    for m in range(n_anchor + 1):
        n = first + stride * m
        if width:
            a, b = int(rng.integers(0, 10 ** width)), int(rng.integers(0, 10 ** width))
        else:
            a, b = draw_operand(rng, dmin, dmax), draw_operand(rng, dmin, dmax)
        P = problem_tokens(a, b, width)
        A = answer_tokens(a, b, width)
        s0 = n - len(P) + 1
        assert s0 >= prev_end + 1, (s0, prev_end)   # >= one newline between
        assert n + len(A) + 1 <= T, (n, len(A))
        x[s0:n + 1] = P
        x[n + 1:n + 1 + len(A)] = A
        x[n + 1 + len(A)] = NL
        prev_end = n + 2 + len(A)
        if m < n_anchor:
            assert len(A) + 1 <= L
            scr[m, :len(A)] = A[::-1]
            slen[m] = len(A)
            ans.append(A)
            ops.append((a, b))
    return dict(x=x, scr=scr, slen=slen, ans=ans, ops=ops)


# ---------------------------------------------------------------- pointer chase
GT, COMMA = 29, 11                            # '>' and ','


def _two(v):
    return [int(DIG[v // 10]), int(DIG[v % 10])]


def chase_block(rng, hops=5, n_distract=1):
    """One pointer-chasing problem (run 45).  A chain s0 -> s1 -> ... -> s_hops of
    distinct 2-digit values is given as shuffled 'ab>cd, ' pairs (plus distractor
    pairs whose keys are not on the chain), then 'Q s0: '.  The answer is s_hops.
    The scratchpad is 's1 s2 ... s_hops' in FORWARD order: each step is a
    content lookup (find 'cur>' and copy what follows) and the answer is the
    last value, so both writing and reading are copy-shaped -- unlike the
    reversed addition scratchpad, which the reader never learned to read."""
    vals = rng.choice(np.arange(10, 100), size=hops + 1 + n_distract, replace=False)
    chain = [int(v) for v in vals[:hops + 1]]
    pairs = [(chain[i], chain[i + 1]) for i in range(hops)]
    others = [v for v in range(10, 100) if v not in chain]
    for d in range(n_distract):
        pairs.append((int(vals[hops + 1 + d]), int(rng.choice(others))))
    order = rng.permutation(len(pairs))
    P = []
    for ii, j in enumerate(order):
        k, v = pairs[j]
        # the last pair ends in a bare newline: the tokenizer merges ', Q' into
        # ' Q' and ',\n' into one token, but a digit and '\n' stay two tokens
        P += _two(k) + [GT] + _two(v) + ([NL] if ii == len(order) - 1 else [COMMA, SP])
    P += [Q, SP] + _two(chain[0]) + [COLON, SP]
    A = _two(chain[-1])
    S = []
    for i, v in enumerate(chain[1:]):
        S += ([SP] if i else []) + _two(v)
    return P, A, S, chain


def make_chase_seq(rng, T=1024, first=64, stride=64, n_anchor=14, L=16, hops=5,
                   n_distract=1, hops_max=0):
    """hops_max > 0 draws hops ~ U{hops..hops_max} PER PROBLEM (run 46's ladder);
    hops_max == 0 keeps run 45's fixed 5-hop behaviour exactly.  The per-anchor hop
    count is returned as `hops` so a probe can bin its results by depth."""
    x = np.full(T + 1, NL, dtype=np.int64)
    scr = np.full((n_anchor, L), T_CLOSE, dtype=np.int64)
    slen = np.zeros(n_anchor, dtype=np.int64)
    ans, ops, nhops = [], [], []
    prev_end = 0
    for m in range(n_anchor + 1):
        n = first + stride * m
        h = int(rng.integers(hops, hops_max + 1)) if hops_max > hops else hops
        P, A, S, chain = chase_block(rng, h, n_distract)
        s0 = n - len(P) + 1
        assert s0 >= prev_end + 1, (s0, prev_end)
        x[s0:n + 1] = P
        x[n + 1:n + 1 + len(A)] = A
        x[n + 1 + len(A)] = NL
        prev_end = n + 2 + len(A)
        if m < n_anchor:
            assert len(S) + 1 <= L
            scr[m, :len(S)] = S
            slen[m] = len(S)
            ans.append(A)
            ops.append(tuple(chain))
            nhops.append(h)
    return dict(x=x, scr=scr, slen=slen, ans=ans, ops=ops,
                hops=np.asarray(nhops, dtype=np.int64))


def eval_set(n_seq, seed=99991, task="add", **kw):
    if task == "chase":
        return [make_chase_seq(np.random.default_rng([seed, i]), **kw)
                for i in range(n_seq)]
    """Held-out problems: a seed stream disjoint from training's
    ([seed, rank, step, micro, 555]); with 6-12 digit operands the chance of a
    training collision is ~1e-5 per problem (negligible, not deduplicated)."""
    return [make_seq(np.random.default_rng([seed, i]), **kw)
            for i in range(n_seq)]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    s = make_seq(np.random.default_rng(0))
    x = s["x"]
    for m in range(3):
        n = 64 + 64 * m
        a, b = s["ops"][m]
        text = f"Q: {a} + {b} = {a + b}\n"
        want = tok(text)["input_ids"]
        got = x[n - len(problem_tokens(a, b)) + 1:n + 2 + len(s["ans"][m])].tolist()
        assert got == want, (got, want)
        assert x[n] == SP and x[n + 1] == s["ans"][m][0]
        assert tok.decode(s["scr"][m, :s["slen"][m]].tolist()) == str(a + b)[::-1]
        print(repr(text), "scratch", tok.decode(s["scr"][m, :s["slen"][m]].tolist()))
    import collections
    ev = eval_set(20)
    print("digits per answer:", collections.Counter(int(l) for e in ev for l in e["slen"]))
    print("OK: tokenization matches the tokenizer, anchors on ' ', answers at n+1")
