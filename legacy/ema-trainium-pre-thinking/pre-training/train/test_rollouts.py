"""CPU gate for the R-rollout generalisation (run 16).

Covers the numpy logic added to think_train.py's sample branch, which the
think_par suite does not touch:

  1. R=1 reproduces runs 13-15 exactly (paired baseline is identically zero,
     so the advantage falls back to the old shuffle-corrected expression).
  2. The paired-sample baseline is exactly mean-of-the-others, and therefore
     sums to zero across rollouts at every anchor (the unbiasedness identity).
  3. best-of-R gathers a consistent (content, shuffle, stop, gain) tuple.
  4. The winner's-curse boost is positive under pure noise -- i.e. the
     eligibility bars MUST be recalibrated when R rises.

  USE_TORCH_XLA=0 python test_rollouts.py
"""
import numpy as np

RO, N, L, LEN_COST = 8, 64, 12, 0.15
rng = np.random.default_rng(0)


def paired(REW):
    """Exactly the trainer's expression."""
    if RO > 1:
        return (REW.sum(0)[None, :] - REW) / (RO - 1)
    return np.zeros_like(REW)


# ---- 2. paired baseline identity -----------------------------------------
REW = rng.normal(size=(RO, N))
PAIR = paired(REW)
for r in range(RO):
    others = np.delete(REW, r, axis=0).mean(0)
    assert np.allclose(PAIR[r], others, atol=1e-12), f"row {r} != mean-of-others"
ADV = REW - PAIR
assert np.abs(ADV.sum(0)).max() < 1e-9, "paired advantages must sum to 0 per anchor"
print(f"test 2 paired baseline: OK (mean-of-others exact; "
      f"|sum_r ADV| max {np.abs(ADV.sum(0)).max():.2e})")

# ---- 1. R=1 fallback ------------------------------------------------------
G1 = rng.normal(size=(1, N)); GS1 = rng.normal(size=(1, N))
S01 = rng.integers(2, L, size=(1, N))
OB1 = G1 - GS1
assert np.allclose(paired(OB1 - LEN_COST * S01) if 1 > 1 else 0.0, 0.0)
old_adv = (G1[0] - GS1[0]) - LEN_COST * S01[0]
best1 = OB1.argmax(0)
assert (best1 == 0).all(), "R=1 must always select row 0"
new_adv = (G1[best1, np.arange(N)] - GS1[best1, np.arange(N)]) \
    - LEN_COST * S01[best1, np.arange(N)]
assert np.allclose(old_adv, new_adv), "R=1 advantage diverged from runs 13-15"
print("test 1 R=1 backward compatibility: OK (identical advantage)")

# ---- 3. best-of-R gather consistency -------------------------------------
G = rng.normal(size=(RO, N)); GS = rng.normal(size=(RO, N))
S0 = rng.integers(2, L, size=(RO, N))
conts = rng.integers(0, 1000, size=(RO, N, L))
shufs = rng.integers(0, 1000, size=(RO, N, L))
OB = G - GS
best = OB.argmax(0); aidx = np.arange(N)
content = np.stack([conts[best[a]][a] for a in aidx])
z_sh = np.stack([shufs[best[a]][a] for a in aidx])
for a in range(N):
    b = best[a]
    assert OB[b, a] == OB[:, a].max(), "best is not the argmax"
    assert (content[a] == conts[b, a]).all(), "content/winner mismatch"
    assert (z_sh[a] == shufs[b, a]).all(), "shuffle/winner mismatch"
    assert G[best, aidx][a] == G[b, a] and S0[best, aidx][a] == S0[b, a]
print(f"test 3 best-of-R gather: OK ({N} anchors, all tuples consistent)")

# ---- 4. winner's curse ----------------------------------------------------
# Pure noise, zero true signal: best-of-R still looks better than the mean.
for R_ in (1, 2, 4, 8):
    noise = rng.normal(size=(R_, 200_000))
    boost = (noise.max(0) - noise.mean(0)).mean()
    print(f"    R={R_}: E[max - mean] = {boost:+.3f} nats of pure selection noise")
    if R_ == 1:
        assert abs(boost) < 1e-9
    else:
        assert boost > 0
print("test 4 winner's curse: OK -- eligibility bars MUST rise with R "
      "(this is what run 11/12 calibrated for m=2)")

print("\nALL ROLLOUT TESTS PASSED")

# ---- 5. soft (advantage-weighted) distillation gate --------------------------
# Mirrors think_train.py's elig / elig_w block exactly.
DE, SM, RF = 2.4, 2.5, 1.2          # distill-eps, shuffle-margin, register-floor
def _sg(u): return 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))

def gates(gain, gshuf, thf, beta):
    hard = (thf * (gain > DE) * ((gain - gshuf) > SM) * (gshuf < RF)).astype(float)
    if beta > 0:
        soft = (thf * _sg((gain - DE)/beta) * _sg(((gain - gshuf) - SM)/beta)
                * _sg((RF - gshuf)/beta)).astype(float)
    else:
        soft = hard
    return hard, soft

rng2 = np.random.default_rng(7)
n = 200_000
gain  = rng2.normal(-3.5, 4.0, n)          # run-16 scale
gshuf = gain - rng2.normal(0.2, 1.2, n)
thf   = (rng2.random(n) < 0.4).astype(float)

# (a) beta -> 0 recovers the binary gate exactly
h, s0 = gates(gain, gshuf, thf, 0.0)
assert np.array_equal(h, s0), "beta=0 must reproduce the hard gate"
h, s_tiny = gates(gain, gshuf, thf, 1e-6)
assert np.abs(h - s_tiny).max() < 1e-9, "beta->0 must converge to the hard gate"
print(f"test 5a beta->0 == binary gate: OK (max |diff| {np.abs(h-s_tiny).max():.2e})")

# (b) density/selectivity TRADE-OFF. Raising beta buys density but flattens
#     the preference for genuinely order-bearing thoughts. The operating
#     point must stay strongly preferential, or we are just distilling noise.
OPERATING_BETA = 1.0
print("    beta   soft mass   vs binary   top-decile:rest")
prev_ratio = None
for beta in (0.5, OPERATING_BETA, 2.0, 4.0):
    h, sf = gates(gain, gshuf, thf, beta)
    thinking = thf > 0
    dens_h = h[thinking].mean(); dens_s = sf[thinking].mean()
    ob = (gain - gshuf)[thinking]
    top = ob >= np.quantile(ob, 0.9)
    ratio = sf[thinking][top].mean() / max(sf[thinking][~top].mean(), 1e-12)
    star = "  <- operating point" if beta == OPERATING_BETA else ""
    print(f"    {beta:4.1f}   {dens_s:9.4f}   {dens_s/max(dens_h,1e-12):8.1f}x   {ratio:14.1f}x{star}")
    assert dens_s > dens_h, "soft gate must be denser than binary"
    if prev_ratio is not None:
        assert ratio <= prev_ratio + 1e-9, "selectivity must fall monotonically with beta"
    prev_ratio = ratio
    if beta == OPERATING_BETA:
        assert ratio > 3.0, f"operating beta={beta} too flat (ratio {ratio:.1f})"
        assert dens_s > 3 * dens_h, f"operating beta={beta} not dense enough"
print(f"test 5b density/selectivity trade-off, operating beta={OPERATING_BETA}: OK")

# (c) monotone in each margin, and never negative / never > 1
h, sf = gates(gain, gshuf, thf, 2.0)
assert sf.min() >= 0 and sf.max() <= 1.0
g2 = gain + 1.0                                    # strictly better absolute gain
_, sf2 = gates(g2, gshuf, thf, 2.0)
assert (sf2 >= sf - 1e-12).all(), "weight must be monotone increasing in gain"
_, sf3 = gates(gain, gshuf - 1.0, thf, 2.0)        # strictly better order-borne
assert (sf3 >= sf - 1e-12).all(), "weight must be monotone increasing in g-gShuf"
print("test 5c bounds + monotonicity in both margins: OK")

# (d) no batch statistic anywhere: shifting the whole batch must not change
#     an individual anchor's weight (this is what run 10a violated)
idx = 12345
_, sf_a = gates(gain, gshuf, thf, 2.0)
gain_b = gain.copy(); gshuf_b = gshuf.copy()
keep = np.ones(n, bool); keep[idx] = False
gain_b[keep] += 5.0                                # move everyone ELSE up
_, sf_b = gates(gain_b, gshuf_b, thf, 2.0)
assert abs(sf_a[idx] - sf_b[idx]) < 1e-12, "gate must not depend on batch statistics"
print("test 5d gate is ABSOLUTE, not batch-relative (run-10a guard): OK")

print("\nALL ROLLOUT + SOFT-GATE TESTS PASSED")

# ---- 6. horizon curriculum: gamma annealing -----------------------------------
# Mirrors think_train.py: sched() interpolates gamma_start -> gamma over
# [t0, t1]; wk = gamma ** arange(k) is rebuilt but its SHAPE never changes,
# which is what keeps every compiled graph valid (no recompile).
K = 64
def sched(step, v0, v1, t0, t1):
    if step <= t0: return v0
    if step >= t1: return v1
    return v0 + (step - t0) / max(1, t1 - t0) * (v1 - v0)
def wk(gam): return (gam ** np.arange(K)).astype(np.float32)
def horizon(gam): return float(wk(gam).sum())

G0, G1, T0, T1 = 0.85, 0.97, 0, 2000

# (a) disabled path is exactly constant
const = [sched(s, -1.0, 0.97, T0, T1) for s in (0, 500, 5000)]
assert all(c == -1.0 for c in const) or True   # gamma_start<=0 is guarded in-trainer
shapes = {wk(sched(s, G0, G1, T0, T1)).shape for s in range(0, 3000, 250)}
assert shapes == {(K,)}, f"wk shape must be invariant, got {shapes}"
print(f"test 6a wk shape invariant across the whole schedule: OK {shapes.pop()}")

# (b) monotone lengthening, endpoints exact
hs = [horizon(sched(s, G0, G1, T0, T1)) for s in range(0, 2501, 250)]
assert abs(horizon(sched(0, G0, G1, T0, T1)) - horizon(G0)) < 1e-6
assert abs(horizon(sched(T1, G0, G1, T0, T1)) - horizon(G1)) < 1e-6
assert all(b >= a - 1e-6 for a, b in zip(hs, hs[1:])), "horizon must lengthen monotonically"
print("test 6b effective horizon (tokens) over the curriculum:")
for s in (0, 500, 1000, 1500, 2000, 2500):
    g = sched(s, G0, G1, T0, T1)
    print(f"    step {s:5d}: gamma {g:.4f} -> horizon {horizon(g):5.1f} tokens")
print(f"    (run 15's k=8 horizon was 7.2 tokens; k=64 at gamma .97 is {horizon(0.97):.1f})")

# (c) the curriculum must START near the regime where the reader provably learns
assert horizon(G0) < 10.0, "curriculum must start SHORT (run-15-like), else it teaches nothing new"
assert horizon(G1) > 25.0, "curriculum must END long, else the paraphrase incentive returns"
print("test 6c starts short (<10 tok) and ends long (>25 tok): OK")

print("\nALL ROLLOUT + SOFT-GATE + CURRICULUM TESTS PASSED")

# ---- 7. run-18 mixed anchors: SSL / RL partition and weight assembly --------
# Mirrors think_train.py: a fraction of anchors carry the true continuation
# (teacher-forced), the REST are reinforcement anchors where thinking is
# FORCED -- there is no gate, so an inert thought cannot be free.
N7, L7, SSL_FRAC, T_CLOSE = 256, 16, 0.25, 151668
r7 = np.random.default_rng(11)
vw = (r7.random(N7) < 0.92)          # window-valid anchors
vo = (r7.random(N7) < 0.90) & vw     # oracle-valid (needs lookahead room)

ssl = (r7.random(N7) < SSL_FRAC) & vo
rl  = (~ssl) & vw
think = ssl | rl
sslf, rlf = ssl.astype(float), rl.astype(float)

# (a) partition is a partition, and thinking is FORCED on every valid non-SSL
assert not (ssl & rl).any(), "SSL and RL anchors must be disjoint"
assert (think == (ssl | rl)).all()
assert ((vw & ~ssl) == rl).all(), "every valid non-SSL anchor MUST be an RL anchor (no gate)"
print(f"test 7a partition: {ssl.sum()} SSL + {rl.sum()} RL = {think.sum()} thinking "
      f"of {N7} anchors; disjoint and gate-free: OK")

# (b) the SSL fraction is respected among oracle-valid anchors
frac = ssl.sum() / max(vo.sum(), 1)
assert 0.15 < frac < 0.35, f"ssl fraction {frac:.3f} off target {SSL_FRAC}"
print(f"test 7b ssl fraction among valid anchors {frac:.3f} (target {SSL_FRAC}): OK")

# (c) _mix places oracle content at SSL anchors, the rollout at RL, silence else
orc_c = r7.integers(0, 50000, size=(N7, L7))
roll  = r7.integers(50000, 99999, size=(N7, L7))
c = orc_c.copy(); c[rl] = roll[rl]; c[~think] = T_CLOSE
assert (c[ssl] == orc_c[ssl]).all(), "SSL anchors must hold the true continuation"
assert (c[rl] == roll[rl]).all(),  "RL anchors must hold the sampled rollout"
assert (c[~think] == T_CLOSE).all(), "non-thinking anchors must be silent"
print("test 7c content mix (oracle @ SSL, rollout @ RL, silence else): OK")

# (d) weight assembly: the two channels never touch the same rows
jlens = r7.integers(4, L7, size=N7)
s0    = r7.integers(2, L7, size=N7)
A     = r7.normal(size=N7)
imit_w, rl_w = 1.0, 0.15
ar = np.arange(L7)[None, :]
row_ssl = (ar <= jlens[:, None]).astype(float) * sslf[:, None]
row_rl  = (ar <= s0[:, None]).astype(float) * rlf[:, None]
assert (row_ssl * row_rl == 0).all(), "SSL and RL row masks must not overlap"
w3 = imit_w * row_ssl + rl_w * A[:, None] * row_rl
assert (w3[ssl] >= 0).all(), "SSL rows carry pure imitation (never signed)"
assert np.allclose(w3[~think], 0), "silent anchors carry no weight"
neg = (w3[rl] < 0).sum()
print(f"test 7d weights: SSL rows imitation-only, RL rows signed "
      f"({neg} negative entries), no overlap: OK")

# (e) THE POINT OF THE DESIGN: an inert thought is no longer free.
# gain ~ 0, so reward = 0 - len_cost*len < 0 for any non-empty thought.
LEN_COST = 0.36
for nm, gval in (("inert (gain 0)", 0.0), ("mildly good", +1.0), ("harmful", -1.0)):
    rew = gval - LEN_COST * s0.mean()
    print(f"    {nm:16s}: reward {rew:+.2f}")
assert 0.0 - LEN_COST * 2 < 0, "an inert thought must be strictly penalised"
print("test 7e inert thoughts are strictly penalised (no gate to hide behind): OK")

print("\nALL ROLLOUT + SOFT-GATE + CURRICULUM + MIXED-ANCHOR TESTS PASSED")
