# Thinking-in-Pretraining on Trainium — Design Decisions (v2)

Supersedes the mechanisms of `RL_PRE_TRAINING.md` where the two disagree; that
file remains the reference for everything not re-specified here (module layout,
logging philosophy, controls). This document records the resolved design after
review, with the rationale for every choice. No code exists yet; this is the
implementation contract.

**Rulings received:**
1. Target model is **Qwen3-0.6B** (our existing from-scratch pipeline).
2. `<T>` emission is **mechanically bound to next-token predictive entropy** —
   not a learned vocab action. Closure is by **length cap** (~64, configurable)
   or by a **gain probe**: close when the thought's improvement on the next
   real token's log-likelihood exceeds a threshold.
3. **KV contamination is accepted and desired** — thoughts may influence the
   stream indirectly past the reward window; no branch duplication.
4. The whitened-reward staging gate (impossible as specified) and the
   fractional-position off-by-one: fixed at my discretion.
5. Remaining tensions from the review: resolved at my discretion, below.

---

## 1. The mechanism (end-to-end)

### 1.1 Trigger (when to think)

At position `n`, thinking triggers when the **pointwise** predictive entropy
`H_n` of the base pass exceeds a threshold:

```
trigger(n)  =  H_n > max(tau_q, tau_abs)
               AND n - last_anchor >= min_gap
               AND window [n+1, n+k] crosses no EOD and no sequence end
               AND anchors_in_row < max_thoughts_per_seq
```

- `tau_q`: running quantile of `H` maintained online so the *candidate* rate
  stays ≈ `q_target` as entropy falls over training.
- `tau_abs` (absolute floor, ~2.0 nats): prevents the quantile from forcing
  thoughts onto genuinely easy text late in training. The quantile keeps the
  rate stable early; the floor lets the rate fall when the model is confident.
- Pointwise `H_n`, not the sequence-EMA of the original spec — matches the
  ruling ("uncertainty of the next token") and is simpler.
- `min_gap >= k` keeps reward windows disjoint.
- Config arithmetic is now consistent by construction: `q_target` is chosen to
  match the slot budget (`max_thoughts_per_seq` fixed-shape slots, §7), instead
  of an independent think-rate that the slot cap silently violates.

The same trigger runs at inference (entropy of the model's own predictive
distribution is available there too). At inference the entropy is computed on
the contaminated stream rather than a clean pass — accepted mismatch, logged.

### 1.2 Generation (sampling the thought)

At a triggered anchor `n`: force-insert `<T>`, autoregressively sample content
tokens at temperature 1.1→0.9, under `no_grad`. Sampling uses **the identical
fractional positions and attention mask as the training pass** — anything else
makes the recomputed `logπ` off-policy and silently biases REINFORCE.

Anchors within one row are sampled **left to right**: anchor #2's prefix
includes anchor #1's realized thought and its contaminated window, exactly as
the training pass and inference will see it. (Sampling all thoughts from clean
prefixes was considered and rejected: off-policy for every anchor after the
first.)

### 1.3 Closure (when to stop thinking) — the new part

The thought at anchor `n` closes at the first of:

1. **Gain probe** (every `probe_every` = 8 sampled tokens, after a minimum
   length `L_min` = 8): append a provisional `</T>`, one incremental forward,
   read `logp(t_{n+1})` at the `</T>` position, and compare with the base pass:

   ```
   gain = logp_think(t_{n+1} | prefix, thought, </T>) - logp_base(t_{n+1})
   close if gain > delta        (delta = 0.5 nats, config)
   ```

   If not closed, the provisional `</T>` is discarded (its KV entry must not
   persist) and generation continues.
2. **Hard cap** `L_max` = 64 tokens (config, may be raised).

The closure decision is **mechanical, not learned** — it contributes no term to
`logπ` and receives no gradient. REINFORCE trains *content only*; content
influences length only indirectly (better content crosses `delta` sooner).

**Honesty note — the gain probe reads the true next token.** This is fine as a
training-time device (pretraining is teacher-forced anyway), but it leaks label
information into the *length* of the thought. Two consequences, both handled in
§9: (a) headline eval numbers must come from an inference-legal closure rule;
(b) at inference, where `t_{n+1}` is unknown, closure uses the **entropy-drop
proxy** — close when the post-`</T>` predictive entropy falls below the
pre-thought entropy by a margin — whose correlation with the true gain we log
throughout training to validate it.

### 1.4 Ephemerality under accepted contamination

- **Direct** attention to thought tokens is still masked for real tokens beyond
  the window `[n+1, n+k]`, and thought KVs are evicted after `k` tokens at
  inference. This keeps the *direct* influence local and the KV cache bounded.
- **Indirect** influence — window tokens attend to the thought, and everything
  downstream attends to those window tokens' contaminated KVs — is allowed and
  intended (per ruling). Single-copy splicing; no branch duplication.
- The strict parity property of the original spec (`pass-2 non-window logits ≡
  pass-1`) is therefore **abandoned by design** and replaced by the weaker,
  still-testable set in §10: exact parity with zero anchors, exact parity for
  all positions before the first `<T>`, and direct mask-structure unit checks.

---

## 2. Two-pass training step

```
PASS 1 (clean, grad ON):   vanilla NTP forward on t_1..t_T
  -> full NTP loss on all NON-window real tokens (chunked CE, as today)
  -> per-position entropies H (computed inside the CE chunk sweep, free)
  -> logp_base at window positions and probe positions (for reward/gain)
SAMPLE (no grad):          triggers from H; left-to-right per row; closure per §1.3
PASS 2 (spliced, grad ON): single contaminated stream with thoughts inserted
  -> mixed NTP on window tokens:  logits_mix = w·logits_think + (1−w)·logits_base
  -> logπ at thought positions (REINFORCE, reward detached)
  -> nothing read at any other position
total = ntp_pass1_nonwindow + ntp_mixed_windows
      + alpha(step) · reinforce + entropy_bonus
```

Loss allocation is unchanged from the original spec and each real token is
counted **exactly once** (unit-tested): non-window from pass 1, window from the
mix. Pass 2's suffix positions receive no loss — under contamination their
logits are no longer replicas of pass 1, but training them would double-count
tokens and teach dependence on thought side-effects beyond the window; the
window is where thought value is measured and mixed in.

**Reward** (unchanged in substance):
`R_raw = Σ_{j=1..k} [logp_think(t_{n+j}) − logp_base(t_{n+j})] − λ_len·L`,
pure pass-2 window log-probs (never mixed — a lazy gate must not be able to
zero the policy gradient), whitened across the batch's thoughts, clipped ±3.

**Credit assignment under contamination (my call):** with accepted
contamination, thought `a` also perturbs windows after its own (through the
stream). Default is **myopic pairing** — `R_a · logπ(z_a)` — accepting a small
bias in exchange for much lower variance; the unbiased reward-to-go variant
(`Σ_{a'≥a} R_{a'} · logπ(z_a)`) stays behind a config flag as an ablation.
Windows are short and disjoint; the indirect term is second-order.

**Rollouts (my call): `m = 1` by default.** Under single-copy contamination, m
rollouts of one anchor need m copies of the *entire spliced row* (the window
hiddens differ per rollout) — pass-2 cost × m, no longer cheap branch rows. The
counterfactual baseline already lives inside `R_raw` (the spec explicitly
allows m=1), so RLOO is not needed for a sane estimator. `m = 2` is the first
knob to turn if gradient variance is bad, at 2× pass-2 cost.

**Mixing head:** per spec — 2-layer MLP (hidden 64) on `[h_base; h_think]`,
sigmoid gate, final bias −2 (w≈0.12), trained by the mixed NTP loss only,
`h_base` input detached. With m=1 the "best-of-m vs mean" question disappears.

---

## 3. Special tokens (verified against our stack)

- **Reuse Qwen3's reserved `<think>` (151667) and `</think>` (151668)** as
  `<T>`/`</T>`. No tokenizer surgery, no embedding resize: the rows exist in
  the 151936-row table and are freshly initialized like everything else in our
  from-scratch training. The original spec's resize-and-mean-init is for
  adapting pretrained models and does not apply.
- **Guard (verified necessary):** raw text containing the literal string
  `<think>` tokenizes to id 151667. Our 3B corpus contains exactly **3** such
  tokens (2× `<think>`, 1× `</think>`, all in fineweb-edu). The dataloader
  remaps ids 151667/151668 → a neutral token id at batch-assembly time and
  asserts the batch contains neither. No re-tokenization needed.

## 4. Fractional positions (fixed)

`eps = 1/(L_max + 3)`. The thought block at anchor `n` is
`<T>, z_1..z_L, </T>` = L+2 tokens; token `i` (0-indexed over the block) sits at

```
pos = n + (i + 1) · eps        max: n + (L_max + 2)/(L_max + 3)  <  n + 1
```

strictly inside `(n, n+1)` for every length up to the cap — the original
`1/(L_max+2)` denominators put a max-length `</T>` exactly on `n+1`, and §0/§3.2
of the spec disagreed by one index; both fixed. Real tokens keep vanilla
integer positions. Verified: our HF Qwen3 RoPE computes angles from
`position_ids.float()` by matmul — no internal `arange`, no integer cast — and
eager attention consumes an additive 4D mask directly. **No model surgery**; a
wrapper passes float `position_ids` and the block mask.

## 5. Staging (replaces Phases 1/2)

The learned `<T>`-emission of the spec's Phase 2 is **gone** (with it, the
P(`<T>`)-vs-NTP softmax tug-of-war and the 1/152k cold-start — the two reasons
it was likely dead on arrival). The entropy trigger is the permanent emission
mechanism. What remains staged is closure complexity:

- **Phase 0 — vanilla warmup.** Our existing loop, T=1024 (§7). Fresh short
  warmup for the experiment arms (controlled comparison); the full 3B
  checkpoint can serve as a stronger starting point in later cycles.
- **Stage A — fixed-length thoughts** (`L = 16`, forced `</T>`, no probe).
  Validates trigger, splice, mask, mixing, REINFORCE with the least machinery.
  `λ_len = 0`.
- **Stage B — adaptive closure** (gain probe + `L_max = 64`). `λ_len = 0.005`.

**Stage gate A→B (fixed per ruling):** EMA (1k steps) of **unwhitened**
`mean R_raw > 0` AND win-rate `P(R_raw > 0) > 0.5` on gated positions. (The
spec gated on the whitened mean, which is 0 by construction — same fix applied
to the M3 milestone criterion.)

`alpha` (REINFORCE weight): linear 0 → 0.5 over the first 10% of Stage A.
Entropy bonus β = 1e-3 annealed to 0. The embedding stop-grad flag of the spec
is **dropped** (my call): with tied embeddings it is leaky anyway (REINFORCE
re-enters the shared table through the lm_head), and α-warmup + grad-clip 1.0
serve the same protective purpose. We log the embedding-row norms of
`<T>`/`</T>` and the top thought tokens; the flag returns only if those blow up.

## 6. Other resolved tensions (from the review, in order)

| # | Tension | Resolution |
|---|---|---|
| 1 | Parity broken by contamination | Contamination accepted by ruling; parity redefined (§1.4, tests §10) |
| 2 | Whitened-mean stage gate ≡ 0 | Gate on unwhitened EMA + win-rate (§5) |
| 3 | eps off-by-one; §0 vs §3.2 | `1/(L_max+3)`, one formula (§4) |
| 4 | think-rate vs slot-cap arithmetic | `q_target` derived from slot budget (§1.1, §7) |
| 5 | P(`<T>`) vs NTP; cold start | Mechanism removed; mechanical trigger (§5) |
| 6 | Whitening breaks gate-reward zero | Moot — no learned gate action remains |
| 7 | Tied embeddings vs stop-grad | Flag dropped; monitored instead (§5) |
| 8a | Windows off sequence end / EOD | Excluded at trigger time (§1.1) |
| 8b | Sampler off-policy positions | Sampler mandated to mirror pass-2 exactly (§1.2) |
| 8c | `test_reward` garbage ≈ 0 flaky | Assert `R(planted) > R(garbage)` instead (§10) |
| 8d | Best-of-m selection bias | Moot at m=1; mean-over-rollouts if m>1 |

## 7. Trainium engineering plan

Everything here exists because Neuron compiles per shape; a shape that varies
with the data is a compilation storm.

- **Fixed anchor slots.** Pass 2 always carries `max_thoughts_per_seq = 8`
  thought blocks of `L_max + 2` tokens each; unused slots and post-`</T>` tail
  are fully masked and positioned harmlessly. Spliced length is the constant
  `T + 8·(L_max+2)` = 1024 + 528 = **1552** — inside the envelope we already
  proved at 2048 with grad-checkpointing + ZeRO + chunked CE.
- **T = 1024 for the thinking experiments** (my call). The corpus is a packed
  token stream — the loader draws any window length, zero data work. Rationale:
  spliced 1552 < 2048 keeps us in known-good HBM territory; sampler attention
  is ~4× cheaper than at 2048; thoughts targeting local uncertainty don't need
  2048 of context for a PoC. Phase 0 for the experiment arms runs at 1024 too,
  so all compute-matched comparisons share T.
- **No full-vocab logits in pass 2, ever.** A [B, 1552, 152k] logit tensor is
  ~944 MB bf16 per row — the allocation that already forced chunked CE at 2048.
  Pass 2 gathers hidden states at the ~200 needed positions (windows + thought
  positions), projects only those. Pass-1 NTP reuses today's chunked CE, with
  per-position entropy accumulated inside the same sweep at no extra memory.
- **Sampler = the critical path.** A fixed-shape, bucketed KV-cache sampler:
  segmented prefill (prefix lengths padded to a small set of buckets), a static
  `L`-step generation loop, probe steps as single-token forwards whose KV entry
  is written to a scratch slot (discarded on continue). This is where the
  engineering time goes after M2; the naive alternatives are either a
  compilation storm (dynamic shapes) or ~L full-length forwards per thought
  (no cache). Worst case per row: 8 anchors × (64 + 8) cached single-token
  steps + bucketed prefills. Build it correct first at Stage A sizes (L=16),
  optimize before Stage B.
- **Graph census per step:** pass-1 fwd+bwd, sampler prefill (few buckets),
  sampler decode step, probe step, pass-2 fwd+bwd, optimizer — each compiled
  once. The trigger/closure logic runs on host (numpy on gathered entropies),
  never in-graph.

## 8. Config defaults (0.6B, T=1024)

```python
T = 1024
k = 8                  # reward window
min_gap = 8            # = k, disjoint windows
max_thoughts_per_seq = 8           # fixed slots (shape constant)
q_target = 0.0075      # trigger quantile ~ slots budget: ~7.7 candidates/row
tau_abs = 2.0          # nats, absolute entropy floor
L_stageA = 16          # fixed length, Stage A
L_max = 64             # Stage B cap (config, may grow)
L_min = 8              # min length before first probe
probe_every = 8
delta = 0.5            # nats, gain-probe closure threshold
m = 1                  # rollouts; 2 = first variance knob (2x pass-2 cost)
alpha_max = 0.5        # REINFORCE weight, linear warmup over 10% of Stage A
beta = 1e-3            # thought-entropy bonus, anneal to 0
lam_len = 0.0 (A) / 0.005 (B)
clip = 3.0             # reward clip after whitening
temp = 1.1 -> 0.9
mix_hidden = 64
mix_bias_init = -2.0
eps = 1/(L_max + 3)
```

## 9. Evaluation protocol (honesty rules)

- All perplexity evals are **teacher-forced two-pass**, mirroring training
  exactly — no train/eval mismatch for the headline metric.
- **Closure-rule leak:** any eval where thought length was chosen by the
  true-token gain probe is optimistically biased on that very token. Every
  result is reported under both closure rules — *train-style* (gain probe;
  diagnostic only) and *inference-legal* (entropy-drop proxy or fixed cap;
  the headline). The gain↔entropy-drop correlation is logged continuously.
- Per-domain (fineweb-edu / finemath / dclm) and difficulty-bucketed (quartiles
  of `−logp_base` over windows) deltas, per the spec — thinking should win on
  hard buckets first.
- Generative demos use thought-stream logits directly (no mixing head at
  inference — it would need a parallel clean stream; noted approximation).
- Controls for the PoC: **(a)** equal-compute vanilla (more steps),
  **(b)** pause tokens (same trigger/slots, content = constant learned
  embedding, no REINFORCE — isolates "extra compute at test positions" from
  "learned content"), **(c)** full method. Success = c > b > a on hard-bucket
  perplexity at equal compute. The spec's further ablations (d–g) are deferred.

## 10. Milestones and tests (updated)

| M | Deliverable | Gate |
|---|---|---|
| M1 | positions, trigger, splice/mask, token-remap guard; unit-tested | tests green |
| M2 | two-pass step, random thoughts, mixing head; Neuron-compiled, stable shapes | parity tests + one recompile-free 50-step run |
| M3 | reward + REINFORCE (Stage A: fixed L, m=1) | unwhitened mean `R_raw` EMA > 0 and rising |
| M4 | fixed-shape KV sampler + probe machinery; logging dashboard; thought dumps | sampler ≤ ~1× pass-1 cost/step; thoughts inspectable |
| M5 | Stage B adaptive closure | closure-by-gain fraction > 0; length distribution sane; gate §5 passed |
| M6 | controls (a)/(b) + comparison table | headline criterion evaluated |

Tests (before M3, per spec discipline):

1. `test_positions` — strict `(n, n+1)` containment **including `L = L_max`**;
   float positions accepted end-to-end.
2. `test_splice_mask` — (i) zero anchors ⇒ pass-2 ≡ pass-1 (atol 1e-5);
   (ii) all positions before the first `<T>` ⇒ identical; (iii) mask-structure
   asserts: thought rows attend prefix + own thought only; suffix rows carry
   −inf at all thought columns; padded slots fully inert.
3. `test_parity` — τ=∞ (no triggers), α=0 ⇒ total ≡ vanilla NTP.
4. `test_ntp_allocation` — every real token's NTP counted exactly once.
5. `test_reward` — `R(planted answer) > R(shuffled garbage)`.
6. `test_reinforce_sign` — synthetic 2-token policy moves the right way.
7. `test_closure` — synthetic gain sequence ⇒ closes at the known probe; caps
   at `L_max`; closure decisions carry no gradient.

## 11. Compute accounting and PoC budget

Per step vs vanilla at T=1024 (C = vanilla fwd+bwd): pass 1 ≈ C; pass 2 ≈
1.5·C (1552 tokens, supralinear attention); sampler ≈ 0.5–1.5·C once the
fixed-shape cache works (the honest uncertainty; it is the M4 gate). Total
**≈ 3–4× vanilla**, vs the spec's 2.3–2.6× (which assumed GPU-style dynamic
shapes and branch-cheap rollouts).

PoC budget on this trn1.2xlarge (measured 4.3k tok/s at T=2048; assume ~5k at
T=1024): Phase 0 ~200M tokens ≈ 11 h; Stage A ~100M real tokens ≈ 1 day;
Stage B ~100M ≈ 1 day; controls (a)+(b) ≈ 1.5 days. **≈ 5 days end-to-end**,
or ~½ day on a trn1.32xlarge (data-parallel only; the code already scales by
launch flag). The full 3B-token thinking run is explicitly out of PoC scope.

## 12. Deliberately deferred

- Reward-to-go credit assignment (flagged config, default myopic).
- m > 1 rollouts / RLOO (first variance knob, 2× pass-2 cost).
- Learned closure or learned trigger (would reintroduce the gate-action
  estimator; only worth it if the mechanical trigger's think-rate placement
  looks wrong in the dumps).
- Quiet-STaR-style parallel thought generation; spec ablations (d)–(g);
  mixing head at generative inference (parallel clean stream).

## 13. Post-incident amendment (2026-08-26): split backward graphs

The first on-device run (2026-08-25) traced pass-1 forward, pass-2 forward,
and both checkpointed backwards into ONE XLA graph. Its HLO was 3.5MB — 2.2×
the vanilla train graph — and neuronx-cc's compilation exhausted the 30GB
host (no swap), livelocking the instance for ~12h until a reboot. Fixes now
in `think_train.py` / `run_poc.sh`:

1. **Two backwards per micro.** Pass-1 CE backwards immediately (its own
   vanilla-sized graph); pass-2's losses backward in a second vanilla-sized
   graph. Gradients accumulate across the two calls, so the optimizer step is
   unchanged.
2. **Pass-1 CE now trains ALL rows** (window predictor rows included). This
   makes `ntp1` exactly the vanilla objective (directly comparable to the
   Phase-0 loss) and is required by the split: zero-weighting window rows
   only made sense when the mixed loss also carried gradient to the base path.
3. **Base path detached in the mixed loss.** `h1`/`lg1` carry no grad in
   `lmix = w·lg2 + (1−w)·lg1`; the mixed window loss trains the thought path
   (`lg2` → pass-2 backbone) and the mixing head only. Deviation from §2's
   "mixed loss is the sole trainer of window rows" — accepted for the PoC;
   the base is simply trained vanilla-style everywhere.
4. **Host protection**: 64GB swapfile (fstab), a supervisor watchdog that
   kills `neuronx-cc` if MemAvailable < 1.5GB, `timeout`-bounded smoke
   attempts with one retry at `--optlevel 1`, purge of partial (neff-less)
   cache modules between attempts, and a cron `@reboot` re-entry with
   marker-file phase resume.

### §13 addendum (14:58 UTC): HBM budget fixes

The graph split fixed compilation, but two further on-device OOMs (622MB
embedding-grad allocation failing at 14.75GB/16GB core utilization) forced
three more changes to `think_train.py`, all Trainium memory engineering with
no algorithmic effect:

- **alpha as a device tensor** — as a Python float the REINFORCE warmup
  coefficient baked into the pass-2 backward graph, compiling one new program
  per step whose loaded copies filled HBM (first OOM, 14:17).
- **bf16 backbone parameters** (ZeRO-1 holds fp32 master shards). The
  thinking step keeps ~5 large programs resident where vanilla needs one;
  at fp32 params+grads+optimizer alone were 9.5GB (second OOM, 14:58).
- **Single backward variant**: grads pre-allocated once and
  `zero_grad(set_to_none=False)`, so only the grad-accumulate variant of each
  backward compiles (the create/accumulate pair kept 4 ~430MB programs
  resident); reward whitening made branch-free (0/1 device scale) so the
  rare n_valid<2 micro cannot pull in yet another variant.

Consequence for the comparison: the vanilla arm trained fp32 params, the
thinking arm trains bf16 params + fp32 master — standard mixed precision;
ntp1 remains directly comparable up to that precision difference.
- **Fused lm_head for all pass-2 grad consumers** (found via byte-diff of two
  same-size HLO dumps by the peer session): window-mix logits and REINFORCE
  logits from two separate lm_head calls gave the tied embedding two
  symmetric gradient contributions; autograd tie-breaks their accumulate-add
  operand order by node address, so the traced HLO hash flipped
  nondeterministically per micro — unbounded recompile churn no memory
  budget can absorb. One concatenated lm_head call (sliced after) leaves
  embedding+lm_head contributions at different depths: deterministic order,
  stable hash.
- **Residual ties are bounded, not eliminated** (15:26-15:35): the fused
  lm_head moved the operand-order tie to h2's two gather consumers and to
  lpt (shared by logpi and the entropy bonus). Each tie is binary, so at
  most 2^n_ties graph variants exist; all orderings compiled once in the
  first two steps and the cache went quiet (0 new modules after 15:35,
  steady 27.3s steps at 600 tok/s). bf16 headroom absorbs the extra
  resident variants. For a larger-scale rerun, make the graph unique by
  construction: one gather over concatenated indices, and entropy computed
  from lgt without sharing lpt's gradient path.
