# Thinking-in-Pretraining: Implementation Plan (REINFORCE)

Implementation plan for adding unsupervised "thinking" to an existing NTP pretraining pipeline.
Written to be handed to Claude Code as the working spec. Assumes an existing repo with:
a small decoder-only transformer (RoPE), a tokenizer, a dataloader producing fixed-length
token batches, and a standard training loop (`train.py` or equivalent) computing NTP loss.

---

## 0. Design summary (frozen decisions)

- **Thought delimiters**: special tokens `<T>`, `</T>` added to the vocab. Thought tokens are
  ordinary vocab tokens (shared embedding table and LM head with real text).
- **Ephemeral thoughts**: a thought anchored after real token `t_n` is attended to ONLY by the
  next `k` real tokens (the reward window). Real tokens beyond `n+k` mask it out. At inference,
  thought KV entries are evicted after `k` tokens.
- **Fractional RoPE**: real tokens keep their vanilla integer positions (identical to a
  no-thinking model). The i-th token of a thought anchored at `n` gets position
  `n + i * eps` with `eps = 1/(L_max + 2)`, i.e. strictly between `n` and `n+1`.
- **Loss on real tokens only**: NTP loss is never applied to thought tokens. Thought content is
  trained exclusively by REINFORCE.
- **Counterfactual baseline**: reward = (log-lik of the true window with thought) minus
  (log-lik without thought). The no-thought pass exists anyway (it is the vanilla NTP pass).
- **Mixing head**: final window logits interpolate between base-pass and thought-pass logits via
  a learned gate, so thinking can never make NTP worse than vanilla. Trained by plain backprop.
- **Three phases**:
  - Phase 0: vanilla NTP warmup (no thoughts).
  - Phase 1: entropy-gated **forced** thinking, fixed thought length `L`, REINFORCE on content.
  - Phase 2: model-controlled thinking (`<T>` emission + free `</T>` termination) with a length
    cost. Only start after Phase 1 shows positive mean reward on gated positions.

---

## 1. Milestone map

| Milestone | Deliverable | Gate to proceed |
|---|---|---|
| M1 | Tokenizer + fractional-RoPE + splicing/masking utilities, fully unit-tested | tests green |
| M2 | Two-pass forward (base + thought) with mixing head; thoughts random; NTP parity check | with-thought NTP loss ≈ base NTP loss when gate w→0 |
| M3 | Reward + REINFORCE loss wired in; Phase 1 training runs | mean whitened reward > 0 and rising on gated positions |
| M4 | Multiple rollouts (RLOO), logging dashboard, thought dumps | variance of policy grad acceptable; thoughts inspectable |
| M5 | Phase 2: learned gating + `</T>` termination + length cost | selective thinking emerges (think-rate correlates with entropy) |
| M6 | Controls & ablations (pause tokens, no-mixing, persistent thoughts) | paper-grade comparison table |

Do them in order. Do not start M3 until M2's parity check passes — almost every silent bug
(masking, positions, off-by-one in windows) is caught there.

---

## 2. Repo changes (file by file)

New module directory: `thinking/`

```
thinking/
  __init__.py
  config.py          # ThinkingConfig dataclass
  tokens.py          # special-token registration
  positions.py       # fractional RoPE position ids
  gate.py            # entropy gating (Phase 1) / learned gate (Phase 2)
  sampler.py         # batched no-grad thought generation
  splice.py          # build spliced batch + block attention mask + window index maps
  reward.py          # reward computation, whitening, clipping, RLOO
  losses.py          # ntp_loss_mixed, reinforce_loss, total_loss
  mixing.py          # mixing head module
  metrics.py         # all logging described in §8
tests/
  test_positions.py
  test_splice_mask.py
  test_parity.py
  test_reward.py
  test_reinforce_sign.py
```

Modified files:
- `model.py`: accept explicit (possibly fractional, float) `position_ids`; accept an explicit
  additive/boolean attention mask instead of assuming causal-only.
- `train.py`: replace single forward with the two-pass step in §6; add phase scheduler.
- `config`: add `ThinkingConfig` (see §7) to the run config.

---

## 3. Component specs

### 3.1 `tokens.py`
- Add `<T>`, `</T>` to tokenizer and resize embeddings. Init the two new rows to the mean of
  existing embeddings + small noise (avoids a giant-norm outlier token early on).
- Guard: these tokens must never appear in raw data; assert during data loading.

### 3.2 `positions.py`
- `build_position_ids(real_len, thoughts) -> FloatTensor` where `thoughts` is a list of
  `(anchor_n, length_L)`. Real token `t_j` gets position `j` (float). Thought token `i` of a
  thought at anchor `n` gets `n + (i+1) * eps`, `eps = 1/(L_max+2)`; `<T>` and `</T>` count as
  thought tokens for positioning.
- RoPE must consume float positions: compute rotation angles directly from the float position
  (no integer cast anywhere). Verify the existing RoPE impl doesn't `arange` internally.

### 3.3 `gate.py` (Phase 1)
- Input: base-pass per-token predictive entropies `H_t` (from pass 1, free).
- Online sequence entropy: EMA over the sequence, `S_t = (1-γ) S_{t-1} + γ H_t`, γ ≈ 0.3.
  Trigger a thought before position `t+1` when `S_t > τ` AND `t - last_trigger >= min_gap`.
- `τ`: maintain as a running quantile (e.g. top-q of `S` values seen) so the think-rate is
  approximately constant at `q` (target ~2–5% of positions) regardless of training progress.
- `min_gap >= k` so reward windows never overlap (keeps rewards independent; simplifies masks).
- Cap thoughts per sequence at `max_thoughts` to bound compute.

### 3.4 `sampler.py`
- `sample_thoughts(model, tokens, anchors, m, L, temp) -> LongTensor[m, num_anchors, L]`
- Runs under `torch.no_grad()` / inference mode. Prefill the prefix once per anchor, then
  autoregressively sample `L` tokens at temperature `temp` (start 1.1, anneal → 0.9).
- Batch across (sequences × anchors × m rollouts). Reuse prefix KV across the m rollouts of the
  same anchor. This is the throughput hotspot; correctness first (M1–M3), then optimize:
  a good target is Quiet-STaR-style parallel generation later, but plain batched sampling is
  fine at 50–100M scale with think-rate ≤5%.
- Phase 1: fixed length `L`, `</T>` forced (banned from sampling, appended). Phase 2: `</T>`
  allowed, hard cap `L_max`.

### 3.5 `splice.py`
- Builds the pass-2 batch: real tokens with thought blocks inserted after their anchors.
- Attention mask (per sequence), on top of causality:
  1. Thought tokens attend to: the real prefix `t_1..t_n` + earlier tokens of the same thought.
     NOT to other thoughts.
  2. Real tokens in the window `t_{n+1}..t_{n+k}` attend to: everything a vanilla model sees
     + the thought anchored at `n`.
  3. Real tokens outside any window attend exactly as vanilla (all thoughts masked out).
- Consequence of (3): logits of non-window real tokens in pass 2 are bit-identical to pass 1.
  This is the parity test, and it means you can skip loss computation for them in pass 2 and
  take their NTP loss from pass 1.
- With m rollouts: replicate only what's needed — practical layout is to treat rollouts as
  extra batch rows for pass 2 (windows are short; cost is modest).
- Emit index maps: `window_token_index[(anchor, j)] -> flat position in spliced batch`,
  `thought_token_index[(anchor, i)] -> flat position`. Everything downstream uses these.

### 3.6 `mixing.py`
- Per window token `j`: `w_j = sigmoid(MLP([h_base_j ; h_think_j]))` (2-layer MLP, hidden 64,
  scalar out). `logits_mix = w_j * logits_think + (1 - w_j) * logits_base`.
- Init final bias to ≈ −2 (w ≈ 0.12): early thoughts are noise; start leaning on the base head.
- Trained by backprop through the NTP loss only. No REINFORCE involvement.

### 3.7 `reward.py`
- Per thought (rollout `i` at anchor `n`):
  `R_raw = Σ_{j=1..k} [ logp_think(t_{n+j}) − logp_base(t_{n+j}) ] − λ_len * L_thought`
  using the **pure** thought-pass log-probs for the first term (not the mixed ones — see §5,
  "which logits feed the reward").
- RLOO across the m rollouts at the same anchor: `R_i ← R_raw_i − mean_{j≠i}(R_raw_j)`
  (skip if m == 1; the counterfactual baseline is already inside `R_raw`).
- Whiten across all thoughts in the batch: subtract mean, divide by (std + 1e-6).
- Clip to `[−c, c]`, c = 3 after whitening.
- `λ_len = 0` in Phase 1 (length is fixed anyway); > 0 in Phase 2.

### 3.8 `losses.py`
- `ntp_loss`: cross-entropy on real tokens only. Window tokens use `logits_mix`; non-window
  tokens use pass-1 logits. Never on thought tokens.
- `reinforce_loss = − mean_over_thoughts( R.detach() * Σ_i logπ(z_i) )` where `logπ(z_i)` are
  the log-probs of the sampled thought tokens, recomputed with grad in pass 2 (they are free:
  pass 2 already produces logits at thought positions).
- Optional exploration bonus: `− β * mean entropy of π at thought positions` added to the loss
  (i.e. encourage entropy), β ≈ 1e-3, anneal to 0.
- `total = ntp_loss + α * reinforce_loss + β_term`, α schedule in §7.
- Gradient-hygiene switch (config flag): optionally stop-grad the REINFORCE loss's flow into
  the embedding table for the first N steps of Phase 1 (`detach` embedding output on the
  thought-policy branch). Default ON for the first 20% of Phase 1.

---

## 4. Phase scheduler

- Phase 0: steps `[0, s0)`. Vanilla NTP only. `s0` ≈ enough to reach clearly-below-unigram
  perplexity (model produces locally coherent samples).
- Phase 1: steps `[s0, s1)`. Entropy-gated forced thinking. α warmup: linear 0 → α_max over the
  first 10% of Phase 1.
- Phase 2: begins only when `mean whitened R_raw > 0` sustained (EMA over 1k steps) AND
  win-rate (fraction of thoughts with `R_raw > 0`) > 0.5. Enable `<T>` as an emittable action:
  at gated-candidate positions, the model's own probability of `<T>` decides (sample it);
  reward for the gate action is the same `R` (think) vs 0 (skip), trained with the same score
  function estimator; `λ_len > 0`. Keep the entropy gate as a candidate filter initially
  (only positions above a low threshold even get the choice), then widen.

---

## 5. Training step (pseudocode, the heart of it)

```python
def train_step(batch, model, cfg, phase):
    # ---- PASS 1: vanilla (no thoughts) ----
    out1 = model(batch.tokens, position_ids=arange, mask=causal)      # grad ON
    logp_base = log_softmax(out1.logits)                               # [B, T, V]
    ntp_loss_flat = nll(logp_base, batch.targets)                      # real tokens
    if phase == 0:
        return ntp_loss_flat.mean()

    H = entropy(logp_base).detach()
    anchors = entropy_gate(H, cfg)                                     # [(b, n), ...]

    # ---- SAMPLE thoughts (no grad) ----
    with torch.no_grad():
        Z = sample_thoughts(model, batch.tokens, anchors, m=cfg.m,
                            L=cfg.L, temp=cfg.temp)                    # discrete tokens

    # ---- PASS 2: spliced (grad ON) ----
    spliced = splice(batch.tokens, anchors, Z)          # tokens, float pos ids, block mask
    out2 = model(spliced.tokens, position_ids=spliced.pos, mask=spliced.mask)
    logp_think = log_softmax(out2.logits)

    # thought-policy log-probs (for REINFORCE), from pass-2 logits at thought positions
    logpi = gather(logp_think, spliced.thought_index, Z)               # [n_thoughts, L]

    # window log-probs of TRUE tokens, per rollout
    lp_win_think = gather(logp_think, spliced.window_index, true_window_tokens)  # [n_th, k]
    lp_win_base  = gather(logp_base,  base_window_index,   true_window_tokens)   # [n_th, k]

    # ---- REWARD (pure thought-pass vs base-pass; detached) ----
    R_raw = (lp_win_think - lp_win_base).sum(-1) - cfg.lam_len * Z.length
    R = clip(whiten(rloo(R_raw, m=cfg.m)), cfg.clip)                   # detached scalar/thought

    # ---- MIXED NTP on windows (backprop trains reader + mixing head) ----
    w = mixing_head(h_base_windows.detach_or_not, h_think_windows)     # cfg flag
    logits_mix = w * logits_think_win + (1 - w) * logits_base_win
    ntp_win = nll(log_softmax(logits_mix), true_window_tokens)
    # best rollout per anchor feeds the mixed loss (or mean over rollouts; cfg flag)

    ntp = combine(ntp_loss_flat_nonwindow, ntp_win).mean()
    reinforce = -(R.detach() * logpi.sum(-1)).mean()
    ent_bonus = -cfg.beta * entropy(logp_think at thought positions).mean()

    return ntp + cfg.alpha(step) * reinforce + ent_bonus
```

Which logits feed the reward: use the **pure** pass-2 window log-probs, not the mixed ones.
With mixed logits, a lazy mixing head (w→0) makes every reward ≈ 0 and kills the policy
gradient exactly when thoughts are weakest. Pure-vs-base keeps the writer's signal alive even
while the reader ignores it; the mixing head independently learns when to start listening.

Which logits feed the NTP loss on windows: the **mixed** ones. That is the safety guarantee.

Detail — `w`'s input can use detached base hiddens (`detach_or_not`) so the mixing head can't
push gradients that degrade the base pass; default detach ON.

---

## 6. Compute accounting

Per step relative to vanilla: pass 1 (1×) + sampling (think_rate × m × L forward-steps with
short suffixes, prefix KV reused) + pass 2 (≈ 1× + spliced overhead). With think-rate 3%,
m = 2, L = 16, k = 8: expect roughly 2.3–2.6× vanilla step cost. Budget experiments so the
**equal-compute vanilla baseline** (same wall-FLOPs, more steps) is always trained alongside —
that comparison decides everything.

---

## 7. `ThinkingConfig` defaults (50–100M model)

```python
L = 16                 # Phase-1 fixed thought length
L_max = 32             # Phase-2 cap
k = 8                  # reward window
m = 2                  # rollouts per anchor (raise to 4 if variance is bad)
think_rate_q = 0.03    # target fraction of positions gated
min_gap = k
max_thoughts_per_seq = 8
gamma_ema = 0.3        # online entropy EMA
alpha_max = 0.5        # REINFORCE weight after warmup
beta = 1e-3            # thought-entropy bonus, anneal to 0
lam_len = 0.0 (P1) / 0.01 (P2)
clip = 3.0
temp = 1.1 -> 0.9
mix_bias_init = -2.0
embed_stopgrad_frac = 0.2   # first fraction of Phase 1
phase0_steps = ~5-10% of total budget
```

## 8. Logging (non-negotiable; this experiment is diagnosed through these)

Per N steps, log:
- mean/std of `R_raw` (unwhitened), win-rate `P(R_raw > 0)`, both overall and bucketed by
  base-pass window difficulty (quartiles of `−lp_win_base`) — thinking should win on hard
  buckets first.
- mixing weight `w`: mean and histogram. The single best health indicator: `w` drifting up
  from 0.12 means the reader is voluntarily using thoughts.
- Δperplexity on window tokens: mixed vs base.
- thought-policy entropy; fraction of thought tokens that are top-100 frequent tokens
  (detects collapse to degenerate filler); distinct-n of thoughts.
- gate stats: think-rate, entropy threshold τ, (Phase 2) correlation of `P(<T>)` with entropy.
- text dumps: 20 random (context-tail, thought, window, R) tuples per eval — the qualitative
  "are thoughts becoming meaningful" question is answered by reading these over training.

## 9. Tests (write before M3)

1. `test_positions`: fractional ids strictly increasing; real ids untouched; RoPE accepts floats.
2. `test_splice_mask`: non-window real-token logits in pass 2 == pass 1 logits (atol 1e-5),
   with random thoughts spliced in. **The** correctness test.
3. `test_parity`: full step with `alpha=0`, `w` forced to 0 → total loss == vanilla NTP loss.
4. `test_reward`: plant a thought that literally contains the window tokens → R_raw > 0;
   plant shuffled garbage → R_raw ≈ 0 mean.
5. `test_reinforce_sign`: synthetic 2-token policy, positive-R thought tokens gain prob after
   one step, negative-R lose. Catches sign errors that otherwise train silently backwards.

## 10. Controls & ablations (M6)

- (a) vanilla NTP, equal compute (more steps).
- (b) pause tokens: identical pipeline, thought tokens replaced by learned constant `<pause>`
  embeddings, no REINFORCE. Isolates "extra forward compute" from "learned content".
- (c) full method.
- (d) full minus mixing head; (e) persistent thoughts (drop window-only masking);
  (f) token-entropy gate vs sequence-entropy gate; (g) m = 1 vs 2 vs 4.
- Eval on held-out hard-token slices (last token of multi-step spans; if using a synthetic
  corpus: answer tokens of arithmetic/reachability items) — report per-slice perplexity,
  not just aggregate.

Headline success criterion: (c) > (b) > (a) on hard-token perplexity at equal compute, and in
Phase 2 the learned think-rate concentrates on high-entropy positions without supervision.