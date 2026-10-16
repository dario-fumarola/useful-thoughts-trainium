# Thinking in Pretraining — Next Steps (post run 34)

Audience: the agent continuing this programme on the cluster. Read §0–§2 before touching anything.
Execute experiments in the order of §7. Every gate below is pre-registered: do not edit a gate after its data exists.

---

## 0. Why the plan changed

Thirty-four runs (see the paper draft) found that a thought written by the model is worth less than content-free filler of the same length, on the same weights, at every checkpoint. This is predicted by an identity, not an engineering failure.

**Setup.** Context $c$, window $W$, writer $q(z\mid c)$, reader $p(W\mid c,z)$, clean pass $p_0(W\mid c)$.

$$
G(z)=\log p(W\mid c,z)-\log p_0(W\mid c),\qquad
m(W\mid c)=\sum_z q(z\mid c)\,p(W\mid c,z),\qquad
r(z\mid c,W)=\frac{q(z\mid c)\,p(W\mid c,z)}{m(W\mid c)} .
$$

**Identity (holds for all $c,W$, any model).**

$$
\mathbb{E}_{z\sim q}[G]=\underbrace{\log\frac{m(W\mid c)}{p_0(W\mid c)}}_{\text{computation term}}-\underbrace{D_{\mathrm{KL}}\big(q(\cdot\mid c)\,\|\,r(\cdot\mid c,W)\big)}_{\text{commitment cost}\,\ge 0}.
$$

Oracle (posterior) thoughts instead get $+D_{\mathrm{KL}}(r\|q)$: that is information, unavailable at inference.

**Consequences.**
1. If thinking buys no computation ($m\le p_0$), a self-sampled thought has $\mathbb{E}[G]\le 0$, and the loss grows with how much the thought says about $W$. This explains own < filler, harm growing with length and with speaking rate, and collapse to register/silence.
2. Our reward $\mathbb{E}_q[G]$ pays the KL. Reasoning-model RL uses a reward linear in probability, $\mathbb{E}_q[p]=m$, which does not.
3. Under log loss the optimal inference-time thought is deterministic and only helps via computation. Useful thoughts are **derivations** (content determined by $c$ but not computable in one pass), not **forecasts** of the uncertain future. Our supervised target (true continuation) trains forecasts.

**The single deciding quantity is the computation term $\log(m/p_0)$.** E0 measures it. E1 checks that the pipeline can harvest computation where it certainly exists. E2 changes the objective only if E0 says there is something to harvest.

---

## 1. Standing rules (inherited; violations void a result)

- **Precision:** fp32 parameters, bf16 compute under autocast. Never bf16 parameters with plain AdamW.
- **Learning rate:** $10^{-5}$, cosine to $5\times10^{-6}$, 200-step warmup. $5\times10^{-5}$ destroys the base.
- **Parallelism:** `--no-zero1` (ZeRO-1 reduce-scatter hangs at 64 ranks on trn2; sharding buys nothing at this size). If the new cluster is different hardware, re-verify, but keep fp32 master weights either way.
- **Sampling:** top-k on host, never on device (Neuron miscompiles top-k over the 152k vocab). Any new fast path needs an on-device replay against CPU ground truth before it is trusted.
- **Positions:** aligned register, thought token $i$ at position $n+1+i$ (`--slot-pos aligned`). Score every checkpoint in the register it was trained in; record the register in every output row.
- **Evaluation splice:** reproduce the training splice exactly (overlapping integer positions, anchor-hiding mask). Plain concatenation is invalid.
- **Held-out ruler in the loop:** fixed 32-sequence held-out CE on every checkpoint. Training loss is not base health.
- **Controls on every claim:** length-matched content-free filler on the same weights and anchors (the filler floor), paired per anchor. Token-matched **and** compute-matched vanilla continuations for any benchmark claim (a thinking step ≈ 4× a vanilla step per token; measure it on the new cluster, do not assume).
- **Statistics:** paired per-anchor differences, report mean ± s.e. and $n$. Quote a gate only against a file probed with the same anchor count, register and flags.
- **No best-of-R numbers as results.** Selection inflates by ≈ +1.4 s.d. at $R=8$. Report unselected rollouts.
- **Provenance:** every dumped thought carries supervised/reinforcement/filler provenance.
- **Abstaining lineages:** read value from the length-matched draft probe, never from the think-mode greedy arm (on those lineages greedy thought = filler string).

---

## 2. Checkpoints

Locate by run id and step. If a checkpoint cannot be found, stop and report; do not substitute silently.

| id | what | tokens | role |
|---|---|---|---|
| `V@8.99B` | vanilla ladder rung, held-out mixture CE 2.775 | 8.99B | **default init for every new training run** |
| `V-cont` | vanilla continued from `V@8.99B` on the same stream | matched per arm | token-matched and compute-matched controls |
| `run25@900` | calibrated reader, reference checkpoint of the paper | — | E0 probe only |
| `run26@1500` | filler-trained reader (cheapest block, +12.6-nat hint value) | — | E0 probe only |
| `run31@800` | best free-running writer, aligned register | — | E0 probe only |
| `run34@800` | final speaking writer, aligned register | 9.957B | E0 probe only |
| `run21@7700` | end of phase A, built register | 9.05B | E0 probe only |

Rationale for `V@8.99B`: every thinking lineage carries inherited damage (LAMBADA −13 on run 34, reader over-trust, register habits). New runs start clean so their effects are attributable.

Model: Qwen3-0.6B architecture (28 layers, hidden 1024, GQA, vocab 151,936, untied head), seq 1024. Data: 74/16/10 FineWeb-Edu / FineMath / DCLM, same held-out split as the paper. Global batch 131,072 tokens/step unless stated.

---

## 3. E0 — Measure the computation term (zero-update, run first)

**Question.** Is $\log(m/p_0)>G(z_\varnothing)$ on any checkpoint, i.e. does a content-bearing thought buy computation beyond what filler buys?

**Estimator.** For each anchor draw $R$ unselected thoughts $z_i\sim q(\cdot\mid c)$ and compute

$$
\hat C_R=\log\frac1R\sum_{i=1}^{R}e^{G_i}.
$$

$\mathbb{E}[\hat C_R]$ is nondecreasing in $R$ and converges to $\log(m/p_0)$ (IWAE bound). $\hat C_1$ in expectation is $\mathbb{E}_q[G]$. The commitment cost is estimated by $\hat C_{R_{\max}}-\overline{G}$.

**Procedure.**
1. Checkpoints: `run25@900`, `run26@1500`, `run31@800`, `run34@800`, `run21@7700`, each in its own register.
2. Anchors: the standard held-out set, $n=465$ (also $n=616$ where a comparison to the paper requires it).
3. Thought length: the checkpoint's trained $L$ ($L=16$), its own stop behaviour.
4. Sampler: **primary** = pure ancestral sampling, temperature 1.0, no top-k, no frequency penalty (so $q$ is the model's actual distribution). **Secondary** = the training sampler (top-512, frequency penalty). The identity holds for whichever $q$ is sampled; report both, decide on the primary.
5. $R\in\{1,2,4,8,16,32\}$ from one pool of 32 samples per anchor (subsample for smaller $R$, average over subsamples).
6. Scores, two variants:
   - (a) undiscounted window log-likelihood, $\gamma=1$, $k=64$. Exact for the identity.
   - (b) discounted per token: for each $j$, $\log\frac1R\sum_i f_j(z_i)-\log p_{0,j}$ with $f_j(z)=p(t_{n+j}\mid c,z,t_{n+1:n+j-1})$, summed with $\gamma^{j-1}$, $\gamma=0.97$. Estimates $\sum_j\gamma^{j-1}\log(m_j/p_{0,j})$ and matches the training quantity.
7. Filler: content-free filler at identical positions, length matched to the mean sampled length. Also report filler binned by the sample length distribution.
8. Also log, per checkpoint, the oracle gain (true 12-token hint at offset 0) as a reader-health check.
9. CIs by bootstrap over anchors (10,000 resamples), paired with filler per anchor.

**Outputs.** `e0_results.csv` (checkpoint, register, sampler, variant, R, $\hat C_R$, s.e., filler gain, $\hat C_R-$filler, s.e.) and a plot of $\hat C_R$ vs $R$ per checkpoint with the filler line.

**Decision (pre-registered, primary sampler, variant a).**
- **NULL:** on every checkpoint, $\hat C_{32}-G(z_\varnothing)$ upper 95% CI $\le 0.05$ nats **and** $\hat C_{32}-\hat C_{16}<0.02$. There is no computation term at this scale and data. Do not run E2 on web text. Run E1.
- **POSITIVE:** on at least two checkpoints, $\hat C_{32}-G(z_\varnothing)>0$ at $>3$ s.e. The objective hides a real capability. Run E1 and E2.
- **AMBIGUOUS:** $\hat C_R$ still rising at $R=32$ but not significant. Extend to $R\in\{64,128\}$ on $n=200$ anchors of the most promising checkpoint, then re-apply the rule.

### 3a. AMENDMENT (2026-09-26, agreed with the user BEFORE any E0 data existed)

**Why.** $\hat C_R$ against a single filler compares an $R$-way mixture with one pass. Log-mean-exp of noisy
gains rises by about $\sigma^2/2$ with no computation at all (lognormal). The within-anchor rollout spread logged
in training (`g_sd`, R = 8) is 1.22 (run 25), 1.68 (run 31), 0.76 (run 34), i.e. $+0.74$, $+1.41$, $+0.29$ nats of
"free" mixture value against own-minus-filler gaps of $-0.4$ to $-0.8$. The original rule would likely read
POSITIVE on runs 25 and 31 from dispersion alone, and a positive from hedging over guesses is exactly the forecast
channel that consequence 3 says is not computation. The original NULL clause ($\hat C_{32}-\hat C_{16}<0.02$) is
also nearly unreachable at $\sigma\approx1.2$--$1.7$.

**Matched-R controls.** At every $R$, the same estimator on four mixtures of the same size:
`own` (the thoughts), `shuf` (each thought's content tokens permuted), `cross` (thoughts sampled for a *different*
held-out sequence at the same anchor slot), `fill` (content-free filler of length $|z_i|$, the length distribution
of the samples). Also reported: greedy thought (the deterministic-writer reading of consequence 3), filler of
length 10 (the paper's floor), the true 12-token continuation (reader health).

**Decision (governs; primary sampler, variant a).** POSITIVE iff on at least two checkpoints
$\hat C_{32}(\text{own})-\hat C_{32}(\text{ctrl}^*)>0$ at $>3$ s.e., paired per anchor, where ctrl* is the matched
control with the highest mean on that checkpoint. Otherwise NOT POSITIVE, and E2 stays parked. The original rule is
still computed and reported for transparency, but it does not license E2.

**Implementation.** `pre-training/train/probe_e0.py` (raw per-token log-probs saved, so every estimator can be
recomputed without the model; the clean prefix is prefilled once per sequence into a ScatterCache and only the
block+window rows are run per condition, verified against `tp.score_buffer` to 3e-5), analysis
`pre-training/train/e0_analyze.py` -> `runs/e0/e0_results.csv`, `runs/e0/e0_curves.png`. Queue:
`runs/e0/run_e0_queue.sh` (3 CPU lanes x 24 threads; r25@900, r31@800, r34@800, r26@1500, r21@7700). Traversal =
`probe_thinkmode.py`'s (data/corpus val, 12 per domain), so anchors pair with every earlier thinkmode file.

---

## 4. E1 — Positive control: a domain with a certain computation term

**Question.** Can the pipeline make a self-written thought beat filler where a scratchpad is known to help? Without this, a web-text null cannot be attributed to the data rather than the machinery.

**Init.** `V@8.99B`.

**Synthetic task (primary).** Multi-digit addition, operands of 6–12 digits, answer written most-significant-digit first in the text. One forward pass handles this poorly because carries propagate from the right; a least-significant-first scratchpad makes it locally computable (Lee et al., 2023, *Teaching Arithmetic to Small Transformers*).
- Text format: `Q: 48213907 + 90771452 = ` followed by the answer and a newline.
- Scratchpad (supervised thought target): the answer digits reversed, least significant first, then `</T>`. At most 13 digits, fits $L=16$.
- Verify the Qwen tokenizer splits digits into single tokens; if not, adapt the format so each digit is one token.
- Build synthetic sequences so that every anchor falls on the token immediately before an answer (one problem per 64-token block, answer starting at anchor $+1$). This keeps the stride-64 geometry and the compiled graph unchanged. The scored window then contains the answer and the next problem.
- Optional second task if time allows: two-operation expressions `a + b - c`, scratchpad = reversed intermediate sum then reversed result.

**Data mixture.** 90% the original mixture, 10% synthetic sequences. Held-out synthetic set of 2,000 problems, disjoint operands.

**Arms** (same init, same data stream, same token budget of 1.0B tokens ≈ 7,600 steps at 131,072 tokens/step):
- **E1-think.** At synthetic anchors: writer imitation on the scratchpad (teacher-forced, weight 1.0); reader trained on the writer's own unselected sampled thought (inference-matched, the run-25 lesson). No RL. At web-text anchors: content-free filler, reader trained (run-26 style), so the web negative does not contaminate the test.
- **E1-filler.** Filler at every anchor, reader trained, no writer. Isolates extra positions.
- **E1-vanilla.** NTP only. Token-matched and compute-matched copies.

**Metrics** (held-out synthetic set unless stated):
1. Answer exact match, greedy, in three modes: clean (no thought), filler, think (own greedy thought).
2. Window gain own − filler, length-matched, paired per anchor, sampled and greedy.
3. Teacher-forced scratchpad in slot: gain over filler (reader ceiling).
4. Free-running scratchpad accuracy (digit-level and exact).
5. Web held-out mixture CE vs `V-cont` token-matched.

**Gates (pre-registered).**
- **G1 at 0.25B tokens (plumbing):** teacher-forced scratchpad gain − filler gain $\ge 2$ nats per window, paired, $>3$ s.e. If it fails, the reader cannot use a correct scratchpad: stop, debug splice/positions/masks with the oracle and slot-position probes before anything else.
- **G2 at 0.5B tokens (the result):** think-mode exact match $\ge$ filler-mode exact match $+10$ points absolute (McNemar $p<0.01$), **and** own − filler window gain $>0$ at $>3$ s.e.
- **G3 (health):** web held-out CE within $+0.10$ nats of token-matched `V-cont`.

**Reading.**
- G1 and G2 pass: the pipeline harvests computation. Together with an E0 NULL, the web-text result is "no computation term at 0.6B on web text," a clean conclusion.
- G1 passes, G2 fails: the writer cannot produce correct scratchpads free-running. Report free-running vs teacher-forced scratchpad accuracy (the exposure-bias signature from the paper). Do not start scheduled sampling (falsified in run 30).
- G1 fails: machinery bug. Everything downstream waits.

### 4a. IMPLEMENTATION NOTES AND AMENDMENTS (2026-09-26, before any E1 data)

- **Run name `think40_add`** (`runs/think40_add/config.sh` carries the full rationale and the gates).
- **Tokenizer.** Qwen3 splits every digit into its own token (verified). `Q: a + b = ` ends with a separate
  `' '` token (220); the anchor sits on it, so the window starts at the answer's first digit. `synth.py`
  asserts that its ids equal the tokenizer's encoding of the text. Gaps between problems are filled with
  newline (198) repeated.
- **Micro schedule.** Synthetic iff `(step*GA + micro) % 10 == 0`: the same micro on every rank, so the 90%
  web micros skip decode and scoring on all ranks at once (a per-rank coin would make every step wait for the
  slowest rank). About 4.7 s/step on average, against about 27 s for runs 20-34.
- **Anchor roles at synthetic anchors.** 25% filler (reader trained), so filler-before-an-answer is in
  distribution and the within-weights filler comparison is fair; of the rest, 50% supervised on the
  scratchpad, 50% the model's own unselected sampled thought (reader trained on it). `--rl-w 0`: no RL.
- **Sampler.** `--repeat-ban 0 --freq-pen 0`. The training sampler banned consecutive repeats and penalised
  frequency, which would corrupt reversed-digit scratchpads (repeats about 10% of the time).
- **Gate steps.** G1 at step 2000 (0.26B tokens), G2 at step 4000 (0.52B tokens): multiples of 500 so the
  checkpoint is archived.
- **Uninformative clause (added).** If filler-mode exact match is >= 0.85 at step 2000, one pass already
  solves the task and the +10 bar fails for the wrong reason: stop, switch to `a + b - c`.
- **Probe.** `pre-training/train/probe_e1.py <ckpt> <out.json>`: held-out set of 143 sequences x 14 = 2,002
  problems (disjoint seed stream). Modes: clean / filler / tf (true scratchpad) / greedy / sample. Exact match =
  every answer digit plus the closing newline is the teacher-forced argmax, which equals greedy decoding.
  The vanilla init scores 0.000 exact match in every mode.
- **Filler and vanilla arms** run after E1-think (the device is single); G1/G2 are read within weights.

---

## 5. E2 — Multi-sample objective (only if E0 is POSITIVE)

**Question.** With the commitment cost removed from the objective, does a thought beat filler on web text?

**Init.** `V@8.99B`.

**Anchors.** Stride 64, $k=64$, $L=16$, aligned register. Supervised fraction $\rho=1/4$ with target = immediate continuation, fixed offset 0 (`--oracle-mode fixed --oracle-off 0`; never the jittered lottery). Remaining anchors are sampled anchors with $R=8$ rollouts. No best-of-R selection anywhere.

**Loss at sampled anchors.** Replace $\mathbb{E}_q[G]$ by the multi-sample bound:

$$
\mathcal L_{\text{IW}}=-\log\frac1R\sum_{i=1}^{R}p(W\mid c,z_i),\qquad z_i\sim q(\cdot\mid c).
$$

- Reader: exact gradient; rollout $i$ weighted by $w_i=\mathrm{softmax}_i(G_i)$.
- Writer: VIMCO (Mnih & Rezende, 2016). With $\hat L=\log\frac1R\sum_j e^{G_j}$ and $\bar G_{-i}=\frac1{R-1}\sum_{j\ne i}G_j$, the learning signal for $z_i$ is

$$
A_i=\hat L-\log\frac1R\Big(\sum_{j\ne i}e^{G_j}+e^{\bar G_{-i}}\Big),
$$

applied to $\nabla\log q(z_i\mid c)$. Keep the paper's weights otherwise (clean NTP 1.0, reader 0.3, writer 0.15 ramped over 600 steps). Use the discounted per-token form if the undiscounted window proves too high-variance; state which.

**Inference (must match the objective).** Draw $R_{\text{inf}}$ thoughts and mix predictions with posterior weights per token:

$$
p_{\text{mix}}(t_{n+j}\mid\cdot)=\sum_i w_i^{(j)}\,p(t_{n+j}\mid c,z_i,t_{n+1:n+j-1}),\qquad w_i^{(j)}\propto p(t_{n+1:n+j-1}\mid c,z_i),
$$

which equals $\frac1{R}\sum_i p(W\mid c,z_i)$ at the sequence level. Report $R_{\text{inf}}\in\{1,4,8\}$.

**Controls.** Filler arm from the same init; `V-cont` token-matched and compute-matched, where compute includes the $R_{\text{inf}}$ decodes at inference.

**Gates (pre-registered).**
- At 0.25B tokens: $\hat C_8$ on held-out (E0 estimator) $>$ its value at step 0, paired, $>3$ s.e. Otherwise stop: the objective is not moving the computation term.
- At 0.5B tokens: mixture think-mode ($R_{\text{inf}}=8$) − filler window gain $>0$ at $>3$ s.e.
- Health: held-out CE within $+0.10$ of token-matched `V-cont`.
- Final: zero-shot suite (LAMBADA, HellaSwag, WinoGrande, ARC-E/C, OpenBookQA, SciQ, BoolQ; 500 each, paired) vs compute-matched `V-cont`. A win against token-matched only is not reported as a win.

---

## 6. E3 — Derivation targets on natural data (only if E1 passes)

Extend E1 from synthetic to natural math: FineMath documents in which a computed quantity appears later in the text and its derivation is checkable (arithmetic in worked solutions). Thought target = the derivation step or intermediate value, not the continuation. Specify the extraction pipeline and its precision before training, and pre-register gates in the E1 form (teacher-forced ceiling first, then own − filler). Do not start E3 without a written spec reviewed by the user.

---

## 7. Order and decision tree

1. **E0** (zero-update). Report to the user before anything else.
2. **E1** regardless of E0's outcome (it is the positive control either way).
3. **E2** only if E0 = POSITIVE.
4. **E3** only if E1 passes G2, and only after the user reviews the spec.

```
E0 ──NULL──────► E1 ──pass──► conclusion: no computation term on web text at 0.6B; propose E3
   │                 └─fail──► report machinery / exposure-bias diagnosis; stop
   ├─POSITIVE──► E1 + E2
   └─AMBIGUOUS─► extend R, re-apply rule
```

Stop and report to the user at every gate, pass or fail. Do not launch an unplanned run onto an axis a probe has closed.

---

## 8. Reporting

For each experiment write `results/<exp>/SUMMARY.md` with: config diff from the defaults above, init checkpoint, tokens trained, every gate with its pre-registered bar, the measured value, $n$, register and verdict, plus paths to raw per-anchor dumps. For the paper: the identity goes into §2; E0's $\hat C_R$ curves become the central figure of §4; N2–N5 are recast as consequences of the identity.