# Useful Thoughts on Trainium

Controlled experiments on whether self-generated intermediate computation improves
language-model predictions. Actual Qwen3-1.7B-Base, real mathematical documents,
and computation diagnostics run on AWS Trainium. Official TEST remains unused.
The [current proof ladder](docs/RESEARCH_ROADMAP.md) separates reader use,
prefix-only computation, actual student benefit, cost, and return to real text.

## Current finding

**Self-generated partial computation helps on fresh synthetic problems.** The
trained128-example writer produces useful work beyond its fitting set, and the
same frozen reader benefits on all four512-item panels. This establishes the
restricted computation channel. A matched-cost advantage and real-text
pretraining gains remain unestablished.

| Fresh panel,512 items each | Reader direct | Empty | Correct supplied state | Student state | Ordinary compact CoT |
|---|---:|---:|---:|---:|---:|
| Interpolation |21|21|343|285|287|
| Unseen structure |18|19|315|238|235|
| Larger operands |8|8|257|140|118|
| Longer composition |2|0|296|151|48|

Entries are complete correct answers, not likelihood scores. Student minus compact
CoT is−0.39percentage points [−4.49,3.52],+0.59 [−3.52,4.69],+4.30 [0.98,7.62],
and+20.12 [15.82,24.41], respectively; intervals pair questions within each panel.
Student beats the same reader's direct, empty and unrelated-note paths on every
panel. Unrelated generated notes yield3/0/0/1correct; expression restatements
yield34/31/9/2. Unsupported student answers are0/1/0/0; compact CoT1/3/1/2.

All2,048 writer attempts stop naturally, with no caps or generation errors.
Exact canonical intermediate states number416/311/288/227 of512. Supplied correct
states still yield only50–67% answer accuracy, so both writer reliability and the
reader's remaining calculation limit performance. The reader's weak direct path
also makes this a mechanism result rather than a comparison with a strong direct
solver.

**The split uses two models and256total training updates; compact CoT uses one
model and128.** Each model makes32passes over128expressions. Inference caps are80total
actions (writer64+reader16 versus CoT80). Actual mean actions are30.2/31.8/36.3/37.8
for the split versus31.2/31.8/37.4/32.9 for CoT. Separate-job timings are not a
matched end-to-end latency benchmark. This is one training seed, synthetic DEV,
and development evidence; official TEST remains unopened.

All18,432 answers,2,048 notes, controls, numerical vectors and paired intervals
passed the independent audit. Identical reader inputs produce identical actual
answers; repeated conditions do not add independent observations. Complete
outputs and the fixed first two examples per panel are public and EBS-archived:
[full result and payloads](validation/partial_state_v6_DEV512_frozen128_20261018.json) ·
[descriptive state-error decomposition](validation/partial_state_v6_DEV512_state_error_analysis_20261018.json).

Next: fresh-B0 training on5,120 verified examples, comparing answer-only,
compact CoT, full-trace CoT and the split at matched total updates, with two seeds.
Both compact and full-trace baselines remain competitors. Qualify full64 recovery
and storage before longer continuation. More capacity stays an option; the useful
signal already exists at1.7B. [Current queue](NEXT_STEPS.md).

## Earlier matched developmental64 comparison

**No overall capability gain is established.** The completed matched development64
comparison shows learned stopping and restricted note use, alongside degraded raw
prediction and no reliable GSM8K gain over the matching empty format. The original
320-update study was interrupted by host RAM exhaustion;64 is developmental only.
[Complete four-panel summary](validation/reader_learning_matched64_summary_20261018.json).

On real DEV documents, using W as both prefix-only writer and reader:

| Panel,256 documents | Interface | Own−raw | Own−empty | Own−unrelated |
|---|---|---:|---:|---:|
| Arithmetic | Ordinary re-anchor |−.616 [−.972,−.274]|+.749 [.414,1.095]|+3.865 [3.374,4.353]|
| Arithmetic | Fixed mixer |+.132 [.015,.256]|+.289 [.188,.397]|+.305 [.200,.419]|
| General | Ordinary re-anchor |−3.337 [−3.869,−2.828]|−1.237 [−1.635,−.864]|−.815 [−1.237,−.412]|
| General | Fixed mixer |−.257 [−.401,−.116]|−.174 [−.267,−.085]|−.151 [−.246,−.062]|

Values are natural16 likelihood gains in nats, with paired-document95% intervals.
Fixed mixing combines raw and note full logits at weight3/8, previously fitted on
separate TRAIN data. The ordinary-interface primary gate fails. W's mixed arithmetic
path is−.175 [−.395,.065] versus untouched B0 raw; its own raw path loses−.307
[−.499,−.092]. Thus its small within-model gain does not establish improvement
above the starting model. Arithmetic W notes cap in7/256 attempts, versus253/256
for C′. General W notes still cap in140/256. Both models' same notes were scored
with B0 and self readers; all vectors, notes, controls and costs are retained:
[arithmetic](validation/reader_learning_compound_DEV256_step64_20261018.json) ·
[general](validation/reader_learning_general_DEV256_step64_20261018.json).

The complete GSM8K TRAIN512 diagnostic gives:

| Model | Direct correct | Empty correct | Thought correct | Unsupported: direct/empty/thought |
|---|---:|---:|---:|---:|
| B0 |429|433|276|11 /18 /147|
| C′64 |415|429|337|4 /4 /43|
| W64 |391|400|412|2 /6 /2|

Every accuracy denominator is512. W thought−empty is+2.34percentage points
[−.98,5.66]; W direct−C′ direct is−4.69 [−8.20,−1.36]. W's large improvement over
C′'s thought path includes repaired stopping/formatting and cannot serve alone as
a capability claim. Direct already requests calculations and includes worked CoT
demonstrations. All3,072 trained outputs were independently re-parsed and remain
available, alongside the1,536B0 paths:
[matched64 results](validation/reader_learning_GSM8K_TRAIN512_step64_20261018.json) ·
[B0 reference](validation/postorder_gsm8k_TRAIN512_B0_20261018.json).

The frozen synthetic256 secondary panel has a stronger likelihood signal. W notes
all stop naturally; fixed mixing gains+6.450 [5.952,6.963] over raw,+4.910
[4.401,5.441] over empty, and+6.376 [5.894,6.878] over unrelated. All four family
own−empty means are positive. Supplied correct partial work, omitting the root
operation, still helps the trained reader: ordinary gain+9.811 [9.369,10.246]
over raw. **This is not free-answer accuracy.** W's notes may contain the answer,
and a post-hoc strict `Result` check finds only73correct,170incorrect and13unsupported
out of256. Exact non-root postorder statements match135/256. Large likelihood gains
can coexist with wrong final calculations.
[All synthetic results, families, notes and vectors](validation/reader_learning_controlled_DEV256_step64_20261018.json).

The subsequent partial-state study uses exact, prefix-computable **intermediate-only** labels, a trained
then frozen reader, a separately supervised writer, and ordinary compact CoT with
the same verified computations as a baseline. Complete-answer accuracy and actual
student-note correctness are primary. Data5,120TRAIN plus separate512-item structure,
range, composition and interpolation DEV panels are prepared; the initial128TRAIN
overfit check is mechanical. The short batched driver completed actual updates1–3,
full save, fresh-process full restore and updates4–6. Independent model/master/
moment byte checks pass; this does not establish long-run memory stability.
All three TRAIN128 fitting runs now finish, with independent terminal model,
master and moment audits passing. The prefix-only writer emits identical native
actions on CPU and Trainium:128/128 exact partial states with natural stops.
This is fitting evidence;
fresh generalization is reported above. The frozen reader answers128/128 with
correct partial states, versus120/128direct and119/128empty; corrupted states
yield4/128. Actual CPU-writer own/other contexts exactly match the executed
teacher/unrelated controls, allowing explicitly documented response reuse:
own128/128, unrelated1/128. **Ordinary compact CoT also gets128/128 using one
training run; the two-model split has not earned its complexity.** All2,688
original B0/reader/CoT answers were independently re-parsed, with unsupported
outputs retained. B0 fails the terse prompt format, so its zeros are not a
broad capability baseline. All fitting jobs and all four fixed fresh
DEV panels are now closed.
[Short-pack hardware evidence](validation/partial_state_v6_training_hardware_20261018.json) ·
[Complete fitting audit](validation/partial_state_v6_TRAIN128_training_20261018.json) ·
[CPU writer outputs](validation/partial_state_v6_TRAIN128_writer_CPU_20261018.json) ·
[Completed Trainium writer and channel](validation/partial_state_v6_TRAIN128_writer_XLA_20261018.json) ·
[Complete fitting answers and exact response reuse](validation/partial_state_v6_TRAIN128_answers_20261018.json) ·
[Intermediate-only design](validation/partial_state_v6_design_20261018.json) ·
[Current queue](NEXT_STEPS.md) · [Experiment summaries](STATUS.md).

The original W220/320 and C′119/320 failures, preserved64model-only snapshots and
read-back verified EBS archives remain documented in the
[host-OOM incident](validation/reader_learning_compound5120_host_OOM_20261018.json).
A four-condition backward-only probe does not reproduce runaway RAM over16 rows.
The complete update loop does: constant/scheduledLR and full/reduced sampled
diagnostics all retain substantial private host memory. Live XLA tensor counts
stay flat while compilation handles keep accumulating under a two-entry cache.
A four-core cache-capacity comparison cuts warm W updates43.466s→1.925s,
with exactly equal losses/norms and94graph reloads replaced by94cache hits. Both
small-topology memory tails are bounded, so this is a speed gain rather than a
repair of the original32-rank failure. The new short reader completed128updates
with full checkpoints in about five minutes; the128TRAIN fitting answers above
now establish mechanics only.
[Backward probe](validation/host_memory_backward_probe_20261018.json) ·
[Full-loop attribution and raw traces](validation/host_memory_full_loop_attribution_20261018.json) ·
[Cache-capacity comparison](validation/host_memory_cache_capacity_20261018.json).
The separate future-aware general-label dataset and CPU-only `general-sft-v5` branch
remain on hold; selected teacher usefulness does not establish prefix-only
attainability. [Complete general teacher batch](validation/general_hindsight_filtered_TRAIN24576_20261018.json).

Earlier B0 real-note likelihood results remain negative:
[arithmetic](validation/postorder_held256_B0_20261018.json),
[general](validation/general_note_held256_B0_20261018.json).
The supplied-partial B0 mechanism check is preserved with its original limits:
[controlled result](validation/controlled_arithmetic_DEV256_B0_20261018.json).

## What led here

- **A copying interface is qualified; arithmetic headroom is limited.** None of six
  ordinary interfaces passed the initial full-span oracle. The TRAIN-fitted fixed
  mixer passes general DEV256 (tax−.154, correct−raw+11.948, correct−corrupt+12.203
  nats), while the original arithmetic panel fails (+.641/.249 gains). General
  oracle notes copy the next16 tokens; they are not prefix-only thoughts. The
  original arithmetic number-only NLL is just .648 nat. DEV reuse makes this
  exploratory interface qualification.
  [Six-interface batch](validation/interface_target_copy_dev256x2_20261018.json) ·
  [Complete mixer fit and controls](validation/interface_scalar_mixer_dev256x2_20261018.json).
- **The initial symbolic filter failed its quota.** Only3,147/14,669 notes beat
  empty-format (21.45%); mean teacher−raw is−1.809nats including the closing boundary.
  The new compound5120 study explicitly teaches the reader as well as the writer,
  using verified compound equations selected without likelihood filtering. It does
  not declare the old filter passed.
  [Failed filter](validation/postorder_symbolic_frozen_reader_filter_20261018.json) ·
  [New design](validation/reader_learning_compound5120_design_20261018.json) ·
  [Data inventory](validation/reader_learning_compound5120_data_20261018.json).
- **Short storage and recovery mechanics are qualified.** FP32 model storage exhausted
  memory in W/C′. BF16 storage with FP32 masters completed three real updates per
  arm, and actual full-state recovery restored all32ranks and performed the next
  ordered update. Earlier numerical-reference and installed-optimizer validation
  failures remain preserved. These short checks did not prevent the later sustained
  host-memory failure. They are execution checks, not long-run stability or capability results.
  [Hardware evidence](validation/useful_note_bf16_storage_hardware_20261018.json) ·
  [Recovery](validation/useful_note_state_recovery_002_20261018.json).
- **Nearby partial notes did not help.** The fixed TRAIN 256 / DEV 64 probe completed
  all 1,920 likelihood cells and 384 numeric paths. Correct−raw numeric likelihood
  was −1.351/−1.600 nats, both intervals negative. DEV correct/raw/empty/corrupt
  initial-scalar counts were 28/31/31/10 of 64, with unsupported 31/22/22/29.
  Eighteen corrupted outputs matched the changed whole value (one also matched
  the original partial): sensitivity existed without net assistance. These bounded
  parser scores exclude some correct multistep calculations; all paths remain saved.
  [Complete likelihood, generation and mechanism evidence](validation/finemath_arithmetic_near_lhs_reader_20261018.json) ·
  [Earlier prepend probe and parser limits](validation/finemath_arithmetic_reader_20261018.json).
- **Initial R study stopped.** Inserting the original thought format carried a
  large initial penalty: B0 DEV512 K1 raw −1.118, empty −7.361, own −11.203 nats.
  Own versus unrelated content was indistinguishable from zero. R1234 stopped
  after 63 updates; R5678 never launched. Both old C runs finished 256 updates.
  The original primary R/C comparison is incomplete.
  [B0 audit](validation/local_dev_B0_20261017.json) ·
  [R execution and preserved checkpoint](validation/science_R_seed1234_stopped_20261017.json).
- **Repairing format did not establish content utility.** Prepending notes
  reduced empty-format tax to approximately zero. On fixed TRAIN128, a
  future-selected best-of-16 tail gained +1.026 nats over empty, while the
  ordinary mean lost 1.072 nats. Oracle selection sees the target.
  [Interface probe](validation/prepend_reader_dev32_20261017.json) ·
  [TRAIN oracle](validation/candidate_prepend_train128_20261017.json) ·
  [DEV oracle](validation/candidate_oracle_dev32_cap256_20261017.json).
- **The first learning comparison was negative.** A 64-update note-cloning arm
  and a raw future16 CE control shared 77 documents and 512 ordered occurrences,
  with unequal labels and compute. Fresh DEV64 notes still hurt the unchanged
  reader: K16 own−empty B0 −1.101, raw −0.956, cloning −0.649 nats. The
  cloning-minus-control change was +0.307 [−0.209,+0.908]. Both terminal states
  are fully backed up. Reader adaptation also failed.
  [Three-arm result](validation/warmstart_dev64_three_arm_20261018.json) ·
  [Reader crosses](validation/warmstart_reader_cross_dev64_20261018.json).
- **Downstream checks also failed.** GSM8K TRAIN128 direct/empty/thought correct
  counts were B0 113/107/63, cloning 92/92/61, raw CE 56/112/76. Every thought
  mode lost to its own empty control. Raw direct had 59 format errors, so its
  score does not isolate arithmetic ability.
  [All nine conditions](validation/warmstart_gsm8k_train128_terminal64_20261018.json).
- **External partial calculations show limited assistance.** A separate TRAIN32
  pilot scored arithmetic-verified manual notes 30/32, empty 27/32, corrupted
  notes 5/32. All three added strict successes repaired answer formatting or
  avoided truncation; none corrected a wrong parsed number. Numeric likelihood
  improved +0.945 [0.258,1.591] nats. One manual note missed a purchasing
  discount despite correct arithmetic; its frozen note and score are unchanged.
  Local 4B notes included final answers. These are qualifications on assistance,
  not evidence of useful student computation.
  [Pilot](validation/gsm8k_calculation_train32_20261018.json) ·
  [Exhaustive posthoc mechanism panel and correction](validation/gsm8k_calculation_train32_mechanism_20261018.json).
- **Short self-generated notes also failed.** On known GSM8K TRAIN32 questions,
  unchanged B0 scored 86/128 sampled paths (67.2%), versus 24/32 (75%) for both
  direct calculation and generate-then-discard. Four paths estimate a single-note
  policy, never best-of-four or voting. Own−direct is −7.81pp [−18.75,+2.34];
  numeric likelihood also declines. All notes, failures and costs are retained.
  Equal token caps do not establish equal FLOPs or latency.
  [Complete self-note result](validation/gsm8k_self_note_train32_20261018.json).

All official final tests remain unused. Failures stay in denominators, controls
are reported alongside raw performance, and oracle results remain separate from
ordinary generated-note utility. Historical implementation and results remain
available in Git and the linked reports.

## Code and plan

| Path | Purpose |
|---|---|
| [configs/weekend.json](configs/weekend.json) | Original stopped experiment configuration |
| [docs/WEEKEND_PLAN.md](docs/WEEKEND_PLAN.md) | Original scientific design and interpretation |
| [train/computable_equations_supervised.py](train/computable_equations_supervised.py) | Verified real equations for supervised labels; future-aware curation |
| [train/interface_oracle_probe.py](train/interface_oracle_probe.py) | Complete six-interface full-span target-copy probe |
| [train/calculation_boundaries.py](train/calculation_boundaries.py) | Bounded exact-arithmetic parser and original stricter curator |
| [train/calculation_boundary_probe.py](train/calculation_boundary_probe.py) | Controlled reader feasibility and exact LHS reanchoring |
| [train/scalar_completion.py](train/scalar_completion.py) | Versioned initial-scalar boundaries; legacy scores remain preserved |
| [train/neuron_driver.py](train/neuron_driver.py) | Historical C/R qualification and driver |
| [train/neuron_backend.py](train/neuron_backend.py) | Causal cache, generation, scoring, and policy replay |
| [train/evaluate.py](train/evaluate.py) | Question-only, sharded math evaluation |
| [train/evaluate_likelihood.py](train/evaluate_likelihood.py) | ARC and full-target LAMBADA likelihood |
| [train/evaluate_local.py](train/evaluate_local.py) | Saved prefix-only notes, content controls, and crossed readers |
| [weekend_thinking/](weekend_thinking/) | Objective, data, checkpoint, and analysis primitives |
| [tests/](tests/) | CPU correctness checks; these do not certify hardware |
| [context/](context/) | Research context and evidence limits |
| [legacy/](legacy/) | Preserved historical source and results; not active launch instructions |

## Running checks

Use a compatible, already working Neuron environment for accelerator work.
Instance paths in `scripts/env.sh` are local conveniences; adapt them to your
own installation. Model weights and corpus acquisition are explicit actions.

```bash
bash scripts/run_checks.sh
```

The CPU checks require PyTorch, NumPy, pytest and a compatible Transformers
installation, and disable accidental XLA initialization. Accelerator qualification,
actual distributed updates, full-step timing, and checkpoint resumption are
separate requirements. Read [START_HERE.md](START_HERE.md) and
[docs/NEURON_IMPLEMENTATION.md](docs/NEURON_IMPLEMENTATION.md) before launching.

Host configuration, credentials, downloaded datasets, weights, compiler caches,
raw traces, and checkpoints remain outside Git under ignored local paths.
Small validation reports describe their provenance and limits.
Verified local checkpoint copies are described in
[CHECKPOINT_BACKUP.md](docs/CHECKPOINT_BACKUP.md).

Original authorship and third-party notices are preserved; see
[NOTICE.md](NOTICE.md). Public availability does not assign a new license to
historical material.
