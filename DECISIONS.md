# Decision log

## 2026-10-16 — handoff baseline
- Start with one mature-model reference condition; do not claim novelty from a
  mature-model success alone.
- Use the prior weekend specification, not a new objective invented during packaging.
- Keep the training/inference/deployment/efficiency claims separate.
- Reuse audited runtime components, not old launchers or entire damaged lineages.
- Package original source and project context so a second upload is unnecessary.
- Add local resumption/scoring/specification tools; no hardware qualification claim.
- Remove the sampler's unnecessary finite upper-tail Gumbel clamp; the intended
  temperature-softmax categorical policy remains unchanged. Device numerical parity
  still needs measurement. This is documented in CHANGELOG.md.

For each later scientific change, record: date, old/new choice, reason,
whether outcomes had been inspected, new experiment ID, and supporting artifact.
Routine formatting changes do not need a decision entry.

## 2026-10-16 — isolated owner-authorized workspace
- Clone the handoff into `/home/ubuntu/dario`, preserving the original sources and history.
- Owner explicitly requested a GitHub repository and updates: created private `dario-fumarola/dario`; this authorizes commits/pushes to that repository, overriding inherited no-push instructions for this scope. Keep assets, credentials, compiler output and run data ignored.
- Reuse existing accelerator environment unchanged; install missing test helpers only under ignored `local/test-deps/`.
- Current session is familiarization only; retain fixed scientific recipe and defer implementation/launch. No research outcomes inspected.

## 2026-10-16 — implementation phase
- Owner extends available time to weeks and explicitly authorizes review/implementation agents and evidence-based adaptation. Preserve one interpretable primary C/R experiment; do not treat six-hour integration or weekend cutoff as current instance expiry. Use bounded diagnostic attempts.
- Download actual released Qwen3-1.7B-Base at immutable commit. Preserve its declared embedding/head tie explicitly after XLA transfer; no independently trained matrices are retied.
- Before scientific outcomes, declare writer support as tokenizer-supported IDs only (mask267model-only rows), identically in sampling/replay and C/R deployment. Reader likelihood and C loss retain released full vocabulary.
- Install no accelerator upgrades. Use hash/version-gated process-local fix for an observed compiler artifact-path bug and explicit restored syncfree scalar-state device placement; preserve failures and numeric-resume limitations.
- Correct URL identity before freeze: preserve content-bearing query parameters. Use audited v2 FineMath asset; retain v1 artifact as evidence of the bug.
- Qualify task answer serialization on TRAIN items only; repeat the original question after thought close to prevent the reader from continuing into a new few-shot example. Preserve failed initial output.

## 2026-10-16 — long-trace precision diagnosis
- Full-cap BF16 cache/replay maxlogp error .21841 exceeds predeclared .2 bound despite tiny typical error; keep failed evidence. FP32 vocabulary head alone does not repair it: exact old trajectory still differs .22614. Do not raise tolerance. Test full FP32 compute at the already declared .001 bound, with both random rollouts and the fixed old failing trajectory, before selecting production precision.
- Replace per-action output lists with static on-device buffers and check only an alive scalar every32steps. Stop padded computation after all four thoughts terminate; keep1024output shape/cap, sampled-stop masks and raw IDs. Count padded work separately. CPU parity/stopping tests pass; new hardware path remains pending.

## 2026-10-16 — select FP32 production candidate before scientific outcomes
- Full FP32 passes the exact old failing actions: maxlogp error2.83718e-5 under the original tighter.001 bound; the same oldoutlier is3.576e-6. FP32head alone fails, so headquantization is not sufficient explanation; BF16 trunk/cache/full arithmetic causes the discrepancy in this stack.
- Select full FP32 parameters/compute for BOTH primary arms; update proposal identity to mature-reference-2026-10-16-v2-fp32 before any scientific launch or final evaluation. Objective,model,data,G4,L1024,K1,temperature,batch,LR remain unchanged. Complete timing and fullmodel resume remain required at selected precision/topology.
- Deep SSD preflight found nonzero highentropy middle blocks despite absent FS signatures; abort before any device write. Continue EBS qualification; investigate read-only before treating devices as disposable scratch.

## 2026-10-16 — initialize verified unused instance-local scratch
- Read-only boot/device evidence establishes fresh currentinstance, allfourinstance-store devices with zero kernelwrites/discards, nofilesystem/partitions/mounts/holders. Middleblockhighentropy is not a zero-read guarantee for discardedNVMe; it alone doesnotestablishuserfiles.
- With existing ownerhardwareautonomy, initialize ONLY `/dev/nvme0n1` as ext4 at `/home/ubuntu/dario/local/nvme`; repeat model,size,zero-write,no-signature/partition/mount guards immediatelybeforewriting. Receipt `validation/nvme_preparation_20261016.json`. OtherthreeNVMe andEBS untouched.
- Use scratch forlargeintermediatecheckpoint/compilerfiles. Scientific terminalcheckpoints and dailylatestcomplete backups also copiedtoEBS; instance-localNVMe is not externaldurablebackup. No cloudcapacity or accountchanges.

## 2026-10-16 — owner-authorized public repository
- Owner explicitly requests public visibility and a clearer name. Rename remote to `dario-fumarola/useful-thoughts-trainium`; publish after tracked-history credential scan and applicable third-party notice/source repair. No weights, corpus, credentials, checkpoints or raw local outputs enterGit.
- Preserve original research history and attribution. Public release is not a new scientific result or blanket relicensing of supplied historical material.

## 2026-10-16 — prospective execution budget with weeks available

- The owner replaced the original weekend constraint with weeks of available
  time. Adopt a 14-day local planning horizon from the first scientific launch;
  this is an autonomous job limit, not a claim about instance shutdown.
- Allow up to 240 hours for the two-seed training schedule, including the
  existing 15% reserve. Keep the remaining horizon for evaluation, verification,
  and recovery. Choose the largest predeclared step count in 256/512/1024/2048
  that fits actual complete-step timing and fresh data. Use timing only; neither
  qualification utility nor final outcomes select exposure.
- After C/R isolation and full-model resume qualify, C1/C2 can run consecutively
  on one partition while R1 runs on the other; then R2 uses the freed partition.
  Check this complete schedule against measured C and R times. If concurrency
  does not qualify, budget the sequential schedule explicitly.
- Before multi-day runs, arrange complete checkpoints at least daily, with
  verified EBS copies of the latest complete state and each terminal result.
  Current immutable qualification jobs keep their original checkpoint schedule.

## 2026-10-16 — runtime identity and checkpoint cadence before science

- Device-memory diagnostics show that the documented legacy Neuron cache limit
  does not control residency in this stack. The installed XLA LRU setting does;
  its full-backward adequacy is still pending. Bind the actual allowlisted cache,
  compiler and CPU-thread environment to each arm's qualification/specification,
  run and resume state. Permit separately declared arm policies only with their
  own measured costs. This changes operations, not the scientific objective.
- Before multi-day science, add synchronized time-triggered complete saves every
  12 hours at update boundaries, in addition to the original milestones. Record
  unmeasured C reward fields and unaudited replay errors as null. Qualify the new
  source's actual warm updates and resume before freezing; CPU tests alone do not
  establish that the added save-decision collective works on the accelerator.
- Preserve the original failed CPU-vector-norm comparison. A full-gradient
  Float64 reduction exposed a biased FP32 reference reduction and passed the
  original tolerance against device update1. Recompute the three-update CPU
  reference with accurate clipping as a separately identified lineage; do not
  change the declared bound or overwrite the original result.

## 2026-10-16 — repair observed device sequence normalization before R

- The installed XLA LRU limit allows the original full-cap backward to fit in
  memory. Its norm nevertheless differs from the independent CPU reference by
  roughly1024 times. Preserve that failed numerical result; R remains gated.
- A reduced device test establishes that the raw int64 mask count is correct,
  but its `clamp_min(1)` returns -1. Both signed losses and gradients reverse
  direction and lose the length normalization. Materializing loss first does
  not repair this defect. Use `mask.float().sum().clamp_min(1.0)` instead: counts
  through1024 are exactly representable. All six fixed-cap1024/head-chunk32
  cases (active lengths1/8/1024, both materialization paths) pass the original
  scalar/full-input/full-head gradient bounds. Separate short-shape failures
  remain retained. Receipt: `validation/p2_integer_length_clamp_20261016.json`.
- This is an implementation correction to the existing signed normalized
  objective, before any scientific R run. Require actual repaired full-Qwen
  loss/norm and signed-gradient sample agreement before launching R qualification.
  Keep the original norm bound; add prospectively declared sample cosine/L2
  checks because a norm cannot detect a sign reversal.
- Qualification R saves every rank/micro's rollout/reward tape after backward,
  and saves the first post-resume baseline checkpoint (R14:3/4/7/14). A resume
  also exports actual restored state before sampling, then verifies the first
  resumed actions and full post-update state under the original tight bounds.

## 2026-10-16 — preserve stochastic continuation across checkpoint observation

- The completed C32 fresh-process comparison passes full model/master/moment
  numerical bounds but fails exact device RNG on all 32 ranks. Each mismatch is
  exactly one installed XLA MarkStep transition. The old path resets the saved
  seed before its final load sync; that sync advances even a deterministic graph.
- Move CPU/device RNG restoration after all device loading and its final sync.
  Checkpoint publication/readback captures the incoming RNG at a settled boundary
  and restores it after all exports and barriers. Preserve unsigned 64-bit seeds;
  add no sync after resetting. Do not substitute per-record reseeding or weaken
  exact stochastic-resume criteria.
- Actual installed Neuron barriers differ from generic upstream XLA rendezvous:
  the measured no-payload barrier and checkpoint exports do not advance RNG.
  A bounded hardware negative control reproduces the old seed/next-draw failure;
  the corrected order reproduces both subsequent random draws and optimizer
  state exactly. Require the integrated helper and then full C/R resume evidence
  before scientific launch; the small probe alone is not full-model qualification.

## 2026-10-16 — separate held-seed numerical accuracy qualification

- Keep the original CPU FP32 three-update per-weight comparison as **v1 FAIL**.
  It is not a reliable ordering of numerical accuracy: the independent FP64
  diagnostic finds smaller full-gradient error on device and resolves most of
  its 754 flagged coordinates in the device's favor. Isolated near-zero gradients
  can change sign, and Adam's first update magnifies that difference. This does
  not establish CPU trajectory identity or excuse arbitrary errors.
- Design a distinct **v2** after that diagnosis and validate it on fixed held
  seed5678. Bind its first128 TRAIN IDs, overlap with seed1234, batch projection,
  source and criteria before computing new outcomes. Use fresh released B0 and
  one ordinary C32 update; scientific data/order and optimization are unchanged.
- Compare the complete first derivative against FP64 differentiable arithmetic
  with frozen native-CPU FP32 RoPE constants. Require absolute loss error≤1e-4;
  norm error≤1e-5+5e-4×reference norm; full-gradient relative L2≤.001 and cosine
  ≥.9999; and relative L2≤.01 for every unique parameter tensor. Compute vector
  reductions in FP64; exactly zero reference tensors require exactly zero device
  gradients, without a denominator floor. Require complete names/shapes/tie,
  finiteness, exact batch/scaling and first Adam step. Publish all tensor metrics.
- This qualifies first-derivative accuracy, not coordinatewise CPU three-step
  identity. Device derivatives recovered from first moments include FP32 clipping
  and moment rounding; frozen CPU RoPE constants are not a unique device truth.
  Same-hardware recovery and identical-gradient optimizer checks remain separate.
  Do not tune criteria after held-seed outcomes; preserve any v2 failure.

## 2026-10-16 — schedule both seeds across existing partitions

- Correct an earlier throughput estimate: R1 and R2 can overlap. With S updates,
  C step cost c and R cost r, running C1→C2 beside R1 then starting R2 takes
  about S(r+2c). Sequential full-node64 with ideal half-time steps takes S(r+c),
  saving only Sc rather than halving the study.
- Two independent 32-core chains, C1→R1 and C2→R2, already take about S(c+r)
  with both chains concurrent. Each R starts from fresh released B0, never from
  its C checkpoint. Prefer this schedule if concurrent-R isolation qualifies;
  preserve global128/accum4, existing sampling streams and qualified topology.
  Do not add world64 qualification solely on the faulty speedup assumption.
- Before freeze, record core placement by seed chain rather than contradictory
  fixed per-arm core ranges; runtime/cache policy remains declared per arm.
- Overlap one bounded upper-partition R update with the already-required lower
  fresh-process resume. Retain exact lower stochastic tapes/RNG and full-state
  checks, real upper update/replica/counters, separate endpoints, and memory/cost
  observations. Use measured coexistence costs and the existing reserve in the
  exposure calculation. No scientific result selects this operational schedule.

## 2026-10-17 — reserve space for serialized science backups

- Retire only the obsolete EBS `qwen-c-fp32-001/step-0000003` duplicate after
  verifying all 33 original NVMe files against its unchanged COMPLETE manifest
  and backup lineage. Preserve the original, retirement receipt and both current
  C backups. This frees 25.65 GiB; EBS now has about 324.5 GiB free.
- The proposed four-run retention plus R qualification leaves about 68 GiB.
  One checkpoint rollover fits its 25.65 GiB payload and 20 GiB reserve;
  simultaneous copies can exceed that allowance. Use one sequential host backup
  queue across the science roster, preserving two verified copies per run and
  all NVMe originals. Bind and review that helper separately before launch.
- Put science monitoring logs on NVMe and include checkpoint storage/rollover
  space in the final exposure plan. The active qualification driver, runtime,
  mathematical recipe and existing R backup supervisor are unchanged.
- Independent review found that one backlogged cycle can copy two unseen
  checkpoints before pruning. Reserve two payloads plus 20 GiB (about 71.3 GiB),
  including actual incomplete staging at any restart. Relocate the old C4 EBS
  checkpoint and its hardlink alias to a fully readback-verified NVMe archive;
  preserve both identities, bytes and relocation receipts before retiring the
  EBS aliases. This recovers another 25.64 GiB; current EBS free space is about
  346.7 GiB. The shared host helper passes six fixtures and independent TERM
  review; no shared science supervisor has launched.

## 2026-10-17 — qualify NVMe temporary program copies

- A bounded read-only capture attributes substantial warm-path delay to EBS
  temporary NEFF writes: 32 workers net-wrote 3.914 GB in 30.10 seconds; the root
  disk wrote 3.948 GB at an observed 125.07 MiB/s, with no physical reads.
  Open worker files were `/tmp/tmp*/model.neff`. Device executions were brief.
  This is observed throughput, not a claim about a configured volume limit.
- The pinned wrapper copies each warm cached NEFF into an outer
  `tempfile.TemporaryDirectory()`. A fresh-process NVMe `TMPDIR` redirects this
  path without changing program bytes, flags, code, synchronization or math.
  It does not relocate the separate hardcoded cold compiler work directory.
- Prepare a new bounded upper-partition R qualification with the same seed1234,
  B0, data, common source and full recipe. Change only temporary-file placement;
  keep the persistent cache unchanged. Bind its exact host I/O environment and
  filesystem separately in site/launch metadata, because the current runtime
  allowlist excludes TMPDIR. Measure the speedup before claiming it.
- R003 remains intact pending reviewed preparation and explicit release. Preserve
  its artifacts and finish a complete optimizer boundary before any cooperative
  stop. New full-update, steady timing and recovery evidence remain required.
  See `validation/r_warm_cache_io_20261017.json`.

## 2026-10-17 — first R update and conditional RAM host qualification

- R003 and R004 complete the same first update with identical numerical metrics;
  all128 rollout tape pairs are byte-identical. R003 cooperatively saves/exit0.
  All serialized model/master/moment state passes inspection; original and EBS
  backup payloads are independently rehashed. Retire its obsolete terminal14
  watcher after verification. Preserve the initial negative mean gain.
- R004 continues through14 for actual steady timing/checkpoints. Its first-step
  time includes programs compiled by R003; core placement and raw-log placement
  also differ. Neither startup time nor partial I/O implies a steady speedup.
- Existing RAM comfortably accommodates temporary NEFF copies. Prepare the
  required upper checkpoint3→4 recovery with a separately bound private RAM
  TMPDIR and the original precreated STOP. Keep every exact restoration/tape/RNG
  and tight update criterion. A longer continuation cannot replace this proof.
- Prepare a lower fresh-B0 seed5678/max14 RAM job for coexistence and ten warm
  timings. Same source, recipe and persistent EBS cache; raw logs/checkpoints stay
  on NVMe. Three upper/seven lower host fixtures and independent review pass.
  No launch until R004 completes, actual topology is free, and a finite root
  release binds concrete input/fork/readiness receipts. No new numerical oracle.
  Select science host policy and exposure from actual measurements.

## 2026-10-17 — adopt the owner's next-day guidance and small first study

- Read owner-provided `CODEX_NEXT_DAY.md` (review snapshot5cd4c79). The referenced
  companion review file is absent locally. Current verified R003 save/backup and
  R004 update1 supersede its historical pending status; active jobs stay unchanged.
- Prospectively target **256 terminal updates per scientific arm/seed**, rather
  than choosing the largest affordable candidate. No scientific pair or outcome
  exists yet. Final freeze waits for R recovery, steady/simultaneous costs and
  storage. If timing makes256 unsuitable, explicitly choose a declared alternative
  before outcomes. Each C and R independently starts from B0 with matched order.
- Keep step64 as a fixed development milestone, without selecting the primary
  checkpoint by its score. Use a fixed held subset, disjoint demonstrations, all
  three modes direct/empty_format/one_thought, errors/truncations and measured
  token/latency costs. Preserve the same-policy primary, R-direct minus C-direct,
  R-thought minus R-empty and each arm versus B0; C degradation alone is not an
  improvement above B0. Direct allows ordinary calculations. Local K1 primary,
  K4/K16 secondary, with absolute clean/empty/own/filler evidence and fixed random
  dev token-class/surprisal descriptions; do not filter whitespace or training.
- One bounded host-storage optimization branch at a time, no more disks, stack
  upgrades or external destinations. Close original missing evidence only; no
  new numerical investigations/gates absent a concrete fault. Keep monitoring
  patient and summaries compact. Long healthy runs are expected.
- Keep the two reviewed RAM preparations unchanged; no extra upper profiling job.
  Report the actual overlap episode honestly, separately from ten warm timings,
  and budget an explicit serial fallback. At256, the initial204 usable training
  hours allow about23.74min/R step even for two serial C/R chains (C about10s).
  That is a feasibility ceiling, not a measured throughput or concurrency claim.
- Required receipt assembly found one original single-accelerator requirement
  lacks a saved counterfactual: future perturbation under fixed writer RNG/model.
  CPU/source-API evidence exists. Prepare one bounded unchanged-backend full-cap
  device check on an idle core, with independent review before launch. This closes
  existing named evidence; it introduces no new oracle, gradients or investigation.


## 2026-10-17 03:04 UTC — reverse only RAM qualification coordination

The upper-first plan could consume its sole strict update while the fresh lower
job is loading or initializing its optimizer. Keep the same two jobs and reverse
their order: prepare/verify the original upper fork/STOP, start lower freshB0 to14,
then release upper strict3→4 immediately after fixed lower COMPLETE3. This gives
initialized lower optimizer state before the overlap. No extra profiling job,
source/math/host-policy change or relaxed recovery criterion is introduced.
All v1 bytes remain immutable; reviewed v2 coordinator/launchers bind them and
actual ancestry/mapping/readiness, with finite root releases. Six host fixtures
and both static probes pass; no RAM full-update result exists. Original fork
hashing remains required. Report the actual overlap and ten lower timings5–14
separately, retaining serial fallback. Public prospective manifest:
`validation/r_ram_sequence_20261017.json` (daf199d0…ac1c6).


## 2026-10-17 04:21 UTC — retain the complete B0 development baseline

The fixed128 TRAIN suite completes all3 modes: direct113/128, empty107/128,
thought63/128. Thought mode has33 format failures and14 answer truncations;
all remain incorrect, alongside the ordinary wrong answers. No runtime errors.
The results show a substantial cost for this unchanged B0 thought protocol;
they do not isolate arithmetic ability or establish a training benefit.
Preserve all outputs and verified EBS copies. Do not change the selected items,
prompt, parser, caps, primary contrast or prospective256 terminal because of this
outcome. Report trained models against B0 in every mode, with R-thought versus
R-empty and R-direct versus C-direct alongside the primary same-policy contrast.
A recovery of formatting or a gain over a degraded control alone must remain
distinct from capability above the starting model. Public result/provenance:
`validation/held_dev_B0_20261017.json`.


## 2026-10-17 05:09 UTC — retain the stopping diagnostic and start RAM qualification

R004 completes14 full updates and verified EBS7/14. Its final512 samples include
509 empty notes after terminator stripping; all stop at base EOS. Mean absolute
raw gain remains negative. Preserve this qualification TRAIN behavior and do not
call shorter notes, centered advantages or a faster update evidence of usefulness.
The prospective256 terminal and unchanged three-mode comparisons remain fixed;
any later apparent benefit must be distinguished from empty-format recovery or
control degradation. All scientific arms still start afresh from B0.

The original strict cp3-only fork/STOP is now verified. Root releases v2 lower
freshB0/seed5678 RAM qualification, deadline13:05:19.591488 UTC. Data owns the
one-shot fixed COMPLETE3 readiness; root owns upper release/launch and the sole
active backup queue. No original v1 launcher is released.


## 2026-10-17 07:41 UTC — choose serial production before outcomes

Use one32-core training group at a time: **C1234, C5678, R1234, R5678**.
Each arm independently loads unchanged B0; within-seed order/exposure, FP32,
qualified host policy and prospective256/64 terminal/development remain fixed.
Keep seed1234 on lower0–31 and5678 on upper32–63. Both controls finishing first
also makes their fixed development checkpoints available during R training.

This is the documented serial fallback. Actual two-R pilot execution passed,
but ordinary MemAvailable reached285.93GiB. Each production rank independently
loads329535 TRAIN records versus3180 in the pilot, so concurrency lacks a
conservative reserve. Do not credit unmeasured restore-memory release or a2x
throughput multiplier. The necessary timed freeze supplies full-loader peak RSS
for the32-rank resource projection before launch. The final ten R timings5–14
remain pending; retain all overlaps and checkpoint/backup costs. No new loader
rewrite, world64 qualification, upper profiling run or numerical gate is needed.
Use idle cores for genuine development evaluation when actual memory permits.

Immutable production source `local/science/source-256-001` matches all61 qualified
active files (SHA244e2f13…287ff7), archived from public commit85c64a9. Its four
finite per-arm launchers passed independent review but have no releases. Freeze,
readiness and drivers must use that archive and absolute evidence/control paths;
root operational docs remain current. Science is not frozen or launched.


## 2026-10-17 08:32 UTC — close qualification and freeze the first study

RAM005 completes14/exit0/cleanup and all ten fixed timings5–14: median704.789s,
P90869.283s, mean735.977s, retaining every shared-host interval. Strict R restore,
tapes/RNG and full continued update pass the unchanged original criteria; actual
sampled two-R execution remains distinct from ten simultaneous warm timings.
Terminal qualification notes are511/512 empty with negative absolute utility.
Preserve this evidence and do not change the prospective256/64 protocol.

Root independently verifies132 source/EBS payload reads for lower7/14. After its
watcher exits, verify all33 NVMe files for each obsolete EBS duplicate and retire
C6/R0031/R0047/RAM0057. All original NVMe states and latest qualification copies
remain; unique upper continued4 now has a verified distinct-filesystem EBS copy.
Off-instance protection remains unresolved.

The first archive freeze fails a host-path check: the unchanged resolved format
has a tokenizer path relative to the live repo. Preserve that failure/log/time.
Create a separate format copy changing only tokenizer_dir to the same absolute
B0 directory. Every other JSON field and format token/text/stop/pad setting is
identical; original e02d296b…61c50 stays intact. The new production hash
dbb5ad67…b9346 and independently reviewed path-only bridge2f2f2f4c…9bb84 are
explicit in the receipt. No numerical source/model/data or format policy changes.

Freeze retry succeeds once, spec **653b4efa43e933839733075eb5c2184a80de6dd24404263a8c9297adb9e241f7**,
finite deadline2026-10-31T08:27:49UTC, terminal256. The original deadline is not
widened by the metadata repair. All nine actual checks and all ten timings bind
the exact immutable source/spec/assets; original check_ready passes. The full
loader peaks at6882680KiB. Conservative32-copy memory projection, observed worker
peaks, RAM temp and128GiB reserve leave372.75GiB extra. Eight retained science EBS
payloads plus two rollover copies, per-payload allowance and20GiB reserve fit;
all-original NVMe saves/logs also fit. A1200s/R-step planning stress scenario plus
C/start/save/backup totals182.09h versus204 usable training hours; this is not a
throughput guarantee or a reason to maximize steps.

Keep serial C1234,C5678,fresh-B0 R1234,fresh-B0 R5678. No scientific arm is yet
released. Start matching shared backup/node leaders, bind fresh actual
topology/ports/temp/memory/storage and then issue the first finite root release.
Public frozen spec, actual evidence receipt, resource projection and format
bridge are in validation/science_256_* and validation/format_path_bridge_*.

### 2026-10-17 22:27 UTC — owner stops original study; investigate reward interface

Owner explicitly superseded the 256-update R plan. Wrote cooperative R1234 STOP
to save the current complete-update boundary; terminated the two active DEV
evaluations (exit143, partial outputs retained). R5678 and the old final matrix
are cancelled/on hold. Preserve original results and identities. Next: small
DEV-only interface-tax probe, genuine candidate oracle, and measured stage
profiling before further training. G1 B0 notes are not a best-of16 bundle; K16
is the future horizon. Finite oracle failure would not prove impossibility.

First repair probe completed22:34UTC: DEV32, five reader interfaces, original G1
notes and global controls,896scores/no failures. Reanchor32 empty−raw−1.080
versus original−6.425 nats; own−empty still−1.906, own−raw-repeat−1.627. No
content-utility claim. Next exploratory candidate search chooses original and
reanchor32 readers using this pilot; genuine16 draws atcap256, no new writer
interface yet. Original paritymax3.815e-5 nats; all400 report contrasts/intervals
independently checked. R stopped at63/exit0/verified27.54GB checkpoint backup;
old shared services retired after backup, with honest interrupted node capture.

The standalone replay profiler failed before its first backward because it
omitted the XLA device-module registration already in the training driver.
Failed output/exact source retained; applied only that registration and relaunched
same actual cases/settings/criteria as replay-padding-dev2-runtimefix-v1. No
numerical discrepancy or speedup established by the failed run. Candidate G16
sampling is separate and passed its first FP32 cache/replay audit(1.097e-5 nats).

Prepared a separate real TRAIN pool before targets:1024 hash-ordered unique
FineMath docs, B0 raw surprisal>=1.7, first128 qualifying/no fallback. This keeps
any later warmstart off DEV. Small one-core scan and a separate published-RLP/
plain-boundary DEV reader probe are running while G16 generation completes.
Both reuse actual FP32 B0, preserve reference parity and fixed controls. Their
results will guide the next explicitly identified short training experiment. G1 tail recheck found71/512
nonempty-note wins vs empty,4 vs raw K1,0 vs raw K16; all498nonempty/512docs
retained. These are descriptive tails, not proof of systematic useful thoughts.


### 2026-10-17 23:11 UTC — real TRAIN candidates; reject support-mass hypothesis

TRAIN raw scan completed1024/1024,237 qualify at frozen>=1.7; selected first128
without fallback. Start a new TRAIN128/G16 corpus, eight finite generation shards
on separate devices. All2048 must close before global controls and future scores;
future-based labels will be training supervision, not deployed selection. No
new student or long RL run yet. This reuses the qualified original writer and
keeps DEV out of the training corpus.

Published instructions did not repair the format tax. Plain-newline+reanchor32
reduces damage, but own−empty remains−1.175 nats on fixed DEV32. Exact released
BF16 input rows for all25 non-EOS added token identities are equal, including
think/end-think. Thus the corrected delimiter gives byte-identical scores. No
training-history or sole-cause inference. An explicit same-logit support probe
then rejected a separate hypothesis: undefined output mass changes the original
empty tax by only.000549 of5.678 nats on DEV8. Keep original full-support reader
scores and do not introduce an unsupported masking fix. Reports and raw outputs
are retained with hash-verified EBS copies; actual useful thoughts still pending.


### 2026-10-17 23:39 UTC — repair the boundary; do not equate selected labels with useful thinking

Genuine G16/cap256 DEV32 closed512samples/4256scores. Reanchor32 best16−empty
K1+.330 has a positive conditional interval, but ordinary mean16 is−1.612;
K16 oracle and every max-own/max-control interval span zero. Original TRAIN128
best16K16 remains−.661 on average; strict future-selected controls yield31labels.
No student was launched from those selected-label statistics.

New reader intervention prepends the fixed original note before the fullprefix.
Empty tax is now essentially zero, but ownK16−empty−1.783 and own−other−1.647.
Rescore the same complete TRAIN candidates under that layout, separately named,
without additional generation. A zero-tax reader may ignore or be distracted by
notes; this repair does not establish useful content. Full-prefix repetition in
the original insertion layout was worse and is not selected.

A pinned open Qwen3-4B teacher was downloaded to authorized existing NVMe.
Prepare first16 TRAIN prefix-only versus hindsight notes as a separate condition,
with CPU/device forward and cached/replay checks before hardware generation.
Hindsight supervision has TRAIN future access, always disclosed. Prefix-only
student and fresh-control evaluation remain necessary. No paid teacher or B0
switch. Masked CE/short ZeRO helpers are ready but the learning recipe/data must
follow measured availability, not the existence of new infrastructure.

Padding diagnosis passed both signs/empty/real89-action/full1.72B gradients;
record the isolated warm improvement, not a distributed throughput claim.

### 2026-10-17 23:50 UTC — learn the writer against a fixed low-tax reader

Prepending the SAME2048 TRAIN candidates yields K16 best16−empty+1.026,
with max-own−max-unrelated+.535 CI[+.325,+.749] and max-own−max-shuffled
+.273 CI[+.088,+.466]. Average own−empty remains−1.072. These are TRAIN
oracle statistics, not a usable prefix-only policy. The boundary repair exposes
a candidate signal worth learning without claiming the present writer helps.

Choose a short positive-only writer behavior-cloning experiment, rather than
joint reader training or another RL run. At each TRAIN doc select highest own
K16 first, then require a nonempty strict win over raw, same-empty and each
same-index content control.77/128 remain, all128 decisions saved. Preserve the
older31-label reanchor dataset as a separate unused artifact. Future labels
serve TRAIN selection only; actual writer tensors contain header/prefix/open/
selected note/canonical close, no true future or reanchor, verified by replacing
every future. Supervise note+synthetic canonical close only at T1/full support.

Release64updates/global8/LR1e-5/FP32 from actual B0 with eight-rank ZeRO,
complete16/64 states, finite6h deadline and monitored first actual update.
Then ordinary future16 CE control from B0 on identical selected IDs/order/
updates/optimizer, reporting unequal CE-token counts and compute. Primary
evaluation uses fresh G1 DEV64 notes with unchanged B0 prepend reader; adapted
reader diagonals/crosses are secondary. Selected TRAIN utility cannot pass it.

Separately release the local Qwen3-4B TRAIN16 prefix-only/hindsight pilot after
CPU reference preparation. Both native/device forward checks pass before any
generation (max.000447). First actual replay gates and all32 notes must close
before content-control scoring. This remains teacher-assisted label evidence.

The first eight-rank launch failed at the initial rendezvous before updates:
the installed list parser correctly remapped ranks, but the sparse collective
device graph had no supported path. Preserve V1 exit1/log/run.json/zero-update
receipt. V2 changes only launch placement to contiguous aligned cores8-15 on
adjacent chips2/3; source/data/pair protocol are identical. The official
[runtime developer guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/nrt-developer-guide.html)
also requires a contiguous resulting core range. Actual V2 execution remains
to be checked; no source-level acceptance is called a hardware pass.

### 2026-10-18 00:03 UTC — distinguish informative labels from a useful writer

Fresh prepend DEV64 B0 baseline closed384 finite rows and exactly reproduces
raw vectors. Empty tax is near zero; ordinary G1 K16 own−empty remains−1.101
CI[−1.728,−.567]. Both32-note teacher generation and176-row B0 reader stage
closed without failures, passing native/device and replay gates. Prefix-only
4B teacher notes remain negative (own−empty−.698). Hindsight notes gain+2.858
CI[+.730,+5.309], with positive matched-control intervals. They see the target;
this establishes target-dependent label information, not deployed reasoning.
Exact4/8gram overlap and all fixed examples are retained. No-copy groups can
still encode future-derived hints, and actual examples include wrong math.
Do not treat continuation likelihood as mathematical correctness. Audit anchor
quality locally before another larger teacher dataset; keep current runs fixed.

The contiguous repair passes the previously failing rendezvous and continues
live first-shape compilation. It consumes only two chips. Release raw_control
concurrently on an isomorphic adjacent two-chip group16-23 at00:01:13, same
immutable pair protocol/source/data/order/optimizer/updates. This prospectively
supersedes earlier local serial/same-core scheduling; keep actual shared-host
compilation/cache contention in costs. Neither first update is claimed yet.
Use completed64 checkpoints for fresh G1, unchanged B0 prepend reader and the
prospectively saved K16 primary/K1 secondary analysis; no best-checkpoint choice.
