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

### 2026-10-18 00:28 UTC — preserve failed parallel control, test the known group serially

Writer updates1-16 pass finite gradients, real sampled weight changes and
sampled-slice replica agreement. First two updates include995/568s cold costs;
later updates take25-27s. Keep the healthy64-update protocol unchanged. Checkpoint16
closed00:26:52; all9 files/27.53GB SHA-readback verified00:28:00. No DEV
selection at16.

Parallel raw_control failed during update2 initialized-optimizer loading after
one complete update. Exact shared NEFF keys match the writer's successful set.
Kernel logs establish eight64MiB contiguous HBM scratchpad allocation failures
on devices4/5. No contemporaneous HBM usage establishes exhaustion, fragmentation
or concurrency as the cause. Preserve the complete failed attempt and separate
kernel/closure annex; all12 observed PIDs are gone. No partial state is a result.
After writer64/exit0/group closure, restart raw_control from actual B0 on exact
writer cores8-15, same immutable source/data/order/settings and a new attempt ID.
This is a serial placement test, not a silent numerical or allocator-policy fix.
Keep actual shared-host/cold/export costs separate from warm step timing.

### 2026-10-18 01:08 UTC — first learned writer remains harmful; repair control allocation only

Writer64 completed/exit0/full27.53GB EBS verified. Fresh G1 DEV64/B0-reader
own−empty−.649 CI[−1.117,−.176], own−other−.636 CI[−1.129,−.136];63/64cap.
Paired change versus B0+.451 CI[−.150,+1.128]. This does not pass useful-content
criterion. Preserve all notes/controls/absolute LL/examples and finish raw64,
reader-cross2×2 and unchanged-parser GSM8K TRAIN128 correctness diagnostics.
Do not equate lower cloning loss or future-selected oracle utility with success.

Serial raw-v2 reproduced contiguous scratchpad growth failure without another
accelerator job, after exactly the same first update as v1. Current5s samples
last showed9.60GiB/24GiB perbank; exact instantaneous layout remains unknown.
Official Neuron memory guidance recommends NEURON_RT_CONTIGUOUS_SCRATCHPAD=0
for failed contiguous growth despite available HBM. Release v3 from B0 with
ONLY this allocator switch, same sources/NEFFs/FP32/LRU2/math, explicit parent
protocol and new canonical spec. Old runtime_policy excludes extra key; root
launcher and actual parent+8worker environment attest it. First2 updates now
pass; step1 matches prior numerics. Sampled max23.326GiB includes3.478GiB
private scratchpad, so long-run/transient headroom is not declared proven.
No blind retry or compiler/page-size change. Final64 and backup remain required.

### 2026-10-18 01:18 UTC — test useful calculation before more writer training

Both adapted-reader crosses are negative; raw K16 drift−3.887 nats. The fixed
first8 selected TRAIN labels contain forum fragments, repetition and document
continuations, not convincing worked calculations. This panel is posthoc and
not a population-frequency estimate. Do not enlarge the same cloning recipe.
Use32 hash-selected new GSM8K TRAIN questions, disjoint from earlier dev/demos,
without B0-failure selection. Compare native local4B prefix-only notes with
independently verified partial calculations and matched numeric corruptions;
fixed B0 prepend reader, silence/content controls, answer LL and correctness.
Freeze all notes before gold. No training, official test, paid teacher, or
FineMath-transfer/self-discovery claim. Preserve all32, including weak notes.

### 2026-10-18 01:29 UTC — natural calculation position for the new QA pilot

Before any reader preparation, model outcome or gold access, change this pilot's
reader serialization to complete four-shot QA + newline + note + newline. Raw
remains complete QA; empty preserves both newlines. The original prototype
would place the calculation before the instruction and four unrelated demos,
far from the question. Root identified this concern and independent review
recommended ordinary partial-CoT continuation as the stronger feasibility test.
Use one interface and the same ten frozen conditions; do not expand a matrix
after results. Teacher source, native prompt, question selection and notes are
unchanged. Accuracy remains distinct from forced canonical-number likelihood.

### 2026-10-18 01:52 UTC — completed matched study remains negative

The raw control completed 64 updates with the documented allocator repair; all
9 terminal files are verified on EBS. Its exact 512 occurrences match the writer.
Fresh DEV64 own−empty is −0.956 for raw versus −0.649 for BC. The paired change
of +0.307 has CI [−0.209,+0.908]. Independent recalculation matches all 330 means
and 100 CIs. No useful-content improvement established. Preserve this negative
study, all generated notes, both failed attempts and actual costs.

All nine GSM8K TRAIN128 cells are closed. B0 direct/empty/thought256 =113/107/63;
BC =92/92/61; raw =56/112/76. Every thought mode loses to its same-model empty
wrapper. Raw direct has 59 format errors, 57 without hash4. Keep the frozen
parser and avoid interpreting the score as isolated arithmetic ability. Fixed
first-four examples remain unselected; a separate posthoc first-format-error
panel documents mechanism without rescoring.

The new 32-question QA pilot continues. Its teacher includes final answers in
the first four notes despite the omission instruction. Keep those outputs and
distinguish copying assistance from verified intermediate-calculation utility
before choosing further training.

### 2026-10-18 02:07 UTC — useful external calculations justify a short self-note probe

The TRAIN32 pilot closed all320 answers and320 finite numeric-LL rows; separate
independent audits exactly reproduce every mean/status and all11 paired CIs.
Verified partial notes score30/32 versus empty27/32, CI[0,7] additional correct;
forced canonical-number LL improves+.945[.258,1.591] nats. Corrupted notes5/32
show content sensitivity and vulnerability, separately from improvement over
empty. Teacher29/32 includes final answers in the fixedfirst4; do not call that
intermediate-only assistance. Source, outputs and closures are verified on EBS.

Next, elicit notes from unchanged B0 using a separate plain3shot calculation
prompt with disjoint TRAIN0/1/2 partial demonstrations, G4/T.7/cap64, deterministic
first-newline/EOS/cap stopping, all original actions retained. Consume each note
after the locked reader Answer: with cap192. Reuse the exact closed raw256
baseline; also empty/discard192, shuffled and unrelated controls. Primary is
mean single-sampled-note policy accuracy minus direct256; average all4 samples
within each of32 questions before bootstrap. Same known questions are explicit
exploratory reuse. No gold-selected sample, model update, teacher or official
TEST. Equal maximum generated-token budgets do not establish equal compute.
Require focused/full CPU checks and actual native/device/replay qualification
before interpreting results; preserve every capped/empty/malformed output.

Posthoc exhaustive discordance inspection narrows the accuracy interpretation:
all3 verified-over-empty wins are2 missing-hash4 repairs and1 avoided cap. No
wrong parsed answer was corrected in that contrast. A separate5-item panel also
retains teacher’s wrong48-hour answer being copied and the verified reader
miscomputing6+22 as26. Keep original reports, strict scores and fixedfirst4
unchanged; publish the transparent posthoc panel separately. Positive forced
numeric LL is distinct from a demonstrated arithmetic-capability gain.

The second shared verified/empty failure exposed a manual-note flaw: carnations
for14 friends were priced as14 singles, missing the additional dozen discount.
Canonical TRAIN counts six dozen+two singles for25; all original readers answer27.
Arithmetic checks passed, but the note did not justify the purchase assumption.
Qualify the historical mode as arithmetic-verified, preserve frozen inputs and
strict scores, and extend the separately versioned posthoc panel to all6 cases
in the discordance-or-verified-failure union. Its originalv1 remains in Git/EBS.
This does not change any active self-note input, source, target or analysis rule.


### 2026-10-18 02:49 UTC — short self-notes fail; test real verified arithmetic

The unchanged-B0 probe closed all128 notes,416 fresh answers and416 finite LL
rows. Mean single-note policy accuracy is67.19%, versus75% direct256 and75%
empty192; own−direct−7.81pp CI[−18.75,+2.34]. NumericLL differences from both
controls are also negative with intervals crossing zero. Beating corrupted text
is not success. Independent audits reproduce every row/mean/CI/source binding;
all original processes are closed and210-file EBS evidence is verified.
Do not scale this policy into long RL or treat final-answer notes as partials.

The next data identity targets exact numerical calculations inside real FineMath
TRAIN/DEV. Hash-select one eligible causal '=' boundary per document before RHS
quality checks, without replacement on rejection. Prefix eligibility rejects
already-visible answers and requires a proper computable subexpression. Safe
exact arithmetic supplies partial labels, with explicit external-supervision
status; the full source result must independently agree. Preserve unsupported,
false-source and alignment rejections separately, exact source tokens and split
identity. Fixed TRAIN256/DEV64 feasibility pools precede model scores. No full
corpus prep until source/fixture review, no training until actual data and reader
feasibility are understood. Future raw control gets full ordinary source CE on
identical accepted chunks. This repairs a concrete data/reward weakness without
converting external labels or oracle measurements into self-discovery claims.


### 2026-10-18 03:02 UTC — causal arithmetic selector and full CPU preparation

Before any corpus preparation or model score, replace the prototype minhash over
all eligible cuts with a stopping-time selector: fixed hash-derived start128..192,
then immediate return at the first prefix-eligible '=' cut. Full-window minhash
was legitimate offline curation, but later candidates could inspect text beyond
the selected cut. The final rule matches the intended causal selection claim.
Chosen-RHS quality failure never triggers another boundary. Sentinel-suffix and
no-fallback tests cover this distinction. Independent review accepts exactsource
177363c6…6ee6af;87 focused tests pass.

Partial labels skip unchanged rational identities, use exact finite decimals
where short, and retain original source spelling/tokens. Numeric source mismatch
is separate from unsupported notation and can reflect a rounding convention.
Reader primary excludes pure-whitespace target tokens; joint scalar and natural16
likelihoods remain separate. Four real CPU/device reference contexts include raw
and correct-note inputs; no source mutation or reduced vocabulary. Reviewer
accepts probe7913b851…b6f36f;20 focused tests pass. Final599 required tests,
toy smoke and compileall pass at local/checks/tests.sBcbKy.

Release one full CPU scan under a finite04:00UTC bound. No accelerator/model or
training release yet; require actual accepted counts and examples first. Source
RHS validation curates a mathematical subset, not unrestricted FineMath utility.
A subsequent learning pilot, if warranted, should add note BC to full-source CE
versus the same source CE alone, with fresh generated-note and downstream tests.


### 2026-10-18 03:26 UTC — repair visible-answer filter before any model outcomes

V1 closed1304 exact numeric examples (TRAIN1220/DEV84). Full arithmetic/source
alignment passed, but root's fixed first4 panel exposed Rs.132600 already in the
writer prefix: the copy filter interpreted .132600 as a decimal. Independent
maximal-span review found3 confirmed misses, all TRAIN, one selected. This shows
why arithmetic/byte conformance does not certify the intended data condition.
Keep completev1, original source/tests and every diagnosis unchanged. No model
was prepared or scored on it. Curatorv2 repairs contextual dots and malformed
maximal comma spans, preserving source salt and causal stopping-time selection.
The repaired source/fixtures passed122 focused tests and independent review;
release fresh CPUv2 under05:00UTC bound, no model launch yet.

Before any model outcome, add384 greedy immediate scalar completions on the same
64DEV×six modes, cap32, alongside1920 source-likelihood cells. Compare complete
initial values exactly; equivalent8.3/2 and4.15 count alike, unlike literal source
NLL. Unsupported/incomplete/error paths stay zero in all denominators. All actual
actions/stops/costs retained. Parser scope and EOS/cap completeness rules are fixed
in source; generation has tokenizer-defined support, LL/native full-head support.
Source review and100 focused tests pass; final714+toy_smoke+compileall pass at
local/checks/tests.zL7PFQ. No hardware or capability result is implied.

### 2026-10-18 03:35 UTC — keep any next warmstart causally separated

While the corrected CPU scan runs, add only a reviewed CPU packer, without a
trainer or learning release. Note labels use the causal prefix and exact proper
intermediate, with the native newline encoded jointly. Mutating or removing
all source-future/RHS fields leaves every writer tensor unchanged. The separate
ordinary-source component supervises all527 next-token targets of the original
528 tokens, including actual EOS tokens, rather than the old future16 control.
An eventual treatment adds separately averaged note CE to the same full-source
CE as its control; that adds labels and compute and is not a FLOPs-matched study.
Packer25 focused tests and independent exact-source review pass; required739
checks, toy smoke and compileall pass at local/checks/tests.WLpLnu. Dataset
publication/order, optimizer and fresh generated-note evaluation remain to be
fixed only after actual reader feasibility. No training has started.

### 2026-10-18 03:44 UTC — qualified corrected data; release actual reader probe

V2 closed all347895 TRAIN/DEV audits, accepted996TRAIN/68DEV. Independent
arithmetic/source/token/pool checks pass. Six leading-comma scanner flags were
ordinary prose/index delimiters or distinct-value TeX notation; preserve the
initial flagged result and separate manual resolutions. No dataset/source edits.
Root inspected all first4 prefixes per split. The claim remains exact local
arithmetic and bounded literal-copy coverage, not trustworthy surrounding prose.
V1 and V2 complete data/evidence have separate SHA-readback EBS archives.

Seal prospective TRAIN512 IDs before any model outcomes, without filtering.
Exact formula overlap affects6/64DEV; disclose it and retain the fixed panel.
Release finite prepare/nativeCPU/four LL/four completion stages with actual
qualification and physical cleanup between stages, no automatic retry. No
training release. Any later arithmetic evaluation keeps its exact writer prompt;
GSM uses the unchanged qualified question-only prompt across all arms and is
explicitly a prompt/domain transfer check.


### 2026-10-18 04:07 UTC — weak external-note result; repair measurement and placement

All1920 LL cells and384 free attempts closed with0 runtime errors, all8 native
and cached gates passed. TRAIN numeric correct−raw−.057nats; DEV+.091 with
interval crossing zero. The declared scalar parser scores27correct/23raw/24empty,
but199/384 are unsupported. Complete numerical lines can be falsely rejected by
later prose ('I' versus 'Answer') or equality. Preserve those original scores;
they are not mathematical capability estimates.

All primary discordances are retained and read. The1/26 gain is parser-only;
3310201/331921→331777 is a real initial-number repair despite the old unsupported
label. Key numeric repairs also occur under the corrupted notes; no proof they
depend on correct intermediate values. No parsed changed-counterfactual output
matches. All source/outputs/control failures are independently checked and EBS
archived before any source integration; public result projection accepted.

Do not scale this into long RL. Next repair the target-independent scalar
completion boundary, retaining old endpoint/results. Any reinterpretation of old
paths is explicitly exploratory. Then isolate note placement/reanchoring on the
same fixed documents and single-step labels, with raw and matched-empty controls.
Do not combine a stronger multi-step external oracle or shrink to current failures.
No next hardware or learning release yet. Ordinary downstream checks remain a
requirement of any later learning pilot, regardless of local likelihood sign.


### 2026-10-18 04:14 UTC — freeze the measurement repair and isolate placement

Preserve the complete old run/source and published result at ae43438 before source
integration. The reviewed new scalar parser certifies local boundaries rather
than arbitrary later prose; arithmetic/grouping/scale ambiguity remains explicit.
Its retrospective audit of old outputs is exploratory:31raw/32empty/35correct,
with131unsupported still retained. Original23/24/27 endpoint remains unchanged.
No claim that the new grammar measures every mathematically equivalent answer.

The next interface repeats only the original LHS-through-equals bytes after the
same note, now following the past. Empty has identical separators/reanchor; raw
unchanged. All donor/shuffle/note assignments and320fixed documents retained.
Real-data CPU check verifies all1920 contexts, maximum549+16 tokens, byte-exact
reanchors and no future fields. Exact source reviews accepted;419 focused and
1058required+toy_smoke+compileall pass. Next prepare explicitly selects both the
interface and parser version, followed by fresh actual CPU/device qualification.
No additional multi-step oracle, train subset filter or learning release.


### 2026-10-18 — owner redirects to full-span interface batch and larger warmstart

The512-document joint draft was superseded before any accelerator qualification
or training. Its reviewed code and CPU data remain preserved. First batch all six
reader interfaces with target-copy/symbolic full-result oracles and controls on
at least256 DEV documents. Likelihood is primary; no per-condition release cycle.
Native tokenizer inspection confirms separate digit tokens. K1 can also be only
leading whitespace, so score the complete target plus closing boundary instead.
Near-LHS partial notes show18 parsed corrupted-counterfactual matches, correcting
the claim that the reader universally ignores notes; net assistance is still absent.
Next learning data must contain>=5000 real examples, with held DEV>=256, joint
writer/reader supervision, BF16 compute/FP32 masters and explicit extra compute.
Zero-weight mixing guarantees zero tax only; negative thinking-augmented CE is
not a proof that every RL approach is impossible. Both remain empirical tests.
Published RLP prompt text is separately attributed under its CC BY-SA4.0 license;
the two closing delimiters in its figure are preserved, not silently repaired.

### 2026-10-18 — Full-span interface question closed; fit scalar mixing next
All six interfaces and controls completed on256+256DEV documents. I3 re-anchoring
strongly enables literal future copying in general text (+15.364nats), while its
empty tax remains−3.007. Arithmetic I3/I5 positive estimates have intervals crossing0;
other arithmetic interfaces are strongly negative. No declared gate passes. The
complete180-CI independent review passes; all cells and verified EBS evidence remain.
Next fit one scalar I3 logit-mixture weight on separate first256uniqueTRAIN records,
fixed nine-weight grid and TRAIN empty-tax constraint−.4; evaluate once on the same
complete DEV panel. This is supplied-target plumbing, not a useful writer result.

### 2026-10-18 — Scalar mixing qualifies general copying; arithmetic reader-learning fallback
I6 w=3/8 is selected on separate TRAIN256 by the frozen nine-point constrained fit.
GENERAL meets all three unchanged gates; ARITHMETIC fails +2 gain/sensitivity.
All3840TRAIN/DEV rows, eight native gates, w0 identities and30 document CIs are
independently verified and archived. Report numeric-only headroom/gains separately;
a wrong note can also improve relative to raw. DEV is reused exploratory evidence.
Use underlying I3 for ordinary differentiable supervised reader training, explicitly
as the permitted reader-learning fallback; do not label that objective I6 training.
Fixed I6 is an additional evaluation control. No fresh-generated utility yet.
New TRAIN-only candidate identities prioritize real compound equations, then true
one-op reserves; exact postorder Fraction substeps replace full-answer-only labels
under a new label identity. Repeated real TRAIN documents may contribute distinct
source cuts: the user's quota is5000examples, not5000unique documents. No repeated
verbalizations. C sees the same source occurrences; document count/exposure reported.
The frozen-reader teacher>empty filter and5120 quota remain unchanged; no shortfall
shrink. Qualification uses a separately declared first64 subset, never a science arm.
The fixed empty profiler timed out during cold compilation with zero complete cases;
no warmed or BF16 throughput claim and no automatic retry. Prioritize supervised
qualification;<=90s complete-step performance remains mandatory before long RL.

### 2026-10-18 — Frozen-reader filter shortfall and stronger plain control
The complete14669-candidate filter yields3147strictly positive full-scalar+boundary
notes (21.45%),1973short of5120. All44007scores/native checks and physical closure
pass; complete EBS archive is verified. No training IDs/dataset were emitted.
Preserve this result while auditing extra true source cuts and token-level effects;
do not lower the threshold or manufacture examples through repeated verbalizations.
Current W/C/T two-update jobs remain unchanged hardware qualifications. Before
science, use C′=raw-next16mean+.5source mean as the main plain control: source-only
C gives far less weight to the anchor than W and confounds the comparison. The
new objective is exactly one source pass with explicit token weights, checked
against dense loss/gradients; integration/hardware qualification remains future work.
W still adds writer supervision and compute; within-reader own−raw/empty/other and
crossed readers remain necessary. No scientific SFT or RL release follows this change.

### 2026-10-18 — Explicit reader-learning pool, without frozen-reader filtering
The closed filter's teacher−empty mean−1.277nat decomposes exactly into numerical
+.380, closing-certificate−1.645 and other−.012. Teacher−raw number-only gain is
+.160, but joint fullspan−1.809 and natural16−4.306: no net assistance claim.
Certificates can contain words as well as punctuation; do not rename that loss.
Requiring the untrained reader to benefit already impedes the permitted reader-
learning fallback. Therefore declare a NEW experiment: first5120 compound
representative TRAIN documents, original frozen order, exact>=2-op proofs and
native notes<=128tokens, WITHOUT likelihood selection.5438are eligible. Preserve
the old3147/14669shortfall and every score unchanged; this is an informed design
adaptation, not a prospective success under the original gate. HeldDEV and GSM
controls remain required; own−raw/other must establish utility after supervision.
Prepare a separate manifest/source identity; active hardware programs stay frozen.

### 2026-10-18 — Hardware memory repair preserves FP32 optimizer masters
World8/accum4 with FP32 model/gradient storage fails W before update1 and C after
update1; T completes2updates and all full-state checks. Every group/device is
closed, source snapshots retained; this is an engineering failure, not a learning
result. W's24GiB bank contains13.620GiB tensors,3GiB scratch and6.589GiB spill rings.
The released310parameter tensors are already BF16. A new parameter-only CPU cast
can preserve exact initial values while retaining FP32 RoPE buffers; whole-module
dtype conversion would round those buffers. Use supported ZeRO FP32 masters/moments,
not private manual master seeding. BF16 gradients/collectives must be reported.
Prefer world32/accum1 for the new qualification; do not claim fit from arithmetic.
Measure actual FP32 master updates, since early BF16 copies can round unchanged.
Saved model compute copies widen for FP32 evaluation; FP32 masters remain necessary
for recovery and must not be silently substituted as a different evaluation model.
New CPU BF16-parameter reference and native distribution checks are prospective;
the old FP32<=.002 gate is not BF16 arithmetic equivalence. No scientific run released.

### 2026-10-18 — New storage-precision qualification and fixed note-prefix ablation
Freeze a new BF16-parameter/FP32-rotary CPU reference before hardware. Gate both
directions of independently FP64-normalized KL<=.01nat for each full-head native
context; log tail errors without claiming rare-target or gradient equivalence.
Three qualification updates cover initialized Adam state and the next non-aliased
warmup LR. Initial master samples, actual master deltas, rounded active copies,
replica agreement and full moments after updates2/3 must pass. No resumption or
steady throughput follows from these tests. The real5120-example science quota
and held/control endpoints remain unchanged.

The first six fixed-order saved B0 notes sometimes compute the right number then
continue into unrelated article text. Declare a separate exploratory ablation on
the same256DEV documents: keep the exact native token prefix through the first
completed nonblank content line, capped at64tokens. The policy receives only
cached note IDs and tokenizer, never source, answer or correctness. Preserve
whole boundary tokens, intact cyclic donors, full/raw/empty controls and original
generation costs. Require positive paired lower95% bounds against BOTH raw and
short unrelated notes. This outcome-informed probe cannot silently change the
primary128-action writer policy, training data or an old result.

### 2026-10-18 — Preserve failed CPU-BF16 reference; use a high-precision golden reference
The v2 W qualification fails before updates: one of four contexts exceeds the
declared .01 KL bound (.010121); all32saved native vectors are finite and replicas
are byte exact. Asymmetric local failure also strands passing ranks before a
later rendezvous. Preserve FAIL, all vectors and physical closure; exact owned
group TERM was needed, with no KILL or effect on the concurrent GSM evaluation.

Three-way comparison on the identical four contexts finds device BF16 within
.003103 of CPU FP32, whereas CPU BF16 itself differs by .014382. Released BF16
weights widen exactly; CPU BF16 intermediate rounding is not the accuracy target.
Declare a new CPU-FP32 golden-reference identity with the SAME .01 bidirectional
KL bound and unchanged BF16 training/storage. This repair is informed by a failed
engineering check, not a prospective pass of the old gate. Persist every rank's
native receipt, collectively agree on failed ranks, and raise uniformly before
ZeRO initialization. Forward sanity still does not establish gradient equivalence.
No training or long-run release follows without actual new hardware qualification.

The closed first-line ablation is independently verified negative in both paths:
shorter notes reduce damage but remain worse than raw and short unrelated notes.
No main-policy change. Existing genuinely general TRAIN scores contain296unique
anchors with B0 K1 surprisal>=1.7nats and no held overlap. Prepare a separately
declared256-document hindsight-label pilot before expanding beyond arithmetic;
it measures teacher-label feasibility, never prefix-only deployment utility.

### 2026-10-18 — General hindsight labels show content signal; plan larger TRAIN-only pool
The complete256-document general pilot has no errors. Under fixedI6, teacher
notes beat raw by+.495nats, empty by+.660 and intact unrelated notes by+.794;
all paired lower95% bounds are positive. OrdinaryI3 loses on average but82/256
notes have strictly positive teacher−empty gain. Independent full-vector, interval,
copying and action-cost audits match. No extracted full16copy, but10notes retain
an8-token future overlap; paraphrased hints and future-aware supervision are explicit.
This supports label feasibility, not a deployed prefix-only policy result.

Keep compound5120 unchanged. Prospectively prepare a separate larger pool:
131072hash-ordered distinct approvedTRAIN anchors, first24576with frozenB0 K1
surprisal>=1.7, same one-teacher/cap64/first-line policy, full reader controls,
and first5120eligible STRICTI3teacher−empty positives. Keep genuine quota shortfall;
no repeated verbalizations or threshold relaxation. These pool sizes provide
margin over the pilot's32% acceptance estimate, without guaranteeing availability.
CPU availability, exact eligibility and timing/storage still need declaration;
consider16independent inference workers on four unshared physical devices to
use the existing node better. No new model launch or main-training change yet.

### 2026-10-18 — General label quota passes; keep reader and writer claims separate
The fixed larger batch supplies8,964eligible strictI3-positive notes; retain its
original first5,120 without ranking or changing the copy/plain-text filter.
Independent all-cell, note, selection, interval and native tensor checks pass.
The whole-pool ordinary reader still loses2.840nats against raw; fixed mixing
improves by.831. Future-aware labels establish training feasibility, not useful
prefix-only thoughts. Retain166selected exact demo imitations as observed label
noise under the prospective filter, rather than silently cleaning after scores.
The isolated native-v5 adapter and actual all5120causal packs are CPU ready;
hardware qualification/training need a fresh resource-specific release.

An optional read-only inventory found only78unexposed compound DEV documents
among existing accepted publications, below256. Count the entire2,048-document
general raw-scoring pool as exposed, not only its selected256. Do not promote
this shortfall to a fresh cohort or substitute TRAIN. Existing real held panels
and the frozen synthetic mechanism panel remain the terminal comparisons.

### 2026-10-18 — Repair C′ host startup without changing its learning recipe
The first matched C′ launch exits before native checks or updates because its
privateRAM TMPDIR plus Python's socket suffix exceeds Linux's107-byte AF_UNIX
limit. All original identities/devices close and the failed attempt is archived.
A separately reviewed/released r3 uses a short unique0700RAM path and an actual
nested Unix socket-bind guard before heavy imports. Data, order, learner sources,
precision, optimizer and objectives are unchanged. Native32/master and three
actual update checks pass; root full-vector/master-sample readback agrees. Keep
r2 failed and r3 active under distinct runtime identities; no automatic retries.

### 2026-10-18 — Preserve global host-OOM interruption; evaluate saved development64 honestly
Kernel logs confirm global host OOM13:16UTC, direct W-worker and user-session
kills; W stops after220updates, C′119. Neither has a full terminal/resumable state.
All original groups/devices are empty. Both exact step64 model-only snapshots,
run/control/source/log evidence are now EBS-backed with source/readback hashes.
The original320-step comparison stays incomplete. A NEW explicit developmental64
identity may evaluate these matched saved models; do not relax the old terminal
identity or invent missing optimizer state. Profile bounded host-memory growth
before restarting long learning. Periodic full checkpoints and a memory-triggered
cooperative STOP are necessary for the next operational recipe. No auto-relaunch.


## 2026-10-18 — intermediate-only computation channel and completed development64

- Owner review supersedes broad general-hindsight learning as the next priority.
  Teacher usefulness does not establish prefix-only attainability; full-answer
  notes do not isolate intermediate computation. Do not scale original RL.
- Finish both preserved matched64 evaluations under a separately declared
  developmental identity; original terminal320 remains incomplete. Both fresh
  notes and B0/self reader crosses are retained regardless of sign.
- Next narrow study: exact prefix-derived partial states excluding root/final
  answer, reader training then freezing, supervised prefix-only writer with stop,
  and the same fixed-reader teacher/student/empty/unrelated/restatement/corrupt
  cross. Ordinary compact CoT uses the same traces and may be the preferred result.
- First128 TRAIN overfit is a mechanics check. Fresh structural/range/composition
  DEV separate generalization questions; complete-answer accuracy is primary for
  this new study, with errors retained, note correctness and measured costs.
- Before long learning, isolate host memory, qualify a bounded complete update
  loop, and add periodic resumable states with an actual restart. Four-condition
  backward-only allocator probe is bounded and makes no full-loop qualification.

2026-10-18 17:52UTC — The closed original-loop2×2 reproduces live host-allocation
growth with both constant/scheduledLR and full/reduced sampled diagnostics;
these are not sufficient fixes. Exactly one rank per adjacent pair grows rapidly.
Flat live tensor counts but repeated compile handles under LRU2 motivate one
explicit world4/global4 cache2/cache256 causal probe, with native/finite gates
and no science model reuse. No runtime/stack change or long learning release.

2026-10-18 18:04UTC — Short-pack mechanics independently pass eight native/master
witnesses, full3/6 raw-state/cast audits, exact actual restore and subsequent
updates. Merge the additive qualified partial-state source to main; active
baseline retains its immutable worktree. Start one complete896-cell TRAIN128
B0 answer characterization with intermediate-only controls, separate offline
gold, exact parser and actual cached-prefill checks. Its outcome cannot block
reader training; it is not held-out or student-utility evidence.

2026-10-18 18:32UTC — LRU256 removes all94warm-tail reloads, preserves every
component loss/global norm exactly, and cuts43.466s to1.925s onworld4/global4.
Both small-topology tails are bounded; do not call this an originalworld32 OOM
repair. Reader128 then completes at~.73s/update with all four full states and
physical closure; freeze its terminal model before fresh writer128. Run ordinary
compactCoT128 independently on the same traces, allowing it to win. These are
TRAIN fitting diagnostics with two-run split training cost, not capability claims.

2026-10-18 19:21UTC — Reader/writer/compactCoT128 all finish, all terminal
active/master/moment bytes audited, frozen weights/nonweight evidence EBS verified.
CPU writer actually emits all128 exact partial states with naturalEOS; fresh
generalization remains unmeasured. All CPU own/other reader contexts equal the
already scheduled teacher/other_teacher controls: verify executed prompt bindings
and label any response reuse explicitly, without extra independent samples.
Trainium prefix and complete-answer matrices remain active. Next-study ordinary
CoT may include the full root calculation; account for two split models/runs.

2026-10-18 19:42UTC — All2,688 TRAIN128 fitting answers close and independently
re-parse. Reader teacher128 exceeds direct120/empty119; actual CPU writer notes
match every executed teacher/unrelated context, so response composition is
explicit reuse, not new inference or extra samples. Ordinary compactCoT128 also
gets128/128 with one training model/run; split has no demonstrated advantage.
Evaluate the fixed four512DEV panels before scaling training, with no outcome
filtering and separate interpolation/structure/range/composition claims. Preserve
all old128fit models and source snapshots. Official TEST remains unused.

2026-10-18 19:58UTC — Actual Trainium writer closes with all128canonical states
and naturalEOS; every action matchesCPU, including intact donors. Frozen-reader
conditional responses may be reused only with explicit exact-input provenance;
composed128/1 does not add inference or samples. CompactCoT remains128/128 with
one training model/run. Fresh fixed DEV writer/CoT sequences run at3301d22 on
physical8/9, bounded01:30UTC Oct19. Keep ordinary partial-CoT as a real cost
competitor when considering the stronger full-trace training variant; extra root
steps may increase answer accuracy and also add inference tokens. No larger
learning or RL release yet.

2026-10-18 20:16UTC — Before reviewing any fresh DEV outcomes, revise the
unreleased larger budget to A640/Bpartial640 versus reader320+writer320: each
system gets640total updates/20,480ordered expressions. A/B each see four passes,
R/W each two; actual supervised labels/chip-time/capacity remain explicit. The
full-root CoT variant is prepared but cannot displace the compact cost baseline.
A320in the first CPU proposal was never launched. The currently active128-model
DEV question retains its original80-action budget/settings; no new learning.

2026-10-18 20:40UTC — Owner asks for the north star, causal steps, papers and
7–8B consideration. Finish the frozen teacher/student/CoT question first; its
failed link chooses the next intervention. Official Qwen8B metadata independently
counts8,190,735,360parameters: BF16 weights+gradients30.51GiB exceed actual24GiB
logical banks. Replicated full SFT cannot reuse the present layout unchanged;
inference/adapters/sharding remain unqualified alternatives. No weights downloaded
or larger run launched. Proof ladder and primary-paper corrections updated;
capacity metadata/lower bounds preserved, with no forward or efficacy claim.

2026-10-18 21:36UTC — All four fresh panels close and independently pass18432
answer/2048note/state/interval/vector audits and full EBS readback. Student correct
285/238/140/151 versus compactCoT287/235/118/48 of512; own beats same-reader raw,
empty and unrelated on every panel. All2048notes stop naturally. This is a useful
fresh synthetic channel, one seed; two-model split received twice the updates.
Keep1.7B for the next varied-data/matched-cost study, with answer-only, compact
partial-CoT and full-root CoT640 each versus reader320+writer320, two seeds.
Full-root CoT is an additional strong baseline, not a relabeling or replacement
of compact CoT. Terminal primary; all four existing DEV panels stay separate and
explicitly developmental. Preserve old negatives. Release first reader only after
source/checks/disk review; full64 save/restart precedes longer continuation.

2026-10-18 22:38UTC — Matched5120 study source57d3220 passes1601checks and is frozen
in a detached worktree. First reader320 completes and independently passes full
64/160/320 state readback, fresh64 restore with all4active/optimizer byte equality,
and genuine ordered65–320; full terminal protected on EBS. Writer320 closes22:29,
independent audit running. No fresh trained DEV outcomes inspected. Three old
engineering EBS duplicate payload sets retired after both copies freshly hash
match; complete originals stay on NVMe, all receipts preserved. Four NVMe alias
pairs already share inodes, so the duplicate check freed0bytes. Do not budget
imaginary102GiB reclamation; preserve unique science with verified lossless archives.

2026-10-18 22:40 UTC — Writer320 independently passes4native gates, all320orders/counters,
full64/160/320 model/master/moment states and full EBS terminal readback. Both R/W
terminals frozen; direct640 starts fresh B0 on physical3, no new DEV review.
Storage investigation finds three unused unformatted local instance SSDs. Initialize
one only after no signatures/partitions/mounts/holders/raw users, and verify a real
write/readback; local/nvme-study adds1.7TiB without cloud changes. Stop unnecessary
compression and restore all41bounded old payloads, every original byte rehashed.
Future baselines use the extra disk; current runs/identities remain in place.

2026-10-18 23:08 UTC — First A640 independently passes640ordered updates, all5full state
publications, native gates and EBS full terminal readback;123368answer labels,
warm median.724s, bounded observed PSS116.8GiB. Its identity gate catches a copied
320digest in the controller's final640receipt. Original receipt/source preserved;
explicit annex corrects the digest and strict640binding passes. Same live compact
S1 template gets an annex after closure; every unstarted640template is corrected.
No training/data changes, no outcome review. Compact640closes, full-CoT640starts.
All2048Wnotes close/EBS readback passes; R/A cache compilation actively progressing.
Keep every original full state, all terminal models and results. EBS retains latest
full resumable state; old EBS optimizer duplicates retire only after fresh both-side
hashes and latest full-state verification. Current retention index is public;
R64/R320/W320 originals intact, models/metadata remain onEBS. A full state protected.

2026-10-18 23:16 UTC — Add one auxiliary unchanged-B0 direct/cot/empty reference before any
new accuracy/state outcome review. The matched four-system/two-seed primary,
training, checkpoints and original design remain fixed. A zero-training reference
is necessary to distinguish a trained-system advantage from improvement over
an adapted/degraded model alone. Single shared B0, not two independent replicas;
6144additional answers, all failures retained, no TEST. Prospective amendment and
registry pin the addition. Independent analysis requires all11controllers closed;
1610CPUchecks pass. Compact640 full state/EBS audit also passes; full-CoT640active.

2026-10-18 23:39 UTC — All5first-seed training models pass independent full updates/state/
native/backup audits and are frozen. Full-CoT640labels941956,warm.728s, observed
bounded PSS121.6GiB; original640closure correct. Second reader320 also passes,
full terminal protected/frozen; second writer320active. All data/source/primary
endpoints fixed, no new accuracy/state review. W1notes/A1answers fully archived;
compact/full evaluations advance while longer note-cache compile actively runs.
First-seed temporary path ps7r2 belongs to prior recovery; use fresh p7s2r/w/a/c/f
for second seed, repaired before any scientific launch. No actual attempt retried.

2026-10-19 00:17 UTC — S1allmodel generation questions close; all output/control/numerical
evidence EBS readback passes, no outcome review. S2W320/A640 full audits/frozen
bindings pass and evals launch; compact640closes, final full-CoT640runs under
finite no-retry queue. Read-only host logs show flat update intervals and~20GiB
retained at later full-save boundaries. Boundary includes inspection/transfers/
serialization/allocator activity; no causal attribution or fix. Public observation
preserved; qualify exporter before much longer loops. Frozen science unchanged.

2026-10-19 00:32 UTC — Allten matched training runs complete; terminal states/ordered
updates/native witnesses/full optimizer masters audited, protected and frozen.
No new math outcomes reviewed. Final full-CoT2eval preparation active; finish all11
controllers before independent whole-question review. Separate four-core export
question released with fixed source/conditions/thresholds:actual state shapes,
six sign versions, exact byte readback after memory points, no training steps.
Baseline must reproduce growth; a candidate is not a training/restart qualification.
No science-source or current evaluation changes. Full-CoT2latest EBS recovery protected.

2026-10-19 00:43 UTC — Tensor-only export followup passes all six exact versions,
source and final live-state checks:current4.809GiB/export versus gc/trim0.0018.
Initial four-condition probe failed xm.save metadata type gate (tuple->list),
parent interrupted peers; preserve all partial files and readback, no pass claimed.
Selected two-condition followup has separate plan. Actual SFT control14 versus
candidate12/close/restore/13-14 now released on isolated fb72065,1601CPUchecks.
First SFTpreflight bound main instead of candidate via cwd shadowing PYTHONPATH;
guard rejected before any worker/allocation. New v2asserts actual imported root,
preserves failed config. Immutable57science and all endpoints remain unchanged.

2026-10-19 01:18 UTC — Close export-memory question without rewriting its strict
failure. All14full checkpoints pass independent tensor/cursor/native checks;
post-export grouped PSS49–139GiB control versus21–22candidate. Saved12 states
match fully. Candidate12/restart/14 differs slightly from uninterrupted14;
one explicit unchanged-exporter restart gives the exact same resumed14 model,
all optimizer bytes, RNG and loss/norm. Cleanup preserves matched recovery;
resumption execution has a separate numerical limit. Both reports and all update
metrics/source are public; originals and terminal recovery/EBS evidence preserved.
No further memory probes. Candidate remains outside immutable scientific source.
All10primary evaluations close; only shared B0 final panel remains. No new outcomes opened.

2026-10-19 01:24 UTC — Begin bounded CPU real-data preparation, not a model run.
Prefix-only re-execution finds5448TRAIN/325DEV compound documents, but raw-proof
subset4872/200 misses5120/256. Isolated9d57df7curator adds minimum2prefix operations
before unchanged RHS validation;1617CPUchecks pass, including wrong-answer/TEST
safeguards. Reprocess same347895auditedrawTRAIN/DEVdocuments on16CPUworkers,
deadline02:30,no retry; all TEST/unlisted bytes remain opaque. Later partial labels
must receive decoded prefix only and omit root, not reuse full-answer oracle fields.
No new learning/current-source changes; matched study waits only on shared B0.

2026-10-19 01:45 UTC — All11math generation controllers close; independent full
question audit underway, no partial accuracy readout. CPU real-data preparation
closes:5885TRAIN/334DEVfull-raw compound sources,5807/329nonroot labels fit64actions;
fixed5120/256 selected,0doc overlap,83long labels retained as exclusions.44DEV
expressions seen inTRAIN;276noveleligibleDEVavailable for future protocol choice.
TeacherlabelAPIsees decodedprefixonly, root operation omitted;full traces separate.
All code/data/provenance EBS-readback, no model/learning release. Keep training
value distinct from deployed note value per re-readBoLT/TPT; current study unchanged.

2026-10-19 02:04 UTC — Complete independent matched study:51,200answers/4096notes,
all20payloads and fullEBSreadback. FullCoT higher on all8seed/panelcells; split
advantage over compact baseline does not survive stronger control. Same-reader
content benefit remains real. Posthoc:all5120TRAINwriter targets have2equations;
composition needs3but Woutputs2in501/512 and508/512. Most failures are correct
subsets, allnotes naturally stop. No emptycollapse/numerics excuse; incomplete
compositional execution. Preserve original primary results and clearly marked
posthoc diagnostics. Choose simpler full-trace single-model approach for real
documents; training-value hypothesis stays separate.1610CPUchecks pass. No RLrelease.


2026-10-19 02:22 UTC — Follow owner steering with one bounded curriculum intervention.
Do not equate two-line behavior with proven EOS causality or abandon all thinking.
Hold the unlaunched real-trace wide-context hardware plan; preserve its CPU work.
Freeze fixed640 versus variable640(depths1..5),same160updates,root omitted,frozen
reader,strict complete-answer gate,and untouched B0/fullCoT references. New
expression-disjoint depth6 tests extrapolation; originalcomposition512 is exposed
DEV. Prospective JSON committed5af3431 before learning;1614checks pass. Start only
disposable save/restart mechanics now; scientific pilots await actual audit.
Real-document usefulness stays the second bounded question,not a synthetic-math
claim or justification for another large writer–reader program.


2026-10-19 02:51UTC — Prioritize single-model training benefit as owner reiterates.
Keep the already-declared variable-depth repair bounded and parallel. Actual
width192/full recovery audit passes;release only its two160update fresh writers.
Real-data source3d8666b passes1622CPUchecks,all15360pack views,held novel/ordinary256
are frozen,EBS verified. Start separate8rank width640 qualification on physical12/13;
training/inference gates remain independent and no320update science is released.
Source data has2–6operations,not a constant trace depth. MuSiQueCPUcandidate
initially projected the wrong raw field;pre-model inspection caught templates.
Use composed_question_text,reject placeholders,and preserve superseded V1inventory.

2026-10-19: Before mainline320update learning, ordered exact-value AST matching found
10TRAINaliases missed by V1surface-string novelty. Freeze V2from265eligible AST-novel
rows,retain256,zeroTRAINAST/document overlap. TRAIN,all previous panels and raw B0
observations remain. Reuse246identical AST-panel and256ordinary B0 observations,
execute10new and2calibration items (calibration excluded fromn). Both original gates
unchanged. Publish complete per-document likelihoods and the correction amendment.
Both bounded math writers completed160;all320updates/sixfullstates independently
verified and terminal models/latestfull state protected. Evaluation V1failed its
runtime-policy preflight before any accelerator worker;retain it and prepare V2
with identical CPU-native vectors plus the actual intended runtime declaration.

2026-10-19: V3depth-repair evaluation now runs five separate four-shard jobs
(fixed/variable notes,frozen-reader controls,fullCoT,B0) with two dependent
same-reader own/other jobs waiting on intact note bundles. All20actual native
checks pass;newshapes are compiling. V2also missed explicitPJRT in the preparation
runtime,again caught before any hardware worker;V3compares the entire runtime map
against the intended execution map before preparing protocols. Native vectors
and all projected inputs remain byte-identical;no repeated scientific observations.

2026-10-19: Complete bounded depth-repair audit passes all9,216 answers/2,048notes.
Variable640 writer gives frozenR306/512 composition answers versusfixed640124;
paired+35.5pp[31.3,40.0],canonical/natural91→409. Declaredrepairgatepasses.
It exceeds frozenfullCoT273 on this panel but not freshdepth2(182versus193),and
unseendepth6 remains10/256(teacher78). One seed and different curricula/target
supervision prevent a matched architecture claim. Keep real-data single-model
mainline;do not expand a split rescue from this narrow positive mechanism.

2026-10-19: All six fixed real-data models completed320 updates without restart.
Independent audit passes12 full states,48 native checks and1,920 ordered updates;
all terminal weights/latest full320 are protected and evaluation bindings frozen.
Finish both predeclared likelihood endpoints independently, without extensions.
Execute the already-planned32-action scalar secondary on all seven fixed models,
using exact original notes and unchanged initial-scalar-v2. All unsupported and
failed outputs remain in256 denominators; no new gate or parser adaptation.
The prospective secondary design precedes trained capability review. Its actual
B0 four-core native/cache672 checks pass; each trained model checks itself again.

2026-10-19 07:15UTC — Both fixed real-document readouts close and independently
reproduce all counts/signs/bootstrap intervals. Neither primary likelihood gate
passes:post raw gain~0.01nat;prior generated gain+0.097/−0.139nat versusrawCPT.
The prospectively declared scalar secondary is positive:216/213 of256 versus
193/196,paired+9.0/+6.6pp with positive intervals;all unsupported outputs retained.
Keep both findings:supervised calculation helps this exact-number path using
extra computation,but training-only/full-continuation improvement is unestablished.
Publish all outputs and controls together;no gate relabeling,RL release or8B jump.
Answer-focused direct/full-trace SFT is a missing practical reference,not yet a run.

2026-10-19 08:00UTC — Release exactlyfourfreshB0 target-SFT models after actual
both-mask qualificationPASS:11finiteupdates,9changed-master/cast witnesses,
6fullstates,andfulloptimizerrestore6→8. SavedCPU RNG6/8 matchesall8ranks;
no independentimmediate deviceRNGreadback ortrajectoryparity claim. Source829d9a2,
320updates/two passes/two seeds,fresh384 prospectively frozen. Prefixunscored;
directfuture16 versusfulltrace+Answer+future16;noCPT/emergenceclaim fromsuccess.
Quietcompiledcache movedbyte-for-byte toexistingNVMe,originalruntimepath retained;
no model/checkpoint/data removed. Diskreserve andfuturebackupbudget verified.
Evaluation41a5bfd/1,664CPUchecks ready;permodel actualnative/cache andfullspan
reference checks still required. MainlineV10mixed result remainsclosedunchanged.

2026-10-19 08:50UTC — Allfour target-SFT runs complete320updates and physically
close. Independent audit passes1,280orderedupdates/eightfullstates/32nativechecks.
Protectallfour terminalmodels/latestfull320;retainoriginal160/320. Allnine fixed
models now released forfresh384 scalar/full16 andordinary256 evaluation. No
partialcapabilityreview,automaticextension,selection orretry. Publishalltraining
traces andmeasuredcosts now;finalreadout waitsallnine controllers/scorejobs.

2026-10-19 09:33UTC — V15 complete and independently checked: CoT342/335 of384
versus direct275/285, same-CoTempty261/263. Declared >=3pp/positivepaired
interval/control gate passes both seeds; zero execution errors, all failures kept.
Most net corrections are between parser-valid numeric outputs (59/45 of67/50).
Preserve the cost: ordinary NLL +0.128/+0.147 nats/token versusB0, full16 belowB0.
This is a useful supervised arithmetic reference, not broad pretraining benefit.
Next bounded question is source-quality retention with separately normalized
source/task objectives. No broadRL,8B or expanded split rescue is released.

2026-10-19 10:50 UTC — Release source-balanced V17 after actual qualification.
The existing two-component CE engine passes eleven real updates, six full-state
readbacks and a fresh-process optimizer restore at the exact declared shape.
Source/task counts and weighted loss are independently verified; four fresh B0
models now run for the frozen320 updates. Fresh confirmation512/ordinary256 and
both-seed accuracy/source-quality gates were published before evaluation/training.
No partial capability review. Retire only older EBS optimizer duplicates after
fresh both-side hashes and newer full protection; all originals/models remain.
The new full checkpoint must be protected before retiring V15 optimizer copies.

2026-10-19 11:13 UTC — Integrate versioned production code into main.
Merge the real-data training/evaluation histories while retaining the bounded
variable-depth path. Resolve schema conflicts by preserving both guarded identities.
All1,699 CPU checks plus smoke/compile pass. Live training/evaluation worktrees
remain clean at37028a1/c9b399b; their qualified source identity is unchanged.
The combined main snapshot has CPU integration evidence, not a new hardware pass.

2026-10-19 11:39 UTC — Close and protect all V17 scientific training.
All four terminal320 runs physically close. Independent audit verifies1,280
updates,32 native checks and all eight model/master/optimizer states, source/task
label counts, ordering and weighted losses. Protect the new full recovery state
first, verify both older copies, retire only their EBS optimizer duplicates,
then protect all terminal models. All originals remain. Publish training-only
receipts and traces; all capability outcomes remain unread until the complete
11-model evaluation. The final exporter verifies this completed publication
without rewriting it. No model, endpoint, gate or training budget changes.

2026-10-19 12:40 UTC — Close V17 as arithmetic-positive, quality-bound-negative.
Wait for all eleven models/panels, then review once. Independent checks reproduce
all native decodes, exact values, counts, bootstrap intervals and literal gates.
Generated calculations score434/439 of512 versus351/357 direct and341/326 empty.
The arithmetic gates pass both seeds; ordinary-NLL upper intervals0.0486/0.0554
exceed the frozen0.03 bound, so the combined gate fails both. Preserve that failure.
Against old target-only models rescored on this same panel, the point estimate of
NLL regression falls75.6%; there is no cross-panel comparison or equivalence claim.
Retain the supervised inference reference and all costs. No RL/8B/split release.
Next specify a bounded single-model document-QA comparison, beginning with evidence
clusters, complete context and trace lengths; actual long-context training must qualify.

- 2026-10-19 13:05UTC — V21 corrects a CPU split-identity interpretation before
  any model outcome: raw component strings can be generic relation programs.
  Use original dataset:seed IDs with held-support text/page exclusions. Freeze
 2,048TRAIN/512 independently supported held questions. All6,144 native views
  and independent raw/count checks pass;1,711 CPU checks pass. Single-model
  direct/prior/post targets retain plain-answer training; post teaches answerEOS
  before its scaffold. Source CE samples256 fixed positions per example, explicitly
  separate from full-document CE. Launch only bounded long-context mechanics at
  source049f53e;held forwards/science await actual recovery,memory and timing.

2026-10-19 13:28UTC — Freeze the proposed single-model QA comparison before any
held forwards. Canonical complete-answer EM separates post-answer parameter
training from generated-CoT inference;both use prospective3pp/paired-CI criteria
per seed,with unchanged-base conventional CoT as an additional practical bar.
All1,735 CPU checks pass;norm/EM/F1 matches pinned author code on519syntheticpairs.
Freeze256 ordinaryWikipedia pages disjoint from allQAvisible pages/text. The
initial ordinary sample failed its unique-page check;fixed hash selection now
explicitly takes one paragraph perpage before any model outcome. Preserve both
attempts. Train snapshot049f53e and evaluationdbd1c10 remain immutable. First
long-context update passes finite/master/replica checks;warm/recovery/memory and
long-prefill decoding checks remain active. No science release from these alone.

2026-10-19 14:47UTC — Complete the alreadyfrozen QA evaluation orchestration,not a
new study. Group v3 isolates protocolTMPDIR and sequential replayports. Seven
full-span panels each use four batch4 shards after own-model TRAIN calibration;
physical14/15 reserved while generation runs. One readout waits ALL20,992 answers
and19,712 LL rows;independent native/text/metric/bootstrap verification precedes
render/protection. Canonicalanswers primary,mirroraliases separately attributed.
Span nativearrays independently recomputed/readbackprotected;first128 fullmodel+
optimizer protected before only older EBS optimizerduplicates retire. No capability
outcomes inspected,no prompt/checkpoint/metric changes,no broader launch release.

2026-10-19 15:23UTC — Allsix QA science runs reachterminal256 and physicallyclose.
Independent all12fullstate/order/label/native auditPASS;copy-new-full-first EBS
protectionPASS,allterminalmodels protected and exactbindings ready. Alloriginal
fullstates retained. Prepare nativeCPUreferences and fixedgeneration groups;
capabilityremainsunread. Publishactualtrainingtraces/nativemapping separately
fromscientificresults. Ownerreview reinforces smallTRAIN2048/formatlearning risk,
canonicalEM+F1/attributedaliases,unknownpretrainingnovelty andexecution-only meaning
ofnumericalchecks. No outcome/gate/data/prompt changes ornewexperiment launch.

2026-10-19 16:40UTC — All20,992GEN/19,712LL executions close. Initialindependent
readoutchecker passes answer/native-decoding/metric/bootstrap checks but stops on
exactPythonfloat equality foroneof60recomputednative summarydicts. Same saved
vectors/source produce<=1.979e-12 scalar differences;both originalgates PASS.
Preserveoriginalchecker/failedlog/diagnostic. ExplicitV2 derivedchecker compares
FP64summaries withinabsolute1e-10;bools,argmax andoriginal.01nativeKL/.001prefill
gates stayliteral. No rawmodeloutput,score,metric,CI,scientificthreshold orbackend
changes;no modelexecutionsrerun. Newrepairsource/attempt thenrender/protect,with
unchangedprimaryreportSHA enforced. This is auditnumericsrepair,not a scientificpass.

2026-10-19 16:58UTC — Close QA-v21 without promoting a failedgate. Own217/227
versusempty190/169 andmatcheddirect201/195;all512pairscomplete. Localgainpositive
bothseeds,butnetfirstseedCIcrosseszero;overalltraining/inference/practicalgatesFAIL.
Post200/186 showsno directaccuracygain,despite+.38/+.93goldLLsecondarybenefit.
OwnLL−7.34/−7.75;hardwrittenwrongclaims andaliases stronglypenalized. DirectCoT's
strict1/1 is formatfailure:posthocgold-independentfirstline rule recovers191/182,
not revisedprimary. Storedtraces/teacherrelations needgroundinginspection. Preserve
allprimaryscores/decisions andsources;publishreviewpackage beforeanothertraining.

2026-10-19 17:18 UTC — Complete a bounded TRAIN-only grounding audit before
another learning release. In 24 fixed-rank, hop-stratified questions, 57 of 72
cited relations are locally supported; eight citations are insufficient, six
relations ambiguous, and one temporally contradictory. Four examples conflate
different counties/cities with the same name. All citations match author-raw
step evidence: these examples are not introduced by our packing. This is one
agent's exploratory annotation, not independent gold or a corpus error rate.
Blind #n expansion does not repair entity identity or unsupported relations.
Prepare a small passage-selection versus format-matched direct comparison;
no intermediate answer claims, no old-score changes, no training released yet.
Fresh evidence is mostly two-hop under existing exclusions; do not promise a
fresh four-hop study. Source-family filters were inventoried, not validated as
semantic curation. Keep V21's failed gates and original report hash unchanged.

2026-10-19 17:38 UTC — Prepare evidence-plan-v23 as a bounded new recipe, not an
explanation of V21. All1,763 CPU tests, smoke/compile and6,144 native views pass.
Both arms preserve source and final-answer labels and their explicit coefficients;
selection has its own0.1 mean CE. This addresses measured V21 answer dilution:
prior-example mean coefficient9.3% of direct,54.6% across the four-pass schedule,
excluding other answer-bearing trace labels and shared-parameter gradient effects.
Use within-epoch mode interleaving and a common generated Final answer field.
The fixed fresh selector exhausted at255/256;preserve V1 failure,declare V2 with
those same255 and unchanged exclusions/ranking/gate. No model outcomes inspected.
CPU readiness does not release science:actual new-objective update/save/restore
and bounded resource review remain. No semantic certificate for support IDs or
claim of connected reasoning follows from the new representation.

2026-10-19 18:11 UTC — Decoder qualification passes on two TRAIN questions/four
attempts with zero execution errors and physical closure. CPU/ordinary/cached
max log-probability difference stays below0.000115 against the unchanged0.001
bound. Three untrained B0 outputs lack the final field; retain them and do not
tune the frozen prompt/parser from these outputs. This is execution evidence.
The actual three-component training check has one finite update and continues
compilation before save/restore; no longer training or held forwards released.
Freeze likelihood diagnostics as canonical pointer-ID replays with invalid-plan
empty fallback explicitly flagged, separate from native-generation observations.
All255 canonical targets fit32tokens includingEOS(max13); no truncation needed.

2026-10-19 18:29 UTC — New-objective qualification passes11 actual updates,
all6full-state audits andfreshprocessrestore. Post-save groupedPSS83–97GiB;
24native witnesses repeat4contexts,not learning evidence orindefinite-memory proof.
Release one two-arm pilot fromfreshB0,seed2026101923,terminal256 primary.
No secondseed until primary/integrity gates;inferenceassistance separately requires
same-model direct andempty gains. Fourchips,expected6chip-hours,hard24upperbound.
Protect firstfull128 andterminal256 before retiring older EBSoptimizerduplicates;
alloriginalstates/allterminalmodels remain. One inactive V21intermediate-model
BACKUP duplicate was copied/readback/fsynced tosecondNVMe before its oldEBS copy
was unlinked;latestfull andsixterminalEBSmodels were untouched. Receipt inrelease.
All1788CPUchecks/smoke/compile pass,including pointerreplay/padding diagnostics.
Noheld model forward ornewcapability outcome at release;V21gates remainunchanged.

2026-10-19 19:39 UTC — Both fresh v23models complete256updates andphysicallyclose.
Independent terminalaudit passesall4original128/256states,16initial nativechecks,
orderedquestion exposure andseparate source/final/selection labelcounts.
Bothterminalmodels/latestfull256 are readback/fsyncprotected andboundforevaluation;
alloriginalstates retained. The earlier128fullEBSoptimizerduplicate retiredonly
afternewfull256verified. Allsmalltrainingwitnesses/metrics andpublicationprotected.
Begin the fixed1530nativeattempts,then3062originalLLrows;noearlyscoreinspection,
checkpointselection orconditionalreplication beforethe declaredpilot readout.
This is completedtraining/recovery evidence,not a newcapability result.

2026-10-19 20:00 UTC — OriginalLLexecution stoppedbeforeanyheldscore loop:
secondTRAINpanel triedtocreate the cumulativecalibrationJSON again(FileExistsError).
This was an agentintroduced IO bug,not an observednumerical/scientificfailure.
All8QAchecks passed;ordinaryresults wereunrecorded. Preserveallv1artifacts/source,
verifyownedprocesses absent/physical6-7empty,archive/readback beforeexplicitrepair.
Workerbf288fe writesimmutablepanelwitnesses andatomicallyupdates onlythisattempt's
summary;twofilesystemregressions includedin1797passingCPUchecks/smoke/compile.
Exactsameinputs/CPUreferences/models/native1530answers;scoringkernel/bounds/gates
unchanged. Newoutputdirectories/registry/readoutroutingv2,originalsuntouched.
All16actualTRAINpanelchecks passinV2 beforeheldscores. Failedcalibrationcompute
accountedseparately. No heldanswer/likelihoodoutcome inspected tochoosethisrepair.

2026-10-19 20:17 UTC — Close v23pilot withnegative allocationdecision. All1530GEN/
3062LLrows andindependentreadout/render/protection complete;primary129/255vs127,
+0.8pp[-4.7,+6.3]. Same-modeldirect127;allfieldscomplete,all255plansvalid,zero caps/
executionerrors. No replication/largelearning/model-size/RLrelease fromfailedgate.
Readeroracle155(+10.2pp[6.3,14.5]vsown) addsIDs only. Exactannotatedsets132/255;
posthoc exactsubgroup+14answers vsdirect,other−12,not a causal estimate.
Suppliedemptyfield regeneratesIDs255/255:qualifycontrol,retain126score andoriginal
figure. Genuine directpaths containzeroIDs;theirnull contrasts/gates unchanged.
CanonicalIDreplayown−empty−.731nats,own−donor+.845;not calibration. Ownactions2.35×
control,sourceNLL+.135vsB0. Oneorderseed/fresh255mostlytwohop;no broadclaims.
PrimaryreportSHAca7cc615495d0cdade77ef70cce0f19055e3456b1c79c460a7f84008dec809b3
unchangedafterstored-outputdiagnosis. Nextboundedquestion:does frozenwriter fit its
ownTRAINlabels bysupport-count,stratified/hash-selected? No new training yet.

2026-10-19 20:27 UTC — V24 frozen TRAIN selector fit diagnostic
V23 failed its net and inference gates; do not replicate it. A fixed hash TRAIN256
cohort oversamples support-count strata128/96/32. Reuse B256 and the unchanged
prompt/decoder, with actual own-model checks: one chip, deadline21:00, no updates
or retries. Annotation agreement measures fitting, not semantic correctness or
generalization. Freeze >=90% two/three-support fitting target before generation;
four-support n32 is descriptive. All1797 CPU checks and projected-file audit pass.

2026-10-19 20:33 UTC — V24 closes: multi-support TRAIN fitting is incomplete
Independent raw-token/cohort/checks pass. Exact sets117/128,59/96,12/32 while TRAIN
answers255/256. Twenty three-support outputs omit annotated IDs with no extra IDs;
identical-input label conflicts and literal duplicate-paragraph replacements do
not explain the gap. Keep semantic-label and sampling limits explicit. Do not
attribute the held failure solely to generalization or capacity. Prepare one
coefficient-only selector-supervision ablation, keeping source/final-answer weights
and data fixed; no new learning is released by this diagnosis alone.

2026-10-19 20:40 UTC — V25 coefficient-only source and qualification released
One treatment changes selection weight0.1→1.0, retaining data/order/exposures and
source/final-answer weights. Existing completed0.1 run is the bound control. New
identity/source5414f8d preserves numerical worker/order/batching functions;1809 CPU
checks pass. Only eight-update/full-restore qualification released, two chips,
deadline22:00 UTC, <=3 chip-hours, no retry. Four full states fit NVMe-study; latest
science full backup untouched. Three inactive engineering model backup duplicates
were copied, fsynced and reread against originals before old duplicates were
unlinked; receipts published. Full treatment requires actual audit and root review.
The already-inspected held255 can only be reused as developmental evidence.

2026-10-19 21:07 UTC — Prospectively separate fitting from answer quality
Before any treatment training, refine V25 readout: held255 is developmental and
will be reported regardless of imperfect-annotation TRAIN fit. Preserve the fitting
rule and all original V23 scores/gates. Matched closed-Evidence + prefilled-Final
empty/own/oracle answer replays use the existing context builder/direct parser;
all765 real contexts pass CPU closure/tokenization checks. Native direct/selection
prompts stay byte-identical. Planning simulation confirms limited small-effect
resolution; it changes no old gate. One actual qualification update passes; initial
component CE matches old exactly, gradient norm44.4→59.1, not a10× gradient share.
Full treatment remains unreleased. A fourth inactive model backup duplicate was
verified/copied/fsynced before old EBS duplicate unlink; original full state retained.

2026-10-19 21:24 UTC — Qualified single treatment released and launched
Actual eight-update/full-restore audit passes all4 states/native16/label counts;
closed evidence and final report protected. Release one fresh B0 selector1.0 model
against the completed0.1 control, identical data/order/source-final weights. Source
5414f8d and all release pipeline hashes frozen. Exec7042/physical2–3, terminal256,
full128/256 protected by prepared watchers, soft deadline22:45:23 and3chip-hour bound.
A fixed six-case TRAIN review finds2 entity collisions and2 root-sufficiency concerns;
this is exploratory, not prevalence or gold. The ablation tests canonical label
fitting and developmental answering, not a semantic-grounding fix. No label filtering,
original-score changes, newsize/RL or campaign expansion.


2026-10-19 22:53 UTC — Close V25 without promoting its failed fitting gate.
The sole coefficient change0.1→1.0 gives TRAIN exact124/128,64/96,15/32;
three-support fitting is still below90% and its paired interval crosses zero.
Developmental own143 versus old direct127 improves the recipe, but own versus
new direct138/clean empty139 remains inconclusive. Ordinary regression halves
to+.067vsB0; it remains a cost. All outputs/checks/backups close cleanly.
The old direct control retains auxiliary0.1, so a bounded format-only1.0 control
is the next causal comparison before scaling or fresh confirmation. No new
learning/official-final-data release follows from this readout. All older scores/gates remain.


2026-10-20 00:46 UTC — V26 closes with a positive developmental training contrast
The original matched-format terminal scores120/255 direct versus completed
selector1 direct138: +7.1pp[2.4,11.8], passing the prospective developmental
criterion. Same closed-empty replay gives+7.8pp[2.7,12.9]; all510 new answers and
256 ordinary rows independently check, with clean controls and no failures.
Selection also preserves ordinary prediction better, but both regress from B0.
The lower-weight format control remains stronger at127; its selector-direct
contrast remains inconclusive. Do not relabel V25 incremental-ID or fitting gates.
This is one order seed and an adaptively inspected panel. Freeze any fresh
confirmation against both direct controls, resolve sample-size/source availability
before launch, and retain complete outcomes. No coefficient sweep, larger model,
RL or official DEV/TEST access is released by closing this comparison.


2026-10-20 01:00 UTC — Fresh confirmation needs more than nominal question counts
V27 metadata finds zero remaining clean TRAIN questions. Author raw DEV download
returns quota HTML; two pinned mirrors agree and all19938 cached TRAIN records
recheck against author raw. DEV has875 source-clean/length-eligible questions,
333 connected evidence groups, greedy343 independent and upper614 from distinct
original IDs. This cannot meet the planned1024 independent items. No smaller
confirmation, relaxed exclusions or model evaluation is released. Original
20-paragraph metadata assumption failed on16 shorter-context rows; a declared
CPU-only repair retains every provided paragraph and preserves failure evidence.
V26 remains promising developmental evidence. Continue the roadmap's controlled
computation reference with variable-depth verified programs and a trained pause
competitor; synthetic mechanism results will not be called general pretraining.


2026-10-20 02:54 UTC — Continue the fixed computation study after actual recovery
All three first-seed64 checkpoints pass independent whole model/master/moment,
mask/count/order and native checks. All12 actual restored optimizer digests match
freshly recomputed saved state. Continue64→4096 under the original06:54UTC wave
deadline; all six terminal models precede held outcomes. This is recovery evidence,
not uninterrupted-trajectory parity or scientific utility. The live evaluator
uses the qualified native decoder without a numerical/parser change, fixed cohorts,
complete failure denominators and separate training/inference claims. Preserve
the V26 developmental result and V27 data-bound qualification as stated by review.

2026-10-20 05:15 UTC — V28 first wave closes at the fixed4096 endpoint. All three
terminal all-state/native/order audits pass; measured reserved time8.296chip-hours.
Release the fresh second data-order seed under the same source/data/optimizer.
The initial second-seed launcher passed a relative release path, which failed
before model loading/output creation/updates after the driver changed cwd.
Preserve all three failed control directories and empty owned temporary directories;
make one explicit absolute-path correction with the original release/deadlines.
Cost0.0017701253chip-hours; no numerical source or science protocol changes.
Continue first64/audit/actual restart/4096; no held inference before all6audits.

2026-10-20 06:15 UTC — Extend only V28's prospective evaluation time allowance,
from18 to30 summed reserved chip-hours (stop29.5, max8chips,8h wall). No held V28
model outcome exists; all6terminal-audit guard remains unmet. Prior TRAIN decoder
qualification gives~0.0693s/action: canonical trained completions plus capped B0
references would already cost18.48idealized chip-hours before load/native checks.
Preserve every model/cohort/mode/cap/parser/gate and frozen numerical source2eb3c1c.
Keep original design/reports unchanged and publish the explicit operational addendum.
The new host wrapper preserves the unchanged builder's original18-hour release,
then writes a separately linked30-hour operational record. CPU source-preservation,
accounting, finite-bound and actual-owner-death checks pass; not8-chip validation.
Do not shrink the powered cohort or wait for partial outcomes to extend execution.
Second-seed three1024readbacks/fsyncs and subsequent updates observed; continue4096.

2026-10-20 08:09 UTC — All six fixed4096 models close0 and pass independent full
model/master/moment/native/order audits;24actual worker restoration comparisons pass.
Science training16.535chip-hours including the pre-load path failure; qualifiers separate.
Freeze and publish all six actual checkpoint identities before any held model outcome.
The original18-hour all-six release is preserved beside the explicitly amended30-hour
operational release. Next prepare the entire fixed23-job comparison and secondary
ordinary-text diagnostic, then execute; no checkpoint, mode or prompt selection.

2026-10-20 08:20 UTC — Preserve an evaluation-preparation IO failure before any task
sample/accelerator call: the host wrapper assumed core prepare() returned its protocol,
but the frozen function writes protocol.json and returns None. Keep original helper,
log and complete first B0 preparation. One explicit IOv2 wrapper reads that file.
All23fixed preparations now close; first-panel prompt/gold bytes and complete CPU
native tensor match the preserved output. All model/source/data/gates/caps unchanged.
Publish exact23protocol bytes and launch the same bounded queue; no automatic retry.

2026-10-20 — Close V28 after all378880attempts and independent full readout.
Inference passes at covered lengths: T18432/18430 versus D11914/1872 of18432.
Direct-inference training benefit fails; lengths7–10 collapse to228/1 of2048.
The second order always writes six steps;2030/2048are correct prefixes,not full executions.
All task arms severely damage ordinary prediction; no ordinary LM term was trained.
Publish both the near-perfect composition result and failed transfer/retention,with
all raw synthetic evidence in release assets and exact portable replay. No RL/size jump.
Before new training,inspect V10/V15/V17 and surviving real-data weights; V17 already
provides a source-balanced arithmetic-positive reference with~0.04nat ordinary cost.
Do not blindly duplicate it. Next actual pretraining/guidance-withdrawal design must
preserve an ordinary LM objective,source separation,strong controls and fresh endpoints.

2026-10-20 — Prepare V29 on real prefix-to-continuation data,without rerunning V28.
Inspect V10/V15/V17 before choosing the intervention. Historical V17 already gives
+0.333/+0.507nat direct-span LL versus its trained control but−0.864/−0.802 versusB0;
retain absolute B0 and both source/format controls. V29 preserves a complete clean
forward including the natural continuation in every arm; auxiliary future and
work receive separate means. Add15360 ordinary windows to the known5120 verified
calculation documents; freeze768 new expression/page-excluded arithmetic DEV and
4096 ordinary DEV documents before any new model outcomes. Independent original
native-token readback and1878CPU checks pass. Work1 is chosen before learning;
placeholder0.1 in the CPU packing snapshot never trained a model or changed data.
Preserve the interrupted CPU validator and cache invariant vocabulary metadata;
2000 native len(tokenizer) calls took41.15s here. No source/stack change needed.
CPU readiness releases no long training. Next same-runtime two-forward update,
actual recovery/memory qualification,and explicit endpoint/power/storage budget.

2026-10-20 — Freeze the V29 eight-update TRAIN-only mechanics qualification before
hardware. Existing causal engine e7c2508 passes1890CPU checks,including actual
clean4+auxiliary1 dense-gradient parity and all three view/guard cases. Prepare
an actual released-B0 BF16 CPU initial-loss reference on32sources/8auxiliary
examples; tolerance.02/component,.025total fixed before device calls. One world8
model halts at4 and resumes in a fresh process to8,with all actual model/optimizer
bytes checked before further updates. New private RAM scratch preserves local
disk reserves without moving caches/deleting states; protect full8 on EBS before
reporting a pass. Two reserved chip-hour cap,finite deadline,no automatic retry.
This is qualification only; source/held/gates for long science remain unreleased.

2026-10-20 — Close V29 numerical/recovery mechanics and trained natural-stop batching.
Actual eight-update qualification,full4/8 audits and terminal EBS protection pass;
original supervisor window refusal and explicit same-budget continuation remain visible.
This releases no scientific learning claim. Freeze the scientific contrasts/power,
checkpoint placement and guarded fixed-arm implementation next. Batching passes on
one trained writer/eight TRAIN prefixes; adopt prospectively only,never into V28.

2026-10-20 — Freeze V29 direct-prediction endpoints and qualify the actual guarded science source.
Six protocols bind both source orders,all3arms,the exact-source recovery receipt and actual TRAIN CPU references.
No learning release follows yet. Full527 scorer hits runtime spill-queue exhaustion; preserve failed attempt,
correct graph-ID/count interpretation,and test explicit synchronization with unchanged inputs and gates.
Full new qualification backup is losslessly archived on EBS; original RAM states remain. Older qualification
backup-only optimizer shards retire after full hash readback; source model/evidence and all original states stay.

2026-10-20 V29 readout integration: independent fixed-TRAIN HF/native/cache gates pass for the four worker paths; no held scoring or learning claim. Source masks reconstruct original token boundaries, including merged whitespace/signs; standalone numeral tokenization cannot locate a source span. Keep all6 terminal identities fixed before held outcomes, and keep recipe, absolute-B0, retention and inference gates separate. The ordinary window has528 source IDs and527 transitions: prior reporting metadata2112/4224 was an off-by-one count, corrected by a new addendum to2108/4216 without modifying any original vector, producer, tolerance or decision.

2026-10-20 — Close V29 without rescuing its failed primary claim. All six fresh real-CPT models and28held phasegroups complete; independent all-context/statistical/98HF checks and extracted public evidence replay pass. Direct continuation gains versus both trained controls reproduce only in one order and both computation models trail B0. Keep the strong separately declared initial-scalar gains643/652 versus raw500/502,with all failures/caps visible. Full527 ordinary retention meets its+.03 target at approximately+.021; the declared suffix16 endpoint regresses approximately+.041 and gets its own public numerical analysis,not a new threshold. Preserve every original score/state and do not launch a recipe extension,RL or size jump from execution success alone.

2026-10-21 — Follow the owner's bounded diagnostic and stopping rule. Freeze V30's fresh TRAIN cohort, G8/temperature.7/cap128 policy and full16 correctness/utility requirements before model calls. Best-of-G is future-aware opportunity, never deployed evidence. Require gains over same-reader raw and unchanged B0, not merely empty format or bad siblings; absent identifiable usable alignment closes this reward branch without RL. A positive diagnostic releases no learning. Read-only V29 reconstruction verifies all40960 ordered exposures per raw run and the exact clean-plus.125 auxiliary-batch objective; selected future tokens have5.47–17.47 times the other per-document token coefficient. This identifies a reweighted raw control and a concrete hypothesis, not the cause of shared ordinary-text regression. No new CPT proposal before that failure is investigated.

2026-10-21 — Close V30 under its prospective stopping rule. All2304candidate attempts and3840full16scores complete; independent all-context/statistical/4HF and model-free projected evidence replay agree. Bestof8 improves greedy+.346nats but doesnotestablishrawgain(+.082) and trailsB0−1.083withnegativeCI. Reward's correct-versus-error contrast ispositive+5.112on25mixed documents; its gate failscoverage32,notdirection. Canonicalhelpfulcoverage98passes. Do notreinterpretthisasnoarithmetic-sensitivity: oracle−corrupt+2.774passes. Fullobjectivebenefit remainsabsent despite healthy mostlycorrect notes; close this rewardbranchwithoutRL or prompt/temperature/checkpoint sweep. PreservealloutputsandV29states; no newlearning/modeljobqueued. Publication-only namespace-file repair uses a new bundle-v2directory and doesnotchange any score,source,gateor model.

2026-10-21 — Read-only frozen raw-loop review finds no intervening gradient reset, unintended extra batch denominator or explicit double parameter-gradient reduction. Exact source hashes match both executed raw protocols; installed ZeRO averages once before clipping and stepping. Existing dense loss/gradient tests support the intended objective but do not diagnose scientific generalization or rule out every numerical issue. The raw-regression cause remains unresolved. Any subsequent CPU-only proposal must change the mechanism and preserve an actual raw reference; it cannot be an EMA-offset change or another prompt/mixer sweep. No model call or learning is released.


2026-10-21 — Finish the bounded selective-reader proposal without another model call. Reject the initial per-token0.03-regret cap: tokenwise hindsight routing plus best-of-eight reaches only0.137nats per16 on the saved V30 panel, below the earlier0.25 target. This is a post-result bound on that fixed proposal/candidate population, not a new V30 endpoint or universal impossibility. The revised draft preserves actual frozen B0 and specifies a small causal supervised head with positive initial mixing, full-span training, constant-blend/raw-ensemble controls and active ordinary retention. It gives up the pointwise guarantee and cannot establish better base weights. Source availability, joint power, actual-model causality/normalization and budget remain unqualified. Follow the owner's latest CPU-proposal-only scope: defer the unexecuted raw-checkpoint-trajectory draft; no data preparation, model calls, feature extraction, head learning, RL or new training are queued.


2026-10-21 — Compute the reviewer-requested unrestricted tokenwise bound from the saved GREEDY calculation only, without a model call, coefficient fit or sampled-candidate search. Verified released score hashes, matched target/document identities and original native target arrays agree: mean hindsight opportunity2.063nats/16, versus actual greedy loss1.429 to B0, on the already-inspected256TRAIN documents. Retain the fixed predictor pair as a candidate; do not interpret this oracle as causal recoverability, fresh confirmation or a V30 rescue. Add closure/gradient instrumentation to distinguish head stalling from absent opportunity. Report prevalence-weighted tradeoffs: hypothetical calculation gain0.25 versus ordinary cost0.48 per16 breaks even only above calculation-window prevalence0.658. No representative prevalence measured and no net-corpus claim follows from local gates.1945CPU tests/smoke/compile pass; execution remains held pending source/power/runtime/budget feasibility.


2026-10-21 — Resolve feasibility rather than repeat the oracle. Freeze4096DEV documents per stratum before new source outcomes: under the explicit normal planning alternative (SD4nats/span, base gain0.5, each control gain0.25, ordinary gain0),8joint requirements have union lower0.917;1024is inadequate at that resolution. This is conditional planning,not unknown-effect power. Start one fixed12-TRAIN-shard CPU source preparation at4cores/3hours/16GiB, retaining old exclusions and unchanged exact curator; no automatic retry, quota reduction or source expansion. Implement and test causal feature/entropy export and the small stable zero-region head on tiny random-model fixtures:1952CPU tests/smoke/compile pass,including a coin negative control and a separable positive control. Actual model qualification remains outstanding. Propose8reservedchip-hours total including qualification on at most2existingchips; old-timing proxy7.653is not a measured new-workload guarantee. All actual B0/T feature calls and experiment-head fitting remain held until feasibility and the whole-question release are concrete.
