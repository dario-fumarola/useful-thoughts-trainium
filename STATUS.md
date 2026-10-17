# Current state — 2026-10-17 01:57 UTC

Real-asset Trainium qualification; **no scientific C/R pair has launched**.
Public repo: https://github.com/dario-fumarola/useful-thoughts-trainium . Work is
isolated in `/home/ubuntu/dario`; original repositories remain unchanged. The
owner authorizes sustained autonomous work and review agents, with weeks available.
The prospective 14-day science schedule is a planning limit, not a shutdown date.

## Completed evidence

- Released Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`,
  actual weights and original embedding tie; FP32 in both proposed arms.
- Audited FineMath production slice: **329,535 train / 18,360 dev / 18,239 test**.
  Benchmark overlap exclusions and fixed evaluation IDs are saved. No official
  final benchmark outcomes have been read.
- Full-cap cached FP32 generation/replay passes the original .001 bound on the
  exact trajectory that failed BF16. The precision change preceded science.
- Full-cap signed full-model backward passes CPU loss/norm and prospectively
  declared signed-gradient cosine/L2 bounds. The original integer normalization
  returned -1 on device; float counting repairs it without changing the objective.
  See `validation/p2_repaired_signed_gradient_20261016.json` and
  `validation/int64_clamp_mechanism_20261016.json`.
- **C32 recovery PASS, common source `3aaaf8a`:** 12 updates; ten steady steps
  average **10.007 seconds**. Fresh-process actual restored state is raw-byte
  exact across model, all 32 optimizer shards, metadata, counters and CPU/device
  RNG. Continued updates through step 12 meet the original numerical tolerance;
  all RNG states and Adam steps remain exact. Continued weights are not bitwise
  identical. Verified EBS backups retain steps 6 and 12. All C workers and supervisor
  exited. See `validation/c_fp32_final_resume_20261016.json`.
- Both full-cap FP32 evaluator suites pass .001 CPU parity: local controls and
  crossed readers, 192 token comparisons, max error .00023079; secondary paths,
  90 token comparisons, max error .000070572, no ranking/token mismatches.
  All workers exited; only development/interface items used. See
  `validation/evaluator_fp32_fullcap_20261016.json`.
- Latest required source checks: **233 CPU tests plus toy smoke PASS**,
  `local/checks/tests.Yr1uiR`. Active source remains unchanged; later commits add
  documentation, receipts and identical reviewed diagnostic copies. Their four
  public-path CPU fixtures also pass.

## C numerical diagnosis and held-seed validation

The original three-step CPU FP32 per-weight comparison remains **FAIL**:
754 of 1,720,574,976 coordinates exceed its fixed bound. Correcting the CPU norm
reduction does not remove this failure. A separate identical-gradient hardware
AdamW test passes; first-step checkpoint moments locate the discrepancy before
Adam, where tiny cancelled gradients can differ in sign.

An independent FP64 differentiable-arithmetic diagnostic is complete (all jobs
exited). It uses exact released FP32 weights cast to double, promoted RMSNorm and
softmax, and frozen native-CPU FP32 RoPE constants. All 310 parameter-gradient tensors are
saved. Against it, device relative gradient L2 error is **.0001083**, versus
**.0003425** for the original CPU FP32 reference. Device is closer on 672 of the
754 flagged coordinates, but 90 selected sign disagreements remain. This is
numerical diagnosis, not a retroactive v1 pass or a scientific result. Receipts:
`validation/c_firstbatch_fp64_derivative_20261016.json` and
`validation/c_fp32_gradient_discrepancy_20261016.json`.

A distinct prospective **v2 PASS** on held seed 5678. Exact first 128 TRAIN IDs
and criteria were committed/pushed before outcomes; 6/128 records overlap the
old batch. All 1,720,574,976 derivatives / 310 tensors are compared once against
FP64. Relative L2 **.00010316**, cosine **.9999999947**; worst tensor relative
L2 **.00493315** is below its fixed .01 bound. Scalar loss/norm checks and all
eight mandatory gates pass; comparator exited 0. Diagnostic sign differences
remain 15,186. Preserve v1 FAIL. Frozen criteria:
`validation/c_fp32_accuracy_v2_20261016.json`; full local result:
`local/qualification/qwen-c-heldseed-001/comparison.result.json`. All 11 targeted
fixtures/source review pass. Independent outcome audit passes, including all
9,920 first-step Adam counters. Public result with all tensor diagnostics:
`validation/c_fp32_accuracy_v2_result_20261016.json`.

## Active work and process ownership

**Data agent: R32 qualification attempt 003**, lower logical cores 0–31.
Launcher 779314, timeout/PGID 779372, driver 779374, 32 workers 779498–779532.
Started 22:51:17 UTC; hard deadline **2026-10-17 06:51:17 UTC** plus 30s grace.
Source `3aaaf8a18baeccb8d1a543d348a53f6863599114`, active code hash
`244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7`.
World 32 / global batch 128 / accumulation 4, G4, cap 1024, FP32, seed 1234, XLA executable LRU 2.
Host store 127.0.0.1:64231; Neuron root 127.0.0.1:64194. Prior attempt 002 failed
before loading because the default host-store port collided with C; its failure
and bounded cleanup are preserved. Attempt 003 has loaded actual weights and
started update 1. Its first backward graph compiled at 23:48:39 UTC after
43 minutes. All 32 ranks have since completed the first backward with zero
recorded execution failures and hold gradient-bearing tensors. First backward
bank allocation was about 19.78 GiB. The next scalar loss passed at 23:54:13.
First accumulated backward `MODULE_7801530274365426598` compiled at 00:10:50
and executed successfully on all 32 workers by 00:14:12: every success counter
increased by one, with zero failure increments. All main banks fell to about
16.52 GiB, consistent with the compiled gradient-buffer aliasing. Third-thought
backward `MODULE_922332460170596362` compiled at 00:32:51 and passed on all ranks.
Fourth-thought backward `MODULE_4430468595289365099` compiled at 00:54:18.
All 32 micro-0 tapes now exist, written after each rank's four synchronized
backwards. Later-micro g0 accumulation `MODULE_16022798442968481532` compiled at
01:15:08 and executed on all 32 workers by 01:19:06, with zero failure increments.
The current bank maximum is about 16.52 GiB. Subsequent sampling, scalar losses
and thought-slot backwards are reusing cached executables; no compiler is active
at the latest observation. Complete micro coverage, initialized optimizer state
and a full update remain pending. Existing logs will distinguish program load,
execution and rank skew once complete warm-step timings exist. Replay audit runs
only at micro 0; later micros skip that audit.
All 128 first-step tapes now exist. At 01:51:26 all 32 workers owned distinct
cold optimizer compile launchers; current banks dropped to about 6.73 GiB.
Transient CCOM RX retries occurred while ranks converged on the collective;
no execution failure is recorded. **No complete R optimizer update yet.**
Separate sysfs memory sampling is read-only.

Read-only performance attribution found substantial EBS temporary-program I/O:
32 workers net-wrote 3.914 GB over 30.10 seconds, while EBS wrote 3.948 GB at
125.07 MiB/s and no physical reads. Open worker files were `/tmp/tmp*/model.neff`;
existing kernel latencies are brief. Receipt:
`validation/r_warm_cache_io_20261017.json`. Distributed prepared
`qwen-r-fp32-nvme-tmp-004`, upper partition, with the same numerical identity and
NVMe TMPDIR/raw logs. This configuration is now running; no complete-step speedup
is measured yet. Its physical placement and monitoring output also differ from
R003, so do not claim a perfectly isolated TMPDIR causal comparison.

**Distributed agent: active R004.** Reviewed preparation SHA
`49c770b7700c070fec3bf7b968dd333ee81e1b5393fb74a4656bbb432b9fec9a`;
root release SHA `cb29f2b8c1cf83ffd8433d3a6f0afee160f2cda1d13f1a4b3a0b1f4e048315a3`.
Fresh preflight passed at 01:38:30 UTC. Launcher 871318, timeout/PGID 871404,
driver 871406, all 32 workers on logical cores 32–63; exact identities in
`local/qualification/qwen-r-fp32-nvme-tmp-004/{processes,rank-processes}.json`.
Same spec `92f866c8…`, source `3aaaf8a` / code hash `244e2f13…`, seed1234,
G4/cap1024/global128/accum4/FP32/LRU2, max14. Ports 64301/64302.
Deadline **2026-10-17 09:38:07 UTC** plus 30s grace.
Output `local/nvme/qualification/qwen-r-fp32-nvme-tmp-004`;
raw logs/monitor in adjacent `qwen-r-fp32-nvme-tmp-004-observation`.
TMPDIR is `local/nvme/tmp/qwen-r-fp32-nvme-tmp-004`, NVMe device 66310;
persistent compiler cache remains EBS device 66305. Host I/O policy SHA
`9d2e1f20f3f38cb30ce41a80f4c3b453f44c4471aed15b382a84a2d8dd8cfb21`
is separately bound. Actual run spec/runtime/environment match the preparation;
All 128 first-step tapes exist by 01:55:12; the optimizer boundary is pending.
Three transient CCOM RX retries were preserved while ranks converged; no fatal
error is recorded. No complete update yet.
Root owns its active backup supervisor: launcher 897722, timeout/PGID 897725,
Python 897726, starts 3586343/3586344, tmux `dario-r-nvme-tmp004-backup`.
Same deadline; state in R004 control directory, events/logs in its NVMe observation
directory. Stop that watcher if R004 fails or stops before terminal backup.

**R003 cooperative stop requested at 01:40:24 UTC**, after R004's actual all-rank
initialization/spec/host-I/O verification. All 96 micro-0/1/2 tapes exist; current
update's micro 3 continues. No signal was sent. It will finish the full optimizer
boundary, save and exit; reason is measured host I/O, not reward/numerical failure.
Receipt `local/qualification/qwen-r-fp32-final-003.cooperative-stop-request.json`.
Data monitors this completion; root will verify its checkpoint backup, then stop
the waiting terminal14 backup supervisor. Preserve every artifact.

The fixed step-1/rank-0/micro-0 TRAIN example is saved in full as
`local/qualification/qwen-r-fp32-final-003.step1-rank0-micro0.example.{json,txt}`,
cross-checked against its binary tape without errors. All four notes are retained:
lengths 342/1024/146/298, three EOS stops and one cap. For this single prefix,
target `+`, raw logp is −.12897 and thought logps range from −6.92639 to −10.05456;
all raw gains are negative and signed LOO advantages have both signs. This is
untrained starting behavior on one TRAIN example, not an efficacy estimate.

Output: `local/nvme/qualification/qwen-r-fp32-final-003`.
Receipts/logs: `local/qualification/qwen-r-fp32-final-003*`.
Root owns the active EBS backup supervisor: launcher 849074, timeout/PGID 849075,
Python 849077, tmux `dario-r-final003-backup`, same 06:51:17 UTC deadline.
It awaits the first COMPLETE checkpoint and retains two verified EBS copies in
`local/backups/qwen-r-fp32-final-003`; source NVMe checkpoints remain untouched.
State: `local/qualification/qwen-r-fp32-final-003.backup-supervisor.state.json`.
Stop this waiting supervisor if the original driver fails or stops early.
The original R003 checkpoint3 resume preparation remains unlaunched and cannot
satisfy its terminal14 guard after this cooperative stop. Rebind recovery to R004:
export actual restored state before sampling, then exactly update4 with STOP
precreated; compare all 32×4 tapes and full post-update checkpoint. Require ten
complete steady updates before exposure selection.
The original upper one-update R overlap smoke is preserved, not launched:
`local/qualification/qwen-r-fp32-upper-overlap-001/prepared.json`, SHA
`dee151f1af85b76f86eb64ebb49c293d796463466e0c801a94e682a8db1d18f4`.
It binds actual lower fork/checkpoint 3/STOP readiness, source/settings and fresh
root release; separate host/Neuron ports 64251/64252, upper cores 32–63.
Its R003-terminal14 guard will not be satisfied after the cooperative stop.
Do not launch it. After R004 completes, rebind fresh upper R004 recovery and
lower NVMe coexistence under new reviewed identities; preserve these old bytes.

**Same-seed host-configuration tape parity PASS at 01:55:31 UTC.** All 128
R003/R004 pairs pass exact actions/masks/record/decode metadata and the existing
1e-7 + 1e-6 floating criterion. Every tensor payload is also raw-byte identical.
Each original was validated against its own run before normalizing only run_id
in memory. Root reverified all 256 tape hashes, bound inputs and coverage. This
does not compare optimizer/checkpoint state or establish training efficacy.
Public receipt: `validation/r_host_io_tape_parity_20261017.json`.

**Conditional RAM host-I/O preparation, not launched:** existing `/dev/shm` has
about 1000 GiB free and observed MemAvailable exceeds 900 GiB. Sixty-four copies
of the largest cached NEFF total about 24.23 GiB. Root is preparing the required
upper R004 checkpoint3→4 recovery with a private RAM TMPDIR and unchanged STOP;
review is preparing a lower fresh seed5678/max14 RAM timing/coexistence job.
Both keep checkpoints/raw logs on NVMe and the existing EBS compiler cache.
They require completed R004, original workers gone, independent review and fresh
root release. Host storage policy is bound separately. No RAM speed claim or
new numerical criterion; final science policy remains unset until measurements.

**Held-seed jobs complete:** C32 update/COMPLETE and FP64 full derivative both
exited 0; comparator exited 0/PASS. All upper-core workers and supervisors are
gone. Checkpoint: `local/nvme/qualification/qwen-c-fp32-heldseed-001/step-0000001`.
C peak sampled device allocation 18.455 GiB, sampled errors zero. Exact preparation
SHA `26a6081f…2313d4b6`; state/receipts under
`local/qualification/qwen-c-heldseed-001/`. Public receipt and independent audit are complete; all original failure receipts
remain.
Root owns integration, decisions and public pushes. Do not edit common active
code during these jobs; later documentation commits do not alter its fingerprint.
The ignored local site draft now binds recorded per-arm runtime policies,
12-hour checkpoint interval, per-seed partitions, NVMe outputs and EBS backups.
Terminal exposure and finite launch deadline remain unset until R qualification.

## Immediate commands and remaining work

```bash
python3 local/qualification/qwen-r-fp32-final-003.status.py
cat local/qualification/qwen-c-heldseed-001/comparison.result.json
```

Complete distributed R004 update/resume/timing and R003's cooperative save.
Overlap the required upper R004 resume with one bounded lower NVMe R update to qualify
two-R coexistence. Prefer parallel seed chains C1→R1 and C2→R2 on the existing
32-core partitions; ideal world 64 saves no study time against that schedule.
Freeze terminal exposure from measured complete-step costs and fresh-data supply before any final test access. Save
all failures, examples and outputs; no outcome-based terminal selection.

Large artifacts remain ignored under `local/`. One verified unused NVMe is
mounted at `local/nvme`; the other three SSDs are untouched. Latest space:
EBS 346.7 GiB free, NVMe 1.3 TiB free. An obsolete EBS C duplicate was retired
after full verification of its preserved NVMe original; receipt:
`local/qualification/ebs-retirement-c-fe-20261017.json`. The science backup queue
will serialize copies across runs; its six host fixtures and independent TERM
review pass, and it remains unlaunched. A backlog can need two new payloads
before pruning. The old C4 checkpoint and its hardlink alias were relocated to
verified NVMe archives, preserving all bytes and receipts:
`local/qualification/c4-checkpoint-relocation-20261017.json`.
EBS backups and scratch share this instance;
no external backup location is approved. The reviewed backup supervisor retains
two verified complete states per run; see `docs/CHECKPOINT_BACKUP.md`.
