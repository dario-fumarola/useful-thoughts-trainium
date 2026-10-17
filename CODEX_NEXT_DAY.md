# Codex handoff: the next 24 hours

Read `REVIEW_AND_NEXT_24H.md` in this review packet, then the live repository's STATUS.md, NEXT_STEPS.md and DECISIONS.md. This handoff reflects public HEAD `5cd4c79169a2f197357bc07e795c12d85bc530f0` and status 2026-10-17 01:41 UTC. Verify changes after that snapshot. Do not act on stale process IDs, commands or launch permissions. This file is review guidance; preserve the owner's actual authorization and the project's active process ownership.

## Goal

Finish the remaining R qualification and obtain the first matched research result. Do not expand the audit program or begin safety/BDI/teacher/compression work. More available weeks do not imply that the first experiment should automatically be the largest affordable one.

## Current facts to preserve

- Actual Qwen3-1.7B-Base weights and ties loaded; audited FineMath assets prepared.
- C32 completed 12 updates and passes fresh-process restoration/continued recovery within its declared distinctions.
- Preserve the CPU-coordinate v1 FAIL and the separately declared v2 derivative PASS. Do not invent v3 merely to improve that record.
- FP32 is the current accepted candidate for both arms. The integer clamp/sign defect is repaired; do not reintroduce it.
- R004 is active on upper cores with NVMe TMPDIR and raw logs. R003 is cooperatively stopping at a complete update/save boundary.
- The original R003-dependent upper-overlap preparation is stale and explicitly must not launch.
- No scientific pair and no complete R update were recorded at the reviewed snapshot. Do not confuse tape/backward counts with optimizer updates.

## Priority order

### 1. Complete the R004 execution test

Inspect actual run/spec/host-I/O identity and current workers. Keep active source immutable. Follow R003's authorized cooperative shutdown, verify its preserved checkpoint and backup, and retire only its now-unneeded waiting supervisor using the existing ownership rules.

Measure whether R004 completes a real optimizer update, including initialized FP32 masters/moments and memory at that boundary. Keep G4, L1024, K1, temperature .7, global128, accumulation4 and the selected FP32 math unchanged.

Do not advertise a TMPDIR-only causal speedup: partition placement and logging also changed. Report measured end-to-end workload performance.

### 2. Close the finite qualification gate

Use actual R004 checkpoints for fresh-process restored-state readback, first resumed stochastic tapes and full post-update comparison. Rebind the two-R coexistence check to that lineage. Preserve original criteria and failed attempts.

Obtain ten complete steady updates and measure simultaneous-R performance. Separate cold compilation, warm executable loading, actual execution, checkpoint and backup work. Qualification has more replay/tape instrumentation than science; distinguish the costs.

No new full-parameter FP64/CPU investigations without a specific new fault. Do not waive remaining R math/recovery checks, but do not invent additional gates merely because the current ones pass.

### 3. Address performance only from the measured breakdown

First use the already-running NVMe placement test. If warm performance remains poor, inspect repeated executable loads and graph variants under LRU2. At most one bounded optimization branch at a time. Possible experiments are canonical materialized per-completion inputs/stable graph boundaries or a bounded larger working-set cache, only if actual memory permits. These are hypotheses, not prescribed patches. Recheck affected signed-gradient/recovery behavior.

Do not upgrade Neuron, add vLLM, switch to 8B/MoE, reduce thought length, rewrite the objective or count cold compilation as warm throughput. Do not add per-token host transfers for profiling.

### 4. Freeze a small first research result

The review recommends the first paired reference terminate at 256 updates, rather than automatically maximizing to 2,048. Record this prospective budget decision before scientific outcomes; if actual timing makes it unsuitable, report the cost and choose a declared alternative explicitly. All scientific arms start from B0, never their corresponding C or qualification checkpoint.

Prefer two concurrent qualified 32-core chains:

- C1234 from B0, then R1234 independently from B0.
- C5678 from B0, then R5678 independently from B0.

Within each seed, C/R share exact source order and exposure. Use measured coexistence costs, fresh-record supply and backup storage in the schedule. Do not spend another day on world64 unless measurements show it is needed.

Scientific terminal checkpoints are primary. For a 256-step run, step64 is an existing fixed development milestone. A dev evaluation there is not a license to select whichever checkpoint wins. Keep the official final test sealed until its protocol/model inventory is fixed.

### 5. Complete the development matrix

Use a fixed, recorded held development subset, separate from demonstrations and final test. Report B0, C and R in all three existing modes: direct, empty_format, one_thought. Preserve errors and truncations, and report actual thought/answer tokens and latency.

`direct` asks the model to show calculations: label it ordinary decoding without the extra thought block, not reasoning forbidden. Thought versus direct also changes prompt structure; use thought versus empty_format to examine the extra note, followed by length/content controls.

Report R-direct minus C-direct, R-thought minus C-thought, R-thought minus R-empty, and each trained arm versus B0. Keep the existing primary same-policy contrast. Do not count a gain caused solely by C degradation as an improvement above the starting model.

For the local audit, preserve absolute clean/empty/own/filler evidence. K1 is primary; K4/K16 remain secondary. Describe token-class/surprisal strata on a fixed random dev slice without filtering training anchors or secretly skipping whitespace. The 16-case arithmetic check demonstrates full-answer sensitivity, not a strong K1 semantic reward. Positive LOO advantages do not mean any sampled thought beats silence.

## Operational details

Keep one code owner and the existing serialized backup queue. Host TMPDIR/effective temp filesystem must be bound separately as in R004; the current runtime whitelist alone does not contain it. Do not format more disks or invent an external backup destination.

A terminal/stop/deadline completion summary with achieved step and total wall time is useful. The driver can exit normally after an intentional early stop; downstream final-evaluation code already checks the exact frozen terminal step, and that protection must remain. Per-step elapsed excludes checkpoint work; do not omit that cost in resource accounting.

Use short status updates: latest complete R step, median/P90 full-step time, active blocker, next checkpoint/evaluation, backup status. Detailed traces belong in evidence files, not a growing narrative of every successful subgraph.

## End-of-day output

Provide one compact report with:

- qualification PASS/FAIL/PENDING for full R update, recovery and coexistence, linked to actual artifacts;
- measured full-step distributions, cold/load/checkpoint costs, active versus padded generation and source exposure;
- launched pair/seed/checkpoint identity, or an exact reason no scientific launch occurred;
- the fixed development comparison when available, with all three inference modes and no final-test leakage;
- latest verified recoverable checkpoint, EBS backup, and the unresolved off-instance-backup requirement.

No model-performance claim follows from a passing unit test, an interesting thought, a positive centered advantage or a cached-program message. The objective is one interpretable research comparison, not simply an occupied instance.
