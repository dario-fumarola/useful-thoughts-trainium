# Current queue — exact partial-computation channel

The next uncompleted task is the **frozen four-panel DEV answer comparison**.
The north star and causal steps are in [RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md).

1. Let the remaining reader controller finish on immutable
   `local/worktrees/partial-state-held-v6` at `3301d22`; never edit that worktree.
   Controls: `local/diagnostics/partial-state-v6-fixed-DEV512-{cot,reader}128-v1`.
   CoT has closed all four panels; the reader runs physical10/logical40–43.
   All2048writer attempts closed20:16UTC, with their complete EBS archive verified.
   At21:24UTC the reader has closed three panels and is generating composition.
   Outcome counts remain unread until the paired question closes. Each reader panel
   consumes its whole512-note bundle, including failures and intact unrelated donors.
   Four separate interpolation/structure/range/composition panels; single greedy
   attempt,80total-action budget (writer64+reader16, direct/CoT80), no TEST.
   Existing finite bound01:30UTC Oct19,3hours per child,512GiB memory floor, no retries.
   Monitor original stage/shard logs, compiler progress, process memory and placement.
2. After **all three** writer/CoT/reader controller `completion.json` files say
   `COMPLETE_ALL_FOUR_FIXED_DEV_PANELS`, execute:

   ```bash
   source scripts/env.sh
   USE_TORCH_XLA=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/audit_partial_state_panels.py --root /home/ubuntu/dario
   ```

   Redirect the full audit log to `local/analysis/`. It independently re-executes
   prefix gold and writer states, re-parses18,432answers, reproduces all intervals
   and actual numerical vectors, and archives complete batches on EBS. All failures
   stay in denominators.1598CPU checks pass; the independent state reparse agrees
   with all128closed fitting traces. No fresh outcomes have been read for this check.
3. Review once for the whole question. Separate teacher reading, student computation,
   composed benefit and comparison with ordinary compact CoT. Retain all four panels
   separately, unsupported counts, actual tokens, first-compile costs and fixed examples.
   Use that diagnosis to select the next learning intervention; publish negative results.

The public `partial-state-study-v7` branch is CPU/data-ready, **not released**.
Four validated local protocol drafts, created before fresh outcome review, are
in `local/diagnostics/partial-state-v7-unreleased-protocols-before-DEV-review`.
It gives each5120training expression one teacher and one plain reader view over
its two passes; both proposed seed schedules are exhaustively audited. The original
128fitting schedule is unchanged. The corrected primary budget is answer-only640,
compact partial-CoT640, versus reader320+writer320:640total updates and20480ordered
expression occurrences per system/seed, two seeds. A/B see four passes; R/W each two.
Full-root CoT is an additional prepared variant, never a replacement for compact CoT.
The proposed96action budget is writer64+reader32 versus single-model96. Actual chip-time,
supervised labels and two-model capacity still matter; equal updates approximate cost.
Before learning: freeze final source/protocols/orders; resolve disk; run one arm at a time;
full-save64, physically close and actually restore before completing320. Preserve a recent
full optimizer state as well as weights. Terminal checkpoints are primary.

Qwen3-8B-Base metadata and actual bank sizing are in
`validation/qwen3_8b_capacity_assessment_20261018.json`:15.26GiB BF16 weights,
30.51GiB weights+gradients versus24GiB/logical core. Full replicated SFT cannot use
this layout unchanged. BF16 inference, adapters and parameter/gradient sharding remain
unqualified options. No8B weights downloaded or run started. Papers and objective
corrections are in `docs/REFERENCE_NOTES.md`; size should address the failed causal link.

Completed evidence is indexed in `STATUS.md` and the README: matched developmental64
is negative overall; all128fitting runs and answers are closed; CPU/Trainium writer
native actions match exactly; reader and ordinary compact CoT both fit128/128.
Original terminal320 was interrupted by host OOM and remains incomplete. The short
world4 save3/restore/update6 is verified; cache256 speeds that small topology22.6×,
but does not prove the originalworld32 OOM fixed or qualify the old RL loop.

Keep35GiB EBS reserve. Full fitting states32/64/96/128 remain on NVMe; frozen models
and complete numerical/output evidence have read-back verified EBS copies. EBS is
same-instance storage. Four inactive engineering qualification duplicates totaling
102.6GiB have complete readback matches at
`local/analysis/inactive-qualification-archive-readback-20261018.json`; nothing removed.
General-sft-v5/broad RL remain held; R1234 stays stopped63 and old launchers retired.
Official TEST remains unused. Root owns live work; subagents are unavailable due
account usage limits. Preserve current science, controls and negative outcomes.
