# Active uniform-LM execution — October 21, 2026

The owner said “go” for qualification, the four fixed training settings if qualification and measured cost pass, and the sealed confirmation sequence. The design/data remain unchanged. Current native control directory: `local/diagnostics/uniform-lm-native-v1`.

- CPU B0 reference is running on 24 cores: PID 2522740, exec session 20743, log `CPU-reference.log`; no native worker yet.
- Native supervision is `scripts/run_uniform_lm.py`; the checker reconstructs initial loss, accumulation/LR/cursors and exact uninterrupted/recovered terminal model plus all eight optimizer shards. The recovery test forks checkpoint 2 into a separate directory with the same run basename; it never removes or rewrites the uninterrupted checkpoint 4.
- Added actual local-optimizer LR and one-reducer telemetry to the existing training engine, and corrected its schema/arm dispatch. No objective change. CPU checks pass (`local/checks/uniform-lm-native-20261021.log`).
- Native stage releases must bind a fresh immutable execution checkout, source hashes, protocols and bounded resources. Maximum two existing physical chips (8/9), 18 total reserved chip-hours including at most 0.5 qualification, original storage/memory floors. Qualification must pass and measured full-workload cost must fit before science fitting.
- Next: finish CPU reference, freeze native source/release, launch qualification, audit and continue the fixed study. No DEV/CONFIRM model result has been observed.

# Active uniform-LM preparation — October 21, 2026

Owner endorsed returning to a sound uniform next-token baseline. CPU source and objective preparation are complete; no new released-model or accelerator call. No goal or new subagents.

- R=/home/ubuntu/dario; F=R/local/diagnostics/uniform-lm-baseline-v1; D=R/local/data/uniform-lm-baseline-v1.
- Source plan `configs/uniform_lm_sources.json`, SHA `95a5c042799a753ea3dc051fe7c8a0e11a3059e98c926be1ee880e6ee623e2d0`. TRAIN 32,768 / DEV 1,024 / CONFIRM 4,096 / QUALIFICATION 1,024. Identity/length-only source/cut choices; all old reader/fresh sources excluded, no replacements or downloads.
- Preparation and independent original-Parquet check cover all 38,912 sources and complete in 121.43 /125.00 seconds on four cores. Dataset 108 MiB. Source preparation manifest SHA `33d54c3f0d91bbef08acd95f69364c48c405aa10988b0225e0e58177661c05df`. All stage PIDs 2521568/2521655 and coordinator session 24092 closed 0. Immutable data-preparation worktree `local/worktrees/uniform-lm-baseline-v1` at `cd29b59f8940cc31d8915f786f2c3e6e6b8f4163`. Do not change those data-generation sources retroactively.
- Current generic engine adds `train/uniform_lm.py`: exact all527 shifted targets, no auxiliary branch, global accumulation indexing and optimizer-step LR. Shared driver uses actual global batch for metrics/resume and avoids the legacy step64 save. Parameters/gradient storage BF16, master/moments and CE FP32, explicit compiler auto-cast=none.
- All 2,023 CPU tests, smoke and compile pass: `local/checks/uniform-lm-final-20261021.log`, session 84391 closed 0. Dense toy loss/all-gradient parity, rank/microstep coverage, actual EOS target, schedules and held-source sealing are tested. Native release guard also requires explicit initial-loss/reducer/master/LR/replica/recovery/precision checks, not just a PASS label.
- Five concrete CPU-validated protocols are in F/protocols (qualification plus four development settings), receipt F/cpu-protocol-check.json. Old derived protocol copies remain in F/protocols-source-cd29b59; use current files only. The updated native release guard is later than the data-only worktree; freeze current main into a fresh native execution checkout before qualification.
- Fixed design `configs/uniform_lm_baseline.json`: all four recipes start from B0, same 32,768 sources/order/17.269M targets, batch 256, accumulation 8, 128 updates. Peak LR 3e-6/1e-5 x warmup-constant/cosine, warmup 13, cosine floor 0.1. All terminals freeze before DEV 1,024 selection; pin chosen checkpoint/recipe before CONFIRM 4,096 scoring. Full 527 NLL is primary, suffix 16/calculation diagnostics separate. One training order/model; no broad replication claim.
- Report and small proof records: `validation/uniform_lm_preparation_20261021/REPORT.md`. Actual native accumulation/LR/recovery and runtime remain unqualified. The proposed 18 total chip-hours (qualification<=0.5) and two existing physical chips are not a native execution release. Full states fit existing /dev/shm under 180 GiB; terminal BF16 copies fit existing nvme-cpt under 16 GiB, with 35 GiB free and 512 GiB host-memory floors. No protected-state deletion.
- **Next work:** bounded native same-shape accumulation 8 qualification, including step 2 full restore and steps 3/4 LR propagation, then actual cost projection. A complete fitting release must bind the unchanged four-setting design/data/protocols and finite resources. Do not treat this CPU source/logic preparation as a learned baseline or numerical hardware pass.

The preceding selector confirmation/publication is complete; its full handoff follows as history.

# Completed selector handoff — October 21, 2026

The fresh frozen-head confirmation is complete. No goal or new subagents. All native work and the coordinator closed successfully; physical devices 8 and 9 have no active Neuron process.

## Result and immutable execution

- R=`/home/ubuntu/dario`; F=`R/local/diagnostics/reader-fresh-confirmation-v1`.
- Immutable execution worktree `R/local/worktrees/reader-fresh-confirmation-v1`, commit `d0bd192b068314158fc58c508a7d512a1ac91097`. No execution source or protocol changed during the run.
- Protocol `F/protocol.json`, SHA `6e4f79175a17af1981e3ff7c98ee2a04686e7b525d403e45f5b8976c27209810`.
- All six fixed calculation heads improve over B0 on 1,024 fresh sources: +0.421–0.506 nats per 16. Five beat every control. Order 45 versus the calculation position schedule is +0.0752 [−0.0418, +0.1876], so the all-six calculation endpoint fails.
- All six lose on 4,096 unenriched natural windows: −0.0195 to −0.0519 nats, every interval negative. The natural endpoint fails. The no-calculation constant ensemble instead gains +0.0244 [0.0155, 0.0340]. Keep that stronger control visible.
- All 12 heads and fitted controls remained fixed. No fit, optimizer or model update. Independent source checks pass for every selected row, without replacement.
- Native output `R/local/nvme-study/runs/reader-fresh-confirmation-v1`: 64 workers, all return codes 0. Supervisor 2440204 and coordinator exec session 52227 closed 0. Runtime status `ALL_FROZEN_HEAD_CONFIRMATION_FEATURE_PROCESSES_CLOSED`.
- Actual 1.7817652013469585 reserved chip-hours including 0.1300066526091703 qualification, below the 4.25 cap. Scientific native wall time 49.6 minutes. Explicit auto-cast=none. New prospective numeric contract passes; old strict failure and original deviations remain unchanged.

## Verification and publication

- Full readout `F/readout/readout.json`, SHA `18a51671da1eff5b5d0373cab6ac387cd79179ddba43777b52941ed3ea7ae8d6`. Independent NumPy reconstruction agrees on every feature-derived head prediction, 80 contrasts and both decisions: `F/independent-readout.json`.
- Source-free public projection `F/public-evidence` and `F/public-projection-replay.json` pass. Include original gate logits: rounded gates near one lose a small base contribution; this was a publication-export issue, not a scientific-score change. The two incomplete export attempts remain under `F/public-evidence-first-rounded-gates` and `F/public-evidence-second-logit-export`.
- Report `validation/reader_fresh_confirmation_complete_20261021/REPORT.md`, all-order/contrast CSVs, complete readout, verification receipts and checked PNG/SVG figure.
- Latest relevant CPU checks pass: 2,004 tests, smoke and compilation. Log `local/checks/reader-confirmation-stable-public-replay-20261021.log`, session 70352 closed 0.
- Four release assets prepared in `F/release-assets`: 14,288,614-byte archive SHA `22a4d0b64673fb4b2d34c87b2a10bff9d3e2e83e65e5bc63c5f5533634d5ba77`, JSON, public replay and SHA256SUMS. Public replay independently checks predictor token scores, stable mixture arithmetic, all 19 absolute arms, routing, document bootstrap and decisions.
- Publication is complete: release `reader-fresh-confirmation-20261021` targets result commit `72631a448218f26b2e1d54aa0ae4e1d8b9b03877`. All four assets and the report were anonymously downloaded, fsynced, read back and SHA-verified. The downloaded archive and script replay all 80 contrasts and both endpoint decisions. Receipt `validation/reader_fresh_confirmation_public_release_20261021.json`; verification session 58424 closed 0. An initial report-ref typo caused a 404 before replay and was corrected; original scientific artifacts were unaffected.
- Current bounded confirmation and publication task is complete. Next research preparation is the uniform-NTP baseline described in NEXT_STEPS.md. The frozen selector study supplies a targeted inference result and a measured negative natural endpoint; retain both.

## Earlier records and future question

The population census release is live and fully verified: `reader-population-census-20261021`, report `validation/reader_population_census_20261021/REPORT.md`, receipt `validation/reader_population_census_public_release_20261021.json`. Strict boundaries occupy 0.001664% of its source-clean frame; not all FineMath or all useful computation. Natural prediction above is measured directly, not transported from arithmetic prevalence.

Earlier hard 6/6 versus smooth 4/6 internal results, the parent failed joint criterion, V29's failed CPT criterion and V30's closed reward branch remain unchanged. The next pretraining baseline needs genuine uniform next-token loss, an unenriched source frame, separate development/confirmation and a finite measured budget. This frozen-inference result supplies neither better backbone weights nor autonomous discovery.

Preserve models, optimizer states, source text, original caches and published records. Existing environment only. Keep root free space at least 35 GiB and available host memory at least 512 GiB. Large data assets stay out of Git. Owner authorizes repository maintenance, pushes and publication.
