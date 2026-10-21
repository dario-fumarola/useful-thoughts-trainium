# Active uniform-LM preparation — October 21, 2026

Owner endorsed returning to a sound uniform next-token baseline. R=/home/ubuntu/dario. Current work is CPU source preparation and a generic plain-LM branch in the existing engine, not a continuation of selector fitting. No new subagents.

- Source plan `configs/uniform_lm_sources.json`: TRAIN 32,768 / DEV 1,024 / CONFIRM 4,096 / QUALIFICATION 1,024. Identity/length-only split/cut, source-clean existing 16-shard frame, all prior reader/fresh sources excluded. No new source download.
- `train/uniform_lm.py`: native all-position packing, global accumulation indexing, explicit optimizer-step warmup/constant/cosine schedules. Existing `partial_state_driver.py` dispatches it, sets LR before each optimizer update and uses actual global batch for metric/resume cursors. Original frozen experiment worktrees remain unchanged.
- Targeted CPU checks pass (48): uniform dense CE/all gradients, distributed accumulation, all-source coverage, schedule/resume and source sealing, plus original V29 tests. Full suite next.
- Design has four development settings (two peak LRs x two schedules), same seed/order/exposure and terminal128. Choose by DEV full527 NLL only after all terminals freeze, then pin checkpoint/recipe before CONFIRM. One model/order; no broad robustness claim.
- Native fitting still needs actual same-shape accumulation/scheduler/recovery qualification and a measured finite projection. Proposed 18 total chip-hours (qualification <=0.5), two existing chips maximum, are not an execution release. Full checkpoints fit in existing /dev/shm under a 180-GiB bound; four BF16 terminal copies fit existing nvme-cpt under16 GiB while preserving35 GiB free and512 GiB host memory. No protected-state deletion.
- Next: commit source rules/code before reconstructing data; start `scripts/prepare_uniform_lm.py` with a 30-minute CPU deadline into `local/data/uniform-lm-baseline-v1`. Then run `scripts/check_uniform_lm_sources.py` and full CPU checks. Record exact processes and outputs before compaction.

The preceding confirmation/publication is complete; its full handoff follows as history.

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
