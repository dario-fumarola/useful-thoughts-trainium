# Current queue

1. Finish publication of the complete V17 result, then specify one broader question.
   Report: validation/real_source_balanced_v17_complete_20261019.json.
   All11 models closed; independent checks and readback protection pass.
   The3 PNG figures have been visually inspected. All45 raw payloads remain.
   Arithmetic:434/439 of512 versus351/357 direct and341/326 empty;
   gains+16.2pp[12.1,20.3]/+16.0pp[11.9,20.1]. Arithmetic gates pass both seeds.
   Ordinary NLL:+0.0380/+0.0429; upper CIs0.0486/0.0554 exceed0.03.
   Combined gate FAILS both. Preserve that conclusion and the original threshold.
   Regression falls75.6% versus old target-only CoT on this same new panel.
   Full16 remains below B0; raw arithmetic does not clearly beat matched direct.
   All28,160 answers/5,632notes/30,976LL rows, failures, costs and14 examples saved.
   No broad RL, model-size jump, split rescue or automatic follow-up release.

2. Specify one single-model document-QA reference.
   North star: compute from the available documents and improve fresh answers
   at worthwhile cost. Keep training benefit and extra inference as separate endpoints.
   First inspect shared-evidence clusters/bootstrap units and compact trace lengths.
   Then freeze an evidence-clean TRAIN/DEV cohort, strong direct/CoT controls,
   complete-answer parser/metrics, data exposures and practical thresholds.
   Select the scientific intervention after reviewing V17; no document model is released.
   Before training: actual long-context backward, memory, periodic full save/restart,
   gradient reduction and timing must qualify under a new experiment identity.
   Preserve the working1.7B stack. Keep full passages; never silently truncate context.
   Do not reuse the128-token math cap for copied support notes(median279 tokens).

Data ready, with limits:

- Author MuSiQue TRAIN has19,938 questions. Policy projection contains only the
  authored question and all visible paragraph titles/texts, excluding gold fields.
- V16's fixed held512 and support-text/title/component-question exclusion leave
  2,983 training questions at3072 complete-input tokens, or502 at2048.
  These are feasibility cohorts; shared evidence within held questions still needs
  a suitable uncertainty unit. No final cohort or long-context model is released.
- V19's pinned TRAIN-only mirror matches every canonical field of all19,938
  author raw records through the pinned official converter, including decompositions.
  It supplies8,872 aliases across5,067 questions, all visible in the passages.
  Alias authenticity is not independently verified against the unavailable official ZIP.
  Keep canonical/alias provenance separate; alias labels are offline gold only.
- Read validation/document_evidence_v16_directed_split_20261019.json and
  validation/document_aliases_v19_TRAIN_readiness_20261019.json before cohort work.
  Official DEV/TEST and the global aliases file were not opened.

Preserve completed boundaries:

- V15 establishes supervised arithmetic benefit with substantial ordinary-NLL cost.
- V10 fails both primary full-span gates; its prospective scalar secondary helps.
- V11's bounded variable-depth repair is closed. No expanded split rescue.
- The original R1234 stays stopped at63. Never resume its old launchers.
- Historical tables and interpretation limits live in docs/EXPERIMENT_HISTORY_20261019.md.

Resources and protection:

Keep35 GiB free per filesystem and512 GiB host availability. All original full
checkpoints remain on NVMe. EBS retains all terminal models and the latest global
full optimizer state: V17 source_direct seed1913 at320. Same-instance protection,
not an off-instance backup. See docs/CHECKPOINT_BACKUP.md.

At publication, root free space is about45.6 GiB. Recalculate the next checkpoint
budget before another scientific release; it cannot yet accommodate a new full
copy while keeping35 GiB free. Older inactive optimizer backup duplicates may
retire only after fresh both-side full hashes and verified newer protection.
Never remove an original or a model weight. Do not format any mounted NVMe volume.
The compile cache already lives on NVMe; no stack upgrades or serving migration.
No cloud/IAM/network/paid-teacher/bucket changes. Agents remain quota-unavailable
until October24; do not retry. Existing-node work and public repo pushes remain authorized.
