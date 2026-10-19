# Current queue

1. Finish the bounded document-QA-v21 long-context qualification.
   Frozen source049f53e: local/worktrees/document-qa-v21. Do not mutate it.
   Control: local/diagnostics/document-QA-v21-world8-width3328-qualification-v1.
   Tmux: document-QA-v21-qualification; supervisor.log and stage logs track it.
   World8×micro1×accum4, global32, width3328, target512, fresh released1.7B.
   Disposable first128 TRAIN only: prior1–6, fresh restore6→8, post1–6.
   Audit all14 updates, source/task counts, seven full states, restoration,
   grouped PSS, actual compile/warm/save time and physical process closure.
   Qualification has a17:00UTC engineering deadline, not an instance shutdown.
   No automatic retry or scientific release follows a completion marker alone.

2. Freeze the single-model QA scientific comparison after actual qualification.
   North star: compute from available documents and improve fresh complete answers
   at worthwhile cost. Separate training benefit from extra inference benefit.
   Prepared arms: direct answers; complete reasoning before answers; reasoning
   after a supervised answer EOS. Augmented arms alternate plain/augmented passes.
   Same source sample and ordered documents; source/task means receive0.5 each.
   Count source labels, task labels, exposures and actual accelerator cost separately.
   Declare complete-answer EM/F1/parser, strong B0 direct/CoT references,
   evidence-removal/question-only controls and practical thresholds before forwards.
   No official DEV/TEST, broad RL, model-size jump or split rescue is released.

Data and CPU evidence:

- validation/document_QA_v21_CPU_readiness_20261019.json:2,048 unique TRAIN and
  512 held questions, complete inputs<=3,072 tokens. All6,144 native views fit
  width3328/target512 without truncation.1,711 CPU checks and independent audit pass.
- Held supporting text/page and actual original seed IDs are disjoint within held
  and absent from all training-visible evidence. Distractors/templates may recur;
  semantic and pretraining novelty are not established.
- V16/V20 raw component_question_texts include generic relation programs. Their
  old tables remain unchanged; V21 uses actual decomposed_instances dataset:IDs.
- Gold traces include the root answer, citing visible passages. All intermediate
  answers in the frozen cohorts occur in their cited passages. This is supervision,
  not deployed evidence selection or semantic entailment.
- MuSiQue author TRAIN and pinned mirror canonical fields agree on all19,938 rows.
  Mirror aliases are attributed gold; not independently authenticated against the
  unavailable official ZIP. Official DEV/TEST/global alias files remain unopened.
- Data/source/evidence protected at local/backups/data/document-QA-v21-TRAIN2048-held512-v1.

Completed scientific boundaries:

- V17 public e1bbff9: arithmetic inference gates PASS both seeds; ordinary-quality
  guard and combined gate FAIL both. Source-CoT434/439 of512 vs direct351/357.
  Ordinary NLL+0.0380/+0.0429 versus B0; thresholds remain unchanged.
- V15 supervised arithmetic gain has substantial ordinary-NLL cost. V10 fails both
  primary full-span gates; its prospective scalar secondary helps. V11 bounded
  variable-depth repair is closed. Original R1234 stays stopped at63.
- Full results, controls and limits:README and docs/EXPERIMENT_HISTORY_20261019.md.

Resources and protection:

Keep35GiB/filesystem and512GiB host availability. Never remove an original state
or model weight. Current complete EBS optimizer recovery is V17 source_direct
seed1913 at320;all scientific terminal models remain protected. Same-instance,
not off-instance. Before another science release, verify full-copy-FIRST space.
Fresh both-side checks verified the original C seed1234 step128 and latest V17
full states before retiring only inactive C128 EBS optimizer duplicates. All
originals and models remain. Root now~64.6GiB free:enough for one new full copy
with35GiB reserve; recalculate remaining terminal-model protection before science.
Receipt:local/diagnostics/document-QA-v21-backup-budget-v1/completion.json.
No cloud/IAM/network/paid-teacher/bucket changes, stack upgrades or live cache moves.
