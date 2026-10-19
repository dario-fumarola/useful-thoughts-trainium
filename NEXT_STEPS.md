# Current queue

1. Finish the fixed single-model real-document evaluation. All six raw/prior/post
   training runs completed320 updates and physically closed. Independent checks
   pass all12 full160/320 states,48 native vectors and1,920 ordered updates.
   All six terminal weights and the latest full320 state are EBS-protected.
   Latest full: real_raw seed2026101914. No additional training or model selection.
   Public training report:validation/real_trace_cpt_v10_terminal_training_20261019.json.

   Control:local/diagnostics/real-trace-cpt-v10-study-v1.
   Immutable training3d8666b; evaluation29c8b24,1,646 CPU checks.
   The release watcher completed all six fixed launches;
   cpt-eval-* generates its original notes, cpt-score-* scores full native16 spans.
   B0 scoring: cpt-v10-B0-CPU-scoring; its ordinary reference is explicitly reused.
   Final archive/readout: cpt-v10-final-readout. Do not duplicate these jobs.
   Prepared readout checker/renderer in local/analysis:
   check_real_trace_readout_v10_20261019.py and render_real_trace_readout_v10_20261019.py.
   After all complete, independently check signs/counts/gates, inspect figures,
   save deterministic examples, then publish complete results and interpretation.

2. Finish the already-declared scalar-continuation secondary. Same seven models,
   same original notes and two separate arithmetic panels; five modes,greedy32.
   Existing initial-scalar-v2 remains unchanged; every error/unsupported/cap stays.
   Public design:validation/real_trace_cpt_v10_scalar_secondary_design_20261019.json.
   Source:local/worktrees/real-trace-answers-v10 at4297c1d;1,657 full CPU checks,
   latest11 affected checks after tighter original-binding validation.
   B0 runs on physical12; actual four-core native/cache672 checks pass.
   Tmux cpt-v10-scalar-release waits for each model's complete original notes,
   then releases its two panels on physical6–11. Each model checks itself again.
   Control:local/diagnostics/real-trace-v10-scalar-secondary-study-v1.
   Tmux cpt-v10-scalar-readout archives/rechecks all17,920 attempts once complete.
   Scalar accuracy is secondary and does not change either likelihood gate.
   Prepared figure/CSV renderer:render_real_trace_scalar_secondary_v10_20261019.py.
   No broad QA, equal-compute or controlled service-latency claim follows.

3. Keep the bounded variable-depth diagnostic closed. Published report:
   validation/variable_depth_writer_v11_complete_20261019.json.
   Composition512: fixed124→variable306, paired+35.5pp[31.3,40.0]; gate passes.
   Canonical/naturally stopped states91→409. Frozen fullCoT273, but different
   curricula/supervision and one diagnostic seed prevent an architecture claim.
   Freshdepth2: variable182/256 versusCoT193; unseen depth6: only10/256.
   All9,216 answers/2,048 notes, controls, figures and examples are preserved.
   Do not launch a larger split rescue or broad RL.

4. Broader document reasoning stays at CPU preparation. MuSiQue TRAIN inventory
   uses composed_question_text and all visible paragraphs; no gold support flags
   or decomposition enter model inputs. Corrected inventory:19,938 questions,
   306 complete inputs<=1536 tokens and4,149<=2048. No model/cohort release.
   Source7c5c264 and corrected readiness are public; official DEV/TEST untouched.
   CPU split audit finds99% of hypothetical held questions reuse a training-visible
   support paragraph. Distinguish question/evidence novelty and include question-only
   controls; validation/document_evidence_v12_split_risks_20261019.json.
   Choose the next scientific intervention after the complete mainline readout.

Fixed mainline data: TRAIN5120 unchanged, SHA437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915.
Primary held V2 manifest22bd05cb3ad07e08a17e30578bebe07629216c9ad3910e9270bf0794b34dff24:
local/nvme-study/data/real-trace-cpt-v10-held-panels-v2.
All256 ordered ASTs are unique and absent from TRAIN. The ten-alias correction
was frozen before learning. Ordinary256 and legacy256 remain separate.
Training and inference gates are independent; report both seeds and absolute B0,
equal-mode controls, ordinary-text regressions and actual token costs.

Storage: keep35 GiB free per filesystem and512 GiB host availability. All original
full checkpoints remain on NVMe. EBS keeps all terminal models and latest full320;
older inactive optimizer duplicates retire only after fresh both-side full hashes
and verified newer protection. Never remove an original or model weight.
local/nvme-cpt is already formatted/mounted, UUID5119e7ab-0ca6-4295-b8f0-e856d87b4b67.
NEVER mkfs again; remount UUID only. EBS is same-instance, not off-instance backup.
No cloud/IAM/network/paid-teacher/bucket changes. R1234 stays stopped at63.
No unqualified8B or officialTEST. Agents remain quota-unavailable; do not retry.

Future RL engineering only, no launch:retired replay still computes all1024slots
for empty notes. A single tiny CPU fixture confirms8-slot cropping preserves
loss/reader scores/gradients; no Trainium timing or complete bucket qualification.
Public:validation/retired_R_empty_padding_CPU_20261019.json. Profile actual phases
and qualify all masks/stops/behavior-logp cases only if a future RL study is justified.
