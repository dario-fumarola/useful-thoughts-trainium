# Current queue

1. V15 is published at e5cdc19; preserve its completed findings and costs.
   All nine models and both endpoints are complete; independent checks pass.
   CoT own342/335 of384 beats direct-SFT raw275/285 and same-CoT empty261/263.
   Paired gains +17.4pp[13.3,21.6] / +13.0pp[8.9,17.2]; declared gates pass both seeds.
   Full16 improves over direct SFT but remains below B0. Ordinary NLL regresses
   0.128/0.147 nats/token; this is not a broad prediction or pretraining improvement.
   Report: validation/real_target_sft_v15_complete_20261019.json; raw payloads,
   checked figures, 15 disclosed examples, all error categories and costs saved.
   Independent readout SHA f8bc04e2c759cbad2a9438754aee828d5f99c6b4747dc41198fa1e4d3a43e033.
   Terminal states and all original outputs/readouts are readback protected.
   Source829d9a2/evaluation41a5bfd remain frozen; producers are physically closed.

2. Finish source-balanced V17 qualification and fresh confirmation preparation.
   Frozen branch37028a1: 1,656 CPU checks and10,240 real native packs pass.
   Reuse existing two-component CE:0.5 source-prefix mean +0.5 task mean perexample.
   Same TRAIN5120/two passes/320updates;source_direct/source_cot,world8×micro4,
   width640,target640. No new scientific training release yet.
   Hardware control:local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2.
   Tmux source-balanced-v17-qualification-v2;actual8workers onphysical12/13.
   CoT1–6/fullrestore6→8 physically closed;direct1–3 still active;independent audit pending.
   V1 PATH-only failure before allocation is retained. Small evidence backedup;
   original qualification full states/models remainNVMe;scientific EBS reserve protected.
   Independent auditor prepared:local/analysis/audit_real_source_balanced_qualification_v17_20261019.py.
   It must verify independent source/task label formulas andweighted loss,allstate bytes,
   restored optimizer digests,memory andphysicalclosure before declaring hardwarePASS.
   Confirmation: pinned HF FineMath4+ TRAIN shards63/62 now complete;1,838 verified source computations.
   Pipeline session useslocal/analysis/*real_source_confirmation_shard_v17_20261019.py.
   Current private source/curation dirs endfresh-source-shard62-v1/fresh-curation-shard62-v1.
   Frozen512 unique fitting expressions plus256 ordinary new documents byfixedhash,
   excluding oldTRAIN/DEV ASTs/pages;keepoldTEST/unlistedtext opaque. No model-outcome selection.
   Design/datahashes published;prepare protected baseline evaluations,confirmstoragebudget,
   releaseonlyqualifiedfourfreshmodels afteractualauditPASS.
   No broadRL,sizejump orsplitrescue. V15 positive inference/negativeNLL result staysunchanged.

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
   Directed held-support exclusion is feasible:at2048/3072 complete-input tokens,
   excluding held support text/title/componentquestions leaves502/2,983trainrows;
   excluding every held-visible paragraph leaves182/1,221. Both hypothetical
   cohorts hold512questions;no model or cohort released. Source/report protected:
   validation/document_evidence_v16_directed_split_20261019.json.
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
