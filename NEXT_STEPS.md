# Current queue

1. Complete the active document-QA-v21 study;monitor health,not capability.
   Master:local/diagnostics/document-QA-v21-study-v1.
   Six tmux science sessions:document-QA-v21-<qa_direct|qa_prior|qa_post>-seed2026101921/1922.
   Physical0–11,world8×micro1×accum4,global32,width3328,target512,constant1e-5.
   Fresh released1.7B per arm;2,048 examples,four passes,256 terminal updates.
   Shared first64 updates match order/loss/norm exactly within seed;do not infer capability.
   After64 objectives differ prospectively;keep terminal checkpoint primary.
   Watchers:document-QA-v21-first128-protection and document-QA-v21-terminal-pipeline.
   They independently inspect full states,copy/read back new full recovery first,
   then retire only older EBS optimizer duplicates;keep originals/all model weights.

2. Complete the fixed evaluation orchestration while training runs.
   B0 generation is already active:tmux document-QA-v21-B0-generation,physical12,
   output local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1.
   Training snapshot049f53e,decoder dbd1c10,span scorer0d1a5b4 are immutable.
   Frozen release,protocols and stronger conventional-CoT controls:
   validation/document_QA_v21_science_release_20261019.json.
   Execution grouping:C/evaluation-execution-plan.json(C is the master above).
   Prepare six audited terminal bindings,then model-native CPU references/prompts.
   Use twelve initial trained-generation groups as planned;replay controls wait
   for each prior-only group to close and reuse its freed physical4/8.
   Use generation-group v2 controller for future groups(protocol-hash TMPDIR);
   leave the active B0 v1 controller/source untouched.
   Likelihood producer prepared:local/analysis/document_QA_v21_likelihood_job_20261019.py.
   It waits for complete original prior attempts;full native answer+EOS and
   ordinary32 targets;four batch4 shards permodel,own-model TRAIN calibration first.
   Still implement its group orchestration/closure audit and final readout.
   Reserve physical14/15 for LL while generation runs;expand after all GEN closes.
   Expected total:20,992 generation attempts and19,712 likelihood rows across7models.
   Review once after all fixed model/mode data close;no checkpoint/prompt selection.

3. Interpret separate endpoints;allow simpler controls to win.
   Primary canonical complete-answer EM,all512 questions,failures retained.
   Training:post.direct−direct.direct. Inference:prior.prior−prior.empty and
   matched direct.direct. Each>=3pp plus positive paired95%lower in both seeds.
   Practical augmentation value additionally needs gains over matched direct.prior,
   matched direct.reference_cot and unchanged B0.reference_cot;all allowed to win.
   CanonicalF1 and attributed mirror-alias scores are separate secondary measures.
   Likelihood separates answer content from EOS;ordinary-page NLL is a measured
   tradeoff,not the inherited FineMath0.03 bound. V17's own gate stays FAIL.

Evidence/protection:

- Actual training/recovery qualification PASS:14 updates,seven full states,
  fresh restore8rank digests,late save PSS growth1.46/1.53GiB<2.
  validation/document_QA_v21_training_hardware_20261019.json.
- FP32 decoder PASS:six TRAIN attempts,zero execution errors,CPU-long maxLP
  error0.0000651<0.001. Two parser-unsupported outputs retained;not accuracy evidence.
  validation/document_QA_v21_decoding_hardware_20261019.json.
- Span B4QA3584/ordinary672:CPU/device parity passes;max token errors
  0.0000799/0.0000584. Warm9.00/0.710s versus single CPU reference69.10/7.14s;
  not a sustained benchmark. Independent publication/closure audit still to finish.
- Cohorts:2,048 distinct train/512 held,allcomplete inputs<=3072;no held supporting
  text/page/actualseedID in training-visible evidence. Distractors/templates recur;
  semantic/pretraining novelty not established. All gold step answers in cited text.
- Raw component strings sometimes generic relations;V21 uses true dataset:seed IDs.
  Canonical author annotations match pinned converter;mirror aliases remain attributed,
  not independently authenticated against official ZIP. OfficialDEV/TEST unopened.
- Keep35GiB/filesystem and512GiB hostavailable. All original full states remainNVMe.
  Latest EBS full is V17 source_direct1913/320 UNTIL first128 watcher records
  new-full readback and older-optimizer retirement. Then use its explicit receipt.
  Root~83GiB pre-first128;all-state/model protection budget verified before release.
  EBS is same-instance,not off-instance. No cloud/IAM/network/paid-teacher changes.

Closed boundaries:V17 arithmetic gatePASS/quality+combinedFAIL;V15 inference gain
with NLLcost;V10 both primary full-span gatesFAIL;V11 bounded repair closed;
old R1234 remains stopped63. Details in README/EXPERIMENT_HISTORY;no new RL/8B release.
