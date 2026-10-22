# Active overnight work — October 22, 2026

The owner explicitly requires continuous autonomous work until **08:00 Eastern / 12:00 UTC**, on the existing node and beyond baselines. Do not end early with a progress-only final. No new agents. Read STATUS.md and NEXT_STEPS.md. Preserve every source, model, optimizer state and historical outcome.

## Completed ordinary baseline

R=/home/ubuntu/dario. C1=R/local/diagnostics/uniform-lm-native-v1. Numerical source 4f9166b944fcc3a7e38df6f32bd45b396d5047ef in local/worktrees/uniform-lm-counter-cache-v1. All four models, DEV choice, fresh CONFIRM, old calculation diagnostic, full readout and NumPy replay are complete. Marker C1/study-coordinator-complete.json. No baseline worker remains active.

Fresh full527 confirmation gain **0.014481206232314262 [0.01390327765920759, 0.015058117946887212] nats/token**; B0 NLL1.4435850404561168, cosine1e-5 model1.4291038342238027. Suffix16 gain0.032716646160013844 [0.013546082354470326,0.05227450553831186]. Actual16.607275863178817 assigned chip-hours within18, including every failed qualification reservation. Full states and four model copies retained. Report validation/uniform_lm_baseline_complete_20261022/REPORT.md. Public source-free package local/releases/uniform-lm-baseline-20261022.tar.gz; publication and anonymous download/hash/replay verification are complete. Receipt validation/uniform_lm_baseline_public_release_20261022.json. Release targets bba4837, tag uniform-lm-baseline-20261022.

## Running matched supervised-learning study

C2=R/local/diagnostics/scaffold-learning-overnight-v1. Active coordinator PID2752939 (session28445) runs C2/run_study.py, log study-coordinator-port-safe.log. It launched raw/format/trace seed11 at03:44–03:45UTC, max3 concurrent two-chip trainers; seed12 follows. Native stage launch/closure receipts are under C2/science-seed11-control/jobs and science-seed12-control/jobs. Training source53adb6e in local/worktrees/scaffold-learning-overnight-v1. Qualified numeric sources must not change.

Six fresh models start from B0, common34,816 windows (32,768 ordinary+2,048 calculations), all527source targets equal and original text preceding optional work. Raw no work labels, formatEOS, traceverified prefix-derived work+EOS. Source targets18,348,032/model. Auxiliary labels0/2,048/73,210. World8, micro2, accumulation17, batch272, width704, targetslots656,128updates,13warmup,cosinepeak1e-5 selected by completed baseline. BF16param/grad, FP32masters/moments, one reducer. Two source orders11/12. This is a post-hoc supervised learning scaffold, not a prefix-only autonomous writer. All six terminals freeze before any new model confirmation score.

Native qualification PASS with exact full-state continuation. CPU/native initial error0.0004257947644765725; actualqualification1.5299846480927792 assignedchip-hours. Source-data independent checks pass all5,120 fresh natural rows and2,112 calculation rows;5,722verifiedsteps. Data in /dev/shm/scaffold-learning-overnight-v1-data. TRAIN34,816;QUAL1,088;freshCONF4,096. All2,035CPUtests,smoke,compilepass (local/checks/scaffold-final-20261022.log).

Resource amendment before long fitting: globalhard40 assignedchip-hours, measuredprojection39.44860793628597. Qualificationphase≤4,seed11≤18,seed12≤18,scoring≤4; maxima cannot all be spent and globalwatchdog sums everyphase. Softstop39.6,hard39.98or11UTC,publication12UTC. Existingnodeonly. Pairassignments0/1,4/5,8/9,12/13;CPUaffinity followsNUMA. Freeportpairs61000–61990. Preserve35GiBfreefilesystems and512GiBhostavailable. FulltrainingRSS~128GiB/pair; three leaves checkpoint-export headroom. Do not increase to six full trainers.

C2/evaluate_and_readout.py is ready, launched by master after all terminals validate. Scoring source b55349f in local/worktrees/scaffold-learning-scoring-v1, with actual imported original worker SHA pinned. Child PYTHONPATH must point to scoring worktree. Eight models (B0,qualifieduniformreference,sixnew) score freshCONF4096 and oldCALC1024. Every original FP32token score and128-updatehistory is exported. Central unique-document bootstrap5000seed17,all26comparisons. Completed readout C2/complete-readout.json. Master completion marker study-complete.json.

C2/replay_scaffold_learning.py is drafted but needs synthetic integrity qualification and actual source-free replay. Human report generator, result publication and anonymousdownload/hash/replay remain to implement. Complete these even for negative results.

## Other work and storage

Optional writer-feedback pilot C3=R/local/diagnostics/writer-feedback-pilot is NOT released or implemented. No actor code, fit or model calls. First inline saved-score diagnostic failed on input shape before output: central stats requires NxK. If pursued, first test the finite TRAIN-only opportunity under historical training-fitted reference-preserving position mixture. Do not jeopardize C1/C2 completion with speculative infrastructure. Details remain in the archived handoff.

The node has16physicalTrainium2chips,64logicalcoresunderLNC2. Assigned-chip-hours are not whole-node billing. C1/whole-node-utilization.jsonl is a read-only monitor until12UTC (session16559). Hostmemory is the concurrency limit; /dev/shm retained states will approach950GiB after C2. Do not add more full optimizer states or model copies without a space plan. Rootfree~36GiB, /dev/shmfree~180GiB before sixnewterminals. Preserve all protected oldV29states, baseline and qualification snapshots. CPUcommands: source scripts/env.sh; USE_TORCH_XLA=0. Python /home/ubuntu/ema-trainium/.venv/bin/python. rg unavailable. No paidteachers, officialTEST, capacitypurchase, IAM/network/stackchange.

[Prior detailed execution handoff](archive/AGENT_HANDOFF_through_uniform_confirmation_20261022.md).

## Saved writer diagnostic decision, 03:56 UTC

C3/saved-candidate-diagnostic.json now completes the TRAIN-only analysis with no model calls. Historical fixed position mixture gains0.228166nats/span versusB0; hindsightbest8adds0.026617[0.008632,0.054787] versusgreedy. ActualB0null+hindsight adds0.069746[0.047502,0.099832]. Most marginal opportunity comes from abstention rather than improved note selection. Do not spend on a new writer run solely from this small diagnostic. Main priority remains the six matched learning models and full publication. Script C3/saved_candidate_diagnostic.py. No actor code or job released.
