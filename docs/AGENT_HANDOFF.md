# Current handoff — 2026-10-19 20:27 UTC

V23 is CLOSED and published on main2c4711d. All1530GEN/3062LL checked/protected;
primary/same-reader-direct gates FAIL. Supplied-empty control regenerates IDs;
original scores/report unchanged and caveat prominent. No replication/training release.
V24 bounded TRAIN-fit generation active,exec65150,physical12/logical48–51;
C=local/diagnostics/document-selector-fit-v24-TRAIN256-v1, O=matching evaluation path.
Frozen local adapter local/analysis/document_selector_fit_v24_20261019.py;
wait C/GEN/closure.json,then source scripts/env.sh and run adapter analyze with
PYTHONPATH=local/worktrees/document-evidence-eval-v23:local/test-deps (absolute paths).
No output-content review until physical closure. Native/cache own-model checks still required.
Deadline21:00UTC,no retry. Main1797CPUchecks/smoke/compile pass tests.DKuZIN.
Exact current paths/session are in local/analysis/CURRENT_V23_SESSION.json.
No parameter updates or new held cohort. Protect/readback outputs and publish diagnosis.

---

# Handoff — 2026-10-19 18:11 UTC

## Latest state at18:11 — continue,do not final

Training qualification C/O below remains active. First update completed after
1225.86s initial compile:loss1.7493615,gradient_norm44.4267,allactual auditsPASS.
Step1 components:source1.8028257/8192labels,answer1.5516232/261,selection.7213700/155;
weights .5/.5/.1. Step2 compiles MODULE8694244300497888072;no failure orcheckpoint yet.
W1 training source remains FROZEN7328141,supervisorPID1358970,exec50403.
Audit watcher86358 waits allthree closures,thenCPUauditonce;do not relaunch.

Decoder qualification COMPLETE/PASS andphysicallyclosed. SourceW2commit
625fb0f6d6c28c11a02d60a9c578d6a5f1cad603 pushed onqa-evidence-eval-v23;rootmain
fast-forwarded locallyto625fb0f,remote main still7328141 until nextpubliccommit.
W2=local/worktrees/document-evidence-eval-v23;all1779checks/smoke/compilePASS,
logs W2/local/checks/tests.42TmmE. Do not unnecessarily repeat.
Decoder control DQ=local/diagnostics/document-evidence-plan-v23-decode-qualification-v1.
Four TRAIN B0 attempts,statusok4,executionerrors0,parser complete1/unsupported3,caps0.
Native/cache/CPUlong checkPASS;maxCPUlogperror.000114845<.001. Notcapabilityresults.
DQ/independent-audit.json PASS;arrays/control/source protectedEBS
local/backups/engineering/document-evidence-plan-v23-decode-qualification-v1,
ARCHIVEsha9a4294307c40c1204a16d47f08a3943f8859e8e1d830744464d0be036a4cbf49.
Public report+annex prepared validation/document_evidence_plan_v23_decode_qualification_20261019.json,
currentlyUNCOMMITTED;include in nextreviewedmainpublicationwith updatedSTATUS/HANDOFF.
Decoderexec96707 andCPUref83944/audit75278 are finished(orready toclose);no decoderGPU actor left.

Secondary likelihood diagnostics are nowprospectively specified,no model calls:
local/diagnostics/document-evidence-plan-v23-CPU-v1/prospective-likelihood-replay-plan.json.
Use canonical replay of valid generated pointerIDs in originalorder,notnative-note LL.
Invalid/missing/error plans fallback toEMPTY with explicitflag;never paste generated
answers/prose intoLLcontext. Other=next-questioncyclic donorIDs,not guaranteedunrelatedfacts.
Oracle=annotatedIDs only. Scoreall255 fullcanonicalanswer+EOS,EOSseparate;no droppedfailures.
All255answers fit32(max13);contexts fbd8605b07d14af1b5b9ef836b6b798baf4206988fc31b431122f01f3f7e832a
in local/analysis/document_evidence_v23_likelihood_contexts.py. CPUchecker
check_document_evidence_v23_likelihood_contexts_20261019.py PASS765native spans;
result local/diagnostics/document-evidence-plan-v23-CPU-v1/likelihood-context-check.json.
Same directory fresh255-target-span-inventory.json PASS;no actual model readouts.

Next useful work while compiler runs:implement frozen likelihoodprepare/run orchestration
using unchanged qualified train/document_qa_span_scores.py,batch4,QA3584/ordinary672,T32,
own-model TRAIN calibration .001 before any heldscore. Old v21 LLworker is reusable
local/analysis/document_QA_v21_likelihood_job_20261019.py buthardcodes512/rowcounts;
new255 requires physicalbatchpadding(lastrow duplicate,drop ONLY synthetic outputs,
logpaddingcount),alloriginal255 retained. Ordinary256panel stays samecostguardrail,
not generalLMnovelty. Freeze endpoints/source BEFORE science launch.
After full trainingqualification/auditprotect/publish,review concrete two-arm freshB0
pilot protocols+budget (256updates,seed2026101923,terminalprimary). Newsource8-update
qualification requiredstatus PASS_ACTUAL_DOCUMENT_EVIDENCE_PLAN_FULL_RESTORE.
No freshheld model calls,256-update pilot orconditionalreplication yet.

## Additional active work since17:41

Training source remains frozen at7328141;qualification still compiling first
update at17:52:36,8 initial native checks PASS,zero completedupdates/errors yet.
Compiler actively used CPU and moved from MODULE18073870128203333353 to
MODULE11834708709892656423;other workers wait on its shared lock. Do not rush it.
Hostavailable~1914GiB;compile ownerPSS~22GiB,workers~4GiB each;healthy.

Independent hardware audit already prepared C/independent-audit.py SHA
1022310e1618ad31b753d9694eb8519d8b9f5049b3dbde29f5983961e150a42e.
Watcher session86358 running local/analysis/watch_document_evidence_v23_qualification_audit_20261019.py.
It waits for all3 physical closures,then runs audit exactly once (CPU),no retries;
logs C/independent-audit.log and independent-audit-exit.json. Stops on originalfailure.
Audit reconstructs all native source/final/selection counts independently,checks
all6 full states/moments/casts andrestore4 digests,24 native witnesses,thenEBS
small evidence. No post-hoc PSSgrowth gate;observe memory andpreserve512GiB floor.

Evaluation development is ISOLATED in local/worktrees/document-evidence-eval-v23,
branch qa-evidence-eval-v23 from7328141;uncommitted,never edit trainingW while active.
New document_evidence_plan_answers.py separates plan syntax from answer scores:
complete requires nonempty unique in-range IDs in one Evidence field beforeFinal,
no extra factual prose;order may differ. Empty controls remain separate.
New document_evidence_plan_evaluate.py prepares projected prompts and delegates
to existing document_qa_evaluate.generate;sharedengine accepts bound schema/sources
and128/256 max_actions. ALL new paths use prior-mode anchoredFinalanswer parser.
Empty/oracle prefixes include ONLY Evidence field,not supplied Finalanswer header;
oracle adds annotatedIDs,never answertext. Release must bind model_specs lineage.
Only26 focused parser checks completedPASS sofar. Full evaluationCPU suite running
exec session5921,log local/analysis/evidence-plan-v23-evaluation-all-CPU-checks.log.
CPU-only decode preparation exec69507 is running --qualification (existingTRAIN
first128,shortest/longest2 questions,modesdirect/selection,B0only). Output
local/diagnostics/document-evidence-plan-v23-decode-qualification-v1;log
local/analysis/evidence-plan-v23-decode-preparation.log. NO decoder GPU job launched.
Next:inspect fullsuite/preparation;freeze/commit evalsource,then bounded4-attempt
TRAIN-only cache/decoder qualification on freephysical14/logical56 inparallel
with trainingphysical12/13. Actualworker/native/cache/closure proofs needed.
No held model forward or256-update science release yet. Needspan scoring/readout
preparation andfinal concrete two-arm resource release after hardwareaudits.

## Active hardware qualification — inspect this first

CPU source public at73281416f9f5c8ab953e279f8940479ab6bbe384;main and
qa-evidence-v23 pushed. Worktree local/worktrees/document-evidence-v23 is FROZEN.
Do not edit its train sources while qualification runs. Main docs may update.
C=local/diagnostics/document-evidence-plan-v23-world8-qualification-v1.
O=local/nvme-cpt/diagnostics/document-evidence-plan-v23-world8-qualification-v1.
Supervisor PID1358970,start_ticks26628280;exec session50403. First owned child
session1359025,start_ticks26628290. Start17:39:50UTC,hard deadline19:00UTC.
Logical48–55 physical12/13 only,one stage at a time,max3reserved chip-hours.
Stages:select-fresh4 halt4;physicalclosure;select-resume8 from4 attempt
actual-restart-001;fresh evidence_direct halt3,output O+'-direct'.
8 initial native receipts already saved at17:40:43;first update compiling/running,
no completed update yet. No held model forward or256-update science release.
Protocol SHA183447dd98f8ada9ca28554a21235f98d099ce5697ec9725736e696c3ed36df5.
Supervisor SHA0ca1db85134c740221a835e07067a91c3b042ea3fc3fa0f3d0d26d66dd1ce0d7.
Next command:tail local/analysis/evidence-plan-v23-hardware-supervisor.log and
C/select-fresh4/run.log;read O/metrics.jsonl and stage exit/closure files.
Supervisor has bounded owned-session cleanup,no automatic retry. If failed,
preserve stage and repair under a new attempt identity;do not reexecute producer.
After all stages close,adapt the existing independent QA qualification audit for
new three masks/weights(.5,.5,.1),interleaved rows and restore4→8. Verify native
witnesses,ordered masks/component arithmetic,counters,checkpoint digests and
FP32 master/moment/cast consistency,then protect small actual evidence onEBS.
CPU data/source already protected atlocal/backups/engineering/document-evidence-plan-v23-CPU-v2/READBACK.json.
All original V21 states/results untouched;oldR1234 remains stopped63.

## Current evidence-plan-v23 preparation

Source worktree local/worktrees/document-evidence-v23,branch qa-evidence-v23.
CPU readiness validation/document_evidence_plan_v23_CPU_readiness_20261019.json.
All1,763 CPU tests/smoke/compile PASS;new driver/CE regression suite complete.
Source code adds separately normalized named components through the existing CE
core,retains two-component API and existing production driver. No new runtime.
All6,144 native views fit;source and final labels identical in every new view.
Four-pass inventories both arms:source2,097,152,final67,408;selection8,192/39,930.
Data local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2;
manifest bd74add7d32df6d591542defc8ab93458d21b2d12c88783f2b9047f494c6910e.
Independent source/label check PASS.255 fresh=205two/46three/4four.
V1 strict fixed metadata selector reached255/256 and failed;V2 explicitly uses
same255,unchanged rank/exclusions/accuracy gate.OriginalV1 plan/failure preserved
local/diagnostics/document-evidence-plan-v23-CPU-v1 and public readiness annex.
No model evaluated on new panel. TRAIN-only qualification now active as above;no science release.
Next:protect CPU data/evidence and commit source,prepare bounded TRAIN128 actual
hardware qualification,pointer fresh1–4/restore4→8 plus direct control1–3;
then audit update/masks/full-state recovery and actual memory. Existing qualified
runtime/environment/topology supervisors can be reused;do not execute old jobs.
Science requires new source/shape/objective qualification and reviewed run budget.
Measured V21 final-token coefficient ratio:prior augmented mean.0928555,
four-pass.5464277;post.0895638/.5447819. This excludes extra root-answer trace
labels and is not total gradient/causal attribution. Evidence in readiness annex.

Earlier next-action notes below describe the completed V22 audit/design step.

User saysworkautonomously untilstopped;weeksavailable. Do not final atmilestones.
Noactivegoal;do notcreateone. Agentquota unavailableuntilOct24,do notretryagents.
Shortreasoning/tool/actioncycles;commentary atleastevery60s. Currentphasepublication
andscientificinspection,not moreruns. NoGPUjobs active;allv21producers CLOSED.

## Current next action: evidence-selection CPU design

V22 TRAIN audit COMPLETE and readback-protected. Public report
validation/document_QA_v22_TRAIN_audit_20261019.json SHA
4c0c4443305bcda5f1d1d5da9062680ed0c195374cd39c0366e17be9205202f3.
24 fixed-rank TRAIN questions/eight per hop;72 cited relations:57 E,8 I,6 A,1 C.
E=locally supported;I=citation insufficient;A=ambiguous;C=contradiction. One agent,
exploratory subset selected after viewing cited traces;not independent gold/error-rate estimate.
Four questions conflate cross-state entities;all72 passages/answers verified against
their own author-raw step contexts and frozen pack. Not our citation-index bug.
All24 manual judgments,72 passages,source scripts and24 availability conditions public.
EBS local/backups/evaluation/document-QA-v22-TRAIN-audit/READBACK.json PASS (10 files).
Raw and allV21 data/primary report unchanged.

At3072:9382 candidates exclude existing TRAIN but not old held evidence. Adding old
held support exclusion leaves658 (531 two/119 three/8 four);all-old-visible exclusion
leaves622 (496/119/7). Page/seed/paragraph independent hash-greedy finds253 for latter,
not an optimal bound. No new cohort frozen. Repeated answers allowed,not official S5.
Source filter inventory:all17299;noTReX5330;noTReX+exact original bridge4625;
NQ/SQuAD-only+exactbridge1833 at3072. No semantic guarantee or selected source filter.
Do NOT do blind #n expansion and call it repaired reasoning.

Next:finish publication of this CPU audit;then freeze one v23 pilot protocol and
implement CPU-only passage-selection targets versus format-matched direct answers.
Proposed mechanism:small list of cited passage IDs before the answer,no intermediate
factual claims. It tests evidence selection,not verified connected reasoning.
Reuse single-model1.7B runtime,source objective and full-answer field parser for ALL
paths. Two fresh models in a bounded pilot;not another6-model same-label campaign.
Need fresh256 held evidence if constructible with deterministic low-conflict greedy
selection from622 candidates;do not relax exclusions silently. Prior512 now development.
No model job released. No other CPU job active. Agentquota unavailableuntilOct24.

Primary papers read:Chain-of-Note2311.09210v2 (GPT-4 labels10K,LLaMA2-7B,hybrid),
Attribute First then Generate2403.17104v3 (source-span selection,not citations-as-proof).
REFERENCE_NOTES updated. HotpotQA human sentence labels considered as an alternative;
official TRAIN HEAD timed out,no dataset downloaded or chosen,no DEV/TEST opened.

## Complete QA-v21 result; already published

Repo/home/ubuntu/dario,publicgithub.com/dario-fumarola/useful-thoughts-trainium.
MasterC=local/diagnostics/document-QA-v21-study-v1. Final-readout-completion.json PASS.
PrimaryF=validation/document_QA_v21_complete_20261019.json;P=Fwithout.json.
ImmutableprimarySHA75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9.
All20,992GEN/19,712LL complete,zeroexecutionerrors;all7models calibrated;allgroups
physicallyclosed. Sixfresh1.7B models,TRAIN2048×4passes,256updates,2dataseeds.
1,536updates/all12fullstates/order/labels/initialmasters/native48 checks audited.
48initialchecks repeat4contexts;numerics is not learning.

Canonicalcorrect/512,seed1921/1922:
- ordinarydirect201/195;priorgenerated217/227;priorplain201/180;priorempty190/169.
- priorfixeddemoCoT224/222 SECONDARY;postdirect200/186;B0fixeddemoCoT92shared.
- own−empty+5.27pp[1.17,9.38]/+11.33[7.42,15.23],allpairscomplete.
- own−ordinarydirect+3.125[-.977,7.227]/+6.25[2.344,10.352].
- postdirect−ordinarydirect−.195[-3.516,3.125]/−1.758[-5.078,1.562].
ALLTHREEboth-seed primarydecisionsFALSE. Seed2inferencepasses,seed1doesnot.
- OwnfullgoldanswerLL−7.337/−7.745vempty;contentdominates,Eosminor.
- PostdirectgoldLL+.377/.931vcontrol positiveCIs,secondarynotaccuracygatepass.
- PriorordinaryNLL+.09747/.11049vB0,betterthandirectby.03636/.02047.
- Prior57.4/56.3actions vsdirect5.37;observedmedian4.96/4.92s vs1.03s concurrent,
  not controlledservinglatency. Training12.56chiphrs,GEN5.19,LL3.07(scienceonly).
- Questiononlyprior11/23;bridge-removed148/144;otherreplay11/33;oracle507/512,512/512
  BUToraclecontainsrootanswer,privileged. Notpurecausalisolation.
- Gainsmostly2hop;no consistent3/4hopgain;held385/104/23,train1717/291/40.
- Direct-trainedreferenceCoTstrict1/1,510unsupported/seed. BaredirectanswerslackFinal
  field. PosthocuniformFinal-else-firstcomplete-line diagnostic recovers191/182;
  not newprimaryscores,not evidencecontrolgeneratedreasoning. Do notclaim40ppwin.

## Audit repair and supplements

Originalanalyzer PASS;originalindependentchecker failed exactdict equality forone
of60 native-summaryobservations. Same savedvectors/source recompute6floatstats
within1.979e-12,both originalgates pass. Originalchecker/failedlog/exit retained.
ExplicitV2 checker usesabsolute1e-10 for summarycomparisonONLY;bools/argmax/nativeKL
.01/cachemaxLP.001/sciencegates/data/scores/CIs unchanged. PrimarySHA unchanged.
Sourceslocal/analysis/check_document_QA_v21_readout_v2_20261019.py,protect...v2,
recover_document_QA_v21_readout_v2_20261019.py. NewattemptC/readout-pipeline-repair-v2.
V2check,originalrender,V2protect allPASS. Core raw/control/publicreadout EBS
local/backups/evaluation/document-QA-v21-complete. Originalfailure copiedP/audit-repair.

Posthocderivedscripts alreadyEXECUTED,no modelexecutions,gatesunchanged:
- account_document_QA_v21_costs_20261019.py → resource-accounting.json/tokenlatencyCSV.
- inspect_document_QA_v21_results_20261019.py → format-alias-hop-inspection.json,
  allunsupported/aliasonlydiagnostics;all17,920QAgoldnativevectorsdecodeexactly,
  identicalacrossmodels/modes;all512canonicalsurfaces roundtripexact.
- diagnose_document_QA_v21_trace_content_20261019.py → stored-trace-content-diagnosis.
  Lastclaimcopied507/509;canonical4steptraceonly7/23,4/23;not semanticcompleteness.
  Literalcitationpresence notentailment. Correctfinals canhavewrongintermediatefacts.
- diagnose_document_QA_v21_serialization_20261019.py → posthoc-serialization-diagnostic;
  uniformgold-independentfallbackforall20,992outputs,clearlyposthocnotnewpass.
- render_document_QA_v21_primary_20261019.py → focusedprimarySVG/PNG,visuallychecked.
Allsources copiedP/analysis-source;fixedhash8fullcontexts/outputs saved.
All1,742CPUtests+smoke/compilePASS;logsP/CPU-checks,original local/checks/tests.kPykH6.

## Completed publication and protected review package

README rewrittencompleteQAwithfailedgates/costs/limits;V17oldfrontpage appended
docs/EXPERIMENT_HISTORY_20261019.md (preserve itsquality+combinedFAIL).
STATUS/NEXT_STEPS/DECISIONS/REFERENCE_NOTES updated with completed publication.
QAcomplete PUSHED atdfa74b017d52965c4e1da416152da60ae667b73b.
Finalpublicfiles readbackprotected underlocal/backups/evaluation/document-QA-v21-complete/FINAL-PUBLICATION.
ReviewZIPdone/CRC+SHAverified:local/review/document-QA-v21-dfa74b0-20261019.zip,
22,042,673bytes SHA307ab27a500b0fd12aea69b1f51fc687cec9b9d7c943a8b70228b02f9f13f162.
Useralreadygivenpubliccommit/ziplinks;do notremake/reannounce zip.
CPU availability and bounded grounding audit is now complete;see current next action above.
The following publication steps1–3 areDONE;step4 produced V22.
1. Verifycurrentdiff/readmelinks/primarySHA;copyALLcurrentresult/docs/source
   supplements to fresh EBS REVIEW-ANNEX withhash/readback/fsync (corearchiveunchanged).
2. Commit/push reviewedQAcomplete data/plots/source/docs;smallP~7.4MiB. Do not add
   privateweights/datasets. No test reruns neededunlessnewcodechanges/failure.
3. Prepare ONE reviewzip underlocal/review using explicitpublicsource/results plus
   frozenTRAIN2048 andheldquestions/gold/manifestforgroundingreview(dataCCBY4.0),
   no modelweights/optimizer/credentials. Includeplainreviewguide+hashes. Letuserknow
   ziplink;do notaskpermission/endturn. Ownerrequestedoccasionalreviewzip.
4. Continue boundedCPUgrounding/eligibilityinspection andpaperbasednextdesign.
   No newtraining/RL/8B releaseyet. Needfreshheldfor anyconfirmatorynewadaptation.
   Current512becomesdevelopmentafterreadingresults;neverretrofitparserandcallprimarypass.

## Scientific findings andreading

Fixed8examplesinclude:JamesCumpston goldOhio citationis91stOhioInfantry,notexplicit
placeofdeath;predictedTennesseeclaimcitesunrelatedbioparagraph,butfinaldatecorrect.
AgnesdeathAmtNeuhaus mistakenforBerlin(birthplace)evenwithrightparagraph.
Wrongrecordlabel/songentityandOlympic1928vs2002returnyear showbinding/extractionerrors.
CorrectMolotov-Ribbentroppactsemanticname differsgoldsurface;EM/F1/aliases limitations.
Do not inferpopulationannotationerrorratefrom8examples.

Re-read primarypapersviawebOct19:
MuSiQue arxiv2108.00573v3 S5excludesorigquestions/answers/paragraphs. Ourresplit
allowsrepeatedanswersandaddspage/allTRAIN-visibleexclusions;notstandardofficialsplit.
EXmodelsubstitutesanswersduringstepsexecution;ourtracessuperviseliteral#nreferences.
Explicitresolutionis candidatehypothesis,notestablishedcause.
TPT2509.20186v1 doc+thinkingordinaryNTP;100Bmidtrainthen350kSFT unlike2kQA.
Do notrejectTPTfromourfailedaccuracygate. Docs/REFERENCE_NOTES hasprimarylinks.
OfficialDEV/TESTdatafiles NEVERopened;paperaggregates/examplesare referencesonly.

## Frozen assets/protection

DO NOT MUTATEworktrees:train049f53e local/worktrees/document-qa-v21;decode dbd1c10
document-qa-eval-v21;spancore0d1a5b4 document-qa-span-v21. No closedproducer reruns.
Original19,938authorTRAINraw local/nvme-study/data/musique-TRAIN-source-v12/
musique_ans_train.raw.jsonl SHAa85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345.
FrozenQAdata local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1.
AliasespinnedTRAINmirror attributed,not independentlyauthenticatedofficialZIP.
LatestEBSfull local/backups/science/document-QA-v21-study-v1/TERMINALS/latest-full256,
qa_prior1922/256 COMPLETEcb861c51561ae48a8c49f14c53dfbf5c58bb83c81e95162fd53c81db6224bd61.
Allsixterminalmodels EBS;all12originalfull128/256optimizerstates remainNVMe.
EBS same-instance,not off-instance. Root~60GiB,NVMe~397GiBfree;keep35/512GiBfloors.
Use scripts/env.sh andvenv/home/ubuntu/ema-trainium/.venv;CPU USE_TORCH_XLA=0.
No cloud/IAM/network/bucket/paidteacher/stack changes. OldR1234 remains stopped63.
