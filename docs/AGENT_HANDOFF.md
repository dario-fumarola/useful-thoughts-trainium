# Handoff — 2026-10-19 15:23 UTC

Read STATUS.md and NEXT_STEPS.md first. Work autonomously until stopped;weeks available.
Do not final at a milestone. No activegoal;do not createone. Agentquota unavailable
untilOct24;do not retryagents. Keep commentary concise and at least every60s;
short reasoning/tool/action cycles. OfficialDEV/TEST unopened,no RL/8B/cloud changes.

## Current study

C=/home/ubuntu/dario/local/diagnostics/document-QA-v21-study-v1.
Frozenrelease SHA5847fc816b7b469f9c236a0acbf8abb2b1a64fb7229ea8940a592a40f12abd1c.
DesignV3 SHA80f8a7f633566289fb74a2631a816819129d199320e4dc0690a2a1f57c1c7da0.
ALLSIX trainingruns CLOSED at256updates,1,536updates total;all12 full128/256 states,
orders,labels,counters,initialmasterwitnesses and48native checks independentlyPASS.
C/terminal-independent-audit.json and terminal-pipeline-completion.json.
Allsix exactterminalbindings C/frozen-models/<qa_direct|qa_prior|qa_post>-seed2026101921/1922.json.
Freshreleased1.7B each;TRAIN2048×fourpasses,global32/world8/micro1/accum4,width3328,
target512,LR1e-5constant,BF16active/grads,FP32masters/Adam,oneZeROreducer.
Loss .5meanCE fixed256originalsourcepositions+.5meanCEtask;not fullsourceCE.
Prior/post use direct on evenpass0/2;prior fulltrace+Finalanswer onodd1/3;
post answerEOS thenpostreasoningtraceEOS onodd. Differenttargetcounts declared.
Publictrainingreport validation/document_QA_v21_training_20261019.json READY,
plus sixall256metricsgz,12fullmanifests,4uniqueactualnativevectors mappedto48observations.
Publicationindependentlyreadbackprotected underTERMINALS/TRAINING_PUBLICATION;commit/push next.

Immutable source/worktrees;DO NOT MUTATE:
- training049f53e atlocal/worktrees/document-qa-v21
- decodingdbd1c10 atlocal/worktrees/document-qa-eval-v21
- spancore0d1a5b4 atlocal/worktrees/document-qa-span-v21
Lastpublicmain9b1d6c9 includes frozen10orchestration/readout sources and independently
recomputed/protected TRAINspanparity arrays. Own-model TRAINcalibration still required
before heldLLforwards. These checks are executionevidence,not scientificvalidation.

## Protection complete

C/terminal-protection.json PASS_ALL_SIX_TERMINAL_MODELS_AND_LATEST_FULL256_PROTECTED.
LatestfullEBS:local/backups/science/document-QA-v21-study-v1/TERMINALS/latest-full256,
qa_prior seed2026101922/256,COMPLETE SHA
cb861c51561ae48a8c49f14c53dfbf5c58bb83c81e95162fd53c81db6224bd61.
Newfullverifiedbefore older128EBS optimizerduplicates retired;all12original fullstates
retainoptimizers onNVMe;allterminalmodels and128model retainedEBS. EBS same-instance,
not off-instance. Root~60.5GiBfree,NVMe~397GiB;keep35GiBfilesystem/512GiBhostavailable.
No mkfs,cloud/IAM/network/paidteacher/bucket orstack changes.

## Evaluation active;capability UNREAD

Tmux document-QA-v21-B0-generation:physical12,fourshards.
Output local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1;
control C/B0-generation. About2,800/3,584attempts currently;count/status only.
B0modes direct,prior,empty,reference_cot,questiononlydirect/prior,bridge_removedprior.
Activecontroller v1 staysimmutable.

Tmux document-QA-v21-trained-generation-pipeline prepares six nativeCPUreferences/
fullpromptprojections now,then launches12fixedgroups. Source
local/analysis/watch_document_QA_v21_trained_generation_20261019.py;
log local/analysis/document-QA-v21-trained-generation-pipeline-20261019.log.
Registry C/trained-generation-registry.json appearsafter allsixprepares/launches.
Futuregroups use local/analysis/supervise_document_QA_v21_generation_group_v3_20261019.py:
protocolhashTMPDIR and replay-specificports;leaveactive originalsunchanged.
Execution C/evaluation-execution-plan.json:direct1921/1922 all4modes→physical0/1;
post1921/1922 all4→2/3;prior1921 prior-only→4,short→5,reference→6,evidence→7;
prior1922 analogous8/9/10/11. Afterprior closes,prepare own/otherreplays,reuse4/8.
Same frozen perquestion source/model/precision/cap;grouping changes schedulingonly.
Controllerownprocesses/actual4allocation+native/cachegates+physicalclosure,zeroautoretry.

Tmux document-QA-v21-likelihood-pipeline ACTIVE waiting for originalGENclosures.
Source local/analysis/watch_document_QA_v21_likelihood_20261019.py;
producer document_QA_v21_likelihood_job_20261019.py,groupcontroller
supervise_document_QA_v21_likelihood_group_20261019.py (allunderlocal/analysis).
Fourbatch4shards/model;704rows/shard,2816/model. Own-model TRAINCPUrefs first,
max-token andabsmean-span drift<=.001 before heldscoreforwards. Fullactualanswer+EOS,
separatecontent/EOS,ordinary256×32. FiveQAmodesdirect/empty/own/other/oracle.
Requires ALL512 originalpriorattempts includingfailures;fixedcyclicdonors,norepair.
Physical14/15 reservedwhileGENactive;expand0–11 onlyafterALL_GENERATION_CLOSED.
Registry C/likelihood-registry.json. ExplicitCPUfallback onlyrootreview onfailure;
no automaticretry,no backendchoice usingheldscores.

Tmux document-QA-v21-readout-pipeline ACTIVE waits BOTH globalclosures:
C/ALL_GENERATION_CLOSED_NO_READOUT.json and ALL_LIKELIHOOD_CLOSED_NO_READOUT.json.
Exactly20,992generationattempts/19,712LLrows across7models required.
Then local/analysis/watch_document_QA_v21_readout_20261019.py runs once:
analyze_document_QA_v21→check_document_QA_v21_readout→render→protect;
stagelogs/exits C/readout-pipeline,final marker C/final-readout-completion.json.
Publicoutputs validation/document_QA_v21_complete_20261019.json +same-stemfolder.
Independentparser/nativedecode/distribution/metrics/vectorizedbootstrap verifiesrawrows;
allerrors,caps,unsupportedretained. No handcorrection orfiniteonlymeans.
Fourstages ASTchecked;parser/cap/error andbootstrapboundarysyntheticchecks pass.
Do not editactive/frozen source. Ifa jobfails,preserve original androotreview smallestrepair.

## Science/interpretation

Primarycanonical complete-answerEM,all512;canonicalF1 and separatelyattributedaliases.
TRAIN2048 might teachformat/datasetpatterns;heldevidencedisjoint doesnotestablish
pretraining/semanticnovelty,generalreasoning or officialbenchmarkperformance.
Reviewerlimits saved C/reviewer-interpretation-boundaries.json;no protocolchange.
Traininggate post.direct−direct.direct>=3pp andpaired95%lower>0 EACHseed.
Inferencegate prior.prior−prior.empty AND−matched direct.direct,same thresholds.
Practicalreasoningsupervision additionally mustbeat matched direct.prior,
direct.reference_cot andunchangedB0.reference_cot;simplercontrols allowedtowin.
20kpairedquestionbootstrap seed2026101921,seedsseparate,checkpointconditional/unadjusted.
LLsecondary fullnativeanswerEOS;oracleincludesrootanswer,privilegednotdeployed.
OrdinaryNLL measuredtradeoff;NEVERreuse/changeV17FineMath .03gate.
Allcurrentcapability unread untilfullclosure;terminal256primary,no checkpointselection.
Fixedhash8exampleoutputs/allrawattempts/likelihoods andcosts preservedbyreadout.
Aftercomplete,rootreviewresults/plots/errors,update README/STATUS/NEXT_STEPS andpush.
No additionalstudy launched untilthis matchedcomparison interpreted.

## Preserved history

V17 completepublic:arithmeticPASSboth,ordinaryquality+combinedFAILboth atoriginal.03bound;
no broadpretrainingclaim. V15 inferencegainwithordinaryNLLcost;V10bothprimaryfull16FAIL.
V11boundedvariabledepthrepairclosed,no splitadvantage;V7fullCoTbeats splitall8cells.
OldR1234 stopped63,NEVERresume. Details README/EXPERIMENT_HISTORY andclosedreports.
