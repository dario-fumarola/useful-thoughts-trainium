# Handoff — 2026-10-19 16:58 UTC

User saysworkautonomously untilstopped;weeksavailable. Do not final atmilestones.
Noactivegoal;do notcreateone. Agentquota unavailableuntilOct24,do notretryagents.
Shortreasoning/tool/actioncycles;commentary atleastevery60s. Currentphasepublication
andscientificinspection,not moreruns. NoGPUjobs active;allv21producers CLOSED.

## Complete QA-v21 result;publish next

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

## Finish concrete publication thencontinue

README rewrittencompleteQAwithfailedgates/costs/limits;V17oldfrontpage appended
docs/EXPERIMENT_HISTORY_20261019.md (preserve itsquality+combinedFAIL).
STATUS/NEXT_STEPS/DECISIONS/REFERENCE_NOTES updated;workingtree uncommitted.
Lastpublicmain9c4d6c5 has completedtrainingproof;QAcomplete NOT pushedyet.
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
