# Current queue

0. Prepare one bounded TRAIN-only selector-fit diagnostic;no new training.
   V23 is closed:generated129/255 vsmatched-direct127,same-modeldirect127;
   primary andinference gates fail. No replication,RL ormodel-size release.
   Readeroracle155/255 shows usable selection headroom;policyexactset132/255.
   Next question:does the frozen selector reproduce its own TRAIN annotations,
   stratified by2/3/4support documents? Use fixedhash cohorts,unchangedprompt/model,
   explicitquestion-onlyprojection andseparatescoringlabels. ReportTRAINfit only.
   This decides whether optimization/label allocation should be inspected before
   data orcapacity scaling. Do notconflate TRAINfitting withcapability/generalization.

1. Publish andpreserve the completed v23 pilot andits negative allocation decision.
   Report:validation/document_evidence_plan_v23_pilot_complete_20261019.json.
   All1530nativeattempts/3062originalLLrows checked/protected;allresponsefieldscomplete,
   zeroexecutionerrors/caps. All255nativeplansvalid;own−direct+0.8pp[-4.7,+6.3].
   Suppliedemptyfield refillsIDs255/255,soqualifythatcontrol withoutrescoring it.
   Genuine directpaths containzeroIDs;theirnull contrasts remainunchanged.
   Posthocexactset group gains14answers,other group loses12;not a causal estimate.
   FixedIDreplay own−empty−.731nats,own−donor+.845;not native-genlikelihood/calibration.
   OrdinaryNLL+.135vsB0;extraactions2.35×. No CoTsuperiority orpretrainingclaim.
   OriginalLLFileExistsError stoppedbeforeheldrows;IO-onlyv2repair preservedallinputs/
   models/nativeobservations/gates. Sourcebf288fe;1797CPUchecks,all16actualTRAINpanels pass.
   Both originalandcontrol-qualifiedplots inspected;originalprimaryreport/hashunchanged.

2. Keep historical results andlimits intact.
   V21:validation/document_QA_v21_complete_20261019.json,originalSHA
   75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9.
   Allthree two-seed primarygates fail;local own−empty positive,netmatched-direct
   firstCIcrosseszero. Olddirect-CoTformatfailure isnot a strongbaselinewin.
   TRAIN relationaudit24questions/72steps isexploratory,not corpuserrorrate/gold.
   Annotationoverlap doesnotprove semanticentailment;source/page exclusion doesnot
   establishpretrainingnovelty. Fresh255 ismostlytwohop;four-hopn4 descriptive only.
   OfficialDEV/TEST datafiles remainunopened. No paidteacher/cloud/IAM/network/stack changes.

Protected state/source:

- C=local/diagnostics/document-evidence-plan-v23-pilot-v1.
  SuccessfulLLregistry likelihood-registry-v2.json;readout-pipeline-io-v2 passes.
  Alloriginalfailedv1registries/logs/source remain;failureEBSarchive recorded inC.
- LatestfullEBS local/backups/science/document-evidence-plan-v23-pilot-v1/TERMINALS/latest-full256,
  evidence_select1923/256. Bothnew/V21sixterminalmodels protected;alloriginalfullstates remainNVMe.
  Older EBSoptimizerduplicates retireonlyafternewfullreadback. Same-instance,notoff-instance.
- V23train frozen7328141:local/worktrees/document-evidence-v23;
  decoder625fb0f:document-evidence-eval-v23;originalspan e0607fd:document-evidence-span-v23;
  repairedspan bf288fe:document-evidence-span-v23-io-v2. Do notmutate/rerunclosed producers.
- V21train049f53e/decoderdbd1c10/span0d1a5b4 remainfrozen.
  ClosedV17 quality/combinedgates FAIL,V15NLLcost,V10primaryFAIL,V7splitloses,V11repairclosed.
  OldR1234 stays stopped63. Keep35GiB/filesystem and512GiBhostavailable floors.
