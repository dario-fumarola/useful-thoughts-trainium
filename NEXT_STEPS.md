# Current queue

1. Publish the completed document-QA-v21 result and concrete review package.
   Primaryreport validation/document_QA_v21_complete_20261019.json,
   SHA75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9.
   All20,992GEN/19,712LL rows close,zeroexecutionerrors;all6trainingruns/12states audited.
   Independentchecks/render/fullreadbackprotection PASS. Originalfailedcheck preserved:
   oneFP64summary differed2e-12;V2compares summarieswithin1e-10 only,allgates/scores unchanged.
   Posthoc format/hop/alias/native-target/trace/cost diagnostics complete;protect REVIEW-ANNEX.
   Focusedprimaryplot visuallychecked. All1,742CPUchecks+smoke/compilePASS.
   Keep oldV17/history andalloriginaloutputs;no rewritten gates or omittedfailures.

2. Inspect supervision and control design before another training release.
   The two-seed primarytraining/inference/practical gates FAIL.
   Local own−empty accuracy+5.3/+11.3pp is positive inbothseeds,allpairscomplete.
   Netown−matcheddirect+3.1/+6.3pp hasfirstCIcrossingzero. Postdirect200/186vs201/195.
   Direct-trained CoT510unsupported/seed is a weakformatcontrol,not a serious solverwin.
   Own gold-answerLL−7.34/−7.75nats;postgold-answerLL+.38/+.93,secondarypredictivebenefit.
   Prior sourceNLL+.097/.110 vsB0;higher-hop gains inconsistent,not broadreasoning.
   Fixedexamples show wrongfact/citationchoices andatleastone goldsupportrelationissue.
   Need bounded TRAIN-only grounding inspection andCPUinventory ofpotentialfreshhelddata.
   Inspect literal#n supervision versus answer-resolved questions;do not assumethiscausedfailure.
   Assess a controltrained toretainFinalanswer formatting withoutthoughtcontent.
   Freeze one falsifiable nextdesign onfreshheld questions after thisreview;no prompt-sweep
   rescue or retrospectiveprimaryparserchanges. No newlearningjob yet.

3. Keep the research boundaries explicit.
   CanonicalEMprimary;canonicalF1 andattributedmirroraliases separate.
   TRAIN2048mayteachformat/datasetpatterns;support/page/seeddisjointness doesnot
   establishpretrainingnovelty orsemanticindependence. Four-contextparity is executiononly.
   OfficialDEV/TEST datafiles unopened;papers/aggregatepublishedtables are referencesonly.
   No broadRL,8B,splitrescue,cloudcapacity/IAM/network/bucket/paidteacher/stack changes.

Protected state/source:

- C=local/diagnostics/document-QA-v21-study-v1;final-readout-completion.json PASS.
- Raw/corepublic readout local/backups/evaluation/document-QA-v21-complete;originals retained.
- LatestfullEBS local/backups/science/document-QA-v21-study-v1/TERMINALS/latest-full256,
  qa_prior1922/256. Allsixterminalmodels protected;all12originalfullstates remainNVMe.
  EBS same-instance,not off-instance. Keep35GiB/filesystem and512GiBhostavailable.
- Frozen train049f53e:local/worktrees/document-qa-v21;decoderdbd1c10:document-qa-eval-v21;
  spancore0d1a5b4:document-qa-span-v21. Do not mutate or rerun closedproducers.
- Closedhistory:V17 arithmeticPASS/quality+combinedFAIL;V15 inferencegainwithNLLcost;
  V10bothprimaryfull16FAIL;V11boundedrepairclosed;V7splitloses. OldR1234 stays stopped63.
