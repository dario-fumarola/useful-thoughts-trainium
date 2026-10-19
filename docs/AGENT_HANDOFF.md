# Handoff — 2026-10-19 14:29 UTC

Read STATUS.md and NEXT_STEPS.md first. User says work autonomously until stopped;
weeks available. Continue to relevant results,not a final acknowledgement at a
milestone. No active goal;do not create one. Agent quota unavailable untilOct24;
do not retry agents. Commentary should remain concise and at least every60s.
Avoid giant speculative analysis:use short reasoning/tool/action cycles.

## Active science

Master C=/home/ubuntu/dario/local/diagnostics/document-QA-v21-study-v1.
Release SHA5847fc816b7b469f9c236a0acbf8abb2b1a64fb7229ea8940a592a40f12abd1c.
Latest designV3 SHA80f8a7f633566289fb74a2631a816819129d199320e4dc0690a2a1f57c1c7da0.
Six fresh B0 jobs started14:01UTC:qa_direct/qa_prior/qa_post × seeds1921/1922.
Each256updates/global32=8192 occurrences,four passes ofTRAIN2048;full128/256.
World8×micro1×accum4,width3328,target512,LR1e-5constant,BF16active/grads,F32masters.
Perexample0.5 meanCE(source fixed256 positions)+0.5 meanCE(task),oneZeROreducer.
Prior/post arms plain on even passes,augmented on odd. Post teaches answerEOS
before post-reasoningEOS. Prior full root trace is conventional CoT,not newarchitecture.
Science tmux document-QA-v21-<arm>-seed2026101921/1922;physical0–11.
Warm~13.5sec/update. All share identical first64 loss,norm,order within seed;
health only. Treatments differ after64. No capability scores read or earlyselection.
Outputs:C/plan.json jobs[*].output underlocal/nvme-cpt/runs/document-QA-v21-*.

Immutable worktrees/source:
- training049f53e atlocal/worktrees/document-qa-v21
- decodingdbd1c10 atlocal/worktrees/document-qa-eval-v21
- span scoring0d1a5b4 atlocal/worktrees/document-qa-span-v21
DO NOT MUTATE them or active controller source. Main integrates these sources.
Actual training14updates/sevenfullstates/freshrestore PASS;PSSgrowth1.46/1.53GiB<2.
Actual decoder CPUlong parity max0.0000651 PASS;sixTRAINattempts,zeroerrors,
4complete/2unsupported(not capability);publicnative arrays explicitly forced intoGit.
Span B4QA3584/ordinary672 parity PASS;QA maxerror0.0000799,mean0.0000126,
ordinarymax0.0000584,mean0.0002913. Warm9.00s/.710s versus singleCPU69.10/7.14s;
limited profile,not steady CPU benchmark. Closure present;independent span archive/
publication still needs root. Control local/diagnostics/document-QA-v21-span-qualification-v1.
CPU checks1742+smoke/compile PASS;core span causality/padding/full-target tests pass.

## Checkpoints and terminal pipeline

Tmux document-QA-v21-first128-protection runs
local/analysis/protect_document_QA_v21_first_full128_20261019.py.
It waits for a complete128,independently inspects all master/moments/casts/counters,
copies new full EBS FIRST,readback/fsync,then freshly hashes oldV17original/EBS
before retiring only old optimizer duplicates. All original/modelweights retained.
Receipt C/first-full128-protection.json;latest EBS full switches ONLY upon PASS.
Root~83GiB before128 copy;NVMe~665GiB before science checkpoints. Keep35GiBfloor.
EBS same-instance,not off-instance. No cloud/IAM/network/paidteacher/bucket changes.
Tmux document-QA-v21-terminal-pipeline waits all six closures and first128 protection,
then runs independent terminal audit→full-first protection→exact binding once.
Sources local/analysis/{audit,protect,bind}_document_QA_v21_terminals_20261019.py.
All12original128/256states stayNVMe. Protect new256full before retiring128EBS
optimizer duplicates;keep its model and all six terminalmodels. No automaticretry.
Final bindings C/frozen-models/<arm>-seed<fullseed>.json have id=jobkey exactly,
status AUDITED_DOCUMENT_QA_V21_TERMINAL256,step256,hashes/audit/protection.

## Evaluation work that remains

B0 already generating onphysical12(logical48–51),four shards:
output local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1.
Controller tmux document-QA-v21-B0-generation;C/B0-generation,rootlog
C/B0-generation-supervisor.log. Actual4process allocation captured. Do not read
answers/accuracy;monitor rowcounts and errors only until full study closes.
Active v1 generation-group controller remains immutable. Future use v2:
local/analysis/supervise_document_QA_v21_generation_group_v2_20261019.py.
It changes TMPDIR identity to protocolhash to avoid sequential replay collisions.

Frozen C/evaluation-execution-plan.json:12 initial trained generation groups.
Direct/post all4modes per model onphysical0/1/2/3. Prior seed1921 groups:
prior-only→4,short(direct/empty/oracle)→5,reference→6,evidence(questiononly/bridge)→7.
Prior seed1922 analogousphysical8/9/10/11. After prior-only groups close,
prepare_replays and reuse their freedphysical4/8. B0 staysphysical12 untilclosed.
This groups the same frozen conditions for speed;no extra observations/selections.
Still implement the post-binding model preparation/subset protocol/controller runner
and replay watchers. Preserve total20,992 generation attempts across7models.
Full model prepare may create all initial prompt projections/native references;
subgroups copy those same bytes/native refs and subset modes. Do not modify DBdcode.
prepare_replays consumes only the completed prior-only group and all512 failures.

LL producer ready,not executed on held:
local/analysis/document_QA_v21_likelihood_job_20261019.py (prepare/run CLI).
prepare --original <closed prior generator directory> --output <fresh scoringdir>.
All512 prior attempts required;usesdirect/empty/own_replay/other_replay/oracle,
full native canonicalanswer+EOS,EOS/content separate;ordinary256×32targets.
Own-model fixedTRAIN CPU calibration69s before any held score;checks checkpoint
signature/lineage,allcontext bounds and everyinput hash. Four shards,batch4;
704rows/shard,2816/model,total19,712LLrows. run --shard0..3 --device xla/cpu.
Source spanworktree0d1a5b4. CandidateXLA onlyafter permodelTRAINcalib max-token/
absolute mean-span errors<=.001;otherwise rootreview/CPUfallback per frozenrelease.
No silent dropped/shortened context ormissingLLestimates. Exceptions retained.
Need implement its group controller/closure and overall queue. Reservephysical14/15
forLL whileGENactive;after all GEN/Replays/B0close,free0–11 mayserve remainingLL.
Need independent allraw-attempt/native/parser/count/CI readout and protected publication.
Do not inspect capability until all7model conditions and LLpanels close.

Primary canonical whole-answerEM,512denominator,allerrors/caps retained;
separate canonicalF1/attributedaliases. Training gate:post.direct−direct.direct.
Inference gate:prior.prior−prior.empty plusmatched direct.direct. Each>=3pp and
paired95%lower>0 separately perseed. Practical augmented solver must ALSO beat
matched direct.prior AND direct.reference_cot AND unchanged B0.reference_cot.
Allarms get fixed-demo reference_cot;native/ref paths separate,no posthocselection.
20kpairedquestionbootstrap seed2026101921,unadjusted/checkpointconditional,no pooledseedCI.
Likelihood is diagnostic/full answer;report EOSpartition. OrdinaryNLL is measured
tradeoff,no inheritedFineMath0.03gate. OfficialDEV/TEST remain unopened.

## Data/provenance

Datalocal/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1.
TRAIN SHAa0a99d866b55c16d7e6f498ee881d87556dfa5eabcab3340feccbf11313476e9.
Held questions SHA7a8ff7129829960a876e8cd39d35f23d9033f2789b2b495023efd9862882c39d.
Allinput<=3072,all6144viewsfit3328/512. Held support text/page/trueoriginalseedIDs
are disjoint withinheld and absent from ALL trainvisible evidence. Distractors and
relationtemplates canrecur;semantic/pretrainingnovelty notestablished.
V16/V20 rawcomponent strings canbe genericrelationprograms;V21 corrects identity
usingdecomposed_instances dataset:id,before anymodel outcome. Oldtables unchanged.
Author rawTRAIN19,938allcanonicalfields matchpinnedauthorconverter;allselectedhop
answersoccurincitedvisiblepassages. Goldtracesinclude root answer,privileged supervision.
Mirror aliases not authenticated against officialZIP;canonical primary avoids relyingon them.
QAdata/source EBSlocal/backups/data/document-QA-v21-TRAIN2048-held512-v1.
Ordinary256 distinctWikipedia pages:local/nvme-cpt/data/document-QA-v21-ordinary256-v2,
SHA7997354318482f4637ff4601f5ff71ecb8cbe2f052fecb5744f4ae13eba58c32.
AllQAvisiblepages/text excluded. Initialselectorfailed pageuniqueness;V2 firstperpage,
no modeloutcomes/thresholdchange;bothattempts protected. Allordinary/design/metric
source protectedlocal/backups/data/document-QA-v21-preparation-annex-v1.

## Environment and prior results

Venv/home/ubuntu/ema-trainium/.venv. source scripts/env.sh thenexplicitPYTHONPATH:
immutableworktree:first,local/analysis,local/test-deps. CPUUSE_TORCH_XLA=0.
TmuxPATH includes actualinterpreterbin/libneuronpjrt-path. No upgrades. No rg;
use targetedPython/grep. No broadprocessarguments/envdumps.16Trn2chips/64logicalLNC2,
24GiBbank;2TiBhost;keep512GiBavailable. NVMe nevermkfs;compilecache alreadyNVMe.
No newzip needed(today's reviewzip alreadyshared). Publicrepo authorized;push reviewedcode.

V17 publice1bbff9:sourceCoT434/439of512 vsdirect351/357;arithmeticPASSboth;
ordinaryNLL+0.0380/+0.0429,upperCI>.03→qualityandcombinedFAILboth. Preservefailure.
Regressionpoint75.6%smallerthanoldtargetonlyonSAMEnewpanel. No clear rawgain;
full16belowB0. Alloriginaloutputs/fullstates preserved andallterminalmodels protected.
V15positiveinferencewithsourcecost;V10primaryLLgatesFAIL;V11boundedrepairclosed.
OriginalR1234staysstopped63. No newRL,8B,expandedwriter-readerrescue orpaperreproductionclaim.
