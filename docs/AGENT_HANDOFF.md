# Handoff — 2026-10-19 09:33 UTC

## Exact next task

Publish the completed V15 result currently prepared onmain, then implement one
bounded source-balanced reference. Current main remote64c0926; local results/docs
are pending reviewed commit. No training or model evaluation remains active.

V15 allnine fixedmodels complete:17,280scalar attempts,3,456notes,19,584LLrows.
PrimaryCoT342/335 versusdirect275/285 andsame-CoTempty261/263 of384. GatePASSboth.
Accuracy gain17.4pp[13.3,21.6]/13.0pp[8.9,17.2]. Allfailures retained;zeroexecerrors.
Both-parse-valid numeric netcorrections59/45 oftotal67/50,posthocdescriptiveonly.
CoTnotes34.1/34.4actions,noempty/caps,allAnswerstops;answercontinuations mostlycap32.
Scalar endpoint iscompletedleadingnumber,notwholeQAreply. Target-onlySFT
ordinaryNLLregresses0.128/0.147nats/tokenversusB0;direct0.149/0.148. Full16CoTown
versusdirect+.603/+.446nat,butversusB0−1.979/−2.119. No broadpretraininggain.

Mainreport:validation/real_target_sft_v15_complete_20261019.json
SHAf8bc04e2c759cbad2a9438754aee828d5f99c6b4747dc41198fa1e4d3a43e033.
Independent native decode/AST/exactnumeric/count/all20kCIs/literalgatesPASS.
All37originalpayloads,2figuresPNG/SVG,CSV,15examples andsourcepublicready.
Twofigures visually inspected;legends/footer areclear. Descriptive errortransition
annex doesnotalterendpoint. Model-source829d9a2/eval41a5bfd stayfrozen.

Originalexampleexportassumedfrozenroworder,butpayloads preserveworkershardorder.
Its failedattempt/exit1 ispreserved;fixedonlypresentationreindexbyimmutableID,
thenexistingdonorpermutation. final-examples-v2 passes;no forwards/scores changed.
Finalcompletionreceipt explicitlyincludesrepair. A prematureprotectionlauncher
usedrelativecwdandfailedbeforewriting;successfulv2readbacklog ispreserved.

Control:local/diagnostics/real-target-eval-v15-study-v1;all9controllers/scorequeue
closed. Alloriginalrawoutputsprotectedatlocal/backups/evaluation/real-target-sft-v15-complete.
PUBLIC-READOUT hascurrentanalysis sources,completepublicfiles,andactual18CPU/XLA
referencepanels(216repeatedrows/3,456tokenpairs),independentlyrecomputedparityPASS.
DESCRIPTIVE-ANNEX protects newerrortransitionJSON/source andclosedreadoutcontrols.
Allfourterminalmodels/latestfull320 EBSprotected,alloriginal160/320 remainNVMe.
Trainingreportvalidation/real_target_sft_v15_training_20261019.json:1,280updates,
8fullstates,32initialnativechecks,allfour320-rowtraces. Eightnativeworkerspermodel
repeatfourcanonicalcontexts;notindependentscientificdocuments.

Nextscientificstep:retainordinarysourceNTPwithseparatelynormalizedtaskloss,
source+direct versus source+CoT. DonotblindlyreuseV10prefix-dominatedlossorV15
source-unscoredloss. Inspectdriverfirst,declareoneboundednewexperimentidentity,
prepareCPUdata/masks,thenactualmask/update/resume/memoryqualificationbeforerelease.
Freshsource-TRAIN confirmationispreferable;existing384alreadydevelopment-exposed.
NoRL/8B/splitexpansion. Thecompletedreferencesarenotemergenceorpretrainingclaims.

## Completed findings that stay unchanged

MainV10:both predeclared full16 likelihood gates FAIL. Post raw gains~0.01nat;
prior-own versusrawCPT+0.097/−0.139nat,intervals crosszero. The prospectively declared
scalar secondary is positive:216/213 of256 versusrawCPT193/196,+9.0/+6.6pp with
positive intervals;unsupported25/24 retained. Same-prior empty175/172. This is a
narrow extra-computation arithmetic-answer gain,not ordinary training benefit.
All19,712 LP rows/3,584notes/17,920scalar attempts,figures,28disclosed examples and
independent bootstrap checks are public in main c6bf246/c60fb3d and follow-ups.
Posthoc partition331ce2e:tail damage~−.21nat inbothseeds;numeral gainclear onlyseed13.
V10models/latest full320 protected;alloriginal160/320 states remain. Never alter raw
reports SHAeb5f6b07aa4b39da6f708a7941fc2bc3bbaf1580b5bc3942979730acab2e1bf4
and c25b6a06f855684622084a9e4744b2c55ef8f7290f31ee108e0aca8537e1855f.
Bounded variable-depth repair is closed and positive only inits narrow scope;
no expanded split rescue. OldR1234 stays stopped63. No broadRL orunqualified8B.
MuSiQue is CPU-only preparation;19,938TRAIN questions,the visible-evidence split
risk is documented. It is not a third released learning program.

## Runtime/storage discipline

Source scripts/env.sh,then explicitly set PYTHONPATH to the DESIRED WORKTREE plus
local/analysis and local/test-deps. Python heredocs run from main otherwise import
main before PYTHONPATH;every preparer must assert ROOT. Use correctcwd. No stack
upgrades,cloud/IAM/network changes,paid teacher services or unapproved bucket.
Do not print broad process args or environment;unrelated services may hold secrets.

Compile cache nowbyte-verified on existingNVMe:
`local/neuron-cache-driver` is a symlink to
`local/nvme-cpt/caches/neuron-cache-driver-20261019`. Original runtimepath unchanged,
all10,908entries/18,495,161,350bytes verified. No model/checkpoint/data removed.
Report:validation/neuron_compile_cache_NVMe_move_20261019.json.
Rootfree~74.6GiB after qualification backups;projected newmodels+latestfull~35.3GiB,
35GiB reserve retained. NVMe has ample room for original states. Keep512GiB host
availability and35GiB perfilesystem. EBS is same-instance,not off-instance backup.
NVMe-cpt UUID5119e7ab-0ca6-4295-b8f0-e856d87b4b67 is ALREADY formatted/mounted;
NEVER mkfs. No new reviewZIP today;one was already shared. Main publication of
V15hardware/active status/cache move is included in the next reviewed commit.
