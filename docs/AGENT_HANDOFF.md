# Handoff — 2026-10-19 08:50 UTC

## Exact active task

Allfour V15science training runs complete320updates. Independent fullstateaudit
passes1,280ordered updates/eightfullstates/32nativechecks. Terminalpipeline closed
08:47UTC;allfourterminalweights/latestfull320 EBSreadbackprotected,originals retained.
Public trainingreport:validation/real_target_sft_v15_training_20261019.json withfour
complete320-rowtraces,models/optimizer receipts andmeasuredconcurrenttrainingcosts.
Training829d9a2 andeval41a5bfd remainfrozen. No capability outcomes inspected.

Allnine evaluation jobs released. Fivebaseline models onphysical8,9,10,11,14;
fournewtargetmodels onphysical0–3. Tmux target-eval-v15-slot0 through8.
Mastercontrol:local/diagnostics/real-target-eval-v15-study-v1.
Mechanicalhealth:python3 local/analysis/live_real_target_v15_health_20261019.py.
All5baseline models completed both nativeLLpanels withactualfirst4CPUpermodeparity.
Four-slot scorequeue target-v15-score-queue waitsfornewnotes;logical48–51 onphysical12.
Per-model registries/paths/frozenbindings are inmasterplan;do not prepare/releaseagain.

Final once-only watcher target-v15-final-readout islive;deadline21:00UTC.
It waitsallnine fixedscalar/controllers ANDnativeLLclosure beforeopeningoutcomes.
Then scripts inlocal/analysis run:analyze_real_target_v15_20261019.py,
check_real_target_readout_v15_20261019.py,render_real_target_v15_20261019.py,
save_real_target_examples_v15_20261019.py. Sources ASTparse;realdatareadout pending.
Logs final-{analyze|independent|render|examples}.log andexitreceipts inmastercontrol.
Anyfailure:preserverawattempt,diagnosesmallrepair,noautomaticretry/modelselection.
Do not inspect partialcorrectness/LL;finishandinterpretallninebeforeanotherlearningrun.

Finalreport target:validation/real_target_sft_v15_complete_20261019.json plus37raw
payloads,independentvectorizedbootstrap/parser/nativecheck,figures,CSV,examples.
Analyzerprotects originals tolocal/backups/evaluation/real-target-sft-v15-complete.
Afterallcomplete:inspectfigures,saveactualsourceannex/readbacks,updateREADME/STATUS,
commit/pushreviewedoutputs. Need verifyfilestracked;publicevidencepaths containing
local/areignoredunlessexplicitlyforce-added. Neveraddmodelweights/assets/secrets.

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
