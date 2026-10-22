# Overnight handoff — October 22, 2026, 11:18 UTC

**Keep working through 08:00 Eastern / 12:00 UTC; do not end the turn early.** The owner grants autonomy on the existing node. No new capacity/stack/IAM/network changes, paid teachers or official TEST. Do not spawn agents. Preserve all original states and outcomes. Repo `/home/ubuntu/dario`; `source scripts/env.sh`; CPU `USE_TORCH_XLA=0`; `rg` unavailable.

## Main program: complete and public

[Plain-language results](OVERNIGHT_RESULTS_20261022.md) · [All twelve model downloads](MODELS_20261022.md) · [Current status](../STATUS.md).

- Ordinary uniform CPT: fresh FineMath gain **0.014481 [0.013903,0.015058] nats/token**. DEV selects cosine / peak LR1e-5 after four terminals freeze. English retention loses. Cost **16.607276 assigned chip-hours**.
- Matched raw/format/computation: six fresh models, two source orders, full source coefficients/context matched. **FAIL**: trace-minus-raw **−0.000943640 /−0.000907604**, negative intervals. Trace beats format but loses to qualified ordinary reference. All arms improve over B0. Cost **32.536880**.
- Separate English diagnostic: all ten mathematical-only models lose. Cost **0.502299**.
- Fixed 50/50 ordinary source mixing: **PASS in both orders**, fresh Math gains **0.011346 /0.011204**, English **0.008550 /0.008756**, positive intervals. Target beats all-math reference by **0.006578 /0.006610**; all-math still better on Math alone. Cost **6.290836**.
- Common panels: all twelve models+B0, same Math4096/English1024. Both mixed models improve both domains; mathematical-only models lose English. Supplemental, no selection or parent-gate change. Cost **0.962545**.
- Standard CPU check: 1,024 fixed documents, all8predictors, **8,192 forwards**. Trace-minus-raw remains negative both orders (**−0.000928 /−0.000890**, negative CIs). Corresponding CPU/native mean differences **+0.000027 /−0.000004**. Largest single-token difference **4.747785nats**; do not claim tokenwise identity.
- Mixture CPU check: four predictors ×32documents ×2domains,256forwards, both numerical checks PASS. Largest mean paired discrepancy Math0.001002/English0.000221. Original native decisions unchanged.

All six source-free archives and the mixture CPU supplement pass anonymous download, hashes and NumPy replay. Receipts `validation/*public_release_20261022.json`; CPU supplement `validation/uniform_mixture_CPU_public_supplement_20261022.json`. Original raw text/token IDs remain outside numerical archives.

All **twelve** model deltas are public at tag **trained-model-deltas-20261022**, source **be71b40774c9d96d7496fb30011c9beed1ea2d4a**. Every asset anonymously downloaded/restored exact BF16 bits, all twelve standard HF loads/forwards PASS. New isolated CPU-only venv with no Neuron/XLA additionally restores selected model and passes HF inference. Existing Trainium environment unchanged. Proofs `validation/trained_models_public_release_20261022.json`, `trained_models_HF_smoke_20261022.json`, `trained_model_clean_CPU_environment_20261022.json`. Codec files remain hash-frozen.

Resource reconciliation `validation/overnight_resource_summary_20261022.json`: **56.899836 assigned chip-hours** within separate summed68.25ceilings;85nativeattempts,8earlier nonzero-exit qualifications. Original failed/missing process-absence flags remain intact; current **16-device inventory is idle**. Twelve fits total1,536updates /213,700,608original target exposures across51,200unique training documents. Source exposures are repeated across arms, not unique corpus tokens. Some qualification costs predate the owner's night window. This is not whole-node billing.

## One optional CPU derivative diagnosis is active

**Session41340**, `local/diagnostics/source-aux-gradient-20261022/run_check.py`; log **coordinator-v2.log**. Started11:10UTC,24threads on cores48–71. At11:12 it had32/272source windows, about3.14seconds per source backward; measured complete projection fits. Expected finish around11:27–11:32, **hard CPU stop11:42UTC /1,800seconds**. No native calls or optimizer steps.

It computes FP32 source and separately normalized work gradients at **unchanged B0 on the exact first native TRAIN batch (272documents, order11)**, then streamed FP64 norms/dots/cosines and zero-moment AdamW directional surrogates. Format/trace work masks use the existing packer. Parameter fingerprint before/after must match. This is one inspected batch, not a held-out model-quality result or unique cause of the failed128-update study. FP32 CPU gradients differ from native BF16 gradients. Do not promote a directional surrogate to actual future training improvement.

Plan `validation/source_auxiliary_gradient_plan_20261022.json` /private prospective-plan.json includes worker SHA, exact batch identities and constants. First local prototype wrongly expected optimizer constants in protocol fields; it was stopped after25seconds of source derivatives, before results, then restarted with the unchanged qualified constants **eps1e-8, clip1**, confirmed from original native driver. Original log/plan retained. No model parameter changed and no native code was modified. Current code is frozen after restart. An actual packing probe validates both work mask interfaces.

Next: inspect **complete-readout.json** when it exists, verify fingerprint/finite values, write a short exploratory report with its actual limits, copy the worker/plan/results for review. Preserve a runtime feasibility stop if the full batch cannot fit; do not silently shrink by outcomes or change a completed scientific decision. This optional derivative work must be closed before the deadline.

## Remaining deadline work

1. Main reports and model catalog are reviewed/committed. Check latest push (`faa884b` at last update). Keep current summary/STATUS/NEXT_STEPS accurate; replace their derivative-pending language once complete. Record any new result without altering original flags.
2. Finalize **read-only utilization** from `local/diagnostics/uniform-lm-native-v1/whole-node-utilization.jsonl`. Monitor session16559 runs until12UTC. Freeze complete JSON lines around11:58, then stop its own monitor cleanly if desired. Source script `monitor_node_utilization.py` has a finally block terminating its `neuron-monitor` child. Use `scripts/summarize_node_utilization.py`: explicit from/to UTC, carry at most10seconds, missing intervals unknown. Process inventory can lag60seconds. Actual peak8/16chips; **not full saturation**. Interim02:53–09:36 coverage98.22%, observed node busy27.85%, min available host598.4GiB. Do not use that interim number as whole-night utilization. Add final coverage/scoped figures and retain compressed raw telemetry.
3. Recheck current local links, actual public receipt/manifest identities, clean Git and remote main. Source/test suite last **2,074PASS**, smoke+compile, local/checks/overnight-final-20261022.log. Production code unchanged since that check except docs. Avoid repeating expensive tests without a new concern.
4. Work through12UTC. All native jobs are closed; no new training campaign near the deadline. Finish with concise self-contained outcomes and results/model links. Do not final early.

## State and source locations

All twelve local native model files and full-state logical paths are cataloged in **validation/trained_model_artifacts_20261022/local-state-registry.json**. Eight new exact model.pt copies are on existing NVMe0; four baseline models on NVMe-CPT. Selected baseline full state moved byte-for-byte to NVMe-study with original directory symlink; all other complete optimizer snapshots remain retained, including RAM. **Do not power off the node or discard states.** Public model weights are remotely verified; local native optimizer snapshots are not a remote backup.

C1 `local/diagnostics/uniform-lm-native-v1`, training source4f9166b; C2 `scaffold-learning-overnight-v1`, source53adb6e /scoringb55349f; C4 `general-web-retention-20261022`, source665dde8; C5 `uniform-mixture-20261022`, original4f9166b /scoring112f1e7; C6 `overnight-common-panels-20261022`; C7 `scaffold-standard-CPU-verification-20261022`; C8 `uniform-mixture-CPU-check-20261022`. All source-free replay/report/package controls remain in those directories. Immutable native workers and codec files stay untouched.

Current free storage roughly root36.6GiB, RAM62.6GiB, NVMe0~41GiB, study39.2GiB, CPT38.6GiB; RUNtmpfs>100GiB. Floors35GiB/512GiBhost remain unchanged. Clean CPU venv and reconstructed downloaded models live in existing `/run/user/1000`, not the Trainium environment. Earlier exhaustive source/publication handoffs are archived under docs/archive.
