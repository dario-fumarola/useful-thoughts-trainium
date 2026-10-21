# Uniform next-token baseline: source and CPU preparation

The plain-text training path and all **38,912 disjoint source windows** are prepared and independently checked. **2,023 CPU tests**, smoke and compilation pass. This stage used zero released-model calls and zero accelerator chip-hours. It supplies an executable objective and fixed comparison, not a learned baseline result.

## One loss for every original token

Each window contains 528 original native tokens. Training predicts all 527 shifted positions once, with no appended delimiter, synthetic EOS, auxiliary forward or extra continuation weight. At batch 256 the coefficient on every eligible token is **1 / 134,912**, including the original final 16 tokens. Real source EOS IDs remain valid targets.

The generic [packing and schedule module](../../train/uniform_lm.py) uses the existing [causal CE kernel](../../train/masked_causal_ce.py) and [training engine](../../train/partial_state_driver.py). Eight accumulated microsteps, eight ranks and four examples per rank cover each global batch once. A tiny Qwen dense reference checks the loss and every parameter gradient against the accumulated rank/microbatch calculation. These CPU tests establish the mathematical path; native BF16 accumulation remains a numerical qualification question.

This changes the definition of the control relative to V29, whose raw arm deliberately duplicated/upweighted selected continuations. It does not identify a hidden V29 implementation bug or prove which earlier hyperparameter caused regression.

## Fixed source allocation

| Cohort | Source documents | Predicted tokens |
|---|---:|---:|
| TRAIN | 32,768 | 17,268,736 |
| DEV | 1,024 | 539,648 |
| CONFIRM | 4,096 | 2,158,592 |
| QUALIFICATION | 1,024 | 539,648 |

A fixed hash ranks eligible unused source identities, allocates these disjoint cohorts, and chooses one bounded native position per document using identity and length only. The selection is frozen before reconstruction. Arithmetic labels, serialization success and model outcomes do not choose or replace windows. All prior reader and fresh-confirmation sources are excluded.

The independent checker reconstructs the complete split/cut choices and retokenizes **every selected original Parquet row**, covering all 16 pinned shards. All 38,912 pass; no row was replaced. Source preparation took **121.43 seconds**, independent checks **125.00 seconds**, each bounded to four CPU cores. The dataset occupies approximately 108 MiB.

This is the existing source-clean, partly depleted FineMath frame, sampled by eligible document with one bounded cut. It is not all-corpus token weighting or proven novelty relative to base pretraining. CONFIRM outcomes are reserved for after recipe/checkpoint selection. [Source plan](../../configs/uniform_lm_sources.json), [preparation](source-preparation.json), [independent check](independent-source-check.json).

## Four development settings, then confirmation

Each recipe starts from unchanged Qwen3-1.7B-Base and sees the same source order once: **17.269 million predicted tokens**, batch 256, 128 optimizer updates. Compare peak LR **3e-6 / 1e-5** with **warmup-constant / warmup-cosine**, 13 warmup updates, and a 0.1 peak-LR floor for cosine. AdamW settings otherwise remain fixed.

All four terminal checkpoints must be frozen before DEV scoring. Choose the lowest full-527-token DEV NLL, with a fixed tie rule. Freeze that recipe and already-trained checkpoint before evaluating CONFIRM against B0. The confirmation endpoint is a positive paired lower confidence bound for the full-window NLL improvement; report the final 16 tokens separately. The existing 1,024 calculation sources are a separate, previously inspected diagnostic, excluded from fitting.

The schedules update on optimizer steps, not microsteps, and are a deterministic function of the saved source cursor on resume. Five concrete protocols (qualification plus four development settings) pass CPU source, order, first/last real-batch and schedule checks. Every development protocol has identical source-order hash and token coefficients. [Design](../../configs/uniform_lm_baseline.json), [protocol checks](cpu-protocol-check.json).

## Native work to qualify

The actual training precision is **BF16 parameters/gradient storage**, with FP32 masters/moments, rotary buffers and CE reduction. Explicit `--auto-cast=none` does not turn BF16-input matrix multiplication into FP32 computation. Native execution must qualify the accumulated gradient/update path, actual base-optimizer LR across warmup/decay, replica agreement, full-state restore at update 2, and measured runtime before the comparison.

The design proposes **18 reserved chip-hours including at most 0.5 for qualification**, at most two existing physical chips. This is a ceiling to assess against measured timing, not a cost measurement or execution release. Existing RAM-backed storage can hold bounded full states; terminal BF16 copies fit the existing NVMe allocation while preserving disk and host-memory reserves. The source/CPU stage created no native release.

A negative four-setting result would remain negative. The plan is a finite comparison, not repeated tuning until B0 is beaten. A positive confirmation would establish a competent ordinary-training reference conditional on one model and training order; it would not itself show a benefit from intermediate computation.
