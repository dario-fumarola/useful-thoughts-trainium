# V15: a stronger supervised reference on real documents

The goal is a better model from intermediate computation, with its cost counted.
V10 established a narrow complete-number benefit from generated calculations,
but failed both full-continuation likelihood gates. It did not compare against
answer-focused supervision. This experiment closes that gap before RL or scale.

1. Keep released Qwen3-1.7B-Base and the exact verified TRAIN5120. No posterior
   teacher or uncertain-future annotation: every trace is executable from the
   original prefix, includes its root operation, and has variable depth.
2. Freeze384 fresh developmental documents from the unused source TRAIN pool.
   Exclude actual training documents and exact ordered value/operator ASTs,
   plus both previous arithmetic panels. Hash order selects384 of499 distinct
   eligible expressions. No model outcome selects the cohort; official TEST
   remains unopened. These are arithmetic documents, not general-language QA.
3. Train two fresh single-model recipes, seeds2026101913/14,320updates each,
   global32,exactly two passes. Direct SFT learns the original future16 only.
   Full-trace CoT SFT learns the exact calculation, `Answer:` boundary and
   original future16. Prefix and supplied opening have no loss. Each recipe
   uses one per-example CE mean,constant1e-5,BF16 active storage/FP32 masters,
   one ZeRO reducer,width640,target128. Full160/320 checkpoints;terminal primary.
4. Qualify the actual new target shape and both masks before science. The CoT
   qualification must save full state,close its process,restore in a fresh
   process and continue ordered updates. The direct arm also needs actual
   finite changed-master updates. Existing CPU or B0 forward-only checks are
   insufficient. Preserve failed attempts and all original outputs.
5. Evaluate B0,the four new models,and the fixed V10 raw/prior models from both
   seeds on the SAME new384 documents. Record raw,empty,generated own,intact
   other,and supplied exact traces. Greedy note cap128,original native stops,
   fixed full `Answer:` closure;greedy continuation32,EOS only. Keep unchanged
   initial-scalar-v2 and every unsupported/error in the denominator. No retries,
   resampling,shortening notes,checkpoint selection or parser changes.
6. Primary contrast: CoT-SFT own minus direct-SFT raw exact scalar accuracy.
   Require at least3 percentage points and paired95% document-bootstrap lower
   bound>0 in EACH seed. Also require positive lower bounds versus same-CoT
   empty and B0raw in each seed,and zero execution errors.20k draws,seed2026101917;
   intervals conditional on checkpoints,not uncertainty across training seeds.
   Failing this gate closes the practical supervised-reference question;do not
   keep tuning until it passes. Passing supports only this supervised reference.
7. Secondary: compare both recipes with frozen V10raw/prior paths. Report full16
   likelihood,predefined numeral/boundary/tail components,and ordinary FineMath
   DEV256 token NLL through the same raw path. Report costs/regressions regardless
   of the primary outcome. The ordinary panel is reused development,not a new
   broad-domain test. No pooled seeds/panels and no post-hoc primary substitution.

Both recipes match document exposures,updates and padded shapes. They differ in
active target counts and inference computation;this is a practical reference,
not an isolated architecture or prefix-mask ablation. CoT uses its trace on both
passes,whereas V10 used one augmented and one plain occurrence. Actual label
inventories,tokens,compile/warm/worker time and all failures must accompany results.
Do not describe a CoT gain as thought-augmented pretraining benefit or emergence.

The result determines whether this small setup establishes a competitive
supervised computation reference. A later training-scaffold question could apply
the SAME post-training objective to raw-CPT/post-CPT starting weights;it is a
different comparison,not released by this design. Broader document reasoning
also remains a separate next step. No8B,broadRL or split-system rescue is released.

The reference choice follows STaR's distinction between directly fine-tuned
answers and rationale training;we are not implementing its iterative sampling
or answer-conditioned rationalization. [STaR](https://arxiv.org/html/2203.14465).
TPT's post-training comparisons also motivate keeping scaffolding benefit
separate from generated computation at inference;this tiny real-data reference
is not its large-scale recipe. [TPT](https://arxiv.org/html/2509.20186v1).
