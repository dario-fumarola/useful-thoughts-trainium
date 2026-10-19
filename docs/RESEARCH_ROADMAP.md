# Long-term proof ladder

## Current causal chain — October 19

The north star is **a model that computes from its available prefix and makes
better fresh predictions at a worthwhile cost**. A thought can reorganize or
derive information from that prefix; a future-aware annotation can additionally
reveal which uncertain continuation occurred. Those are different experiments.
The synthetic study isolates the computation channel. The current mainline tests
training and inference effects separately on verified real mathematical documents.

| Step | Evidence required | Current state and next action |
|---|---|---|
| 1. The reader can use computation | Correct states improve the same reader's answers with content controls. | Pass in the two-seed study: teacher accuracy48–83%, with substantial remaining calculation errors. |
| 2. The writer can supply it | Prefix-only computation is correct, relevant, complete and naturally stopped. | The bounded varied-depth repair reaches409/512 correct, complete, naturally stopped composition states versus91/512 for its fixed-depth control. At unseen depth6 only24/256 states qualify; length extrapolation remains weak. |
| 3. The composed channel is useful | Student notes beat direct, empty and unrelated notes on the same reader. | The repaired writer gives the same frozen reader306/512 composition answers versus124/512 with the fixed writer,empty7,and unrelated1. The declared repair gate passes. Longer-depth answers remain10/256. |
| 4. The system earns its cost | Compare strong direct, compact and full CoT with all models, updates and tokens accounted for. | No matched split advantage in the completed two-seed study. The new one-seed curriculum diagnostic exceeds frozen fullCoT on composition, but curricula/target supervision differ and it loses at familiar depth. Keep single-model methods as the mainline. |
| 5. The mechanism survives real data | Source-matched real-document endpoints, with genuine prefix-derived or explicitly posterior labels. | The completed two-seed raw/prior/post study fails both declared full16 likelihood gates. Its prospective scalar secondary improves by9.0/6.6pp over raw CPT with positive paired intervals. Training-only improvement remains unestablished. The completed fresh384 reference gives CoT342/335 versus direct-SFT275/285 and empty261/263; both declared gates pass. Ordinary NLL regresses0.128/0.147 nats/token versus B0. |
| 6. Self-improvement adds value | Improve beyond the supervised reference without an external teacher at deployment. | Held. First complete the real-document single-model reference; distinguish training-scaffold value from extra-inference value. |

The [bounded repair](../validation/variable_depth_writer_v11_complete_20261019.json)
is a positive curriculum intervention, not a reason to reopen an unlimited split
rescue. It supports useful prefix-computable states within the covered depth range;
it does not establish robust longer execution or a matched architecture advantage.
All9,216answers/2,048notes, including unsupported outputs, are published.

The completed fixed-depth comparison is
[fully public](../validation/partial_state_v7_matched5120_complete_20261018.json).
Its [post-hoc trace analysis](../validation/partial_state_v7_matched5120_descriptive_20261019.json)
finds a fixed-depth behavior consistent with learned stopping shortcuts. Larger
training exposure on the same trace depth did not establish compositional
execution. This is a measured boundary, not evidence that all intermediate
computation or thought-augmented pretraining fails.

The mainline uses one model and verified real text. A full-trace CoT-style arm tests
computation before prediction; a post-document explanatory arm tests training
value. Both need a strong ordinary continuation control, explicit label/update
costs and source-matched held endpoints. Exact posterior-note recovery is not a
universal requirement for latent training; see [paper notes](REFERENCE_NOTES.md).
The completed [likelihood study](../validation/real_trace_cpt_v10_complete_20261019.json)
and [prospective scalar secondary](../validation/real_trace_cpt_v10_scalar_secondary_20261019.json)
retain their separate endpoints. A stronger
[direct-SFT versus full-CoT-SFT reference](../validation/real_target_sft_v15_design_20261019.json)
now completes all nine models on fresh384 documents. Its
[declared gate passes in both seeds](../validation/real_target_sft_v15_complete_20261019.json),
with a material ordinary-text regression. This establishes a supervised inference
mechanism, not thought-augmented pretraining benefit. Preserving source prediction
while retaining computation is the next bounded question.
The [live status](../STATUS.md) defines the next action; this roadmap adds no launches.

Broader document reasoning remains a data-only preparation. A
[TRAIN-only split audit](../validation/document_evidence_v16_directed_split_20261019.json)
shows that held supporting evidence can be excluded from every training-visible
paragraph while retaining a candidate learning set. It supplies neither model
results nor longer-context hardware readiness. Finish the current complete
comparison before releasing that separate question.

**Capacity is an intervention, not an assumed remedy.** A7–8B writer may produce
more reliable work, while a stronger reader may already solve more without help.
Use the completed teacher/student contrast to choose the intervention. Qwen3-8B-Base
shares the qualified architecture family, but its BF16 weights plus gradients
alone exceed the current24GiB-per-logical-core replicated layout. BF16 inference,
frozen-backbone adapters, or actual parameter/gradient sharding need their own
bounded qualification; changing only a model name is insufficient. The metadata
and lower-bound sizing are in
[the capacity assessment](../validation/qwen3_8b_capacity_assessment_20261018.json).
No8B weights or training have been launched.

## Central question

**What initial competence and training signal allow a language model to discover useful intermediate computation, and can that computation later be internalized or compressed?**

Published work already shows benefits from thought-related objectives in mature models. A positive mature-model reference is therefore an enabling result, not the headline novelty. The valuable next contribution is to determine when it becomes possible, what changes, and which benefits survive tighter constraints.

## Step 1: a trustworthy reference condition

Establish a correctly initialized, synchronized, end-to-end procedure on a mature base. Publish or preserve the exact data, prompts, update, labels, sampled actions, masks and evaluation. Reproduce at least one capability outcome over independent training seeds. A negative local likelihood curve does not cancel a separately measured downstream gain.

This stage addresses the reference condition. A small exposure remains a feasibility study. Infrastructure correctness is necessary, not itself proof of scientific efficacy.

## Step 2: distinguish the source of the gain

Separate ordinary-inference training benefit, same-weight inference benefit, same-inference-budget deployment benefit, and content-sensitive effects. Use crossed writer/readers, format/filler/order controls, and selectively altered intermediate facts. Do not require every beneficial thought to transfer between mismatched readers; that is a test of portability, not a universal definition of utility.

A fixed filler can itself provide useful extra hidden computation. Beating one arbitrary filler is not definitive proof of semantic reasoning; ultimately compare against an appropriately trained pause/extra-compute control and relevant supervised baselines. An attractive trace is not a causal argument.

If training benefits appear but direct thought utility does not, retain that result. Test matched post-training and examine whether the objective learned useful representations or merely specialized to data/style.

## Step 3: locate the onset of useful thought

Hold model size, architecture, tokenizer and pretraining recipe fixed. Compare checkpoints at different training ages, each with its own matched ordinary continuation and identical thought-training allowance. The old young 0.6B model versus released 0.6B base is an initial contrast, but their different data/training histories mean it is not a clean token-count intervention.

A controlled series such as 9B, 30B and 100B tokens is a future design proposal, not a commitment that these are the relevant thresholds. Measure separately: reader benefit from supplied usable hints, frequency of useful self-generated samples, gain under ordinary deployment, and downstream benefit after thought training. Those components may reach usefulness at different stages.

A result that larger models work while young ones fail does not by itself show why. Use these measurements to distinguish available computation, writer discovery, reader compatibility and optimization noise.

## Step 4: quantify the minimum useful guidance

If unassisted discovery fails but a semantic seed works, vary a capped amount of teacher/prefix-derived guidance. Do not confuse handing over the exact future token with teaching a useful intermediate computation.

Compare the same seed alone against seed plus autonomous thought training. Remove the teacher completely in the latter phase. Use fresh held-out items and account for all teacher data and compute. Check whether the student improves beyond the warm-start checkpoint, rather than merely retaining distilled behavior.

A hindsight teacher may use answers during training, but target notes must be learnable from the student's available prefix for the desired deployment claim. Partial overlap with the final answer is not automatically leakage; hidden future access at inference is. Construct matched ablations rather than declaring hints invalid by word overlap alone.

BoLT is a relevant precedent, but its substantial teacher warm start is not evidence that a tiny dose is sufficient. Measuring the dose and when it can be removed could be a useful contribution.

## Step 5: compress after usefulness exists

Thought length L, prediction horizon K and the offset before the scored future window are different variables. Change one at a time. More target tokens than thought tokens does not force reasoning or semantic compression, especially under teacher forcing.

Start from the successful condition. Reduce L under a declared curriculum or distill to a shorter writer. Report performance versus actual generated tokens and latency, not just maximum cap. Compare against more direct-answer samples and a trained pause baseline at the same inference budget.

Then assess whether the training gain survives removing explicit thoughts altogether or after identical post-training. That distinguishes a useful training scaffold from a required deployment feature.

## Step 6: social reasoning as a controlled extension

A strong first task is explicit information asymmetry: one person sees an object move and another does not. The thought should distinguish reality, that person's belief and their goal. Change only who observed the event, then test whether the inferred belief and predicted action change appropriately.

Compare state-aware thoughts against response-only training and equally sized ordinary summaries. Do not reward invented confidence about underspecified motives. Evaluate new names, narratives, event orderings and held-out compositions. Structured belief accuracy and downstream action prediction are separate outcomes.

This is a better initial social claim than generic-looking BDI prose. It also supplies a causal test bed for the general computation mechanism. It is not itself a safety result: knowing another person's beliefs can support helpful or harmful behavior.

## Step 7: safety only with an explicit decision target

Define a policy, context-sensitive labels and the behavior being improved. Compare response-only, verdict-plus-response and explanation-plus-response conditions under matched data and appropriate compute accounting. Include benign near-neighbors and overrefusal, not only harmful compliance.

Distinguish assessing a supplied draft, predicting risk from the prefix, and assessing a generated draft before release. The true future can label training data but cannot enter a supposedly prospective inference policy.

A generic NTP penalty on an unsafe target still pushes imitation; a 'tax' is a measured tradeoff, not a mechanism that teaches safer behavior. Masking, safe alternatives and explicit preferences have different meanings and need separate comparisons.

Safety-reflection pretraining and deliberative safety objectives already have close precedents. New work on thinking tokens and safety also cautions that visible deliberation can be disconnected from the final decision. A stronger contribution would establish a specific causal benefit of the intermediate representation, retention through later training, or better generalization to context-sensitive distinctions. No 'safe from first principles' claim follows from a weekend accuracy gain.

## Paper discipline

The eventual paper should center on the strongest supported mechanism or boundary, not the number of attempted runs. Keep a compact main claim and a reproducible implementation. Preserve historical failures with their actual exposure and known defects. Do not rewrite them into clean replications, and do not let an audit ledger become the research product by default.
