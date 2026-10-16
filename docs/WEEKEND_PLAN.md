# The weekend experiment

## 1. What Monday should answer

The strongest attainable outcome is a reproducible mature-model reference condition in which thought training improves a specified capability, with separate evidence about whether the generated content is useful at inference. A useful negative outcome identifies a tested boundary: initialization, implementation, reader compatibility, writer learning, deployment policy, or insufficient exposure. It does not condemn all pretraining-time reasoning.

We will not try to prove early emergence, zero-supervision bootstrapping, social intelligence, safety and cheap inference simultaneously.

## 2. Starting condition

Use **Qwen/Qwen3-1.7B-Base**, actual released pretrained weights. This size has a published RLP success condition and is much less ambitious to integrate than a new 8B/MoE model. The specific model/data/layout proposed here are not an exact reproduction of that work. The public repository checked on October 16 still lists training code as forthcoming.

If memory or backend integration fails within the allocated qualification window, switch **both arms** to **Qwen/Qwen3-0.6B-Base** and create a new experiment ID. Do not compare the two model sizes as a pure maturity test. The later maturity experiment holds 0.6B architecture, tokenizer and controlled pretraining history fixed across checkpoint ages.

Preserve the intended embedding/head tie of the released checkpoint, including after transfer to XLA. Historical untied raw project checkpoints need different handling. Never silently retie two independently trained matrices.

## 3. Data

Prefer the surviving, audited mathematical raw-text data when document boundaries, tokenizer identity and held-out exclusions can be established. Otherwise use a pinned **FineMath-4+** subset. This is a math-rich proof-of-feasibility setting, not a claim of arbitrary-web generality.

FineMath's authors describe benchmark decontamination, but still audit our selected data against the exact benchmark revisions. The supplied record script does exact dedup and document grouping, not near-duplicate or benchmark decontamination. Existing base-model contamination cannot be removed by checking only the new training slice; common initialization controls some comparison concerns, not every interpretation.

Create a fixed shuffled manifest with one context anchor per document, chosen independently of model entropy and target difficulty. Use the last 512 context tokens, store 16 future tokens for analysis, and train the reward on the first one. Reject too-short documents and log their count. All versions of a source page belong to one split. Save document IDs, dataset/tokenizer revisions, anchor IDs and file hashes. Require enough new documents to avoid accidental cycling in the data-matched run; any repeat is logged as a repeat, not new tokens.

The writer accepts prefix tokens only. Reader scoring can see the true future under teacher forcing. No answer-aware selector is allowed at deployment. The ordinary CPT arm sees exactly the same prefix plus the trained one-token future, not the extra 15 tokens stored only for analysis.

## 4. Primary arms

| Arm | Training | Required checkpoint |
|---|---|---|
| B0 | None; unchanged released base | Starting reference |
| C | Ordinary next-token continuation | At exactly the same source-record count as R |
| R | Signed group-relative thought-policy update | At the preregistered terminal step |

Both trained arms start from the same model snapshot and use the same ordered records within each seed. Give the second seed its own fixed, recorded permutation of the same data pool; share that permutation within the second pair. Use the same initial learning rate in the primary objective comparison. A strong claim against CPT ultimately needs sensible tuning, not only identical hyperparameters. A cheap, predeclared higher-LR CPT control is appropriate after the primary pair fits; do not spend Friday tuning R extensively while leaving C intentionally weak.

Save C at the data-matched boundary before continuing it, when practical, to the same measured resource allowance as R. Call that a **resource-matched** comparison, not exact FLOP matching. Count repeated data separately. If it is not completed, make no training-efficiency claim.

## 5. Proposed initial recipe

These are explicit starting choices, not guaranteed effective hyperparameters:

| Setting | Choice |
|---|---|
| Model | Released Qwen3-1.7B-Base |
| Context | 512 real tokens plus fixed prompt delimiters |
| Thought cap | 1,024 generated actions; natural stop allowed |
| Independent samples | 4 per prefix |
| Reward horizon | 1 real next token |
| Temperature | 0.7, same distribution in generation and gradient replay |
| Prefixes per update | 128, e.g. 32 ranks × 4 accumulation micros × 1 prefix |
| Backward microbatch | One thought sequence at a time if needed |
| Learning rate | Constant 1e-6 in primary R and C |
| AdamW | beta=(0.9,0.95), weight decay=0.1; record epsilon/version |
| Gradient clipping | Global norm 1 after the single intended reduction |
| Precision | FP32 parameters/master accumulation; qualified BF16 compute |
| Sharding | Verified ZeRO-1 for 1.7B; do not assume full Adam fits each core |
| Extra losses in R | None: no next-token, reader, gate or teacher-hint auxiliary |
| Primary checkpoint | Fixed terminal step, not whichever test score is highest |

RLP's length ablation motivates not returning immediately to six- or 64-token thoughts. G=4 and L=1,024 are a combined engineering choice; their individually reported ablations do not guarantee that combination succeeds. If they do not fit, reduce unique-prefix batch first while preserving group size; use more accumulation when possible. Do not secretly make L=32 and call the outcome a test of the long-thought hypothesis.

A later L=2,048 condition gets its own identity. No within-run compression or horizon curriculum this weekend. Evaluate K=4 and K=16 as declared out-of-training-horizon measurements.

## 6. The update

For context c, generate G independent thoughts z_i from the current policy. Score the true next token w without gradients:

    s_i = log p_current(w | c, z_i)
    b   = log p_current(w | c)
    r_i = s_i - b
    A_i = r_i - mean_{j != i}(r_j)

Update all sampled thoughts with signed advantages, not only winners:

    loss = - mean_i [ A_i * mean_{sampled actions t} log q_current(z_i,t | c,z_i,<t) ]

The first sampled stop action belongs to the policy loss. Padding after it does not. At a hard cap, do not invent a sampled stop action. Keep EOS and thought-end semantics explicit. The supplied format declares both the thought-end token and base EOS as terminators. Include the actually sampled terminator in the policy loss, remove it from note content, and render the reader close delimiter consistently. Record which terminator occurred.

One fresh update gives the same first-step gradient as the paper's tokenwise clipped surrogate evaluated at behavior=current. Multiple policy epochs are a different implementation and require stored behavior log probabilities and tested importance ratios. Per-completion normalization is the default, matching the paper's equation; fixed-cap normalization exists only as a named later ablation.

**Baseline cancellation:** any baseline common to all samples of one context cancels exactly in A_i. Therefore we omit EMA storage from this minimal gradient path rather than pretend it changes these advantages. This equivalence holds for the written update, not for pipelines with additional clipping, gating or asymmetric scaling before centering. It does NOT prevent a changing shared reader from changing the reward landscape. Report absolute evidence and evaluate frozen-reader transfers separately.

## 7. Hardware and wall-clock plan

A trn2.48xlarge contains 16 Trainium2 chips. At the default LNC2 arrangement it has 64 logical NeuronCores with about 24 GB per logical core. Total HBM is not a single process's usable memory.

Start with two isolated 32-logical-core partitions only after runtime visibility and resource isolation are verified. If partitioning is troublesome, run arms sequentially on the full node. Preserve global batch with accumulation changes and record all topology changes. Do not debug partitioning for half a day.

Use the existing working XLA/Neuron environment. Current vLLM/Neuron availability does not certify custom training or rapid rollout weight transfer; this is not the weekend to migrate serving platforms. The doctor script is read-only and captures a whitelist of environment variables without credentials.

### Friday: qualification, bounded to roughly 6 hours

Inventory the actual device layout, software versions, available checkpoints/corpus and storage. Pin model and data revisions. Verify actual pretrained loading, tokenizer IDs, embedding ties, correct labels, prefix-only generation, same-policy replay log probabilities, one gradient reduction, finite parameter updates, and exact checkpoint/resume behavior. Run a tiny overfit test and a reader-interface check on non-evaluation examples.

Measure at least 10 steady-state **complete optimizer steps** after warmup. Include rollout generation, scoring, backward, all-reduce and optimizer work. A peak accelerator FLOP number does not predict this runtime. Detect continuous recompilation; use static padded shapes and cached decoding. Do not copy vocabulary-sized logits to the CPU each generated token.

Then choose the largest step budget among **256, 512, 1,024, 2,048** that fits an approximately 12-hour per-pair execution allowance with a 15% reserve. This selection uses timing only, not validation outcomes. Lock the number and rerun the primary pair from clean initialization after any algorithm-changing debug work. Small budgets are explicitly feasibility pilots; they do not replicate the paper's much larger exposure.

### Friday night / Saturday: first paired run

Run C and R, save quarter/half/terminal checkpoints, and record metrics. Terminal is the primary outcome. Periodic development evaluation is allowed; final held-out results are not used to select checkpoints. A small CE increase or negative initial utility is not an automatic kill condition.

Once C reaches its data-matched boundary, its partition can evaluate B0/C, perform the resource-matched extension, or prepare replication. Do not use all remaining capacity for another reward variant.

### Saturday / Sunday: confirmation and evaluation

Repeat the fixed pair with seed 5678 when the measured budget permits. The same pretrained initialization is intentional; independent training randomness is the replication. If only a partial second run fits, report it as partial rather than a matched confirmation.

Reserve the last 10–12 hours for inference evaluation, integrity checks, paired analysis and backups. This is a budget reservation, not an estimate of how quickly the full suite will run. Prioritize the locked primary task and required controls if actual evaluation throughput is lower than expected.

## 8. Evaluation matrix

Use unchanged base B0, data-matched C, and R. For each, evaluate direct answering and one thought-enabled answer at the same inference budget. Compare R versus C **within each mode**. R+thought beating C+direct alone does not isolate a training improvement.

The primary downstream contrast is **R with one thought minus C with one thought** at the data-matched terminal checkpoints. Required secondary contrasts are R-direct minus C-direct and R-thought minus R-direct. This prevents choosing whichever inference comparison happens to win.

Primary downstream: GSM8K official test, one fixed prompt family, one generated path, predeclared answer extraction and length limits. Qualify formatting on development/training examples before final testing. Failure to format answers is reported, never silently discarded. Secondary: ARC-Challenge candidate likelihood and LAMBADA as a regression sentinel. The existing project benchmark is CPU-only and tailored to the old model; its run script is not a plug-and-play 1.7B/long-thought evaluator. Keep its item definitions but qualify the model loader and accelerator scoring path.

Local inference audit: 2,048 fresh documents, K=1 primary, K=4/16 secondary. Report raw silence, empty format, own samples, length-matched filler, unrelated note and shuffled note. Average independently generated samples per document for random-policy expected utility. Never treat samples as independent documents. Use one greedy note for deterministic deployment evaluation. Report maximum/oracle scores only in a separate diagnostic section.

On a smaller fixed development subset, cross the initial and trained writer/readers. This separates changes in note content from reader adaptation and clean-baseline drift. It is a localization test, not a requirement that all useful thoughts transfer to a mismatched reader. A supplied correct next token is a plumbing control, not evidence of semantic reader competence.

## 9. Monday decisions

| Result | Interpretation | Next move |
|---|---|---|
| R direct beats C direct | Training benefit | Replicate; identical short post-training comparison |
| R thought beats R direct and content controls | Inference benefit in this setting | Replicate; diagnose actual thought content and cost |
| R beats C only when R alone gets more inference tokens | Extra-compute comparison only | Give C the same thought budget |
| Oracle tail rises but ordinary policy does not | Useful candidates may exist; deployed policy still fails | Frozen/crossed-reader analysis before adding a selector |
| Own beats filler but loses to raw silence | Content signal remains below insertion cost | Reader/interface experiment |
| No gain after very few updates | Underexposed pilot | Extend the fixed recipe; do not declare the idea impossible |
| Numeric/layout checks fail | Implementation result | Repair and restart; do not classify as scientific failure |

Run paired, document/item-level confidence intervals and show each training seed. A CI over items cannot substitute for replication over training runs. Two seeds are a start, not a final power guarantee. Report actual denominators, unscored failures, truncation rates, latency and data/resource counts.

## 10. Stop changing the question

No paper rewrites during the run. No safety mixture, BDI target, 8B migration, latent-state redesign, selector training, or compression schedule in the primary experiment. No autonomous agent may change the objective, losses, split, model size, stopping rule or chosen checkpoint without creating a new experiment ID and obtaining a human decision.

If Neuron integration is not qualified by the timebox, do not launch a weekend-long unverified RL job. Complete the mature model baseline, evaluate surviving late checkpoints if available, and finish a short controlled plumbing run. That is a useful infrastructure deliverable, but must be labeled as such.
