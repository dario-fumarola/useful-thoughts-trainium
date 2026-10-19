# Useful Thoughts on Trainium

**The document-QA study is complete: generated reasoning helps the same trained
model, but its advantage over matched direct training does not pass the declared
two-seed gate. Post-answer reasoning supervision shows no direct-accuracy improvement here.**

This is a small supervised reference using Qwen3-1.7B-Base, real MuSiQue documents,
and Trainium. It does not establish broad reasoning, pretraining benefit, or
novelty relative to the model's original pretraining. Official DEV/TEST are unopened.

## Completed document-QA comparison

Six fresh models, two data-order seeds, the same 2,048 training questions and four
passes (256 updates). Every example combines 0.5 mean source CE over 256 fixed
source positions and 0.5 mean task CE. Direct training predicts the answer;
prior training mixes direct answers with full-trace CoT; post training mixes direct
answers with reasoning after an answer stop. Task-label counts differ by arm.
Terminal checkpoints are primary; the held panel has 512 questions with disjoint
supporting evidence, pages and original question identities relative to training.

| Model / path | Seed1921 correct /512 | Seed1922 correct /512 |
|---|---:|---:|
| Unchanged B0, fixed-demo conventional CoT (shared) |92|92|
| Matched direct training, direct answer |201|195|
| Prior training, direct answer |201|180|
| Prior training, empty reasoning format |190|169|
| **Prior training, generated reasoning** |**217**|**227**|
| Prior training, fixed-demo conventional CoT (secondary) |224|222|
| Post-answer training, direct answer |200|186|

Canonical whole-answer EM is primary; all failures remain in denominators.
Canonical token F1 and separately attributed answer-alias metrics are also saved.
Generated reasoning's F1 is 0.509/0.533 versus 0.490/0.470 for matched direct training.
Its attributed-alias EM is 238/512 and 247/512; these do not replace canonical scores.

| Predeclared contrast | Seed1921 gain, pp [95% CI] | Seed1922 gain, pp [95% CI] |
|---|---:|---:|
| Post direct − matched direct |−0.2 [−3.5,3.1]|−1.8 [−5.1,1.6]|
| Generated reasoning − same-model empty |+5.3 [1.2,9.4]|+11.3 [7.4,15.2]|
| Generated reasoning − matched direct |+3.1 [−1.0,7.2]|+6.3 [2.3,10.4]|

The gate requires a point gain of at least 3pp and a positive paired lower bound
in **each** seed. The overall inference and practical-value gates therefore fail;
the post-answer training gate also fails. The local reasoning-versus-empty gain
is positive in both seeds, entirely among pairs with complete answer fields.
Intervals use 20,000 paired question bootstrap draws, conditional on each checkpoint,
without pooling seeds or adjusting intervals for multiple comparisons.

![Primary contrasts](validation/document_QA_v21_complete_20261019/primary-contrasts-focused.svg)

## What the controls and outputs show

The ordinary-trained model's fixed-demo CoT path scores only **1/512 in each seed**,
with 510 unsupported outputs each. It usually fails the required answer format.
Large gains over that degraded path are not evidence of a strong CoT baseline win.
A labelled posthoc rule—use the anchored final field, otherwise the first complete
line—recovers191/182 on those same outputs, below native direct201/195. It changes
no primary score or gate; a future parser must be fixed before fresh evaluation.
All paths and format counts remain in the [complete tables](validation/document_QA_v21_complete_20261019/all-answer-failures-costs.csv).

Prior-model accuracy falls to 11/23 without documents and 148/144 when one annotated
bridge paragraph is removed. Own-note replay scores 217/227 versus 11/33 with an
unrelated note. These are evidence/content diagnostics, not perfect causal isolation.
Supplied oracle traces score 507/512 and 512/512, but contain the root answer.

The additional gain is concentrated in two-hop questions. Three- and four-hop
cells show no consistent advantage; there are only 104/23 such questions. Stored
traces also contain wrong intermediate facts and citations. Some correct final
answers follow visibly incorrect intermediate claims. Literal presence of an
answer in a cited paragraph does not establish that the stated relation is supported.
One fixed example’s gold citation contains “Ohio” in a military unit name without
stating the asked place-of-death relation. Teacher grounding also needs inspection.

Post-answer training does improve gold-answer log likelihood over matched direct
training by 0.38/0.93 nats, with positive paired intervals, despite no direct-accuracy
gain. This secondary predictive result does not pass the declared accuracy gate.

Accuracy and likelihood disagree: own notes reduce gold-answer span likelihood
versus empty by **7.34/7.75 nats**, mostly in answer content rather than EOS.
Correct canonical answers have positive average gains, while other written claims
can make the gold surface much less likely. This does not justify restarting the
old likelihood-reward RL recipe. Exact-match and exact-surface likelihood also
penalize valid alternative phrasing; alias and F1 diagnostics remain separate.

## Cost, limits and reproducibility

Generated reasoning uses 57.4/56.3 actions per question versus 5.37/5.37 for matched
direct training. Observed median execution times are about 4.96/4.92 versus 1.03/1.03
seconds under concurrent jobs; this is not a controlled serving-latency benchmark.
Prior training's ordinary-page NLL is 0.097/0.110 nat/token worse than B0, although
it is better than matched direct training by 0.036/0.020. This is specialization
with a measured cost, not broad language-model improvement.

All **20,992 generation attempts and 19,712 likelihood rows** completed with zero
execution errors. Raw outputs, unsupported cases, native targets, examples and
analysis sources are preserved and read back on EBS. All six terminal model files
and a complete optimizer recovery state are protected; EBS is same-instance.
Numerical parity establishes execution integrity, not learning. A derived checker
repair replaced exact FP64 summary equality with a 1e-10 comparison after an
observed 2e-12 reduction difference; original scientific and numerical gates,
outputs, metrics, intervals and the primary report hash remain unchanged.

- [Complete result and literal gates](validation/document_QA_v21_complete_20261019.json)
- [Frozen design and release](validation/document_QA_v21_science_release_20261019.json)
- [Actual training and checkpoint evidence](validation/document_QA_v21_training_20261019.json)
- [Independent readout check](validation/document_QA_v21_complete_20261019/independent-readout-check.json)
- [Fixed-hash examples](validation/document_QA_v21_complete_20261019/fixed-hash-eight-examples.json)
- [Format, alias and hop diagnostics](validation/document_QA_v21_complete_20261019/format-alias-hop-inspection.json)
- [Posthoc serialization diagnostic](validation/document_QA_v21_complete_20261019/posthoc-serialization-diagnostic.json)
- [Stored-trace diagnosis](validation/document_QA_v21_complete_20261019/stored-trace-content-diagnosis.json)
- [Resource accounting](validation/document_QA_v21_complete_20261019/resource-accounting.json)
- [Earlier results, including V17's unchanged failed quality gate](docs/EXPERIMENT_HISTORY_20261019.md)
- [Current queue](NEXT_STEPS.md)

The follow-up [TRAIN-only audit](validation/document_QA_v22_TRAIN_audit_20261019.json)
examines 24 questions and 72 cited relations: 57 are locally supported, eight have
insufficient citations, six are ambiguous, and one has a temporal contradiction.
Four questions switch between different counties or cities sharing a name.
All 72 citations match the original source annotations. This is an exploratory
single-agent review, not independent adjudication or a population error estimate;
insufficient citations do not establish that the answers themselves are false.

The active [bounded pilot](validation/document_evidence_plan_v23_pilot_release_20261019.json) tests passage
selection before answering, with the same answer format and explicit final-answer
coefficient in both arms. Its [CPU preparation](validation/document_evidence_plan_v23_CPU_readiness_20261019.json)
passes data, native packing and gradient checks. The [TRAIN-only decoder check](validation/document_evidence_plan_v23_decode_qualification_20261019.json)
passes native/cache/CPU comparisons; the [training and restore check](validation/document_evidence_plan_v23_training_hardware_20261019.json)
passes 11 actual updates, six complete state audits and a fresh-process restore.
Both fresh models have [completed training](validation/document_evidence_plan_v23_training_20261019.json)
for one pilot seed. All four periodic full states are audited; terminal models and
the latest full recovery state are protected. Fixed evaluation is being prepared;
there is no new capability result.
The primary contrast is selection versus matched direct training on 255 fresh questions.
An inference-assistance claim also requires gains over the selection model's own direct
and empty-selection paths, as [declared before training](configs/document_evidence_v23_readout.json).
Selecting annotated evidence will not itself prove connected reasoning or correct
attribution. Replication is conditional; no RL or model-size jump is released.
