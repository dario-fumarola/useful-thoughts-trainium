# Useful Thoughts on Trainium

**The passage-selection pilot is complete and misses its gate: generated selection
answers 129/255 correctly versus 127/255 for matched direct training. Supplied
support IDs help the reader, but self-generated selection has no established net benefit.**

This is a small supervised reference using Qwen3-1.7B-Base, real MuSiQue documents,
and Trainium. It does not establish broad reasoning, pretraining benefit, or
novelty relative to the model's original pretraining. Official DEV/TEST are unopened.

## Latest completed passage-selection pilot

Two fresh Qwen3-1.7B-Base models, one data-order seed, the same TRAIN2048 questions
and four passes. Source, final-answer labels and their explicit loss coefficients
match; pointer supervision has its own loss. Terminal256 is primary. The fresh
255-question panel is mostly two-hop; official DEV/TEST remain unopened.

| Model / path | Correct /255 |
|---|---:|
| Matched direct training, direct answer |127|
| Matched direct training, empty-field response |126|
| Selection training, direct answer |127|
| Selection training, generated IDs |129|
| Selection training, supplied-empty prompt† |126|
| Selection training, annotated support IDs |155|

The primary gain is **+0.8pp [−4.7,+6.3]**, below the declared three-point/positive
lower-bound gate. Selection versus the same model's direct path is also
+0.8pp [−4.3,+5.9]. All answer fields are complete, all255 generated selections
are syntactically valid, and no generation is capped. Replication is not released.
A failed gate does not establish that the true effect is zero.

†The supplied-empty prompt generates a new ID list on **255/255** questions.
It does not isolate selection omission. Its original score is preserved; both
models' direct paths genuinely generate no IDs. This caveat does not change
the failed primary or same-model direct comparison.

![Pilot contrasts with operational control caveat](validation/document_evidence_plan_v23_pilot_complete_20261019/primary-contrasts-with-control-caveat.svg)

The reader can use supplied selections: annotated IDs improve it by
**10.2pp [6.3,14.5]** over its own generated IDs, without adding answer text.
Generated sets exactly match the annotation on132/255 questions; mean support
recall is77.0%. In a posthoc split, exact-set selections gain14 answers relative
to direct answering, while other selections lose12. These groups are descriptive,
not causal estimates or semantic verification of the annotations.

Canonical pointer replay is more useful than donor IDs in likelihood
(+0.845 nats [0.326,1.373]) but worse than a genuinely empty replay
(−0.731 [−1.192,−0.299]). These are fixed-context likelihood diagnostics,
not native-generation likelihood or calibration. Generated selection uses
17.4 actions/question versus7.41 for direct training, about2.35×. Ordinary-text
NLL is still worse than untouched B0 by0.135 nats/token, despite a small advantage
over the matched control. No broad pretraining or ordinary-CoT superiority is shown.

All1,530 native attempts and3,062 likelihood rows are complete, independently
checked and protected. The original pre-held calibration-file failure and its
narrow IO repair are preserved; models, observations and gates are unchanged.
Successful training/evaluation used4.723 reserved chip-hours, plus0.062 for that
failed calibration attempt; hardware qualification is separate. Concurrent timings
are not a controlled serving-latency comparison.

- [Complete primary report](validation/document_evidence_plan_v23_pilot_complete_20261019.json)
- [Stored-output and empty-control diagnosis](validation/document_evidence_plan_v23_pilot_complete_20261019/stored-selection-and-empty-prefix-diagnosis.json)
- [Fixed examples](validation/document_evidence_plan_v23_pilot_complete_20261019/fixed-hash-eight-examples.json)
- [Independent check](validation/document_evidence_plan_v23_pilot_complete_20261019/independent-readout-check.json)

## Selector fitting follow-up

A frozen-model check on a fixed hash sample of the original training questions
finds incomplete support-set fitting, especially beyond two passages:

| Annotated support count | Exact selected sets | Correct TRAIN answers |
|---|---:|---:|
| Two |117/128 (91.4%)|127/128|
| Three |59/96 (61.5%)|96/96|
| Four |12/32 (37.5%)|32/32|

The predeclared fitting target fails on three-support questions. All256 answer
fields complete without errors or caps; one selection has duplicate IDs.
There are no identical rendered TRAIN inputs with conflicting support labels.
Twenty of the three-support outputs omit a required ID while choosing only
annotated supports. These are annotation-agreement diagnostics, not semantic
verification. Near-perfect TRAIN answers do not establish evidence selection or
generalization; the oversampled TRAIN panel cannot be pooled with held questions.

The check used the unchanged terminal model and prompt, no parameter updates,
and0.048 reserved chip-hours. Raw outputs, fixed examples and independent checks
are saved in the [complete fitting report](validation/document_selector_fit_v24_complete_20261019.json).
This narrows the next question to supervision/optimization before model scaling.

## Earlier document-QA comparison

**The document-QA study is complete: generated reasoning helps the same trained
model, but its advantage over matched direct training does not pass the declared
two-seed gate. Post-answer reasoning supervision shows no direct-accuracy improvement here.**

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

The completed passage-selection pilot above repaired final-answer weighting and
answer serialization, and removed supervised intermediate factual claims.
Its [training/state qualification](validation/document_evidence_plan_v23_training_20261019.json)
and [decoder qualification](validation/document_evidence_plan_v23_decode_qualification_20261019.json)
are execution evidence. Its failed capability gate remains the allocation decision;
no replication, RL escalation or model-size jump follows from the engineering passes.
