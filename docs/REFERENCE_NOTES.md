# Reference notes and verification dates

## Prior reviewed scientific basis (carried forward, not rerun here)

RLP motivates a mature Qwen3-1.7B base and thought-policy learning from future-text
predictive reward. Our G=4/L=1024/context=512/single-update configuration is a
combined engineering choice. It is not the complete published recipe, a tested
reproduction, or a guaranteed positive result. Common-baseline cancellation is
an algebraic property of the chosen LOO advantages; it does not fix a moving
reader or make variable-length normalization unbiased.

Quiet-STaR already uses a multi-token non-myopic prediction objective; extending
beyond the immediate next token is not novelty by itself. BoLT's teacher warm start,
TPT's thought-augmented data, and mature reasoning checkpoints in other work all
mean that different sources of initial competence must be separated experimentally.
Fast Quiet-STaR is relevant to later compression/internalization, not proof that
an extremely short initial thought budget will bootstrap this model.

Safety reflection and deliberative safety methods are already active research
areas. Keep their objectives distinct from predicting generic text or producing
plausible social commentary. The roadmap identifies narrower causal questions.
The primary citations and versioned paper links are in SOURCES.md.

## Warm-start interpretation — checked 2026-10-18

[BoLT v2](https://arxiv.org/html/2503.18866v2) trains both the prefix-conditioned
joint model and a future-conditioned posterior, with different formatting. Our
writer/reader/source loss does not reproduce that posterior training. Its appendix
also reports degrading few-shot GSM8K across some bootstrap iterations; likelihood
and downstream improvement need separate measurement. Its 480M unique-token,
8B training-token study is much larger than our initial interface warm start.

[TPT](https://arxiv.org/html/2509.20186v1) concatenates each original document
followed by its generated thinking trajectory. Our T arm interleaves a note at an
anchor and repeats the prefix tail. Keep calling it augmented causal CE rather
than a TPT reproduction; a negative small warm start cannot refute the paper.

## Packaging-time official documentation checks — 2026-10-16

- Codex reads project AGENTS.md instructions. The root file here is kept short;
  the launch script passes the separate opening task without changing permissions.
  https://developers.openai.com/codex/guides/agents-md
  https://developers.openai.com/codex/cli
- AWS documents trn2 logical-core configuration and runtime visibility controls.
  Device isolation must still be verified on the actual process layout.
  https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.1/about-neuron/arch/neuron-features/logical-neuroncore-config.html
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/configuration-guide.html
- The XLA ZeRO-1 tutorial says its optimizer step handles distributed reduction;
  adding xm.optimizer_step or manual reduction is not appropriate for that path.
  The page also warns of a future XLA-training support transition. Preserve the
  working installed stack; do not infer that a newly installed release is compatible.
  https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html
- Hugging Face documents commit-specific snapshot downloads and streaming datasets.
  The bundled helpers use explicit bounded network actions; live download APIs
  were not executed during packaging and still need instance-side checks.
  https://huggingface.co/docs/huggingface_hub/guides/download
  https://huggingface.co/docs/datasets/stream

These checks support interface choices, not hardware performance or scientific
claims. Pin resolved asset revisions and inspect installed library APIs at use.

## 2026-10-18: check the bootstrap mechanisms against the original papers

Quiet-STaR trains the prediction/mixing path with a likelihood loss alongside
rationale REINFORCE. Its published algorithm also drops negative rationale
advantages, and its learned boundary embeddings are part of the adaptation.
These are material differences from the stopped writer-only signed-advantage
recipe, not evidence that another unqualified RL run will work.
[Quiet-STaR, sections4.3–4.4](https://arxiv.org/html/2403.09629v2).

BoLT trains both a latent-conditioned joint model and a future-aware posterior
through two sequence orderings, after supervised initialization. Its later
candidate weights use the joint probability divided by the posterior proposal
probability, rather than continuation likelihood alone.
[BoLT, sections3.2 and4.1, Algorithm1](https://arxiv.org/html/2503.18866v1).
Our inference: a note selected only for predicting an observed future need not
be recoverable by a prefix-only writer. If returning to general text, distinguish
posterior usefulness from prior/deployed utility; do not describe the previous
positive-reader filter as the BoLT weighting rule.

STaR's arithmetic study distinguishes ordinary fine-tuning, rationale
bootstrapping and answer-conditioned rationalization, including digit-length
generalization. It does not establish that arbitrary future-aware annotations
will be useful under a prefix-only policy.
[STaR, sections4.1–4.3](https://arxiv.org/html/2203.14465v2).
For our current experiment, exact prefix-computable partial states eliminate
that information mismatch, while separately held structure/range/composition
panels test transfer. Correct teacher reading, actual student benefit and a gain
over ordinary compact CoT remain three different claims.

## Capacity and objective corrections — checked 2026-10-18

[RLP v2, sections2.4 and10](https://arxiv.org/html/2510.01265v2) explicitly uses
thought-policy updates without ordinary NTP, sibling-relative advantages and a
sample-length-normalized clipped surrogate. Its main model is Qwen3-1.7B-Base;
reported gains therefore do not establish7B as a prerequisite. Its introduction
mentions interleaved likelihood training, in tension with those explicit method
statements. Treat the equations and experimental settings as the recipe, and do
not present our stopped small study as a clean replication or rejection.
Its EMA ablation cannot be explained by varying only the shared scalar baseline
in the written sibling-centered update: that term cancels. This discrepancy
needs implementation detail; it does not justify adding an EMA as a remedy.

[RPT, section3.3](https://arxiv.org/html/2506.08007v1) starts from an already
reasoning-distilled14B model and filters low-entropy positions with a separate
small proxy. It is not evidence that the same bootstrap works on an unadapted
small base. [Quiet-STaR's appendixA](https://arxiv.org/html/2403.09629v2) uses
Mistral7B, learned boundaries and a learned mixing network; its capacity and
initialization differ together from our old experiment.

Our algebra/engineering implications: subtracting an empty-format or EMA constant
still cancels inside per-context LOO advantages; a real no-note choice or another
update rule is required to change that behavior. Saved behavior log-probabilities
remove recomputation of the old-policy denominator, but training still needs
differentiable new-policy log-probabilities. A zero thought-mixing weight also
zeros content-dependent reward until the prediction/mixing path learns to use it.
These observations guide the next design; they are not new training results.

## Training value and deployed thought value — checked 2026-10-19

BoLT models a latent marginal at prediction time. Its Appendix B.1 reports an
improving bound even while conditional reconstruction worsens slightly: better
prior/posterior agreement contributes to the gain.
[BoLT v2, sections 3.3 and B.1](https://arxiv.org/html/2503.18866v2).
For prefix X, continuation Y and thought Z, the conditional variational identity is

$$
\log p(Y\mid X)\ge
\mathbb E_{q(Z\mid X,Y)}[\log p(Y\mid X,Z)]
-D_{KL}(q(Z\mid X,Y)\Vert p(Z\mid X)).
$$

Our inference: maximizing hindsight utility alone omits prior compatibility.
Conversely, copying each hindsight thought exactly is not necessary for a useful
latent marginal. Single-greedy-note utility and thought-augmented training value
are separate endpoints. The current partial-state experiment isolates the former;
it does not settle every latent-training hypothesis.

TPT trains on document followed by generated thinking using ordinary causal CE.
Its training benefit should not be gated by our inserted-note inference test.
[TPT section 2](https://arxiv.org/html/2509.20186v1).
Keep both hypotheses explicit without changing the current fixed comparison.

## Scale and document reasoning — rechecked 2026-10-19

TPT's mid-training comparison uses 100 billion tokens followed by 350,000-example SFT.
Its smaller 1.5B thought-generator ablation outperforms its 7B generator.
Neither result makes 7B a prerequisite for our experiment. Our roughly 4.8 million
supervised-token treatment tests a small, verified arithmetic augmentation;
it cannot reproduce or refute that larger training recipe.
[TPT sections3.3 and5](https://arxiv.org/html/2509.20186v1).

MuSiQue explicitly studies shortcuts through question-only, context-only and
single-paragraph controls. Its gold decomposition is training supervision, not
input available to a deployed generator. Our inference: a future document
reasoning study should test whether connected evidence is actually used, rather
than treating fluent fact extraction or benchmark accuracy alone as proof.
The current corpus inventory releases no learning or evaluation job.
[MuSiQue sections4 and7.2](https://arxiv.org/html/2108.00573v3).

## Trained extra-compute controls — rechecked 2026-10-19

The pause-token paper omits loss on supplied pause tokens. Its strongest gains
use delays during both pretraining and finetuning; adding them only at finetuning
has mixed effects, and removing delays at inference can break performance.
Our inference: untrained filler is insufficient for the stronger content-versus-
computation claim, and plain-path retention needs measurement. A later comparison
must match our causal attention: the paper uses bidirectional prefix attention
in downstream tasks. No pause arm is added to the current fixed study.
[Pause tokens, sections3–5](https://arxiv.org/html/2310.02226v3).

## Original question identities — checked 2026-10-19

[MuSiQue section5, S5](https://arxiv.org/html/2108.00573v3#S5) separates original
single-hop questions, answers and associated paragraphs to reduce memorization.
Our raw TRAIN component_question_texts sometimes contain a generic relation
program such as a location relation. Those strings are not original question
identities. V21 uses dataset plus original hop ID from decomposed_instances,
alongside supporting passage/page exclusions. Historical V16/V20 counts remain
as originally computed. Our held split permits repeated answers and distractors;
it is a TRAIN-derived development study, not the official benchmark split.

## QA-v21 interpretation — checked 2026-10-19

MuSiQue's step-execution model substitutes intermediate answers while executing
the decomposition. V21 instead supervised a single textual trace containing
literal `#n` placeholders. Explicit binding is a possible new treatment, not a
retroactive explanation established by our results. The paper's split excludes
shared original questions, answers and paragraphs; our within-TRAIN resplit
allows repeated answers and adds page/visible-context exclusions. These are
different protocols, not a replication of its official split.
[MuSiQue sections5 and7.2](https://arxiv.org/html/2108.00573v3).

TPT concatenates a raw document and generated thinking for ordinary next-token
training. Its reported mid-training uses100B tokens followed by350k-example SFT.
Our2,048-question post-answer supervision is a much smaller,different experiment;
its failed accuracy gate does not refute TPT.
[TPT sections2 and3.3](https://arxiv.org/html/2509.20186v1).
