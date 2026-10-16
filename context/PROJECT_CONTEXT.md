# Project context for a new coding session

## The question and the owner's goal

Dario and Emanuele/Manuel are investigating useful self-generated intermediate
computation during language-model pretraining. In the original setup a writer
samples a short thought from a prefix; a reader predicts the actual continuation;
training rewards thoughts that help that prediction. Writer and reader commonly
share parameters. The end goal is not merely fluent explanations.

The staged ambition is: establish useful computation, determine how early and
with how little supervision it can develop, improve its efficiency, then examine
social understanding and safety. A mature-model experiment is an enabling
reference condition, not proof of emergence from scratch.

The October 13 call explicitly accepts a modest NTP cost if a real downstream
capability is gained. It questions the old practice of refusing to run downstream
benchmarks until a local thought-likelihood probe became positive. Keep that
correction. Neither generic CE nor a nice-looking thought is the sole success
criterion.

Source: project-only Italian transcript excerpt in this directory. It preserves
original wording and source physical line numbers; exploratory ideas and claimed
hardware specifications in the call are not independently verified facts.

## What the previous program did

The legacy project used a Qwen3-0.6B architecture trained from scratch, with
important later experiments starting after roughly 9B vanilla tokens. Calling
that checkpoint “untrained” usually meant before thought-specific updates, not
random initialization. Models and local paths in archival notes are historical.

The implementation evolved through many objectives, geometries, teacher/hint
conditions, gates, and continuations. “98 runs” is a research history, not 98
independent controlled replications. The current repository manuscript reaches
run 98; the separately uploaded manuscript represents an earlier period.

The old code and manuscript disclose a gradient-synchronization boundary:
runs 13–74 did not perform the intended global update. Each saved replica saw a
much smaller effective batch/exposure than nominal accounting suggested. Later
runs added synchronization but used a changed design family. This qualifies
training conclusions; it does not erase every measurement made on those weights.

Several positive signals deserve preservation: content/order contrasts, synthetic
reader capabilities, and especially run 95's early oracle-selectable tail. They
have not established a net benefit under the ordinary generated-thought policy.
The recent terminal local audit still loses to silence. See the evidence ledger
for units and limitations; do not turn this summary into a new empirical result.

## Run 95 is informative, but the old narrative overclaimed

Captured fixed-anchor output compares 382 anchors across 48 sequences. The
oracle best-of-six with a silence option rises from approximately 2.449 to 5.857
nats per sequence, a paired increase of 3.408 ± 0.552. The ordinary first-sample
policy remains negative. These are recorded historical outputs, not rerun here.

The fixed-anchor script scores K=1, while the run trained a different horizon.
Both writer and reader changed, and the oracle sees the true continuation.
Run 95 trained using 128 samples; six is this evaluation's sample count. The
claimed +0.30 contribution from ordinary pretraining compared probes of the same
base checkpoint, not a valid before/after pretraining intervention.

A crossed writer/reader analysis at the trained horizon remains a useful bounded
follow-up if the weights survived. Recovering it is not a prerequisite to a fresh
mature-model comparison.

## What we found in implementation

1. `model_init.build_model()` explicitly builds from config and initializes new
   random weights. Changing its model ID cannot load a mature base.
2. A final clean-token label in the OGeo-based late trainer family was left as
   zero while receiving NTP weight. The kit repairs that one target and relevant
   callers; the original geometry stays unchanged. This does not quantify its
   contribution to historical degradation or certify the whole trainer.
3. Gradient reduction must have exactly one owner. Historical missing reduction
   and possible double reduction in a port are separate failure modes.
4. BF16 compute is not a license for rounded-away updates. Check actual parameter
   changes and preserve a qualified FP32 master/update path.
5. Cached generation, reward scoring, and gradient replay must use the same
   weights, positions, temperature, token mask, and stopping semantics.

## Why the weekend plan changed the starting condition

Use released Qwen3-1.7B-Base, where related mature-model work provides a favorable
reference. This changes size and maturity together; it is not a controlled token-age
experiment. The 0.6B released-base fallback is a separate identity. Later, hold
architecture and pretraining history controlled when comparing checkpoint ages.

The proposed thought cap is now generous enough not to require discovering and
compressing useful computation at the same time. Thought length L, reward horizon
K, and offset before the scored future are separate variables. Do not conflate
lengthening a thought with extending its prediction horizon.

The current recipe is deliberately minimal: four independent prefix-only samples,
up to 1,024 actions, K=1 reward, signed leave-one-out advantages, one fresh update,
no best-of-only training, no joint NTP/reader/hint/gate auxiliary. It is an explicit
engineering proposal, not a reproduced author implementation or a guarantee.

## The four distinct outcomes

- Training benefit: R-direct versus matched C-direct.
- Inference benefit: R-thought versus the same R-direct weights.
- Same-policy deployment benefit: R-thought versus C-thought. This is primary.
- Efficiency: quality under matched measured resources; equal raw documents alone
  do not establish equal computation.

B0 is the unchanged base reference. C and R share each seed's data order; the
second seed uses another fixed order. Terminal checkpoints are primary. Better
oracle-selected candidates do not count as deployment unless a prefix-available
selection policy actually recovers that benefit.

## Social and safety ideas are recorded, not added to this run

Social BDI thoughts should distinguish reality, what an agent knows/believes,
and its goals, then improve a downstream prediction. A clean intervention changes
who saw an event while holding the rest fixed. Compare against an ordinary summary;
do not reward unsupported confidence about hidden intentions.

Safety needs an explicit policy and appropriate target behavior. Recognizing
harmful text is not the same as choosing a safe response. Training on an unsafe
NTP target still rewards that target unless the objective changes. A “tax” names
a tradeoff, not a learning mechanism. Hindsight judgment, prefix-only risk
prediction, and checking the model's own draft are distinct setups.

The future roadmap specifies response-only, verdict, and reasoning comparisons,
benign near-neighbors/overrefusal, and causal rather than cosmetic evidence. No
weekend result justifies “safe from first principles.”

## Operational facts and unknowns

The user has a large Trainium instance and a weekend allocation. The call discussed
an October 16, 2026 capacity start, but exact instance identity and cutoff require
local verification. A `48xlarge` suffix is not a chip count. Do not copy old core
allocations blindly. Check the actual runtime and LNC layout.

Bedrock is explicitly excluded. Do not use a paid teacher API or provision new
capacity without a new user instruction. The call leaves old checkpoint backup
completeness uncertain. Discover actual disks/manifests and an approved durable
backup destination; do not infer survival from a path in a log.

The research process previously spent too much effort on expanding records and
stopping rules. This repo intentionally uses one short status, one decision log,
and a finite qualification checklist. The objective is a decisive comparison,
not a larger audit archive. Preserve evidence without letting it dominate work.
