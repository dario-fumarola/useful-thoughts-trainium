# Operations and restart procedure

## Scope and environment

Use the existing authorized instance. Do not create/terminate capacity, change
IAM/networking, or use Bedrock. The owner authorizes maintaining the public
`dario-fumarola/useful-thoughts-trainium` repository. Keep the known-good Neuron
Python environment. Public documentation may describe a different release; inspect
installed versions and source before adapting an API. Current XLA documentation
also notes a future support transition; that is not a reason to migrate a working
weekend experiment to a new backend simultaneously.

The first-run script prints missing helper dependencies. For an already-qualified
stack that only lacks pytest, installing that small helper is a different action
from upgrading Torch/Transformers. Prefer an isolated helper environment for data
acquisition where dependency conflicts exist. Never use an unrestricted
`pip install -U ...` to “fix” an accelerator import.

## Data and model acquisition

Use existing pinned local assets when available. Otherwise the explicit scripts:

```bash
python scripts/fetch_model.py --out local/assets/qwen3-1.7b-base
python scripts/fetch_corpus.py --max-documents 1000 --out local/data/plumbing_raw.jsonl
```

These are real network downloads when YOU run them, not actions performed during
packaging. The 1,000-document corpus is only for plumbing. Full exposure requires
at least `terminal_steps × global_unique_prefixes_per_update` fresh train records,
plus documents filtered out and held-out splits. Do not use the tiny slice as the
scientific dataset. The streamer has a bounded shuffle, not uniform whole-corpus
sampling. Review the dataset license and source structure.

The model helper resolves `main` once to a commit and downloads that commit. Both
arms reuse its manifest. No remote model code is executed. A local model snapshot
must include config, tokenizer, weights, and an accurate manifest.

Run exact/near-duplicate and benchmark exclusion audits before tokenization.
`scripts/audit_benchmark_overlap.py` is a conservative word n-gram screen, not a
complete near-duplicate or paraphrase decontaminator. The dataset's upstream
filtering is useful context, not a guarantee for our slice or pretrained weights.
Do not inspect final-test answers while tuning prompts or training choices.

```bash
python scripts/prepare_records.py local/data/audited_raw.jsonl \
  --tokenizer-dir local/assets/qwen3-1.7b-base \
  --tokenizer-identity 'Qwen/Qwen3-1.7B-Base@ACTUAL_COMMIT' \
  --source-revision ACTUAL_DATA_COMMIT \
  --out local/data/records.jsonl
python scripts/make_format.py --tokenizer-dir local/assets/qwen3-1.7b-base \
  --out local/resolved_format.json
```

`ACTUAL_COMMIT` values must be read from asset manifests, not literally copied.
A full production dataset is chosen before observing treatment outcomes. All
versions of a document stay in one split; each seed has one deterministic order
shared by C and R. Record filtering and truncation are part of the experiment.

## Resolve, qualify, freeze

Set paths and a finite timezone-aware planning deadline in `configs/site.local.json`.
The owner states weeks are available; the 14-day study schedule is not an instance
expiry. Do not infer an actual shutdown date from the planning deadline.
Resolve an approved backup destination early. Local disk is not durable if the
instance/storage disappears.

Measure complete steps using the implemented driver. The owner's next-day
guidance prospectively targets the first comparison at **256 updates**, with
step64 a fixed development milestone. Check measured timing, fresh records and
backup space before freezing; choose an explicit alternative only if needed.
The supplied budget utility estimates capacity but does not predict convergence.
Then freeze real assets and source, using the actual chosen step count:

```bash
python scripts/freeze_experiment.py --experiment-id mature-rlp-pair-001 \
  --steps 256 --out local/experiment.json
```

The 256 target is prospective and still requires measured qualification.
The freezer rejects wrong tokenizer IDs, insufficient fresh records, unresolved
assets, inconsistent global batches, and changed checksums. It hashes model files
once per freeze; budget that I/O rather than repeatedly hashing during each step.

Populate `local/qualification.json` from the pending template only after running
the named checks. Include command, measurements, evidence file/checksum, source
hash, and frozen-spec hash. There is no automatic pass from template copying.

```bash
python scripts/check_ready.py --frozen local/experiment.json \
  --receipt local/qualification.json
```

This checks evidence bindings and assets. It cannot certify the truth of a manually
written observation. Read actual results. Once code changes materially, requalify
and refresh the source/spec binding; do not reuse stale evidence.

## Long jobs and interruption

Launch the qualified driver in a named `tmux` session or an existing job manager
so an SSH disconnect does not kill the training process. The driver must support
a stop file and save at the next safe update boundary. It must not catch a stop
signal and exit without accounting for partially completed updates.

Record job/session name, PID, start time, run directory, visible cores, and deadline
in STATUS.md. Do not start a duplicate because the terminal looks quiet; check
owned processes and the instance's actual device report. Never use broad `pkill`,
`killall`, cloud termination, or an infinite relaunch loop. Stop only this job.

No command here claims that this chat session continues working in the background.
The actual instance job and Codex session are processes you start and supervise.

For multi-day science, use one node-wide monitor shared by both chains and
stream its output through gzip onto NVMe. Existing qualification monitors are
left intact. Their uncompressed node-wide output grows about 0.6 GiB/hour per
monitor, so duplicating it per science arm needlessly consumes scratch space.
Record the monitor/compressor PIDs, exit status and finite deadline with the
science launch, and preserve the separate raw driver logs.

## Checkpoints

Every rank writes its own optimizer/RNG state. Model, plan, source, data cursor,
format, and actual counters must be recoverable. Publish COMPLETE only after all
shards are finished and the real process barrier has been reached. Write to a new
checkpoint directory; do not overwrite the only known-good checkpoint.

`python scripts/verify_checkpoint.py <dir>` checks the packaged manifest format.
It does not verify numerical equivalence or support arbitrary legacy checkpoints.
Never deserialize an untrusted third-party pickle. The CPU loader uses
`weights_only=True` for its own tensor/simple-value format.

Before capacity ends, verify the latest full and data-matched checkpoints at the
approved durable destination. File name/size alone does not prove a successful
backup: verify content checksums and a small restore path. Do not invent an S3
bucket or reuse an archived bucket automatically. A local verified checkpoint is
still at risk if no durable destination is available; state that plainly.

## Compact end-of-session handoff

Update STATUS.md with: current stage; actual tests; active job; verified checkpoint;
remaining blocker; exact next command. Put one material decision in DECISIONS.md
only when needed. Evidence belongs in local logs, not a growing narrative. Keep
separate the terminal primary checkpoint and any development-selected result.

## Final deliverables

One result table per training seed: C/R direct and thought accuracy, three primary
contrasts, CE costs, local thought controls, failures/truncations, source/generated
work, and measured time. Include partial outcomes as partial. End with the next
experiment selected by the evidence, not another undirected list of models.
