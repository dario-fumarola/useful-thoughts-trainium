# Checkpoint protection

The current studies retain every original full checkpoint on NVMe. EBS under
`local/backups/science/` protects every scientific terminal model and the latest
complete optimizer checkpoint across the studies. Evaluation originals, reports,
examples and analysis sources have separate readback-protected archives.
Both storage locations belong to the same instance; this is not an off-instance
backup. Git contains source and small evidence, not full model/optimizer payloads.

The current EBS full recovery state is V17 source-direct seed1913 at update320:
`local/backups/science/real-source-balanced-v17-study-v1/TERMINALS/latest-full320`.
All four V17 terminal models and this full optimizer state passed independent
audit and protection; [completed training evidence](../validation/real_source_balanced_v17_training_20261019.json)
contains the receipts.
See [live status](../STATUS.md) for the current handoff.

Protection follows this order:

1. Independently inspect the original complete scientific states, including
   payload hashes, optimizer tensors, counters, lineage and active/master casts.
2. Copy the newest full terminal state to EBS first. Read back every payload,
   verify its size/hash, and flush files and directory metadata.
3. Only after that succeeds, freshly verify the older full state on both NVMe
   and EBS. Retire only its inactive EBS optimizer duplicates, with a recorded
   intent and completion receipt. Keep its model and every original file.
4. Protect the remaining terminal models. A model shared with the latest full
   state may use a hard link within EBS; it is not a second independent copy.

Keep at least 35 GiB free on each filesystem throughout the procedure. An error
leaves the originals and last verified protection intact; failed attempts retain
all evidence and are reviewed before any separately identified repair.

An older EBS directory whose optimizer duplicates were retired contains
`OPTIMIZER_DUPLICATES_RETIRED.json`. Its original `COMPLETE.json` and archive
receipt describe the historical full copy. The sidecar identifies the unchanged
NVMe recovery source and the newer protected full state. Do not treat that older
EBS directory as a currently complete optimizer backup.

The V17 procedure is preserved with its
[hardware and release evidence](../validation/real_source_balanced_v17_hardware_20261019.json).
Hash verification establishes byte integrity. Actual optimizer restoration is a
separate hardware check; the qualified V17 save/restart evidence is linked there.
The original 256-step copy/watch scripts remain historical tools and do not define
the active study's checkpoint schedule or retention policy.
