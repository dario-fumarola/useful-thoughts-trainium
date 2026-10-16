# Local checkpoint backup

Large intermediates live on the instance SSD mounted at `local/nvme`. Copy each
scientific terminal checkpoint and the daily latest complete checkpoint to EBS
under `local/backups`. Keep the source until the copy is verified. This is local
redundancy; both locations are still tied to this instance.

```bash
python scripts/backup_checkpoint.py \
  local/nvme/runs/<run>/step-0000256 \
  --destination local/backups/<run>/step-0000256
```

The command requires a `neuron-v1` COMPLETE manifest and another filesystem by
default. It verifies file names, sizes, and hashes, copies in bounded chunks,
flushes files, reads the copied data back, and publishes the complete destination
last. It refuses changed source files and preserves existing backups. The
default free-space reserve is 20 GiB. Nothing is uploaded.

An interrupted attempt leaves `.<step>.incomplete` beside the destination. Repeat
the same command to reuse verified whole files; partial files are copied again.
Existing complete destinations are verified and left intact. Hash verification
establishes byte integrity, not numerical optimizer-resume correctness; that is a
separate hardware qualification.

For a multi-day run, `scripts/watch_checkpoint_backups.py` supervises this copy
process. Give it the exact source/run, destination, frozen spec hash, terminal
step, deadline, and the same stop-file used by the training driver. Record the
supervisor PID and arguments in the launch receipt and keep its log monitored.
The driver saves complete states every 12 hours at update boundaries and at its
milestones; the supervisor polls every 30 seconds.

The supervisor keeps the newest two verified EBS copies available for each run,
including its terminal state. It backfills both if started late. Before deleting
a redundant EBS copy, it verifies both survivors and the redundant copy's original
source files. It removes only its own marked directories and rejects unknown
files. All source checkpoints remain on scratch. On copy/verification failure,
deadline expiry, or a termination signal, it requests the driver's stop-file and
exits; it does not automatically retry a failed operation. A restart re-verifies
the retained copies. Both storage locations remain instance-local.
