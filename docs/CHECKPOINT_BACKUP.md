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
