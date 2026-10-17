This host-only helper is outside the active training source fingerprint. Freeze
the Python helper hash, launcher hash, and exact roster hash separately in the
site/launch receipt. No real backup has been launched by preparing these files.
Bind helper/launcher hashes in the frozen site; after training freeze, bind the
roster containing that resulting training spec hash in the launch receipt.

The immutable roster has schema `shared-checkpoint-backup-v1`, one `spec_hash`,
one `terminal_step`, a timezone-aware `deadline_utc`, a pinned `source_root` and
`source_sha256`, absolute `lock_file`/`log_file`/`state_file` paths, and exactly
four `runs`. Each run contains `run_id`, `source`, `destination`, and `stop_file`.
Use the same one lock path for the whole four-run roster. Put event/raw logs on
NVMe and regular STOP/state/lock files in a separate control directory. Controls
must stay outside every driver output and backup directory.

After verifying the frozen launcher hash, the finite launch command is:

```bash
bash validation/orchestration/launch_shared_checkpoint_backups.sh \
  <frozen-python-helper-sha256> <immutable-roster.json> <frozen-roster-sha256>
```

The launcher independently enforces the UTC deadline with GNU `timeout`, TERM,
and 30-second kill grace. The supervisor checks the clock before/after every
complete existing `scripts.watch_checkpoint_backups.cycle()` call. Explicit TERM
or another leader-owned failure requests all regular STOP files and exits; it
preserves staging and never relaunches. A duplicate leader refuses to run without
stopping the valid leader. Invalid startup input never establishes a leader and
must block the science launcher; require a matching `RUNNING` state first.

Only one run is copied/verified/pruned at a time. Each cycle uses the existing
distinct-filesystem check, 20 GiB free-space reserve, hash/readback, incomplete-copy
recovery and two-copy retention. NVMe originals are never removed. Completed runs
are reverified once more before the shared supervisor reports `COMPLETE`. Missing
future source directories are observed without creating them. Deadline success
requires all four terminal checkpoints and two verified backups for every run.

Allow **two new checkpoint payloads plus the 20 GiB reserve** for a queued cycle.
On restart or backlog, the existing cycle can copy both unseen survivors before
pruning older copies. Shared serialization prevents concurrent cycles from adding
to that peak; it does not reduce the within-cycle backlog requirement. With
25.65 GiB payloads, this needs roughly 71.3 GiB free at rollover.
