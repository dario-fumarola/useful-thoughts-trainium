One shared process streams the installed `/opt/aws/neuron/bin/neuron-monitor`
through `/usr/bin/gzip -1 -c` onto NVMe for the full four-run schedule. Do not
start a node monitor per science arm. Qualification monitors remain unchanged.
These files are outside the active training source fingerprint; freeze helper
and launcher SHA256s separately in the site/launch configuration.

After training freeze, create one immutable config with:

```json
{
  "schema": "shared-node-monitor-v1",
  "frozen_experiment": "/absolute/path/to/frozen-experiment.json",
  "frozen_experiment_sha256": "ACTUAL_FILE_SHA256",
  "spec_hash": "ACTUAL_FROZEN_SPEC_HASH",
  "deadline_utc": "EXACT_FROZEN_DEADLINE",
  "helper_sha256": "FROZEN_HELPER_SHA256",
  "launcher_sha256": "FROZEN_LAUNCHER_SHA256",
  "nvme_root": "/home/ubuntu/dario/local/nvme",
  "output_dir": "/home/ubuntu/dario/local/nvme/science/shared-node-monitor"
}
```

Create the output parent beforehand; the monitor creates only a fresh output
directory. It verifies the mounted NVMe filesystem, frozen file/spec/deadline,
and both helper hashes. One fixed node lock is
`/home/ubuntu/dario/local/science/shared-node-monitor.lock`; a second science
leader fails without touching the valid capture. No release or live monitor is
created merely by preparing these files.

After checking the frozen launcher SHA, the finite command is:

```bash
bash validation/orchestration/launch_shared_node_monitor.sh CONFIG_JSON CONFIG_SHA256
```

The helper stops capture at the exact frozen deadline. The outer GNU timeout
allows30s for producer shutdown/gzip drain, then TERM and a final5s KILL bound
on the shared process group. Direct owned child PIDs are reaped on TERM/INT;
the producer stops first so gzip can consume EOF and write its trailer.
No trainer is started or stopped by this helper.

`launch.json` and `state.json` record config/spec/source/helper bindings, actual
supervisor/launcher-parent/producer/compressor PID/startticks/PPID/PGID/argv,
UTC and elapsed times, signals, forced kills and exit codes. Raw stdout is only
`node.jsonl.gz.partial` during capture. Only an orderly deadline stop plus
successful gzip close publishes `node.jsonl.gz` and `DEADLINE_COMPLETE` with
`complete=true`. Require that final state as well as the file. Early producer or
compressor exit, TERM/INT before deadline, forced cleanup and exceptions retain
the partial suffix and an incomplete/failure state. A SIGKILL can leave RUNNING
state and a partial file; it is never a complete receipt. If the science roster
finishes early, TERM retains a valid interrupted prefix when drain succeeds;
it does not claim capture through the later frozen deadline.

The host tests use fake producers and real gzip only. They cover deadline drain,
both early child failures, TERM/INT cleanup, and duplicate leadership. They are
host plumbing checks, not accelerator or scientific qualification evidence.
