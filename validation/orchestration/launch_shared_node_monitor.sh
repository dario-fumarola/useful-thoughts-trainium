#!/usr/bin/env bash
# One shared node stream; no model/device allocation and no per-arm monitor.
set -euo pipefail
if [[ $# != 2 ]]; then
  printf '%s\n' 'Usage: launch_shared_node_monitor.sh CONFIG_JSON CONFIG_SHA256' >&2
  exit 2
fi
monitor_python=/home/ubuntu/ema-trainium/.venv/bin/python
monitor_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
monitor_helper="$monitor_directory/shared_node_monitor.py"
monitor_seconds=$("$monitor_python" - "$monitor_helper" "$0" "$1" "$2" <<'PY'
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
from pathlib import Path
import sys
helper,launcher,config,expected=sys.argv[1:]
raw=Path(config).read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Frozen monitor config changed')
settings=json.loads(raw)
for path,key in [(helper,'helper_sha256'),(launcher,'launcher_sha256')]:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=settings[key]:raise ValueError('Frozen monitor helper/launcher changed')
spec=importlib.util.spec_from_file_location('shared_node_monitor',helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
settings=module.read_config(config,expected)
remaining=math.ceil((module.utc_deadline(settings['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
if remaining<=0:raise TimeoutError('Frozen node-monitor deadline expired')
print(remaining)
PY
)
# The helper stops capture at the exact frozen deadline. Reserve30s to stop the
# producer and drain gzip before the outer whole-group TERM/+5s KILL fallback.
monitor_bound=$((monitor_seconds + 30))
exec timeout --signal=TERM --kill-after=5s "${monitor_bound}s" "$monitor_python" -u "$monitor_helper" \
  --config "$1" --config-sha256 "$2"
