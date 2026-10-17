#!/usr/bin/env bash
# Host-only; exact helper, launcher and roster hashes belong in the frozen launch receipt.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf '%s\n' 'Usage: launch_shared_checkpoint_backups.sh HELPER_SHA256 ROSTER_JSON ROSTER_SHA256' >&2
  exit 2
fi
backup_python=/home/ubuntu/ema-trainium/.venv/bin/python
backup_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backup_helper="$backup_directory/shared_checkpoint_backups.py"
backup_seconds=$("$backup_python" - "$backup_helper" "$1" "$2" "$3" <<'PY'
from datetime import datetime, timezone
import hashlib, importlib.util, math
from pathlib import Path
import sys
helper, helper_sha, roster, roster_sha = sys.argv[1:]
if hashlib.sha256(Path(helper).read_bytes()).hexdigest() != helper_sha:
    raise ValueError('Frozen shared backup helper changed')
spec = importlib.util.spec_from_file_location('shared_backups', helper)
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config = module.read_roster(roster, roster_sha)
deadline = datetime.fromisoformat(config['deadline_utc'].replace('Z', '+00:00'))
remaining = math.floor((deadline-datetime.now(timezone.utc)).total_seconds())
if remaining <= 0:
    raise TimeoutError('Shared backup launch deadline has expired')
print(remaining)
PY
)
exec timeout --signal=TERM --kill-after=30s "${backup_seconds}s" "$backup_python" -u "$backup_helper" \
  --roster "$2" --roster-sha256 "$3"
