"""Host-only cost accounting and fail-closed liveness for the fixed evaluation."""
import json
from pathlib import Path

MAX_CHIPS=8
MAX_CHIP_HOURS=18.
STOP_CHIP_HOURS=17.5
HEARTBEAT_SECONDS=60.


def reserved_hours(records,now):
    total=0.
    for value in records:
        start=value['started_monotonic'];end=value.get('closed_monotonic',now)
        if end<start:raise ValueError('Monotonic duration cannot be negative')
        total+=end-start
    return total/3600


def liveness(path,now,same_identity):
    path=Path(path)
    try:value=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    try:
        owner=value['owner']
        if not all(type(owner[k]) is int and owner[k]>0 for k in ('pid','start_ticks','pgid','session')):return False
        return bool(value.get('status')=='ACTIVE_FIXED_V28_EVALUATION'
            and value.get('maximum_chips')==MAX_CHIPS
            and value.get('maximum_reserved_chip_hours')==MAX_CHIP_HOURS
            and 0<=now-value.get('monotonic',-1)<=HEARTBEAT_SECONDS
            and value.get('reserved_chip_hours',MAX_CHIP_HOURS)<STOP_CHIP_HOURS
            and same_identity(owner))
    except (KeyError,TypeError,ValueError):return False
