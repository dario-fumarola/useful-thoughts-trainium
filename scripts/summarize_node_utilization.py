"""Summarize a frozen read-only telemetry window without filling missing samples."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def summarize(rows, start, end, maximum_sample_seconds=10.0):
    if not start < end or maximum_sample_seconds <= 0:
        raise ValueError('A positive observation window and sample limit are required')
    times = [datetime.fromisoformat(row['utc']).timestamp() for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Telemetry timestamps must be strictly increasing')
    covered = weighted_busy = occupied_chip_seconds = 0.0
    minimum_memory = math.inf
    intervals = []
    for i, (row, timestamp) in enumerate(zip(rows, times)):
        next_time = times[i+1] if i+1 < len(times) else end
        first = max(start, timestamp)
        last = min(end, next_time, timestamp+maximum_sample_seconds)
        if last <= first:
            continue
        chips = row['total_physical_chips']
        occupied = row['occupied_physical_chips']
        busy = row['node_mean_core_busy_percent']
        if (chips != 16 or row['logical_cores'] != 64
                or len(occupied) != len(set(occupied))
                or any(not isinstance(v, int) or not 0 <= v < chips for v in occupied)
                or not math.isfinite(busy) or not 0 <= busy <= 100
                or not math.isfinite(row['available_host_GiB'])):
            raise ValueError('Unexpected topology or invalid telemetry')
        seconds = last-first
        covered += seconds
        weighted_busy += seconds*busy
        occupied_chip_seconds += seconds*len(occupied)
        minimum_memory = min(minimum_memory, row['available_host_GiB'])
        intervals.append(dict(utc=row['utc'], covered_seconds=seconds,
                              occupied_chips=len(occupied), busy_percent=busy,
                              available_host_GiB=row['available_host_GiB']))
    if covered == 0:
        raise ValueError('No valid samples cover the requested window')
    window = end-start
    return dict(
        window_seconds=window, observed_seconds=covered,
        unobserved_seconds=window-covered, observation_coverage=covered/window,
        observed_time_weighted_node_busy_percent=weighted_busy/covered,
        observed_time_weighted_occupied_chips=occupied_chip_seconds/covered,
        observed_process_occupied_chip_hours=occupied_chip_seconds/3600,
        maximum_observed_process_occupied_chips=max(v['occupied_chips'] for v in intervals),
        minimum_observed_available_host_GiB=minimum_memory,
        maximum_sample_carry_seconds=maximum_sample_seconds,
        scope=('Read-only hardware telemetry over covered intervals only. Missing '
               'intervals remain unknown. Process inventory refreshes at most once '
               'per minute and can lag launches/exits. Busy percentage uses all 64 '
               'logical cores, including idle cores; it is not FLOP efficiency. '
               'Observed process occupancy is distinct from assigned experiment '
               'chip-hours and the whole-node invoice.'),
        intervals=intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--from-utc', required=True)
    parser.add_argument('--to-utc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.telemetry.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('Use a frozen complete telemetry snapshot')
    rows = [json.loads(line) for line in raw.splitlines()]
    first, last = (datetime.fromisoformat(v).timestamp()
                   for v in (args.from_utc, args.to_utc))
    result = summarize(rows, first, last)
    result.update(status='OBSERVED_NODE_UTILIZATION_SUMMARY',
                  telemetry_sha256=hashlib.sha256(raw).hexdigest(),
                  from_utc=args.from_utc, to_utc=args.to_utc,
                  generated_utc=datetime.now(timezone.utc).isoformat())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'intervals'}, indent=2))


if __name__ == '__main__':
    main()
