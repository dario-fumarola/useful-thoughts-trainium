#!/usr/bin/env python3
"""Copy a complete scratch checkpoint to another local filesystem, with hashes."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.checkpoint_backup import copy_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--reserve-gib', type=float, default=20.)
    parser.add_argument('--allow-same-filesystem', action='store_true',
                        help='Diagnostic copies only; scratch backup normally requires another filesystem.')
    args = parser.parse_args()
    result = copy_checkpoint(args.source, args.destination,
                             reserve_bytes=int(args.reserve_gib * 1024**3),
                             require_distinct_filesystems=not args.allow_same_filesystem)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
