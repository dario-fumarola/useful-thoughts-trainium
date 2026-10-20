"""Fetch one pinned public TRAIN shard for bounded CPU preparation only.

No source text is selected or parsed here. No model call, training, benchmark,
official TEST, package installation or capacity change is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

REVISION = 'e92b25a616738fe95dc186b64dfb19f9c8525594'
DATASET = 'HuggingFaceTB/finemath'
FILENAME = 'finemath-4plus/train-00061-of-00064.parquet'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = args.repo_root / 'local/data/sources/HuggingFaceTB--finemath/SOURCE.json'
    pinned = json.loads(metadata.read_text())
    assert pinned['resolved_revision'] == REVISION
    original = next(row for row in pinned['files'] if row['path'] == FILENAME)
    assert original['bytes'] < 512 * 2**20
    assert shutil.disk_usage(args.output.parent).free > 35 * 2**30 + original['bytes']
    args.output.mkdir(exist_ok=False)
    write(args.output / 'download-plan.json', dict(
        status='FROZEN_PINNED_TRAIN_DOWNLOAD_CPU_PREPARATION_ONLY',
        utc=datetime.now(timezone.utc).isoformat(), dataset=DATASET,
        revision=REVISION, filename=FILENAME, split='train',
        expected_bytes=original['bytes'], source_inventory_sha256=sha(metadata),
        script_sha256=sha(__file__),
        selection='Next unused descending shard after prior source shards63 and62; no model outcomes used.',
        proposed_CPU_bounds=dict(first_parquet_rows=65536, maximum_raw_characters=65536,
                                 maximum_native_tokens=8208, minimum_binary_operations=2),
        limits='Download only. Retained-source page/content and exact-expression exclusions, native-label checks and a frozen diagnostic design precede any model call.'))
    base = 'https://huggingface.co/datasets/' + DATASET
    api = 'https://huggingface.co/api/datasets/' + DATASET + '/tree/' + REVISION + '/finemath-4plus?recursive=false&limit=1000'
    with urllib.request.urlopen(api, timeout=60) as response:
        entries = json.load(response)
    entry = next(row for row in entries if row['path'] == FILENAME)
    assert entry['size'] == original['bytes']
    expected = entry['lfs']['oid']
    assert re.fullmatch('[0-9a-f]{64}', expected)
    write(args.output / 'public-file-metadata.json', entry)
    partial = args.output / (Path(FILENAME).name + '.download-part')
    started = time.monotonic(); digest = hashlib.sha256(); size = 0
    request = urllib.request.Request(base + '/resolve/' + REVISION + '/' + FILENAME,
                                    headers={'User-Agent': 'useful-thoughts-trainium-source-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as stream:
        for chunk in iter(lambda: response.read(8 * 2**20), b''):
            assert time.monotonic() - started < 600, 'Finite download deadline; partial retained.'
            size += len(chunk); assert size <= original['bytes']
            digest.update(chunk); stream.write(chunk)
        stream.flush(); os.fsync(stream.fileno())
    assert size == original['bytes'] and digest.hexdigest() == expected
    assert sha(partial) == expected
    target = args.output / Path(FILENAME).name
    partial.rename(target)
    result = dict(status='PASS_PINNED_PUBLIC_TRAIN_SHARD_BYTES_LFS_SHA_READBACK_FSYNC',
        path=str(target.resolve()), bytes=size, sha256=expected, dataset=DATASET,
        revision=REVISION, filename=FILENAME, split='train',
        public_metadata_sha256=sha(args.output / 'public-file-metadata.json'),
        script_sha256=sha(__file__),
        limits='No rows decoded, no model/accelerator/officialTEST calls. CPU filtering and qualification still pending.')
    write(args.output / 'download-complete.json', result)
    fd = os.open(args.output, os.O_RDONLY); os.fsync(fd); os.close(fd)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
