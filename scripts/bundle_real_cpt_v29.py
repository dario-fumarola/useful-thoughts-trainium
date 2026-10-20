"""Package checked V29 numerical evidence without redistributing source windows.

This publication step changes neither scores nor scientific decisions. Original
rows remain local; the public projection omits source-bearing token ID arrays,
records their hashes, and retains every attempt, model output, and token score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def array_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def project(source, target, fields=None):
    """Keep all rows in order, with explicit hashes for omitted source arrays."""
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open() as incoming, target.open('x') as outgoing:
        for line in incoming:
            row = json.loads(line)
            if fields is not None:
                row = {key: value for key, value in row.items() if key in fields}
            omitted = {}
            for key in ('prompt_ids', 'context_ids', 'target_ids', 'prefix_ids',
                        'continuation16_ids', 'full_calculation_ids',
                        'numeric_surface_ids', 'closing_boundary_ids'):
                if key in row:
                    value = row.pop(key)
                    omitted[key] = dict(length=len(value), canonical_json_sha256=array_hash(value))
            row['omitted_source_arrays'] = omitted
            outgoing.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            count += 1
        outgoing.flush()
        os.fsync(outgoing.fileno())
    return dict(original_sha256=sha(source), projected_sha256=sha(target), rows=count)


def archive(root, destination, paths):
    expected = {str(path.relative_to(root)): sha(path) for path in paths}
    with tarfile.open(destination, 'x:gz', compresslevel=6) as stream:
        for path in paths:
            assert path.is_file() and not path.is_symlink()
            stream.add(path, arcname='v29-evidence/' + str(path.relative_to(root)), recursive=False)
    seen = set()
    with tarfile.open(destination, 'r:gz') as stream:
        for member in stream:
            name = member.name.removeprefix('v29-evidence/')
            assert member.isfile() and name in expected and name not in seen
            h = hashlib.sha256()
            content = stream.extractfile(member)
            for chunk in iter(lambda: content.read(8 * 2**20), b''):
                h.update(chunk)
            assert h.hexdigest() == expected[name]
            seen.add(name)
    assert seen == set(expected)
    with destination.open('rb') as stream:
        os.fsync(stream.fileno())
    return dict(file=destination.name, bytes=destination.stat().st_size,
                sha256=sha(destination), contents=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'readout', 'independent-check', 'scalar', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    index, report, checked = read(args.index), read(args.readout), read(args.independent_check)
    scalar = read(args.scalar)
    assert report['status'] == 'COMPLETE_V29_ALL_FIXED_MODEL_READOUT'
    assert checked['status'] == 'PASS_ALL_PAIRED_DOCUMENT_CONTRASTS_DECISIONS_AND_FIXED_HF_SPOTS'
    assert checked['report_sha256'] == sha(args.readout) and checked['index_sha256'] == sha(args.index)
    assert scalar['primary_decisions_unchanged'] == report['decision']
    assert scalar['pins'][str(args.readout.resolve())] == sha(args.readout)
    for path, digest in report['source_pins'].items():
        assert sha(path) == digest, path
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / 'projection'
    root.mkdir()
    records, portable = {}, dict(models={}, original_index_sha256=sha(args.index))
    for model, jobs in index['models'].items():
        portable['models'][model] = {}
        phases = [(panel, Path(jobs[panel])) for panel in ('writer', 'math', 'ordinary')]
        phases += [('scalar-' + str(n), Path(path)) for n, path in enumerate(jobs['scalar'])]
        for panel, source in phases:
            relative = Path('models') / model / panel
            target = root / relative
            target.mkdir(parents=True)
            filename = ('notes.jsonl' if panel == 'writer' else
                        'answers.jsonl' if panel.startswith('scalar-') else 'logps.jsonl')
            record = project(source / filename, target / filename)
            expected = 768 if panel == 'writer' else 576 if panel.startswith('scalar-') else 4608 if panel == 'math' else 4096
            assert record['rows'] == expected
            records[str(relative / filename)] = record
            portable['models'][model][panel] = str(relative / filename)
            for path in sorted(source.iterdir()):
                if path.name in ('notes-bundle.json', filename):
                    continue
                # These fixed numerical witnesses contain tensors/scores or
                # runtime metadata, not the original source input windows.
                assert path.suffix in ('.json', '.pt'), path
                shutil.copyfile(path, target / path.name)
        reference = Path(read(index['release'])['references'][model])
        target = root / 'references' / model
        target.mkdir(parents=True)
        for name in ('reference.json', 'CPU-native.pt'):
            shutil.copyfile(reference / name, target / name)
        write(target / 'omitted-TRAIN-inputs.json',
              dict(reason='Original real source windows retained locally.',
                   original_sha256=sha(reference / 'TRAIN-inputs.json')))
    data = Path(index['data'])
    fields = {'item_id', 'document_id', 'kind', 'split', 'full_calculation_text',
              'normalized_expression', 'source_file_sha256', 'source_record_sha256',
              'provenance', 'numeric_value', 'numeric_surface_text',
              'prefix_ids', 'continuation16_ids', 'full_calculation_ids',
              'numeric_surface_ids', 'closing_boundary_ids'}
    for name in ('DEV_calculation.jsonl', 'DEV_ordinary.jsonl'):
        records['cohorts/' + name] = project(data / name, root / 'cohorts' / name, fields)
    shutil.copyfile(data / 'manifest.json', root / 'cohorts' / 'manifest.json')
    results = root / 'results'
    results.mkdir()
    for source, name in ((args.readout, 'readout.json'), (args.independent_check, 'independent-check.json'),
                         (args.scalar, 'scalar-paired.json'), (args.index, 'original-index.json'),
                         (Path(index['release']), 'release.json')):
        shutil.copyfile(source, results / name)
    portable['cohorts'] = dict(calculation='cohorts/DEV_calculation.jsonl', ordinary='cohorts/DEV_ordinary.jsonl')
    write(root / 'index-portable.json', portable)
    write(root / 'projection-receipt.json', dict(
        status='COMPLETE_ALL_ORIGINAL_NUMERICAL_ROWS_PROJECTED_NO_ATTEMPT_FILTERING', files=records,
        array_hash_encoding='UTF-8 JSON, separators comma/colon, sorted object keys; no newline.',
        omissions='Original source context/target/prompt token arrays and TRAIN reference inputs remain local. Their hashes are preserved. Full generated outputs and token likelihood vectors are retained.',
        attribution='FineMath, HuggingFaceTB/SmolLM2 authors; pinned revision e92b25a616738fe95dc186b64dfb19f9c8525594. Dataset metadata specifies ODC-By-1.0 and Common Crawl terms. Original web text is not redistributed in this bundle.',
        source_sha256=sha(__file__)))
    files = sorted(path for path in root.rglob('*') if path.is_file())
    numerical = [path for path in files if path.name.endswith('.jsonl')]
    witnesses = [path for path in files if path.suffix == '.pt']
    metadata = [path for path in files if path not in numerical and path not in witnesses]
    archives = []
    for name, paths in (('v29-numerical-evidence.tar.gz', numerical),
                        ('v29-native-witnesses.tar.gz', witnesses),
                        ('v29-metadata-and-checks.tar.gz', metadata)):
        value = archive(root, args.output / name, paths)
        archives.append(value)
        print(json.dumps({k: v for k, v in value.items() if k != 'contents'}), flush=True)
    write(args.output / 'manifest.json', dict(status='PASS_ALL_ARCHIVES_FULL_MEMBER_READBACK_AND_FSYNC',
          readout_sha256=sha(args.readout), independent_check_sha256=sha(args.independent_check),
          source_sha256=sha(__file__), archives=archives))
    with (args.output / 'SHA256SUMS').open('x') as stream:
        for item in archives:
            stream.write(item['sha256'] + '  ' + item['file'] + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(args.output, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


if __name__ == '__main__':
    main()
