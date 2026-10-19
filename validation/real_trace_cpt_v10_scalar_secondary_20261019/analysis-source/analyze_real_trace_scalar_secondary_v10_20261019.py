"""Preserve and recheck all fixed scalar continuations before secondary readout."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import os
import shutil
from collections import Counter

from train.real_trace_answers import classify, MODES, sources
from weekend_thinking.artifacts import sha256_file
from weekend_thinking.stats import paired_cluster_bootstrap

R = Path('/home/ubuntu/dario')
S = R/'local/diagnostics/real-trace-v10-scalar-secondary-study-v1'
registry = json.loads((S/'registry.json').read_text())
assert registry['status'] == 'ALL_SEVEN_FIXED_SCALAR_MODELS_RELEASED'
assert len(registry['models']) == 7
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).open()]
for model in registry['models'].values():
    assert read(Path(model['control'])/'completion.json')['status'] == 'COMPLETE_BOTH_FIXED_REAL_SCALAR_PANELS'

A = R/'local/backups/evaluation/real-trace-v10-scalar-secondary/COMPLETE'
A.mkdir(exist_ok=False)
public = R/'validation/real_trace_cpt_v10_scalar_secondary_20261019'
public.mkdir(exist_ok=False)
archive_files = {}
def protect(source, target):
    shutil.copytree(source, target)
    for file in source.rglob('*'):
        if file.is_file():
            copied = target/file.relative_to(source)
            digest = sha256_file(file)
            assert sha256_file(copied) == digest
            with copied.open('rb') as stream:
                os.fsync(stream.fileno())
            archive_files[str(copied.relative_to(A))] = dict(bytes=copied.stat().st_size, sha256=digest)

protect(S, A/'study-control')
protect(R/'local/nvme-cpt/data/real-trace-v10-scalar-secondary-v1', A/'scoring-only-gold')
cells, tables, payloads, source_receipts = {}, {}, {}, {}
for key, model in registry['models'].items():
    control = Path(model['control'])
    plan = read(control/'plan.json')
    protect(control, A/key/'control')
    cells[key], tables[key] = {}, {}
    for step in plan['steps']:
        output, panel = Path(step['output']), step['panel']
        protocol, summary = read(output/'protocol.json'), read(output/'summary.json')
        assert protocol['source_files'] == sources()
        assert summary['status'] == 'COMPLETE_ALL_FIXED_SCALAR_CONTINUATIONS'
        assert summary['rows'] == 1280 and summary['panel'] == panel
        assert summary['protocol_sha256'] == sha256_file(output/'protocol.json')
        assert sha256_file(output/'scored.jsonl') == summary['scored_sha256']
        assert summary['model_id'] == ('B0' if key == 'B0' else model['model_id'])
        assert sha256_file(step['gold']) == protocol['gold_sha256']
        source_receipts[key+'/'+panel] = dict(model_binding=protocol['model'], protocol_sha256=sha256_file(output/'protocol.json'), summary_sha256=sha256_file(output/'summary.json'), gold_sha256=protocol['gold_sha256'], original_scored_sha256=summary['scored_sha256'])
        gold = {row['item_id']: row['numeric_value'] for row in rows(step['gold'])}
        values = rows(output/'scored.jsonl')
        assert len(values) == 1280
        for value in values:
            check = classify(value, gold[value['item_id']])
            assert all(value[name] == field for name, field in check.items())
        cells[key][panel], tables[key][panel] = {}, {}
        for mode in MODES:
            selected = [v for v in values if v['mode'] == mode]
            assert len(selected) == len({v['document_id'] for v in selected}) == 256
            assert len({v['item_id'] for v in selected}) == 256
            cell = dict(documents=256, correct=sum(v['correct'] for v in selected),
                        categories=dict(Counter(v['category'] for v in selected)),
                        capped=sum(v['stop_type'] == 'length_cap' for v in selected),
                        generated_actions=sum(v['n_actions'] for v in selected),
                        wall_seconds_sum=sum(v['wall_seconds'] for v in selected))
            assert cell == summary['modes'][mode]
            tables[key][panel][mode] = cell
            cells[key][panel][mode] = [dict(item_id=v['item_id'], document_id=v['document_id'], score=int(v['correct'])) for v in selected]
        # All outputs/failures are public; the repeated source prompt IDs stay in the full archive.
        visible = [{**{k: v for k, v in value.items() if k != 'prompt_ids'},
                    'expected_numeric_value': gold[value['item_id']]} for value in values]
        file = public/f'{key}-{panel}-all1280-scalar-continuations.jsonl.gz'
        file.write_bytes(gzip.compress((''.join(json.dumps(v, separators=(',', ':'))+'\n' for v in visible)).encode(), mtime=0))
        payloads[str(file.relative_to(R))] = dict(rows=1280, sha256=sha256_file(file))
        protect(output, A/key/panel)

def compare(panel, reference, treatment, amode='raw', bmode='raw'):
    return paired_cluster_bootstrap(cells[reference][panel][amode], cells[treatment][panel][bmode],
                                    repeats=20000, seed=2026101915)

contrasts = {}
for seed in (2026101913, 2026101914):
    raw, prior, post = (f'{arm}-seed{seed}' for arm in ('real_raw', 'real_prior', 'real_post'))
    contrasts[str(seed)] = {}
    for panel in ('DEV_AST256', 'DEV_legacy256'):
        contrasts[str(seed)][panel] = dict(
            post_raw_minus_raw_raw=compare(panel, raw, post),
            prior_own_minus_raw_raw=compare(panel, raw, prior, bmode='own'),
            prior_own_minus_prior_raw=compare(panel, prior, prior, bmode='own'),
            prior_own_minus_prior_empty=compare(panel, prior, prior, amode='empty', bmode='own'),
            prior_own_minus_prior_other=compare(panel, prior, prior, amode='other', bmode='own'),
            prior_own_minus_raw_own=compare(panel, raw, prior, amode='own', bmode='own'),
            post_raw_minus_B0raw=compare(panel, 'B0', post),
            raw_raw_minus_B0raw=compare(panel, 'B0', raw),
            prior_own_minus_B0raw=compare(panel, 'B0', prior, bmode='own'),
            prior_own_minus_B0own=compare(panel, 'B0', prior, amode='own', bmode='own'))

for name in ('analyze_real_trace_scalar_secondary_v10_20261019.py',
             'prepare_real_trace_scalar_secondary_v10_20261019.py',
             'watch_real_trace_scalar_secondary_v10_20261019.py',
             'check_real_trace_scalar_readout_v10_20261019.py',
             'render_real_trace_scalar_secondary_v10_20261019.py'):
    source, target = R/'local/analysis'/name, A/'analysis-source'/name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256_file(source) == sha256_file(target)
    with target.open('rb') as stream:
        os.fsync(stream.fileno())
    archive_files[str(target.relative_to(A))] = dict(bytes=target.stat().st_size, sha256=sha256_file(target))

archive = dict(status='ALL_FIXED_SCALAR_ORIGINALS_READBACK_VERIFIED_AND_FSYNCED', files=archive_files,
               utc=datetime.now(timezone.utc).isoformat())
with (A/'ARCHIVE.json').open('x') as stream:
    json.dump(archive, stream, indent=2)
    stream.flush()
    os.fsync(stream.fileno())
for directory in [p for p in A.rglob('*') if p.is_dir()]+[A]:
    fd = os.open(directory, os.O_RDONLY|os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)

report = dict(status='COMPLETE_ALL_SEVEN_MODELS_BOTH_FIXED_SCALAR_PANELS',
              utc=datetime.now(timezone.utc).isoformat(), scalar_continuations=17920,
              tables=tables, contrasts=contrasts, payloads=payloads, source_receipts=source_receipts,
              scoring_gold_manifest=read(R/'local/nvme-cpt/data/real-trace-v10-scalar-secondary-v1/manifest.json'),
              design_sha256=sha256_file(R/'validation/real_trace_cpt_v10_scalar_secondary_design_20261019.json'),
              source_files=sources(), archive_sha256=sha256_file(A/'ARCHIVE.json'),
              limits=['Secondary initial-scalar accuracy at curated real-document boundaries, not general QA.',
                      'All256 documents and all errors/unsupported outputs remain in every denominator.',
                      'Existing initial-scalar-v2 unchanged; no searching, answer repair, arithmetic evaluation or gold-dependent stopping.',
                      'Each seed/panel is separate; paired document intervals condition on checkpoints, not training-seed uncertainty. B0 is shared once.',
                      'Original notes and references reused exactly, not new independent observations. Supplied full traces include the result.',
                      'Own/other use prior generated notes plus32 continuation actions. Raw/empty/oracle use32 continuation actions. Not equal inference compute or controlled service latency.',
                      'The primary full16 likelihood gates are unchanged. Same-instance EBS protection is not off-instance backup.'])
assert sum(v['rows'] for v in payloads.values()) == 17920 and len(payloads) == 14
with (R/'validation/real_trace_cpt_v10_scalar_secondary_20261019.json').open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(dict(status=report['status'], tables=tables), indent=2), flush=True)
