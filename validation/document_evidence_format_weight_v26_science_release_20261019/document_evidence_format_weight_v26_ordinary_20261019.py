"""Fixed ordinary256 cost of V26; unchanged qualified scorer, no QA reward claims."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from train import document_evidence_plan_likelihood as engine
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/document-evidence-span-v23-io-v2'
C = R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
CONTROL = C/'ORDINARY'
O = R/'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-ordinary256-v1'
OLD_C = R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read = engine.read
rows = engine.rows


def sources():
    return {**engine.sources(), str(Path(__file__).resolve().relative_to(R)): sha256_file(__file__)}


def old_groups():
    registry = read(OLD_C/'likelihood-registry-v2.json')['groups']
    assert len(registry) == 2
    return {v['arm']: Path(v['output']) for v in registry}


def prepare():
    import torch
    from train.document_qa_span_scores import score_spans
    from train.evaluate_local import setup_model
    assert engine.ROOT == W
    contract = read(R/'validation/document_evidence_format_weight_v26_ordinary_readiness_20261019.json')
    assert contract['status'] == 'PASS_CPU_FIXED_ORDINARY256_INPUTS_AND_REUSED_BASELINE_IDENTITIES'
    assert contract['sources'] == sources()
    assert contract['controller_sha256'] == sha256_file(R/'local/analysis/supervise_document_evidence_format_weight_v26_ordinary_20261019.py')
    review_file = C/'evaluation-release.json'
    review = read(review_file)
    assert review['status'] == 'RELEASED_FORMAT_WEIGHT_V26_TERMINAL_FIXED_EVALUATION'
    binding = C/'frozen-models/evidence_direct-format1-seed2026101923.json'
    model = read(binding)
    assert model['status'] == 'AUDITED_EVIDENCE_FORMAT_WEIGHT_V26_TERMINAL256'
    assert model['checkpoint_sha256'] == review['model_checkpoint_sha256'] == sha256_file(model['checkpoint'])
    assert sha256_file(model['audit_file']) == model['audit_sha256']
    assert sha256_file(model['protection_file']) == model['protection_sha256']
    original = old_groups()['evidence_select']
    p = read(original/'protocol.json')
    assert p['source_files'] == engine.sources()
    assert sha256_file(original/'ordinary-raw.jsonl') == p['files']['ordinary-raw.jsonl']
    examples = rows(original/'ordinary-raw.jsonl')
    assert len(examples) == len({r['item_id'] for r in examples}) == 256
    assert all(r['panel'] == 'ordinary' and r['mode'] == 'raw' and len(r['context_ids']) == 256 and len(r['target_ids']) == 32 for r in examples)
    assert sha256_file(original/'calibration-inputs.json') == p['files']['calibration-inputs.json']
    calibration = {'ordinary': read(original/'calibration-inputs.json')['ordinary']}
    assert len(calibration['ordinary']) == 4
    torch.set_num_threads(16)
    backend, _, loaded, signature, _ = setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),
        checkpoint=Path(model['checkpoint']), device='cpu'))
    assert all(loaded[k] == model[k] for k in ('run_id', 'step', 'spec_hash'))
    values = score_spans(backend, [r['context_ids'] for r in calibration['ordinary']],
        [r['target_ids'] for r in calibration['ordinary']], width=672)
    assert len(values) == 4 and all(math.isfinite(v['full_logp']) for v in values)
    del backend
    O.mkdir(parents=True, exist_ok=False)
    CONTROL.mkdir(exist_ok=False)
    os.link(original/'ordinary-raw.jsonl', O/'ordinary-raw.jsonl')
    atomic_json(O/'calibration-inputs.json', calibration)
    atomic_json(O/'calibration-CPU-reference.json', dict(signature=signature, values={'ordinary': values}))
    release = dict(status='RELEASED_FIXED_V26_ORDINARY256_CURRENT_MODEL_ONLY', utc=utc_now(),
        parent_review_sha256=sha256_file(review_file), model_binding_sha256=sha256_file(binding),
        source_files=sources(), controller_sha256=sha256_file(R/'local/analysis/supervise_document_evidence_format_weight_v26_ordinary_20261019.py'),
        old_control_protocol_sha256=sha256_file(original/'protocol.json'),
        input_sha256=p['files']['ordinary-raw.jsonl'],
        interpretation='Same already-inspected ordinary256 continuation cost, paired pages. No answer or reasoning outcome gate.')
    atomic_json(CONTROL/'release.json', release)
    files = {name: sha256_file(O/name) for name in ('ordinary-raw.jsonl', 'calibration-inputs.json', 'calibration-CPU-reference.json')}
    protocol = dict(schema='document-evidence-plan-v23-pointer-replay-likelihood',
        purpose='FIXED_V26_ORDINARY256_COST_ONLY', created_utc=utc_now(),
        model=model, model_dir=p['model_dir'], signature=signature, shards=4, batch=4, target_slots=32,
        widths=dict(QA=3584, ordinary=672), source_files=engine.sources(), files=files,
        inventories={'ordinary-raw.jsonl': dict(rows=256, targets=8192)},
        release_file=str(CONTROL/'release.json'), release_sha256=sha256_file(CONTROL/'release.json'),
        interpretation='Ordinary panel only; current-model CPU references, original qualified numerical kernel. No synthetic scoring observations.')
    atomic_json(O/'protocol.json', protocol)
    print(json.dumps(dict(status=release['status'], protocol_sha256=sha256_file(O/'protocol.json'))))


def observations(output):
    values = []
    for shard in range(4):
        d = output/'scores'/f'shard-{shard}'
        report = read(d/'completion.json')
        assert report['scores_sha256'] == sha256_file(d/'scores.jsonl')
        for row in rows(d/'scores.jsonl'):
            if row['panel'] == 'ordinary':
                assert row['mode'] == 'raw' and row['status'] == 'ok'
                assert len(row['target_ids']) == row['tokens'] == len(row['token_logps']) == 32
                assert math.isclose(sum(row['token_logps']), row['full_logp'], rel_tol=0, abs_tol=1e-8)
                assert math.isfinite(row['full_logp']) and not row['synthetic_padding_observation']
                values.append(row)
    assert len(values) == len({v['item_id'] for v in values}) == 256
    return {v['item_id']: v for v in values}


def analyze():
    assert read(CONTROL/'GEN/closure.json')['actual_rows'] == 256
    p = read(O/'protocol.json')
    assert read(CONTROL/'release.json')['source_files'] == sources()
    current = observations(O)
    baseline = {name: observations(path) for name, path in old_groups().items()}
    # The original B0 panel was measured and retained by the closed V23 readout.
    reference = read(OLD_C/'readout-execution-plan-v2.json')['B0_ordinary_reference']
    original_b0 = Path(reference['output'])
    assert all(sha256_file(original_b0/name) == value for name,value in reference['file_sha256'].items())
    assert sha256_file(original_b0/'ordinary-raw.jsonl') == p['files']['ordinary-raw.jsonl']
    b0 = {}
    for shard in range(4):
        d = original_b0/'scores'/f'shard-{shard}'
        report = read(d/'completion.json')
        assert report['scores_sha256'] == sha256_file(d/'scores.jsonl')
        for row in rows(d/'scores.jsonl'):
            if row['panel'] == 'ordinary':
                assert row['status'] == 'ok' and row['tokens'] == 32
                assert len(row['token_logps']) == 32 and math.isclose(sum(row['token_logps']), row['full_logp'], abs_tol=1e-8)
                assert row['item_id'] not in b0
                b0[row['item_id']] = row
    baseline['B0'] = b0
    baseline['selector1'] = observations(R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-ordinary256-v1')
    tables = {'format1': dict(mean_nll_per_token=float(np.mean([-v['full_logp']/32 for v in current.values()])), items=256, tokens=8192)}
    paired = {}
    ids = sorted(current)
    for index, (name, control) in enumerate(sorted(baseline.items())):
        assert set(control) == set(current)
        assert all(control[k]['target_ids'] == current[k]['target_ids'] for k in ids)
        differences = np.array([(-current[k]['full_logp']+control[k]['full_logp'])/32 for k in ids])
        rng = np.random.default_rng(2026101926+100+index)
        bootstrap = np.mean(differences[rng.integers(0,256,(20000,256))], axis=1)
        paired[name] = dict(mean_nll_change=float(differences.mean()), CI95=list(map(float,np.quantile(bootstrap,[.025,.975]))),
            interpretation='Positive means worse ordinary NLL; paired pages, conditional on checkpoints.', seed=2026101926+100+index)
        tables[name] = dict(mean_nll_per_token=float(np.mean([-v['full_logp']/32 for v in control.values()])), items=256, tokens=8192)
    atomic_json(CONTROL/'result.json', dict(status='COMPLETE_FIXED_ORDINARY256_COST', utc=utc_now(),
        protocol_sha256=sha256_file(O/'protocol.json'), tables=tables, paired=paired,
        note='No synthetic rows or pooled tokens as independent items; ordinary pages are the paired units.'))
    print(json.dumps(dict(tables=tables,paired=paired)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','analyze'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else analyze)()
