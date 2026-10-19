"""Evaluate the weight treatment on the unchanged, already inspected TRAIN cohort."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

from train import document_qa_evaluate as engine
from train.document_evidence_plan_answers import parse_evidence, annotated_support_overlap
from train.document_qa_answers import parse_answer, score
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/document-evidence-eval-v23'
SCIENCE = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
C = SCIENCE/'TRAIN-fit'
O = R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-TRAIN256-v1'
OLD = R/'local/diagnostics/document-selector-fit-v24-TRAIN256-v1'
OLD_O = R/'local/nvme-cpt/evaluation/document-selector-fit-v24-TRAIN256-v1'
SCHEMA = 'document-evidence-weight-v25-TRAIN-fit'
COUNTS = {2: 128, 3: 96, 4: 32}


def sources():
    names = ('train/document_evidence_plan_evaluate.py',
             'train/document_evidence_plan_packing.py',
             'train/document_evidence_plan_answers.py')
    return {**engine.sources(), **{n: sha256_file(W/n) for n in names},
            str(Path(__file__).resolve().relative_to(R)): sha256_file(__file__)}


def prepare():
    import torch
    from train.evaluate_local import setup_model
    binding = SCIENCE/'frozen-models/evidence_select-weight1-seed2026101923.json'
    model = engine.read(binding)
    assert model['status'] == 'AUDITED_EVIDENCE_WEIGHT_V25_TERMINAL256' and model['step'] == 256
    assert sha256_file(model['checkpoint']) == model['checkpoint_sha256']
    assert sha256_file(model['audit_file']) == model['audit_sha256']
    assert sha256_file(model['protection_file']) == model['protection_sha256']
    assert engine.read(model['audit_file'])['status'] == 'PASS_SINGLE_TERMINAL256_EVIDENCE_WEIGHT_TRAINING_STATES_NATIVE8_ORDER_COUNTERS_AND_LABELS'
    assert engine.read(model['protection_file'])['status'] == 'PASS_SINGLE_TERMINAL_MODEL_AND_LATEST_FULL256_PROTECTED'
    original = engine.read(OLD_O/'protocol.json')
    refs = engine.read(OLD_O/'native-contexts.json')['refs']
    torch.set_num_threads(16)
    backend, _, loaded, signature, _ = setup_model(SimpleNamespace(
        model_dir=Path(original['model_dir']), checkpoint=Path(model['checkpoint']), device='cpu'))
    assert all(loaded[k] == model[k] for k in ('step', 'run_id', 'spec_hash'))
    native = []
    with torch.no_grad():
        for ref in refs:
            x = torch.full((1, 1024), 151643, dtype=torch.long)
            x[0, :len(ref['context_ids'])] = torch.tensor(ref['context_ids'])
            hidden = backend.hidden(x)
            native.append(backend.model.lm_head(hidden[:, len(ref['context_ids'])-1]).float().log_softmax(-1))
    native = torch.cat(native).cpu()
    del backend, hidden, x
    C.mkdir(exist_ok=False)
    O.mkdir(parents=True, exist_ok=False)
    torch.save(native, O/'native-reference.pt')
    for name in ('TRAIN256-labels.jsonl', 'TRAIN256-questions-only.jsonl'):
        os.link(OLD/name, C/name)
    for name in ('native-contexts.json', 'prompts-selection.jsonl'):
        assert sha256_file(OLD_O/name) == original['files'][name]
        os.link(OLD_O/name, O/name)
    assert sha256_file(OLD/'native-and-scored.jsonl') == engine.read(OLD/'result.json')['scored_sha256']
    release = dict(status='RELEASED_WEIGHT_V25_FIXED_TRAIN_FIT_ONLY', utc=utc_now(),
        model=model, parent_release_sha256=sha256_file(SCIENCE/'release.json'),
        source_files=sources(), model_modes={'evidence_select': ['selection']},
        original_TRAIN_fit_report_sha256=sha256_file(OLD/'result.json'),
        original_native_scored_sha256=sha256_file(OLD/'native-and-scored.jsonl'),
        labels_sha256=sha256_file(C/'TRAIN256-labels.jsonl'),
        scope='Same TRAIN cohort, prompts, decoder and128 cap; only terminal model changes. Not a generalization result.')
    atomic_json(C/'release.json', release)
    original.update(schema=SCHEMA, purpose='WEIGHT_V25_FROZEN_TRAIN_FIT_ONLY', created_utc=utc_now(),
        model=model, signature=signature, source_files=sources(),
        files={n: sha256_file(O/n) for n in ('native-reference.pt', 'native-contexts.json', 'prompts-selection.jsonl')},
        release_file=str(C/'release.json'), release_sha256=sha256_file(C/'release.json'),
        question_file=str(C/'TRAIN256-questions-only.jsonl'), question_file_sha256=sha256_file(C/'TRAIN256-questions-only.jsonl'))
    atomic_json(O/'protocol.json', original)
    print(json.dumps(dict(status='PREPARED_UNCHANGED_TRAIN_COHORT_WITH_ACTUAL_CURRENT_MODEL_CPU_REFERENCE', protocol_sha256=sha256_file(O/'protocol.json'))))


def analyze():
    import numpy as np
    assert engine.read(C/'GEN/closure.json')['actual_attempts'] == 256
    release = engine.read(C/'release.json')
    assert sha256_file(C/'TRAIN256-labels.jsonl') == release['labels_sha256']
    assert sha256_file(OLD/'native-and-scored.jsonl') == release['original_native_scored_sha256']
    labels = {v['item_id']: v for v in engine.rows(C/'TRAIN256-labels.jsonl')}
    control = {v['native']['item_id']: v for v in engine.rows(OLD/'native-and-scored.jsonl')}
    values = [v for shard in range(4) for v in engine.rows(O/'answers'/f'shard-{shard}'/'attempts.jsonl')]
    assert len(values) == len({v['item_id'] for v in values}) == 256 and set(labels) == {v['item_id'] for v in values}
    scored = []
    for v in values:
        gold = labels[v['item_id']]
        parsed = parse_evidence(v['raw_text'], len(gold['question']['contexts']))
        answer = parse_answer(v['raw_text'], 'prior', terminated=v['stop_type'] not in ('length_cap', 'error'), capped=v['stop_type'] == 'length_cap')
        scored.append(dict(native=v, support_count=len(gold['support_indices']), selection=asdict(parsed),
            overlap=annotated_support_overlap(parsed, gold['support_indices']), answer=asdict(answer),
            answer_score=score(answer, gold['answer'])))
    strata = {}
    for n in COUNTS:
        sub = sorted((v for v in scored if v['support_count'] == n), key=lambda v: v['native']['item_id'])
        delta = np.array([v['overlap']['exact_match']-control[v['native']['item_id']]['overlap']['exact_match'] for v in sub])
        rng = np.random.default_rng(2026101925+n)
        bootstrap = delta[rng.integers(0, len(sub), size=(20000, len(sub)))].mean(1)
        lo, hi = np.quantile(bootstrap, [.025, .975])
        strata[str(n)] = dict(items=len(sub), exact=sum(v['overlap']['exact_match'] for v in sub),
            control_exact=sum(control[v['native']['item_id']]['overlap']['exact_match'] for v in sub),
            paired_gain=float(delta.mean()), paired_CI95=[float(lo), float(hi)],
            **{k: sum(v['overlap'][k] for v in sub)/len(sub) for k in ('precision', 'recall', 'f1')},
            output_ID_counts=dict(Counter(len(v['selection']['indices']) for v in sub)),
            selection_status=dict(Counter(v['selection']['status'] for v in sub)),
            answer_status=dict(Counter(v['answer']['status'] for v in sub)),
            execution_errors=sum(v['native']['status'] != 'ok' for v in sub),
            stop_types=dict(Counter(v['native']['stop_type'] for v in sub)),
            canonical_TRAIN_correct=sum(v['answer_score']['canonical_em'] for v in sub))
    with (C/'native-and-scored.jsonl').open('x') as stream:
        for v in scored:
            stream.write(json.dumps(v)+'\n')
    result = dict(status='COMPLETE_WEIGHT_V25_FIXED_TRAIN_FIT', utc=utc_now(), strata=strata,
        fitting_target_pass=all(strata[str(n)]['exact']/strata[str(n)]['items'] >= .9 for n in (2, 3)) and strata['3']['paired_CI95'][0] > 0.,
        protocol_sha256=sha256_file(O/'protocol.json'), scored_sha256=sha256_file(C/'native-and-scored.jsonl'),
        bootstrap=dict(draws=20000, seed_base=2026101925, stratum_seed='base+support_count', unit='paired TRAIN question', conditional_on_checkpoints=True),
        limits=['Exact author annotation agreement is not semantic entailment.', 'TRAIN fitting is not generalization or useful reasoning.', 'Oversampled strata are separate; four-support n32 descriptive.'])
    atomic_json(C/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'analyze'))
    parser.add_argument('--output', type=Path, default=O)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--until')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'analyze': analyze()
    else:
        assert args.shard is not None and args.until
        engine.generate(args.output, args.shard, args.until, schema=SCHEMA, expected_sources=sources())
