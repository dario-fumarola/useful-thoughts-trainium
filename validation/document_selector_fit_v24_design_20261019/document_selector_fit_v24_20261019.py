"""Bounded frozen-model TRAIN annotation-fit diagnostic; no parameter updates."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from train import document_qa_evaluate as engine
from train.document_evidence_plan_evaluate import prompt_text
from train.document_evidence_plan_answers import parse_evidence, annotated_support_overlap
from train.document_qa_answers import parse_answer, score
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/document-evidence-eval-v23'
C = R/'local/diagnostics/document-selector-fit-v24-TRAIN256-v1'
O = R/'local/nvme-cpt/evaluation/document-selector-fit-v24-TRAIN256-v1'
D = R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
OLD = R/'local/nvme-cpt/evaluation/document-evidence-plan-v23-evidence_select-seed2026101923-held255-v1'
SCHEMA = 'document-selector-fit-v24-TRAIN'
SALT = 'document-selector-fit-v24-TRAIN-v1:'
COUNTS = {2: 128, 3: 96, 4: 32}


def sources():
    names = ('train/document_evidence_plan_evaluate.py',
             'train/document_evidence_plan_packing.py',
             'train/document_evidence_plan_answers.py')
    return {**engine.sources(), **{n: sha256_file(W/n) for n in names},
            str(Path(__file__).resolve().relative_to(R)): sha256_file(__file__)}


def rank(row):
    return hashlib.sha256((SALT+row['item_id']).encode()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    rows = engine.rows(D/'TRAIN2048.jsonl')
    assert len(rows) == len({x['item_id'] for x in rows}) == 2048
    selected = sorted([x for n, count in COUNTS.items()
                       for x in sorted((v for v in rows if len(v['support_indices']) == n), key=rank)[:count]], key=rank)
    assert Counter(len(x['support_indices']) for x in selected) == Counter(COUNTS)
    assert all(set(x['question']) == {'item_id', 'question', 'contexts'}
               and x['question']['item_id'] == x['item_id']
               and all(set(v) == {'title', 'text'} for v in x['question']['contexts']) for x in selected)
    p = engine.read(OLD/'protocol.json')
    assert p['model']['step'] == 256 and p['model']['arm'] == 'evidence_select'
    assert sha256_file(p['model']['checkpoint']) == p['model']['checkpoint_sha256']
    assert all(sha256_file(W/n) == v for n, v in p['source_files'].items())
    C.mkdir(parents=True, exist_ok=False)
    O.mkdir(parents=True, exist_ok=False)
    with (C/'TRAIN256-labels.jsonl').open('x') as f:
        for x in selected:
            f.write(json.dumps(x)+'\n')
    with (C/'TRAIN256-questions-only.jsonl').open('x') as f:
        for x in selected:
            f.write(json.dumps(x['question'])+'\n')
    tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
    lengths = []
    with (O/'prompts-selection.jsonl').open('x') as f:
        for x in selected:
            ids = tok.encode(prompt_text(x['question'], 'selection'), add_special_tokens=False)
            assert len(ids) <= 3584 and not set(ids) & {151643, 151645}
            lengths.append(len(ids))
            f.write(json.dumps(dict(item_id=x['item_id'], mode='selection', prompt_ids=ids, parser_mode='prior'))+'\n')
    for name in ('native-contexts.json', 'native-reference.pt'):
        assert sha256_file(OLD/name) == p['files'][name]
        os.link(OLD/name, O/name)
    release = dict(status='RELEASED_FROZEN_TRAIN_SELECTOR_FIT_ONLY_V24', utc=utc_now(),
        question='Does the frozen terminal selector reproduce its own TRAIN support annotations?',
        items=256, strata=COUNTS, salt=SALT,
        training_file_sha256=sha256_file(D/'TRAIN2048.jsonl'),
        cohort_labels_sha256=sha256_file(C/'TRAIN256-labels.jsonl'),
        cohort_questions_sha256=sha256_file(C/'TRAIN256-questions-only.jsonl'),
        model=p['model'], source_files=sources(), model_modes={'evidence_select': ['selection']},
        primary='Exact annotation-set agreement, separately for 2, 3 and 4 support paragraphs.',
        fit_target='At least 90% exact annotation sets in BOTH 2- and 3-support strata; 4-support n32 descriptive.',
        secondary=['Annotation precision/recall/F1, emitted-ID count, termination, unsupported/error counts.',
                   'Canonical answer accuracy is TRAIN fitting only, never a capability endpoint.'],
        decision='Below target: inspect optimization/supervision before scaling. Above: investigate generalization; no automatic training release.',
        limits=['Imperfect author annotations are not semantic gold.',
                'Oversampled TRAIN strata cannot be pooled against the differently mixed held panel.',
                'No parameter updates, new held evaluation, retries or new calibration kernel.'],
        original_generation_protocol_sha256=sha256_file(OLD/'protocol.json'))
    atomic_json(C/'release.json', release)
    p.update(schema=SCHEMA, purpose='FROZEN_TRAIN_SELECTOR_FIT_ONLY_V24', created_utc=utc_now(),
        items=256, modes=['selection'], shards=4, source_files=sources(),
        files={n: sha256_file(O/n) for n in ('native-contexts.json', 'native-reference.pt', 'prompts-selection.jsonl')},
        ordered_item_ids=[x['item_id'] for x in selected], maximum_prompt_tokens={'selection': max(lengths)},
        question_file=str(C/'TRAIN256-questions-only.jsonl'), question_file_sha256=release['cohort_questions_sha256'],
        release_file=str(C/'release.json'), release_sha256=sha256_file(C/'release.json'), limits=release['limits'])
    atomic_json(O/'protocol.json', p)
    atomic_json(C/'preparation.json', dict(status='PASS_FIXED_HASH_TRAIN_PROJECTION_AND_IDENTICAL_DECODER', utc=utc_now(),
        protocol_sha256=sha256_file(O/'protocol.json'), strata=COUNTS,
        pool_strata=dict(Counter(len(x['support_indices']) for x in rows)),
        prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths),
        no_labels_in_generation_files=True, original_model_and_native_references_unchanged=True))
    print(json.dumps(engine.read(C/'preparation.json')))


def analyze():
    assert engine.read(C/'GEN/closure.json')['actual_attempts'] == 256
    p = engine.read(O/'protocol.json'); release = engine.read(C/'release.json')
    assert sha256_file(C/'TRAIN256-labels.jsonl') == release['cohort_labels_sha256']
    labels = {v['item_id']: v for v in engine.rows(C/'TRAIN256-labels.jsonl')}
    values = [v for shard in range(4) for v in engine.rows(O/'answers'/f'shard-{shard}'/'attempts.jsonl')]
    assert len(values) == len({v['item_id'] for v in values}) == 256 and set(labels) == {v['item_id'] for v in values}
    scored = []
    for v in values:
        gold = labels[v['item_id']]
        parsed = parse_evidence(v['raw_text'], len(gold['question']['contexts']))
        answer = parse_answer(v['raw_text'], 'prior', terminated=v['stop_type'] not in ('length_cap', 'error'), capped=v['stop_type'] == 'length_cap')
        scored.append(dict(native=v, support_count=len(gold['support_indices']),
            selection=asdict(parsed), overlap=annotated_support_overlap(parsed, gold['support_indices']),
            answer=asdict(answer), answer_score=score(answer, gold['answer'])))
    strata = {}
    for n in COUNTS:
        sub = [v for v in scored if v['support_count'] == n]
        strata[str(n)] = dict(items=len(sub), exact=sum(v['overlap']['exact_match'] for v in sub),
            **{k: sum(v['overlap'][k] for v in sub)/len(sub) for k in ('precision', 'recall', 'f1')},
            output_ID_counts=dict(Counter(len(v['selection']['indices']) for v in sub)),
            selection_status=dict(Counter(v['selection']['status'] for v in sub)),
            answer_status=dict(Counter(v['answer']['status'] for v in sub)),
            execution_errors=sum(v['native']['status'] != 'ok' for v in sub),
            stop_types=dict(Counter(v['native']['stop_type'] for v in sub)),
            canonical_TRAIN_correct=sum(v['answer_score']['canonical_em'] for v in sub))
    with (C/'native-and-scored.jsonl').open('x') as f:
        for v in scored:
            f.write(json.dumps(v)+'\n')
    result = dict(status='COMPLETE_FROZEN_TRAIN_SELECTOR_FIT_V24', utc=utc_now(),
        release_sha256=sha256_file(C/'release.json'), protocol_sha256=sha256_file(O/'protocol.json'),
        scored_sha256=sha256_file(C/'native-and-scored.jsonl'), strata=strata,
        operational_TRAIN_fit_target_pass=all(strata[str(n)]['exact']/strata[str(n)]['items'] >= .9 for n in (2, 3)),
        limits=release['limits'])
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
