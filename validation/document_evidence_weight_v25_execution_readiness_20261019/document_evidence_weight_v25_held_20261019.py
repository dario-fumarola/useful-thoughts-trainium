"""Fixed developmental held readout, with matched closed-field answer replays."""
import argparse
import json
import os
from pathlib import Path
from train import document_qa_evaluate as engine
from train.document_evidence_plan_evaluate import prompt_text
from document_evidence_v23_likelihood_contexts import context_text, selected_ids
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/document-evidence-eval-v23'
SCIENCE = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D = R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
TRAIN = R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-TRAIN256-v1'
OLD = R/'local/nvme-cpt/evaluation/document-evidence-plan-v23-evidence_select-seed2026101923-held255-v1'
NATIVE = R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-native-v1'
REPLAY = R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-answer-replay-v1'
SCHEMAS = {'native': 'document-evidence-weight-v25-native-held',
           'replay': 'document-evidence-weight-v25-answer-replay'}


def sources():
    names = ('train/document_evidence_plan_evaluate.py',
             'train/document_evidence_plan_packing.py',
             'train/document_evidence_plan_answers.py')
    local = [Path(__file__).resolve(), R/'local/analysis/document_evidence_v23_likelihood_contexts.py']
    return {**engine.sources(), **{n: sha256_file(W/n) for n in names},
            **{str(p.relative_to(R)): sha256_file(p) for p in local}}


def prepare(kind):
    from transformers import AutoTokenizer
    assert kind in SCHEMAS
    parent = SCIENCE/'evaluation-release.json'
    review = engine.read(parent)
    assert review['status'] == 'RELEASED_WEIGHT_V25_TERMINAL_FIXED_EVALUATION'
    assert review['held_source_files'] == sources()
    p = engine.read(TRAIN/'protocol.json')
    assert p['model']['checkpoint_sha256'] == review['model_checkpoint_sha256']
    assert sha256_file(p['model']['checkpoint']) == review['model_checkpoint_sha256']
    manifest = engine.read(D/'manifest.json')
    assert sha256_file(D/'manifest.json') == review['data_manifest_sha256']
    for key in ('held_questions', 'held_gold'):
        info = manifest['files'][key]
        assert sha256_file(info['path']) == info['sha256']
    questions = engine.rows(manifest['files']['held_questions']['path'])
    assert len(questions) == 255 and all(set(q) == {'item_id', 'question', 'contexts'} for q in questions)
    plans = {}
    if kind == 'replay':
        closure = SCIENCE/'HELD-native/GEN/closure.json'
        assert engine.read(closure)['actual_attempts'] == 510
        for shard in range(4):
            d = NATIVE/'answers'/f'shard-{shard}'
            report = engine.read(d/'run.json')
            assert report['status'] == 'complete_all_attempts' and report['attempts_sha256'] == sha256_file(d/'attempts.jsonl')
            for row in engine.rows(d/'attempts.jsonl'):
                if row['mode'] == 'selection':
                    assert row['item_id'] not in plans
                    plans[row['item_id']] = row
        assert set(plans) == {q['item_id'] for q in questions}
        gold = {v['item_id']: v for v in engine.rows(manifest['files']['held_gold']['path'])}
    output = NATIVE if kind == 'native' else REPLAY
    control = SCIENCE/('HELD-native' if kind == 'native' else 'HELD-replay')
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(exist_ok=False)
    modes = ['direct', 'selection'] if kind == 'native' else ['empty', 'selected_id_replay', 'oracle']
    tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
    lengths, metadata, files = {}, [], {}
    for name in ('native-contexts.json', 'native-reference.pt'):
        assert sha256_file(TRAIN/name) == p['files'][name]
        os.link(TRAIN/name, output/name)
        files[name] = sha256_file(output/name)
    for mode in modes:
        path = output/f'prompts-{mode}.jsonl'
        sizes = []
        with path.open('x') as stream:
            for q in questions:
                if kind == 'native':
                    text, parser_mode = prompt_text(q, mode), 'prior'
                else:
                    indices = None
                    if mode == 'selected_id_replay':
                        indices, fallback = selected_ids(plans[q['item_id']], len(q['contexts']))
                        metadata.append(dict(item_id=q['item_id'], indices=indices, **fallback))
                    elif mode == 'oracle':
                        indices = gold[q['item_id']]['support_indices']
                    text, parser_mode = context_text(q, mode, indices), 'direct'
                    assert text.endswith('Final answer:\n')
                ids = tok.encode(text, add_special_tokens=False)
                assert len(ids) <= 3584 and not set(ids) & {151643, 151645}
                sizes.append(len(ids))
                stream.write(json.dumps(dict(item_id=q['item_id'], mode=mode,
                    prompt_ids=ids, parser_mode=parser_mode))+'\n')
        files[path.name] = sha256_file(path)
        lengths[mode] = max(sizes)
        if kind == 'native':
            assert files[path.name] == engine.read(OLD/'protocol.json')['files'][path.name]
    atomic_json(control/'replay-selection-metadata.json', dict(items=metadata,
        interpretation='Canonical pointer IDs only; no generated answer is pasted into a prompt.'))
    release = dict(status='RELEASED_FIXED_WEIGHT_V25_HELD_GENERATION', utc=utc_now(),
        parent_review_sha256=sha256_file(parent), kind=kind, schema=SCHEMAS[kind],
        source_files=sources(), model_modes={'evidence_select': modes},
        native_protocol_sha256=None if kind == 'native' else sha256_file(NATIVE/'protocol.json'),
        model_checkpoint_sha256=p['model']['checkpoint_sha256'],
        interpretation='Previously inspected held255, developmental only. Closed-field replay uses bare-answer parsing, distinct from native generation.')
    atomic_json(control/'release.json', release)
    p.update(schema=SCHEMAS[kind], purpose='FIXED_WEIGHT_V25_DEVELOPMENTAL_HELD_'+kind.upper(),
        created_utc=utc_now(), source_files=sources(), items=255, modes=modes, shards=4,
        files=files, ordered_item_ids=[q['item_id'] for q in questions], maximum_prompt_tokens=lengths,
        question_file=manifest['files']['held_questions']['path'],
        question_file_sha256=manifest['files']['held_questions']['sha256'],
        release_file=str(control/'release.json'), release_sha256=sha256_file(control/'release.json'),
        limits=['Held255 was previously inspected; no untouched confirmation claim.',
                'All errors/caps/unsupported/fallbacks remain in denominators.',
                'Replay contexts prefill both Evidence and Final answer; no hard token ban.',
                'Oracle IDs are imperfect privileged annotations, not semantic proof.'])
    atomic_json(output/'protocol.json', p)
    print(json.dumps(dict(status='PREPARED_FIXED_DEVELOPMENTAL_HELD', kind=kind,
        items=255, modes=modes, protocol_sha256=sha256_file(output/'protocol.json'),
        replay_fallbacks=sum(v['fallback'] for v in metadata))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-native', 'prepare-replay', 'generate'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--until')
    args = parser.parse_args()
    if args.command.startswith('prepare-'):
        prepare(args.command.split('-')[1])
    else:
        assert args.output and args.shard is not None and args.until
        p = engine.read(args.output/'protocol.json')
        assert p['schema'] in SCHEMAS.values()
        engine.generate(args.output, args.shard, args.until, schema=p['schema'], expected_sources=sources())
