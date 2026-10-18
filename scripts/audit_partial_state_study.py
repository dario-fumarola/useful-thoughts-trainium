"""Independent complete-question audit of the prospective two-seed v7 study.

Require every registered four-panel generation controller to close before
opening outcomes. Keep the four panels and both seeds separate. This script
does not train, generate, select checkpoints or repair failed attempts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

from scripts.audit_partial_state_panels import (
    exact_trace, independent_score, independent_state, paired, read, rows, sha, write,
)

PANELS = ('DEV_interpolation', 'DEV_structure', 'DEV_range', 'DEV_composition')
SEEDS = ('2026101806', '2026101807')
ARMS = ('writer', 'reader', 'direct', 'cot', 'cot_full')
READER_MODES = ['direct', 'empty', 'teacher', 'corrupt', 'other_teacher',
                'restatement', 'own', 'other']


def require_complete(registry):
    """Fail before any outcome access if any part of the fixed question is open."""
    if (registry['schema'] != 'partial-state-v7-matched5120-evaluation-registry-v1'
            or tuple(registry['seeds']) != SEEDS or tuple(registry['panels']) != PANELS
            or registry['action_budget'] != 96):
        raise ValueError('The declared two-seed/four-panel/96-action study is required')
    reference = registry.get('untouched_reference')
    if (not isinstance(reference, dict) or set(reference['outputs']) != set(PANELS)
            or reference['modes'] != ['direct', 'cot', 'empty']):
        raise ValueError('The prospectively added untouched reference is required')
    if read(Path(reference['control'])/'completion.json')['status'] != 'COMPLETE_ALL_FOUR_FIXED_DEV_PANELS':
        raise ValueError('The untouched reference must close before outcome review')
    for seed in SEEDS:
        if set(registry['jobs'][seed]) != set(ARMS):
            raise ValueError('Both ordinary CoT competitors and all split controls are required')
        for arm in ARMS:
            job = registry['jobs'][seed][arm]
            if set(job['outputs']) != set(PANELS):
                raise ValueError('All four fixed panels are required')
            if read(Path(job['control'])/'completion.json')['status'] != 'COMPLETE_ALL_FOUR_FIXED_DEV_PANELS':
                raise ValueError('Every original controller must close before outcome review')


def numerical(path, shard, *, reference=None, notes=False):
    import torch
    from train.useful_note_precision import compare_distributions
    directory = path/('notes' if notes else 'answers')/f'shard-{shard}'
    ref = torch.load((reference or path)/'native-reference.pt', weights_only=True, map_location='cpu')
    actual = torch.load(directory/'native-logps.pt', weights_only=True, map_location='cpu')
    native = compare_distributions(ref[shard:shard+1], actual)[0]
    if notes:
        modes = ['prefill']
    else:
        requested = read(path/'protocol.json')['modes']
        modes = [m for m in ('direct', 'cot') if m in requested]
        conditioned = [m for m in requested if m not in ('direct', 'cot')]
        if conditioned:
            modes.append(conditioned[0])
    cache = {}
    for mode in modes:
        vectors = torch.load(directory/f'cache-{mode}-logps.pt', weights_only=True, map_location='cpu')
        cache[mode] = compare_distributions(vectors['ordinary'], vectors['cached'])[0]
    if not native['passed'] or any(v['max_logp_error'] > .001 for v in cache.values()):
        raise ValueError('Saved full native/cache vectors fail independent recomputation')
    return dict(native=native, cache=cache, actual_vectors_recomputed=True)


def audit_untouched_reference(root, registry, tokenizer, public, copy):
    """One shared, zero-training reference; never count it as two seed replicas."""
    from train.partial_state_answers import score_answer
    from train.partial_state_evaluation import prompt_tokens, reader_tokens
    job = registry['untouched_reference']
    source_root = Path(registry['source_root'])
    reference, allcells, payloads = {}, {}, {}
    copy('B0/control', Path(job['control']))
    for index, panel in enumerate(PANELS):
        path = Path(job['outputs'][panel])
        stage = Path(job['control'])/f'stage-{index}'
        completed = read(stage/'completion.json')
        if (completed['status'] != 'COMPLETE_FIXED_PANEL_ALL512_ATTEMPTS'
                or completed['output'] != str(path)
                or completed['result_sha256'] != sha(path/'summary.json')
                or read(path/'generation-closure.json')['status'] !=
                'ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED'):
            raise ValueError('Original untouched-reference closure is required')
        copy('B0/'+panel, path)
        p = read(path/'protocol.json')
        if (p['schema'] != 'partial-state-complete-answers-v5'
                or p['reader'] != dict(id='B0', checkpoint=None)
                or p['modes'] != ['direct', 'cot', 'empty'] or p['panel'] != panel
                or p['direct_and_cot_cap'] != 96 or p['reader_answer_cap'] != 32):
            raise ValueError('Unchanged B0 and the exact declared inference budget are required')
        for name, digest in p['source_files'].items():
            if sha(source_root/name) != digest:
                raise ValueError('Original B0 evaluation source changed')
        for name, digest in p['files'].items():
            if sha(path/name) != digest:
                raise ValueError('Original B0 inputs/references changed')
        questions = rows(path/'questions.jsonl')
        ids = [q['item_id'] for q in questions]
        if len(ids) != 512 or len(set(ids)) != 512 or ids != p['ordered_item_ids']:
            raise ValueError('A complete fixed untouched-reference cohort is required')
        question = {q['item_id']: q for q in questions}
        gold = {q['item_id']: exact_trace(q['expression_tree'])[0] for q in questions}
        if gold != {v['item_id']: int(v['answer']) for v in rows(path/'gold.jsonl')}:
            raise ValueError('Reference gold differs from prefix arithmetic')
        cells, raw, checks = {}, [], []
        for shard in range(4):
            exited = read(stage/f'shard-{shard}/exit.json')
            directory = path/f'answers/shard-{shard}'
            run = read(directory/'run.json')
            if (exited['exit_code'] != 0 or exited['original_session_remaining']
                    or run['status'] != 'complete_all_attempts' or run['rows'] != 384
                    or sha(directory/'answers.jsonl') != run['answers_sha256']):
                raise ValueError('All original reference attempts must complete and close')
            checks.append(numerical(path, shard))
            for value in rows(directory/'answers.jsonl'):
                key = (value['item_id'], value['mode'])
                if key in cells or value['item_id'] not in gold or value['mode'] not in p['modes']:
                    raise ValueError('Unexpected or duplicated reference answer')
                q, mode = question[value['item_id']], value['mode']
                plain = mode in ('direct', 'cot')
                prompt = (prompt_tokens(q, tokenizer, cot=mode == 'cot') if plain else
                          reader_tokens(q, [], tokenizer))
                if (value['prompt_ids'] != prompt or value['cap'] != (96 if plain else 32)
                        or value['prefill_width'] != (32 if plain else 128)
                        or value['n_actions'] != len(value['actions'])
                        or value['n_writer_actions'] != 0 or value['source_note'] is not None):
                    raise ValueError('Actual B0 input, budget, or no-writer reference changed')
                independent = independent_score(value, gold[value['item_id']])
                production = score_answer(value, gold[value['item_id']], cot=mode == 'cot')
                if independent != {k: production[k] for k in ('score', 'status')}:
                    raise ValueError('Reference complete-answer parsers disagree')
                cells[key] = {**value, 'generation_status': value['status'], **independent}
                raw.append(dict(arm='B0', **cells[key]))
        if len(cells) != 1536:
            raise ValueError('All reference outcomes must stay in their denominator')
        metrics = {}
        summary = read(path/'summary.json')
        for mode in p['modes']:
            vs = [cells[key, mode] for key in ids]
            metrics[mode] = dict(items=512, correct=sum(v['score'] for v in vs),
                unsupported=sum(v['status'] != 'ok' for v in vs),
                status_counts=dict(Counter(v['status'] for v in vs)),
                generated_actions=sum(v['n_actions'] for v in vs), writer_actions=0,
                decode_wall_seconds=sum(v['wall_seconds'] for v in vs), writer_wall_seconds=0)
            if metrics[mode] != summary['results'][mode]:
                raise ValueError('Original B0 aggregate does not reproduce')
        for name, result in summary['contrasts'].items():
            b, a = name.split('_minus_')
            actual = paired([cells[key, b]['score']-cells[key, a]['score'] for key in ids])
            if actual['difference'] != result['difference'] or actual['ci95'] != result['ci95']:
                raise ValueError('Original B0 interval does not reproduce')
        file = public/f'B0-{panel}-all1536-original-answers.jsonl.gz'
        with gzip.GzipFile(filename=str(file), mode='wb', mtime=0) as stream:
            for value in raw:
                stream.write((json.dumps(value, separators=(',', ':'))+'\n').encode())
        payloads[str(file.relative_to(root))] = dict(rows=len(raw), bytes=file.stat().st_size, sha256=sha(file))
        reference[panel] = dict(results=metrics, numerical_checks=checks,
            model_identity='unchanged_B0', shared_across_both_trained_seeds=True,
            additional_independent_seed_replications=False)
        allcells[panel] = cells
    return reference, allcells, payloads


def audit(root, registry_file):
    root, registry_file = Path(root).resolve(), Path(registry_file).resolve()
    registry = read(registry_file)
    require_complete(registry)  # No gold, answer, or note reads precede this gate.
    design = root/'validation/partial_state_v7_matched_design_20261018.json'
    if sha(design) != registry['design_sha256']:
        raise ValueError('Prospective scientific design changed')
    amendment = root/'validation/partial_state_v7_untouched_reference_amendment_20261018.json'
    if sha(amendment) != registry['reference_amendment_sha256']:
        raise ValueError('The auxiliary unchanged-reference amendment changed')
    source_root = Path(registry['source_root'])
    from transformers import AutoTokenizer
    from train.partial_state_answers import score_answer
    from train.partial_state_checkpoint import verify
    from train.partial_state_evaluation import prompt_tokens, reader_tokens
    from train.partial_state_note_driver import load_bundle
    tokenizer = AutoTokenizer.from_pretrained(root/'local/assets/qwen3-1.7b-base',
                                             local_files_only=True, trust_remote_code=False)
    public = root/'validation/partial_state_v7_matched5120_complete_20261018'
    public.mkdir(exist_ok=False)
    archive = root/'local/backups/evaluation/partial-state-v7-matched5120-closed'
    archive.mkdir(parents=True, exist_ok=False)
    archived, payloads, results = {}, {}, {}
    total_answers = total_notes = 0

    def copy(label, source):
        size = sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
        if shutil.disk_usage(archive).free < 35*2**30 + size:
            raise ValueError('Archive must preserve the declared EBS reserve')
        destination = archive/label
        shutil.copytree(source, destination)
        for file in source.rglob('*'):
            if file.is_file():
                out = destination/file.relative_to(source)
                digest = sha(file)
                if sha(out) != digest:
                    raise ValueError('Archive readback differs')
                archived[str(out.relative_to(archive))] = dict(bytes=file.stat().st_size, sha256=digest)

    reference, reference_cells, reference_payloads = audit_untouched_reference(
        root, registry, tokenizer, public, copy)
    payloads.update(reference_payloads)
    for seed in SEEDS:
        jobs = registry['jobs'][seed]
        for arm in ARMS:
            copy(seed+'/control-'+arm, Path(jobs[arm]['control']))
        seed_results = {}
        for index, panel in enumerate(PANELS):
            paths = {arm: Path(jobs[arm]['outputs'][panel]) for arm in ARMS}
            for arm, path in paths.items():
                stage = Path(jobs[arm]['control'])/f'stage-{index}'
                completed = read(stage/'completion.json')
                result = path/('notes-bundle.json' if arm == 'writer' else 'summary.json')
                if (completed['status'] != 'COMPLETE_FIXED_PANEL_ALL512_ATTEMPTS'
                        or completed['output'] != str(path)
                        or completed['result_sha256'] != sha(result)
                        or read(path/'generation-closure.json')['status'] !=
                        'ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED'):
                    raise ValueError('Original stage/device closure is missing or changed')
                copy(seed+'/'+panel+'/'+arm, path)
            protocols = {arm: read(path/'protocol.json') for arm, path in paths.items()}
            for protocol in protocols.values():
                for name, digest in protocol['source_files'].items():
                    if sha(source_root/name) != digest:
                        raise ValueError('Original frozen evaluation source changed')
                for name, digest in protocol.get('input_pins', {}).items():
                    if sha(name) != digest:
                        raise ValueError('Frozen writer/native preparation changed')
            rp = protocols['reader']
            questions = rows(paths['reader']/'questions.jsonl')
            ids = [q['item_id'] for q in questions]
            if len(ids) != 512 or len(set(ids)) != 512:
                raise ValueError('One complete unique fixed cohort is required')
            by_id = {q['item_id']: q for q in questions}
            gold = {q['item_id']: exact_trace(q['expression_tree'])[0] for q in questions}
            if gold != {v['item_id']: int(v['answer']) for v in rows(paths['reader']/'gold.jsonl')}:
                raise ValueError('Gold differs from the independent prefix interpreter')
            controls = rows(paths['reader']/'controls.jsonl')
            control = {c['item_id']: c for c in controls}
            bundle = load_bundle(paths['writer']/'notes-bundle.json', ids)
            if bundle['execution_device'] != 'xla' or rp['writer'] != bundle['writer']:
                raise ValueError('Actual unchanged native writer notes are required')
            writer = verify(bundle['writer'])
            if (writer['arm'] != 'writer' or writer['step'] != 320
                    or read(writer['training_protocol'])['seed'] != int(seed)):
                raise ValueError('Actual terminal writer belongs to another arm, step, or seed')
            for q, c, generated in zip(questions, controls, bundle['controls']):
                expected_writer_prompt = prompt_tokens(q, tokenizer, cot=True)
                if generated['own']['prompt_ids'] != expected_writer_prompt:
                    raise ValueError('Actual writer input is not the prefix-only projection')
                if generated['own']['n_actions'] != len(generated['own']['actions']):
                    raise ValueError('Writer action inventory differs from its native trace')
                _, trace, expression = exact_trace(q['expression_tree'])
                partial = trace[:-1]
                teacher = ''.join(f'{a} {op} {b} = {v}\n' for a, op, b, v in partial)
                a, op, b, value = partial[-1]
                corrupt = ''.join(f'{a} {op} {b} = {v}\n' for a, op, b, v in
                                  partial[:-1]+[(a, op, b, value+1)])
                expected = {'empty': [], 'teacher': tokenizer.encode(teacher, add_special_tokens=False),
                            'corrupt': tokenizer.encode(corrupt, add_special_tokens=False),
                            'restatement': tokenizer.encode(expression+'\n', add_special_tokens=False)}
                if c['item_id'] != q['item_id'] or any(c['notes'][m] != v for m, v in expected.items()):
                    raise ValueError('Supplied state control differs from exact prefix computation')
                for mode in ('own', 'other'):
                    if (c['notes'][mode] != generated[mode]['content_ids']
                            or c['note_generation'][mode]['item_id'] != generated[mode]['item_id']):
                        raise ValueError('A student/donor note was changed or dropped')
            if sorted(c['other_teacher_item_id'] for c in controls) != sorted(ids):
                raise ValueError('Teacher donors do not preserve the intact population')
            for c in controls:
                donor = c['other_teacher_item_id']
                if donor == c['item_id'] or c['notes']['other_teacher'] != control[donor]['notes']['teacher']:
                    raise ValueError('Unrelated supplied state is changed or self-paired')
            writer_prep = Path(protocols['writer']['model_preparation'])
            copy(seed+'/'+panel+'/writer-native-preparation', writer_prep)
            for shard in range(4):
                stage = Path(jobs['writer']['control'])/f'stage-{index}'
                exited = read(stage/f'shard-{shard}/exit.json')
                if exited['exit_code'] != 0 or exited['original_session_remaining']:
                    raise ValueError('Every original writer session must complete and close')
            checks = {'writer': [numerical(paths['writer'], shard, reference=writer_prep, notes=True)
                                  for shard in range(4)]}
            state = {q['item_id']: independent_state(note['own'], q['expression_tree'])
                     for q, note in zip(questions, bundle['controls'])}
            if state != bundle['state_diagnostics']:
                raise ValueError('Saved writer-state diagnostics differ from independent execution')
            cells, metrics, raw = {}, {}, []
            for arm in ('reader', 'direct', 'cot', 'cot_full'):
                path, p = paths[arm], protocols[arm]
                modes = READER_MODES if arm == 'reader' else ['direct' if arm == 'direct' else 'cot']
                expected_step = 320 if arm == 'reader' else 640
                if (p['schema'] != 'partial-state-complete-answers-v5' or p['modes'] != modes
                        or p['ordered_item_ids'] != ids or p['direct_and_cot_cap'] != 96
                        or p['reader_answer_cap'] != 32 or p['reader']['arm'] != arm
                        or p['reader']['step'] != expected_step):
                    raise ValueError('Fixed model/condition/cohort/budget differs')
                for name, digest in p['files'].items():
                    if sha(path/name) != digest:
                        raise ValueError('Original projected inputs or references changed')
                binding = verify(p['reader'])
                if read(binding['training_protocol'])['seed'] != int(seed):
                    raise ValueError('Model belongs to another training seed')
                checks[arm] = []
                for shard in range(4):
                    stage = Path(jobs[arm]['control'])/f'stage-{index}'
                    exit = read(stage/f'shard-{shard}/exit.json')
                    directory = path/f'answers/shard-{shard}'
                    run = read(directory/'run.json')
                    if (exit['exit_code'] != 0 or exit['original_session_remaining']
                            or run['status'] != 'complete_all_attempts'
                            or sha(directory/'answers.jsonl') != run['answers_sha256']):
                        raise ValueError('Every original shard must complete and physically close')
                    checks[arm].append(numerical(path, shard))
                    for value in rows(directory/'answers.jsonl'):
                        key = (arm, value['item_id'], value['mode'])
                        if key in cells or value['item_id'] not in gold or value['mode'] not in modes:
                            raise ValueError('Unknown or duplicated answer')
                        q, mode = by_id[value['item_id']], value['mode']
                        plain = mode in ('direct', 'cot')
                        prompt = (prompt_tokens(q, tokenizer, cot=mode == 'cot') if plain else
                                  reader_tokens(q, control[q['item_id']]['notes'][mode], tokenizer))
                        if (value['prompt_ids'] != prompt or value['cap'] != (96 if plain else 32)
                                or value['prefill_width'] != (32 if plain else 128)
                                or value['n_actions'] != len(value['actions'])):
                            raise ValueError('Actual context, budget, or action inventory differs')
                        independent = independent_score(value, gold[value['item_id']])
                        production = score_answer(value, gold[value['item_id']], cot=mode == 'cot')
                        if independent != {k: production[k] for k in ('score', 'status')}:
                            raise ValueError('Independent complete-answer parsers disagree')
                        cells[key] = {**value, 'generation_status': value['status'], **independent}
                        raw.append(dict(arm=arm, **cells[key]))
                summary = read(path/'summary.json')
                metrics[arm] = {}
                for mode in modes:
                    vs = [cells[arm, key, mode] for key in ids]
                    result = dict(items=512, correct=sum(v['score'] for v in vs),
                                  unsupported=sum(v['status'] != 'ok' for v in vs),
                                  status_counts=dict(Counter(v['status'] for v in vs)),
                                  generated_actions=sum(v['n_actions'] for v in vs),
                                  writer_actions=sum(v['n_writer_actions'] for v in vs),
                                  decode_wall_seconds=sum(v['wall_seconds'] for v in vs),
                                  writer_wall_seconds=sum(0 if v['source_note'] is None else
                                                          v['source_note']['wall_seconds'] for v in vs))
                    if result != summary['results'][mode]:
                        raise ValueError('Original aggregate does not reproduce')
                    metrics[arm][mode] = result
                for name, result in summary['contrasts'].items():
                    b, a = name.split('_minus_')
                    actual = paired([cells[arm, key, b]['score']-cells[arm, key, a]['score'] for key in ids])
                    if actual['difference'] != result['difference'] or actual['ci95'] != result['ci95']:
                        raise ValueError('Original interval does not reproduce')
            if len(cells) != 512*11:
                raise ValueError('Every outcome must stay in the denominator')
            contrasts = {}
            for b, a in [('own', 'direct'), ('own', 'empty'), ('own', 'other'),
                         ('teacher', 'own'), ('teacher', 'direct'), ('teacher', 'corrupt'),
                         ('teacher', 'other_teacher'), ('own', 'restatement')]:
                contrasts[b+'_minus_'+a] = paired([cells['reader', key, b]['score']-
                                                  cells['reader', key, a]['score'] for key in ids])
            for arm in ('direct', 'cot', 'cot_full'):
                mode = 'direct' if arm == 'direct' else 'cot'
                contrasts['split_minus_'+arm] = paired([cells['reader', key, 'own']['score']-
                                                       cells[arm, key, mode]['score'] for key in ids])
            for mode in ('direct', 'cot', 'empty'):
                if {key for key, _ in reference_cells[panel]} != set(ids):
                    raise ValueError('Untouched and trained cohorts must match exactly')
                contrasts['split_minus_untouched_B0_'+mode] = paired([
                    cells['reader', key, 'own']['score']-reference_cells[panel][key, mode]['score']
                    for key in ids])
            identical, duplicates = {}, 0
            for (arm, _, _), value in cells.items():
                if arm != 'reader' or value['status'] == 'error':
                    continue
                key = (tuple(value['prompt_ids']), value['cap'], value['prefill_width'])
                response = (value['actions'], value['text'], value['raw_text'], value['stop_type'])
                if key in identical:
                    if identical[key] != response:
                        raise ValueError('Identical actual inputs produce different deterministic answers')
                    duplicates += 1
                else:
                    identical[key] = response
            notes = [c['own'] for c in bundle['controls']]
            for suffix, values in [('answers', raw), ('notes', notes)]:
                file = public/f'seed{seed}-{panel}-all-original-{suffix}.jsonl.gz'
                with gzip.GzipFile(filename=str(file), mode='wb', mtime=0) as stream:
                    for value in values:
                        stream.write((json.dumps(value, separators=(',', ':'))+'\n').encode())
                payloads[str(file.relative_to(root))] = dict(rows=len(values), bytes=file.stat().st_size,
                                                            sha256=sha(file))
            seed_results[panel] = dict(results=metrics, contrasts=contrasts, numerical_checks=checks,
                writer_state_diagnostics=dict(items=512, canonical_nonroot_exact=sum(v['canonical_nonroot_exact'] for v in state.values()),
                    canonical_and_natural_stop=sum(v['canonical_and_natural_stop'] for v in state.values()),
                    equations_supported=sum(v['equations_supported'] for v in state.values()),
                    all_stated_equations_correct=sum(v['all_stated_equations_correct'] for v in state.values()),
                    root_operation_transmitted=sum(v['root_operation_transmitted'] for v in state.values()),
                    stop_counts=dict(Counter(v['stop_type'] for v in notes)), status_counts=dict(Counter(v['status'] for v in notes))),
                identical_reader_inputs=dict(duplicate_attempts=duplicates, all_matched=True, additional_independent_observations=False),
                examples_fixed_first_two=[dict(question=q, writer=notes[i],
                    answers={arm: cells[arm, q['item_id'], 'direct' if arm == 'direct' else 'own' if arm == 'reader' else 'cot']
                             for arm in ('reader', 'direct', 'cot', 'cot_full')}) for i, q in enumerate(questions[:2])])
            total_answers += len(cells)
            total_notes += len(notes)
        results[seed] = seed_results
    if total_answers != 45056 or total_notes != 4096:
        raise ValueError('Both seeds and all panels are required')
    write(archive/'ARCHIVE.json', dict(status='COMPLETE_MATCHED_STUDY_EBS_READBACK_VERIFIED', files=archived))
    report = dict(schema='partial-state-v7-matched5120-complete-v1',
        status='COMPLETE_BOTH_SEEDS_ALL_FOUR_PANELS_NO_FILTERING',
        finished_utc=datetime.now(timezone.utc).isoformat(), results=results,
        untouched_reference=reference, primary_answer_attempts=total_answers,
        untouched_reference_answer_attempts=6144,
        answer_attempts=total_answers+6144, writer_attempts=total_notes,
        audit_script_sha256=sha(__file__), registry_sha256=sha(registry_file),
        design_sha256=sha(design), public_payloads=payloads,
        untouched_reference_amendment_sha256=sha(amendment),
        EBS_archive_receipt_sha256=sha(archive/'ARCHIVE.json'),
        limitations=['Two training seeds, same four development-exposed synthetic panels; never pool item/seed uncertainty or claim a pristine final test.',
                     'Split has two models. Equal total updates/action caps do not equal FLOPs or end-to-end latency; raw supervised token counts and timings are separate costs.',
                     'Unchanged B0 is a single shared zero-training reference, not two independent baseline replicas. It is auxiliary to the original matched-training primary comparisons.',
                     'All errors, caps, unsupported outputs and wrong student states remain in denominators.',
                     'Independent complete-answer and writer-state parsers re-execute prefix math; all saved numerical vectors are recomputed.',
                     'Root-operation counts cover fully parsed equation notes; they do not establish absence of final-answer text in unsupported formats.',
                     'This is a synthetic computation study, not evidence of general natural-text pretraining improvement.'])
    write(root/'validation/partial_state_v7_matched5120_complete_20261018.json', report)
    return dict(status=report['status'], answer_attempts=total_answers+6144, writer_attempts=total_notes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.registry), indent=2))
