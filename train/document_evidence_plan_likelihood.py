"""Canonical pointer-replay diagnostics, not native-generation likelihood.

The native answer observations stay unchanged. Invalid/missing generated plans
use an explicitly flagged empty replay;no generated answer is pasted back in.
"""
from pathlib import Path
import argparse
import json
import math
import time
from types import SimpleNamespace

from train.document_evidence_plan_answers import parse_evidence
from train.document_evidence_plan_packing import prompt, FINAL
from train.useful_note_sft_data import EOS
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

MODES = ('direct', 'empty', 'selected_id_replay', 'other_id_replay', 'oracle')
ROOT = Path(__file__).resolve().parents[1]


def selected_ids(attempt, paragraph_count):
    if attempt.get('status') != 'ok':
        return [], dict(fallback=True, reason='generation_error')
    parsed = parse_evidence(attempt.get('raw_text', ''), paragraph_count)
    if parsed.status != 'complete':
        return [], dict(fallback=True, reason='plan_'+parsed.status, parser_reason=parsed.reason)
    return list(parsed.indices), dict(fallback=False, reason='valid_nonempty_plan')


def context_text(question, mode, indices=None):
    if mode not in MODES:
        raise ValueError('Declared diagnostic condition required')
    if mode == 'direct':
        if indices is not None:
            raise ValueError('No pointer labels in the ordinary direct condition')
        return prompt(question, 'direct') + FINAL
    if mode == 'empty':
        if indices is not None:
            raise ValueError('Empty condition cannot inspect supplied pointer labels')
        indices = []
    if (not isinstance(indices, list) or len(indices) != len(set(indices))
            or any(type(value) is not int or not 1 <= value <= len(question['contexts']) for value in indices)):
        raise ValueError('Complete unique in-range pointers required;never clip/repair')
    field = 'Evidence:' + ''.join(f' [{value}]' for value in indices) + '\n'
    return prompt(question, 'selection') + field + FINAL


def scoring_example(question, answer, mode, tokenizer, indices=None):
    text = context_text(question, mode, indices)
    past = tokenizer.encode(text, add_special_tokens=False)
    joint = tokenizer.encode(text + answer, add_special_tokens=False)
    if joint[:len(past)] != past:
        raise ValueError('Native answer boundary changed;never repair silently')
    target = joint[len(past):] + [EOS]
    if (not answer or not past or not 1 < len(target) <= 32
            or len(past) + len(target) > 3584 or EOS in joint):
        raise ValueError('Complete fixed native span required;never shorten/drop a question')
    return dict(item_id=question['item_id'], panel='QA', mode=mode,
                context_ids=past, target_ids=target,
                interpretation='Canonical pointer-ID replay,not the native generation observation')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sources():
    names = ('train/document_evidence_plan_likelihood.py', 'train/document_evidence_plan_answers.py',
        'train/document_evidence_plan_packing.py', 'train/document_qa_packing.py',
        'train/document_evidence_data.py', 'train/document_qa_span_scores.py',
        'train/evaluate_local.py', 'train/evaluate.py', 'train/neuron_backend.py',
        'train/runtime_compat.py', 'weekend_thinking/pretrained.py',
        'weekend_thinking/artifacts.py')
    return {name: sha256_file(ROOT/name) for name in names}


def physical_batch(batch, size=4):
    """Pad only execution, preserving the exact number of original observations."""
    if not batch or type(size) is not int or not 1 <= len(batch) <= size:
        raise ValueError('A nonempty bounded original batch is required')
    return [*batch, *([batch[-1]] * (size-len(batch)))], len(batch)


def prepare(original, output, data_dir, ordinary_manifest, calibration_dir):
    import torch
    from transformers import AutoTokenizer
    from train.document_evidence_plan_evaluate import SCHEMA, sources as generation_sources
    from train.document_qa_span_scores import score_spans
    from train.evaluate_local import setup_model

    original, output, data_dir, ordinary_manifest, calibration_dir = (Path(p).resolve()
        for p in (original, output, data_dir, ordinary_manifest, calibration_dir))
    p = read(original/'protocol.json')
    if (p['schema'] != SCHEMA or p['purpose'] != 'FIXED_HELD_EVIDENCE_PLAN_V23'
            or p['items'] != 255 or p['shards'] != 4 or 'selection' not in p['modes']
            or p['source_files'] != generation_sources()
            or not p['release_file'] or sha256_file(p['release_file']) != p['release_sha256']):
        raise ValueError('Closed, source-bound released selection generation is required')
    release = read(p['release_file'])
    if (sha256_file(data_dir/'manifest.json') != p['data_manifest_sha256']
            or release['data_manifest_sha256'] != p['data_manifest_sha256']
            or sha256_file(ordinary_manifest) != release['ordinary_manifest_sha256']):
        raise ValueError('Frozen scientific/ordinary data binding differs')
    notes, origins = {}, {}
    for shard in range(4):
        directory = original/'answers'/f'shard-{shard}'
        report = read(directory/'run.json')
        if (report['status'] != 'complete_all_attempts'
                or report['protocol_sha256'] != sha256_file(original/'protocol.json')
                or report['attempts_sha256'] != sha256_file(directory/'attempts.jsonl')):
            raise ValueError('All original shards must close before replay preparation')
        origins[str(directory/'attempts.jsonl')] = report['attempts_sha256']
        for record in rows(directory/'attempts.jsonl'):
            if record['mode'] == 'selection':
                if record['item_id'] in notes:
                    raise ValueError('Duplicate native selection attempt')
                notes[record['item_id']] = record
    data = read(data_dir/'manifest.json')
    for name in ('held_questions', 'held_gold', 'training'):
        if sha256_file(data['files'][name]['path']) != data['files'][name]['sha256']:
            raise ValueError('Frozen data file changed')
    questions = rows(data['files']['held_questions']['path'])
    gold = {r['item_id']: r for r in rows(data['files']['held_gold']['path'])}
    if ([q['item_id'] for q in questions] != p['ordered_item_ids']
            or set(notes) != set(gold) or len(notes) != 255):
        raise ValueError('Every original question, including failed selections, is required')
    tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
    output.mkdir(parents=True, exist_ok=False)
    files, inventories = {}, {}

    def save_examples(name, examples):
        path = output/name
        path.write_text(''.join(json.dumps(row)+'\n' for row in examples))
        files[name] = sha256_file(path)
        inventories[name] = dict(rows=len(examples), targets=sum(len(r['target_ids']) for r in examples))

    for mode in MODES:
        examples = []
        for index, question in enumerate(questions):
            key = question['item_id']
            indices, donor, flag = None, None, None
            if mode == 'oracle':
                indices = gold[key]['support_indices']
            elif mode in ('selected_id_replay', 'other_id_replay'):
                donor_index = index if mode == 'selected_id_replay' else (index+1) % len(questions)
                donor_question = questions[donor_index]
                donor = donor_question['item_id']
                indices, flag = selected_ids(notes[donor], len(donor_question['contexts']))
                if any(i > len(question['contexts']) for i in indices):
                    indices, flag = [], dict(fallback=True, reason='donor_ids_outside_recipient_range')
            example = scoring_example(question, gold[key]['answer'], mode, tok, indices)
            example.update(donor_item_id=donor, selection_replay_status=flag)
            examples.append(example)
        save_examples(f'QA-{mode}.jsonl', examples)
    ordinary = read(ordinary_manifest)
    if sha256_file(ordinary['data_file']) != ordinary['data_sha256']:
        raise ValueError('Frozen ordinary continuation file changed')
    examples = [dict(item_id=r['item_id'], panel='ordinary', mode='raw',
                    context_ids=r['prefix_ids'], target_ids=r['target_ids']) for r in rows(ordinary['data_file'])]
    if len(examples) != 256 or any(len(r['context_ids']) != 256 or len(r['target_ids']) != 32 for r in examples):
        raise ValueError('The existing complete ordinary256 panel is required')
    save_examples('ordinary-raw.jsonl', examples)
    # Reuse the unchanged qualified scoring kernel/shapes, not old model logits.
    qualification = read(calibration_dir/'protocol.json')
    closure = read(calibration_dir/'closure.json')
    completed = read(calibration_dir/'workload-completion.json')
    if (not closure['physical14_empty']
            or completed['status'] != 'ALL_TRAIN_ONLY_SPAN_CPU_PARITY_CHECKS_PASS_PENDING_PHYSICAL_CLOSURE'
            or sha256_file(calibration_dir/'examples.json') != qualification['examples_sha256']):
        raise ValueError('Actual original scoring-shape qualification is required')
    for name in ('train/document_qa_span_scores.py', 'train/neuron_backend.py',
                 'train/evaluate_local.py', 'weekend_thinking/pretrained.py', 'train/runtime_compat.py'):
        if sha256_file(ROOT/name) != qualification['source_files'][name]:
            raise ValueError('Qualified scoring implementation changed')
    cal = read(calibration_dir/'examples.json')
    training_ids = {r['item_id'] for r in rows(data['files']['training']['path'])}
    if any(r['item_id'] not in training_ids for r in cal['QA']):
        raise ValueError('Own-model calibration must use existing TRAIN examples')
    torch.set_num_threads(16)
    model, checkpoint = p['model'], p['model']['checkpoint']
    if checkpoint and sha256_file(checkpoint) != model['checkpoint_sha256']:
        raise ValueError('Original terminal model changed')
    backend, _, loaded, signature, _ = setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),
        checkpoint=Path(checkpoint) if checkpoint else None, device='cpu'))
    if signature != p['signature'] or checkpoint and any(loaded[k] != model[k] for k in ('run_id', 'step', 'spec_hash')):
        raise ValueError('Loaded own-model identity differs')
    refs = {}
    for label, records in cal.items():
        refs[label] = score_spans(backend, [r['context_ids'] for r in records],
            [r['target_ids'] for r in records], width=3584 if label == 'QA' else 672)
    (output/'calibration-inputs.json').write_bytes((calibration_dir/'examples.json').read_bytes())
    atomic_json(output/'calibration-CPU-reference.json', dict(signature=signature, values=refs))
    files.update({name: sha256_file(output/name) for name in ('calibration-inputs.json', 'calibration-CPU-reference.json')})
    protocol = dict(schema='document-evidence-plan-v23-pointer-replay-likelihood', created_utc=utc_now(),
        model=model, model_dir=p['model_dir'], signature=signature, shards=4, batch=4, target_slots=32,
        widths=dict(QA=3584, ordinary=672), source_files=sources(), files=files, inventories=inventories,
        original_generation_protocol_sha256=sha256_file(original/'protocol.json'), original_selection_attempts=origins,
        release_file=p['release_file'], release_sha256=p['release_sha256'],
        interpretation='Canonical pointer-ID replay,invalid plans explicitly empty. Not native-generation likelihood or calibration. All original questions retained;synthetic physical padding never counted.')
    atomic_json(output/'protocol.json', protocol)
    return protocol


def calibration_check(actual, reference):
    """Keep the existing token and mean-span bounds, including finite checks."""
    aligned = (len(actual) == len(reference) == 4
        and all(a['tokens'] == b['tokens'] == len(a['token_logps']) == len(b['token_logps'])
                for a, b in zip(actual, reference)))
    if not aligned:
        return dict(passed=False, reason='native_target_alignment_differs')
    differences = [abs(x-y) for a, b in zip(actual, reference)
                   for x, y in zip(a['token_logps'], b['token_logps'])]
    spans = [a['full_logp']-b['full_logp'] for a, b in zip(actual, reference)]
    if not differences or not all(math.isfinite(v) for v in [*differences, *spans]):
        return dict(passed=False, reason='nonfinite_calibration')
    maximum, mean = max(differences), abs(sum(spans)/4)
    return dict(max_token_error=maximum, absolute_mean_span_error=mean,
                passed=maximum <= .001 and mean <= .001)


def score_original_batch(backend, batch, width, scorer):
    """Return one observation per original row, never per physical padding row."""
    execution, originals = physical_batch(batch)
    values = scorer(backend, [r['context_ids'] for r in execution],
        [r['target_ids'] for r in execution], width=width)
    if len(values) != 4:
        raise RuntimeError('Physical scoring batch returned an incomplete result')
    return values[:originals], 4-originals


def run(output, shard, device):
    from train.document_qa_span_scores import score_spans
    from train.evaluate_local import setup_model

    output = Path(output).resolve()
    p = read(output/'protocol.json')
    if (p['schema'] != 'document-evidence-plan-v23-pointer-replay-likelihood'
            or p['source_files'] != sources() or type(shard) is not int or not 0 <= shard < 4
            or p['shards'] != 4 or p['batch'] != 4 or p['target_slots'] != 32
            or p['widths'] != dict(QA=3584, ordinary=672)
            or sha256_file(p['release_file']) != p['release_sha256']):
        raise ValueError('Frozen released scoring protocol and shapes required')
    for name, expected in p['files'].items():
        if sha256_file(output/name) != expected:
            raise ValueError('Frozen likelihood file changed: '+name)
    model, path = p['model'], p['model']['checkpoint']
    if path and sha256_file(path) != model['checkpoint_sha256']:
        raise ValueError('Original terminal model changed')
    backend, _, loaded, signature, _ = setup_model(SimpleNamespace(
        model_dir=Path(p['model_dir']), checkpoint=Path(path) if path else None, device=device))
    if signature != p['signature'] or path and any(loaded[k] != model[k] for k in ('run_id', 'step', 'spec_hash')):
        raise ValueError('Loaded own-model identity differs')
    directory = output/'scores'/f'shard-{shard}'
    directory.mkdir(parents=True, exist_ok=False)
    checks, references = {}, read(output/'calibration-CPU-reference.json')['values']
    for panel, records in read(output/'calibration-inputs.json').items():
        try:
            actual = score_spans(backend, [r['context_ids'] for r in records],
                [r['target_ids'] for r in records], width=p['widths'][panel])
            checks[panel] = dict(actual=actual, **calibration_check(actual, references[panel]))
        except Exception as error:
            checks[panel] = dict(passed=False, error=f'{type(error).__name__}: {error}')
        # Persist the original failed witness before raising; never retry silently.
        atomic_json(directory/'own-model-TRAIN-calibration.json', checks)
        if not checks[panel]['passed']:
            raise RuntimeError('Own-model TRAIN calibration failed before held scoring; preserve this attempt')
    count, errors, padding, expected = 0, 0, 0, 0
    started = time.monotonic()
    with (directory/'scores.jsonl').open('x') as stream:
        for name, inventory in p['inventories'].items():
            original = rows(output/name)
            if len(original) != inventory['rows']:
                raise ValueError('Frozen original inventory differs')
            panel = 'QA' if name.startswith('QA-') else 'ordinary'
            selected = original[shard::4]
            expected += len(selected)
            for index in range(0, len(selected), 4):
                batch = selected[index:index+4]
                before, exception = time.monotonic(), None
                padding += 4-len(batch)
                try:
                    values, _ = score_original_batch(backend, batch, p['widths'][panel], score_spans)
                except Exception as error:
                    values = [None]*len(batch)
                    exception = f'{type(error).__name__}: {error}'
                seconds = time.monotonic()-before
                for row, value in zip(batch, values):
                    result = {k: row.get(k) for k in ('item_id', 'panel', 'mode',
                        'donor_item_id', 'selection_replay_status')}
                    result.update(status='error' if exception else 'ok',
                        context_tokens=len(row['context_ids']), target_ids=row['target_ids'],
                        batch_seconds=seconds, synthetic_padding_observation=False)
                    if exception:
                        result['error'] = exception
                        errors += 1
                    else:
                        result.update(value)
                        if panel == 'QA':
                            result.update(answer_content_logp=sum(value['token_logps'][:-1]),
                                          EOS_logp=value['token_logps'][-1])
                    stream.write(json.dumps(result, allow_nan=False)+'\n')
                    count += 1
                stream.flush()
                print(json.dumps(dict(shard=shard, panel=panel, original_rows=count,
                    execution_errors=errors, synthetic_execution_rows=padding)), flush=True)
    atomic_json(directory/'completion.json', dict(
        status='COMPLETE_ALL_POINTER_REPLAY_LIKELIHOOD_ATTEMPTS_NOT_READOUT',
        rows=count, expected_original_rows=expected, execution_errors=errors,
        synthetic_execution_rows=padding, wall_seconds=time.monotonic()-started,
        device=device, source_protocol_sha256=sha256_file(output/'protocol.json'),
        scores_sha256=sha256_file(directory/'scores.jsonl')))
    if count != expected or errors:
        raise RuntimeError('Incomplete or failed scoring; preserve all rows, no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    for name in ('original', 'data-dir', 'ordinary-manifest', 'calibration-dir'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--device', choices=('cpu', 'xla'), default='xla')
    args = parser.parse_args()
    if args.command == 'prepare':
        paths = (args.original, args.data_dir, args.ordinary_manifest, args.calibration_dir)
        if any(path is None for path in paths):
            parser.error('Explicit closed generation, data, ordinary and TRAIN calibration paths required')
        prepare(args.original, args.output, args.data_dir, args.ordinary_manifest, args.calibration_dir)
    else:
        if args.shard is None:
            parser.error('--shard required')
        run(args.output, args.shard, args.device)


if __name__ == '__main__':
    main()
