"""Fixed real-document scalar continuation, separately generated and scored.

Generation opens only native contexts and model references. Gold and the
unchanged conservative scalar parser are used only by the score command.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from train.real_trace_contexts import reader_context
from train.real_trace_scope import panel_scope,protocol_items
from train.real_trace_notes import read, rows, sources as note_sources
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now, require_utc_deadline

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'real-trace-native-scalar-secondary-v10'
MODES = ('raw', 'empty', 'oracle', 'own', 'other')


def sources():
    return {**note_sources(), **{name: sha256_file(ROOT/name) for name in
            ('train/real_trace_answers.py', 'train/scalar_completion.py')}}


def prepare(output, origin, inputs, gold):
    """Reuse exact note/model/native evidence; never sample another note."""
    from transformers import AutoTokenizer
    from train.partial_state_evaluation import donor_permutation
    prior = read(origin/'protocol.json')
    bundle = read(origin/'notes-bundle.json')
    panel = prior['panel']
    schema, items = panel_scope(panel,'scalar')
    manifest = read(inputs/'manifest.json')
    if (bundle['status'] != 'COMPLETE_REAL_PREFIX_ONLY_NOTE_BUNDLE'
            or bundle['model_id'] != prior['model']['id'] or prior['source_files'] != note_sources()):
        raise ValueError('Complete fixed-model arithmetic notes required')
    for name, digest in bundle['source_pins'].items():
        if sha256_file(name) != digest:
            raise ValueError('Original note evidence changed')
    questions = inputs/f'{panel}-questions.jsonl'
    oracle = inputs/f'{panel}-controls.jsonl'
    for file in (questions, oracle):
        if sha256_file(file) != manifest['files'][file.name]['sha256']:
            raise ValueError('Fixed question/oracle inputs changed')
    qs, cs = rows(questions), rows(oracle)
    ids = [q['item_id'] for q in qs]
    if (len(qs) != items or len(set(ids)) != items
            or len({q['document_id'] for q in qs}) != items
            or ids != prior['ordered_item_ids'] or ids != bundle['ordered_item_ids']
            or [c['item_id'] for c in cs] != ids
            or [c['item_id'] for c in bundle['controls']] != ids):
        raise ValueError('All original ordered documents required')
    originals = {v['item_id']: v for name in bundle['source_pins']
                 if Path(name).name == 'notes.jsonl' for v in rows(name)}
    if set(originals) != set(ids):
        raise ValueError('Every original note attempt required')
    donors = donor_permutation([len(originals[k]['content_ids']) for k in ids], ids)
    tok = AutoTokenizer.from_pretrained(prior['model_dir'], local_files_only=True)
    projected = []
    for q, supplied, notes, donor in zip(qs, cs, bundle['controls'], donors):
        if (notes['own'] != originals[q['item_id']] or notes['other'] != originals[ids[donor]]
                or notes['other_item_id'] != ids[donor]):
            raise ValueError('Own/other notes must remain native and intact')
        choices = dict(raw=None, empty=[], oracle=supplied['oracle_ids'],
                       own=notes['own']['content_ids'], other=notes['other']['content_ids'])
        contexts = {m: reader_context(q, tok, choices[m]) for m in MODES}
        if any(not ctx or len(ctx) > 672 for ctx in contexts.values()):
            raise ValueError('No truncation of native reader context')
        projected.append(dict(item_id=q['item_id'], document_id=q['document_id'], contexts=contexts,
                              note_status={m: notes[m]['status'] for m in ('own', 'other')}))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'contexts.jsonl').open('x') as stream:
        for value in projected:
            stream.write(json.dumps(value, separators=(',', ':'))+'\n')
    for name, digest in prior['native_files'].items():
        if sha256_file(origin/name) != digest:
            raise ValueError('Actual own-model CPU native reference changed')
        shutil.copyfile(origin/name, output/name)
    p = dict(schema=schema, created_utc=utc_now(), items=items, shards=4, panel=panel,
             model=prior['model'], model_dir=prior['model_dir'], assets=prior['assets'],
             signature=prior['signature'], runtime_policy=prior['runtime_policy'],
             source_files=sources(), modes=list(MODES), max_actions=32, prefill_width=672,
             stop_ids=[151643], ordered_item_ids=ids, gold_sha256=sha256_file(gold),
             files={name: sha256_file(output/name) for name in
                    ('contexts.jsonl', 'native-contexts.json', 'native-reference.pt')},
             reused_origin=str(origin), origin_protocol_sha256=sha256_file(origin/'protocol.json'),
             original_notes_sha256=sha256_file(origin/'notes-bundle.json'),
             limits=['Initial complete scalar at a curated continuation boundary, not general QA.',
                     'Generation never opens gold. Existing initial-scalar-v2 is scoring-only.',
                     'Original native notes and CPU references are reused, not new independent observations.'])
    atomic_json(output/'protocol.json', p)
    return p


def load(output):
    p = read(output/'protocol.json')
    items = protocol_items(p,'scalar')
    if (p['shards'] != 4
            or p['modes'] != list(MODES) or p['max_actions'] != 32 or p['prefill_width'] != 672
            or p['stop_ids'] != [151643] or p['source_files'] != sources()
            or set(p['files']) != {'contexts.jsonl', 'native-contexts.json', 'native-reference.pt'}):
        raise ValueError('Exact fixed secondary decoding identity required')
    # No gold file is opened or hashed by the generation loader.
    for name, digest in p['files'].items():
        if sha256_file(output/name) != digest:
            raise ValueError('Frozen native contexts/references changed')
    origin = Path(p['reused_origin'])
    if (sha256_file(origin/'protocol.json') != p['origin_protocol_sha256']
            or sha256_file(origin/'notes-bundle.json') != p['original_notes_sha256']
            or read(origin/'protocol.json')['model'] != p['model']):
        raise ValueError('Exact original note/model binding changed')
    values = rows(output/'contexts.jsonl')
    if (len(values) != items or [v['item_id'] for v in values] != p['ordered_item_ids']
            or len({v['item_id'] for v in values}) != items
            or len({v['document_id'] for v in values}) != items
            or any(set(v['contexts']) != set(MODES) or set(v['note_status']) != {'own', 'other'}
                   or any(not ctx or len(ctx) > 672 or any(type(t) is not int or t < 0 or t == 151643 for t in ctx)
                          for ctx in v['contexts'].values()) for v in values)):
        raise ValueError('Complete original context inventory required')
    return p, values


def generate(output, shard, until):
    import torch
    from train.evaluate import GreedyDecoder, GenerationFailure, asset_provenance
    from train.evaluate_local import setup_model
    from train.partial_state_answer_driver import cache_prefill_check
    from train.useful_note_precision import compare_distributions
    from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy
    p, contexts = load(output)
    deadline = require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite scalar-generation deadline; no retry')
    check()
    if not 0 <= shard < 4:
        raise ValueError('Explicit four-shard generation required')
    require_runtime_policy(p['runtime_policy'], runtime_policy())
    if asset_provenance(Path(p['model_dir'])) != p['assets']:
        raise ValueError('Released model assets changed')
    model = p['model']
    if model['checkpoint'] and sha256_file(model['checkpoint']) != model['checkpoint_sha256']:
        raise ValueError('Frozen model changed')
    d = output/'answers'/f'shard-{shard}'
    d.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', started_utc=utc_now(), rows=0, shard=shard,
                  protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json', report)
    started = time.monotonic()
    try:
        backend, tok, loaded, signature, _ = setup_model(SimpleNamespace(
            model_dir=Path(p['model_dir']), checkpoint=Path(model['checkpoint']) if model['checkpoint'] else None,
            device='xla'))
        if signature != p['signature']:
            raise ValueError('Transferred model differs from its own CPU reference')
        if model['checkpoint'] and (loaded['sha256'] != model['checkpoint_sha256']
                or any(loaded[k] != model[k] for k in ('run_id', 'step', 'spec_hash'))):
            raise ValueError('Actual generation checkpoint differs')
        refs = read(output/'native-contexts.json')['refs']
        reference = torch.load(output/'native-reference.pt', map_location='cpu', weights_only=True)
        context = refs[shard]['context_ids']
        x = torch.full((1, 1024), 151643, dtype=torch.long)
        x[0, :len(context)] = torch.tensor(context)
        with torch.no_grad():
            hidden = backend.hidden(x.to(backend.device))
            lp = backend.model.lm_head(hidden[:, len(context)-1]).float().log_softmax(-1)
            backend.sync()
            actual = lp.cpu()
        native = compare_distributions(reference[shard:shard+1], actual)[0]
        torch.save(actual, d/'native-logps.pt')
        atomic_json(d/'native.json', native)
        if not native['passed']:
            raise RuntimeError('Actual own-model native parity failed')
        del x, hidden, lp, actual
        decoder = GreedyDecoder(backend.model, tok, sync=backend.sync, banned_ids=backend.banned_ids)
        audit, vectors = cache_prefill_check(backend, decoder, contexts[shard]['contexts']['own'], width=672, cap=32)
        torch.save(vectors, d/'cache-prefill-logps.pt')
        atomic_json(d/'cache-prefill.json', audit)
        if audit['max_logp_error'] > 1e-3:
            raise RuntimeError('Actual cached scalar shape differs from causal forward')
        with (d/'answers.jsonl').open('x') as stream:
            for i in range(shard, p['items'], 4):
                q = contexts[i]
                for mode in MODES:
                    check()
                    prompt = q['contexts'][mode]
                    before = time.monotonic()
                    value = dict(item_id=q['item_id'], document_id=q['document_id'], mode=mode,
                                 status='error', actions=[], content_ids=[], text='', raw_text='',
                                 stop_type='error', prompt_ids=prompt,
                                 source_note_status=q['note_status'].get(mode, 'ok'))
                    try:
                        value.update(asdict(decoder.decode(prompt, max_new_tokens=32, prefill_width=672,
                                     pad_id=151643, stop_ids=(151643,), answer_line_stop=False)), status='ok')
                    except Exception as error:
                        if isinstance(error, GenerationFailure):
                            value.update(asdict(error.partial))
                        value['error'] = f'{type(error).__name__}: {error}'
                    value.update(n_actions=len(value['actions']), wall_seconds=time.monotonic()-before)
                    stream.write(json.dumps(value, allow_nan=False)+'\n')
                    stream.flush()
                    report['rows'] += 1
                print(json.dumps(dict(utc=utc_now(), shard=shard, rows=report['rows'])), flush=True)
        report.update(status='complete_all_attempts', answers_sha256=sha256_file(d/'answers.jsonl'))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(finished_utc=utc_now(), wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json', report, overwrite=True)


def classify(value, expected):
    """The fixed parser sees emitted text and stop only, never expected value."""
    from train.scalar_completion import parse_initial_scalar_v2
    parsed = parse_initial_scalar_v2(value['text'], value['stop_type'])
    execution_error = value['status'] != 'ok' or value['source_note_status'] != 'ok'
    correct = (not execution_error and parsed['status'] == 'valid'
               and Fraction(parsed['value']) == Fraction(expected))
    return dict(parsed=parsed, correct=bool(correct),
                category='execution_error' if execution_error else parsed['status'])


def score(output, gold):
    from transformers import AutoTokenizer
    p, contexts = load(output)
    if read(output/'generation-closure.json')['status'] != 'ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED':
        raise ValueError('All generation processes must physically close first')
    if sha256_file(gold) != p['gold_sha256']:
        raise ValueError('Frozen scoring-only gold changed')
    gold_rows = rows(gold)
    if [g['item_id'] for g in gold_rows] != p['ordered_item_ids']:
        raise ValueError('Complete fixed scoring-only gold required')
    expected = {g['item_id']: g['numeric_value'] for g in gold_rows}
    tokenizer = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True)
    defined = set(tokenizer.get_vocab().values())
    observed = []
    for shard in range(4):
        d = output/'answers'/f'shard-{shard}'
        run = read(d/'run.json')
        values = rows(d/'answers.jsonl')
        order = [(q['item_id'], mode) for q in contexts[shard::4] for mode in MODES]
        expected_rows = p['items']//4 * len(MODES)
        if (run['status'] != 'complete_all_attempts' or run['rows'] != expected_rows or len(values) != expected_rows
                or run['protocol_sha256'] != sha256_file(output/'protocol.json')
                or sha256_file(d/'answers.jsonl') != run['answers_sha256']
                or [(v['item_id'], v['mode']) for v in values] != order
                or not read(d/'native.json')['passed']
                or read(d/'cache-prefill.json')['max_logp_error'] > 1e-3):
            raise ValueError('Complete original attempts and hardware checks required')
        by_id = {q['item_id']: q for q in contexts[shard::4]}
        for value in values:
            q, mode, actions = by_id[value['item_id']], value['mode'], value['actions']
            stopped = bool(actions and actions[-1] == 151643)
            if (value['document_id'] != q['document_id'] or value['prompt_ids'] != q['contexts'][mode]
                    or value['source_note_status'] != q['note_status'].get(mode, 'ok')
                    or value['n_actions'] != len(actions) or len(actions) > 32
                    or any(type(t) is not int or t not in defined for t in actions)
                    or 151643 in actions[:-1] or value['content_ids'] != (actions[:-1] if stopped else actions)
                    or value['status'] not in ('ok', 'error')):
                raise ValueError('Native continuation attempt changed')
            if value['status'] == 'ok' and (value['stop_type'] != ('token:151643' if stopped else 'length_cap')
                    or not stopped and len(actions) != 32):
                raise ValueError('No parser-dependent early stopping permitted')
            if value['text'] != tokenizer.decode(value['content_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False):
                raise ValueError('Saved text differs from native continuation')
            observed.append({**value, **classify(value, expected[value['item_id']])})
    with (output/'scored.jsonl').open('x') as stream:
        for value in observed:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
    modes = {}
    for mode in MODES:
        selected = [v for v in observed if v['mode'] == mode]
        assert len(selected) == len({v['document_id'] for v in selected}) == p['items']
        modes[mode] = dict(documents=p['items'], correct=sum(v['correct'] for v in selected),
                           categories=dict(Counter(v['category'] for v in selected)),
                           capped=sum(v['stop_type'] == 'length_cap' for v in selected),
                           generated_actions=sum(v['n_actions'] for v in selected),
                           wall_seconds_sum=sum(v['wall_seconds'] for v in selected))
    summary = dict(status='COMPLETE_ALL_FIXED_SCALAR_CONTINUATIONS', model_id=p['model']['id'], panel=p['panel'],
                   rows=len(observed), modes=modes, scored_sha256=sha256_file(output/'scored.jsonl'),
                   protocol_sha256=sha256_file(output/'protocol.json'),
                   limits=p['limits'] + [f"All failures remain in{p['items']} denominators. No gate substitution."])
    atomic_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'score'))
    for name in ('output', 'origin', 'inputs', 'gold'):
        parser.add_argument('--'+name, type=Path, required=name == 'output')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--deadline-utc')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output, args.origin, args.inputs, args.gold)
    elif args.command == 'generate':
        result = generate(args.output, args.shard, args.deadline_utc)
    else:
        result = score(args.output, args.gold)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
