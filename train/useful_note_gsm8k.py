"""Four model jobs, each running all three modes on its fixed TRAIN512 shard.

Generation uses only question projection, worked-example prompt and format.
Gold is opened by summarize only after all1536 attempted paths are complete.
This is development, with known outcomes on the inherited first128 questions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from train import evaluate as evaluator
from train import useful_note_evaluation as shared
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format
from weekend_thinking.evaluation import score_predictions
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'useful-note-gsm8k-train512-v1'
PANEL_SHA256 = 'c8cadad08a84bc813566085ad0d7b1d5d1f289f22287683b8f2bae1002c12231'
N, SHARDS, THOUGHT_CAP, ANSWER_CAP = 512, 4, 128, 512
MODES = ('direct', 'empty_format', 'one_thought')
MODEL_IDS = ('B0', 'W', 'Cprime', 'T')
STATUSES = ('ok', 'truncated', 'error', 'format_error')


def sources():
    return {**shared.sources(), 'train/useful_note_gsm8k.py': sha256_file(__file__)}


def validate_identity(identity):
    """No legacy B0/vanilla/thought alias can describe a new trained arm."""
    key = identity.get('id')
    if key not in MODEL_IDS:
        raise ValueError('Require actual B0/W/Cprime/T model identity, never legacy arm aliases')
    if key == 'B0':
        if (identity.get('checkpoint') is not None or identity.get('checkpoint_sha256') is not None
                or identity.get('loaded_checkpoint') is not None):
            raise ValueError('Only unchanged released B0 has no checkpoint')
    elif (not identity.get('checkpoint') or not identity.get('checkpoint_sha256')
            or Path(identity['checkpoint']).name != 'model.pt'
            or len(identity['checkpoint_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in identity['checkpoint_sha256'])
            or not identity.get('provenance') or not identity.get('checkpoint_files')
            or not identity.get('training_source_files') or not identity.get('run_id')
            or not identity.get('spec_hash') or type(identity.get('step')) is not int or identity['step'] < 1):
        raise ValueError('Actual completed science checkpoint/source/full-shard identity required')
    return identity


def resolve_identity(model_id, checkpoint=None, checkpoint_sha256=None, complete_sha256=None, training_protocol=None):
    if model_id == 'B0':
        if any(v is not None for v in (checkpoint, checkpoint_sha256, complete_sha256, training_protocol)):
            raise ValueError('B0 cannot disguise a trained checkpoint')
        identity = dict(id='B0', checkpoint=None, checkpoint_sha256=None)
    else:
        if model_id not in MODEL_IDS or any(v is None for v in (checkpoint, checkpoint_sha256, complete_sha256, training_protocol)):
            raise ValueError('Explicit actual trained model and all terminal lineage pins required')
        # Root updates this shared validator for final science schema/arms only
        # after the active held baseline closes. No local gate bypass is used.
        identity = shared.trained_identity(checkpoint, checkpoint_sha256, complete_sha256, training_protocol)
        if identity['id'] != model_id:
            raise ValueError('Actual checkpoint arm differs from requested model ID')
    return validate_identity(identity)


def validate_format(tokenizer, data):
    fmt = Format.from_dict(data['format'])
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if (list(fmt.header_ids) or fmt.pad_id != 151643 or fmt.stop_id != 151668
            or tuple(fmt.additional_stop_ids) != (151643,) or tokenizer.eos_token_id != 151643
            or decode(fmt.open_ids) != '\n<think>' or decode(fmt.close_ids) != '</think>\n'):
        raise ValueError('Exact native no-header first-close/EOS format required')
    return fmt


def measure_widths(questions, prompt, fmt, tokenizer):
    """Common answer shape for direct/empty/own; no current answers are inputs."""
    lengths = [len(tokenizer.encode(evaluator.question_prompt(q['question'], prompt), add_special_tokens=False))
               for q in questions]
    suffixes = [len(tokenizer.encode(evaluator.reader_suffix(q['question']), add_special_tokens=False))
                for q in questions]
    if not lengths or min(lengths) < 1:
        raise ValueError('Nonempty native question prompts required')
    width = max(lengths)
    answer = width + len(fmt.header_ids) + len(fmt.open_ids) + THOUGHT_CAP + len(fmt.close_ids) + max(suffixes)
    return dict(question_prefill_width=width, answer_prefill_width=answer,
        question_prompt_lengths=dict(min=min(lengths), max=max(lengths), total=sum(lengths)),
        reader_suffix_lengths=dict(min=min(suffixes), max=max(suffixes), total=sum(suffixes)))


def prepare(output, panel, model_dir, native_reference, until, model_id, **checkpoint_args):
    from transformers import AutoTokenizer
    from train.interface_oracle_probe import load as load_native, reference
    shared.check(until); output = Path(output); panel = Path(panel); model_dir = Path(model_dir)
    if sha256_file(panel/'manifest.json') != PANEL_SHA256 or output.exists():
        raise ValueError('Exact closed TRAIN512 panel and fresh output required')
    panel_manifest = shared.read(panel/'manifest.json')
    if (panel_manifest['schema'] != 'gsm8k-train512-development-v1'
            or panel_manifest['thought_cap'] != THOUGHT_CAP or panel_manifest['max_answer_tokens'] != ANSWER_CAP
            or panel_manifest['compute'] != 'fp32' or panel_manifest['head_precision'] != 'fp32'
            or panel_manifest['parser'] != 'Existing strict_final_hash4_number_v1; no recovered-number accuracy.'):
        raise ValueError('Unchanged prepared panel/caps/FP32/strict parser required')
    paths = [panel/'manifest.json']
    for name in ('questions.jsonl','prompt.json','format.json','selection.json'):
        path = panel/name
        shared.verify_pins({str(path): panel_manifest['output_hashes'][name]}); paths.append(path)
    questions = evaluator.read_questions(panel/'questions.jsonl', qualification=True)
    ids = [q['item_id'] for q in questions]; selection = shared.read(panel/'selection.json')
    if (len(ids) != N or len(set(q['document_id'] for q in questions)) != N
            or ids != selection['ordered_item_ids'] or ids != panel_manifest['selection']['ordered_item_ids']
            or selection['ordered_item_ids_sha256'] != canonical_hash(ids)):
        raise ValueError('Complete fixed ordered512 TRAIN questions, no selection/filtering')
    prompt = shared.read(panel/'prompt.json'); evaluator.validate_prompt_spec(prompt, set(ids))
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    fmt_data = shared.read(panel/'format.json'); fmt = validate_format(tok, fmt_data)
    widths = measure_widths(questions, prompt, fmt, tok)
    config = shared.read(model_dir/'config.json')
    if widths['answer_prefill_width']+ANSWER_CAP > config['max_position_embeddings']:
        raise ValueError('Common answer prefill/continuation exceeds released context')
    assets = evaluator.asset_provenance(model_dir)
    old, native = load_native(Path(native_reference)); ref, _ = reference(Path(native_reference), old, native)
    if old['assets'] != assets or len(native['qualification_refs']) < SHARDS:
        raise ValueError('Actual released B0 native FP32 reference required')
    identity = resolve_identity(model_id, **checkpoint_args)
    shared.check(until); output.mkdir(parents=True, exist_ok=False)
    # All three projections stay exact. Generation never opens gold.jsonl.
    for name in ('questions.jsonl','prompt.json','format.json'):
        (output/name).write_bytes((panel/name).read_bytes())
    protocol = dict(schema=SCHEMA, created_utc=utc_now(), purpose='Nonfinal GSM8K TRAIN development; official TEST unused',
        items=N, shards=SHARDS, modes=list(MODES), model=identity, model_dir=str(model_dir.resolve()),
        assets=assets, model_manifest_sha256=assets['manifest_sha256'], base_signature=ref['signature'],
        native_reference=str(Path(native_reference).resolve()),
        native_inputs=shared.pin([Path(native_reference)/'protocol.json',Path(native_reference)/'cpu-reference/reference.json',
                                 Path(native_reference)/'cpu-reference/next-logps.pt']),
        source_files=sources(), input_hashes=shared.pin(paths),
        projection_hashes={name:sha256_file(output/name) for name in ('questions.jsonl','prompt.json','format.json')},
        ordered_item_ids=ids, ordered_document_ids=[q['document_id'] for q in questions],
        gold_file=str((panel/'gold.jsonl').resolve()), gold_sha256=panel_manifest['output_hashes']['gold.jsonl'],
        compute='fp32', head_precision='fp32', selection_policy='single_greedy', support='Tokenizer-defined IDs only',
        thought_cap=THOUGHT_CAP, max_answer_tokens=ANSWER_CAP, answer_stop=evaluator.ANSWER_STOP,
        parser='strict_final_hash4_number_v1', **widths, bootstrap=dict(repeats=5000, seed=17), deadline_utc=until,
        limitations=['First128 TRAIN questions have known prior development outcomes; this is not blind or official TEST.',
            'Direct/empty/one-thought share one model load per question shard; decoding itself reuses the sequential qualified decoder.',
            'Thought128 is TOTAL sampled actions including native close/EOS; all capped content is retained.',
            'Strict parser retains execution errors, truncations and format failures in every512 denominator.',
            'Conditional document-bootstrap intervals are not training-seed uncertainty.',
            'Complete module receipts are separate from external owned-process/physical closure evidence.'])
    atomic_json(output/'protocol.json', protocol); return protocol


def load(output):
    output = Path(output); p = shared.read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or p['items'] != N or p['shards'] != SHARDS
            or p['modes'] != list(MODES) or p['thought_cap'] != THOUGHT_CAP or p['max_answer_tokens'] != ANSWER_CAP
            or p['compute'] != 'fp32' or p['head_precision'] != 'fp32' or p['parser'] != 'strict_final_hash4_number_v1'):
        raise ValueError('Frozen GSM wrapper/policy/source changed')
    validate_identity(p['model']); shared.verify_pins(p['input_hashes']); shared.verify_pins(p['native_inputs'])
    shared.verify_pins({str(output/name):digest for name,digest in p['projection_hashes'].items()})
    questions = evaluator.read_questions(output/'questions.jsonl', qualification=True)
    if ([q['item_id'] for q in questions] != p['ordered_item_ids']
            or [q['document_id'] for q in questions] != p['ordered_document_ids'] or len(questions) != N):
        raise ValueError('Complete original question order/documents required')
    return p, questions


def generate(output, shard, until, device='xla'):
    import torch
    shared.check(until); output = Path(output); p, questions = load(output)
    if type(shard) is not int or shard not in range(SHARDS) or until != p['deadline_utc']:
        raise ValueError('Fixed shard/deadline required')
    d = output/'generation'/f'shard-{shard}'; d.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', rows=0, started_utc=utc_now(), shard=shard,
        protocol_sha256=sha256_file(output/'protocol.json'), model=p['model'])
    atomic_json(d/'run.json', report); started = time.monotonic()
    try:
        before = time.monotonic(); backend, tok, actual = shared.load_model(p, p['model'], device)
        validate_identity(actual); report['model'] = actual; report['model_setup_seconds'] = time.monotonic()-before
        if any(v.dtype != torch.float32 for v in backend.model.parameters()):
            raise ValueError('GSM inference requires actual FP32 parameters/head')
        before = time.monotonic(); report['gate'] = shared.base_gate(backend, p, shard, d, actual)
        report['native_gate_seconds'] = time.monotonic()-before
        fmt = validate_format(tok, shared.read(output/'format.json')); prompt = shared.read(output/'prompt.json')
        widths = measure_widths(questions, prompt, fmt, tok)
        if any(widths[k] != p[k] for k in widths):
            raise ValueError('Actual native common prompt shapes changed')
        decoder = evaluator.GreedyDecoder(backend.model, tok, sync=backend.sync, banned_ids=backend.banned_ids)
        local = evaluator.shard_questions(questions, shard, SHARDS)
        report['path_generation_started_utc'] = utc_now(); path_started = time.monotonic()
        with (d/'predictions.jsonl').open('x') as stream:
            for mode in MODES:
                for question in local:
                    shared.check(until)
                    row = evaluator.generate_item(question, decoder=decoder, prompt_spec=prompt, fmt=fmt,
                        mode=mode, max_answer_tokens=ANSWER_CAP, thought_cap=THOUGHT_CAP,
                        question_prefill_width=p['question_prefill_width'], answer_prefill_width=p['answer_prefill_width'])
                    row.update(protocol_sha256=report['protocol_sha256'], model_id=actual['id'],
                        model_identity_sha256=canonical_hash(actual), checkpoint_sha256=actual['checkpoint_sha256'])
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
                    report['rows'] += 1
                    print(json.dumps({k:row[k] for k in ('item_id','mode','status','n_generated_tokens','wall_seconds')}), flush=True)
        report.update(status='complete_attempts', path_generation_seconds=time.monotonic()-path_started,
            predictions_sha256=sha256_file(d/'predictions.jsonl'))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report.update(finished_utc=utc_now(), stage_wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json', report, overwrite=True)


def complete_predictions(output, p, questions):
    """All four whole jobs/all1536 paths before any gold read; errors count."""
    output = Path(output); values = []; reports = []; pins = {}; identity = None
    for shard in range(SHARDS):
        d = output/'generation'/f'shard-{shard}'; run = shared.read(d/'run.json'); rows = shared.rows(d/'predictions.jsonl')
        expected = [(q['item_id'],mode) for mode in MODES for q in evaluator.shard_questions(questions,shard,SHARDS)]
        if (run['status'] != 'complete_attempts' or run['shard'] != shard or run['rows'] != len(expected)
                or run['protocol_sha256'] != sha256_file(output/'protocol.json')
                or run['predictions_sha256'] != sha256_file(d/'predictions.jsonl')
                or [(r['item_id'],r['mode']) for r in rows] != expected):
            raise ValueError('All four complete original question/mode path shards required')
        validate_identity(run['model'])
        if any(run['model'].get(k) != v for k,v in p['model'].items()) or (identity is not None and run['model'] != identity):
            raise ValueError('Actual model identity differs across shards/protocol')
        identity = run['model']; known = {q['item_id']:q for q in questions}
        for row in rows:
            q = known[row['item_id']]
            if (row['document_id'] != q['document_id'] or row['split'] != 'train' or row['task'] != 'gsm8k'
                    or row['status'] not in STATUSES or row['selection_policy'] != 'single_greedy'
                    or row['protocol_sha256'] != run['protocol_sha256'] or row['model_id'] != identity['id']
                    or row['model_identity_sha256'] != canonical_hash(identity)
                    or row['checkpoint_sha256'] != identity['checkpoint_sha256']
                    or row['answer_cap'] != ANSWER_CAP or row['thought_cap'] != (THOUGHT_CAP if row['mode']=='one_thought' else 0)
                    or row['n_answer_tokens'] != len(row['answer_actions']) or len(row['answer_actions']) > ANSWER_CAP
                    or row['n_thought_tokens'] != len(row['thought_actions']) or len(row['thought_actions']) > THOUGHT_CAP
                    or row['n_generated_tokens'] != row['n_answer_tokens']+row['n_thought_tokens']
                    or row['mode'] != 'one_thought' and row['thought_actions']
                    or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0):
                raise ValueError('Original path identity/status/action-count contract changed')
        values.extend(rows); reports.append(run); pins.update(shared.pin([d/'run.json', d/'predictions.jsonl']))
    if len(values) != N*len(MODES) or len({(r['item_id'],r['mode']) for r in values}) != len(values):
        raise ValueError('No missing, duplicated or filtered paths')
    return values, reports, pins, identity


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return dict(total=float(a.sum()), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a,.9)), min=float(a.min()), max=float(a.max()))


def reduce_predictions(gold, predictions, item_ids):
    if ([r['item_id'] for r in gold] != item_ids or len(gold) != N
            or any(r.get('split') != 'train' for r in gold)):
        raise ValueError('Exact ordered TRAIN512 projected gold required')
    if len(predictions) != N*len(MODES) or len({(r['item_id'],r['mode']) for r in predictions}) != len(predictions):
        raise ValueError('All1536 distinct attempted paths required; no complete-case subset')
    scored = {}; absolute = {}
    for mode in MODES:
        group = [r for r in predictions if r['mode']==mode]
        values, result = score_predictions(gold, group); scored[mode] = values
        result.update(correct=sum(int(r['score']) for r in values),
            unsupported=sum(result['statuses'][s] for s in ('truncated','error','format_error')),
            thought_actions=sum(r['n_thought_tokens'] for r in group), answer_actions=sum(r['n_answer_tokens'] for r in group),
            thought_caps=sum(r.get('thought_stop_type')=='length_cap' for r in group),
            answer_caps=sum(r.get('answer_stop_type')=='length_cap' for r in group),
            wall_seconds=distribution([r['wall_seconds'] for r in group]),
            thought_action_distribution=distribution([r['n_thought_tokens'] for r in group]),
            answer_action_distribution=distribution([r['n_answer_tokens'] for r in group]))
        absolute[mode] = result
    contrasts = {f'thought_minus_{mode}': paired_cluster_bootstrap(scored[mode],scored['one_thought'],repeats=5000,seed=17)
                 for mode in ('empty_format','direct')}
    return absolute, contrasts, scored


def summarize(output):
    output = Path(output); p, questions = load(output)
    values, reports, pins, identity = complete_predictions(output,p,questions)  # Complete barrier BEFORE gold.
    shared.verify_pins({p['gold_file']:p['gold_sha256']})
    gold = shared.rows(p['gold_file'])
    absolute, contrasts, scored = reduce_predictions(gold,values,p['ordered_item_ids'])
    with (output/'scores.jsonl').open('x') as stream:
        for mode in MODES:
            for row in scored[mode]:
                stream.write(json.dumps(dict(row,mode=mode,model_id=identity['id'],model_identity_sha256=canonical_hash(identity)))+'\n')
    result = dict(schema=SCHEMA,status='complete_all_attempts',created_utc=utc_now(),purpose=p['purpose'],
        items=N,paths=N*len(MODES),model=identity,protocol_sha256=sha256_file(output/'protocol.json'),
        source_files=p['source_files'],input_hashes=p['input_hashes'],prediction_pins=pins,gold_sha256=p['gold_sha256'],
        scores_sha256=sha256_file(output/'scores.jsonl'),absolute=absolute,contrasts=contrasts,
        model_setup_seconds=sum(r['model_setup_seconds'] for r in reports),native_gate_seconds=sum(r['native_gate_seconds'] for r in reports),
        sum_job_stage_wall_seconds=sum(r['stage_wall_seconds'] for r in reports),
        sum_path_generation_seconds=sum(r['path_generation_seconds'] for r in reports),limitations=p['limitations'])
    shared.verify_pins(pins); atomic_json(output/'summary.json',result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    for name in ('output','panel','model-dir','native-reference'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-id',choices=MODEL_IDS,required=True); p.add_argument('--deadline-utc',required=True)
    p.add_argument('--checkpoint',type=Path); p.add_argument('--checkpoint-sha256'); p.add_argument('--complete-sha256'); p.add_argument('--training-protocol',type=Path)
    p = sub.add_parser('generate'); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-index',type=int,required=True); p.add_argument('--deadline-utc',required=True); p.add_argument('--device',choices=('cpu','xla'),default='xla')
    p = sub.add_parser('summarize'); p.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    if a.command=='prepare':
        result=prepare(a.output,a.panel,a.model_dir,a.native_reference,a.deadline_utc,a.model_id,
            checkpoint=a.checkpoint,checkpoint_sha256=a.checkpoint_sha256,complete_sha256=a.complete_sha256,training_protocol=a.training_protocol)
        print(json.dumps(dict(items=result['items'],modes=result['modes'],model=result['model']['id'],answer_prefill_width=result['answer_prefill_width'])))
    elif a.command=='generate':generate(a.output,a.shard_index,a.deadline_utc,a.device)
    else:print(json.dumps(summarize(a.output),indent=2))


if __name__=='__main__':main()
