"""Materialize existing compound TRAIN anchors, without a new corpus/model pass.

This is a new supervised candidate inventory, not a training release. Native
payloads come unchanged from the two closed publications. No new notes are made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file

SCHEMA = SALT = 'finemath-compound-anchor-candidates-v1'
ALL_SCHEMA = 'compound-priority-equation-candidates-v1'
ROOT = Path(__file__).resolve().parents[1]
HEADER = re.compile(rb'^\{"item_id": "([^"\\]+)", "document_id": "([^"\\]+)", "split": "(train|dev|test)"')
SPECS = (
    dict(publication='finemath-computable-equations-supervised-v2', priority=1,
         manifest_sha256='ae3c2b1a8c75b09fc00dbbaeb7ddcc8ad65f02d55202b337a917a34c4c9b431f',
         accepted_sha256='87b15e2bea5004234bda03e73677b487e62f0700a552b4980396ed37a16927f7'),
    dict(publication='finemath-computable-equations-raw-v3', priority=0,
         manifest_sha256='794260d8cc934bfdf5f13fc471d03c4062ee0d14af546cc08f05af2f6a5502d1',
         accepted_sha256='e4f08549c299452760e6dc6c9204061780c5a70ab0082a41b497646e5e14ad29'),
)


def sources():
    return {name: sha256_file(ROOT/name) for name in (
        'train/compound_anchor_candidates.py', 'weekend_thinking/artifacts.py',
        'weekend_thinking/data.py')}


def candidate_hash(document_id, absolute_cut):
    return canonical_hash(dict(salt=SALT, role='candidate', document_id=document_id,
                               absolute_source_cut=absolute_cut))


def document_hash(document_id):
    return canonical_hash(dict(salt=SALT, role='document_order', document_id=document_id))


def absolute_cut(row, publication):
    doc = row['document_id']
    if publication == SPECS[0]['publication']:
        match = re.fullmatch(re.escape(doc)+r':([0-9]+)', row['source_record_id'])
        if match is None or int(match[1]) < 512:
            raise ValueError('Old source_record_id is not a native anchor identity')
        cut = int(match[1])-512+row['boundary']['cut']
    elif publication == SPECS[1]['publication']:
        cut = row['absolute_source_cut']
        if row['source_chunk_token_start']+row['boundary']['cut'] != cut:
            raise ValueError('Raw absolute/local native cuts disagree')
    else:
        raise ValueError('Unrecognized closed publication')
    if type(cut) is not int or cut < 128:
        raise ValueError('Invalid absolute source cut')
    return cut


def validate_payload(row):
    """Internal identities only; no tokenizer, source reconstruction or relabeling."""
    cut = row['boundary']['cut']; original = row['original_source_ids']
    if row['split'] != 'train' or type(cut) is not int or not 128 <= cut <= 512 or len(original) != 528:
        raise ValueError('Expected unchanged TRAIN native528 source window')
    if row['prefix_ids'] != original[:cut] or row['boundary']['prefix_ids'] != row['prefix_ids']:
        raise ValueError('Causal prefix differs from original source window')
    if row['natural_future16_ids'] != original[cut:cut+16] or len(row['natural_future16_ids']) != 16:
        raise ValueError('Natural16 differs from the same source cut')
    target = row['target_ids']; mask = row['numeric_token_mask']; proof = row['verified_numeric_rhs']
    if not 1 <= len(target) <= 16 or target != row['natural_future16_ids'][:len(target)]:
        raise ValueError('Primary numeral+boundary target is not native future prefix')
    if len(mask) != len(target) or any(type(v) is not bool for v in mask) or not any(mask):
        raise ValueError('Invalid numeric overlap mask')
    if proof['token_ids'] != target or proof['numeric_token_mask'] != mask:
        raise ValueError('Saved scalar proof differs from native target')
    start = proof['closing_boundary_token_start']
    if not 0 <= start < len(target) or proof['closing_boundary_ids'] != target[start:]:
        raise ValueError('Invalid saved boundary token span')
    if row['closing_boundary_ids'] != target[start:] or row['closing_boundary_token_start'] != start:
        raise ValueError('Top-level boundary identity differs')
    if row['correct_note_ids'] != target or row['correct_note_text'] != proof['serialized_text']:
        raise ValueError('Preexisting target-copy payload differs from saved target')
    if row['copy_note_ids'] != target or row['copy_note_text'] != proof['serialized_text']:
        raise ValueError('Preexisting copy alias differs from saved target')
    if len(row['numeric_value']) > 200 or Fraction(row['numeric_value']) != Fraction(proof['value']):
        raise ValueError('Saved numeric values differ')
    if row['boundary']['numeric_value'] != row['numeric_value']:
        raise ValueError('Boundary arithmetic value differs')
    if type(row['boundary']['binary_ops']) is not int or not 1 <= row['boundary']['binary_ops'] <= 6:
        raise ValueError('Invalid frozen safe-AST operation count')


def equality_projection(row):
    """Compare semantic/native target identity, allowing original past crops to differ."""
    proof_keys = ('serialized_text', 'literal_text', 'numeral_char_start', 'numeral_char_end',
                  'closing_boundary_char', 'closing_boundary_char_offset',
                  'closing_boundary_token_start', 'boundary_token_mask', 'completion_kind')
    keys = ('natural_future16_ids', 'target_ids', 'numeric_token_mask', 'numeric_surface_text',
            'numeric_value', 'closing_boundary_ids', 'reanchor_ids')
    return dict(payload={k: row[k] for k in keys}, boundary={k: row['boundary'][k] for k in (
        'expression', 'normalized_expression', 'numeric_value', 'binary_ops')},
        target_relative_proof={k: row['verified_numeric_rhs'][k] for k in proof_keys})


def read_train(path, spec, audit, check=lambda: None, minimum_ops=2):
    """Opaque skip of every non-TRAIN payload, even if it is not valid JSON."""
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if number % 256 == 0: check()
            match = HEADER.match(line)
            if match is None: raise ValueError('Accepted-row canonical identity header missing')
            if match[3] != b'train': continue
            row = json.loads(line)
            if row['item_id'] != match[1].decode() or row['document_id'] != match[2].decode():
                raise ValueError('Canonical header identity differs from TRAIN payload')
            origin = dict(publication=spec['publication'], accepted_file=str(Path(path).resolve()),
                accepted_sha256=spec['accepted_sha256'], original_item_id=row['item_id'],
                line_number=number, original_line_sha256=hashlib.sha256(line).hexdigest())
            ops = row['boundary']['binary_ops']
            if type(ops) is not int: raise ValueError('Invalid binary operation count')
            if ops < minimum_ops:
                audit.append(dict(**origin, document_id=row['document_id'], binary_ops=ops,
                                  status='one_op_not_in_compound_inventory'))
                continue
            validate_payload(row)
            cut = absolute_cut(row, spec['publication'])
            audit.append(dict(**origin, document_id=row['document_id'], binary_ops=ops,
                              absolute_source_cut=cut, status='candidate_input'))
            yield row, origin, cut


def assemble(inputs, check=lambda: None, compound_priority=False):
    candidates = {}; audit = []; duplicates = []
    for path, spec in inputs:
        for row, origin, cut in read_train(path, spec, audit, check, 1 if compound_priority else 2):
            doc = row['document_id']; key = (doc, cut)
            if key in candidates:
                existing = candidates[key]
                if equality_projection(existing['source_payload']) != equality_projection(row):
                    raise ValueError(f'Same native cut disagrees: {doc}:{cut}')
                duplicates.append(dict(document_id=doc, absolute_source_cut=cut,
                    first_origin=existing['origins'][0], additional_origin=origin,
                    native_future_target_arithmetic_equal=True))
                existing['origins'].append(origin)
                if spec['priority'] < existing['_priority']:
                    existing['source_payload'] = row; existing['_priority'] = spec['priority']
                if existing['source_payload'] is row: existing['retained_publication'] = spec['publication']
            else:
                id_kind = 'compound-priority-v1' if compound_priority else 'compound-v1'
                candidates[key] = dict(candidate_id=f'{doc}:{id_kind}:{cut}', document_id=doc,
                    split='train', absolute_source_cut=cut, binary_ops=row['boundary']['binary_ops'],
                    expression=row['boundary']['expression'], normalized_expression=row['boundary']['normalized_expression'],
                    numeric_value=row['numeric_value'], candidate_hash=candidate_hash(doc, cut),
                    source_payload=row, origins=[origin], retained_publication=spec['publication'], _priority=spec['priority'])
    compound_docs = {r['document_id'] for r in candidates.values() if r['binary_ops'] >= 2}
    ordered = sorted(candidates.values(), key=lambda r: (
        int(r['document_id'] not in compound_docs) if compound_priority else 0,
        document_hash(r['document_id']),
        int(r['binary_ops'] < 2) if compound_priority else 0, r['candidate_hash']))
    seen = set(); representatives = []
    for row in ordered:
        row.pop('_priority'); doc = row['document_id']; row['representative'] = doc not in seen
        if row['representative']: representatives.append(row['candidate_id'])
        seen.add(doc); row['source_payload_sha256'] = canonical_hash(row['source_payload'])
    return ordered, representatives, audit, duplicates


def duplicate_groups(rows, key):
    groups = defaultdict(list)
    for row in rows: groups[key(row)].append(row['candidate_id'])
    return [dict(payload_sha256=key, candidate_ids=ids) for key, ids in sorted(groups.items()) if len(ids) > 1]


def prepare(supervised, raw, output, deadline_utc, specs=SPECS, expected=None, compound_priority=False):
    deadline = require_utc_deadline(deadline_utc); started = time.monotonic()
    def check():
        if datetime.now(timezone.utc) >= deadline: raise TimeoutError('Finite CPU materialization deadline')
    schema = ALL_SCHEMA if compound_priority else SCHEMA
    expected = expected or ((14680, 13182) if compound_priority else (5833, 5448))
    check(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    pinned = sources(); bindings = []; paths = []
    try:
        for directory, spec in zip((supervised, raw), specs):
            directory = Path(directory).resolve(); manifest = directory/'manifest.json'; accepted = directory/'accepted.jsonl'
            if sha256_file(manifest) != spec['manifest_sha256'] or sha256_file(accepted) != spec['accepted_sha256']:
                raise ValueError('Closed publication input hash differs')
            m = json.loads(manifest.read_text())
            if m['schema'] != spec['publication'] or m['accepted_sha256'] != spec['accepted_sha256']:
                raise ValueError('Closed manifest identity differs')
            bindings.append(dict(publication=spec['publication'], manifest_file=str(manifest),
                manifest_sha256=spec['manifest_sha256'], accepted_file=str(accepted), accepted_sha256=spec['accepted_sha256']))
            paths.append((accepted, spec))
        rows, representatives, audit, duplicates = assemble(paths, check, compound_priority)
        docs = [r['document_id'] for r in rows if r['representative']]
        if (len(rows), len(docs)) != expected: raise ValueError('Frozen full candidate/document counts differ')
        assert len(set(r['candidate_id'] for r in rows)) == len(rows) and len(set(docs)) == len(docs)
        check()
        with (output/'candidates.jsonl').open('x') as stream:
            for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
        with (output/'audit.jsonl').open('x') as stream:
            for entry in audit: stream.write(json.dumps(entry, allow_nan=False)+'\n')
        rule = ('Compound-bearing documents first, then one-op-only; fixed document hash within each group. '
                'Minimum compound candidate hash/document when available, otherwise minimum one-op hash; all documents retained.'
                if compound_priority else 'All documents in fixed document-hash order; lowest candidate hash/document. No reduction to5120.')
        atomic_json(output/'one-per-document.ids.json', dict(schema=schema, split='train',
            selected_count=len(docs), ordered_document_ids=docs, ordered_candidate_ids=representatives,
            ordered_document_ids_sha256=canonical_hash(docs), ordered_candidate_ids_sha256=canonical_hash(representatives),
            rule=rule))
        window_groups = duplicate_groups(rows, lambda r: canonical_hash(r['source_payload']['original_source_ids']))
        prefix_groups = duplicate_groups(rows, lambda r: canonical_hash(r['source_payload']['prefix_ids']))
        full_groups = duplicate_groups(rows, lambda r: r['source_payload_sha256'])
        atomic_json(output/'duplicate-audit.json', dict(same_cut_duplicates=duplicates,
            same_cut_native_target_arithmetic_disagreements=0, exact_original_window_groups=window_groups,
            exact_causal_prefix_groups=prefix_groups, exact_complete_accepted_payload_groups=full_groups,
            repeated_windows_retained=True,
            limitations='Repeated source windows across distinct equation cuts remain explicit exposure; repeated cuts are deduplicated.'))
        for binding in bindings:
            if sha256_file(binding['manifest_file']) != binding['manifest_sha256'] or sha256_file(binding['accepted_file']) != binding['accepted_sha256']:
                raise ValueError('Closed inputs changed during materialization')
        if sources() != pinned: raise ValueError('Materializer source changed')
        check()
        files = {name: sha256_file(output/name) for name in (
            'candidates.jsonl', 'audit.jsonl', 'one-per-document.ids.json', 'duplicate-audit.json')}
        manifest = dict(schema=schema, status='COMPLETE_CPU_CANDIDATE_INVENTORY_NOT_TRAINING_RELEASE',
            created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            split='train', candidates=len(rows), documents=len(docs), minimum_binary_ops=1 if compound_priority else 2,
            candidates_file=str(output/'candidates.jsonl'), candidates_sha256=files['candidates.jsonl'],
            representatives_file=str(output/'one-per-document.ids.json'), representatives_sha256=files['one-per-document.ids.json'],
            candidate_ids_sha256=canonical_hash([r['candidate_id'] for r in rows]),
            representative_ids_sha256=canonical_hash(representatives), source_files=pinned, input_publications=bindings,
            output_files={name: dict(file=str(output/name), sha256=value) for name, value in files.items()},
            selection_salt=SALT, candidate_hash_rule='canonical_hash({salt,role:candidate,document_id,absolute_source_cut})',
            document_order_rule='canonical_hash({salt,role:document_order,document_id})',
            representative_rule=rule, compound_priority=compound_priority,
            compound_documents=len({r['document_id'] for r in rows if r['binary_ops'] >= 2}),
            duplicate_priority='Prefer raw-v3 complete native row on same-cut equality; otherwise preserve sole old row. All origins retained.',
            binary_ops_histogram=dict(sorted(Counter(r['binary_ops'] for r in rows).items())),
            unique_normalized_expressions=len({r['normalized_expression'] for r in rows}),
            unique_numeric_values=len({r['numeric_value'] for r in rows}),
            unique_expression_value_pairs=len({(r['normalized_expression'], r['numeric_value']) for r in rows}),
            same_cut_duplicates=len(duplicates), duplicate_original_windows=len(window_groups),
            duplicate_causal_prefixes=len(prefix_groups), duplicate_complete_payloads=len(full_groups),
            input_train_rows=len(audit), one_op_input_rows=sum(r['status']=='one_op_not_in_compound_inventory' for r in audit),
            old_datasets_and_fresh_held256_unchanged=True, no_new_raw_read_tokenization_model_or_notes=True,
            limitations=['Structural compound curriculum, not measured numerical difficulty or a hard-math claim.',
                'Original payloads retain external exact-arithmetic/target-copy supervision and all existing note fields; no alternate notes generated.',
                'Future-aware source-label curation is inherited; native causal prefixes and natural16 remain unchanged.',
                'Different accepted source crops at the same cut can have different past lengths; only a complete existing row is retained, never spliced.',
                'No positive-gain filtering, training release or alteration of the old TRAIN5120/held DEV256.'])
        atomic_json(output/'manifest.json', manifest)
        atomic_json(output/'closure.json', dict(status=manifest['status'], manifest_sha256=sha256_file(output/'manifest.json'),
            candidates=len(rows), documents=len(docs), input_hashes_before_after_pass=True,
            native_internal_alignment_pass=True, same_cut_equality_pass=True, no_payload_decoding_of_nontrain=True,
            source_files=pinned, output_files=files))
    except BaseException as error:
        atomic_json(output/'materialization.failed.json', dict(status='PARTIAL_NOT_READY',
            error=f'{type(error).__name__}: {error}', source_files=pinned, inputs=bindings))
        raise
    print(json.dumps(dict(output=str(output), manifest_sha256=sha256_file(output/'manifest.json'),
        closure_sha256=sha256_file(output/'closure.json'), candidates=len(rows), documents=len(docs))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervised', type=Path, default=ROOT/'local/nvme/data'/SPECS[0]['publication'])
    parser.add_argument('--raw', type=Path, default=ROOT/'local/nvme/data'/SPECS[1]['publication'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-op-output', type=Path, help='Optional separate compound-priority fallback; preserves the compound publication.')
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    prepare(args.supervised, args.raw, args.output, args.deadline_utc)
    if args.all_op_output:
        prepare(args.supervised, args.raw, args.all_op_output, args.deadline_utc, compound_priority=True)


if __name__ == '__main__': main()
