"""CPU preparation of declared compound supervision, without a gain filter.

This is a new reader-learning design informed by closed TRAIN diagnostics.
It preserves the failed frozen-reader filter and never reads its decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import torch

from train import useful_note_sft_data as packing
from train.calculation_boundaries import parse_numeric
from train.symbolic_note_labels import POSTORDER, teacher_note
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'useful-note-supervised-train-v3'


def sources():
    # Match the existing packer's causal helpers, without modifying them.
    from train.filter_symbolic_notes import sources as helper_sources
    return {**helper_sources(), 'train/prepare_reader_learning_data.py': sha256_file(__file__)}


def check_deadline(value):
    if datetime.now(timezone.utc) >= require_utc_deadline(value):
        raise TimeoutError('Finite CPU preparation expired; partial evidence retained')


def pinned(path, digest):
    path = Path(path).resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f'Immutable input changed: {path}')
    return path


def select_representatives(candidate_file, requested, check=lambda: None):
    """First eligible rows in publication order; no scores are an input.

    Scan the complete publication before returning. Invalid or repeated eligible
    representatives are failures, never reasons to replace an earlier document.
    The returned raw lines preserve every selected label/source field exactly.
    """
    if type(requested) is not int or requested < 1:
        raise ValueError('Positive declared inventory size required')
    selected = []; eligible_ids = []; eligible_docs = set(); cuts = set(); ids = set()
    all_ops = Counter(); eligible_ops = Counter(); total = 0
    with Path(candidate_file).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            check()
            if not line.strip():
                raise ValueError('Canonical candidate publication has no empty rows')
            row = json.loads(line); total += 1
            if row.get('split') != 'train':
                raise ValueError('TRAIN-only publication required before payload inspection')
            provenance = row['candidate_provenance']
            ops = provenance['binary_ops']; representative = provenance['representative']
            cut = (row['document_id'], row['absolute_source_cut'])
            if (type(ops) is not int or ops < 1 or type(representative) is not bool
                    or provenance['candidate_id'] != row['item_id']
                    or provenance['document_id'] != row['document_id']
                    or provenance['split'] != 'train'
                    or provenance['absolute_source_cut'] != row['absolute_source_cut']
                    or row['item_id'] in ids or cut in cuts):
                raise ValueError('Candidate identity/provenance or distinct-cut violation')
            ids.add(row['item_id']); cuts.add(cut); all_ops[ops] += 1
            if ops < 2 or not representative:
                continue
            parsed = parse_numeric(row['boundary']['expression'])
            if (parsed.binary_ops != ops or parsed.value != Fraction(row['numeric_value'])
                    or row.get('teacher_note_kind') != POSTORDER
                    or row['document_id'] in eligible_docs):
                raise ValueError('Invalid compound math or duplicate representative document')
            eligible_docs.add(row['document_id']); eligible_ids.append(row['item_id']); eligible_ops[ops] += 1
            if len(selected) < requested:
                selected.append((line_number, line, row))
    if len(eligible_ids) < requested:
        raise ValueError(f'Compound representative shortfall: {len(eligible_ids)} < {requested}; no fallback')
    inventory = dict(publication_items=total, eligible_compound_representatives=len(eligible_ids),
        eligible_unique_documents=len(eligible_docs), eligible_item_ids_sha256=canonical_hash(eligible_ids),
        eligible_binary_ops=dict(sorted(eligible_ops.items())), publication_binary_ops=dict(sorted(all_ops.items())),
        selected_items=requested, unselected_eligible=len(eligible_ids)-requested,
        selected_publication_lines=[n for n, _, _ in selected])
    return selected, inventory


def verify_source_rows(selected, union_file, check=lambda: None):
    """Compare native rows to the pinned union's exact original payloads."""
    wanted = {r['item_id']: r for _, _, r in selected}; seen = set()
    with Path(union_file).open('rb') as stream:
        for line in stream:
            check(); wrapper = json.loads(line)
            if wrapper.get('split') != 'train':
                raise ValueError('Upstream candidate union must remain TRAIN only')
            key = wrapper['candidate_id']
            if key not in wanted:
                continue
            if key in seen:
                raise ValueError('Repeated upstream candidate identity')
            seen.add(key); row = wanted[key]; payload = wrapper['source_payload']
            provenance = {k: v for k, v in wrapper.items() if k != 'source_payload'}
            if canonical_hash(payload) != wrapper['source_payload_sha256']:
                raise ValueError('Original source payload hash changed')
            expected = dict(payload, original_item_id=payload['item_id'], item_id=key,
                absolute_source_cut=wrapper['absolute_source_cut'], teacher_note_kind=POSTORDER,
                candidate_provenance=provenance, teacher_note_text=row['teacher_note_text'],
                teacher_note_ids=row['teacher_note_ids'])
            if row != expected:
                raise ValueError('Native row changed original source/natural continuation/provenance')
    if seen != set(wanted):
        raise ValueError('Selected native row missing from upstream source publication')


def validate_native(row, tokenizer):
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    note = packing.note_ids(row, tokenizer)
    original = row['original_source_ids']; past = row['prefix_ids']; cut = len(past)
    target = row['target_ids']; mask = row['numeric_token_mask']
    if (len(original) != 528 or not 128 <= cut <= 512 or original[:cut] != past
            or len(row['natural_future16_ids']) != 16
            or row['natural_future16_ids'] != original[cut:cut+16]
            or not 1 <= len(target) <= 16 or target != row['natural_future16_ids'][:len(target)]
            or len(mask) != len(target) or any(type(v) is not bool for v in mask) or not any(mask)
            or note != row['teacher_note_ids'] or teacher_note(row) != row['teacher_note_text']
            or decode(original) != row['original_source_text'] or decode(past) != row['prefix_text']
            or decode(row['reanchor_ids']) != row['reanchor_text']):
        raise ValueError('Native original528/natural16/note/math/serialization mismatch')
    return note


def measure_packing(rows, tokenizer, parts, check=lambda: None):
    maxima = {c: [0, 0] for c in packing.COMPONENTS}; notes = []
    for row in rows:
        check(); note = validate_native(row, tokenizer); notes.append(len(note))
        prompt = packing.writer_prompt(row['prefix_ids'], 'I3', parts)
        context = packing.reader_context(row, note, 'I3', parts)
        measures = dict(source=(528, 527), writer=(len(prompt)+len(note)+1, len(note)+1),
            reader=(len(context)+16, 16), augmented=(len(context)+528-len(row['prefix_ids']),
                len(context)+527-len(row['prefix_ids'])))
        for c, (length, labels) in measures.items():
            maxima[c] = [max(maxima[c][0], length), max(maxima[c][1], labels)]
    round32 = lambda n: ((n+31)//32)*32
    shapes = {c: dict(width=round32(n), target_slots=16 if c == 'reader' else round32(t))
              for c, (n, t) in maxima.items()}
    counts = {c: dict(examples=0, active_labels=0, note_close_labels=0, non_note_labels=0)
              for c in packing.COMPONENTS}
    for row in rows:
        check()
        for c in packing.COMPONENTS:
            batch = packing.pack_example(row, tokenizer, c, interface='I3', parts=parts, **shapes[c])
            note = batch['note_close_mask']; other = batch['future_mask']; active = note | other
            labels = batch['labels'][active].tolist(); inputs = batch['input_ids'][0]
            if ((note & other).any() or not active.any() or (batch['labels'][~active] != -100).any()
                    or not torch.equal(batch['labels'][active], batch['input_ids'].gather(1, batch['target_positions'])[active])
                    or (c == 'source' and labels != row['original_source_ids'][1:])
                    or (c == 'reader' and labels != row['natural_future16_ids'])
                    or (c == 'writer' and (other.any() or labels != row['teacher_note_ids']+[151668]))
                    or (c != 'writer' and note.any())
                    or (c == 'augmented' and labels != inputs[1:len(labels)+1].tolist())):
                raise ValueError('Invalid native causal component labels/masks')
            record = counts[c]; record['examples'] += 1; record['active_labels'] += int(active.sum())
            record['note_close_labels'] += int(note.sum()); record['non_note_labels'] += int(other.sum())
    return shapes, counts, dict(min=min(notes), max=max(notes), total=sum(notes))


def prepare(declaration_file, expected_declaration_sha256, candidates, filter_summary, model_dir, output, deadline_utc):
    from transformers import AutoTokenizer
    check = lambda: check_deadline(deadline_utc)
    check(); declaration_file = pinned(declaration_file, expected_declaration_sha256)
    declaration = json.loads(declaration_file.read_text()); candidates = Path(candidates).resolve(); output = Path(output).resolve()
    if (declaration['items'] != 5120 or declaration['unique_documents'] != 5120
            or declaration['selection_uses_likelihood'] is not False or declaration['interface'] != 'I3'
            or declaration['teacher_note_kind'] != POSTORDER or output.exists()):
        raise ValueError('Exact declared5120/I3/postorder/no-likelihood selection and fresh output required')
    inputs = {str(declaration_file): expected_declaration_sha256}
    for name, digest in declaration['input_pins'].items():
        path = pinned(ROOT/name, digest); inputs[str(path)] = digest
    manifest_file = candidates/'manifest.json'; candidate_file = candidates/'candidates.jsonl'
    if str(manifest_file) not in inputs or str(candidate_file) not in inputs:
        raise ValueError('Actual candidate publication must be pinned by declaration')
    cm = json.loads(manifest_file.read_text())
    if (cm['status'] != 'complete_CPU_labels_NOT_final_training_inventory'
            or cm['teacher_note_kind'] != POSTORDER or cm['candidates_sha256'] != inputs[str(candidate_file)]):
        raise ValueError('Complete immutable native postorder publication required')
    for name, digest in cm['input_hashes'].items():
        path = pinned(name, digest); inputs[str(path)] = digest
    filter_summary = pinned(filter_summary, declaration['old_filter']['summary_sha256'])
    old = json.loads(filter_summary.read_text())
    if (old['status'] != 'complete_filter_training_SHORTFALL' or old['selected_item_ids']
            or old['training_sha256'] is not None
            or any(old[k] != declaration['old_filter'][k] for k in ('positive', 'attempted', 'shortfall'))):
        raise ValueError('Preserve the actual failed original filter; no acceptance reinterpretation')
    inputs[str(filter_summary)] = sha256_file(filter_summary); frozen_sources = sources()
    selected, inventory = select_representatives(candidate_file, 5120, check)
    if inventory['publication_items'] != cm['candidate_count']:
        raise ValueError('Full publication coverage changed')
    union_manifests = [Path(p) for p in cm['input_hashes'] if Path(p).name == 'manifest.json']
    if len(union_manifests) != 1:
        raise ValueError('One original candidate-union manifest required')
    union_manifest = json.loads(union_manifests[0].read_text()); union_file = pinned(union_manifest['candidates_file'], union_manifest['candidates_sha256'])
    verify_source_rows(selected, union_file, check)
    held_paths = [Path(p) for p in inputs if Path(p).name == 'postsft-held256.ids.json']
    if len(held_paths) != 1:
        raise ValueError('Declared fresh DEV document inventory required')
    held = json.loads(held_paths[0].read_text()); rows = [r for _, _, r in selected]
    ids = [r['item_id'] for r in rows]; docs = [r['document_id'] for r in rows]
    if set(docs) & set(held['selected_document_ids']):
        raise ValueError('TRAIN and fresh held DEV document overlap')
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    vocab_hash = canonical_hash(tok.get_vocab())
    if vocab_hash != cm['tokenizer_vocab_sha256']:
        raise ValueError('Original native tokenizer changed')
    check(); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inventory.json', inventory)
    try:
        shapes, counts, lengths = measure_packing(rows, tok, cm['parts'], check)
        with (output/'training.jsonl').open('xb') as stream:
            for _, line, _ in selected:
                check(); stream.write(line)
        for name, digest in inputs.items(): pinned(name, digest)
        if sources() != frozen_sources:
            raise ValueError('Preparation/helper source changed during CPU preparation')
        check()
        result = dict(schema=SCHEMA, status='complete_CPU_inventory_NOT_Neuron_training_qualification',
            created_utc=utc_now(), experiment_id=declaration['experiment_id'], purpose='science', items=5120,
            unique_documents=5120, unique_source_cuts=5120, split='train', interface='I3', parts=cm['parts'],
            teacher_note_kind=POSTORDER, selection_uses_likelihood=False,
            selection=declaration['selection'], selection_informed_by_closed_TRAIN_diagnostics=True,
            declaration_file=str(declaration_file), declaration_sha256=expected_declaration_sha256,
            old_filter_file=str(filter_summary), old_filter_sha256=sha256_file(filter_summary), old_filter=declaration['old_filter'],
            eligible_inventory=inventory, packing=shapes, component_counts=counts, teacher_content_token_lengths=lengths,
            ordered_item_ids_sha256=canonical_hash(ids), ordered_document_ids_sha256=canonical_hash(docs),
            training_file=str(output/'training.jsonl'), training_sha256=sha256_file(output/'training.jsonl'),
            selected_original_line_sha256=canonical_hash([hashlib.sha256(line).hexdigest() for _, line, _ in selected]),
            tokenizer_vocab_sha256=vocab_hash, model_dir=str(Path(model_dir).resolve()),
            input_hashes=inputs, filter_input_hashes=inputs, source_files=frozen_sources,
            legacy_filter_input_hashes_semantics='Provenance alias only. No likelihood acceptance/filtering/ranking is performed.',
            writer_stop_ids=list(packing.writer_stop_ids('I3',cm['parts'])), reader_target='Original natural16',
            mask_semantics='future_mask is the legacy non-note role: source/augmented label every ordinary declared token; reader labels natural16 only.',
            limitations=[declaration['adaptation'], 'External exact symbolic supervision and future-derived answer surface, not generated-policy evidence.',
                'Same occurrences/updates are not equal labels, FLOPs or latency.', 'CPU inventory is not accelerator qualification or training release.'])
        atomic_json(output/'manifest.json', result); return result
    except Exception as exc:
        atomic_json(output/'FAILED.json', dict(status='failed_CPU_preparation_no_retry', error=repr(exc),
            finished_utc=utc_now(), declaration_sha256=expected_declaration_sha256))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','candidates','filter-summary','model-dir','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-declaration-sha256', required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    result = prepare(args.declaration, args.expected_declaration_sha256, args.candidates,
        args.filter_summary, args.model_dir, args.output, args.deadline_utc)
    print(json.dumps(dict(items=result['items'], eligible=result['eligible_inventory']['eligible_compound_representatives'],
                          packing=result['packing'], component_counts=result['component_counts'])), flush=True)


if __name__ == '__main__':
    main()
