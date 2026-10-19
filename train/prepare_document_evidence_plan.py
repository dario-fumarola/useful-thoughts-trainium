"""Freeze TRAIN pointer targets and a fresh question-only panel; CPU only."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from train.document_evidence_data import question_projection
from train.document_evidence_plan_data import fresh_candidates, select_fresh_panel, training_row
from train.document_evidence_plan_packing import COMPONENTS, prompt, training_pack
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return dict(path=str(path.resolve()), sha256=sha256_file(path), rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'native-lengths', 'previous-data', 'aliases', 'design', 'model-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required; preserve failed preparations')
    design = json.loads(args.design.read_text())
    if (design.get('experiment_id') != 'document-evidence-plan-v23'
            or design.get('design_revision') != 2
            or design['fresh_panel']['count'] != 255):
        raise ValueError('Explicit V2 design required; V1 selection failure remains separate')
    args.output.mkdir(parents=True)
    atomic_json(args.output / 'preparation-plan.json', dict(status='CPU_ONLY_NOT_TRAINING_RELEASE',
        utc=utc_now(), design_sha256=sha256_file(args.design),
        inputs={name: dict(path=str(path.resolve()), sha256=sha256_file(path)) for name, path in {
            'raw': args.raw, 'native_lengths': args.native_lengths, 'aliases': args.aliases,
            'old_training': args.previous_data / 'TRAIN2048.jsonl',
            'old_held_questions': args.previous_data / 'held512-questions-only.jsonl',
            'old_held_gold': args.previous_data / 'held512-gold-only.jsonl'}.items()}))
    (args.output / 'prospective-design.json').write_bytes(args.design.read_bytes())
    if (sha256_file(args.raw) != 'a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
            or sha256_file(args.previous_data / 'TRAIN2048.jsonl') !=
            'a0a99d866b55c16d7e6f498ee881d87556dfa5eabcab3340feccbf11313476e9'):
        raise ValueError('Unchanged author source and V21 TRAIN cohort required')
    raw_rows = read(args.raw)
    raw = {row['id']: row for row in raw_rows}
    old_train = read(args.previous_data / 'TRAIN2048.jsonl')
    old_held = read(args.previous_data / 'held512-questions-only.jsonl')
    lengths = {row['item_id']: row['native_input_tokens'] for row in read(args.native_lengths)}
    aliases = {row['item_id']: row for row in read(args.aliases)}
    candidates = fresh_candidates(raw_rows, lengths,
        {row['item_id'] for row in old_train}, {row['item_id'] for row in old_held})
    chosen = select_fresh_panel(candidates, count=255, higher_hop_count=32)
    training = [training_row(raw[row['item_id']]) for row in old_train]
    assert len(training) == 2048
    assert all(row['question'] == previous['question'] and row['answer'] == previous['answer']
               for row, previous in zip(training, old_train))
    questions, gold = [], []
    for metadata in chosen:
        item = raw[metadata['item_id']]
        question = question_projection(item)
        alias = aliases[item['id']]
        assert alias['canonical_answer'] == item['answer_text']
        questions.append(question)
        gold.append(dict(item_id=item['id'], answer=item['answer_text'],
            answer_aliases=alias['answer_aliases'],
            support_indices=training_row(item)['support_indices'],
            hops=metadata['hops'], original_seed_ids=sorted(metadata['seed_ids'])))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                             trust_remote_code=False)
    counts = {view: Counter() for view in ('direct', 'empty', 'selected')}
    maxima = Counter()
    for index, row in enumerate(training):
        packs = {view: training_pack(row, tokenizer, view) for view in counts}
        for component in ('source', 'answer'):
            labels = [pack['labels'][pack[component + '_mask']].tolist() for pack in packs.values()]
            assert all(labels[0] == other for other in labels[1:])
            assert labels[0]
        for view, pack in packs.items():
            for component in COMPONENTS:
                counts[view][component] += int(pack[component + '_mask'].sum())
            masks = [pack[name + '_mask'] for name in COMPONENTS]
            assert sum(mask.int() for mask in masks).max().item() == 1
            source_count = int(pack['source_mask'].sum())
            assert source_count == 256
            active = sum(mask.int() for mask in masks).bool()
            labels = pack['labels'][active]
            assert (labels == pack['input_ids'][0, pack['target_positions'][active]]).all()
            maxima['active_labels'] = max(maxima['active_labels'], len(labels))
            maxima['last_target_position'] = max(maxima['last_target_position'],
                                                 int(pack['target_positions'][active].max()))
            maxima[view + '_answer_labels'] = max(maxima[view + '_answer_labels'],
                                                  int(pack['answer_mask'].sum()))
        if (index + 1) % 256 == 0:
            print(json.dumps(dict(checked_training_rows=index + 1)), flush=True)
    for question in questions:
        for mode in ('direct', 'selection'):
            encoded = tokenizer.encode(prompt(question, mode), add_special_tokens=False)
            assert len(encoded) <= 3584
            maxima['held_prompt_tokens'] = max(maxima['held_prompt_tokens'], len(encoded))
    files = dict(training=write_rows(args.output / 'TRAIN2048.jsonl', training),
        held_questions=write_rows(args.output / 'held255-questions-only.jsonl', questions),
        held_gold=write_rows(args.output / 'held255-gold-only.jsonl', gold))
    inventories = {}
    for arm, augmented in [('evidence_direct', 'empty'), ('evidence_select', 'selected')]:
        inventories[arm] = {name: 2 * (counts['direct'][name] + counts[augmented][name])
                            for name in COMPONENTS}
    assert inventories['evidence_direct']['source'] == inventories['evidence_select']['source']
    assert inventories['evidence_direct']['answer'] == inventories['evidence_select']['answer']
    names = ('train/prepare_document_evidence_plan.py', 'train/document_evidence_plan_data.py',
        'train/document_evidence_plan_packing.py', 'train/document_evidence_data.py',
        'train/document_qa_data.py', 'train/document_qa_packing.py',
        'train/useful_note_sft_data.py', 'weekend_thinking/artifacts.py')
    manifest = dict(status='PASS_ALL_6144_NATIVE_POINTER_VIEWS_AND_FRESH255_CPU_ONLY',
        utc=utc_now(), experiment_id=design['experiment_id'], design_revision=2,
        design_sha256=sha256_file(args.design), items=2048, held=255,
        candidates=len(candidates), selected_hops=dict(Counter(row['hops'] for row in gold)),
        files=files, native_view_counts=counts, four_pass_inventory=inventories, maxima=maxima,
        same_final_labels_and_coefficients_in_all_views=True,
        source_files={name: sha256_file(ROOT / name) for name in names},
        tokenizer_vocab_sha256=canonical_hash(tokenizer.get_vocab()),
        tokenizer_files={name: sha256_file(args.model_dir / name)
            for name in ('tokenizer.json', 'tokenizer_config.json')},
        previous_data=str(args.previous_data.resolve()),
        preparation_plan_sha256=sha256_file(args.output / 'preparation-plan.json'),
        limits='CPU data/packing checks only, not hardware qualification or learning. Author support labels are not semantically certified. All old outcomes unchanged; V1 metadata failure retained. No model forward pass or official DEV/TEST opened.')
    atomic_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
