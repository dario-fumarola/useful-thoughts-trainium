"""Package a bounded, exploratory TRAIN-only audit; never change V21 outcomes."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/home/ubuntu/dario')
AUDIT = ROOT / 'local/diagnostics/document-QA-v22-data-availability-v1'
DATA = ROOT / 'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
RAW = ROOT / 'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
NOTES = ROOT / 'local/analysis/document_QA_v22_TRAIN24_manual_notes_20261019.json'
OUT = ROOT / 'validation/document_QA_v22_TRAIN_audit_20261019'
PRIMARY = ROOT / 'validation/document_QA_v21_complete_20261019.json'
PRIMARY_SHA = '75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


assert sha(PRIMARY) == PRIMARY_SHA
original_hashes = {p.name: sha(p) for p in [DATA / 'TRAIN2048.jsonl',
    DATA / 'held512-questions-only.jsonl', DATA / 'held512-gold-only.jsonl']}
raw = {r['id']: r for r in rows(RAW)}
train = {r['item_id']: r for r in rows(DATA / 'TRAIN2048.jsonl')}
held = {r['item_id'] for r in rows(DATA / 'held512-questions-only.jsonl')}
template = rows(AUDIT / 'TRAIN96-grounding-review.jsonl')
notes = json.loads(NOTES.read_text())
expected = list(range(8)) + list(range(32, 40)) + list(range(64, 72))
assert [r['index'] for r in notes['reviews']] == expected
counts, source_counts = Counter(), {}
reviewed = []
for judgment in notes['reviews']:
    sample = template[judgment['index']]
    item = raw[sample['item_id']]
    assert sample['item_id'] in train and sample['item_id'] not in held
    assert len(judgment['labels']) == len(judgment['notes']) == sample['hops']
    assert set(judgment['labels']) <= set(notes['labels'])
    copy = {k: v for k, v in sample.items() if k != 'all_visible_contexts'}
    copy['template_index'] = judgment['index']
    copy['chain_flags'] = judgment['chain_flags']
    for i, step in enumerate(copy['steps']):
        original = item['decomposed_instances'][i]
        # Check the actual author-raw step evidence, not just the frozen pack.
        supports = [c for c in original['contexts'] if c['is_supporting']]
        assert any(c['paragraph_text'] == step['paragraph'] and
                   c['wikipedia_title'] == step['title'] for c in supports)
        assert original['answer_text'] == step['canonical_answer']
        visible = train[sample['item_id']]['question']['contexts'][step['citation'] - 1]
        assert visible['text'] == step['paragraph'] and visible['title'] == step['title']
        step['manual_entailment_label'] = judgment['labels'][i]
        step['manual_note'] = judgment['notes'][i]
        step['source_dataset'] = original['dataset']
        step['source_single_hop_id'] = original['id']
        counts[judgment['labels'][i]] += 1
        source_counts.setdefault(original['dataset'], Counter())[judgment['labels'][i]] += 1
    reviewed.append(copy)
assert len(reviewed) == 24 and sum(counts.values()) == 72
assert Counter(r['hops'] for r in reviewed) == {2: 8, 3: 8, 4: 8}
availability = json.loads((AUDIT / 'result.json').read_text())
assert availability['status'] == 'CPU_AVAILABILITY_INVENTORY_COMPLETE_NO_COHORT_OR_TRAINING_RELEASE'
assert sha(RAW) == 'a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
OUT.mkdir(exist_ok=False)
write(OUT / 'TRAIN24-cited-relations-and-chains.json', {
    'status': notes['status'], 'scope': notes['scope'], 'labels': notes['labels'],
    'examples': reviewed})
write(OUT / 'availability.json', availability)
shutil.copyfile(AUDIT / 'plan.json', OUT / 'availability-plan.json')
shutil.copyfile(AUDIT / 'manual-review-rubric.json', OUT / 'unexecuted-TRAIN96-rubric.json')
source = OUT / 'analysis-source'
source.mkdir()
for path in [Path(__file__), NOTES,
             ROOT / 'local/analysis/audit_document_QA_v22_data_availability_20261019.py']:
    shutil.copyfile(path, source / path.name)
report = {
    'status': 'CPU_TRAIN_AUDIT_COMPLETE_NO_NEW_COHORT_MODEL_OR_SCORE_RELEASE',
    'utc': datetime.now(timezone.utc).isoformat(),
    'question': 'Are the existing reasoning labels grounded well enough to justify simply resolving references or repeating training?',
    'manual_review': {
        'examples': 24, 'cited_steps': 72, 'examples_per_hop': {'2': 8, '3': 8, '4': 8},
        'selection': notes['scope'], 'cited_relation_counts': dict(counts),
        'cited_relation_counts_by_original_source': source_counts,
        'confirmed_cross_state_entity_conflation_examples': [r['item_id'] for r in reviewed
            if any(f.startswith('confirmed_cross_state') for f in r['chain_flags'])],
        'raw_provenance': 'All 72 cited passages and answers match their own author-raw single-hop annotations and frozen visible citation indices. Observed examples are not introduced by a shuffled paragraph index in our pack.',
        'limits': 'One coding agent, no independent semantic adjudication. Exploratory stratified convenience subset within a fixed-hash sample; no corpus error-rate estimate. A marks ambiguity, I citation insufficiency, not necessarily false answers. Other visible passages not exhaustively adjudicated. Whole-chain identity can fail even when every individual cited relation is locally supported.'},
    'availability': {
        'source_rows': availability['source_rows'],
        'at_qualified_3072_window': [r for r in availability['conditions'] if r['window'] == 3072],
        'limits': 'Eligible counts precede within-cohort exclusions. Greedy counts are constructive lower bounds, not optima or proofs of infeasibility. No candidates selected into a new frozen cohort. Repeated answers allowed; not the official MuSiQue split. 4096-token inventory is sizing only and exceeds the qualified width.'},
    'decision': 'Do not launch another run using the same canonical traces with blind answer substitution. First design a bounded grounded-evidence supervision/control comparison. Preserve V21 as a negative result; a changed label or parser needs a new identity and fresh held evidence.',
    'next_constraints': [
        'Separate literal citation presence, local relation entailment and cross-step entity identity.',
        'Use one prospective answer serialization/parser for all competing paths; report unsupported counts.',
        'Do not present answer-resolved questions as verified reasoning, or copying source text as proof of connected reasoning.',
        'A fresh cohort under the previous evidence exclusions is predominantly two-hop; it cannot support a strong four-hop generalization claim.',
        'No new training, RL, capacity change or official DEV/TEST file opened by this audit.'],
    'source_data': {'dataset': 'MuSiQue author TRAIN only, with train_patch already included',
        'authors_repository': 'https://github.com/stonybrooknlp/musique',
        'paper': 'https://arxiv.org/abs/2108.00573',
        'raw_sha256': sha(RAW), 'frozen_data_sha256': original_hashes},
    'unchanged_v21_primary_sha256': PRIMARY_SHA,
    'files_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}}
assert sha(PRIMARY) == PRIMARY_SHA
assert original_hashes == {p.name: sha(p) for p in [DATA / name for name in original_hashes]}
write(OUT.with_suffix('.json'), report)
print(json.dumps({'status': report['status'], 'reviewed_examples': len(reviewed),
    'cited_steps': sum(counts.values()), 'cited_relation_counts': dict(counts),
    'confirmed_cross_state_rows': len(report['manual_review']['confirmed_cross_state_entity_conflation_examples']),
    'public_report': str(OUT.with_suffix('.json'))}, indent=2))
