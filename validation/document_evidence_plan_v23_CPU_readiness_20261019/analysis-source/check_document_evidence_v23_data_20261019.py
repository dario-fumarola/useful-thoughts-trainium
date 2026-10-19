"""Independent raw-string/source checks, without the cohort-selection helpers."""
from collections import Counter
from pathlib import Path
import hashlib
import json

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/document-evidence-v23'
D = R / 'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
V = R / 'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
S = R / 'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows = lambda path: [json.loads(line) for line in Path(path).open()]
norm = lambda text: ' '.join(text.casefold().split())
raw = {row['id']: row for row in rows(S)}
manifest = json.loads((D / 'manifest.json').read_text())
for record in manifest['files'].values():
    assert sha(record['path']) == record['sha256']
for name, digest in manifest['source_files'].items():
    assert sha(W / name) == digest
old_train = rows(V / 'TRAIN2048.jsonl')
old_held = rows(V / 'held512-questions-only.jsonl')
train, questions, gold = rows(D / 'TRAIN2048.jsonl'), rows(D / 'held255-questions-only.jsonl'), rows(D / 'held255-gold-only.jsonl')
assert len(train) == 2048 and len(questions) == len(gold) == 255
assert len({row['item_id'] for row in train}) == 2048
assert len({row['item_id'] for row in questions}) == 255
old_ids = {row['item_id'] for row in old_train + old_held}
blocked_text, blocked_titles, blocked_seeds, blocked_questions = set(), set(), set(), set()
for key in old_ids:
    item = raw[key]
    blocked_text.update(norm(c['paragraph_text']) for c in item['contexts'])
    blocked_titles.update(norm(c['wikipedia_title']) for c in item['contexts'])
    blocked_seeds.update((h['dataset'], h['id']) for h in item['decomposed_instances'])
    blocked_questions.add(norm(item['composed_question_text']))
for row, old in zip(train, old_train):
    assert row['item_id'] == old['item_id']
    assert set(row) == {'item_id', 'split', 'question', 'answer', 'support_indices'}
    assert row['split'] == 'train' and row['answer'] == old['answer']
    assert row['question'] == old['question']
seen_text, seen_titles, seen_seeds, seen_questions = set(), set(), set(), set()
for question, label in zip(questions, gold):
    key = question['item_id']
    assert key == label['item_id'] and key not in old_ids
    item = raw[key]
    expected = dict(item_id=key, question=item['composed_question_text'],
        contexts=[dict(title=c['wikipedia_title'], text=c['paragraph_text']) for c in item['contexts']])
    assert question == expected
    support = [c for c in item['contexts'] if c['is_supporting']]
    texts = {norm(c['paragraph_text']) for c in support}
    titles = {norm(c['wikipedia_title']) for c in support}
    seeds = {(h['dataset'], h['id']) for h in item['decomposed_instances']}
    q = norm(item['composed_question_text'])
    assert not texts & (blocked_text | seen_text)
    assert not titles & (blocked_titles | seen_titles)
    assert not seeds & (blocked_seeds | seen_seeds)
    assert q not in blocked_questions | seen_questions
    assert label['answer'] == item['answer_text']
    assert label['support_indices'] == [i+1 for i, c in enumerate(item['contexts']) if c['is_supporting']]
    assert label['hops'] == len(item['decomposed_instances'])
    assert label['original_seed_ids'] == sorted(f'{name}:{value}' for name, value in seeds)
    seen_text.update(texts); seen_titles.update(titles); seen_seeds.update(seeds); seen_questions.add(q)
counts = manifest['four_pass_inventory']
assert counts['evidence_direct']['source'] == counts['evidence_select']['source'] == 2097152
assert counts['evidence_direct']['answer'] == counts['evidence_select']['answer'] == 67408
primary = R / 'validation/document_QA_v21_complete_20261019.json'
assert sha(primary) == '75e5caf4638ab4de27b53663707b99482ed8ac84f4ab95584549f4537f2da3a9'
result = dict(status='PASS_INDEPENDENT_SOURCE_SPLIT_AND_LABEL_IDENTITY_CPU_ONLY',
    training=2048, held=255, held_hops=dict(Counter(r['hops'] for r in gold)),
    checks='Raw normalized strings and tuple seed IDs, exact question projections, directed and within-held exclusions, unchanged ordered training questions/answers, pointer labels, source bindings and V21 primary hash.',
    limits='Does not independently certify semantic support, repeat native packing or qualify hardware/learning.',
    manifest_sha256=sha(D / 'manifest.json'), source_sha256=sha(__file__))
target = D / 'independent-source-check.json'
assert not target.exists()
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
