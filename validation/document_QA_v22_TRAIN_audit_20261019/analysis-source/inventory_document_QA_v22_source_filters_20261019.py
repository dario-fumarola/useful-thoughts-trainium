"""Exploratory source-policy counts, not semantic validation or a chosen filter."""
from collections import Counter
from pathlib import Path
import json
import re

ROOT = Path('/home/ubuntu/dario')
SOURCE = ROOT / 'local/nvme-study/data/musique-TRAIN-source-v12'
RAW = [json.loads(line) for line in (SOURCE / 'musique_ans_train.raw.jsonl').open()]
LENGTH = {r['item_id']: r['native_input_tokens'] for r in map(json.loads,
    (SOURCE / 'input-inventory-v2.jsonl').open())}
TRAIN = {r['item_id'] for r in map(json.loads, (ROOT /
    'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1/TRAIN2048.jsonl').open())}


def normalized(text):
    return ' '.join(text.casefold().split())


def exact_bridge(row):
    for hop in row['decomposed_instances']:
        for marker, original in hop['question_reverse_replacement'].items():
            match = re.fullmatch(r'\[\[RQS\]\] (\d+) \[\[RQE\]\]', marker)
            if not match or normalized(original) != normalized(
                    row['decomposed_instances'][int(match[1])]['answer_text']):
                return False
    return True


result = []
for policy in ['all', 'no_trex', 'nq_squad_only']:
    for bridge in [False, True]:
        selected = []
        for row in RAW:
            datasets = {h['dataset'] for h in row['decomposed_instances']}
            valid = (policy == 'all' or policy == 'no_trex' and 'trex' not in datasets
                     or policy == 'nq_squad_only' and datasets <= {'natural_questions', 'squad2'})
            if valid and LENGTH[row['id']] <= 3072 and (not bridge or exact_bridge(row)):
                selected.append(row)
        result.append(dict(source_policy=policy, exact_original_bridge_text=bridge,
            rows_at_3072=len(selected),
            hops=dict(Counter(len(r['decomposed_instances']) for r in selected)),
            in_old_train=sum(r['id'] in TRAIN for r in selected)))

# The first inventory was executed interactively. Reproduction must agree;
# preserve that result instead of silently replacing it.
original = ROOT / 'local/diagnostics/document-QA-v22-data-availability-v1/source-policy-inventory.json'
assert json.loads(original.read_text()) == json.loads(json.dumps(result))
print(json.dumps({'status': 'EXPLORATORY_COUNTS_REPRODUCED_NOT_SEMANTIC_FILTER',
                  'conditions': result}, indent=2))
