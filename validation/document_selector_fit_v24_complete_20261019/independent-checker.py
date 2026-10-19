"""Independent raw-token/count check for the frozen TRAIN-only fit diagnostic."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import string
from transformers import AutoTokenizer

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-selector-fit-v24-TRAIN256-v1'
O = R/'local/nvme-cpt/evaluation/document-selector-fit-v24-TRAIN256-v1'
read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(v) for v in p.read_text().splitlines()]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
release = read(C/'release.json'); protocol = read(O/'protocol.json')
assert sha(C/'release.json') == protocol['release_sha256']
assert read(C/'GEN/closure.json')['assigned_physical_device_empty']
report = read(C/'result.json')
assert report['protocol_sha256'] == sha(O/'protocol.json')
assert report['scored_sha256'] == sha(C/'native-and-scored.jsonl')
train = rows(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2/TRAIN2048.jsonl')
labels = rows(C/'TRAIN256-labels.jsonl')
expected = []
for n, count in ((2, 128), (3, 96), (4, 32)):
    pool = [v for v in train if len(v['support_indices']) == n]
    expected.extend(sorted(pool, key=lambda v: hashlib.sha256((release['salt']+v['item_id']).encode()).hexdigest())[:count])
assert {v['item_id'] for v in expected} == {v['item_id'] for v in labels}
assert {v['item_id']: v for v in expected} == {v['item_id']: v for v in labels}
labels = {v['item_id']: v for v in labels}
tokenizer = AutoTokenizer.from_pretrained(protocol['model_dir'], local_files_only=True, trust_remote_code=False)
actual = []
for shard in range(4):
    d = O/'answers'/f'shard-{shard}'
    run = read(d/'run.json')
    assert run['status'] == 'complete_all_attempts' and run['rows'] == 64
    assert run['attempts_sha256'] == sha(d/'attempts.jsonl')
    assert read(d/'native.json')['passed']
    assert read(d/'cache-prefill.json')['max_logp_error'] <= .001
    shard_rows = rows(d/'attempts.jsonl')
    assert [v['item_id'] for v in shard_rows] == protocol['ordered_item_ids'][shard::4]
    actual.extend(shard_rows)
assert len(actual) == len({v['item_id'] for v in actual}) == 256
strata = {n: [] for n in (2, 3, 4)}
def norm(text):
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', ''.join(v for v in text.lower() if v not in string.punctuation)).split())
for v in actual:
    assert v['status'] == 'ok' and v['n_actions'] == len(v['actions']) <= 128
    assert tokenizer.decode(v['content_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False) == v['raw_text']
    assert v['content_ids'] == (v['actions'][:-1] if v['stop_type'].startswith('token:') else v['actions'])
    g = labels[v['item_id']]
    fields = list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*', v['raw_text']))
    ids = []
    valid = False
    answer = ''
    if len(fields) == 1:
        prefix = v['raw_text'][:fields[0].start()]
        match = re.fullmatch(r'\s*Evidence:[ \t]*((?:\[\d+\][ \t]*)*)\r?\n\s*', prefix, re.I)
        if match:
            ids = list(map(int, re.findall(r'\[(\d+)\]', match[1])))
            valid = bool(ids) and len(set(ids)) == len(ids) and all(1 <= z <= len(g['question']['contexts']) for z in ids)
        tail = v['raw_text'][fields[0].end():].lstrip()
        if tail and ('\n' in tail or v['stop_type'].startswith('token:')):
            answer = tail.split('\n', 1)[0].strip()
    chosen = set(ids) if valid else set()
    gold = set(g['support_indices']); overlap = len(chosen & gold)
    precision = overlap/len(chosen) if chosen else 0.
    recall = overlap/len(gold)
    strata[len(gold)].append(dict(exact=int(chosen == gold), precision=precision, recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0., count=len(ids),
        canonical=int(bool(answer) and norm(answer) == norm(g['answer']))))
for n, sub in strata.items():
    s = report['strata'][str(n)]
    assert s['items'] == len(sub) and s['exact'] == sum(v['exact'] for v in sub)
    assert s['output_ID_counts'] == {str(k): v for k, v in Counter(v['count'] for v in sub).items()}
    assert s['canonical_TRAIN_correct'] == sum(v['canonical'] for v in sub)
    for k in ('precision', 'recall', 'f1'):
        assert abs(s[k]-sum(v[k] for v in sub)/len(sub)) <= 1e-12
assert report['operational_TRAIN_fit_target_pass'] == all(report['strata'][str(n)]['exact']/report['strata'][str(n)]['items'] >= .9 for n in (2, 3))
result = dict(status='PASS_INDEPENDENT_RAW_TOKEN_COHORT_COUNTS_OVERLAP_AND_FIT_CHECK',
    report_sha256=sha(C/'result.json'), checker_sha256=sha(Path(__file__)), actual_attempts=256,
    native_checks=4, cached_prefill_checks=4,
    limit='This verifies stored measurements, not semantic annotations, generalization or capability.')
with (C/'independent-check.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
