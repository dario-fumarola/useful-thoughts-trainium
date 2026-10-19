"""Read-only TRAIN structure/overlap audit; does not release a cohort or model."""
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path('/home/ubuntu/dario')
source = R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'

class Components:
    def __init__(self):
        self.parent, self.size, self.first = [], [], {}
    def add(self, paragraphs):
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(1)
        for p in paragraphs:
            if p in self.first:
                self.union(i, self.first[p])
            else:
                self.first[p] = i
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]: a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
    def summary(self):
        counts = Counter(self.find(i) for i in range(len(self.parent)))
        return dict(components=len(counts), singleton_questions=sum(v == 1 for v in counts.values()),
                    largest_question_components=sorted(counts.values(), reverse=True)[:10],
                    unique_exact_paragraph_texts=len(self.first))

def paragraph_id(context):
    # Exact text after whitespace normalization; not an assertion of factual independence.
    return hashlib.sha256(' '.join(context['paragraph_text'].split()).encode()).digest()

support_graph, visible_graph = Components(), Components()
positions, support_patterns, context_counts, setnames, questions = Counter(), Counter(), Counter(), Counter(), Counter()
records, training_visible, training_support = [], set(), set()
all_ids = set()
for line in source.open():
    row = json.loads(line)
    assert row['id'] not in all_ids and row['answerable']
    all_ids.add(row['id'])
    visible = [paragraph_id(c) for c in row['contexts']]
    support = [p for p, c in zip(visible, row['contexts']) if c['is_supporting']]
    assert support
    support_graph.add(support)
    visible_graph.add(visible)
    index = tuple(i+1 for i, c in enumerate(row['contexts']) if c['is_supporting'])
    positions.update(index)
    support_patterns[index] += 1
    context_counts[len(visible)] += 1
    setnames[row['setname']] += 1
    questions[' '.join(row['composed_question_text'].lower().split())] += 1
    train = int(hashlib.sha256(('musique-v12-audit-only:'+row['id']).encode()).hexdigest(), 16) % 1000 < 800
    if train:
        training_visible.update(visible)
        training_support.update(support)
    records.append(dict(train=train, visible=set(visible), support=set(support)))

held = [r for r in records if not r['train']]
result = dict(status='READ_ONLY_TRAIN_SPLIT_RISK_AUDIT_NO_COHORT_OR_MODEL_RELEASE',
              utc=datetime.now(timezone.utc).isoformat(), rows=len(records), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              setnames=dict(setnames), context_counts=dict(context_counts), supporting_position_counts=dict(positions),
              most_common_support_position_patterns=[dict(positions=list(k), questions=v) for k, v in support_patterns.most_common(10)],
              exact_question_text_duplicate_excess=sum(n-1 for n in questions.values()),
              support_sharing_components=support_graph.summary(), any_visible_sharing_components=visible_graph.summary(),
              hypothetical_question_hash_split=dict(training_questions=sum(r['train'] for r in records), held_questions=len(held),
                held_with_any_support_seen_in_training_visible=sum(bool(r['support'] & training_visible) for r in held),
                held_with_all_support_seen_in_training_visible=sum(r['support'] <= training_visible for r in held),
                held_with_any_support_seen_as_training_support=sum(bool(r['support'] & training_support) for r in held),
                held_with_all_support_seen_as_training_support=sum(r['support'] <= training_support for r in held)),
              limits=['Only authors TRAIN inspected; no official DEV/TEST downloaded or read.',
                      'Hypothetical80/20 question hash is an audit, not an adopted split or released cohort.',
                      'Exact paragraph overlap is not the same as answer leakage or semantic duplication. All facts remain visible at inference.',
                      'Gold support flags are offline audit labels only; never policy inputs. No model results inspected.'])
out = R/'local/analysis/musique-TRAIN-split-risk-audit-v12-20261019.json'
with out.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2), flush=True)
