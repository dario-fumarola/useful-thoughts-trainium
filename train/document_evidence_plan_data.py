"""Offline passage-selection labels and prospective fresh-evidence selection.

Support annotations may be imperfect. A support ID is not an entailment label.
Only question_projection() outputs may enter a model prompt.
"""
from collections import defaultdict
import hashlib

from train.document_evidence_data import question_projection
from train.document_qa_data import evidence_identity

FRESH_SALT = 'document-evidence-plan-v23-fresh-held-v1:'


def training_row(raw):
    """Replace decomposition claims with IDs of already-visible paragraphs."""
    question = question_projection(raw)
    support = [i + 1 for i, context in enumerate(raw['contexts']) if context['is_supporting']]
    answer = raw['answer_text']
    if not support or not isinstance(answer, str) or not answer.strip():
        raise ValueError('Nonempty author answer and visible support annotations required')
    return dict(item_id=question['item_id'], split='train', question=question,
                answer=answer, support_indices=support)


def _features(row):
    return ({('paragraph', value) for value in row['support_text']}
            | {('title', value) for value in row['support_titles']}
            | {('seed', value) for value in row['seed_ids']}
            | {('question', row['question'])})


def fresh_candidates(raw_rows, native_lengths, old_train_ids, old_held_ids,
                     *, max_input_tokens=3072):
    """OFFLINE directed exclusions against ALL prior visible evidence."""
    raw = {row['id']: row for row in raw_rows}
    if len(raw) != len(raw_rows) or not old_train_ids.isdisjoint(old_held_ids):
        raise ValueError('Unique source identities and disjoint old cohorts required')
    excluded_ids = old_train_ids | old_held_ids
    if not excluded_ids <= raw.keys():
        raise ValueError('Every prior item must bind to the source')
    metadata = {key: dict(evidence_identity(row), hops=len(row['decomposed_instances']))
                for key, row in raw.items()}
    seen_questions = {metadata[key]['question'] for key in excluded_ids}
    blocked = {field: set().union(*(metadata[key][field] for key in excluded_ids))
               for field in ('visible_text', 'visible_titles', 'seed_ids')}
    result = []
    for key, row in metadata.items():
        if key in excluded_ids or row['question'] in seen_questions:
            continue
        length = native_lengths[key]
        if type(length) is not int or length < 1:
            raise ValueError('Complete positive native input lengths required')
        if (length > max_input_tokens or row['support_text'] & blocked['visible_text']
                or row['support_titles'] & blocked['visible_titles']
                or row['seed_ids'] & blocked['seed_ids']):
            continue
        result.append(row)
    return result


def select_fresh_panel(candidates, *, count=256, higher_hop_count=32):
    """Fixed, metadata-only two-stage greedy algorithm; no adaptive fallback."""
    if (type(count) is not int or type(higher_hop_count) is not int
            or not 0 <= higher_hop_count <= count
            or len({row['item_id'] for row in candidates}) != len(candidates)):
        raise ValueError('Explicit sizes and unique candidate IDs required')
    owners = defaultdict(set)
    features = {row['item_id']: _features(row) for row in candidates}
    for key, values in features.items():
        for value in values:
            owners[value].add(key)
    degree = {key: len(set().union(*(owners[value] for value in values))) - 1
              for key, values in features.items()}
    rank = lambda row: hashlib.sha256((FRESH_SALT + row['item_id']).encode()).hexdigest()
    selected, occupied = [], set()

    def add(row):
        values = features[row['item_id']]
        if values & occupied:
            return False
        selected.append(row)
        occupied.update(values)
        return True

    high = sorted((row for row in candidates if row['hops'] >= 3),
                  key=lambda row: (-row['hops'], degree[row['item_id']], rank(row)))
    for row in high:
        if len(selected) == higher_hop_count:
            break
        add(row)
    if len(selected) != higher_hop_count:
        raise ValueError('Insufficient independent higher-hop candidates; no cohort released')
    for row in sorted(candidates, key=lambda row: (degree[row['item_id']], rank(row))):
        if len(selected) == count:
            break
        add(row)
    if len(selected) != count:
        raise ValueError(f'Fixed fresh selection reached {len(selected)}/{count} rows; no cohort released')
    return selected
