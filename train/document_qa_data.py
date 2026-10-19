"""MuSiQue annotation identities and grounded, single-model supervision.

Question projection remains in document_evidence_data. Everything below that
reads an answer, decomposition or support flag is OFFLINE curation only.
"""
import hashlib
import re

from train.document_evidence_data import question_projection


def normalized(text):
    return ' '.join(text.casefold().split())


def fingerprint(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def original_seed_ids(row):
    """Original question IDs, never the generic component relation programs."""
    hops = row['decomposed_instances']
    result = {f'{hop["dataset"]}:{hop["id"]}' for hop in hops}
    if not hops or len(result) != len(hops):
        raise ValueError('Distinct original single-hop question identities required')
    return result


def evidence_identity(row):
    """OFFLINE split metadata. Do not pass this mapping to a policy."""
    question = question_projection(row)
    support = [c for c in row['contexts'] if c['is_supporting']]
    if not support:
        raise ValueError('Answerable evidence-bearing rows required')
    return dict(
        item_id=question['item_id'], question=normalized(question['question']),
        support_text={fingerprint(c['paragraph_text']) for c in support},
        support_titles={normalized(c['wikipedia_title']) for c in support},
        visible_text={fingerprint(c['paragraph_text']) for c in row['contexts']},
        visible_titles={normalized(c['wikipedia_title']) for c in row['contexts']},
        seed_ids=original_seed_ids(row))


def supervised_trace(row, canonical):
    """Complete canonical decomposition, citations and answers, including root.

    Gold supervision is privileged. The generator receives only the projected
    authored question and ALL visible paragraphs. #n references retain the
    authors' dependency notation; they must point to an earlier step.
    """
    hops, linked = row['decomposed_instances'], canonical['question_decomposition']
    if len(hops) != len(linked) or not 2 <= len(hops) <= 4:
        raise ValueError('Complete two-to-four-hop annotated decomposition required')
    lines = []
    for i, (raw, hop) in enumerate(zip(hops, linked), 1):
        index = hop['paragraph_support_idx']
        if (hop['id'] != raw['id'] or hop['answer'] != raw['answer_text']
                or type(index) is not int or not 0 <= index < len(row['contexts'])
                or not row['contexts'][index]['is_supporting']
                or any(not 1 <= int(ref) < i for ref in re.findall(r'#(\d+)', hop['question']))
                or any(marker in hop['question'] for marker in ('[[CQS]]', '[[RQS]]', '[SEP]'))):
            raise ValueError('Trace identity, visible support or backward dependency differs')
        lines.append(f'Step {i}: {hop["question"]}\nEvidence [{index+1}]: {hop["answer"]}')
    if linked[-1]['answer'] != row['answer_text'] or canonical['answer'] != row['answer_text']:
        raise ValueError('Root answer must agree with the annotated final answer')
    return '\n'.join(lines)


def select_cohort(records, *, held_count=512, train_count=2048):
    """Fixed hash order, independent held support/seed IDs, directed exclusions.

    Repeated distractors and relation templates are permitted. No held gold
    supporting text/page or seed question may occur in training-visible data.
    Exact normalized composed questions are unique in each returned cohort.
    """
    ranked = sorted(records, key=lambda r: (
        hashlib.sha256(('document-evidence-v16-audit-only:'+r['item_id']).encode()).hexdigest(),
        r['item_id']))
    held, seen_questions = [], set()
    support, titles, seeds = set(), set(), set()
    for row in ranked:
        if (row['question'] in seen_questions or row['support_text'] & support
                or row['support_titles'] & titles or row['seed_ids'] & seeds):
            continue
        held.append(row); seen_questions.add(row['question'])
        support.update(row['support_text']); titles.update(row['support_titles']); seeds.update(row['seed_ids'])
        if len(held) == held_count:
            break
    if len(held) != held_count:
        raise ValueError('Insufficient independent held questions')
    held_ids = {r['item_id'] for r in held}
    training = []
    for row in ranked:
        if (row['item_id'] in held_ids or row['question'] in seen_questions
                or row['visible_text'] & support or row['visible_titles'] & titles
                or row['seed_ids'] & seeds):
            continue
        training.append(row); seen_questions.add(row['question'])
        if len(training) == train_count:
            break
    if len(training) != train_count:
        raise ValueError('Insufficient evidence-clean training questions')
    return training, held
