"""Selection-format diagnostics, separate from canonical complete-answer scores.

Nonempty, unique, in-range pointers can still cite irrelevant evidence. This
parser never reads gold answers or support annotations and never repairs IDs.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedEvidence:
    status: str
    indices: tuple[int, ...] = ()
    reason: str = ''


def parse_evidence(text, paragraphs):
    if not isinstance(text, str) or type(paragraphs) is not int or paragraphs < 1:
        raise ValueError('Decoded response and visible paragraph count required')
    finals = list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*', text))
    if len(finals) != 1:
        return ParsedEvidence('unsupported', reason='missing_or_multiple_final_fields')
    prefix = text[:finals[0].start()]
    match = re.fullmatch(r'\s*Evidence:[ \t]*((?:\[[0-9]+\][ \t]*)*)\r?\n\s*',
                         prefix, re.IGNORECASE)
    if match is None:
        return ParsedEvidence('unsupported', reason='not_one_selection_field_before_final')
    indices = tuple(map(int, re.findall(r'\[([0-9]+)\]', match[1])))
    if not indices:
        return ParsedEvidence('empty')
    if len(indices) != len(set(indices)):
        return ParsedEvidence('unsupported', indices, 'duplicate_ids')
    if any(not 1 <= value <= paragraphs for value in indices):
        return ParsedEvidence('unsupported', indices, 'out_of_range_ids')
    return ParsedEvidence('complete', indices)


def annotated_support_overlap(parsed, support_indices):
    """Offline annotation agreement, explicitly not semantic correctness."""
    gold = set(support_indices)
    if not gold or any(type(value) is not int or value < 1 for value in gold):
        raise ValueError('Nonempty positive annotated support IDs required')
    chosen = set(parsed.indices) if parsed.status == 'complete' else set()
    common = len(chosen & gold)
    precision = common/len(chosen) if chosen else 0.
    recall = common/len(gold)
    return dict(precision=precision, recall=recall,
                exact_match=int(chosen == gold),
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
