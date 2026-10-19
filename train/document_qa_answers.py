"""Complete, designated QA answer fields; no search for gold-like substrings.

Direct mode scores the first nonempty output line after supplied Answer:.
Prior mode requires one anchored Final answer: field (case insensitive). The
answer may start on that line or the next nonempty line. A newline or sampled
terminator must close the field. A bare length-capped suffix is incomplete.

EM/F1 normalization follows the MuSiQue/SQuAD definitions, reimplemented here:
https://github.com/stonybrooknlp/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py
Canonical-only and attributed-alias metrics remain separately named.
"""
from collections import Counter
from dataclasses import dataclass
import re
import string


@dataclass(frozen=True)
class ParsedAnswer:
    status: str
    answer: str = ''
    end: int | None = None
    reason: str = ''


def parse_answer(text, mode, *, terminated=False, capped=False):
    if mode not in ('direct', 'prior') or not isinstance(text, str):
        raise ValueError('Declared direct/prior decoded text required')
    terminal = terminated or capped
    start = 0
    if mode == 'prior':
        fields = list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*', text))
        if len(fields) != 1:
            return ParsedAnswer('unsupported' if terminal else 'pending',
                                reason='missing_final_field' if not fields else 'multiple_final_fields')
        start = fields[0].end()
    while start < len(text) and text[start].isspace():
        start += 1
    newline = text.find('\n', start)
    if newline >= 0:
        value = text[start:newline].strip()
        return ParsedAnswer('complete', value, newline+1) if value else ParsedAnswer('unsupported', reason='empty_answer')
    value = text[start:].strip()
    if terminated and value:
        return ParsedAnswer('complete', value, len(text))
    return ParsedAnswer('unsupported' if terminal else 'pending',
                        reason='empty_answer' if not value else 'unterminated_cap')


def normalize_answer(text):
    lowered = text.lower()
    unpunctuated = ''.join(c for c in lowered if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', unpunctuated, flags=re.UNICODE).split())


def exact_match(prediction, gold):
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    return 2.*common/(len(p)+len(g))


def score(parsed, canonical, aliases=()):
    if not isinstance(canonical, str) or not canonical.strip() or any(not isinstance(v,str) or not v.strip() for v in aliases):
        raise ValueError('Nonempty attributed gold answers required')
    if parsed.status != 'complete':
        return dict(canonical_em=0, canonical_f1=0., attributed_alias_em=0, attributed_alias_f1=0.)
    alternatives = [canonical, *aliases]
    return dict(canonical_em=exact_match(parsed.answer, canonical),
        canonical_f1=token_f1(parsed.answer, canonical),
        attributed_alias_em=max(exact_match(parsed.answer, g) for g in alternatives),
        attributed_alias_f1=max(token_f1(parsed.answer, g) for g in alternatives))
