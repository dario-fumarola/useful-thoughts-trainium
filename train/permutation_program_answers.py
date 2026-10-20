"""Gold-independent complete-symbol parser and separate execution diagnostics.

A final answer is one A..P symbol in its designated field. State diagnostics
never alter answer scores. All errors, caps and unsupported fields remain failures.
"""
from dataclasses import dataclass
import re

NODES = 'ABCDEFGHIJKLMNOP'
FINAL = re.compile(r'(?im)^Final answer:[ \t]*(?:\r?\n)?')
TRANSITION = re.compile(r'([rs]): ([A-P]) -> ([A-P])')


@dataclass(frozen=True)
class ParsedAnswer:
    status: str
    answer: str = ''
    reason: str = ''
    end: int | None = None


def parse_answer(text, mode, *, terminated=False, capped=False):
    """The prompt supplies Final answer for direct/empty/pause/partial controls.

    Trace mode must generate that exact field once. Intermediate states are not
    searched for an answer. Whitespace and node case are harmless; explanatory prose is unsupported.
    """
    if not isinstance(text, str) or mode not in ('direct', 'empty', 'pause', 'trace',
                                               'partial', 'corrupt', 'donor'):
        raise ValueError('Declared generated text and mode required')
    final = terminated or capped
    suffix = text
    if mode == 'trace':
        fields = list(FINAL.finditer(text))
        if len(fields) != 1:
            return ParsedAnswer('unsupported' if final else 'pending',
                                reason='missing_or_multiple_final_fields')
        suffix = text[fields[0].end():]
    else:
        fields = list(FINAL.finditer(text))
        if fields:
            if len(fields) != 1 or text[:fields[0].start()].strip():
                return ParsedAnswer('unsupported', reason='multiple_or_misplaced_answer_headers')
            suffix = text[fields[0].end():]
    suffix = suffix.lstrip(' \t\r\n')
    lines = suffix.splitlines(keepends=True)
    if not lines:
        return ParsedAnswer('unsupported' if final else 'pending', reason='empty_answer')
    first = lines[0].strip().upper()
    closed = lines[0].endswith(('\n', '\r')) or terminated
    if len(first) != 1 or first not in NODES:
        return ParsedAnswer('unsupported' if final or closed else 'pending', reason='not_one_node')
    if not closed:
        return ParsedAnswer('unsupported' if capped else 'pending', reason='unterminated_answer')
    if any(line.strip() for line in lines[1:]):
        return ParsedAnswer('unsupported', reason='trailing_content')
    return ParsedAnswer('complete', first, end=len(text))


def score_answer(prediction, expected, mode):
    if expected not in NODES or len(expected) != 1:
        raise ValueError('One exact gold node required for offline scoring')
    stop = prediction['stop_type']
    if stop not in ('token:151643', 'token:151645', 'qa_answer_line', 'length_cap', 'error'):
        raise ValueError('Unknown generation stop')
    parsed = parse_answer(prediction['text'], mode,
                          terminated=stop.startswith('token:'), capped=stop == 'length_cap')
    status = ('error' if prediction.get('status', 'ok') != 'ok' or stop == 'error' else
              'capped' if stop == 'length_cap' else parsed.status)
    return dict(status=status, parsed_answer=parsed.answer or None,
                score=int(status == 'complete' and parsed.answer == expected),
                reason=parsed.reason)


def execution_diagnostic(text, question):
    """Offline checking of all claimed transitions; does not adjudicate answers.

    The expression-only input has no labelled fields. Missing states, wrong
    operations, incorrect source/destination and extra prose are kept separate.
    """
    from train.permutation_program_data import Question
    if type(question) is not Question or not isinstance(text, str):
        raise ValueError('Available-input Question and original decoded trace required')
    fields = list(FINAL.finditer(text))
    body = text[:fields[0].start()] if len(fields) == 1 else text
    lines = body.splitlines()
    claims = [TRANSITION.fullmatch(line) for line in lines]
    valid = all(match is not None for match in claims)
    exact = 0; local = 0; current = question.start; chain_intact = True
    for i, match in enumerate(claims):
        if match is None:
            chain_intact = False
            continue
        op, src, dst = match.groups(); a, b = NODES.index(src), NODES.index(dst)
        local += int(getattr(question, op)[a] == b)
        wanted = i < len(question.program) and op == question.program[i]
        correct = wanted and chain_intact and a == current and getattr(question, op)[a] == b
        exact += int(correct)
        chain_intact = bool(correct)
        current = b
    complete = valid and len(lines) == len(question.program) and exact == len(lines)
    return dict(claimed_steps=len(lines), expected_steps=len(question.program),
                syntactically_valid=valid, locally_correct_transitions=local,
                correct_execution_prefix_steps=exact,
                complete_correct_execution=bool(complete))
