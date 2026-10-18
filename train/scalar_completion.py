"""Conservative initial-scalar syntax, independent of gold or token generation.

V2 accepts one maximal ASCII signed decimal or decimal/decimal Fraction. No
leading Markdown/LaTeX wrappers, thousands grouping, exponent or mixed-number
syntax is interpreted. A newline terminates a numeric line, but a visible
operator/second number after that newline still makes it ambiguous. Following
prose is never classified by pronoun or answer-marker wording.

EOS certifies a bare literal. A cap does not; an earlier newline, explicit single
'=' relation, or sentence boundary can certify it. Equality certifies only the
emitted initial value, without parsing/evaluating its following derivation.
Sentence punctuation supports '.' and ';' with whitespace and nonnumeric
lookahead (or EOS/line termination). Commas need whitespace and actual prose
lookahead; comma/digits and numeric lists remain ambiguous. Space plus an
ordinary multi-letter unit/prose word also certifies a scalar; scale words such
as 'million' do not. '!' can mean factorial; '*' can mean multiplication
or Markdown; both are rejected, including across line breaks. A decimal ending
at '.' is only a sentence dot with that explicit boundary evidence.
"""
from __future__ import annotations

from fractions import Fraction
import re
import unicodedata

PARSER_VERSION = 'initial-scalar-v2'
MAX_TEXT, MAX_LITERAL, MAX_DIGITS = 8192, 128, 64
EOS_STOPS = frozenset(('eos', 'token:151643'))
WHITESPACE = ' \t\r\n'
NUMBER = r'[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)'
INITIAL = re.compile(rf'({NUMBER})(?:[ \t]*/[ \t]*({NUMBER}))?')
MATH_WORDS = frozenset(('plus','minus','times','over','divide','divided','dividing',
    'multiply','multiplied','multiplying','mod','modulo','square','squared','cube','cubed',
    'power','powers','raised','root','sqrt','percent','percentage','factorial','choose',
    'point','dot','by','add','added','adding','subtract','subtracted','subtracting',
    'sin','cos','tan','log','ln','e','exp','and','or','to'))
SCALE_WORDS = frozenset(('hundred','hundreds','thousand','thousands','million','millions',
    'billion','billions','trillion','trillions','quadrillion','quadrillions','quintillion',
    'quintillions','dozen','dozens','score','scores','lakh','lakhs','lac','lacs','crore',
    'crores','grand','mn','bn'))


def _result(status, **fields):
    return dict(parser_version=PARSER_VERSION, status=status, **fields)


def _arithmetic_tail(tail):
    """Conservative visible continuations, even after a local line boundary."""
    if not tail: return False
    first=tail[0]
    if (first in '+-*/^×÷−%‰_\\()[]{}<>:=!.,;'
            or first.isnumeric() or unicodedata.category(first)=='Sm'):
        return True
    word=re.match(r'[A-Za-z]+\b',tail)
    return bool((word and word.group().lower() in MATH_WORDS|SCALE_WORDS)
        or re.match(r'[eE][ \t]*[+-]?[0-9]',tail)
        or (tail[0] in 'eExX' and tail[1:2].isnumeric())
        or re.match(r'[xX][ \t]*[+-]?(?:[0-9]|\.[0-9])',tail))


def parse_initial_scalar_v2(text, stop_type):
    """Return status/value/literal/completion; never search past initial text.

    Status is valid, incomplete, unsupported, format_error or error. A valid
    rational string is suitable for exact Fraction comparison by a later scorer;
    this function receives no target and never stops/changes a generator.
    """
    if not isinstance(text,str) or len(text)>MAX_TEXT: return _result('unsupported')
    if stop_type not in EOS_STOPS and stop_type!='length_cap': return _result('error')
    terminal=stop_type in EOS_STOPS; source=text.lstrip(WHITESPACE); match=INITIAL.match(source)
    if match is None:
        if not terminal and source in ('','+','-','.','+.','-.'): return _result('incomplete')
        numeric_like=bool(source and (source[0].isnumeric() or source[0] in '−＋－*$\\('))
        return _result('unsupported' if numeric_like else 'format_error')
    literal=match.group(); tail=source[match.end():]; rest=tail.lstrip(WHITESPACE)
    if len(literal)>MAX_LITERAL or sum(c.isdigit() for c in literal)>MAX_DIGITS:
        return _result('unsupported')
    try: value=Fraction(match[1])/(Fraction(match[2]) if match[2] is not None else 1)
    except (ValueError,ZeroDivisionError): return _result('unsupported')
    whitespace=tail[:len(tail)-len(rest)]; line=any(c in '\r\n' for c in whitespace)
    completion=None
    if not rest:
        completion='eos' if terminal else 'line' if line else None
    elif rest.startswith('='):
        after=rest[1:].lstrip(WHITESPACE)
        if after[:1] in ('=','<','>'): return _result('unsupported')
        completion='equality'
    elif rest[0] in '.,;':
        following=rest[1:]; lookahead=following.lstrip(WHITESPACE)
        if following and following[0] not in WHITESPACE: return _result('unsupported')
        if rest[0]==',' and not re.match(r'[A-Za-z]+\b',lookahead): return _result('unsupported')
        if _arithmetic_tail(lookahead): return _result('unsupported')
        if lookahead: completion='sentence'
        elif terminal: completion='eos'
        elif any(c in '\r\n' for c in following): completion='line'
    elif _arithmetic_tail(rest): return _result('unsupported')
    elif not whitespace: return _result('unsupported')  # Adjacent identifier or undefined suffix.
    elif line: completion='line'
    elif re.match(r'[A-Za-z]{2,}\b',rest): completion='word'
    else: return _result('unsupported')  # A single letter can be a variable or exponent.
    if completion is None: return _result('incomplete')
    return _result('valid',value=str(value),literal=literal,completion=completion)
