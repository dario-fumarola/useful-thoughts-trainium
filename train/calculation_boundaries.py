"""Conservative real-FineMath arithmetic boundaries; no model-based selection.

One hashed search start/document fixes the first eligible causal boundary before
source-RHS curation, with no replacement on failure. TEST rows are never parsed.
External exact arithmetic supplies a proper intermediate, not spontaneous thought.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
from pathlib import Path
import re
import unicodedata

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = 'finemath-arithmetic-boundaries-train-dev-v1'
SALT = 'finemath-arithmetic-boundaries-v1'
SOURCE_SHA = 'bd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff'
MAX_OPS, MAX_NODES, MAX_POWER, MAX_VALUE, MAX_PART = 6, 63, 8, 10**12, 10**18
MIN_PAST, SOURCE_PREFIX, SOURCE_FUTURE, MAX_RHS = 128, 512, 16, 16
HEADER = re.compile(r'^\{\s*"id"\s*:\s*"([^"\\]+)"\s*,\s*"document_id"\s*:\s*"([^"\\]+)"\s*,\s*"split"\s*:\s*"(train|dev|test)"\s*,\s*"prefix"\s*:\s*')
NUM = r'(?:\d+(?:\.\d+)?|\.\d+)'
RATIONAL = rf'[+\-\u2212]?\s*{NUM}(?:\s*/\s*[+\-\u2212]?\s*{NUM})?'
MATH_SUFFIX = re.compile(r'[\d\s.()+*/^\-+\u2212\u00d7\u00f7]+$')
FUNCTIONS = {'sin','cos','tan','cot','sec','csc','asin','acos','atan','arcsin','arccos','arctan',
             'sinh','cosh','tanh','log','ln','exp','sqrt','abs','max','min'}


class Unsupported(ValueError):
    """Unsupported/ambiguous arithmetic is not a claim that a source is false."""


@dataclass(frozen=True)
class Parsed:
    expression: str
    value: Fraction
    binary_ops: int
    has_product: bool
    intermediate_expression: str | None
    intermediate_value: Fraction | None


def render_value(value):
    """Exact terminating decimal up to24 places; never round a rational."""
    if value.denominator==1:return str(value.numerator)
    denominator=value.denominator;twos=fives=0
    while denominator%2==0:twos+=1;denominator//=2
    while denominator%5==0:fives+=1;denominator//=5
    places=max(twos,fives)
    if denominator!=1 or places>24:return f'{value.numerator}/{value.denominator}'
    scaled=abs(value.numerator)*(2**(places-twos))*(5**(places-fives))
    whole,fraction=divmod(scaled,10**places)
    return f"{'-' if value<0 else ''}{whole}.{str(fraction).zfill(places).rstrip('0')}"


def bounded(value):
    if abs(value) > MAX_VALUE or abs(value.numerator) > MAX_PART or value.denominator > MAX_PART:
        raise Unsupported('value_bound')
    return value


def parse_numeric(expression: str) -> Parsed:
    """No eval. Decimal literals become exact Fractions from their source text."""
    text = expression.strip().replace('\u2212', '-').replace('\u00d7', '*').replace('\u00f7', '/')
    if not text or len(text) > 160 or '\n' in text or re.search(r'[^\d\s.()+*/^\-+]', text):
        raise Unsupported('unsupported_expression_characters')
    text = text.replace('^', '**')
    try:
        tree = ast.parse(text, mode='eval')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Unsupported('incomplete_or_non_numeric_expression') from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise Unsupported('node_bound')
    operations = []; products = []
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(text, node)
            if not re.fullmatch(NUM, literal or ''):
                raise Unsupported('unsupported_numeric_literal')
            return bounded(Fraction(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise Unsupported('non_numeric_AST')
        a, b = walk(node.left), walk(node.right)
        try:
            if isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b) > MAX_POWER:
                    raise Unsupported('power_bound_or_noninteger_exponent')
                v = a ** int(b)
            elif isinstance(node.op, ast.Mult): v = a*b
            elif isinstance(node.op, ast.Div): v = a/b
            elif isinstance(node.op, ast.Add): v = a+b
            else: v = a-b
        except ZeroDivisionError as exc:
            raise Unsupported('division_by_zero') from exc
        bounded(v); operations.append((node, v))
        if isinstance(node.op, (ast.Mult, ast.Div, ast.Pow)): products.append(node)
        if len(operations) > MAX_OPS: raise Unsupported('operation_bound')
        return v
    value = walk(tree.body)
    def form(node):
        if isinstance(node, ast.Constant): return ('number', Fraction(ast.get_source_segment(text,node)))
        if isinstance(node, ast.UnaryOp):
            child=form(node.operand)
            if isinstance(node.op,ast.UAdd): return child
            return ('number',-child[1]) if child[0]=='number' else ('negative',child)
        return (type(node.op).__name__,form(node.left),form(node.right))
    def fraction_form(value):
        if value.denominator==1:return ('number',value)
        return ('Div',('number',Fraction(value.numerator)),('number',Fraction(value.denominator)))
    proper = [(node, v) for node, v in operations if node is not tree.body
              and v != value and form(node) != fraction_form(v)]
    first = proper[0] if proper else None  # First simplifying, non-whole proper node, left-to-right postorder.
    return Parsed(text, value, len(operations), bool(products),
        ast.get_source_segment(text, first[0]) if first else None, first[1] if first else None)


def maximal_lhs(past: str) -> str:
    """Never retry a shorter suffix after the maximal numeric region fails."""
    end = re.search(r'=\s*$', past)
    if end is None: raise Unsupported('cut_not_after_equals')
    left = past[:end.start()].rstrip(); match = MATH_SUFFIX.search(left)
    if match is None: raise Unsupported('no_numeric_LHS')
    raw = match.group(); expression = raw.strip(); before = left[:match.start()]
    if not expression: raise Unsupported('no_numeric_LHS')
    # A single-letter/escaped identifier before a numeric fragment, or any
    # immediately adjacent identifier, can be an omitted variable/operator.
    immediate = before[-1:] and not raw[:1].isspace()
    identifier = re.search(r'(\\?[^\W\d]\w*)$', before.rstrip(), flags=re.UNICODE)
    if (immediate and before[-1:].isalnum()) or (identifier and
            (len(identifier.group()) == 1 or identifier.group().lower() in FUNCTIONS
             or '\\' in identifier.group() or re.search(r'\d', identifier.group())
             or not identifier.group().isascii())):
        raise Unsupported('variable_or_identifier_suffix')
    if expression[0] in '+-\u2212' and before.rstrip() and before.rstrip()[-1] not in ':;\n':
        raise Unsupported('ambiguous_unary_suffix')
    if before.rstrip() and before.rstrip()[-1] in '%,/\\^*]}': raise Unsupported('unsupported_operator_before_suffix')
    return expression


def whole_value_visible(past: str, value: Fraction) -> str | None:
    """Conservative numeric-copy check, not semantic proof of a hidden answer.

    Whole decimal/rational spans, their components, scientific literals and
    percentage interpretations are checked. Ambiguous LaTeX/vulgar fractions or
    mixed-number spans are rejected. Computed expressions/prose paraphrases can
    still encode the answer; this filter does not claim to detect all of them.
    """
    if re.search(r'\\(?:[dt]?frac)\b|[\u00bc-\u00be\u2150-\u215e]', past):
        return 'ambiguous_fraction_copy_span'
    if any(c.isnumeric() and c not in '0123456789' for c in past):
        return 'ambiguous_unicode_numeric_copy_span'
    if re.search(rf'{NUM}\s+{NUM}\s*/\s*{NUM}', past):
        return 'ambiguous_mixed_number_copy_span'
    # Thousands/exponents are accepted here only to avoid missing visible
    # numbers; they are not accepted by the LHS or source-label grammar.
    atom = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
    atom = rf'(?:{atom})(?:[eE][+-]?\d+)?'
    if re.search(r'[eE]\s*\u2212\s*\d',past): return 'ambiguous_numeric_copy_span'
    for m in re.finditer(rf'[+\-\u2212]?\s*{atom}(?:\s*/\s*[+\-\u2212]?\s*{atom})?\s*%?', past):
        text = m.group().strip(); percent = text.endswith('%'); text = text.rstrip('%').strip()
        try:
            parts = text.split('/')
            if any(len(v)>80 or any(abs(int(e))>18 for e in re.findall(r'[eE]([+-]?\d+)',v)) for v in parts):
                return 'ambiguous_or_unbounded_numeric_copy_span'
            values = [Fraction(re.sub(r'\s+','',v).replace(',', '').replace('\u2212','-')) for v in parts]
            candidates = list(values) + [abs(v) for v in values]  # Binary/unary signs can be ambiguous in prose.
            v = values[0]/values[1] if len(values) == 2 else values[0]; candidates += [v, abs(v)]
            if percent: candidates += [v/100, abs(v)/100]
        except (ValueError, ZeroDivisionError): return 'ambiguous_numeric_copy_span'
        if value in candidates: return 'whole_value_already_visible'
    return None


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def note_pair(parsed, tok):
    make = lambda v: f'Compute {parsed.intermediate_expression} = {render_value(v)}.'
    correct = make(parsed.intermediate_value); ids = tok.encode(correct, add_special_tokens=False)
    for delta in [v for n in range(1, 33) for v in (n, -n)]:
        wrong = parsed.intermediate_value + delta
        if wrong == parsed.value: continue
        bad = make(wrong); bad_ids = tok.encode(bad, add_special_tokens=False)
        if len(bad_ids) == len(ids):
            return dict(correct_text=correct, correct_ids=ids, correct_value=render_value(parsed.intermediate_value),
                corrupted_text=bad, corrupted_ids=bad_ids, corrupted_value=render_value(wrong),
                correct_value_fraction=str(parsed.intermediate_value),corrupted_value_fraction=str(wrong),
                corruption_rule='First +1,-1,+2,-2,...,+32,-32 equal-token-length value; exclude whole answer', delta=delta)
    raise Unsupported('no_equal_token_length_corruption')


def search_start(document_id,item_id):
    h=canonical_hash(dict(salt=SALT,document_id=document_id,item_id=item_id,role='search_start'))
    return MIN_PAST+int(h,16)%65,h


def choose_boundary(prefix_ids, document_id, item_id, tok):
    """Stop at first eligible cut after fixed start; never decode its suffix."""
    start,start_hash=search_start(document_id,item_id);audit=[];cache={}
    for cut in range(start, len(prefix_ids)+1):
        token = prefix_ids[cut-1]
        if token not in cache: cache[token] = decode(tok, [token])
        if re.search(r'=\s*$', cache[token]) is None: continue
        try:
            past_ids = prefix_ids[:cut]; past = decode(tok, past_ids)
            if tok.encode(past, add_special_tokens=False) != past_ids:
                raise Unsupported('past_token_roundtrip_failed')
            expression = maximal_lhs(past); parsed = parse_numeric(expression)
            if parsed.binary_ops < 2 or not parsed.has_product: raise Unsupported('insufficient_operation_structure')
            if parsed.intermediate_value is None or parsed.intermediate_value == parsed.value:
                raise Unsupported('no_simplifying_nonwhole_proper_intermediate')
            reason = whole_value_visible(past, parsed.value)
            if reason: raise Unsupported(reason)
            pair = note_pair(parsed, tok)
            key = canonical_hash(dict(salt=SALT, document_id=document_id, item_id=item_id, cut=cut, expression=expression))
            candidate = dict(cut=cut, expression=expression, normalized_expression=parsed.expression,
                value=str(parsed.value), binary_ops=parsed.binary_ops, intermediate_expression=parsed.intermediate_expression,
                intermediate_value=str(parsed.intermediate_value), prefix_ids=past_ids, prefix_text=past,
                prefix_sha256=canonical_hash(past_ids), note=pair, selection_hash=key,
                search_start=start,search_start_hash=start_hash,selection_policy='First prefix-eligible cut after fixed hash start128..192; return immediately; no RHS retry')
            audit.append(dict(cut=cut,status='eligible',selection_hash=key))
            return candidate,audit
        except Unsupported as error: audit.append(dict(cut=cut, status='ineligible', reason=str(error)))
    return None,audit


def source_rhs(selected, source_ids, tok):
    """Curation only. A complete original-token label must have right lookahead."""
    cut = selected['cut']; past = selected['prefix_text']; full = decode(tok, source_ids)
    tail = decode(tok, source_ids[cut:])
    if full != past+tail or not full.startswith(past): raise Unsupported('source_cut_not_lossless')
    match = re.match(rf'\s*({RATIONAL})', tail)
    if match is None: raise Unsupported('unsupported_or_missing_RHS')
    literal = match.group(1); after = tail[match.end():]
    if not after: raise Unsupported('RHS_truncated_no_lookahead')
    # Do not bless a numeric prefix of a longer expression, exponent, mixed
    # number, decimal, identifier, percent, or unsupported mathematical form.
    next_text = after.lstrip()
    period = next_text.startswith('.') and len(next_text)>1 and next_text[1].isspace()
    math_continuation = ('0123456789.+-*/^_%\\([{!:<>\u00d7\u00f7\u2212\u2217\u22c5\u00b7\u2044\u2215\u2264\u2265\u2248\u2260\u00b2\u00b3')
    if (not next_text or (next_text[0] in math_continuation and not period) or
            unicodedata.category(next_text[0]) in {'Sm','No'} or re.match(r',\s*\d',next_text)
            or re.match(r'(?:[^\W\d]\b|times\b|plus\b|minus\b|over\b|divided\b|multiplied\b)',next_text) or
            (after[:1] and not after[0].isspace() and (after[0].isalnum() or after[0] == '_'))):
        raise Unsupported('unsupported_or_ambiguous_RHS_span')
    parts=re.sub(r'\s+', '', literal).replace('\u2212','-').split('/')
    value=bounded(Fraction(parts[0])/Fraction(parts[1]) if len(parts)==2 else Fraction(parts[0]))
    # Original token label includes only leading whitespace, the COMPLETE
    # numeric literal and optional trailing whitespace, never delimiter text.
    ends = []
    for width in range(1, min(MAX_RHS, len(source_ids)-cut)+1):
        label = decode(tok, source_ids[cut:cut+width])
        if label[:match.end()] == tail[:match.end()] and not label[match.end():].strip() and len(label) >= match.end():
            if decode(tok, source_ids[:cut+width]) == past+label: ends.append(width)
    if not ends: raise Unsupported('RHS_not_original_token_aligned_within16')
    width = min(ends)
    ids=source_ids[cut:cut+width];serialized=decode(tok,ids)
    mask=[bool(decode(tok,[t]).strip()) for t in ids]
    if (re.sub(r'\s+','',serialized)!=re.sub(r'\s+','',literal) or not any(mask)
            or not any(re.search(r'\d',decode(tok,[t])) for t in ids)):
        raise Unsupported('numeric_target_character_or_mask_mismatch')
    return dict(text=literal, value=str(value), token_ids=ids,numeric_token_mask=mask,
        serialized_text=serialized, token_count=width,
        source_char_start=len(past)+match.start(1), source_char_end=len(past)+match.end(1))


def curate_selected(record, selected, tok, source_file, source_sha256):
    """No alternate candidate is accepted if this already-selected RHS fails."""
    if record['split'] not in {'train', 'dev'}: raise ValueError('Never curate TEST')
    if len(record['prefix']) != SOURCE_PREFIX or len(record['future']) != SOURCE_FUTURE:
        raise ValueError('Expected original512+16 source tokens')
    if record['prefix'][:selected['cut']] != selected['prefix_ids']: raise ValueError('Chosen past changed')
    ordinary = record['prefix']+record['future']
    try:
        rhs = source_rhs(selected, ordinary, tok)
    except (Unsupported, ValueError, ZeroDivisionError) as error:
        return None, dict(status='unsupported_source_label', reason=str(error), selected=selected)
    if Fraction(rhs['value']) != Fraction(selected['value']):
        return None, dict(status='source_numeric_mismatch', reason='Complete_supported_exact_RHS_differs_from_LHS', rhs=rhs, selected=selected)
    row = dict(item_id=f"{record['id']}:arithmetic:{selected['cut']}", document_id=record['document_id'],
        split=record['split'], source_file=str(source_file), source_sha256=source_sha256,
        source_record_id=record['id'], source_record_sha256=canonical_hash(record), original_prefix_ids=record['prefix'],
        original_future_ids=record['future'], ordinary_source_ids=ordinary, ordinary_source_text=decode(tok, ordinary),
        prefix_ids=selected['prefix_ids'], prefix_text=selected['prefix_text'],
        writer_prefix_ids=selected['prefix_ids'], writer_prefix_text=selected['prefix_text'],
        natural_future16_ids=ordinary[selected['cut']:selected['cut']+16], numeric_target_ids=rhs['token_ids'],
        numeric_token_mask=rhs['numeric_token_mask'],
        note_ids=selected['note']['correct_ids'], note_text=selected['note']['correct_text'],
        corrupted_note_ids=selected['note']['corrupted_ids'], corrupted_note_text=selected['note']['corrupted_text'],
        natural_continuation_ids=ordinary[selected['cut']:], natural_continuation_text=decode(tok, ordinary[selected['cut']:]),
        verified_numeric_rhs=rhs, boundary=selected,
        supervision='External exact arithmetic from past operands; actual future used ONLY for declared source-label curation',
        ordinary_control='FULL ordinary source causal CE on identical accepted source528-token chunks; not future16-only')
    if (len(row['natural_future16_ids'])!=16 or row['numeric_target_ids']!=row['natural_future16_ids'][:len(row['numeric_target_ids'])]
            or len(row['numeric_token_mask'])!=len(row['numeric_target_ids']) or not any(row['numeric_token_mask'])):
        raise ValueError('Natural/source scalar token alignment changed')
    return row, dict(status='accepted', selected=selected, rhs=rhs)


def prefix_records(path):
    """Hashing may cover asset bytes, but TEST JSON/prefix/futures are unparsed."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip(): continue
            match = HEADER.match(line)
            if match is None: raise ValueError('Require canonical record header before any array parsing')
            item, doc, split = match.group(1,2,3)
            if split == 'test': continue
            prefix, _ = decoder.raw_decode(line, match.end())
            if len(prefix) != SOURCE_PREFIX or any(type(t) is not int or t < 0 for t in prefix):
                raise ValueError('Invalid original source prefix')
            yield dict(item_id=item, document_id=doc, split=split, prefix=prefix), line


def check_deadline(deadline):
    if datetime.now(timezone.utc)>=require_utc_deadline(deadline):
        raise TimeoutError('Finite arithmetic preparation deadline reached; partial files retained')


def prepare(source, output, tokenizer_dir, deadline_utc=None):
    """Full-corpus command; do not invoke until the prospective design is reviewed."""
    from transformers import AutoTokenizer
    source=Path(source).resolve();output=Path(output).resolve();tokenizer_dir=Path(tokenizer_dir).resolve()
    deadline_utc=deadline_utc or (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    check_deadline(deadline_utc)
    if sha256_file(source) != SOURCE_SHA: raise ValueError('Audited source snapshot differs')
    check_deadline(deadline_utc)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    output.mkdir(parents=True, exist_ok=False); accepted=[]; seen={}; counts={}
    try:
        with (output/'audit.jsonl').open('x') as log, (output/'accepted.jsonl').open('x') as stream:
            for header, line in prefix_records(source):
                check_deadline(deadline_utc);doc=header['document_id']
                if doc in seen: raise ValueError('Require upstream one-record/document and TRAIN/DEV disjointness')
                seen[doc]=header['split']
                selected, boundary_audit = choose_boundary(header['prefix'],doc,header['item_id'],tok)
                start,start_hash=search_start(doc,header['item_id'])
                audit=dict(item_id=header['item_id'],document_id=doc,split=header['split'],boundary_audit=boundary_audit,
                    search_start=start,search_start_hash=start_hash)
                if selected is None: audit.update(status='no_eligible_past_boundary')
                else:
                    check_deadline(deadline_utc)
                    record=json.loads(line)  # First future-array parse, after immutable hash choice.
                    row, status=curate_selected(record,selected,tok,source,SOURCE_SHA);audit.update(status)
                    if row is not None:
                        stream.write(json.dumps(row,allow_nan=False)+'\n');accepted.append(row)
                log.write(json.dumps(audit,allow_nan=False)+'\n');counts[audit['status']]=counts.get(audit['status'],0)+1
        check_deadline(deadline_utc)
    except BaseException as error:
        atomic_json(output/'preparation.failed.json',dict(status='partial_not_ready',deadline_utc=deadline_utc,
            error=f'{type(error).__name__}: {error}',counts=counts,
            partial_hashes={f.name:sha256_file(f) for f in (output/'accepted.jsonl',output/'audit.jsonl') if f.exists()}))
        raise
    selection={}
    for split,need in (('train',256),('dev',64)):
        values=sorted((r for r in accepted if r['split']==split),key=lambda r:canonical_hash(dict(salt=SALT,split=split,document_id=r['document_id'],item_id=r['item_id'])))
        selection[split]=dict(requested=need,available=len(values),selected_item_ids=[r['item_id'] for r in values[:need]],reserve_item_ids=[r['item_id'] for r in values[need:]])
    manifest=dict(schema=SCHEMA,status='complete_preparation_not_model_qualification',created_utc=utc_now(),deadline_utc=deadline_utc,
        source_file=str(source.resolve()),source_sha256=SOURCE_SHA,
        source_files={'train/calculation_boundaries.py':sha256_file(__file__)},tokenizer_dir=str(tokenizer_dir.resolve()),
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),selection_salt=SALT,selection=selection,counts=counts,
        boundary_selection='SHA256(salt,document_id,item_id,role=search_start) modulo65 +128, then first prefix-eligible cut; immediate return, no RHS quality retry',
        number_display='Exact terminating decimal when <=24 fractional digits; otherwise reduced fraction; integers unchanged',
        ready_for_feasibility=all(v['available']>=v['requested'] for v in selection.values()),
        accepted_sha256=sha256_file(output/'accepted.jsonl'),audit_sha256=sha256_file(output/'audit.jsonl'),
        limits=['Strict plaintext grammar and original-token roundtrip can reject legitimate math; unsupported is not false.',
            'Exact source_numeric_mismatch can reflect legitimate rounding/approximation conventions; it is not a blanket factual-error claim.',
            'Whole-answer filter detects literal/rational copies and rejects selected ambiguous notations, not all semantic disclosures.',
            'Original source window is512+16 tokens, not the full document. TRAIN/DEV split and document identities are unchanged.',
            'One fixed hash-start first-eligible stopping-time cut/document; failed source labels are never replaced; no model/benchmark/entropy selection.',
            'Unchanged rational identities are skipped structurally, independent of display; no reasoning utility follows from simplification.',
            'Future training/control protocol is not released; correct intermediate is external arithmetic supervision.'])
    atomic_json(output/'manifest.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer-dir',type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();prepare(a.source,a.output,a.tokenizer_dir,a.deadline_utc)


if __name__=='__main__': main()
