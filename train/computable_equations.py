"""Real-source numeric equalities for an explicitly full-answer-copy oracle.

The first causal prefix-eligible cut is fixed before RHS validation. No source
label failure is replaced. Notes copy the complete verified primary target
including its actual boundary, NOT a proper intermediate. Corruption changes
only the numeral substring and preserves the original leading/closing bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import re
import time

from train import calculation_boundaries as safe
from train.scalar_completion import parse_initial_scalar_v2
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = SALT = 'finemath-computable-equations-v1'
TRAIN_ITEMS, DEV_ITEMS = 5120, 512
SOURCE_SHA = safe.SOURCE_SHA
MAX_TARGET = 16


def sources():
    root = Path(__file__).resolve().parents[1]
    names = ('train/computable_equations.py', 'train/calculation_boundaries.py',
             'train/scalar_completion.py', 'weekend_thinking/artifacts.py')
    return {name: sha256_file(root/name) for name in names}


def search_start(document_id, item_id):
    digest = canonical_hash(dict(salt=SALT, document_id=document_id,
                                 item_id=item_id, role='search_start'))
    return 128 + int(digest, 16) % 65, digest


def choose_boundary(prefix_ids, document_id, item_id, tok):
    """Prefix-only stopping rule: return immediately, without reading a suffix."""
    start, digest = search_start(document_id, item_id)
    audit, pieces = [], {}
    for cut in range(start, len(prefix_ids)+1):
        token = prefix_ids[cut-1]
        if token not in pieces: pieces[token] = safe.decode(tok, [token])
        if re.search(r'=\s*$', pieces[token]) is None: continue
        try:
            ids = prefix_ids[:cut]; past = safe.decode(tok, ids)
            if tok.encode(past, add_special_tokens=False) != ids:
                raise safe.Unsupported('past_token_roundtrip_failed')
            expression = safe.maximal_lhs(past); parsed = safe.parse_numeric(expression)
            if parsed.binary_ops < 1: raise safe.Unsupported('no_binary_operation')
            reason = safe.whole_value_visible(past, parsed.value)
            if reason: raise safe.Unsupported(reason)
            equal = re.search(r'=\s*$', past)
            left = past[:equal.start()].rstrip()
            assert left.endswith(expression)
            lhs_start = len(left)-len(expression)
            selected = dict(cut=cut, prefix_ids=ids, prefix_text=past,
                prefix_sha256=canonical_hash(ids), expression=expression,
                normalized_expression=parsed.expression, numeric_value=str(parsed.value),
                binary_ops=parsed.binary_ops, lhs_char_start=lhs_start,
                reanchor_text=past[lhs_start:], search_start=start,
                search_start_hash=digest,
                selection_policy='First prefix-eligible causal cut after fixed hash start128..192; no RHS retry')
            audit.append(dict(cut=cut, status='eligible'))
            return selected, audit
        except safe.Unsupported as error:
            audit.append(dict(cut=cut, status='ineligible', reason=str(error)))
    return None, audit


def source_target(selected, original_ids, tok):
    """Complete source numeral plus the token covering its closing certificate.

    The frozen scalar grammar validates maximality using actual source lookahead
    (never treats an arbitrary source-window end as EOS). The target is the
    shortest ORIGINAL token prefix including the numeral and a line/sentence/
    equality/word boundary character. A token can contain additional source
    characters; none are split, re-encoded, or silently truncated.
    """
    cut = selected['cut']; past = selected['prefix_text']
    tail = safe.decode(tok, original_ids[cut:])
    if safe.decode(tok, original_ids) != past+tail:
        raise safe.Unsupported('source_cut_not_lossless')
    parsed = parse_initial_scalar_v2(tail, 'length_cap')
    if parsed['status'] != 'valid':
        raise safe.Unsupported('source_scalar_'+parsed['status'])
    value = safe.bounded(Fraction(parsed['value']))
    start = len(tail)-len(tail.lstrip(' \t\r\n'))
    literal = parsed['literal']; end = start+len(literal)
    if tail[start:end] != literal: raise safe.Unsupported('source_literal_offset_mismatch')
    following = tail[end:]; completion = parsed['completion']
    if completion == 'line':
        match = re.search(r'[\r\n]', following)
        if match is None: raise safe.Unsupported('missing_source_line_boundary')
        boundary = end+match.start()
    else:
        boundary = end+len(following)-len(following.lstrip(' \t\r\n'))
        expected = {'equality': '=', 'sentence': '.,;', 'word': None}
        if completion not in expected or boundary >= len(tail):
            raise safe.Unsupported('missing_actual_source_boundary')
        if expected[completion] and tail[boundary] not in expected[completion]:
            raise safe.Unsupported('source_boundary_kind_mismatch')
    widths = []
    for n in range(1, min(MAX_TARGET, len(original_ids)-cut)+1):
        text = safe.decode(tok, original_ids[cut:cut+n])
        if tail.startswith(text) and len(text) > boundary: widths.append(n)
    if not widths: raise safe.Unsupported('complete_numeral_and_boundary_not_within16_tokens')
    n = min(widths); ids = original_ids[cut:cut+n]
    text = safe.decode(tok, ids); offsets = [0]
    for i in range(1, n+1):
        part = safe.decode(tok, ids[:i])
        if not text.startswith(part): raise safe.Unsupported('nonmonotonic_target_token_decode')
        offsets.append(len(part))
    mask = [a < end and b > start and bool(tail[max(a,start):min(b,end)].strip())
            for a,b in zip(offsets, offsets[1:])]
    if not any(mask): raise safe.Unsupported('no_numeric_target_token')
    closing_start = next(i for i,(a,b) in enumerate(zip(offsets,offsets[1:])) if a <= boundary < b)
    return dict(token_ids=ids, numeric_token_mask=mask, value=str(value),
        literal_text=literal, serialized_text=text, token_count=n,
        numeral_char_start=start, numeral_char_end=end,
        closing_boundary_char=tail[boundary], closing_boundary_char_offset=boundary,
        closing_boundary_source_char=len(past)+boundary,
        closing_boundary_token_start=closing_start, closing_boundary_ids=ids[closing_start:],
        completion_kind=completion, boundary_token_mask=[a <= boundary < b
            for a,b in zip(offsets, offsets[1:])],
        source_char_start=len(past)+start, source_char_end=len(past)+end)


def corrupt_surface(target, tok):
    """Fixed numeral mutation in full source target; no score-based search.

    Prefer equal native token length; if no digit mutation has that length,
    retain the first valid one with an explicit flag. Never filter a document
    merely to obtain equal note lengths.
    """
    literal=target['literal_text'];value=Fraction(target['value'])
    source_text=target['serialized_text'];first=None
    start,end=target['numeral_char_start'],target['numeral_char_end']
    for index in range(len(literal)-1, -1, -1):
        if literal[index] not in '0123456789': continue
        for digit in '0123456789':
            if digit == literal[index]: continue
            text = literal[:index]+digit+literal[index+1:]
            parsed = parse_initial_scalar_v2(text, 'eos')
            if parsed['status'] != 'valid': continue
            wrong = Fraction(parsed['value'])
            if wrong == value: continue
            try: safe.bounded(wrong)
            except safe.Unsupported: continue
            full=source_text[:start]+text+source_text[end:]
            encoded = tok.encode(full, add_special_tokens=False)
            if safe.decode(tok, encoded) != full: continue
            candidate=dict(text=full,ids=encoded,value=str(wrong),numeral_text=text,
                equal_token_length=len(encoded)==len(target['token_ids']),
                rule='Rightmost digit first; replacement0..9; first different bounded value at equal native length, otherwise first valid mutation')
            if first is None:first=candidate
            if candidate['equal_token_length']:return candidate
    if first is not None:return first
    raise ValueError('No valid numeric digit corruption; fail preparation, never drop/replace item')


def curate(record, selected, tok, source_file, source_sha):
    if record['split'] not in {'train','dev'}: raise ValueError('Never curate TEST')
    if len(record['prefix']) != 512 or len(record['future']) != 16:
        raise ValueError('Expected immutable512+16 source chunk')
    if record['prefix'][:selected['cut']] != selected['prefix_ids']:
        raise ValueError('Chosen prefix changed')
    original = record['prefix']+record['future']
    try: target = source_target(selected, original, tok)
    except (safe.Unsupported, ValueError, ZeroDivisionError) as error:
        return None, dict(status='unsupported_source_label', reason=str(error))
    if target['value'] != selected['numeric_value']:
        return None, dict(status='source_numeric_mismatch', target=target)
    literal = target['literal_text']; copy_ids = tok.encode(literal, add_special_tokens=False)
    if not copy_ids or safe.decode(tok, copy_ids) != literal:
        raise ValueError('Full-answer note does not roundtrip')
    bad = corrupt_surface(target, tok)
    reanchor = selected['reanchor_text']; reanchor_ids = tok.encode(reanchor, add_special_tokens=False)
    if safe.decode(tok, reanchor_ids) != reanchor: raise ValueError('Reanchor bytes changed')
    natural = original[selected['cut']:selected['cut']+16]
    if len(natural) != 16 or target['token_ids'] != natural[:target['token_count']]:
        raise ValueError('Native target is not exact natural16 prefix')
    row = dict(item_id=f"{record['id']}:equation:{selected['cut']}",
        document_id=record['document_id'], split=record['split'],
        prefix_ids=selected['prefix_ids'], prefix_text=selected['prefix_text'],
        natural_future16_ids=natural, target_ids=target['token_ids'],
        numeric_token_mask=target['numeric_token_mask'], numeric_value=target['value'],
        correct_note_ids=target['token_ids'],correct_note_text=target['serialized_text'],
        copy_note_ids=target['token_ids'],copy_note_text=target['serialized_text'],
        numeric_surface_ids=copy_ids,numeric_surface_text=literal,
        corrupt_note_ids=bad['ids'], corrupt_note_text=bad['text'], corrupt_numeric_value=bad['value'],
        corrupt_numeric_surface_text=bad['numeral_text'],equal_note_token_length=bad['equal_token_length'],
        corruption_rule=bad['rule'], reanchor_ids=reanchor_ids, reanchor_text=reanchor,
        verified_numeric_rhs=target, boundary=selected,
        closing_boundary_ids=target['closing_boundary_ids'],
        closing_boundary_token_start=target['closing_boundary_token_start'],
        source_file=str(Path(source_file).resolve()), source_sha256=source_sha,
        source_record_id=record['id'], source_record_sha256=canonical_hash(record),
        original_source_ids=original, original_source_text=safe.decode(tok, original),
        supervision='FULL externally computed answer; source RHS validates equality and supplies exact numeral surface; NOT a proper intermediate',
        primary_target='Complete original numeral PLUS actual closing-boundary token; correct note deliberately copies this FULL span',
        ordinary_control='Identical complete original528-token source causal CE, not future16-only')
    return row, dict(status='accepted')


def select_pools(rows):
    result = {}
    for split, count in (('train',TRAIN_ITEMS),('dev',DEV_ITEMS)):
        ordered = sorted((r for r in rows if r['split']==split), key=lambda r:
            canonical_hash(dict(salt=SALT, role='pool', split=split,
                                document_id=r['document_id'], item_id=r['item_id'])))
        result[split] = dict(requested=count, available=len(ordered),
            selected_item_ids=[r['item_id'] for r in ordered[:count]],
            reserve_item_ids=[r['item_id'] for r in ordered[count:]])
    return result


def prepare(source, output, tokenizer_dir, deadline_utc):
    from transformers import AutoTokenizer
    source=Path(source).resolve(); output=Path(output).resolve()
    tokenizer_dir=Path(tokenizer_dir).resolve(); deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('CPU preparation deadline; partials retained')
    check(); source_files=sources()
    if sha256_file(source) != SOURCE_SHA: raise ValueError('Audited source SHA differs')
    check(); tok=AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    output.mkdir(parents=True, exist_ok=False); counts=Counter(); seen=set(); accepted=[]
    split_counts={name:Counter() for name in ('train','dev')}
    started=time.monotonic()
    try:
        with (output/'audit.jsonl').open('x') as log, (output/'accepted.jsonl').open('x') as stream:
            for header, line in safe.prefix_records(source):
                check(); doc=header['document_id']
                if doc in seen: raise ValueError('Repeated document or split overlap')
                seen.add(doc)
                chosen, reasons=choose_boundary(header['prefix'],doc,header['item_id'],tok)
                audit=dict(item_id=header['item_id'],document_id=doc,split=header['split'],boundary_audit=reasons)
                if chosen is None: audit['status']='no_eligible_past_boundary'
                else:
                    # First original future-array parse occurs only after stopping.
                    row, status=curate(json.loads(line),chosen,tok,source,SOURCE_SHA)
                    audit.update(status);audit['selected']=chosen
                    if row is not None:
                        stream.write(json.dumps(row,allow_nan=False)+'\n');accepted.append(row)
                counts[audit['status']]+=1;split_counts[header['split']][audit['status']]+=1
                log.write(json.dumps(audit,allow_nan=False)+'\n')
                if len(seen)%5000==0:
                    print(json.dumps(dict(documents=len(seen),counts=counts,elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
        check()
        if sources()!=source_files or sha256_file(source)!=SOURCE_SHA:
            raise ValueError('Source changed during preparation')
    except BaseException as error:
        atomic_json(output/'preparation.failed.json',dict(status='partial_not_ready',error=f'{type(error).__name__}: {error}',
            counts=dict(counts),documents=len(seen),source_files=source_files))
        raise
    pools=select_pools(accepted); dev=pools['dev']['selected_item_ids']
    manifest=dict(schema=SCHEMA,status='complete_CPU_preparation_not_model_or_training_qualification',
        created_utc=utc_now(),deadline_utc=deadline_utc,elapsed_seconds=time.monotonic()-started,
        source_file=str(source),source_sha256=SOURCE_SHA,source_files=source_files,
        tokenizer_dir=str(tokenizer_dir),tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),
        accepted_file=str(output/'accepted.jsonl'),accepted_sha256=sha256_file(output/'accepted.jsonl'),
        audit_file=str(output/'audit.jsonl'),audit_sha256=sha256_file(output/'audit.jsonl'),
        documents=len(seen),counts=dict(counts),counts_by_split={k:dict(v) for k,v in split_counts.items()},
        selection_salt=SALT,selection=pools,
        selected_train_item_ids=pools['train']['selected_item_ids'],selected_dev_item_ids=dev,
        phase1_dev_item_ids=dev[:256],held_post_sft_dev_item_ids=dev[256:512],
        ready_for_requested_pools=all(p['available']>=p['requested'] for p in pools.values()),
        boundary_selection='First prefix-eligible cut after hash-start128..192, before selected-RHS validation; no fallback',
        target_definition='Shortest native continuation prefix covering complete verified numeral plus actual local closing certificate; <=16tokens',
        note_definition='EXACT full verified primary target including leading/closing bytes; FULL-ANSWER/COPY oracle, NOT proper intermediate',
        limitations=['Exact numeric LHS/RHS subset, not all FineMath; unsupported and exact numeric mismatch are distinct.',
            'V2 bounded copy checks reject known contextual-dot/grouping ambiguities; not exhaustive semantic disclosure detection.',
            'Only one existing528-token window/document; no new raw-document or TEST array access.',
            'Whole-target likelihood primary; overlap-mask numeric likelihood secondary; tokens containing both numeral and boundary cannot be split.',
            'Fixed5120TRAIN/512DEV quotas are not shrunk if short; separate full-document identity needed on shortfall.',
            'Full-answer arithmetic oracle/target-copy plumbing is not deployed reasoning or a no-answer partial-note result.'])
    atomic_json(output/'manifest.json',manifest)
    print(json.dumps(dict(manifest=str(output/'manifest.json'),counts=dict(counts),
                         pools={k:dict(available=v['available'],requested=v['requested']) for k,v in pools.items()})),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','tokenizer-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();prepare(a.source,a.output,a.tokenizer_dir,a.deadline_utc)


if __name__=='__main__':main()
