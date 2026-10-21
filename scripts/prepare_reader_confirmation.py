"""Prepare fixed fresh calculation and unenriched native-window panels on CPU.

Natural sources and their cuts depend only on source identity and length. No
candidate label, predictive score or serialization outcome selects/replaces a
natural window. Every chosen row must validate or preparation stops.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

SALT = 'reader-fresh-confirmation-20261021'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''): h.update(block)
    return h.hexdigest()


def priority(doc, role):
    return hashlib.sha256((SALT+':'+role+':'+doc).encode()).hexdigest()


def select_natural(rows, count):
    """Candidate flags are deliberately absent from the selection key."""
    eligible = [dict(document_id=r['document_id'], cut=r['chosen_cut'], positions=r['positions'])
                for r in rows if not r['used'] and r['positions'] > 0]
    eligible.sort(key=lambda r: priority(r['document_id'], 'natural'))
    if len(eligible) < count: raise ValueError('Insufficient natural source identities')
    return eligible[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args(); p = json.loads(args.plan.read_text())
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None or p['identity'] != SALT: raise ValueError('Frozen finite plan')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4]); started = time.monotonic()
    def check():
        if datetime.now(timezone.utc) >= deadline: raise TimeoutError('Finite source preparation')
        if shutil.disk_usage(args.output.parent).free < 35*2**30: raise RuntimeError('Disk reserve')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 512*2**30: raise RuntimeError('Host-memory reserve')
        if args.output.exists() and sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file()) > 256*2**20:
            raise RuntimeError('Finite source output')
    def bound(spec):
        if sha(spec['path']) != spec['sha256']: raise ValueError('Changed input: '+spec['path'])
        return Path(spec['path'])
    if sha(__file__) != p['preparer_sha256']: raise ValueError('Frozen preparer source')
    check(); args.output.mkdir(exist_ok=False)
    projection = bound(p['census_projection'])
    with gzip.open(projection, 'rt') as stream:
        natural = select_natural((json.loads(line) for line in stream), p['natural_documents'])
    natural_ids = {r['document_id'] for r in natural}
    reserve = [json.loads(line) for line in bound(p['strict_reserve']).open()]
    calculations = [r for r in reserve if r['document_id'] not in natural_ids]
    calculations.sort(key=lambda r: priority(r['document_id'], 'calculation'))
    if len(calculations) < p['calculation_documents']:
        raise ValueError('Source allocation is insufficient; never replace a natural row')
    calculations = calculations[:p['calculation_documents']]
    chosen = {r['document_id']: dict(r, kind='calculation', cut=r['provenance']['absolute_source_cut'])
              for r in calculations}
    chosen.update({r['document_id']: dict(r, kind='natural') for r in natural})
    if len(chosen) != p['natural_documents']+p['calculation_documents']:
        raise ValueError('Panels require distinct source documents')
    # Freeze selected identities and positions before raw-source/token checks.
    selection = {doc: dict(kind=r['kind'], cut=r['cut']) for doc, r in chosen.items()}
    (args.output/'selection.json').write_text(json.dumps(selection, sort_keys=True)+'\n')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(p['tokenizer_directory'], local_files_only=True,
                                              trust_remote_code=False)
    sources = {}; counts = Counter()
    for spec in p['source_pools']:
        membership = bound(spec['membership']); raw_path = bound(spec['raw'])
        metadata = {}
        for line in membership.open():
            m = json.loads(line)
            if m['document_id'] in chosen:
                if m['document_id'] in sources: raise ValueError('Repeated source identity')
                metadata[m['document_id']] = m
        with raw_path.open('rb') as stream:
            for doc, m in metadata.items():
                check(); r = chosen[doc]; stream.seek(m['raw_offset']); line = stream.read(m['raw_bytes'])
                raw = json.loads(line); text = raw['text']
                if raw['url'] != m['url'] or hashlib.sha256(' '.join(text.split()).encode()).hexdigest() != doc:
                    raise ValueError('Native source identity mismatch')
                tokens = tokenizer(text[:65536], add_special_tokens=False, truncation=True,
                                   max_length=8208, return_attention_mask=False)['input_ids']
                cut = r['cut']; native = tokens[cut-512:cut+16]
                if not 512 <= cut <= min(8192, len(tokens)-16) or len(native) != 528:
                    raise ValueError('Chosen cut does not have its complete native window')
                if r['kind'] == 'calculation' and (native[:512] != r['prefix_ids'] or native[512:] != r['continuation16_ids']):
                    raise ValueError('Stored arithmetic view differs from original native source')
                # These are real native token windows. No decode/re-encode step
                # alters partial UTF8 tokens at a boundary on natural windows.
                if tokenizer.eos_token_id in native[:512]:
                    raise ValueError('Chosen real source contains EOS in prefix; retain selection and stop, never replace')
                row = dict(item_id=doc+':confirm:'+str(cut), document_id=doc,
                    kind=r['kind'], split='fresh_confirmation', prefix_ids=native[:512],
                    continuation16_ids=native[512:], stop_id=tokenizer.eos_token_id,
                    provenance=dict(source_pool=spec['name'], source_sha256=spec['raw']['sha256'],
                        raw_source_line=m['source_line'], raw_line_sha256=hashlib.sha256(line).hexdigest(),
                        source_url=m['url'], absolute_source_cut=cut,
                        source_locator=dict(filename=m['parquet_filename'], parquet_row=m['parquet_row'])))
                if r['kind'] == 'calculation':
                    for key in ('normalized_expression', 'full_calculation_text', 'full_calculation_ids', 'numeric_value'):
                        row[key] = r[key]
                sources[doc] = row; counts[r['kind']] += 1
    if set(sources) != set(chosen): raise ValueError('A chosen source is missing')
    files = {}
    for name, selected in [('CONFIRM_natural', natural), ('CONFIRM_calculation', calculations)]:
        labels = args.output/(name+'.jsonl'); questions = args.output/(name+'-questions.jsonl')
        with labels.open('x') as out, questions.open('x') as writer:
            for selected_row in selected:
                r = sources[selected_row['document_id']]
                out.write(json.dumps(r, separators=(',', ':'))+'\n')
                writer.write(json.dumps({k:r[k] for k in ('item_id', 'document_id', 'prefix_ids')}, separators=(',', ':'))+'\n')
        files[name] = dict(rows=len(selected), labels_sha256=sha(labels), questions_sha256=sha(questions))
    result = dict(status='COMPLETE_FRESH_READER_SOURCE_PANELS_NOT_MODEL_RELEASE',
        plan_sha256=sha(args.plan), selection_sha256=sha(args.output/'selection.json'),
        cohorts=files, strict_calculation_reserve_after_natural_exclusion=len(reserve)-len(natural_ids & {r['document_id'] for r in reserve}),
        counts=dict(counts), wall_seconds=time.monotonic()-started, model_calls=0,
        accelerator_chip_hours=0,
        limits='Fresh sources, not novelty relative to base-model pretraining. Natural source/cut selection ignores calculation labels; no label-driven row replacement. Independent pinned-Parquet validation remains required.')
    (args.output/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__': main()
