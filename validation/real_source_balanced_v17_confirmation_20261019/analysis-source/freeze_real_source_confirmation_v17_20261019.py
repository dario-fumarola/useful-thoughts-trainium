"""Freeze source-TRAIN confirmation after complete,unchanged equation curation."""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
from urllib.parse import urlsplit,unquote
from transformers import AutoTokenizer
from train import computable_equations_raw as curator
from train.calculation_boundaries import maximal_lhs,parse_numeric
from train.symbolic_note_labels import postorder_steps

R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data'
W=R/'local/worktrees/real-prefix-partial-v9'
assert Path(curator.__file__).resolve()==W/'train/computable_equations_raw.py'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
SALT='real-source-balanced-v17-confirmation-fixed-hash-v1'


def page(url):
    p=urlsplit(url)
    return p.netloc.casefold().removeprefix('www.')+unquote(p.path).rstrip('/').casefold()


def expression_key(expression):
    def walk(node):
        if isinstance(node,ast.Constant):
            assert type(node.value) in (int,float)
            return ('number',str(Fraction(ast.get_source_segment(expression,node))))
        if isinstance(node,ast.UnaryOp):
            a=walk(node.operand)
            if isinstance(node.op,ast.UAdd):return a
            assert isinstance(node.op,ast.USub)
            return ('number',str(-Fraction(a[1]))) if a[0]=='number' else ('negate',a)
        assert isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow))
        return (type(node.op).__name__,walk(node.left),walk(node.right))
    return walk(ast.parse(expression,mode='eval').body)


def exact_value(key):
    if key[0]=='number':return Fraction(key[1])
    if key[0]=='negate':return -exact_value(key[1])
    a,b=exact_value(key[1]),exact_value(key[2])
    if key[0]=='Add':return a+b
    if key[0]=='Sub':return a-b
    if key[0]=='Mult':return a*b
    if key[0]=='Div':return a/b
    assert key[0]=='Pow' and b.denominator==1 and abs(b.numerator)<=100
    return a**b.numerator


def rank(kind,doc):return hashlib.sha256(f'{SALT}:{kind}:{doc}'.encode()).hexdigest()


if __name__=='__main__':
    rule=read(D/'real-source-balanced-v17-confirmation-extension-rule.json')
    assert rule['arithmetic_confirmation_documents']==512 and rule['ordinary_confirmation_documents']==256
    sources=[]
    for number in rule['shard_order']:
        suffix='' if number==63 else f'-shard{number:02d}'
        raw=D/f'real-source-balanced-v17-fresh-source{suffix}-v1'
        curated=D/f'real-source-balanced-v17-fresh-curation{suffix}-v1'
        if not (curated/'completion.json').exists():break
        c=read(curated/'completion.json');f=read(raw/'filtered-complete.json')
        assert c['status']=='COMPLETE_FRESH_COMPOUND_CURATION_NO_COHORT_OR_MODEL_RELEASE'
        assert sha(curated/'accepted.jsonl')==c['accepted_sha256']
        assert sha(f['raw_file'])==f['raw_sha256'] and sha(f['membership_file'])==f['membership_sha256']
        sources.append(dict(shard=number,raw=raw,curated=curated,filtered=f))
    assert sources and sources[0]['shard']==63
    previous=[D/'real-prefix-partial-labels-v9/TRAIN.jsonl',D/'real-prefix-partial-labels-v9/DEV.jsonl',
        D/'real-trace-cpt-v10-held-panels-v2/DEV_novel_expression256.jsonl',D/'real-target-sft-v15-held-v1/DEV_fresh384.jsonl']
    forbidden,prior_docs=set(),set()
    for p in previous:
        for row in rows(p):forbidden.add(expression_key(row['normalized_expression']));prior_docs.add(row['document_id'])
    tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True)
    candidates=[];exclusions=Counter();source_pins={}
    for source in sources:
        file=source['curated']/'accepted.jsonl';source_pins[str(file)]=sha(file)
        for line_number,line in enumerate(file.open(),1):
            row=json.loads(line)
            assert row['source_sha256']==source['filtered']['raw_sha256']
            assert row['document_id'] not in prior_docs
            assert tok.decode(row['prefix_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==row['prefix_text']
            expression=parse_numeric(maximal_lhs(row['prefix_text'])).expression
            key=expression_key(expression)
            value,steps=postorder_steps(expression)
            assert exact_value(key)==value==Fraction(row['numeric_value'])
            if key in forbidden:exclusions['prior_ordered_AST']+=1;continue
            partial='\n'.join(steps[:-1])+'\n';full='\n'.join(steps)+'\n'
            partial_ids=tok.encode(partial,add_special_tokens=False);full_ids=tok.encode(full,add_special_tokens=False)
            if len(partial_ids)+1>64:exclusions['same_partial64_fit_rule']+=1;continue
            if len(full_ids)+1>128:exclusions['full_note_stop_exceeds128']+=1;continue
            if len(row['prefix_ids'])+len(tok.encode('\nCalculation:\n',add_special_tokens=False))+len(full_ids)+2+16>672:
                exclusions['reader_width672']+=1;continue
            result=dict(item_id=row['item_id'],document_id=row['document_id'],split='followup_dev',
                prefix_ids=row['prefix_ids'],prefix_text=row['prefix_text'],continuation16_ids=row['natural_future16_ids'],
                numeric_surface_ids=row['numeric_surface_ids'],numeric_surface_text=row['numeric_surface_text'],
                closing_boundary_ids=row['closing_boundary_ids'],numeric_value=row['numeric_value'],
                normalized_expression=expression,partial_note_text=partial,partial_note_ids=partial_ids,
                full_calculation_text=full,full_calculation_ids=full_ids,stop_id=tok.eos_token_id,
                source_accepted_file=str(file),source_accepted_line=line_number,
                source_accepted_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                source_provenance={k:row[k] for k in ['raw_line_sha256','raw_text_sha256','raw_source_line','source_url',
                    'prefix_source_char_end','source_chunk_char_start','source_chunk_char_end','absolute_source_cut','raw_rhs_validation']})
            candidates.append((rank('arithmetic',row['document_id']),key,page(row['source_url']),result))
    selected=[];pages=set();expressions=set();documents=set()
    for _,key,url,row in sorted(candidates,key=lambda v:(v[0],v[3]['item_id'])):
        if url in pages or key in expressions or row['document_id'] in documents:continue
        selected.append(row);pages.add(url);expressions.add(key);documents.add(row['document_id'])
        if len(selected)==512:break
    if len(selected)<512:
        f=D/f'real-source-balanced-v17-confirmation-insufficient-through-shard{sources[-1]["shard"]}.json'
        with f.open('x') as stream:json.dump(dict(status='INSUFFICIENT_FIXED_CONFIRMATION_NO_COHORT',
            processed_shards=[v['shard'] for v in sources],unique_fitting_pages_and_ASTs=len(selected),
            exclusions=dict(exclusions),next_action='Follow declared next shard;do not relax eligibility or use model outcomes.'),stream,indent=2)
        print(f.read_text(),flush=True);raise SystemExit(2)
    # Select ordinary documents independently by fixed hash;exclude every selected
    # arithmetic page and document before any text/likelihood eligibility checks.
    metadata=[];raw_handles={};offsets={};raw_shas={}
    for source in sources:
        f=source['filtered'];raw_file=Path(f['raw_file']);positions=[]
        with raw_file.open('rb') as stream:
            while True:
                pos=stream.tell();line=stream.readline()
                if not line:break
                positions.append(pos)
        offsets[str(raw_file)]=positions;raw_handles[str(raw_file)]=raw_file.open('rb')
        raw_shas[str(raw_file)]=f['raw_sha256']
        for row in rows(f['membership_file']):
            if row['document_id'] not in documents and page(row['url']) not in pages:
                metadata.append((rank('ordinary',row['document_id']),str(raw_file),row))
    ordinary=[];ordinary_pages=set();ordinary_docs=set();ordinary_exclusions=Counter()
    try:
        for _,file,meta in sorted(metadata,key=lambda v:(v[0],v[2]['document_id'])):
            url=page(meta['url']);doc=meta['document_id']
            if url in ordinary_pages or doc in ordinary_docs:continue
            stream=raw_handles[file];stream.seek(offsets[file][meta['source_line']-1]);line=stream.readline()
            raw=json.loads(line);text=raw['text'][:curator.MAX_CHARACTERS]
            encoded=tok(text,add_special_tokens=False,truncation=True,max_length=curator.MAX_TOKENS,
                return_offsets_mapping=True,return_attention_mask=False)
            ids=encoded['input_ids'];count=len(ids)-512-16+1
            if count<1:ordinary_exclusions['too_short_for512plus16']+=1;continue
            cut=512+int(rank('ordinary_cut',doc),16)%count
            try:window=curator.source_window(ids,encoded['offset_mapping'],text,cut,tok)
            except curator.base.safe.Unsupported:ordinary_exclusions['not_exact_native_UTF8_window']+=1;continue
            if set(window['ids'])&set(tok.all_special_ids):ordinary_exclusions['special_token']+=1;continue
            ordinary.append(dict(item_id=f'{doc}:ordinary-v17:{cut}',document_id=doc,split='followup_dev',
                prefix_ids=window['ids'][:512],continuation16_ids=window['ids'][512:528],
                source_file=file,source_sha256=raw_shas[file],raw_source_line=meta['source_line'],source_url=meta['url'],
                source_chunk_char_start=window['char_start'],source_chunk_char_end=window['char_end'],
                raw_line_sha256=hashlib.sha256(line).hexdigest(),absolute_source_cut=cut))
            ordinary_pages.add(url);ordinary_docs.add(doc)
            if len(ordinary)==256:break
    finally:
        for handle in raw_handles.values():handle.close()
    assert len(ordinary)==256 and not documents&ordinary_docs and not pages&ordinary_pages
    O=D/'real-source-balanced-v17-confirmation-v1';O.mkdir(exist_ok=False);files={}
    for name,values in [('DEV_confirm512.jsonl',selected),('DEV_ordinary256.jsonl',ordinary),
                        ('DEV_confirm512-questions.jsonl',[{k:r[k] for k in ['item_id','document_id','prefix_ids']} for r in selected])]:
        file=O/name
        with file.open('x') as stream:
            for value in values:stream.write(json.dumps(value,separators=(',',':'))+'\n')
        files[name]=dict(rows=len(values),bytes=file.stat().st_size,sha256=sha(file))
    manifest=dict(status='FROZEN_FRESH_SOURCE_TRAIN_CONFIRMATION_NO_MODEL_FORWARD',utc=datetime.now(timezone.utc).isoformat(),
        schema='real-source-balanced-v17-confirmation-v1',script_sha256=sha(__file__),selection_salt=SALT,
        completed_shards=[s['shard'] for s in sources],source_accepted_files=source_pins,
        prior_files={str(p):sha(p) for p in previous},curator_source_files=curator.sources(),
        arithmetic_documents=512,ordinary_documents=256,arithmetic_exclusions=dict(exclusions),ordinary_exclusions=dict(ordinary_exclusions),
        trace_operations=dict(Counter(len(v['full_calculation_text'].splitlines()) for v in selected)),files=files,
        limits=['New public source TRAIN-derived confirmation,not officialTEST. No model outputs or scores select documents.',
                'Ordered exact-value ASTs and canonical pages exclude prior retained data;one page/expression per arithmetic item. Operator patterns/equal answers can recur.',
                'Old TEST/unlisted raw text remains opaque;prior identifiers used only for exclusion. Unknown semantic duplicates and pretraining exposure are not ruled out.',
                'Source RHS verification is supervised curation;teacher traces receive only the causal prefix. Writer projection contains only identifiers/prefix IDs.',
                'Ordinary documents use one deterministic complete512+16 native window from bounded source text;all panel denominators fixed before new learning.'])
    (O/'manifest.json').write_text(json.dumps(manifest,indent=2))
    A=R/'local/backups/data/real-source-balanced-v17-confirmation-v1';A.mkdir(exist_ok=False);receipts={}
    for source in [*O.iterdir(),Path(__file__)]:
        assert shutil.disk_usage(R).free>35*2**30+source.stat().st_size
        target=A/source.name;shutil.copyfile(source,target)
        with target.open('rb') as stream:os.fsync(stream.fileno())
        assert sha(source)==sha(target);receipts[target.name]=sha(target)
    (A/'READBACK.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(dict(status=manifest['status'],completed_shards=manifest['completed_shards'],files=files),indent=2),flush=True)
