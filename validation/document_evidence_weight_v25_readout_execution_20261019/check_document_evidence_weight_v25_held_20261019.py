"""Independently recompute V25 held answers and paired counts from native tokens."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import string
import numpy as np
from transformers import AutoTokenizer

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def normalize(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())

def parse(text,mode,terminated,capped):
    start=0
    terminal=terminated or capped
    if mode=='prior':
        fields=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
        if len(fields)!=1:return ('unsupported' if terminal else 'pending'),''
        start=fields[0].end()
    else:assert mode=='direct'
    tail=text[start:].lstrip()
    if '\n' in tail:return ('complete',tail.split('\n',1)[0].strip()) if tail.split('\n',1)[0].strip() else ('unsupported','')
    if terminated and tail.strip():return 'complete',tail.strip()
    return ('unsupported' if terminal else 'pending'),''

def f1(a,b):
    a,b=normalize(a).split(),normalize(b).split()
    return float(a==b) if not a or not b else 2*sum((Counter(a)&Counter(b)).values())/(len(a)+len(b))

report=read(C/'held-native-and-replay-result.json')
scored=read(C/'held-native-and-replay-scored.json')
assert report['status']=='COMPLETE_DEVELOPMENTAL_WEIGHT_V25_NATIVE_AND_CLOSED_FIELD_ANSWERS'
gold={v['item_id']:v for v in rows(D/'held255-gold-only.jsonl')}
assert len(gold)==255
manifest=read(D/'manifest.json')
assert sha(D/'held255-gold-only.jsonl')==manifest['files']['held_gold']['sha256']
recomputed={};native_checks=0;attempts_total=0
for kind,count in [('native',510),('answer-replay',765)]:
    O=R/f'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-{kind}-v1'
    control=C/('HELD-native' if kind=='native' else 'HELD-replay')
    assert read(control/'GEN/closure.json')['actual_attempts']==count
    p=read(O/'protocol.json');tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    assert sha(O/'protocol.json')==report['inputs'][kind+'_protocol_sha256']
    for name,value in p['files'].items():assert sha(O/name)==value
    allrows=[]
    for shard in range(4):
        q=O/'answers'/f'shard-{shard}';run=read(q/'run.json')
        assert run['status']=='complete_all_attempts' and run['attempts_sha256']==sha(q/'attempts.jsonl')
        assert read(q/'native.json')['passed'] and read(q/'cache-prefill.json')['max_logp_error']<=.001
        native_checks+=1
        values=rows(q/'attempts.jsonl');assert len(values)==run['rows']
        assert {(v['mode'],v['item_id']) for v in values}=={(m,k) for m in p['modes'] for k in p['ordered_item_ids'][shard::4]}
        allrows.extend(values)
    assert len(allrows)==count;attempts_total+=count
    for mode in p['modes']:
        name=kind+':'+mode;values=[v for v in allrows if v['mode']==mode]
        assert len(values)==len({v['item_id'] for v in values})==255
        assert {v['item_id'] for v in values}==set(gold)
        result={};statuses=Counter();caps=errors=prefixids=eh=fh=brackets=0
        for v in values:
            key=v['item_id'];g=gold[key];s=scored[name][key]
            assert v==s['native'] and v['n_actions']==len(v['actions'])<=128
            assert v['content_ids']==(v['actions'][:-1] if v['stop_type'].startswith('token:') else v['actions'])
            assert tok.decode(v['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==v['raw_text']
            expected_mode='prior' if kind=='native' else 'direct';assert v['parser_mode']==expected_mode
            status,answer=parse(v['raw_text'],expected_mode,v['stop_type'].startswith('token:'),v['stop_type']=='length_cap')
            assert status==s['answer']['status'] and answer==s['answer']['answer']
            if status=='complete' and v['status']=='ok':
                alternatives=[g['answer'],*g['answer_aliases']]
                metrics=dict(canonical_em=int(normalize(answer)==normalize(g['answer'])),canonical_f1=f1(answer,g['answer']),
                    attributed_alias_em=max(int(normalize(answer)==normalize(x)) for x in alternatives),
                    attributed_alias_f1=max(f1(answer,x) for x in alternatives))
            else:metrics=dict(canonical_em=0,canonical_f1=0.,attributed_alias_em=0,attributed_alias_f1=0.)
            assert all(abs(metrics[k]-s['score'][k])<=1e-12 for k in metrics)
            result[key]=metrics;statuses[status]+=1;caps+=v['stop_type']=='length_cap';errors+=v['status']!='ok'
            if kind=='native':
                field=re.search(r'(?im)^[ \t]*Final answer:[ \t]*',v['raw_text'])
                prefix=v['raw_text'][:field.start()] if field else v['raw_text'];prefixids+=bool(re.search(r'\[\d+\]',prefix))
            else:
                first=v['raw_text'].strip().split('\n',1)[0]
                eh+=bool(re.search(r'(?im)^\s*Evidence:',v['raw_text']));fh+=bool(re.search(r'(?im)^\s*Final answer:',v['raw_text']))
                brackets+=bool(re.fullmatch(r'(?:\[\d+\]\s*)+',first))
        table=report['tables'][name]
        assert table['canonical_correct']==sum(v['canonical_em'] for v in result.values())
        assert abs(table['mean_canonical_F1']-sum(v['canonical_f1'] for v in result.values())/255)<=1e-12
        assert table['alias_correct']==sum(v['attributed_alias_em'] for v in result.values())
        assert table['parsed_status']==dict(statuses) and table['unsupported']==sum(n for k,n in statuses.items() if k!='complete')
        assert table['execution_errors']==errors and table['capped']==caps
        assert table['generated_actions']==sum(v['n_actions'] for v in values)
        assert table['generated_prefix_ID_rows']==prefixids and table['repeated_evidence_headers']==eh
        assert table['repeated_final_headers']==fh and table['leading_bracket_ID_lines']==brackets
        recomputed[name]=result
for index,contrast in enumerate(report['contrasts']):
    ids=sorted(gold);a,b=contrast['treatment'],contrast['control']
    def value(name,k):return recomputed[name][k]['canonical_em'] if name in recomputed else scored[name][k]['score']['canonical_em']
    delta=np.array([value(a,k)-value(b,k) for k in ids]);rng=np.random.default_rng(2026101925+100+index)
    bootstrap=delta[rng.integers(0,255,(20000,255))].mean(1)
    assert abs(float(delta.mean())-contrast['paired_gain'])<=1e-12
    assert max(abs(x-y) for x,y in zip(np.quantile(bootstrap,[.025,.975]),contrast['CI95']))<=1e-12
    assert contrast['corrected']==int((delta>0).sum()) and contrast['regressed']==int((delta<0).sum())
result=dict(status='PASS_INDEPENDENT_NATIVE_TOKEN_ANSWER_COST_COUNT_AND_PAIRED_HELD_READOUT',
    attempts=attempts_total,native_and_cache_checks=native_checks,contrast_count=len(report['contrasts']),
    checker_sha256=sha(__file__),result_sha256=sha(C/'held-native-and-replay-result.json'),
    scored_sha256=sha(C/'held-native-and-replay-scored.json'),
    limitation='Independent recomputation of measurements; not semantic adjudication or untouched confirmation.')
with (C/'held-independent-check.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
