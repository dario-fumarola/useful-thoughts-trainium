"""Independent token/parser/count/bootstrap reconstruction of the V26 contrast."""
from collections import Counter
from pathlib import Path
import hashlib,json,re,string
import numpy as np
from transformers import AutoTokenizer

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
O=R/'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-held255-v1'
OLD=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def normalize(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def parse(text,mode,terminated,capped):
    start=0;terminal=terminated or capped
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

report=read(C/'held-result.json');scored=read(C/'held-scored.json');review=read(C/'evaluation-release.json')
assert report['status']=='COMPLETE_DEVELOPMENTAL_FORMAT_WEIGHT_V26_ANSWERS'
gold={v['item_id']:v for v in rows(D/'held255-gold-only.jsonl')};assert len(gold)==255
assert sha(D/'held255-gold-only.jsonl')==read(D/'manifest.json')['files']['held_gold']['sha256']
assert sha(OLD/'held-native-and-replay-scored.json')==review['comparison_scored_sha256']==report['comparison_scored_sha256']
previous=read(OLD/'held-native-and-replay-scored.json')
for name,key in [('selector1:direct','native:direct'),('selector1:own','native:selection'),('selector1:empty','answer-replay:empty'),('format0.1:direct','old-matched-direct')]:
    expected=previous[key]
    if name=='format0.1:direct':expected={k:dict(v,answer=v['score']['parsed']) for k,v in expected.items()}
    assert scored[name]==expected
p=read(O/'protocol.json');assert sha(O/'protocol.json')==report['protocol_sha256']
assert read(C/'HELD/GEN/closure.json')['actual_attempts']==510
tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
for name,digest in p['files'].items():assert sha(O/name)==digest
current={}
for shard in range(4):
    d=O/'answers'/f'shard-{shard}';run=read(d/'run.json');values=rows(d/'attempts.jsonl')
    assert run['status']=='complete_all_attempts' and sha(d/'attempts.jsonl')==run['attempts_sha256'] and len(values)==run['rows']
    assert read(d/'native.json')['passed'] and read(d/'cache-prefill.json')['max_logp_error']<=.001
    assert {(v['mode'],v['item_id']) for v in values}=={(m,k) for m in p['modes'] for k in p['ordered_item_ids'][shard::4]}
    for v in values:
        key=('format1:'+v['mode'],v['item_id']);assert key not in current;current[key]=v
assert len(current)==510
recomputed={}
for name,records in scored.items():
    assert set(records)==set(gold)
    results={};statuses=Counter();caps=errors=actions=prefixids=eheaders=fheaders=brackets=0
    for key,s in records.items():
        v=s['native'];g=gold[key]
        if name.startswith('format1:'):assert v==current[(name,key)]
        assert v['n_actions']==len(v['actions'])<=128
        assert v['content_ids']==(v['actions'][:-1] if v['stop_type'].startswith('token:') else v['actions'])
        assert tok.decode(v['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==v['raw_text']
        mode='direct' if name in ('selector1:empty','format1:canonical_empty') else 'prior';assert v['parser_mode']==mode
        status,answer=parse(v['raw_text'],mode,v['stop_type'].startswith('token:'),v['stop_type']=='length_cap')
        assert status==s['answer']['status'] and answer==s['answer']['answer']
        if status=='complete' and v['status']=='ok':
            alternatives=[g['answer'],*g['answer_aliases']]
            metrics=dict(canonical_em=int(normalize(answer)==normalize(g['answer'])),canonical_f1=f1(answer,g['answer']),attributed_alias_em=max(int(normalize(answer)==normalize(a)) for a in alternatives),attributed_alias_f1=max(f1(answer,a) for a in alternatives))
        else:metrics=dict(canonical_em=0,canonical_f1=0.,attributed_alias_em=0,attributed_alias_f1=0.)
        assert all(abs(metrics[k]-s['score'][k])<=1e-12 for k in metrics)
        results[key]=metrics;statuses[status]+=1;caps+=v['stop_type']=='length_cap';errors+=v['status']!='ok';actions+=v['n_actions']
        if name.startswith('format1:'):
            text=v['raw_text'];field=re.search(r'(?im)^[ \t]*Final answer:[ \t]*',text);prefix=text[:field.start()] if field else text
            prefixids+=name=='format1:direct' and bool(re.search(r'\[\d+\]',prefix))
            eheaders+=bool(re.search(r'(?im)^\s*Evidence:',text))
            fheaders+=name=='format1:canonical_empty' and bool(re.search(r'(?im)^\s*Final answer:',text))
            brackets+=bool(re.fullmatch(r'(?:\[\d+\]\s*)+',text.strip().split('\n',1)[0]))
    table=report['tables'][name]
    assert table['canonical_correct']==sum(v['canonical_em'] for v in results.values()) and table['alias_correct']==sum(v['attributed_alias_em'] for v in results.values())
    assert abs(table['mean_canonical_F1']-sum(v['canonical_f1'] for v in results.values())/255)<=1e-12
    assert table['parsed_status']==dict(statuses) and table['unsupported']==sum(n for k,n in statuses.items() if k!='complete')
    assert (table['capped'],table['execution_errors'],table['generated_actions'])==(caps,errors,actions)
    if name.startswith('format1:'):assert (table['generated_prefix_ID_rows'],table['generated_evidence_headers'],table['regenerated_final_headers'],table['leading_bracket_ID_lines'])==(prefixids,eheaders,fheaders,brackets)
    recomputed[name]=results
expected=[('selector1:direct','format1:direct'),('selector1:own','format1:direct'),('selector1:empty','format1:canonical_empty'),('format1:direct','format0.1:direct')]
assert [(v['treatment'],v['control']) for v in report['contrasts']]==expected
for index,c in enumerate(report['contrasts']):
    a,b=expected[index];ids=sorted(gold);delta=np.array([recomputed[a][k]['canonical_em']-recomputed[b][k]['canonical_em'] for k in ids]);seed=2026101926+index
    rng=np.random.default_rng(seed);boot=delta[rng.integers(0,255,(20000,255))].mean(1)
    assert c['seed']==seed and abs(c['paired_gain']-float(delta.mean()))<=1e-12
    assert max(abs(x-y) for x,y in zip(c['CI95'],np.quantile(boot,[.025,.975])))<=1e-12
    assert (c['corrected'],c['regressed'])==(int((delta>0).sum()),int((delta<0).sum()))
primary=report['contrasts'][0]
assert report['primary_developmental_criterion_pass']==(primary['paired_gain']>=.03 and primary['CI95'][0]>0.)
assert report['operational_controls']==dict(native_direct_generates_no_IDs=report['tables']['format1:direct']['generated_prefix_ID_rows']==0,
    canonical_empty_stays_in_answer_field=all(report['tables']['format1:canonical_empty'][k]==0 for k in ('generated_evidence_headers','regenerated_final_headers','leading_bracket_ID_lines')))
result=dict(status='PASS_INDEPENDENT_V26_TOKENS_ANSWERS_CONTROLS_COUNTS_AND_PAIRED_CONTRASTS',new_attempts=510,all_new_and_reused_scored_rows=1530,native_and_cache_checks=4,contrasts=4,result_sha256=sha(C/'held-result.json'),scored_sha256=sha(C/'held-scored.json'),source_sha256=sha(__file__),limit='Measurement consistency, not semantic adjudication or fresh confirmation.')
with (C/'held-independent-check.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
