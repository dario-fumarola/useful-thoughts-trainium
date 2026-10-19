"""Fixed developmental contrasts: content-supervised model versus format1 control."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import json,re
import numpy as np
from train.document_qa_answers import parse_answer,score
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
O=R/'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-held255-v1'
OLD=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
review=read(C/'evaluation-release.json')
assert review['status']=='RELEASED_FORMAT_WEIGHT_V26_TERMINAL_FIXED_EVALUATION'
assert read(C/'HELD/GEN/closure.json')['actual_attempts']==510
manifest=read(D/'manifest.json');gold={v['item_id']:v for v in rows(D/'held255-gold-only.jsonl')}
assert len(gold)==255 and sha256_file(D/'held255-gold-only.jsonl')==manifest['files']['held_gold']['sha256']
old=read(OLD/'held-native-and-replay-scored.json')
assert sha256_file(OLD/'held-native-and-replay-scored.json')==review['comparison_scored_sha256']
assert read(OLD/'held-independent-check.json')['status']=='PASS_INDEPENDENT_NATIVE_TOKEN_ANSWER_COST_COUNT_AND_PAIRED_HELD_READOUT'
scored={name:old[key] for name,key in [('selector1:direct','native:direct'),('selector1:own','native:selection'),('selector1:empty','answer-replay:empty'),('format0.1:direct','old-matched-direct')]}
# V23 stored its parsed answer inside the score record; retain that source record.
scored['format0.1:direct']={k:dict(v,answer=v['score']['parsed']) for k,v in scored['format0.1:direct'].items()}
p=read(O/'protocol.json');attempts=[]
for shard in range(4):
    d=O/'answers'/f'shard-{shard}';run=read(d/'run.json')
    assert run['status']=='complete_all_attempts' and run['attempts_sha256']==sha256_file(d/'attempts.jsonl')
    assert read(d/'native.json')['passed'] and read(d/'cache-prefill.json')['max_logp_error']<=.001
    attempts.extend(rows(d/'attempts.jsonl'))
assert len(attempts)==510
for mode in ('direct','canonical_empty'):
    result={};values=[v for v in attempts if v['mode']==mode]
    assert len(values)==len({v['item_id'] for v in values})==255 and {v['item_id'] for v in values}==set(gold)
    for v in values:
        g=gold[v['item_id']]
        answer=parse_answer(v['raw_text'],v['parser_mode'],terminated=v['stop_type'].startswith('token:'),capped=v['stop_type']=='length_cap')
        metrics=score(answer,g['answer'],g['answer_aliases'])
        if v['status']!='ok':metrics={k:0. for k in metrics}
        field=re.search(r'(?im)^[ \t]*Final answer:[ \t]*',v['raw_text'])
        prefix=v['raw_text'][:field.start()] if field else v['raw_text']
        first=v['raw_text'].strip().split('\n',1)[0]
        result[v['item_id']]=dict(native=v,answer=asdict(answer),score=metrics,
            generated_prefix_ID_count=len(re.findall(r'\[\d+\]',prefix)) if mode=='direct' else 0,
            generated_evidence_header=bool(re.search(r'(?im)^\s*Evidence:',v['raw_text'])),
            generated_final_header=bool(re.search(r'(?im)^\s*Final answer:',v['raw_text'])) if mode=='canonical_empty' else False,
            leading_bracket_ID_line=bool(re.fullmatch(r'(?:\[\d+\]\s*)+',first)))
    scored['format1:'+mode]=result
tables={}
for name,result in scored.items():
    assert set(result)==set(gold)
    tables[name]=dict(items=255,canonical_correct=sum(v['score']['canonical_em'] for v in result.values()),
        mean_canonical_F1=sum(v['score']['canonical_f1'] for v in result.values())/255,
        alias_correct=sum(v['score']['attributed_alias_em'] for v in result.values()),
        parsed_status=dict(Counter(v['answer']['status'] for v in result.values())),
        unsupported=sum(v['answer']['status']!='complete' for v in result.values()),execution_errors=sum(v['native']['status']!='ok' for v in result.values()),
        capped=sum(v['native']['stop_type']=='length_cap' for v in result.values()),generated_actions=sum(v['native']['n_actions'] for v in result.values()),
        generated_prefix_ID_rows=sum(v.get('generated_prefix_ID_count',0)>0 for v in result.values()),
        generated_evidence_headers=sum(v.get('generated_evidence_header',False) for v in result.values()),
        regenerated_final_headers=sum(v.get('generated_final_header',False) for v in result.values()),
        leading_bracket_ID_lines=sum(v.get('leading_bracket_ID_line',False) for v in result.values()))
comparisons=[('selector1:direct','format1:direct'),('selector1:own','format1:direct'),('selector1:empty','format1:canonical_empty'),('format1:direct','format0.1:direct')]
contrasts=[];ids=sorted(gold)
for index,(a,b) in enumerate(comparisons):
    delta=np.array([scored[a][k]['score']['canonical_em']-scored[b][k]['score']['canonical_em'] for k in ids]);seed=2026101926+index
    rng=np.random.default_rng(seed);boot=delta[rng.integers(0,255,(20000,255))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
    contrasts.append(dict(treatment=a,control=b,paired_gain=float(delta.mean()),CI95=[float(lo),float(hi)],corrected=int((delta>0).sum()),regressed=int((delta<0).sum()),seed=seed))
primary=contrasts[0]
summary=dict(status='COMPLETE_DEVELOPMENTAL_FORMAT_WEIGHT_V26_ANSWERS',utc=utc_now(),tables=tables,contrasts=contrasts,
    primary_developmental_criterion_pass=primary['paired_gain']>=.03 and primary['CI95'][0]>0.,
    operational_controls=dict(native_direct_generates_no_IDs=tables['format1:direct']['generated_prefix_ID_rows']==0,
        canonical_empty_stays_in_answer_field=all(tables['format1:canonical_empty'][k]==0 for k in ('generated_evidence_headers','regenerated_final_headers','leading_bracket_ID_lines'))),
    protocol_sha256=sha256_file(O/'protocol.json'),comparison_scored_sha256=sha256_file(OLD/'held-native-and-replay-scored.json'),source_sha256=sha256_file(__file__),
    limitations=['One order seed and previously inspected held255, not fresh confirmation. Intervals conditional on checkpoints and unadjusted.',
        'Matched coefficients do not match gradient magnitudes or label information. All20 passages remain visible.',
        'Primary concerns direct answering after training. It cannot establish an incremental benefit from emitting IDs.',
        'All earlier reports and failed gates remain unchanged. Failures retained in denominators.'])
atomic_json(C/'held-result.json',summary)
with (C/'held-scored.json').open('x') as f:json.dump(scored,f)
print(json.dumps(summary,indent=2))
