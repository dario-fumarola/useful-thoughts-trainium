"""Independent raw-action, node, numerical-witness and paired-gate check.

Run only after all23 original executions and the main readout close. No model
calls, error exclusions, parser rescue, or changes to the frozen scientific gate.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,re
import numpy as np
import torch
from tokenizers import Tokenizer
from train.useful_note_precision import compare_distributions

read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def complete_score(value,mode,gold):
    if value['status']!='ok' or value['stop_type'] in ('length_cap','error'):return 0
    text=value['text'];fields=list(re.finditer(r'(?im)^Final answer:[ \t]*(?:\r?\n)?',text))
    if mode=='trace':
        if len(fields)!=1:return 0
        text=text[fields[0].end():]
    elif fields:
        if len(fields)!=1 or text[:fields[0].start()].strip():return 0
        text=text[fields[0].end():]
    lines=text.lstrip(' \t\r\n').splitlines(keepends=True)
    if not lines or not(lines[0].endswith(('\n','\r')) or value['stop_type'].startswith('token:')):return 0
    symbol=lines[0].strip().upper()
    return int(len(symbol)==1 and symbol in 'ABCDEFGHIJKLMNOP' and symbol==gold and not any(v.strip() for v in lines[1:]))


def bootstrap(a,b,expected):
    assert a.keys()==b.keys();groups={}
    for key in sorted(a):
        assert a[key][0]==b[key][0]
        groups.setdefault(a[key][0],[]).append(b[key][1]-a[key][1])
    assert set(map(len,groups.values()))=={2}
    delta=np.array([sum(v)/2 for v in groups.values()],dtype=np.float64)
    rng=np.random.default_rng(17);draws=[]
    for _ in range(50):
        indices=rng.integers(0,len(delta),size=(100,len(delta)))
        draws.extend(delta[indices].mean(1).tolist())
    interval=np.quantile(draws,[.025,.975])
    assert expected['documents']==len(delta) and expected['items']==2*len(delta)
    assert expected['bootstrap_draws']==5000 and expected['bootstrap_seed']==17
    assert abs(float(delta.mean())-expected['difference_b_minus_a'])<1e-10
    assert np.allclose(interval,expected['ci95'],atol=1e-10,rtol=0)
    return bool(delta.mean()>=.03 and interval[0]>0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','readout','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();paths=read(args.index);result=read(args.readout)
    assert result['status']=='COMPLETE_FIXED_V28_ALL_TERMINAL_READOUT' and len(paths)==len(set(paths))==23
    scores={};attempt_count=0;native_count=0;cache_count=0
    for path in paths:
        root=Path(path);p=read(root/'protocol.json');key=(p['model']['id'],p['panel'],p['diagnostic'])
        assert key not in scores;scores[key]={mode:{} for mode in p['modes']}
        tok=Tokenizer.from_file(str(Path(p['model_dir'])/'tokenizer.json'))
        gold={v['item_id']:v for v in rows(root/'gold.jsonl')};reference=torch.load(root/'native-reference.pt',weights_only=True,map_location='cpu')
        assert len(gold)==p['items'] and sha(root/'gold.jsonl')==p['gold_file_sha256']
        for shard in range(4):
            folder=root/'answers'/f'shard-{shard}';run=read(folder/'run.json')
            assert run['status']=='complete_all_attempts' and run['attempts_sha256']==sha(folder/'attempts.jsonl')
            assert run['protocol_sha256']==sha(root/'protocol.json')
            actual=torch.load(folder/'native-logps.pt',weights_only=True,map_location='cpu')
            check=compare_distributions(reference[shard:shard+1],actual)[0]
            assert check['passed'] and read(folder/'native.json')['passed'];native_count+=1
            vectors=torch.load(folder/'cache-prefill-logps.pt',weights_only=True,map_location='cpu')
            check=compare_distributions(vectors['ordinary'],vectors['cached'])[0]
            assert check['passed'] and check['max_logp_error']<=.001;cache_count+=1
            values=rows(folder/'attempts.jsonl')
            assert len(values)==len(p['ordered_item_ids'][shard::4])*len(p['modes'])
            for v in values:
                item=v['item_id'];mode=v['mode'];assert item not in scores[key][mode]
                assert v['n_actions']==len(v['actions'])<=128
                assert tok.decode(v['content_ids'],skip_special_tokens=False)==v['raw_text']==v['text']
                if v['status']=='ok':
                    terminal=v['stop_type'].startswith('token:')
                    assert v['content_ids']==(v['actions'][:-1] if terminal else v['actions'])
                    if terminal:assert v['stop_type']==f"token:{v['actions'][-1]}" and v['actions'][-1] in (151643,151645)
                    if v['stop_type']=='length_cap':assert v['n_actions']==128
                g=gold[item];q=g['question'];node=q['start']
                for op in q['program']:node=q[op][node]
                target='ABCDEFGHIJKLMNOP'[node];assert target==g['answer']
                scores[key][mode][item]=(g['family_id'],complete_score(v,mode,target));attempt_count+=1
        for mode,values in scores[key].items():
            assert set(values)==set(gold) and set(Counter(v[0] for v in values.values()).values())=={2}
            table=next(v for v in result['absolute'] if (v['model'],v['panel'],v['diagnostic'],v['mode'])==(*key,mode))
            assert table['correct']==sum(v[1] for v in values.values()) and table['items']==len(values)
    decisions={'inference':True,'training':True}
    for seed in (2026102028,2026102029):
        d=f'program-direct-seed{seed}-v1';t=f'program-trace-seed{seed}-v1';p=f'program-pause-seed{seed}-v1'
        get=lambda model,mode:scores[model,'DEV_composition',False][mode]
        comparisons={'inference':{'trace_minus_direct_model':((d,'direct'),(t,'trace')),
            'trace_minus_trained_pause':((p,'pause'),(t,'trace')),'trace_minus_same_direct':((t,'direct'),(t,'trace'))},
            'training':{'trained_direct_minus_direct_model':((d,'direct'),(t,'direct')),
            'trained_direct_minus_pause_direct':((p,'direct'),(t,'direct'))}}
        for claim,conditions in comparisons.items():
            for name,(a,b) in conditions.items():
                passed=bootstrap(get(*a),get(*b),result['primary_'+claim][str(seed)][name])
                decisions[claim]&=passed
    assert decisions==result['gates']
    report=dict(status='PASS_INDEPENDENT_NATIVE_ACTIONS_NODES_WITNESSES_FAMILY_BOOTSTRAPS_AND_GATES',
        utc=datetime.now(timezone.utc).isoformat(),model_calls=0,attempts=attempt_count,
        native_vectors=native_count,cache_vectors=cache_count,independent_gates=decisions,
        readout_sha256=sha(args.readout),index_sha256=sha(args.index),checker_sha256=sha(__file__),
        limits=['Independent arithmetic/count/parser/token/vector/bootstrap reconstruction,not semantic or broad capability validation.',
                'No fresh model calls or rescoring rule changes. All original failures remain in denominators.'])
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ('status','attempts','native_vectors','cache_vectors')}))


if __name__=='__main__':main()
