"""Post-hoc whole-cohort transition/stopping diagnosis; no new model calls."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
from tokenizers import Tokenizer
from train.permutation_program_answers import parse_answer
from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/permutation-program-v28-complete-readout-io-v2'
INDEX=R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1/index.json'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
LINE=re.compile(r'([rs]): ([A-P]) -> ([A-P])')
NODES='ABCDEFGHIJKLMNOP'


def transitions(q):
    node=q['start'];result=[]
    for op in q['program']:
        nxt=q[op][node];result.append((op,NODES[node],NODES[nxt]));node=nxt
    return result


def main():
    assert read(C/'independent-check.json')['status']=='PASS_INDEPENDENT_NATIVE_ACTIONS_NODES_WITNESSES_FAMILY_BOOTSTRAPS_AND_GATES'
    summary={};partials={};examples={};fixed_ids=None
    for folder in read(INDEX):
        root=Path(folder);p=read(root/'protocol.json');model=p['model']['id']
        if '-trace-' not in model:continue
        gold={v['item_id']:v for v in rows(root/'gold.jsonl')}
        if p['panel']=='DEV_depth' and not p['diagnostic'] and fixed_ids is None:
            fixed_ids=[]
            for depth in (7,8,9,10):
                eligible={v['family_id'] for v in gold.values() if len(v['question']['program'])==depth}
                family=min(eligible,key=lambda key:hashlib.sha256(key.encode()).hexdigest())
                fixed_ids.extend(sorted(key for key,g in gold.items() if g['family_id']==family))
            assert len(fixed_ids)==8
        if p['diagnostic']:
            tok=Tokenizer.from_file(str(Path(p['model_dir'])/'tokenizer.json'))
            supplied={}
            for mode in p['modes']:
                supplied[mode]={}
                for v in rows(root/f'prompts-{mode}.jsonl'):
                    text=tok.decode(v['prompt_ids'],skip_special_tokens=False)
                    body=text.rsplit('Work:\n',1)[1].split('Final answer:\n',1)[0]
                    claims=[LINE.fullmatch(line) for line in body.splitlines()]
                    assert claims and all(claims)
                    supplied[mode][v['item_id']]=claims[-1].group(3)
            counts=defaultdict(Counter)
        else:
            counts=defaultdict(Counter)
        for shard in range(4):
            folder=root/'answers'/f'shard-{shard}'
            assert read(folder/'run.json')['attempts_sha256']==sha256_file(folder/'attempts.jsonl')
            for value in rows(folder/'attempts.jsonl'):
                g=gold[value['item_id']];q=g['question'];expected=transitions(q)
                assert expected[-1][2]==g['answer']
                mode=value['mode']
                if p['diagnostic']:
                    parsed=parse_answer(value['text'],mode,terminated=value['stop_type'].startswith('token:'),capped=value['stop_type']=='length_cap')
                    count=counts[mode];count['items']+=1;count['unsupported']+=int(parsed.status!='complete')
                    count['correct_answers']+=int(parsed.status=='complete' and parsed.answer==g['answer'])
                    last=supplied[mode][value['item_id']]
                    count['copies_last_supplied_destination']+=int(parsed.status=='complete' and parsed.answer==last)
                    count['last_supplied_destination_is_gold']+=int(last==g['answer'])
                    continue
                if mode!='trace':continue
                body=value['text'].split('Final answer:',1)[0]
                claims=[LINE.fullmatch(line) for line in body.splitlines()]
                valid=all(claims)
                actual=[m.groups() if m is not None else None for m in claims]
                length=len(actual);depth=len(expected);prefix=0
                for a,b in zip(actual,expected):
                    if a!=b:break
                    prefix+=1
                parsed=parse_answer(value['text'],'trace',terminated=value['stop_type'].startswith('token:'),capped=value['stop_type']=='length_cap')
                count=counts[str(depth)];count['items']+=1
                count['correct_answers']+=int(parsed.status=='complete' and parsed.answer==g['answer'])
                count['complete_correct_execution']+=int(valid and actual==expected)
                count[f'claimed_steps_{length}']+=1
                count['shorter_than_required']+=int(length<depth)
                count['longer_than_required']+=int(length>depth)
                count['correct_prefix_then_stop_early']+=int(valid and length<depth and prefix==length)
                count['all_claims_correct_execution_prefix']+=int(valid and length<=depth and prefix==length)
                count['syntactically_valid']+=int(valid)
                count['correct_prefix_steps_total']+=prefix
                count['copies_last_claim_destination']+=int(bool(actual) and actual[-1] is not None and parsed.status=='complete' and parsed.answer==actual[-1][2])
                if p['panel']=='DEV_depth' and value['item_id'] in fixed_ids:
                    item=examples.setdefault(value['item_id'],dict(item_id=value['item_id'],family_id=g['family_id'],question=q,
                        expected_answer=g['answer'],expected_transitions=expected,models={}))
                    item['models'][model]=dict(original_attempt=value,claimed_steps=length,correct_execution_prefix_steps=prefix)
        if p['diagnostic']:partials[model]={k:dict(v) for k,v in counts.items()}
        else:summary[f"{model}/{p['panel']}"]={k:dict(v) for k,v in counts.items()}
    assert len(examples)==8 and all(len(v['models'])==2 for v in examples.values())
    report=dict(status='COMPLETE_POSTHOC_ALL_STORED_TRACE_AND_SUPPLIED_STATE_DIAGNOSIS',model_calls=0,
        readout_sha256=sha256_file(C/'readout.json'),source_sha256=sha256_file(__file__),
        whole_trace_cohorts=summary,supplied_state_controls=partials,
        longer_examples=dict(selection='Post-hoc,one lowest-SHA256 family per depth7/8/9/10,both variants,all two trace models. Original prospective composition examples remain unchanged.',cases=[examples[k] for k in fixed_ids]),
        limits=['Exploratory diagnosis of existing outputs,not new primary endpoints or fresh confirmation.',
            'Supplied partial notes omit the root operation; that answer-prefilled interface was not a trained partial-completion objective.',
            'Claims about faithful internal computation require more than visible transition agreement. No scientific scores or gates change.'])
    atomic_json(C/'stored-trace-diagnosis.json',report)
    print(json.dumps(dict(status=report['status'],longer={k:v for k,v in summary.items() if k.endswith('DEV_depth')},supplied_state_controls=partials)))


if __name__=='__main__':main()
