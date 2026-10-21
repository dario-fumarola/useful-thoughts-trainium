"""Reconstruct all V30 contexts/statistics and fixed ordinary-HF score checks.

Independent context construction and bootstrap implementation; canonical trace
classification uses the frozen, tested arithmetic contract. This is verification
within the repository, not an external replication or semantic judge.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train.candidate_utility_v30 import correctness
from train.calculation_boundaries import render_value
from train.evaluate import load_own_checkpoint
from weekend_thinking.artifacts import atomic_json, sha256_file
from weekend_thinking.pretrained import load_base


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).open() if s.strip()]


def independent_stat(values):
    ordered=np.array([v for _,v in sorted(values)],dtype=np.float64)
    rng=np.random.default_rng(2026102130)
    means=[ordered[rng.integers(0,len(ordered),len(ordered))].mean() for _ in range(5000)]
    return float(ordered.mean()),np.quantile(means,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','runs','readout','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); protocol=read(a.protocol/'protocol.json'); report=read(a.readout/'readout.json')
    assert report['protocol_sha256']==sha256_file(a.protocol/'protocol.json')
    for name,digest in protocol['files'].items():assert sha256_file(a.protocol/name)==digest
    for name,digest in report['source_pins'].items():assert sha256_file(name)==digest
    labels=rows(a.protocol/'diagnostic-labels.jsonl'); questions=rows(a.protocol/'diagnostic-questions.jsonl')
    assert len(labels)==len(questions)==256
    pool=rows(protocol['pool_file']); ordered=sorted(pool,key=lambda r:hashlib.sha256(
        ('v30-fixed-cohort:'+r['item_id']).encode()).hexdigest())
    assert labels==ordered[8:264]
    assert questions==[{k:r[k] for k in ('item_id','document_id','prefix_ids')} for r in labels]
    n={};s={}
    for shard in range(4):
        for r in rows(a.runs/f'generate-{shard}/notes.jsonl'):
            key=(r['item_id'],r['mode']);assert key not in n;n[key]=r
        for tag in ('score-T','score-B0'):
            for r in rows(a.runs/f'{tag}-{shard}/scores.jsonl'):
                key=(r['item_id'],r['mode']);assert key not in s;s[key]=r
    assert len(n)==2304 and len(s)==3840
    torch.set_num_threads(16)
    model,tok=load_base('/home/ubuntu/dario/local/assets/qwen3-1.7b-base'); model.eval()
    opening=tok.encode('\nCalculation:\n',add_special_tokens=False)
    closing=tok.encode('Answer:\n',add_special_tokens=False);stops=(closing[0],tok.eos_token_id)
    contrasts={k:[] for k in report['contrasts']};mixed=[];helpful=0;contexts=0
    for index,row in enumerate(labels):
        item=row['item_id'];past=row['prefix_ids'];target=row['continuation16_ids']
        for mode in ['greedy']+[f'sample{i}' for i in range(8)]:
            note=n[(item,mode)];actions=note['actions'];stopped=actions[-1] in stops
            assert not any(t in stops for t in actions[:-1])
            assert note['content_ids']==(actions[:-1] if stopped else actions)
            assert tok.decode(note['content_ids'],skip_special_tokens=False,
                clean_up_tokenization_spaces=False)==note['text']
            assert note['prompt_sha256']==hashlib.sha256(json.dumps(past+opening,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        oracle=row['full_calculation_ids'];lines=row['full_calculation_text'].splitlines(keepends=True)
        left,right=lines[-1].rstrip('\n').rsplit(' = ',1)
        lines[-1]=left+' = '+render_value(Fraction(right)+1)+'\n'
        corrupt=tok.encode(''.join(lines),add_special_tokens=False)
        donor=labels[(index+1)%256]['item_id']
        contents=dict(raw=None,B0_raw=None,empty=[],oracle=oracle,corrupt=corrupt,
            other=n[(donor,'greedy')]['content_ids'])
        contents.update({mode:n[(item,mode)]['content_ids'] for mode in ['greedy']+[f'sample{i}' for i in range(8)]})
        for mode,note in contents.items():
            score=s[(item,mode)];expected=past if note is None else past+opening+note+closing
            assert score['context_ids']==expected and score['target_ids']==target
            assert score['document_id']==row['document_id'] and score['tokens']==16
            assert abs(score['full_logp']-sum(score['token_logps']))<1e-10
            contexts+=1
        value=lambda mode:s[(item,mode)]['full_logp']
        sampled=[f'sample{i}' for i in range(8)];best=max(sampled,key=value)
        d=dict(best8_minus_raw=value(best)-value('raw'),best8_minus_B0_raw=value(best)-value('B0_raw'),
            best8_minus_greedy=value(best)-value('greedy'),mean8_minus_raw=sum(value(m) for m in sampled)/8-value('raw'),
            greedy_minus_raw=value('greedy')-value('raw'),oracle_minus_corrupt=value('oracle')-value('corrupt'),
            oracle_minus_raw=value('oracle')-value('raw'))
        for name,v in d.items():contrasts[name].append((item,v))
        category={m:correctness(n[(item,m)]['text'],row['full_calculation_text'],row['numeric_value'])['category'] for m in sampled}
        good=[m for m in sampled if category[m]=='verified_canonical'];bad=[m for m in sampled if category[m]=='arithmetic_error']
        helpful+=any(value(m)>max(value('raw'),value('B0_raw')) for m in good)
        if good and bad:mixed.append((item,sum(value(m) for m in good)/len(good)-sum(value(m) for m in bad)/len(bad)))
    checks={}
    for name,values in contrasts.items():
        mean,interval=independent_stat(values);expected=report['contrasts'][name]
        assert abs(mean-expected['difference_b_minus_a'])<1e-10
        assert max(abs(x-y) for x,y in zip(interval,expected['ci95']))<1e-10
        checks[name]=dict(mean=mean,ci95=interval)
    assert helpful==report['correct_helpful_documents'] and len(mixed)==report['alignment']['mixed_documents']
    if len(mixed)>=2:
        mean,interval=independent_stat(mixed);expected=report['alignment']['conditional_contrast']
        assert abs(mean-expected['difference_b_minus_a'])<1e-10
        assert max(abs(x-y) for x,y in zip(interval,expected['ci95']))<1e-10
    expected_gates=dict(complete_execution=True,
        opportunity=all(checks[k]['mean']>=.25 and checks[k]['ci95'][0]>0 for k in
            ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy')),
        correct_helpful_coverage=helpful>=64,
        alignment=len(mixed)>=32 and mean>=.25 and interval[0]>0,
        plumbing=checks['oracle_minus_corrupt']['mean']>=0 and checks['oracle_minus_corrupt']['ci95'][0]>0)
    assert expected_gates==report['gates'] and all(expected_gates.values())==report['all_required_passed']
    # Fixed before outcomes: first four cohort documents, sample0/sample7/raw/empty.
    loaded=load_own_checkpoint(model,Path(protocol['model']['checkpoint']));model.eval()
    assert loaded['sha256']==protocol['model']['checkpoint_sha256']
    hf=[]
    with torch.no_grad():
        for row,mode in zip(labels[:4],('sample0','sample7','raw','empty')):
            stored=s[(row['item_id'],mode)];context=stored['context_ids'];target=stored['target_ids']
            ids=torch.tensor([context+target]);lp=model(ids).logits[0,len(context)-1:-1].float().log_softmax(-1)
            actual=lp.gather(-1,ids[0,len(context):,None])[:,0].tolist()
            error=max(abs(x-y) for x,y in zip(actual,stored['token_logps']));assert error<=.001
            hf.append(dict(item_id=row['item_id'],mode=mode,max_token_error=error))
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'independent-check.json',dict(status='PASS_ALL_CONTEXTS_AND_PRIMARY_STATISTICS_FIXED_HF',
        contexts=contexts,notes=len(n),contrasts=checks,conditional_mixed_documents=len(mixed),
        correct_helpful_documents=helpful,gates=expected_gates,HF_checks=hf,report_sha256=sha256_file(a.readout/'readout.json'),
        checker_sha256=sha256_file(__file__),limits=__doc__))
    print('PASS_CONTEXTS',contexts,'NOTES',len(n),'HF',len(hf),flush=True)


if __name__=='__main__':main()
