"""Replay the public V30 verdict with NumPy and frozen source; no Torch/models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).open()]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired(values,expected):
    if len(values)<2:
        assert expected['status']=='unidentifiable';return
    x=np.array([v for _,v in sorted(values)],dtype=np.float64)
    rng=np.random.default_rng(2026102130)
    means=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(5000)]
    assert abs(float(x.mean())-expected['difference_b_minus_a'])<1e-10
    assert max(abs(a-b) for a,b in zip(np.quantile(means,[.025,.975]),expected['ci95']))<1e-10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=read(a.manifest);root=a.evidence
    assert manifest['status']=='PASS_V30_ARCHIVE_FULL_MEMBER_READBACK_AND_FSYNC'
    for name,sha in manifest['files'].items():assert digest(root/name)==sha
    sys.path.insert(0,str(root/'host-sources'))
    from train.candidate_utility_v30 import correctness
    report=read(root/'results/readout.json');assert digest(root/'results/readout.json')==manifest['report_sha256']
    sources=rows(root/'cohort-math.jsonl');notes={};scores={}
    for shard in range(4):
        for r in rows(root/f'outputs/generate-{shard}.jsonl'):notes[(r['item_id'],r['mode'])]=r
        for tag in ('score-T','score-B0'):
            for r in rows(root/f'outputs/{tag}-{shard}.jsonl'):scores[(r['item_id'],r['mode'])]=r
    assert len(notes)==2304 and len(scores)==3840 and len(sources)==256
    deltas={k:[] for k in report['contrasts']};components={k:[] for k in report['best8_native_components_vs_raw']}
    mixed=[];helpful=0;classes=Counter()
    for r in sources:
        item=r['item_id'];v=lambda mode:scores[(item,mode)]['full_logp']
        sampled=[f'sample{i}' for i in range(8)];best=max(sampled,key=v)
        for mode in ['B0_raw','raw','empty','greedy','oracle','corrupt','other']+sampled:
            row=scores[(item,mode)];assert abs(sum(row['token_logps'])-v(mode))<1e-10
            assert row['omitted_source_arrays']['target_ids']['length']==16
        values=dict(best8_minus_raw=v(best)-v('raw'),best8_minus_B0_raw=v(best)-v('B0_raw'),
            best8_minus_greedy=v(best)-v('greedy'),mean8_minus_raw=sum(v(m) for m in sampled)/8-v('raw'),
            greedy_minus_raw=v('greedy')-v('raw'),oracle_minus_corrupt=v('oracle')-v('corrupt'),oracle_minus_raw=v('oracle')-v('raw'))
        for name,value in values.items():deltas[name].append((item,value))
        category={m:correctness(notes[(item,m)]['text'],r['full_calculation_text'],r['numeric_value'])['category'] for m in sampled}
        classes.update(category.values());good=[m for m in sampled if category[m]=='verified_canonical'];bad=[m for m in sampled if category[m]=='arithmetic_error']
        helpful+=any(v(m)>max(v('raw'),v('B0_raw')) for m in good)
        if good and bad:mixed.append((item,sum(v(m) for m in good)/len(good)-sum(v(m) for m in bad)/len(bad)))
        for name in components:
            components[name].append((item,sum(x-y for x,y,b in zip(scores[(item,best)]['token_logps'],scores[(item,'raw')]['token_logps'],r['native_masks'][name]) if b)))
    for name,values in deltas.items():paired(values,report['contrasts'][name])
    for name,values in components.items():paired(values,report['best8_native_components_vs_raw'][name])
    paired(mixed,report['alignment']['conditional_contrast'])
    assert len(mixed)==report['alignment']['mixed_documents'] and helpful==report['correct_helpful_documents']
    assert dict(classes)==report['candidate_categories']
    yes=lambda name:report['contrasts'][name]['difference_b_minus_a']>=.25 and report['contrasts'][name]['ci95'][0]>0
    align=report['alignment']['conditional_contrast']
    gates=dict(complete_execution=True,opportunity=all(yes(n) for n in ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy')),
        correct_helpful_coverage=helpful>=64,alignment=len(mixed)>=32 and align['difference_b_minus_a']>=.25 and align['ci95'][0]>0,
        plumbing=report['contrasts']['oracle_minus_corrupt']['difference_b_minus_a']>=0 and report['contrasts']['oracle_minus_corrupt']['ci95'][0]>0)
    assert gates==report['gates'] and all(gates.values())==report['all_required_passed']
    assert 'torch' not in sys.modules
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'public-replay.json').write_text(json.dumps(dict(status='PASS_V30_PUBLIC_NUMERICAL_REPLAY_NO_TORCH',
        files=len(manifest['files']),attempts=len(notes),scores=len(scores),gates=gates,
        all_required_passed=all(gates.values()),limits='Verifies projected numerical evidence, not omitted source windows or model execution.'),indent=2)+'\n')
    print('PASS_PUBLIC_REPLAY',gates,flush=True)


if __name__=='__main__':main()
