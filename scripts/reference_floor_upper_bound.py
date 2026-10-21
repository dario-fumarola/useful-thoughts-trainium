"""Post-result proposal bound from public V30 scores; no models or policy sweep.

This oracle knows each gold token. It is only a ceiling on a future causal gate
over the fixed saved candidates at the single, previously recorded gate cap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','proposal','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();proposal=json.loads(a.proposal.read_text())
    cap=proposal['mechanism']['gate_bound']['maximum_a'];score={};pins={}
    for tag in ('score-T','score-B0'):
        for shard in range(4):
            path=a.evidence/f'outputs/{tag}-{shard}.jsonl'
            pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.open():
                row=json.loads(line);key=(row['item_id'],row['mode'])
                assert key not in score;score[key]=row
    ids=sorted({k[0] for k in score});assert len(ids)==256
    rows=[]
    for item in ids:
        base=np.array(score[(item,'B0_raw')]['token_logps'],dtype=np.float64)
        def upper(mode):
            assisted=np.array(score[(item,mode)]['token_logps'],dtype=np.float64)
            delta=np.logaddexp(math.log1p(-cap)+base,math.log(cap)+assisted)-base
            return float(np.maximum(delta,0).sum())
        rows.append(dict(item_id=item,greedy_with_hindsight_token_gate_upper=upper('greedy'),
            best8_with_hindsight_note_and_token_gate_upper=max(upper(f'sample{i}') for i in range(8)),
            supplied_correct_with_hindsight_token_gate_upper=upper('oracle')))
    summary={}
    for key in rows[0]:
        if key=='item_id':continue
        x=np.array([v[key] for v in rows]);summary[key]=dict(mean_nats_per16=float(x.mean()),
            median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()),
            documents_above_point25=int((x>.25).sum()))
    result=dict(status='EXPLORATORY_CPU_MATHEMATICAL_UPPER_BOUND_NOT_A_DEPLOYED_RESULT',
        proposal_sha256=hashlib.sha256(a.proposal.read_bytes()).hexdigest(),fixed_maximum_gate=cap,
        documents=256,summary=summary,rows=rows,source_pins=pins,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        derivation='For a fixed candidate and gold target, log((1-a)*p0+a*p1) is monotone in a. Maximize over[0,cap] at0 orcap, sum these tokenwise hindsight maxima, then optionally maximize over the8saved notes.',
        decision='Do not launch the proposed pointwise-capped gate over these fixed candidates on this evidence.',
        limits=['Inspected V30 TRAIN data; post-result proposal arithmetic,not confirmation or a new V30 endpoint.',
            'The gate sees the current gold token and best8 sees the continuation; neither is a deployed causal policy.',
            'The pointwise bound is stricter than an average retention constraint. This does not rule out other gates, readers or candidate distributions.',
            'Only a single cap chosen from the recorded mathematical retention bound is used. No policy sweep, new model call, training or reopening of V30.'])
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
