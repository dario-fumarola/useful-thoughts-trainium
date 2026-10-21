"""Independent HF/FP64 behavior-LP check across every qualification batch row.

Checks actual cached actions against ordinary full-prefix forwards, including
rows1..7 after sampled histories diverge. It does not assess task correctness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from train.evaluate import load_own_checkpoint
from train.real_trace_contexts import writer_prompt
from weekend_thinking.artifacts import atomic_json, sha256_file
from weekend_thinking.pretrained import load_base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args(); protocol=json.loads((a.protocol/'protocol.json').read_text())
    complete=json.loads((a.qualification/'sampling-complete.json').read_text())
    assert complete['notes_sha256']==sha256_file(a.qualification/'notes.jsonl')
    assert complete['protocol_sha256']==sha256_file(a.protocol/'protocol.json')
    questions={r['item_id']:r for r in map(json.loads,(a.protocol/'qualification-questions.jsonl').open())}
    attempts=list(map(json.loads,(a.qualification/'notes.jsonl').open()))
    assert len(questions)==8 and len(attempts)==72
    assert {(r['item_id'],r['mode']) for r in attempts}=={
        (k,m) for k in questions for m in ['greedy']+[f'sample{i}' for i in range(8)]}
    torch.set_num_threads(16)
    model,tok=load_base('/home/ubuntu/dario/local/assets/qwen3-1.7b-base')
    loaded=load_own_checkpoint(model,Path(protocol['model']['checkpoint']))
    assert loaded['sha256']==protocol['model']['checkpoint_sha256']
    model.eval()
    defined=set(tok.get_vocab().values());banned=sorted(set(range(model.config.vocab_size))-defined)
    checks=[]; deadline=datetime.fromisoformat(a.deadline_utc)
    with torch.no_grad():
        for r in attempts:
            assert datetime.now(timezone.utc)<deadline
            context=writer_prompt(questions[r['item_id']],tok); actions=r['actions']
            assert all(t in defined for t in actions)
            ids=torch.tensor([context+actions]); logits=model(ids).logits[0,len(context)-1:-1].double()/.7
            logits[:,banned]=-torch.inf
            lp=(logits-logits.logsumexp(-1,keepdim=True)).gather(-1,ids[0,len(context):,None])[:,0]
            assert len(lp)==len(r['behavior_logps'])==len(actions)
            error=max(abs(x-y) for x,y in zip(lp.tolist(),r['behavior_logps']))
            assert error<=.001
            checks.append(dict(item_id=r['item_id'],mode=r['mode'],tokens=len(actions),max_selected_logp_error=error))
            if len(checks)%9==0: print('CHECKED_ACTUAL_QUALIFICATION_ATTEMPTS',len(checks),flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'all-batch-rows.json',dict(status='PASS_ALL72_ACTUAL_ROLLOUTS_HF_FP64_BEHAVIOR_LOGPS',
        attempts=len(checks),tokens=sum(r['tokens'] for r in checks),checks=checks,
        maximum_selected_logp_error=max(r['max_selected_logp_error'] for r in checks),
        notes_sha256=complete['notes_sha256'],protocol_sha256=complete['protocol_sha256'],
        checker_sha256=sha256_file(__file__),limits='Execution fidelity only; no correctness, learning or generalization claim.'))


if __name__=='__main__':main()
