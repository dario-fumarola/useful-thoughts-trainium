"""Interrupted log/partial-state preservation and uniform recovery refusal."""
import json
from pathlib import Path

import pytest
import torch

from train import useful_note_sft_driver as d
from weekend_thinking.artifacts import canonical_hash,sha256_file


def interrupted(tmp_path):
    output=tmp_path/'T-run';checkpoint=output/'step-0000003';checkpoint.mkdir(parents=True)
    p=dict(world_size=32,accumulation=1,global_batch=32,steps=320,items=5120,seed=17)
    rows=[dict(item_id=f'item-{i}') for i in range(5120)];order=list(range(5120))*2
    run=dict(schema=d.SCHEMA,run_id=output.name,arm='T',protocol_sha256='protocol',
        spec_hash=canonical_hash(p),order_hash=canonical_hash([rows[i]['item_id'] for i in order]),
        world_size=32,accumulation=1,purpose='science',items=5120,scientific_steps=320,qualification=False)
    (output/'run.json').write_text(json.dumps(run))
    files={}
    for name in ['model.pt']+[f'rank-{rank:05d}.pt' for rank in range(32)]:
        path=checkpoint/name;path.write_bytes(name.encode())
        files[name]=dict(bytes=path.stat().st_size,sha256=sha256_file(path))
    m=dict(schema='neuron-v1',run_id=output.name,step=3,world_size=32,spec_hash=canonical_hash(p),
        metadata=run,files=files)
    (checkpoint/'COMPLETE.json').write_text(json.dumps(m))
    lines=[json.dumps(dict(step=i+1,ordered_item_ids=[rows[k]['item_id'] for k in order[i*32:(i+1)*32]],
        counters=dict(source_records=(i+1)*32)))+'\n' for i in range(4)]
    raw=(''.join(lines)+'{"step":5,"interrupted":').encode();(output/'metrics.jsonl').write_bytes(raw)
    (output/'completion.json').write_text(json.dumps(dict(status='stopped',actual_steps=3)))
    partial=output/'step-0000320';partial.mkdir();(partial/'model.pt').write_bytes(b'partial terminal payload')
    dev=output/'development/step-0000064';dev.mkdir(parents=True);(dev/'COMPLETE.json').write_text('{"resumable":false}')
    return output,checkpoint,p,rows,order,raw,b''.join(line.encode() for line in lines[:3])


def test_resume_preserves_entire_tail_and_partial_checkpoints_without_counting_them(tmp_path):
    output,checkpoint,p,rows,order,raw,prefix=interrupted(tmp_path)
    m,segment=d.prepare_resume(output,checkpoint,'attempt-2',p,'T','protocol',rows,order)
    assert m['step']==3 and (segment/'previous-metrics.jsonl').read_bytes()==raw
    assert (output/'metrics.jsonl').read_bytes()==prefix
    assert (segment/'abandoned/step-0000320/model.pt').read_bytes()==b'partial terminal payload'
    assert (segment/'abandoned/development/step-0000064/COMPLETE.json').is_file()
    assert (checkpoint/'COMPLETE.json').is_file() and (output/'run.json').is_file()
    assert json.loads((segment/'previous-completion.json').read_text())['actual_steps']==3
    receipt=json.loads((segment/'prepared.json').read_text())
    assert receipt['abandoned_metric_bytes']==len(raw)-len(prefix)
    assert receipt['checkpoint_complete_sha256']==sha256_file(checkpoint/'COMPLETE.json')


@pytest.mark.parametrize('defect',['corrupt_last_shard','wrong_arm','wrong_protocol','bad_metric_order','older_than_complete','completed','unsafe_attempt','reused_attempt'])
def test_resume_rejects_invalid_inputs_before_rewriting_metrics(tmp_path,defect):
    output,checkpoint,p,rows,order,raw,prefix=interrupted(tmp_path);attempt='new';arm='T';protocol='protocol'
    if defect=='corrupt_last_shard':(checkpoint/'rank-00031.pt').write_bytes(b'corrupt')
    elif defect=='wrong_arm':arm='W'
    elif defect=='wrong_protocol':protocol='other'
    elif defect=='bad_metric_order':
        lines=(output/'metrics.jsonl').read_bytes().splitlines(keepends=True)
        value=json.loads(lines[0]);value['ordered_item_ids'][0]='wrong';lines[0]=(json.dumps(value)+'\n').encode()
        raw=b''.join(lines);(output/'metrics.jsonl').write_bytes(raw)
    elif defect=='older_than_complete':
        later=output/'step-0000004';later.mkdir();(later/'COMPLETE.json').write_text('{}')
    elif defect=='completed':(output/'completion.json').write_text('{"status":"complete"}')
    elif defect=='unsafe_attempt':attempt='../outside'
    else:(output/'recoveries/new').mkdir(parents=True)
    with pytest.raises(ValueError):d.prepare_resume(output,checkpoint,attempt,p,arm,protocol,rows,order)
    assert (output/'metrics.jsonl').read_bytes()==raw
    assert (output/'step-0000320/model.pt').read_bytes()==b'partial terminal payload'
    assert not (tmp_path/'outside').exists()


@pytest.mark.parametrize('stage',['payload','restored'])
def test_one_bad_recovery_rank_stops_every_rank_after_retaining_local_evidence(tmp_path,stage):
    errors=[None,'bad saved counter',None]
    for rank,error in enumerate(errors):
        directory=tmp_path/str(rank);directory.mkdir()
        class XM:
            def all_gather(self,value):
                assert value.item()==float(bool(error))
                return torch.tensor([0.,1.,0.])
            def rendezvous(self,tag):
                assert tag==f'recovery-{stage}-consensus-published'
                assert (directory/f'{stage}-consensus-rank-{rank:05d}.json').is_file()
        with pytest.raises(RuntimeError,match='all ranks stop'):
            d.recovery_consensus(directory,rank,3,error,XM(),'cpu',stage)
        local=json.loads((directory/f'{stage}-local-rank-{rank:05d}.json').read_text())
        global_=json.loads((directory/f'{stage}-consensus-rank-{rank:05d}.json').read_text())
        assert local['error']==error and global_['failed_ranks']==[1]


@pytest.mark.parametrize('stop_requested',[False,True])
def test_stop_and_sync_observations_finish_before_final_rng_reset(stop_requested):
    import random
    import numpy as np
    from weekend_thinking.checkpoints import capture_cpu_rng,restore_cpu_rng
    from types import SimpleNamespace
    original=capture_cpu_rng()
    try:
        random.seed(73);np.random.seed(74);torch.manual_seed(75)
        saved=dict(cpu=capture_cpu_rng(),device=902)
        expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
        events=[];device_state=[1000]
        def advance(name):
            events.append(name);device_state[0]+=1
            random.random();np.random.rand();torch.rand(())
        def stopped():advance('stop-collective');return stop_requested
        def sync(**kwargs):advance('pending-load-sync')
        def reset(value,device):events.append('reset');device_state[0]=value
        xm=SimpleNamespace(set_rng_state=reset)
        assert d.resume_boundary(stopped,saved,xm,'cpu',sync) is (not stop_requested)
        assert events==['stop-collective','pending-load-sync','reset']
        assert device_state[0]==902
        assert (random.random(),float(np.random.rand()),float(torch.rand(())))==expected
    finally:restore_cpu_rng(original)
