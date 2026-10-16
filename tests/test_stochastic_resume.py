"""CPU fixtures for first resumed-step evidence; no device/resume claim."""
from __future__ import annotations

import json

import pytest
import torch

from scripts.compare_stochastic_resume import compare_runs
from weekend_thinking.artifacts import canonical_hash


@pytest.fixture
def outputs(tmp_path):
    roots = [tmp_path/side/'same-run' for side in ('baseline','resumed')]
    spec = dict(stage='qualification', arm='thought', seed=1234, precision='fp32',
                world_size=2,global_prefixes=4,steps=5,cap=7)
    spec_hash, order_hash = canonical_hash(spec), 'a'*64
    run = dict(spec=spec,spec_hash=spec_hash,run_id='same-run',arm='thought',seed=1234)
    metadata = dict(spec_hash=spec_hash,arm='thought',seed=1234,global_batch=4,
                    accumulation=2,order_hash=order_hash)
    manifest = dict(schema='neuron-v1',step=3,run_id='same-run',world_size=2,
                    spec_hash=spec_hash,metadata=metadata)
    for root in roots:
        (root/'step-0000003').mkdir(parents=True)
        (root/'run.json').write_text(json.dumps(run))
        (root/'step-0000003/COMPLETE.json').write_text(json.dumps(manifest))
        tapes = root/'qualification-tapes/step-0000004'
        tapes.mkdir(parents=True)
        for rank in range(2):
            for micro in range(2):
                lengths = torch.tensor([[1,3,7,5]])
                mask = torch.arange(7)[None,None,:] < lengths[:,:,None]
                actions = torch.arange(28).reshape(1,4,7)
                tape = dict(schema='qualification-rollout-v1',run_id='same-run',
                    spec_hash=spec_hash,order_hash=order_hash,arm='thought',seed=1234,
                    step=4,rank=rank,world_size=2,micro=micro,global_batch=4,
                    accumulation=2,record_id=f'record-{rank}-{micro}',
                    ordered_record_index=12+micro*2+rank,policy_version=3,
                    temperature=.7,decode_iterations=7,actions=actions,action_mask=mask,
                    behavior_logps=torch.full((1,4,7),-2.),clean=torch.tensor([-3.]),
                    thought=torch.tensor([[-2.,-4.,-3.,-3.]]),
                    advantages=torch.tensor([[1.,-1.,0.,0.]]),lengths=lengths)
                torch.save(tape,tapes/f'rank-{rank:05d}-micro-{micro:03d}.pt')
    criteria = tmp_path/'criteria.json'
    criteria.write_text(json.dumps(dict(atol=1e-7,rtol=1e-6)))
    return roots, criteria


def compare(outputs):
    roots, criteria = outputs
    return compare_runs(*roots,resume_step=3,criteria_path=criteria)


def mutate(outputs, field, function, rank=1, micro=1):
    path = outputs[0][1]/f'qualification-tapes/step-0000004/rank-{rank:05d}-micro-{micro:03d}.pt'
    tape = torch.load(path,weights_only=True)
    tape[field] = function(tape[field])
    torch.save(tape,path)


def test_exact_all_rank_micro_tapes_pass_without_full_state_claim(outputs):
    result = compare(outputs)
    assert result['pass'] and result['all_tapes_bitwise_equal']
    assert len(result['rows']) == 4
    assert not result['full_state_resume_qualified']


def test_last_rank_micro_action_divergence_is_failure(outputs):
    mutate(outputs,'actions',lambda x: x+1)
    result = compare(outputs)
    assert not result['pass']
    assert result['first_divergence'] == dict(step=4,rank=1,micro=1,field='actions',
        kind='exact',index=[0,0,0],expected=0,actual=1)


def test_tight_logp_tolerance_is_separate_from_replay_ceiling(outputs):
    mutate(outputs,'behavior_logps',lambda x: x+1e-5)
    result = compare(outputs)
    assert not result['pass']
    assert result['first_divergence']['field'] == 'behavior_logps'
    assert result['rows'][-1]['numeric']['behavior_logps']['max_abs'] < .001


def test_coherent_changed_stop_mask_still_requires_exact_match(outputs):
    path = outputs[0][1]/'qualification-tapes/step-0000004/rank-00001-micro-001.pt'
    tape = torch.load(path,weights_only=True)
    tape['action_mask'][0,1,2] = False
    tape['lengths'][0,1] = 2
    torch.save(tape,path)
    result = compare(outputs)
    assert not result['pass']
    assert result['first_divergence']['field'] == 'action_mask'
    assert result['first_divergence']['kind'] == 'exact'


def test_numeric_within_predeclared_bound_passes_but_is_not_exact(outputs):
    mutate(outputs,'thought',lambda x: x+5e-7)
    result = compare(outputs)
    assert result['pass'] and not result['all_tapes_bitwise_equal']


@pytest.mark.parametrize('field,change',[
    ('policy_version',lambda x:x+1),('ordered_record_index',lambda x:x+1),
    ('order_hash',lambda x:'b'*64),('decode_iterations',lambda x:x-1),
    ('lengths',lambda x:x+1),('action_mask',lambda x:~x),
    ('behavior_logps',lambda x:torch.full_like(x,float('nan'))),
])
def test_identity_masks_lengths_and_finiteness_fail_closed(outputs,field,change):
    mutate(outputs,field,change)
    assert not compare(outputs)['pass']


def test_complete_coverage_required_and_first_step_not_skipped(outputs):
    root = outputs[0][1]
    (root/'qualification-tapes/step-0000004/rank-00001-micro-001.pt').unlink()
    result = compare(outputs)
    assert not result['pass'] and result['first_divergence']['kind'] == 'coverage'
    assert result['compared_step'] == 4


def test_baseline_tape_hardlink_is_not_fresh_process_evidence(outputs):
    roots, _ = outputs
    relative = 'qualification-tapes/step-0000004/rank-00000-micro-000.pt'
    (roots[1]/relative).unlink()
    (roots[1]/relative).hardlink_to(roots[0]/relative)
    result = compare(outputs)
    assert not result['pass'] and 'aliases' in result['first_divergence']['error']


def test_relaxed_criteria_and_changed_resume_manifest_rejected(outputs):
    roots, criteria = outputs
    original = criteria.read_text()
    criteria.write_text(json.dumps(dict(atol=.001,rtol=1e-6)))
    with pytest.raises(ValueError,match='retain'):
        compare(outputs)
    criteria.write_text(original)
    path = roots[1]/'step-0000003/COMPLETE.json'
    manifest = json.loads(path.read_text())
    manifest['step'] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='manifests differ'):
        compare(outputs)
