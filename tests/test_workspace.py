from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file,source_fingerprint,require_utc_deadline
from weekend_thinking.checkpoints import capture_cpu_rng,restore_cpu_rng,save_rank,finalize_checkpoint,verify_checkpoint,load_cpu_rank
from weekend_thinking.order import seed_order,RecordCursor,distributed_batch_indices
from weekend_thinking.session import toy_records,run_reference
from weekend_thinking.evaluation import numeric_answer,score_predictions,downstream_contrasts
from weekend_thinking.experiment import freeze,validate_qualification,REQUIRED_CHECKS
from weekend_thinking.core import Format
from weekend_thinking.data import write_records


def test_atomic_no_clobber(tmp_path):
    path=tmp_path/'a.json';atomic_json(path,{'x':1})
    with pytest.raises(FileExistsError):atomic_json(path,{'x':2})
    assert json.loads(path.read_text())=={'x':1}
    atomic_json(path,{'x':3},overwrite=True)
    assert json.loads(path.read_text())=={'x':3}


def test_no_nan_and_ordered_hash(tmp_path):
    assert canonical_hash({'a':1,'b':2})==canonical_hash({'b':2,'a':1})
    with pytest.raises(ValueError):atomic_json(tmp_path/'bad',{'n':float('nan')})


def test_deadline_timezone():
    with pytest.raises(ValueError):require_utc_deadline('2026-10-18T12:00:00')
    assert str(require_utc_deadline('2026-10-18T12:00:00-04:00')).startswith('2026-10-18 16:00:00')


def test_source_only(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts'/'x.py').write_text('a=1\n')
    a=source_fingerprint(tmp_path)
    (tmp_path/'note.md').write_text('this is not code')
    assert source_fingerprint(tmp_path)==a
    (tmp_path/'scripts'/'x.py').write_text('a=2\n')
    assert source_fingerprint(tmp_path)!=a


def test_rng_roundtrip():
    state=capture_cpu_rng();expected=(random.random(),np.random.random(),torch.rand(3))
    restore_cpu_rng(state);actual=(random.random(),np.random.random(),torch.rand(3))
    assert expected[:2]==actual[:2];assert torch.equal(expected[2],actual[2])


def test_checkpoint_shards_and_corruption(tmp_path):
    save_rank(tmp_path/'rank-00000.pt',{'x':torch.tensor([1,2])})
    with pytest.raises(ValueError):finalize_checkpoint(tmp_path,world_size=2,step=1,run_id='t',spec_hash='h')
    save_rank(tmp_path/'rank-00001.pt',{'x':torch.tensor([3])})
    finalize_checkpoint(tmp_path,world_size=2,step=1,run_id='t',spec_hash='h')
    assert verify_checkpoint(tmp_path)['step']==1
    assert torch.equal(load_cpu_rank(tmp_path,1)['x'],torch.tensor([3]))
    with pytest.raises(ValueError):verify_checkpoint(tmp_path,expected_spec_hash='different')
    (tmp_path/'rank-00000.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):verify_checkpoint(tmp_path)


def test_order_and_no_cycle():
    records=toy_records(50)
    assert seed_order(records,1)==seed_order(records,1)
    assert seed_order(records,1)!=seed_order(records,2)
    c=RecordCursor(records,seed_order(records,1));assert len(c.take(50))==50
    with pytest.raises(ValueError):c.take(1)


@pytest.mark.parametrize('world,accum,micro',[(1,8,1),(2,4,1),(4,2,1),(2,2,2)])
def test_distributed_order_covers_batch(world,accum,micro):
    indices=[j for rank in range(world) for j in distributed_batch_indices(3,rank,world,accum,micro)]
    assert sorted(indices)==list(range(24,32))


@pytest.mark.parametrize('arm',['thought','vanilla'])
def test_exact_cpu_session_resume(tmp_path,arm):
    continuous=run_reference(out=tmp_path/'all',arm=arm,steps=4)
    partial=run_reference(out=tmp_path/'resumed',arm=arm,steps=4,stop_after=2)
    resumed=run_reference(out=tmp_path/'resumed',arm=arm,steps=4,resume=partial['checkpoint'])
    a=load_cpu_rank(continuous['checkpoint']);b=load_cpu_rank(resumed['checkpoint'])
    for k in a['model']:assert torch.equal(a['model'][k],b['model'][k])
    assert continuous['metrics']==resumed['metrics']
    assert continuous['order_hash']==resumed['order_hash']
    with pytest.raises(ValueError):run_reference(out=tmp_path/'wrong',arm=arm,steps=5,resume=partial['checkpoint'])


@pytest.mark.parametrize('text,expected',[
    ('work\n#### 42','42'),('#### -1,250.50','-1250.5'),('#### 0.00','0'),
    ('#### .5','.5'),('#### 42.','42'),('The answer is 42',None),
    ('#### 42 then 13',None),('#### 1/2',None),('#### NaN',None),('#### 12,34',None)])
def test_numeric_parser(text,expected):
    if expected=='.5':expected='0.5'
    assert numeric_answer(text)==expected


def test_prediction_failures_stay_in_denominator():
    gold=[{'item_id':str(i),'answer':'#### 2'} for i in range(4)]
    pred=[{'item_id':'0','generated_text':'#### 2','status':'ok'},
          {'item_id':'1','generated_text':'#### 2','status':'truncated'},
          {'item_id':'2','generated_text':'2','status':'ok'},
          {'item_id':'3','generated_text':'','status':'error'}]
    rows,result=score_predictions(gold,pred)
    assert result['accuracy']==.25 and result['items']==4
    assert result['statuses']=={'ok':1,'truncated':1,'error':1,'format_error':1}
    with pytest.raises(ValueError):score_predictions(gold,pred[:-1])


def test_all_downstream_contrasts():
    rows=lambda v:[{'item_id':str(i),'document_id':str(i),'score':x} for i,x in enumerate(v)]
    a,b,c,d=map(rows,[[0,0,1],[0,1,1],[1,1,1],[1,1,1]])
    result=downstream_contrasts(a,b,c,d,repeats=100)
    assert result['primary_R_thought_minus_C_thought']['difference_b_minus_a']==pytest.approx(1/3)
    assert result['training_R_direct_minus_C_direct']['difference_b_minus_a']==pytest.approx(2/3)
    assert result['inference_R_thought_minus_R_direct']['difference_b_minus_a']==0


@pytest.fixture
def synthetic_frozen(tmp_path):
    # Entirely synthetic filesystem fixture; this is NOT evidence of Neuron execution.
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts'/'driver.py').write_text('# test fixture\n')
    model=tmp_path/'model';model.mkdir()
    atomic_json(model/'config.json',{'model_type':'qwen3','vocab_size':31})
    (model/'model.safetensors').write_bytes(b'synthetic fixture, not model weights')
    asset={'model_id':'Qwen/Qwen3-1.7B-Base','resolved_revision':'a'*40,
           'files':{x.name:{'sha256':sha256_file(x)} for x in model.iterdir()}}
    atomic_json(model/'ASSET_MANIFEST.json',asset)
    records=tmp_path/'records.jsonl';write_records(records,toy_records(8))
    atomic_json(records.with_suffix('.manifest.json'),{'output_sha256':sha256_file(records),
                'tokenizer_identity':asset['model_id']+'@'+asset['resolved_revision']})
    fmt=tmp_path/'format.json';atomic_json(fmt,{'format':asdict(Format((29,),(30,),30,0)),'tokenizer_dir':str(model)})
    proposal={'optimization':{'step_budget_candidates':[2],'global_unique_prefixes_per_update':2,'micro_prefixes_per_rank':1},
              'model':{'id':asset['model_id']},'data':{'prefix_tokens':6,'stored_future_tokens':4},
              'thought_arm':{'reward_horizon':1}}
    site={'model_dir':str(model),'records_file':str(records),'format_file':str(fmt),
          'world_size_per_arm':1,'gradient_accumulation':2,'deadline_utc':'2099-01-01T00:00:00Z'}
    from weekend_thinking.runtime_policy import runtime_policy
    site.update(runtime_policy_by_arm={arm:runtime_policy({}) for arm in ('vanilla','thought')},
                checkpoint_interval_seconds=43200)
    frozen=freeze(tmp_path,proposal,site,experiment_id='unit-test-only',steps=2)
    return tmp_path,proposal,site,frozen


def test_freeze_and_data_binding(synthetic_frozen):
    root,proposal,site,frozen=synthetic_frozen
    assert frozen['training_exposure_per_arm']['prefix_visits']==4
    bad=dict(site,gradient_accumulation=1)
    with pytest.raises(ValueError):freeze(root,proposal,bad,experiment_id='unit-test-2',steps=2)
    Path(site['records_file']).write_text('changed')
    with pytest.raises(ValueError):freeze(root,proposal,site,experiment_id='unit-test-3',steps=2)


def test_hardware_receipt_cannot_use_cpu(synthetic_frozen):
    root,_,_,frozen=synthetic_frozen
    with pytest.raises(ValueError,match='CPU/synthetic'):
        validate_qualification(root,frozen,{'platform':'cpu','synthetic':True})


def test_receipt_evidence_and_source_binding(synthetic_frozen):
    root,_,_,frozen=synthetic_frozen
    log=root/'fixture.log';log.write_text('UNIT TEST ONLY - no device execution')
    checks={n:{'passed':True,'command':'synthetic unit test','observations':'not a real run',
               'evidence_file':str(log),'evidence_sha256':sha256_file(log)} for n in REQUIRED_CHECKS}
    receipt={'platform':'neuron','synthetic':False,'spec_hash':frozen['spec_hash'],
             'source_sha256':frozen['source']['sha256'],'checks':checks,'full_step_seconds':[1.0]*10}
    validate_qualification(root,frozen,receipt)
    log.write_text('changed')
    with pytest.raises(ValueError,match='Evidence changed'):validate_qualification(root,frozen,receipt)
