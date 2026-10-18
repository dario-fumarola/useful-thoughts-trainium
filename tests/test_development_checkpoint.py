"""A model-only milestone must remain distinct from complete scientific state."""
import copy
import json

import pytest

from train import development_checkpoint as d
from weekend_thinking.artifacts import canonical_hash, sha256_file


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    source=tmp_path/'train/frozen.py';source.parent.mkdir();source.write_text('frozen source\n')
    data=tmp_path/'training.jsonl';data.write_text('immutable data\n')
    dataset=tmp_path/'dataset.json';dataset.write_text('{}\n')
    p=dict(schema='useful-note-sft-v4',purpose='science',steps=320,global_batch=32,
           world_size=32,items=5120,seed=17,data_file=str(data),data_sha256=sha256_file(data),
           dataset_file=str(dataset),dataset_sha256=sha256_file(dataset),
           source_files={'train/frozen.py':sha256_file(source)})
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(p))
    monkeypatch.setattr(d,'PROTOCOL_SHA256',sha256_file(protocol))
    run=tmp_path/'actual-W';snap=run/'development/step-0000064';snap.mkdir(parents=True)
    model=snap/'model.pt';model.write_bytes(b'exact model bytes; real model loading is separately strict')
    r=dict(schema=p['schema'],purpose='science',qualification=False,arm='W',run_id=run.name,
           spec_hash=canonical_hash(p),protocol_sha256=sha256_file(protocol),order_hash='a'*64)
    (run/'run.json').write_text(json.dumps(r))
    metrics=[dict(step=i,counters=dict(source_records=i*32,source_tokens=i*32*528),
                  sampled_replicas_bitwise_equal=True,sampled_active_model_matches_cast_master=True) for i in range(1,65)]
    (run/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metrics))
    metadata=dict(qualification=False,purpose='science',arm='W',seed=17,spec_hash=r['spec_hash'],
                  order_hash=r['order_hash'],protocol_sha256=r['protocol_sha256'],
                  data_sha256=p['data_sha256'],dataset_sha256=p['dataset_sha256'],
                  parameter_dtype='bf16',gradient_dtype='bf16',optimizer_dtype='fp32',
                  global_batch=32,items=5120,counters=metrics[-1]['counters'])
    c=dict(schema='useful-note-model-only-v1',resumable=False,step=64,run_id=run.name,
           spec_hash=r['spec_hash'],metadata=metadata,
           files={'model.pt':dict(bytes=model.stat().st_size,sha256=sha256_file(model))})
    complete=snap/'COMPLETE.json';complete.write_text(json.dumps(c))
    return dict(model=model,complete=complete,protocol=protocol,run=run,manifest=c,source=source,metrics=metrics)


def resolve(s):
    return d.identity(s['model'],sha256_file(s['model']),sha256_file(s['complete']),s['protocol'])


def test_saved64_identity_has_no_optimizer_or_terminal_claim(saved):
    v=resolve(saved)
    assert v['step']==64 and v['developmental_only'] is True and v['resumable'] is False
    assert v['training_plan_terminal_step']==320 and len(v['checkpoint_files'])==1
    assert d.verify(v)==v
    for field,value in [('step',320),('id','Cprime'),('developmental_only',False),('resumable',True)]:
        with pytest.raises(ValueError):d.verify(dict(v,**{field:value}))


@pytest.mark.parametrize('field,value', [('step',63),('step',320),('step',True),
    ('resumable',True),('schema','neuron-v1'),('run_id','different-run')])
def test_hash_consistent_wrong_milestone_is_rejected(saved,field,value):
    c=copy.deepcopy(saved['manifest']);c[field]=value;saved['complete'].write_text(json.dumps(c))
    with pytest.raises(ValueError):resolve(saved)


def test_future_counters_missing_updates_or_source_drift_are_rejected(saved):
    c=copy.deepcopy(saved['manifest']);c['metadata']['counters']['source_records']=4096
    saved['complete'].write_text(json.dumps(c))
    with pytest.raises(ValueError,match='first64'):resolve(saved)
    saved['complete'].write_text(json.dumps(saved['manifest']))
    rows=copy.deepcopy(saved['metrics']);rows[10]['step']=42
    (saved['run']/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='first64'):resolve(saved)
    (saved['run']/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in saved['metrics']))
    saved['source'].write_text('changed after training')
    with pytest.raises(ValueError,match='source closure'):resolve(saved)


def test_model_bytes_and_model_symlink_cannot_change(saved):
    digest=sha256_file(saved['model']);saved['model'].write_bytes(b'different')
    with pytest.raises(ValueError,match='hashes'):
        d.identity(saved['model'],digest,sha256_file(saved['complete']),saved['protocol'])
    target=saved['model'].with_suffix('.data');saved['model'].rename(target);saved['model'].symlink_to(target)
    with pytest.raises(ValueError,match='Ordinary'):resolve(saved)


def test_terminal_validator_still_rejects_model_only64(saved):
    from train.useful_note_evaluation import trained_identity
    with pytest.raises((ValueError,FileNotFoundError)):
        trained_identity(saved['model'],sha256_file(saved['model']),sha256_file(saved['complete']),saved['protocol'])


def test_protocol_cannot_be_rebound_to_another_learning_recipe(saved):
    p=json.loads(saved['protocol'].read_text());p['lr']=100
    saved['protocol'].write_text(json.dumps(p))
    with pytest.raises(ValueError,match='original seed'):resolve(saved)
