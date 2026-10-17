"""Fresh B4/G1 semantics, complete-bundle barrier, and explicit model bindings."""
import json
from types import SimpleNamespace

import pytest
import torch

import train.warmstart_evaluate as probe
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def test_distinct_prefix_batch_g1_rng_and_failed_replay_keep_genuine_samples():
    fmt=Format((2,),(3,),5,0,additional_stop_ids=(6,))
    prefixes=[[10+i,20+i] for i in range(4)]
    class Backend:
        device=torch.device('cpu')
        def __init__(self):self.events=[];self.bad=False
        def sync(self):self.events.append('sync')
        def seed(self,value):self.events.append('seed');torch.manual_seed(value)
        def sample(self,value,**kw):
            self.events.append('sample');assert value.tolist()==prefixes
            assert kw['group_size']==1 and kw['max_actions']==256 and kw['temperature']==.7
            actions=torch.randint(10,30,(4,1,256));mask=torch.zeros_like(actions,dtype=torch.bool)
            for i,n in enumerate([1,3,256,2]):
                mask[i,0,:n]=True
                if n<256:actions[i,0,n-1]=6 if i==3 else 5
            return SimpleNamespace(actions=actions,action_mask=mask,behavior_logps=torch.full(actions.shape,-2.),decode_iterations=256)
        def replay(self,value,rollout,**kw):
            assert value.tolist()==prefixes
            return rollout.behavior_logps+(.01 if self.bad else 0.)
    backend=Backend();seed=probe.batch_seed(['a','b','c','d'])
    first=probe.generate_batch(backend,prefixes,fmt=fmt,seed=seed,set_seed=backend.seed,audit=True)
    assert backend.events[:4]==['sync','seed','sample','sync']
    assert all(r['status']=='ok' and r['batch_row']==i and r['cache_replay_max_error']==0 for i,r in enumerate(first))
    assert first[0]['content_ids']==[] and first[2]['stop_type']=='length_cap' and first[3]['stop_type']=='token:6'
    backend.bad=True
    failed=probe.generate_batch(backend,prefixes,fmt=fmt,seed=seed,set_seed=backend.seed,audit=True)
    assert all(r['status']=='error' for r in failed)
    assert [r['actions'] for r in first]==[r['actions'] for r in failed]
    assert [r['behavior_logps'] for r in first]==[r['behavior_logps'] for r in failed]
    assert seed!=probe.batch_seed(['b','a','c','d'])


def protocol():
    ids=[f'item-{i:02d}' for i in range(64)]
    return dict(selection=dict(item_ids=ids,document_ids=['doc:'+k for k in ids]),writer=probe.model_identity('B0'),controls_seed=17)


def test_fixed_shards_and_global_donors_cross_shard_boundaries():
    p=protocol();ids=p['selection']['item_ids']
    assert sum((probe.shard_ids(p,s) for s in range(4)),[])==ids
    assert len({probe.batch_seed(ids[i:i+4]) for i in range(0,64,4)})==16
    notes={k:dict(content_ids=[i+10]) for i,k in enumerate(ids)}
    controls=probe.make_controls(p,notes,[9])
    assert controls[15]['donor_item_id']==ids[16] and controls[15]['content_ids']['other']==[26]
    assert controls[-1]['donor_item_id']==ids[0] and controls[-1]['content_ids']['other']==[10]
    assert controls==probe.make_controls(p,notes,[9])


def test_missing_notes_block_future_and_model_access(tmp_path,monkeypatch):
    p=protocol();fmt=Format((2,),(3,),5,0)
    monkeypatch.setattr(probe,'read_protocol',lambda _: (p,fmt))
    monkeypatch.setattr(probe,'load_selected',lambda *a,**k:pytest.fail('Future loading before complete notes'))
    monkeypatch.setattr(probe,'load_model',lambda *a,**k:pytest.fail('Model loading before complete notes'))
    with pytest.raises(ValueError,match='All four complete'):
        probe.score(tmp_path,0,'cpu',probe.model_identity('B0'))


def test_complete_notes_bind_documents_stop_metadata_and_actual_writer(tmp_path):
    p=protocol();fmt=Format((2,),(3,),5,0);(tmp_path/'protocol.json').write_text('{}')
    identity=dict(**p['writer'],loaded_checkpoint=None,model={'fingerprint':'x'},base_manifest_sha256='a'*64)
    for shard in range(4):
        d=tmp_path/'notes'/f'shard-{shard}';d.mkdir(parents=True)
        ids=probe.shard_ids(p,shard);rows=[]
        for i,key in enumerate(ids):
            batch=ids[i//4*4:i//4*4+4]
            rows.append(dict(item_id=key,document_id='doc:'+key,split='dev',status='ok',actions=[10,5],content_ids=[10],behavior_logps=[-1.,-1.],action_mask=[True,True]+[False]*254,n_actions=2,n_content_tokens=1,stop_type='token:5',batch_item_ids=batch,batch_seed=probe.batch_seed(batch),batch_row=i%4,cache_replay_max_error=0. if i<4 else None))
        (d/'notes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (d/'run.json').write_text(json.dumps(dict(status='complete',writer=identity,protocol_sha256=sha256_file(tmp_path/'protocol.json'),notes_sha256=sha256_file(d/'notes.jsonl'),item_ids=ids)))
    notes,writer=probe.load_notes(tmp_path,p,fmt);assert len(notes)==64 and writer==identity
    path=tmp_path/'notes/shard-3/notes.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['document_id']='wrong';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rpath=path.parent/'run.json';report=json.loads(rpath.read_text());report['notes_sha256']=sha256_file(path);rpath.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Invalid complete'):
        probe.load_notes(tmp_path,p,fmt)


def test_checkpoint_hash_is_explicit_and_checked_before_model_access(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Only unchanged B0'):
        probe.model_identity('student')
    path=tmp_path/'model.pt';path.write_bytes(b'actual checkpoint bytes')
    identity=probe.model_identity('student',path,'0'*64)
    monkeypatch.setattr(probe,'setup_model',lambda *a,**k:pytest.fail('Unverified checkpoint loaded'))
    with pytest.raises(ValueError,match='SHA mismatch'):
        probe.load_model({},identity,'cpu')


@pytest.mark.parametrize('returned',[[float('nan')]*16,[-1.]*15])
def test_invalid_reader_vector_is_retained_as_failed_row(tmp_path,monkeypatch,returned):
    p=protocol();p['width']=1024;p['selection']['records_file']='unopened-records.jsonl'
    fmt=Format((2,),(3,),5,0);(tmp_path/'protocol.json').write_text('{}')
    notes={k:dict(content_ids=[10],prefix_sha256=canonical_hash([10,11]),status='ok',n_actions=2,n_content_tokens=1,stop_type='token:5') for k in p['selection']['item_ids']}
    values=probe.make_controls(p,notes,[9]);path=tmp_path/'controls.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in values))
    (tmp_path/'controls.complete.json').write_text(json.dumps(dict(status='complete',writer=p['writer'],items=64,filler_ids=[9],protocol_sha256=sha256_file(tmp_path/'protocol.json'),notes_sha256=canonical_hash(notes),controls_sha256=sha256_file(path))))
    monkeypatch.setattr(probe,'read_protocol',lambda _: (p,fmt))
    monkeypatch.setattr(probe,'load_notes',lambda *a: (notes,p['writer']))
    key=p['selection']['item_ids'][0]
    monkeypatch.setattr(probe,'load_selected',lambda *a,**k: [dict(item_id=key,document_id='doc:'+key,prefix=[10,11],future=[17]*16)])
    monkeypatch.setattr(probe,'load_model',lambda *a: (None,None,p['writer'],None))
    monkeypatch.setattr(probe,'score_continuations',lambda *a,**k: [SimpleNamespace(token_log_likelihoods=returned)])
    with pytest.raises(RuntimeError,match='Scoring failure retained'):
        probe.score(tmp_path,0,'cpu',p['writer'])
    d=tmp_path/'readers/B0/shard-0'
    rows=[json.loads(x) for x in (d/'scores.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['status']=='error' and 'sixteen finite' in rows[0]['error']
    report=json.loads((d/'run.json').read_text());assert report['status']=='failed' and report['rows']==1
