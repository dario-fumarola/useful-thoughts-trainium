"""Synthetic held-note barriers, causal contexts, identities and denominators.

No real dataset, model/checkpoint payload, accelerator or outcome is read.
"""
import copy
import json
from pathlib import Path

import pytest

from train import useful_note_evaluation as e
from weekend_thinking.artifacts import canonical_hash,sha256_file


class ByteTokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


def test_prefix_projection_is_variable_length_and_independent_of_future_labels():
    row=dict(item_id='a',document_id='d',split='dev',prefix_ids=[2,3],reanchor_ids=[3],
             target_ids=[8],natural_future16_ids=[8]*16,numeric_value='8')
    other=copy.deepcopy(row);other.update(target_ids=[999],natural_future16_ids=[999]*16,numeric_value='999')
    assert e.prefix_row(row)==e.prefix_row(other)
    assert set(e.prefix_row(row))=={'item_id','document_id','split','prefix_ids','reanchor_ids'}
    long=dict(row,prefix_ids=[2]*512);assert len(e.prefix_row(long)['prefix_ids'])==512


def test_I3_contexts_keep_raw_unchanged_and_teacher_distinct_from_target_copy():
    row=dict(item_id='a',prefix_ids=[7,8],reanchor_ids=[8],target_ids=[5,10],
             boundary={'expression':'2+3'},numeric_value='5',numeric_surface_text='5')
    p=dict(parts={'old_open':[100],'old_close':[151668,198]})
    contexts=e.contexts(row,p,{'a':{'content_ids':[77]}},[88,89],ByteTokenizer())
    assert contexts['raw']==[7,8]
    assert contexts['empty']==[7,8,100,151668,198,8]
    assert contexts['own']==[7,8,100,77,151668,198,8]
    assert contexts['other']==[7,8,100,88,89,151668,198,8]
    assert contexts['target_copy']==[7,8,100,5,10,151668,198,8]
    assert contexts['symbolic_teacher']!=contexts['target_copy']
    assert bytes(contexts['symbolic_teacher'][3:-3]).decode()=='Compute: 2 + 3 = 5. Result: 5.'


def test_global_other_is_intact_permutation_with_actual_empty_donor():
    p=dict(ordered_item_ids=['a','b','c'])
    notes={'a':{'content_ids':[1,2,3]},'b':{'content_ids':[]},'c':{'content_ids':[9]}}
    v=e.other_controls(p,notes)
    assert v['other_donors']=={'a':'b','b':'c','c':'a'}
    assert v['other_content_ids']=={'a':[],'b':[9],'c':[1,2,3]}
    assert sorted(map(len,v['other_content_ids'].values()))==sorted(len(n['content_ids']) for n in notes.values())


def test_incomplete_notes_block_future_source_and_model_before_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'check',lambda _:None)
    monkeypatch.setattr(e,'load',lambda _:({},[]))
    def incomplete(*args):raise ValueError('All256 complete attempted notes required')
    monkeypatch.setattr(e,'complete_notes',incomplete)
    monkeypatch.setattr(e,'selected_rows',lambda _:pytest.fail('Future source touched before notes close'))
    monkeypatch.setattr(e,'load_model',lambda *args:pytest.fail('Model allocated before notes close'))
    with pytest.raises(ValueError,match='All256'):e.score(tmp_path,0,'B0','unused')
    assert not (tmp_path/'scores').exists()


def note_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'N',4);monkeypatch.setattr(e,'SHARDS',1)
    (tmp_path/'protocol.json').write_text('{}')
    p=dict(writer={'id':'B0','checkpoint':None,'checkpoint_sha256':None},vocab_size=151936,banned_ids=[999],
           parts={'old_open':[100],'old_close':[151668,198]})
    prefixes=[dict(item_id=f'i{i}',document_id=f'd{i}',prefix_ids=[1]*(i+1),reanchor_ids=[1]) for i in range(4)]
    notes=[]
    for i,q in enumerate(prefixes):
        actions=[2,151668] if i<3 else [151643]
        content=actions[:-1]
        notes.append(dict(item_id=q['item_id'],document_id=q['document_id'],status='ok',actions=actions,content_ids=content,
            prefix_sha256=canonical_hash(q['prefix_ids']),writer_prompt_sha256=canonical_hash(q['prefix_ids']+[100]),
            n_actions=len(actions),n_content_tokens=len(content),stop_type=f'token:{actions[-1]}'))
    d=tmp_path/'notes/shard-0';d.mkdir(parents=True)
    def write():
        (d/'notes.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in notes))
        (d/'run.json').write_text(json.dumps(dict(status='complete_attempts',rows=4,writer=p['writer'],
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),notes_sha256=sha256_file(d/'notes.jsonl'))))
    write();return p,prefixes,notes,write


def test_complete_note_loader_keeps_immediate_EOS_empty_and_counts_every_path(tmp_path,monkeypatch):
    p,prefixes,notes,write=note_fixture(tmp_path,monkeypatch)
    actual,_,_=e.complete_notes(tmp_path,p,prefixes)
    assert len(actual)==4 and actual['i3']['content_ids']==[]
    notes[0].update(status='error',actions=[2],content_ids=[2],n_actions=1,n_content_tokens=1,stop_type='error')
    write();assert e.complete_notes(tmp_path,p,prefixes)[0]['i0']['status']=='error'


def test_cap_is128_total_actions_and_capped_content_is_retained(tmp_path,monkeypatch):
    p,prefixes,notes,write=note_fixture(tmp_path,monkeypatch)
    assert e.CAP==128 and 544+e.CAP==672
    notes[0].update(actions=[2]*128,content_ids=[2]*128,n_actions=128,n_content_tokens=128,stop_type='length_cap')
    write();assert len(e.complete_notes(tmp_path,p,prefixes)[0]['i0']['content_ids'])==128


@pytest.mark.parametrize('defect',['gap','undefined','changed_prompt','early_stop','writer'])
def test_note_loader_rejects_missing_or_corrupt_actual_paths(tmp_path,monkeypatch,defect):
    p,prefixes,notes,write=note_fixture(tmp_path,monkeypatch)
    if defect=='gap':notes.pop()
    elif defect=='undefined':notes[0].update(actions=[999,151668],content_ids=[999])
    elif defect=='changed_prompt':notes[0]['writer_prompt_sha256']='wrong'
    elif defect=='early_stop':notes[0].update(actions=[151668,2,151668],content_ids=[151668,2],n_actions=3,n_content_tokens=2)
    elif defect=='writer':p['writer']['id']='W'
    write()
    # Writer mutation must disagree with the fixed protocol expectation.
    if defect=='writer':p['writer']['id']='B0'
    with pytest.raises(ValueError):e.complete_notes(tmp_path,p,prefixes)


def test_three_metrics_use_same_full16_vector_and_actual_boundary_prefix():
    row=dict(target_ids=[7,8,9],natural_future16_ids=list(range(7,23)),numeric_token_mask=[True,True,False])
    lp=[-float(i) for i in range(1,17)]
    assert e.metric_values(lp,row)==dict(natural16_ll=-136.,scalar_boundary_ll=-6.,numeric_only_ll=-3.)
    with pytest.raises(ValueError):e.metric_values(lp[:15],row)
    row['target_ids']=[999]
    with pytest.raises(ValueError):e.metric_values(lp,row)


def score_fixture():
    ids=['a','b','c'];docs=['D1','D1','D2'];values=[]
    for i,(key,doc) in enumerate(zip(ids,docs)):
        for mode in e.MODES:
            z=float(i+(mode=='own'))
            values.append(dict(item_id=key,document_id=doc,path='I3',mode=mode,status='ok',**{k:z for k in e.METRICS}))
    return ids,docs,values


def test_document_pairing_and_failures_never_turn_into_complete_case_primary():
    ids,docs,values=score_fixture();report=e.reduce_rows(values,ids,docs,['I3'])['I3']
    contrast=report['contrasts']['own_minus_raw/natural16_ll']
    assert contrast['difference_b_minus_a']==1 and contrast['ci95']==[1,1]
    assert contrast['items']==3 and contrast['documents']==2
    failed=next(v for v in values if v['item_id']=='b' and v['mode']=='own');failed.clear()
    failed.update(item_id='b',document_id='D1',path='I3',mode='own',status='error')
    report=e.reduce_rows(values,ids,docs,['I3'])['I3']
    assert report['absolute']['own/natural16_ll']==dict(items=3,errors=1,mean=None)
    assert report['contrasts']['own_minus_raw/natural16_ll']['available'] is False
    assert report['absolute']['raw/natural16_ll']['items']==3


@pytest.mark.parametrize('defect',['missing','duplicate','unknown_status','wrong_doc'])
def test_metric_matrix_requires_exact_complete_keys(tmp_path,defect):
    ids,docs,values=score_fixture()
    if defect=='missing':values.pop()
    elif defect=='duplicate':values[-1]=copy.deepcopy(values[0])
    elif defect=='unknown_status':values[0]['status']='skipped'
    elif defect=='wrong_doc':values[0]['document_id']='different'
    with pytest.raises(ValueError):e.reduce_rows(values,ids,docs,['I3'])


def checkpoint_fixture(tmp_path,monkeypatch,*,items=5120,steps=320):
    from train import useful_note_sft_driver as driver
    from weekend_thinking.runtime_policy import runtime_policy
    run=tmp_path/'W-run';d=run/f'step-{steps:07d}';d.mkdir(parents=True)
    source_map={'train/useful_note_sft_driver.py':sha256_file(e.ROOT/'train/useful_note_sft_driver.py')}
    monkeypatch.setattr(driver,'source_files',lambda:source_map)
    policy=runtime_policy({});policy['environment'].update(PJRT_DEVICE='NEURON',XLA_COMPILATION_CACHE_SIZE='2')
    p=dict(schema='useful-note-sft-v1',purpose='science',items=items,epochs=2,steps=steps,tail_policy='exact',world_size=8,accumulation=4,
        global_batch=32,lr=1e-5,warmup_steps=20,betas=[.9,.95],eps=1e-8,weight_decay=.1,clip=1.,loss_weights=copy.deepcopy(driver.WEIGHTS),
        parameter_dtype='fp32',optimizer_dtype='fp32',compute_dtype='bf16',schedule='linear_warmup_then_cosine',interface='I3',
        packing=dict(source=dict(width=544,target_slots=544),writer=dict(width=640,target_slots=128),reader=dict(width=704,target_slots=16),augmented=dict(width=704,target_slots=704)),
        checkpoint_policy={'development_model_only':[64],'full_state':'terminal_or_STOP'},runtime_policy=policy,
        allocator_environment={'NEURON_RT_CONTIGUOUS_SCRATCHPAD':'0'},source_files=source_map,seed=17)
    protocol=tmp_path/'training-protocol.json';protocol.write_text(json.dumps(p))
    spec=canonical_hash(p);ph=sha256_file(protocol);counts={'source_records':steps*32}
    (run/'run.json').write_text(json.dumps(dict(arm='W',qualification=False,purpose='science',spec_hash=spec,protocol_sha256=ph,scientific_steps=steps,order_hash='order')))
    (run/'completion.json').write_text(json.dumps(dict(status='complete',actual_steps=steps,planned_steps=steps,protocol_sha256=ph,counters=counts)))
    for name in ['model.pt']+[f'rank-{i:05d}.pt' for i in range(8)]:(d/name).write_bytes(name.encode())
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha256_file(f)) for f in d.iterdir()}
    complete=d/'COMPLETE.json'
    complete.write_text(json.dumps(dict(schema='neuron-v1',run_id=run.name,step=steps,spec_hash=spec,world_size=8,files=files,
        metadata=dict(arm='W',qualification=False,parameter_dtype='fp32',order_hash='order',counters=counts))))
    return d/'model.pt',sha256_file(d/'model.pt'),sha256_file(complete),protocol


def test_checkpoint_identity_hashes_all_master_shards_without_loading_tensors(tmp_path,monkeypatch):
    args=checkpoint_fixture(tmp_path,monkeypatch);identity=e.trained_identity(*args)
    assert identity['id']=='W' and identity['step']==320 and len(identity['checkpoint_files'])==9
    (args[0].parent/'rank-00007.pt').write_bytes(b'corrupt master')
    with pytest.raises(ValueError,match='master/moment integrity'):e.trained_identity(*args)


def test_forged_short_science_identity_fails_actual_driver_recipe_validation(tmp_path,monkeypatch):
    args=checkpoint_fixture(tmp_path,monkeypatch,items=64,steps=2)
    with pytest.raises(ValueError,match='at least5000'):e.trained_identity(*args)


def test_trained_reader_is_never_compared_against_base_native_logits():
    assert 'not compared with B0' in e.base_gate(None,{},0,Path('/unused'),{'id':'W'})['scope']


def test_saved_costs_count_shared_paths_once_and_include_partial_attempt_actions():
    notes={'a':dict(actions=[1,151668],content_ids=[1],wall_seconds=2.),'b':dict(actions=[3],content_ids=[3],wall_seconds=4.)}
    values=[dict(timing_group='a/raw',wall_seconds=5.,shared_raw_forward_seconds=3.,shared_by_paths=['I3','I6w0.375'],path=p) for p in ('I3','I6w0.375')]
    costs=e.saved_costs(notes,values)
    assert costs['total_generated_actions']==3 and costs['total_content_tokens']==2
    assert costs['mean_actions_per_attempt']==1.5 and costs['mean_content_tokens_per_attempt']==1
    assert costs['note_generation_seconds_sum']==6 and costs['scoring_cell_seconds_sum']==5 and costs['raw_full_logits_seconds_sum']==3
    values[-1]['wall_seconds']=0.
    with pytest.raises(ValueError,match='Shared timing'):e.saved_costs(notes,values)


def test_actual_full_logits_mixer_score_path_preserves_axes_and_shared_raw_cost(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from train.interface_mixer_probe import full_logits
    class TinyBackend:
        device=torch.device('cpu')
        def __init__(self):
            torch.manual_seed(3);self.model=SimpleNamespace(lm_head=torch.nn.Linear(3,32,bias=False))
        def hidden(self,ids):
            x=ids.float().remainder(13)/100
            return torch.stack((x,x*x,x/3),-1).cumsum(1)
        def sync(self):pass
    backend=TinyBackend();target=list(range(16));past=[7,8,9]
    actual=full_logits(backend,past,target)
    expected=backend.model.lm_head(backend.hidden(torch.tensor([past+target]))[:,len(past)-1:len(past)+15])[0]
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)
    monkeypatch.setattr(e,'SHARDS',1);monkeypatch.setattr(e,'N',2);monkeypatch.setattr(e,'check',lambda _:None)
    (tmp_path/'protocol.json').write_text('{}');held=tmp_path/'held.json';held.write_text('{}')
    identity={'id':'B0','checkpoint':None,'checkpoint_sha256':None}
    p=dict(writer=identity,ordered_item_ids=['a','b'],held_file=str(held),deadline_utc='deadline',parts={'old_open':[100],'old_close':[151668,198]})
    data=[dict(item_id=key,document_id='d'+key,split='dev',prefix_ids=[7]*n,reanchor_ids=[7],target_ids=target[:2],natural_future16_ids=target,
        numeric_token_mask=[True,False],boundary={'expression':'2+3'},numeric_value='5',numeric_surface_text='5') for key,n in [('a',2),('b',7)]]
    prefixes=[e.prefix_row(r) for r in data];notes={key:dict(status='ok',content_ids=[2,3],actions=[2,3,151668],wall_seconds=.1) for key in ('a','b')}
    monkeypatch.setattr(e,'load',lambda _:(p,prefixes));monkeypatch.setattr(e,'complete_notes',lambda *_:(notes,{},identity))
    monkeypatch.setattr(e,'selected_rows',lambda _:(data,None));monkeypatch.setattr(e,'load_model',lambda *args:(backend,ByteTokenizer(),identity));monkeypatch.setattr(e,'base_gate',lambda *args:{'scope':'synthetic'})
    control=dict(status='complete',note_pins={},writer=identity,protocol_sha256=sha256_file(tmp_path/'protocol.json'),**e.other_controls(p,notes))
    (tmp_path/'controls.json').write_text(json.dumps(control))
    e.score(tmp_path,0,'B0','deadline',device='cpu',with_mixer=True)
    d=tmp_path/'scores/B0/with-mixer/shard-0';values=e.rows(d/'scores.jsonl');run=e.read(d/'run.json')
    assert run['status']=='complete_attempts' and len(values)==24 and all(v['status']=='ok' for v in values)
    raw=[v for v in values if v['mode']=='raw']
    for key in ('a','b'):
        pair=[v for v in raw if v['item_id']==key]
        assert pair[0]['token_logps']==pair[1]['token_logps']
        assert pair[0]['wall_seconds']==pair[1]['wall_seconds']>0
        assert pair[0]['wall_seconds']>=pair[0]['shared_raw_forward_seconds']>0
        row=next(r for r in data if r['item_id']==key)
        raw_logits=full_logits(backend,row['prefix_ids'],target)
        own_context=e.contexts(row,p,notes,e.other_controls(p,notes)['other_content_ids'][key],ByteTokenizer())['own']
        note_logits=full_logits(backend,own_context,target)
        for path,logits in [('I3',note_logits),('I6w0.375',raw_logits+.375*(note_logits-raw_logits))]:
            expected=logits.log_softmax(-1).gather(-1,torch.tensor(target)[:,None])[:,0]
            saved=next(v for v in values if (v['item_id'],v['mode'],v['path'])==(key,'own',path))
            torch.testing.assert_close(torch.tensor(saved['token_logps']),expected,atol=2e-7,rtol=1e-6)
    costs=e.saved_costs(notes,values)
    assert costs['scoring_shared_timing_groups']==12
    assert costs['raw_full_logits_seconds_sum']==run['shared_raw_forward_seconds']
