"""CPU contracts only: objective scaling, inventory, state and publication."""
import copy
from contextlib import nullcontext
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from train import useful_note_sft_driver as d
from weekend_thinking.artifacts import canonical_hash,sha256_file
from weekend_thinking.checkpoints import capture_cpu_rng,restore_cpu_rng


def protocol():
    from weekend_thinking.runtime_policy import runtime_policy
    policy=runtime_policy({});policy['environment'].update(PJRT_DEVICE='NEURON',XLA_COMPILATION_CACHE_SIZE='2')
    return dict(schema=d.SCHEMA,purpose='qualification',seed=17,items=64,epochs=2,steps=4,
        global_batch=32,world_size=8,accumulation=4,lr=1e-5,warmup_steps=20,betas=[.9,.95],eps=1e-8,
        weight_decay=.1,clip=1.,loss_weights=copy.deepcopy(d.WEIGHTS),parameter_dtype='fp32',optimizer_dtype='fp32',
        compute_dtype='bf16',schedule='linear_warmup_then_cosine',interface='I3',parts={'old_close':[151668,198]},
        packing=dict(source=dict(width=544,target_slots=544),writer=dict(width=640,target_slots=128),
            reader=dict(width=704,target_slots=16),augmented=dict(width=704,target_slots=704)),
        checkpoint_policy={'development_model_only':[64],'full_state':'terminal_or_STOP'},
        qualification_state_policy='small_slices',runtime_policy=policy,
        allocator_environment={'NEURON_RT_CONTIGUOUS_SCRATCHPAD':'0'},source_files={'fixed':'hash'})


def test_generic_recipe_accepts_explicit_qualifier_and_science_seed_count(monkeypatch):
    monkeypatch.setattr(d,'source_files',lambda:{'fixed':'hash'})
    p=protocol();d.validate_protocol(p)
    p.update(purpose='science',items=5120,steps=320,seed=5678,tail_policy='exact')
    d.validate_protocol(p)
    p.update(world_size=32,accumulation=1);d.validate_protocol(p)


@pytest.mark.parametrize('field,value',[
    ('items',63),('steps',2),('seed',True),('interface','I6'),('compute_dtype','fp16'),
    ('accumulation',1),('qualification_state_policy','implicit'),('optimizer_dtype','bf16')])
def test_recipe_refuses_implicit_quota_topology_or_objective_changes(monkeypatch,field,value):
    monkeypatch.setattr(d,'source_files',lambda:{'fixed':'hash'});p=protocol();p[field]=value
    with pytest.raises(ValueError):d.validate_protocol(p)


def test_science_tail_policy_and_minimum_are_explicit():
    p=protocol();p.update(purpose='science',items=5001,tail_policy='exact')
    with pytest.raises(ValueError,match='whole global'):d.planned_steps(p)
    p['tail_policy']='ceil_repeat';assert d.planned_steps(p)==313
    p['tail_policy']='drop_tail';assert d.planned_steps(p)==312
    p['items']=4999
    with pytest.raises(ValueError,match='at least5000'):d.planned_steps(p)


def test_warmup_cosine_has_positive_final_update_and_zero_next_rate():
    p=protocol();p.update(purpose='science',items=5120,steps=320,tail_policy='exact')
    assert d.learning_rate(1,p)==pytest.approx(5e-7)
    assert d.learning_rate(20,p)==p['lr']
    rates=[d.learning_rate(i,p) for i in range(20,322)]
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert d.learning_rate(320,p)>0 and d.learning_rate(321,p)==0
    assert d.learning_rate(2,protocol())==pytest.approx(1e-6)


def test_world8_accum4_and_world32_share_the_same_global_order():
    p=protocol();p['steps']=2
    from train.warmstart_driver import order_index
    order=[order_index(i,64,p['seed']) for i in range(64)]
    for step in range(2):
        eight=[d.batch_indices(step,r,p,order=order)[micro] for micro in range(4) for r in range(8)]
        thirtytwo=dict(p,world_size=32,accumulation=1)
        assert eight==[d.batch_indices(step,r,thirtytwo,order=order)[0] for r in range(32)]
        assert eight==order[step*32:(step+1)*32]
    assert sorted(order)==list(range(64))


def inventory_fixture():
    p=protocol();p['data_sha256']='data'
    rows=[dict(item_id=f'i{i}',document_id=f'd{i//2}',absolute_source_cut=128+i%2,
        split='train',teacher_note_kind='exact-postorder-v1') for i in range(64)]
    manifest=dict(schema='useful-note-positive-train-v2',status='complete_CPU_inventory_NOT_Neuron_training_qualification',
        purpose='qualification',items=64,unique_documents=32,split='train',training_sha256='data',
        ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in rows]),
        ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in rows]),
        **{k:p[k] for k in ('packing','interface','parts')})
    return p,manifest,rows


def test_inventory_allows_repeated_documents_with_unique_causal_cuts():
    p,m,rows=inventory_fixture();assert d.validate_inventory(p,m,rows)==32


@pytest.mark.parametrize('mutation',['duplicate_cut','duplicate_id','dev','wrong_order','wrong_purpose','missing_kind','bool_cut'])
def test_inventory_rejects_replacement_split_or_cut_alias(mutation):
    p,m,rows=inventory_fixture()
    if mutation=='duplicate_cut':rows[1]['absolute_source_cut']=128
    elif mutation=='duplicate_id':rows[1]['item_id']=rows[0]['item_id']
    elif mutation=='dev':rows[1]['split']='dev'
    elif mutation=='wrong_order':rows.reverse()
    elif mutation=='wrong_purpose':m['purpose']='science'
    elif mutation=='missing_kind':rows[0].pop('teacher_note_kind')
    elif mutation=='bool_cut':rows[0]['absolute_source_cut']=True
    with pytest.raises(ValueError):d.validate_inventory(p,m,rows)


class TinyBackend:
    device=torch.device('cpu');head_chunk=32;head_amp=staticmethod(nullcontext)
    def __init__(self):
        self.model=torch.nn.Module();self.model.config=SimpleNamespace(vocab_size=17)
        self.model.embedding=torch.nn.Embedding(17,6);self.model.lm_head=torch.nn.Linear(6,17,bias=False)
        self.model.lm_head.weight=self.model.embedding.weight
    def hidden(self,ids):return self.model.embedding(ids).cumsum(1)/64


def packed(row,component,p):
    shape=p['packing'][component];width=shape['width'];slots=shape['target_slots']
    count={'source':527,'writer':7,'reader':16,'augmented':681}[component]
    offset={'source':0,'writer':3,'reader':6,'augmented':9}[component]+row['offset']
    ids=((torch.arange(width)+offset)%17).reshape(1,-1).long()
    positions=(torch.arange(width-count,width) if component in ('writer','reader') else torch.arange(1,count+1)).reshape(1,-1)
    label=ids.gather(1,positions);inactive=slots-count
    note=torch.full_like(positions,component=='writer',dtype=torch.bool);future=~note
    return dict(input_ids=ids,target_positions=F.pad(positions,(0,inactive)),labels=F.pad(label,(0,inactive),value=-100),
        note_close_mask=F.pad(note,(0,inactive)),future_mask=F.pad(future,(0,inactive)))


@pytest.mark.parametrize('arm',['W','C','T'])
def test_component_accumulation_then_one_mean_reducer_matches_dense_global_objective(monkeypatch,arm):
    from train import useful_note_sft_data
    torch.manual_seed(8);initial=TinyBackend();reference=copy.deepcopy(initial);workers=[copy.deepcopy(initial) for _ in range(2)]
    p=protocol();p.update(world_size=2,accumulation=2)
    calls=[]
    def pack(row,tok,component,**kwargs):calls.append(component);return packed(row,component,p)
    monkeypatch.setattr(useful_note_sft_data,'pack_example',pack)
    for model in [reference.model]+[b.model for b in workers]:
        for parameter in model.parameters():parameter.grad=torch.full_like(parameter,.125)
    for rank,backend in enumerate(workers):
        for micro in range(2):d.backward_components({'offset':2*rank+micro},None,arm,backend,p,lambda:None)
    terms=[]
    for offset in range(4):
        for component,weight in d.WEIGHTS[arm].items():
            batch=packed({'offset':offset},component,p);active=batch['note_close_mask']|batch['future_mask']
            positions=batch['target_positions'][active]-1
            hidden=reference.hidden(batch['input_ids'][:,:-1])[0,positions]
            terms.append(weight/4*F.cross_entropy(reference.model.lm_head(hidden),batch['labels'][active]))
    sum(terms).backward()
    for index,expected in enumerate(reference.model.parameters()):
        actual=torch.stack([list(b.model.parameters())[index].grad for b in workers]).mean(0)
        torch.testing.assert_close(actual,expected.grad,atol=3e-6,rtol=3e-6)
    assert len(calls)==4*len(d.WEIGHTS[arm])


def test_signed_zero_replica_mismatch_is_not_called_bitwise_equal():
    assert d.replicas_bitwise_equal(torch.tensor([[0.,1.],[0.,1.]]))
    assert not d.replicas_bitwise_equal(torch.tensor([[0.,1.],[-0.,1.]]))


def optimizer_fixture():
    model=torch.nn.Module();model.a=torch.nn.Parameter(torch.linspace(-1,1,100));model.b=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    masters={i:p.detach().clone() for i,p in enumerate(model.parameters())}
    state=dict(sharded_master_weights=masters,base_state={i:dict(step=torch.tensor(2.),exp_avg=torch.ones_like(t)*.1,
        exp_avg_sq=torch.ones_like(t)*.2) for i,t in masters.items()})
    return model,SimpleNamespace(state_dict=lambda:state),state


def test_optimizer_checks_full_fp32_state_and_saves_only_declared_slices():
    model,opt,state=optimizer_fixture();report,_=d.inspect_optimizer(opt,2,2)
    assert report['full_master_and_moment_finite'] and report['master_numel']==103
    value=d.qualification_slices(model,opt,2,{'spec_hash':'spec'},0,1)
    assert value['resumable'] is False and value['optimizer_samples'].numel()<3*103
    assert value['optimizer_full_state_checks']['adam_step']==2


@pytest.mark.parametrize('mutation',['tail_nan_master','inf_first_moment','negative_second','step','dtype','missing'])
def test_optimizer_inspection_rejects_corruption_outside_the_saved_slice(mutation):
    _,opt,s=optimizer_fixture()
    if mutation=='tail_nan_master':s['sharded_master_weights'][0][-1]=float('nan')
    elif mutation=='inf_first_moment':s['base_state'][0]['exp_avg'][-1]=float('inf')
    elif mutation=='negative_second':s['base_state'][0]['exp_avg_sq'][-1]=-1
    elif mutation=='step':s['base_state'][0]['step']=torch.tensor(1.)
    elif mutation=='dtype':s['base_state'][0]['exp_avg_sq']=s['base_state'][0]['exp_avg_sq'].bfloat16()
    elif mutation=='missing':s['base_state'].pop(1)
    with pytest.raises(RuntimeError):d.inspect_optimizer(opt,2,2)


class AdvancingXM:
    def __init__(self):self.seed=10220601738208571245
    def get_rng_state(self,device):return self.seed
    def set_rng_state(self,seed,device):self.seed=seed
    def rendezvous(self,tag):
        self.seed=(1012031+self.seed*7012063)%2**64
        random.random();np.random.random();torch.rand(2)


def test_small_qualification_is_atomic_nonresumable_and_rng_transparent(tmp_path):
    model,opt,_=optimizer_fixture();xm=AdvancingXM();incoming=capture_cpu_rng()
    expected=(random.random(),float(np.random.random()),torch.rand(3));restore_cpu_rng(incoming)
    path=d.save_qualification_slices(tmp_path,model,opt,2,{'spec_hash':'spec'},0,1,xm,'cpu')
    manifest=json.loads((path/'COMPLETE.json').read_text());payload=torch.load(path/'rank-00000.pt',weights_only=True)
    assert manifest['resumable'] is payload['resumable'] is False and set(manifest['files'])=={'rank-00000.pt'}
    assert sha256_file(path/'rank-00000.pt')==manifest['files']['rank-00000.pt']['sha256']
    assert xm.seed==10220601738208571245
    assert random.random()==expected[0] and float(np.random.random())==expected[1] and torch.equal(torch.rand(3),expected[2])


def test_development_model_preserves_tie_and_is_explicitly_not_a_resume(tmp_path):
    model=torch.nn.Module();model.model=torch.nn.Module();model.model.embed_tokens=torch.nn.Embedding(5,3)
    model.lm_head=torch.nn.Linear(3,5,bias=False);model.lm_head.weight=model.model.embed_tokens.weight
    xm=AdvancingXM();path=d.save_development_model(tmp_path,model,64,{'spec_hash':'spec'},0,xm,'cpu')
    manifest=json.loads((path/'COMPLETE.json').read_text());saved=torch.load(path/'model.pt',weights_only=True)
    assert manifest['resumable'] is False and set(manifest['files'])=={'model.pt'}
    assert saved['model']['lm_head.weight'].data_ptr()==saved['model']['model.embed_tokens.weight'].data_ptr()
    assert xm.seed==10220601738208571245


def test_dense_proxy_counts_executed_head_and_trunk_slots_separately():
    values={'source':{'trunk_slots':9,'head_slots':3}}
    assert d.dense_matmul_proxy(values,100,4,10)==8*(60*9+40*3)
