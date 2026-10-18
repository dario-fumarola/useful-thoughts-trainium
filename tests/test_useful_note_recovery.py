"""Recovery preserves FP32 masters that differ from rounded BF16 model copies."""
import copy
from types import SimpleNamespace

import pytest
import torch

from train import useful_note_recovery as r
from weekend_thinking.artifacts import canonical_hash
from weekend_thinking.checkpoints import capture_cpu_rng


class Tied(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(41)
        self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Embedding(8,4,dtype=torch.bfloat16)
        self.lm_head=torch.nn.Linear(4,8,bias=False,dtype=torch.bfloat16)
        self.lm_head.weight=self.model.embed_tokens.weight
    def get_input_embeddings(self):return self.model.embed_tokens
    def get_output_embeddings(self):return self.lm_head


class CPUZeROContract:
    """Real AdamW on separate FP32 masters; no claim of an XLA fixture."""
    def __init__(self,model):
        self.model=model;self.master=torch.nn.Parameter(next(model.parameters()).float().clone())
        self.base_optimizer=torch.optim.AdamW([self.master],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    def update(self,step):
        self.model.zero_grad(set_to_none=True);self.base_optimizer.zero_grad(set_to_none=True)
        self.base_optimizer.param_groups[0]['lr']=step*1e-5/20
        ids=torch.tensor([step%8,(step+1)%8])
        loss=(self.model.model.embed_tokens(ids).float()-torch.arange(4).float()/10).square().mean()
        loss.backward();self.master.grad=next(self.model.parameters()).grad.float().clone()
        self.base_optimizer.step()
        with torch.no_grad():next(self.model.parameters()).copy_(self.master.bfloat16())
        return float(loss.detach())
    def state_dict(self):
        s=self.base_optimizer.state_dict()
        return dict(param_groups=s['param_groups'],base_state=s['state'],sharded_master_weights={0:self.master.detach()})
    def load_state_dict(self,state):
        state=copy.deepcopy(state)
        self.base_optimizer.load_state_dict(dict(param_groups=state['param_groups'],state=state['base_state']))
        with torch.no_grad():
            self.master.copy_(state['sharded_master_weights'][0])
            next(self.model.parameters()).copy_(self.master.bfloat16())


def fixture():
    model=Tied();opt=CPUZeROContract(model)
    for i in range(1,4):opt.update(i)
    p=dict(world_size=1,steps=4,seed=17,global_batch=32,accumulation=32,
        data_sha256='data',dataset_sha256='inventory',precision_contract={'version':'fixture'},
        runtime_policy={'fixture':'CPU'},allocator_environment={'fixture':'CPU'},purpose='qualification',
        items=64,betas=[.9,.95],eps=1e-8,weight_decay=.1)
    identity=dict(spec_hash=canonical_hash(p),order_hash='order',arm='T',seed=17,global_batch=32,accumulation=32,
        protocol_sha256='protocol-bytes',data_sha256='data',dataset_sha256='inventory',parameter_dtype='bf16',gradient_dtype='bf16',
        optimizer_dtype='fp32',compute_dtype='bf16',precision_contract=p['precision_contract'],runtime_policy=p['runtime_policy'],
        allocator_environment=p['allocator_environment'],purpose='qualification',items=64,
        counters=dict(source_records=96,source_tokens=96*528,sampled_actions=0,
            labels=dict(source=0,writer=0,reader=0,augmented=96*600,continuation=0)))
    manifest=dict(schema='neuron-v1',run_id='T-qualification',world_size=1,spec_hash=canonical_hash(p),step=3,
        files={'model.pt':{},'rank-00000.pt':{}},metadata=copy.deepcopy(identity))
    weights=dict(model=copy.deepcopy(model.state_dict()),step=3,run_id='T-qualification',spec_hash=canonical_hash(p))
    saved=dict(optimizer=copy.deepcopy(opt.state_dict()),rng=dict(cpu=capture_cpu_rng(),device=912),step=3,rank=0,world_size=1,**identity)
    kwargs=dict(protocol=p,protocol_sha256='protocol-bytes',arm='T',rank=0,run_id='T-qualification',order_hash='order',expected_lr=1e-5*3/20)
    return model,opt,manifest,weights,saved,kwargs


def test_roundtrip_actual_fp32_masters_not_rounded_bf16_model_and_next_update(tmp_path):
    model,opt,manifest,weights,saved,kwargs=fixture()
    assert (saved['optimizer']['sharded_master_weights'][0]!=weights['model']['model.embed_tokens.weight'].float()).any()
    torch.save(dict(weights=weights,saved=saved),tmp_path/'state.pt')
    payload=torch.load(tmp_path/'state.pt',weights_only=True)
    new=Tied();newopt=CPUZeROContract(new)
    assert r.validate_payload(manifest,payload['weights'],payload['saved'],new,**kwargs)==3
    events=[]
    xm=SimpleNamespace(set_rng_state=lambda value,device:events.append(('reset',value)))
    def sync(**kwargs):events.append(('sync',kwargs))
    result=r.restore_and_verify(new,newopt,payload['weights'],payload['saved'],xm=xm,device='cpu',sync=sync)
    assert result['full_optimizer_bytes_equal'] and result['full_active_model_bytes_equal']
    assert events[-1]==('reset',912) and events[-2][0]=='sync'
    assert opt.update(4)==newopt.update(4)
    assert r.optimizer_digest(opt.state_dict())==r.optimizer_digest(newopt.state_dict())
    assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(model.parameters(),new.parameters()))


@pytest.mark.parametrize('defect',['arm','rank','world','order','data','step','counter','bf16_master','nan_moment','negative_v','wrong_lr','untied_values','missing_master','missing_model','model_only','rng'])
def test_payload_rejects_mixed_or_incomplete_state_before_mutation(defect):
    model,opt,m,w,s,k=fixture();before={n:r.tensor_digest(t) for n,t in model.state_dict().items()}
    assert r.validate_payload(m,w,s,model,**k)==3
    if defect=='arm':k['arm']='W'
    elif defect=='rank':s['rank']=1
    elif defect=='world':m['world_size']=2
    elif defect=='order':s['order_hash']='other'
    elif defect=='data':s['data_sha256']='other'
    elif defect=='step':s['step']=2
    elif defect=='counter':s['counters']['source_records']=64
    elif defect=='bf16_master':s['optimizer']['sharded_master_weights'][0]=s['optimizer']['sharded_master_weights'][0].bfloat16()
    elif defect=='nan_moment':s['optimizer']['base_state'][0]['exp_avg'].flatten()[0]=float('nan')
    elif defect=='negative_v':s['optimizer']['base_state'][0]['exp_avg_sq'].flatten()[0]=-1
    elif defect=='wrong_lr':s['optimizer']['param_groups'][0]['lr']=1e-6
    elif defect=='untied_values':w['model']['lm_head.weight']=w['model']['lm_head.weight']+1
    elif defect=='missing_master':s['optimizer']['sharded_master_weights']={}
    elif defect=='missing_model':w['model'].pop('lm_head.weight')
    elif defect=='model_only':m.update(schema='useful-note-model-only-v1',files={'model.pt':{}})
    else:s.pop('rng')
    with pytest.raises(ValueError):r.validate_payload(m,w,s,model,**k)
    assert before=={n:r.tensor_digest(t) for n,t in model.state_dict().items()}


def test_supported_loader_that_loses_master_precision_is_detected():
    model,opt,m,w,s,k=fixture();new=Tied();bad=CPUZeROContract(new)
    original=bad.load_state_dict
    def rounded(state):
        state=copy.deepcopy(state)
        state['sharded_master_weights'][0]=state['sharded_master_weights'][0].bfloat16().float()
        original(state)
    bad.load_state_dict=rounded
    xm=SimpleNamespace(set_rng_state=lambda *args:pytest.fail('Must not restore RNG after failed state audit'))
    with pytest.raises(RuntimeError,match='FP32 masters'):
        r.restore_and_verify(new,bad,w,s,xm=xm,device='cpu',sync=lambda **kwargs:None)
