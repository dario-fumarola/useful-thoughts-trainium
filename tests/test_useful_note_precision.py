"""CPU precision/state contracts, without importing or allocating XLA."""
import copy
import json
import struct
from types import SimpleNamespace

import pytest
import torch

from train import useful_note_precision as p
from weekend_thinking.artifacts import sha256_file


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(5,3);self.lm_head=torch.nn.Linear(3,5,bias=False)
        self.lm_head.weight=self.emb.weight
        self.register_buffer('inv_freq',torch.tensor([.10000000149,.00316227766],dtype=torch.float32),persistent=False)


def test_parameter_only_cast_preserves_tie_and_full_precision_rotary_buffer():
    m=TiedModel();m.emb.weight.data=m.emb.weight.data.bfloat16().float();identity=id(m.emb.weight)
    old=m.inv_freq.clone();old_weights=m.emb.weight.clone();report=p.cast_parameters(m)
    assert m.emb.weight.dtype==torch.bfloat16 and m.emb.weight is m.lm_head.weight and id(m.emb.weight)==identity
    assert torch.equal(m.emb.weight.float(),old_weights)
    assert m.inv_freq.dtype==torch.float32 and torch.equal(m.inv_freq.view(torch.uint8),old.view(torch.uint8))
    assert not torch.equal(m.inv_freq,m.inv_freq.bfloat16().float())
    assert report['parameter_objects_preserved']


def test_cast_refuses_already_cast_or_noncanonical_buffers():
    m=TiedModel();p.cast_parameters(m)
    with pytest.raises(ValueError):p.cast_parameters(m)
    m=TiedModel();m.inv_freq=m.inv_freq.bfloat16()
    with pytest.raises(ValueError):p.cast_parameters(m)


def test_released_header_requires_every_tensor_bf16(tmp_path):
    def write(dtype):
        header=json.dumps({'weight':{'dtype':dtype,'shape':[5,3],'data_offsets':[0,30]}}).encode()
        (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header)
    write('BF16');r=p.released_bf16_header(tmp_path);assert r['tensors']==1 and r['parameters']==15
    write('F32')
    with pytest.raises(ValueError,match='all released weights BF16'):p.released_bf16_header(tmp_path)


def test_full_head_kl_normalizes_independently_and_checks_each_reference():
    r=torch.tensor([[2.,-1.,0.],[2.,-1.,0.]],dtype=torch.float64)
    a=r.clone();a[0]+=1000;a[1]=torch.tensor([-2.,5.,0.])
    rows=p.compare_distributions(r,a)
    assert rows[0]['passed'] and rows[0]['max_bidirectional_kl_nats']<1e-12
    assert not rows[1]['passed'] and rows[1]['max_bidirectional_kl_nats']>.01
    rr=r-r.logsumexp(-1,keepdim=True);aa=a-a.logsumexp(-1,keepdim=True)
    assert rows[1]['reference_to_device_kl_nats']==pytest.approx(float((rr[1].exp()*(rr[1]-aa[1])).sum()))
    assert rows[1]['device_to_reference_kl_nats']==pytest.approx(float((aa[1].exp()*(aa[1]-rr[1])).sum()))


def test_tail_logp_maximum_is_diagnostic_and_all_vocab_rows_enter_kl():
    r=torch.tensor([[0.,-30.,-30.]])
    a=torch.tensor([[0.,-90.,-30.]])
    row=p.compare_distributions(r,a)[0]
    assert row['passed'] and row['max_logp_error']>50
    # The last row could be undefined by a tokenizer; it is still in the full head.
    bad=a.clone();bad[0,-1]=10
    assert not p.compare_distributions(r,bad)[0]['passed']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_full_head_rejects_nonfinite_values(value):
    with pytest.raises(RuntimeError):p.compare_distributions(torch.zeros(1,3),torch.tensor([[0.,value,1.]]))


def test_standard_shards_preserve_initial_fp32_values_padding_and_master_rounding():
    m=TiedModel();m.emb.weight.data=1+torch.arange(15).reshape(5,3).float()/8;p.cast_parameters(m)
    for rank in range(4):
        expected=p.shard_expectations(m,rank,4);padded=torch.nn.functional.pad(m.emb.weight.detach(),(0,0,0,3))
        state={'sharded_master_weights':{0:padded.chunk(4)[rank].float().clone()}}
        opt=SimpleNamespace(state_dict=lambda:state)
        samples=p.master_samples(opt,expected)
        assert torch.equal(samples,expected['values'])
        assert torch.equal(p.active_shard_samples(m,expected),samples.bfloat16())
        if rank==0:
            state['sharded_master_weights'][0].add_(1e-7)
            after=p.master_samples(opt,expected)
            assert (after-samples).ne(0).any() and torch.equal(after.bfloat16(),samples.bfloat16())
            assert state['sharded_master_weights'][0].dtype==torch.float32
    expected=p.shard_expectations(m,0,4)
    bad=SimpleNamespace(state_dict=lambda:{'sharded_master_weights':{0:torch.ones(2,3,dtype=torch.bfloat16)}})
    with pytest.raises(RuntimeError,match='FP32 master'):p.master_samples(bad,expected)


def test_cpu_reference_uses_canonical_fp32_parameters_full_head_and_receipt(monkeypatch,tmp_path):
    from train import evaluate,neuron_backend
    from weekend_thinking import pretrained
    m=TiedModel();m.emb.weight.data=m.emb.weight.data.bfloat16().float()
    m.get_input_embeddings=lambda:m.emb;m.get_output_embeddings=lambda:m.lm_head
    monkeypatch.setattr(pretrained,'load_base',lambda path:(m,None))
    monkeypatch.setattr(evaluate,'asset_provenance',lambda path: {'model_id':'tiny'} if path.is_dir() else None)
    monkeypatch.setattr(p,'released_bf16_header',lambda path:{'dtype':'BF16','tensors':1,'parameters':15})
    class Backend:
        def __init__(self,model,**kwargs):self.model=model
        def hidden(self,ids):return self.model.emb(ids%5)
    monkeypatch.setattr(neuron_backend,'NeuronCausalBackend',Backend)
    contexts=tmp_path/'contexts.json';contexts.write_text(json.dumps({'qualification_refs':[{'context_ids':[1,2+i]} for i in range(4)]}))
    report=p.cpu_reference(contexts,tmp_path,tmp_path/'reference','2099-01-01T00:00:00+00:00',1)
    values=torch.load(tmp_path/'reference/next-logps.pt',weights_only=True)
    assert values.shape==(4,5) and values.dtype==torch.float32 and torch.isfinite(values).all()
    assert report['precision']==p.contract() and report['conversion']['parameter_dtype']=='torch.float32' and report['next_logps_sha256']==sha256_file(tmp_path/'reference/next-logps.pt')
    assert all(row=={'hidden':'torch.float32','head':'torch.float32','logps':'torch.float32'} for row in report['dtypes'])
    assert report['conversion']['buffers']['inv_freq']['dtype']=='torch.float32'
