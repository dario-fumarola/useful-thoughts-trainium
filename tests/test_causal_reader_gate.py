import math

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.causal_reader_gate import SelectiveHead, extract, gate_inputs
from train.neuron_backend import NeuronCausalBackend


def fixture_backend():
    torch.manual_seed(118)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=23,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()
    return NeuronCausalBackend(model,checkpoint_trunk=False,head_chunk=3)


def test_features_and_scores_match_independent_hf_without_vocabulary_mask():
    b=fixture_backend();contexts=[[1,2,3],[3,4,5,6,7]];targets=[list(range(1,17)),list(range(2,18))]
    actual=extract(b,contexts,targets,width=40,pad_id=0,capture_first=True)
    for index,(past,future) in enumerate(zip(contexts,targets)):
        with torch.no_grad():
            x=torch.tensor([[*past,*future]])
            hidden=b.model.model(input_ids=x,use_cache=False).last_hidden_state[0,len(past)-1:-1]
            lp=b.model(x).logits[0,len(past)-1:-1].float().log_softmax(-1)
            entropy=-(lp.exp()*lp).sum(-1)
            gold=lp.gather(-1,torch.tensor(future)[:,None])[:,0]
        assert torch.allclose(actual.features.hidden[index],hidden,atol=1e-6)
        assert torch.allclose(actual.features.entropy[index],entropy,atol=1e-6)
        assert torch.allclose(actual.target_logps[index],gold,atol=1e-6)
        assert torch.allclose(actual.first_logps[index],lp[0],atol=1e-6)


def test_current_future_perturbation_cannot_change_gate_features():
    b=fixture_backend();past=[[1,2,3]];future=[list(range(1,17))]
    original=extract(b,past,future,width=40,pad_id=0)
    for position in range(16):
        changed=[[t if i<position else (t+3)%23 for i,t in enumerate(future[0])]]
        alternative=extract(b,past,changed,width=40,pad_id=0)
        assert torch.equal(original.features.hidden[:,:position+1],alternative.features.hidden[:,:position+1])
        assert torch.equal(original.features.entropy[:,:position+1],alternative.features.entropy[:,:position+1])
        assert original.target_logps[0,position]!=alternative.target_logps[0,position]
    padded=extract(b,past,future,width=48,pad_id=0)
    assert torch.allclose(original.features.hidden,padded.features.hidden,atol=1e-6)


def test_head_backward_cannot_update_reader_or_take_gold_scores_as_inputs():
    b=fixture_backend();target=[list(range(1,17))]
    raw=extract(b,[[1,2,3]],target,width=40,pad_id=0)
    assisted=extract(b,[[1,2,3,8,9]],target,width=40,pad_id=0)
    features=gate_inputs(raw.features,assisted.features,note_actions=[2],capped=[False],vocabulary_size=23)
    assert features.shape==(1,16,37) and not features.requires_grad
    head=SelectiveHead(37)
    loss=head.loss(features,raw.target_logps,assisted.target_logps)
    loss.backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert all(p.grad is None for p in b.model.parameters())


def test_closure_and_saturation_have_consistent_values_and_finite_gradients():
    head=SelectiveHead(1);boundary=torch.tensor(head.closing_logit,dtype=torch.float64)
    logits=torch.stack((boundary,torch.nextafter(boundary,torch.tensor(math.inf)),
        boundary-1.,torch.tensor(-1000.),torch.tensor(1000.),torch.tensor(0.))).double().requires_grad_()
    raw=torch.full_like(logits,-1.);assisted=torch.full_like(logits,-5.)
    mixed=head.target_logps(logits,raw,assisted);g=head.gate(logits)
    assert g[0]==0 and g[1]>0 and g[2]==0 and g[3]==0 and g[4]==1
    assert torch.equal(mixed[[0,2,3]],raw[[0,2,3]])
    mixed.sum().backward()
    assert torch.isfinite(mixed).all() and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[[0,2,3]],torch.zeros(3,dtype=torch.float64))
    with torch.no_grad():head.bias.fill_(-1000)
    head.loss(torch.zeros(2,16,1),torch.full((2,16),-1.),torch.full((2,16),-5.)).backward()
    assert head.weight.grad.eq(0).all() and head.bias.grad==0


def test_full_distribution_normalization_exact_bypass_and_target_consistency():
    head=SelectiveHead(1);torch.manual_seed(2)
    raw=torch.randn(2,16,23,dtype=torch.float64).log_softmax(-1)
    assisted=torch.randn_like(raw).log_softmax(-1)
    assert head.predict_logps(None,raw,enabled=False) is raw
    closed=torch.full((2,16),head.closing_logit,dtype=torch.float64)
    assert head.predict_logps(closed,raw,assisted) is raw
    logits=closed.clone();logits[0]=0;logits[1,8:]=1000
    mixed=head.predict_logps(logits,raw,assisted)
    assert torch.allclose(mixed.exp().sum(-1),torch.ones(2,16,dtype=torch.float64),atol=1e-12)
    assert torch.equal(mixed[1,:8],raw[1,:8])
    labels=torch.arange(16)[None,:,None].expand(2,-1,-1)
    lp0=raw.gather(-1,labels)[:,:,0];lp1=assisted.gather(-1,labels)[:,:,0]
    assert torch.allclose(mixed.gather(-1,labels)[:,:,0],head.target_logps(logits,lp0,lp1),atol=1e-12)


def fit_toy(features,raw,assisted):
    head=SelectiveHead(1)
    optimizer=torch.optim.AdamW([{'params':[head.weight],'weight_decay':.0001},
                                {'params':[head.bias],'weight_decay':0}],
                               lr=.01,betas=(.9,.999),eps=1e-8)
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        head.loss(features,raw,assisted).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        optimizer.step()
    return head


def test_hindsight_coin_advantage_does_not_become_a_causal_gain():
    # Identical causal inputs, balanced unpredictable outcomes. The second
    # predictor's 90/10 bias gives a positive oracle bound but no usable signal.
    features=torch.zeros(2,16,1,dtype=torch.float64)
    raw=torch.full((2,16),math.log(.5),dtype=torch.float64)
    assisted=torch.tensor([math.log(.9),math.log(.1)])[:,None].expand(2,16).double()
    head=fit_toy(features,raw,assisted)
    assert .5*math.log(.9/.5)>0
    assert head.loss(features,raw,assisted)>=-raw.mean()-1e-12
    assert head.gate(head(features)).max()<=1e-8


def test_same_fixed_head_recipe_can_learn_a_separable_causal_signal():
    # Now a prospective feature distinguishes when the extra predictor helps.
    features=torch.tensor([1.,-1.])[:,None,None].expand(2,16,1).double()
    raw=torch.full((2,16),math.log(.5),dtype=torch.float64)
    assisted=torch.tensor([math.log(.9),math.log(.1)])[:,None].expand(2,16).double()
    head=fit_toy(features,raw,assisted)
    gain=raw.neg().mean()-head.loss(features,raw,assisted)
    assert gain>.2
    assert head.gate(head(features))[1].eq(0).all()
