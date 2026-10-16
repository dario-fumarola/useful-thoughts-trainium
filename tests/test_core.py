import itertools
import pytest
import torch
from weekend_thinking.core import *
from weekend_thinking.reference import TinyCausalLM, UpdateConfig, generate, replay_log_probs, thought_update, vanilla_update

@pytest.fixture
def fmt(): return Format((1,), (2,), 3, 0)


def test_last_real_token_is_supervised_from_extra_source_token():
    b = causal_batch([[5, 6, 7, 8]], [1], pad_id=0, total_length=7)
    assert b.labels[0, :3].tolist() == [6, 7, 8]
    assert b.loss_mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert b.input_ids[0, :4].tolist() == [5, 6, 7, 8]


@pytest.mark.parametrize('actions,expected', [([4,3,0,0],[1,1,0,0]), ([3,0,0,0],[1,0,0,0]), ([4,5,6,7],[1,1,1,1])])
def test_stop_and_cap(actions, expected):
    assert action_mask(torch.tensor(actions), 3).int().tolist() == expected


def test_scorer_predicts_true_horizon_only(fmt):
    r = Record('x','doc','test',(5,6,7),(8,9,10))
    b = scoring_batch([r], [[20,21]], fmt, 2)
    supervised = b.labels[b.loss_mask.bool()].tolist()
    assert supervised == [8,9]
    assert b.input_ids[0].tolist() == [5,6,7,1,20,21,2,8,9]


def test_empty_format_not_raw_silence(fmt):
    r = Record('x','doc','test',(5,6),(8,))
    raw = scoring_batch([r],[None],fmt,1)
    empty = scoring_batch([r],[[]],fmt,1)
    assert raw.input_ids.shape[1] + 2 == empty.input_ids.shape[1]


def test_changing_future_cannot_change_writer_input_or_samples(fmt):
    a = Record('x','doc','test',(5,6,7),(8,9))
    b = Record('y','doc2','test',a.prefix,(20,21))
    assert writer_inputs([a.prefix],fmt) == writer_inputs([b.prefix],fmt)
    torch.manual_seed(5); model = TinyCausalLM()
    cfg = UpdateConfig(max_thought_tokens=6)
    torch.manual_seed(7); x,_ = generate(model,[list(a.prefix)],fmt,cfg)
    torch.manual_seed(7); y,_ = generate(model,[list(b.prefix)],fmt,cfg)
    assert torch.equal(x,y)


def test_loo_cancels_common_baseline_and_detaches():
    r = torch.tensor([[-9.,-8.,-7.,-6.]], requires_grad=True)
    a = loo_advantages(r)
    assert torch.allclose(a, loo_advantages(r-125.))
    assert not a.requires_grad and a[0,-1] > 0 and r[0,-1] < 0
    assert torch.allclose(a.sum(-1), torch.tensor([0.]))


def test_loo_exact_enumeration_matches_true_bandit_gradient():
    # Two iid Bernoulli thought samples. Helpful action reward=1, bad=0.
    theta = torch.tensor(0.3, requires_grad=True)
    p = theta.sigmoid(); expected_loss = 0.0
    for a,b in itertools.product((0,1),repeat=2):
        actions = torch.tensor([a,b], dtype=torch.float32)
        logps = actions*torch.log(p)+(1-actions)*torch.log(1-p)
        weight = ((p if a else 1-p)*(p if b else 1-p)).detach()
        adv = loo_advantages(actions)
        expected_loss = expected_loss - weight*(logps*adv).mean()
    g = torch.autograd.grad(expected_loss,theta)[0]
    assert torch.allclose(g, -p.detach()*(1-p.detach()), atol=1e-6)


def test_masked_tail_has_no_gradient():
    lp = torch.zeros(1,2,4,requires_grad=True)
    mask = torch.tensor([[[1,1,0,0],[1,1,1,1]]],dtype=torch.bool)
    policy_loss(lp,mask,torch.tensor([[1.,-1.]])).backward()
    assert lp.grad[0,0,2:].abs().sum() == 0
    assert lp.grad[0,0,0] < 0 and lp.grad[0,1,0] > 0


def test_length_normalizations_are_named_and_different():
    lp = torch.tensor([[[-2.,-2.,-2.,-2.],[-3.,-3.,-3.,-3.]]])
    mask = torch.tensor([[[1,0,0,0],[1,1,1,1]]],dtype=torch.bool)
    adv = torch.tensor([[1.,-1.]])
    assert policy_loss(lp,mask,adv,normalization='sequence') != policy_loss(lp,mask,adv,normalization='fixed_cap')
    with pytest.raises(ValueError): policy_loss(lp,mask,adv,normalization='mystery')


def test_fresh_ppo_equals_reinforce_gradient():
    a = torch.randn(2,4,7,requires_grad=True)
    mask = torch.ones_like(a,dtype=torch.bool); adv = torch.randn(2,4)
    ga = torch.autograd.grad(policy_loss(a,mask,adv),a,retain_graph=True)[0]
    gb = torch.autograd.grad(policy_loss(a,mask,adv,old_lp=a.detach()),a)[0]
    assert torch.allclose(ga,gb)


def test_sampling_mask_and_temperature_agree():
    torch.manual_seed(2); logits = torch.randn(32,11)
    token,lp = sample_actions(logits,0.7,(0,5))
    expected = policy_log_probs(logits,0.7,(0,5)).gather(-1,token[:,None])[:,0]
    assert torch.allclose(lp,expected)
    assert not torch.isin(token,torch.tensor([0,5])).any()


@pytest.mark.parametrize('bans', [tuple(range(4)), (-1,), (4,)])
def test_bad_vocab_masks_rejected(bans):
    with pytest.raises(ValueError): policy_log_probs(torch.zeros(4),0.7,bans)


def test_controls_match_length_and_have_no_answers():
    notes = [[5,6,7,8],[10,11],[]]
    controls = matched_controls(notes,filler_ids=[20,21],seed=4)
    for value in controls.values(): assert list(map(len,value)) == list(map(len,notes))
    assert sorted(controls['shuffled'][0]) == sorted(notes[0])
    assert controls['other'][0] == [10,11,10,11]


def test_behavior_replay_equivalence(fmt):
    torch.manual_seed(11); model=TinyCausalLM()
    cfg=UpdateConfig(max_thought_tokens=9, banned_ids=(0,1))
    prefixes=[[6,7,8],[9,10,11]]
    actions,old=generate(model,prefixes,fmt,cfg)
    new=replay_log_probs(model,prefixes,actions,fmt,cfg)
    mask=action_mask(actions,fmt.stop_id)
    assert torch.allclose(old[mask],new[mask],atol=1e-6)


def test_both_actual_updates_change_weights_and_remain_finite(fmt):
    torch.manual_seed(9)
    rows=[Record(str(i),str(i),'train',(5+i,6,7),(9+i,10)) for i in range(2)]
    for update in (thought_update,vanilla_update):
        model=TinyCausalLM(); before=[x.detach().clone() for x in model.parameters()]
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        result=update(model,opt,rows,fmt,UpdateConfig(max_thought_tokens=6))
        assert any(not torch.equal(x,y) for x,y in zip(before,model.parameters()))
        assert all(torch.isfinite(x).all() for x in model.parameters())
        assert result['gradient_norm_before_clip'] >= 0


def test_padding_does_not_change_valid_causal_scores(fmt):
    torch.manual_seed(3); model=TinyCausalLM()
    r=Record('x','doc','test',(4,5,6),(7,8))
    a=scoring_batch([r],[[11,12]],fmt,2)
    b=scoring_batch([r],[[11,12]],fmt,2,max_length=32)
    la=target_log_likelihood(model(a.input_ids),a.labels,a.loss_mask)
    lb=target_log_likelihood(model(b.input_ids),b.labels,b.loss_mask)
    assert torch.allclose(la,lb,atol=1e-6)


def test_header_matches_generation_and_scoring():
    fmt=Format((1,),(2,),3,0,header_ids=(20,21))
    r=Record('x','doc','train',(5,6),(7,))
    assert writer_inputs([r.prefix],fmt)[0]==[20,21,5,6,1]
    scored=scoring_batch([r],[[9]],fmt,1)
    assert scored.input_ids[0].tolist()==[20,21,5,6,1,9,2,7]
    raw=scoring_batch([r],[None],fmt,1)
    assert raw.input_ids[0].tolist()==[5,6,7]


def test_accumulated_micro_gradients_equal_global_batch():
    torch.manual_seed(55)
    lp=torch.randn(4,4,8,requires_grad=True)
    mask=torch.rand(4,4,8)>0.2
    adv=torch.randn(4,4)
    global_grad=torch.autograd.grad(policy_loss(lp,mask,adv),lp,retain_graph=True)[0]
    # Simulated 2 ranks, 2 accumulation micros each. The reducer averages ranks.
    split_loss=sum(policy_loss(lp[i:i+1],mask[i:i+1],adv[i:i+1])/4 for i in range(4))
    split_grad=torch.autograd.grad(split_loss,lp)[0]
    assert torch.allclose(global_grad,split_grad)


def test_explicit_eos_is_a_real_stop_action():
    actions=torch.tensor([[4,8,9,0,0]])
    assert action_mask(actions,3,(9,)).int().tolist()==[[1,1,1,0,0]]
    assert content_from_actions(actions[0].tolist(),3,(9,))==[4,8]


def test_multiple_stop_reference_is_consistent():
    torch.manual_seed(77)
    fmt=Format((1,),(2,),3,0,additional_stop_ids=(9,))
    cfg=UpdateConfig(max_thought_tokens=12)
    model=TinyCausalLM(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    rows=[Record('a','d','train',(5,6),(7,))]
    report=thought_update(model,opt,rows,fmt,cfg)
    assert report['max_behavior_replay_error'] < 1e-5


def test_format_json_roundtrip():
    from dataclasses import asdict
    import json
    fmt=Format((1,),(2,),3,0,(20,21),(9,))
    assert Format.from_dict(json.loads(json.dumps(asdict(fmt))))==fmt
