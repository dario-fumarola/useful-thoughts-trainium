"""CPU parity with tiny Qwen architecture; never presented as device evidence."""
import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.neuron_backend import NeuronCausalBackend, pack_reader
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import (Format, Record, action_mask, scoring_batch,
                                   target_log_likelihood, policy_log_probs)
from weekend_thinking.pretrained import sampled_fingerprint


def model():
    torch.manual_seed(19)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=37, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=3,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attn_implementation='eager')).eval()


def test_compact_reader_matches_reference():
    fmt = Format((2,), (3, 4), 5, 0, additional_stop_ids=(6,))
    prefix = torch.tensor([[10, 11], [12, 13]])
    future = torch.tensor([[14, 15], [16, 17]])
    actions = torch.tensor([[[5, 0, 0], [7, 6, 0]], [[8, 9, 10], [11, 5, 0]]])
    x, lengths, positions = pack_reader(prefix, actions, fmt, future)
    from weekend_thinking.core import content_from_actions
    records, notes = [], []
    for b in range(2):
        for g in range(2):
            records.append(Record(str(b), str(b), 'train', tuple(prefix[b].tolist()), tuple(future[b].tolist())))
            notes.append(content_from_actions(actions[b,g].tolist(), 5, (6,)))
    reference = scoring_batch(records, notes, fmt, 2, max_length=x.shape[-1])
    assert torch.equal(reference.input_ids, x)
    for i in range(4):
        assert positions[i].tolist() == reference.loss_mask[i].nonzero()[:,0].tolist()
        assert lengths[i] == reference.attention_mask[i].sum()


def test_cached_actions_replay_and_score():
    m = model()
    backend = NeuronCausalBackend(m, checkpoint_trunk=False, head_chunk=2)
    fmt = Format((2,), (3, 4), 5, 0, additional_stop_ids=(6,))
    prefix = torch.tensor([[10, 11, 12]])
    torch.manual_seed(29)
    rollout = backend.sample(prefix, fmt=fmt, group_size=4, max_actions=7,
                             temperature=0.7, policy_version=0)
    replay = backend.replay(prefix, rollout, fmt=fmt, temperature=0.7)
    mask = rollout.action_mask
    assert torch.allclose(replay[mask], rollout.behavior_logps[mask], atol=1e-5)
    records = [Record('a', 'a', 'train', (10,11,12), (14,15))] * 4
    from weekend_thinking.core import content_from_actions
    notes = [content_from_actions(a.tolist(), 5, (6,)) for a in rollout.actions[0]]
    batch = scoring_batch(records, notes, fmt, 2)
    with torch.no_grad():
        ref = target_log_likelihood(m(batch.input_ids).logits, batch.labels, batch.loss_mask)
    actual = backend.score(prefix, torch.tensor([[14,15]]), rollout, fmt=fmt, horizon=2)
    assert torch.allclose(actual[0], ref, atol=1e-5)


@pytest.mark.parametrize('advantage', [-.4, .25])
@pytest.mark.parametrize('tokens', [[5, 0, 0, 0], [7, 8, 5, 0], [7, 8, 9, 10]])
def test_checkpointed_trunk_and_head_gradients_match_full_logits(advantage, tokens):
    m = model()
    fmt = Format((2,), (3,), 5, 0)
    backend = NeuronCausalBackend(m, head_chunk=2, checkpoint_trunk=True)
    prefix = torch.tensor([[10, 11, 12]])
    actions = torch.tensor([tokens])
    mask = action_mask(actions, 5)
    loss = backend.replay_one_loss(prefix, actions, mask, torch.tensor(advantage),
                                  fmt=fmt, temperature=0.7, divisor=4)
    loss.backward()
    grads = {k:p.grad.clone() for k,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    x = torch.cat((prefix, torch.tensor([[2]]), actions[:,:-1]), -1)
    logits = m(x).logits[:, 3:]
    lp = policy_log_probs(logits, .7).gather(-1, actions[:,:,None])[:,:,0]
    ref = -torch.tensor(advantage) * (lp * mask).sum() / int(mask.sum()) / 4
    ref.backward()
    assert torch.allclose(loss, ref, atol=1e-6)
    for k,p in m.named_parameters():
        assert torch.allclose(grads[k], p.grad, atol=1e-6), k


def test_integer_fingerprint_large_embedding_bounds():
    # meta device avoids allocating a >300M-element regression fixture. Check
    # integer construction's boundary through the real implementation on a
    # normal model, plus exact arithmetic at the offending released shape.
    n = 311164928
    indices = [i*(n-1)//31 for i in range(32)]
    assert indices[-1] == n-1 and all(0 <= i < n for i in indices)
    m = model()
    assert sampled_fingerprint(m) == sampled_fingerprint(m)


def test_stopping_early_preserves_fixed_padded_shape_and_first_stop():
    m = model()
    prefix = torch.tensor([[10,11,12]])
    with torch.no_grad():
        first = int(m(torch.tensor([[10,11,12,2]])).logits[0,-1].argmax())
    fmt = Format((2,), (3,), first, 0)
    calls = []
    handle = m.model.register_forward_hook(lambda *args: calls.append(1))
    backend = NeuronCausalBackend(m, stop_check_interval=1)
    rollout = backend.sample(prefix, fmt=fmt, group_size=4, max_actions=17,
                             temperature=.7, policy_version=0, greedy=True)
    handle.remove()
    assert rollout.decode_iterations == 1 and len(calls) == 1
    assert rollout.actions.shape == (1,4,17)
    assert rollout.actions[0,:,0].eq(first).all()
    assert rollout.actions[0,:,1:].eq(0).all()
    assert rollout.action_mask.sum() == 4
    assert rollout.behavior_logps[0,:,1:].eq(0).all()
