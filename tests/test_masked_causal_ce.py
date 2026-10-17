"""Native dense causal CE/gradient parity; synthetic CPU models and tokens only."""
import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.masked_causal_ce import masked_causal_ce
from train.neuron_backend import NeuronCausalBackend


def backend(head_chunk=3):
    torch.manual_seed(81)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=19, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=0, attn_implementation='eager')).eval()
    return NeuronCausalBackend(model, head_chunk=head_chunk, checkpoint_trunk=True)


def metadata(x, positions, note, future):
    p = torch.tensor(positions, dtype=torch.long)
    n, f = torch.tensor(note, dtype=torch.bool), torch.tensor(future, dtype=torch.bool)
    y = x.gather(1, p.clamp(0, x.shape[1]-1))
    y = torch.where(n | f, y, -100)
    return p, y, n, f


def dense(model, x, p, y, n, f):
    logits = model(x[:, :-1]).logits.float()
    all_ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction='none')
    results = []
    for mask in (n, f):
        dense_mask = torch.zeros_like(all_ce)
        for row in range(len(x)):
            dense_mask[row, p[row][mask[row]]-1] = 1.
        counts = dense_mask.sum(-1)
        results.append((all_ce*dense_mask).sum(-1)/counts.clamp_min(1.))
    return results


def assert_gradients(a, b, atol=1e-6):
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        assert p.grad is not None and q.grad is not None, name
        assert torch.allclose(p.grad, q.grad, atol=atol, rtol=1e-5), name


def test_shifted_dense_ce_and_all_tied_gradients_match_variable_future_positions():
    b = backend(); reference = copy.deepcopy(b.model)
    x = torch.tensor([[10,11,2,7,0,3,4,11,12,13,0,0],
                      [11,12,2,8,9,7,3,4,12,13,14,15]])
    args = metadata(x, [[3,4,5,6,9,10,0,0], [4,5,6,7,10,11,0,0]],
        [[1,1,1,1,0,0,0,0]]*2, [[0,0,0,0,1,1,0,0]]*2)
    actual = masked_causal_ce(b, x, *args, note_weight=.7, future_weight=1.3)
    n, f = dense(reference, x, *args)
    expected = .7*n.mean()+1.3*f.mean()
    assert actual.note_close_counts.tolist() == [4.,4.]
    assert actual.future_counts.tolist() == [2.,2.]
    assert args[1][0,1] == 0  # Legal EOS=pad remains a supervised note token.
    assert torch.allclose(actual.loss, expected, atol=1e-6)
    assert torch.allclose(actual.note_close_per_example, n, atol=1e-6)
    actual.loss.backward(); expected.backward()
    assert b.model.get_input_embeddings().weight is b.model.lm_head.weight
    assert len(list(b.model.parameters())) == len({id(p) for p in b.model.parameters()})
    assert_gradients(b.model, reference)


def test_separate_example_means_and_absent_components_keep_fixed_batch_denominator():
    b = backend(); reference = copy.deepcopy(b.model)
    x = torch.tensor([[10,11,2,7,3,4,12,13], [11,12,2,8,9,3,4,15]])
    args = metadata(x, [[3,4,5,6,7], [3,4,5,6,7]],
        [[1,1,0,0,0], [0,0,0,0,0]], [[0,0,0,1,1], [0,0,0,0,1]])
    actual = masked_causal_ce(b, x, *args)
    n, f = dense(reference, x, *args)
    assert actual.note_close_counts.tolist() == [2.,0.]
    assert actual.future_counts.tolist() == [2.,1.]
    assert actual.note_close_per_example[1] == 0.
    assert torch.allclose(actual.note_close_loss, n.sum()/2, atol=1e-6)
    assert torch.allclose(actual.future_loss, f.mean(), atol=1e-6)
    actual.loss.backward(); (n.mean()+f.mean()).backward()
    assert_gradients(b.model, reference)


def test_full_256_content_plus_two_close_targets_are_not_truncated_to_action_cap():
    b = backend(head_chunk=32); reference = copy.deepcopy(b.model)
    x = torch.tensor([[10,11,2]+[7]*256+[3,4]+[10,11]+[8,9]])
    positions = [list(range(3,261))+[263,264]]
    args = metadata(x, positions, [[1]*258+[0]*2], [[0]*258+[1]*2])
    actual = masked_causal_ce(b, x, *args)
    n, f = dense(reference, x, *args)
    assert actual.note_close_counts.item() == 258.
    assert actual.future_counts.item() == 2.
    assert torch.allclose(actual.loss, n.mean()+f.mean(), atol=1e-6)
    actual.loss.backward(); (n.mean()+f.mean()).backward()
    assert_gradients(b.model, reference)


def test_own_and_later_future_inputs_cannot_predict_the_first_future_label():
    a = backend(); b = backend()
    x = torch.tensor([[10,11,2,7,3,4,11,12,8,9]])
    args = metadata(x, [[3,4,5,8]], [[1,1,1,0]], [[0,0,0,1]])
    changed = x.clone(); changed[0,8:] = torch.tensor([16,17])
    # Labels deliberately remain fixed: neither own target input nor later
    # tokens may change its prediction. Only predecessor context is observable.
    first = masked_causal_ce(a, x, *args)
    second = masked_causal_ce(b, changed, *args)
    assert torch.equal(first.loss, second.loss)
    first.loss.backward(); second.loss.backward()
    assert_gradients(a.model, b.model, atol=0.)


@pytest.mark.parametrize('slots', [0,3])
def test_all_zero_masks_have_finite_differentiable_zero_and_zero_gradients(slots):
    b = backend()
    x = torch.tensor([[10,11,2,7,3,4]])
    p = torch.zeros((1,slots), dtype=torch.long)
    y = torch.full_like(p, -100)
    mask = torch.zeros_like(p, dtype=torch.bool)
    result = masked_causal_ce(b, x, p, y, mask, mask)
    assert result.loss.item() == 0. and result.note_close_counts.item() == 0.
    assert result.future_counts.item() == 0.
    result.loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in b.model.parameters())


@pytest.mark.parametrize('kind', ['position0','position_end','label','overlap','duplicate','weight'])
def test_invalid_supervision_is_rejected_before_hidden_forward(kind, monkeypatch):
    b = backend()
    x = torch.tensor([[10,11,2,7,3,4]])
    p,y,n,f = metadata(x, [[3,4]], [[1,1]], [[0,0]])
    weights = {}
    if kind == 'position0': p[0,0] = 0
    if kind == 'position_end': p[0,0] = x.shape[1]
    if kind == 'label': y[0,0] = 19
    if kind == 'overlap': f[0,0] = True
    if kind == 'duplicate': p[0,1] = p[0,0]
    if kind == 'weight': weights['future_weight'] = float('nan')
    monkeypatch.setattr(b, 'hidden', lambda _: pytest.fail('Invalid metadata reached model'))
    with pytest.raises(ValueError): masked_causal_ce(b, x, p,y,n,f, **weights)
