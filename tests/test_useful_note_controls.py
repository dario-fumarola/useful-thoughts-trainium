import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from train.useful_note_controls import pack_raw_continuation, plain_control_ce
from train.useful_note_sft_data import packed


def test_raw_continuation_dense_causal_loss_and_gradient():
    """The compact targets agree with ordinary shifted CE, including real EOS."""
    past = [3, 4] * 64
    future = [5] * 7 + [151643] + [6] * 8
    batch = pack_raw_continuation(past, future, width=160)
    # A cumulative embedding is a small causal model. Only the selected future
    # positions contribute, as they would in a dense prefix-masked CE pass.
    generator = torch.Generator().manual_seed(20261018)
    embedding = torch.randn(151669, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    head = torch.randn(2, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    # Remap target vocabulary for the tiny head while keeping the actual EOS
    # input and its active target-mask position in the packer.
    targets = torch.tensor([5] * 7 + [0] + [6] * 8)
    hidden = embedding[batch['input_ids']].cumsum(1) / 128
    positions = batch['target_positions'][0] - 1
    compact = F.cross_entropy(hidden[0, positions] @ head, targets)
    dense_logits = hidden[0, :-1] @ head
    dense_targets = torch.full((159,), -100)
    dense_targets[len(past) - 1:len(past) + 15] = targets
    dense = F.cross_entropy(dense_logits, dense_targets, ignore_index=-100)
    compact_grads = torch.autograd.grad(compact, (embedding, head), retain_graph=True)
    dense_grads = torch.autograd.grad(dense, (embedding, head))
    assert torch.equal(batch['input_ids'][0, :144], torch.tensor(past + future))
    assert torch.equal(batch['labels'][0], torch.tensor(future))
    assert not batch['note_close_mask'].any()
    assert batch['future_mask'].sum().item() == 16
    assert batch['labels'][0, 7].item() == 151643
    assert torch.allclose(compact, dense, atol=1e-12, rtol=0)
    for a, b in zip(compact_grads, dense_grads):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_later_gold_tokens_cannot_change_earlier_predictions():
    past = [3] * 128
    original = pack_raw_continuation(past, [4] * 16, width=160)
    changed = pack_raw_continuation(past, [4] * 8 + [5] * 8, width=160)
    # The causal prefix available to each of the first nine predictions is
    # exactly unchanged, including the prediction of the first changed label.
    for index in range(9):
        position = original['target_positions'][0, index].item()
        assert torch.equal(original['input_ids'][0, :position], changed['input_ids'][0, :position])
    assert set(inspect.signature(pack_raw_continuation).parameters) == {
        'prefix_ids', 'continuation_ids', 'width', 'target_slots'}


@pytest.mark.parametrize('cut', [128, 512])
def test_single_pass_plain_control_matches_dense_objective_and_gradient(cut):
    torch.manual_seed(42)
    embedding = torch.nn.Embedding(11, 3)
    head = torch.nn.Linear(3, 11, bias=False)
    backend = SimpleNamespace(device=torch.device('cpu'), head_chunk=128,
                              model=SimpleNamespace(config=SimpleNamespace(vocab_size=11), lm_head=head),
                              hidden=lambda ids: embedding(torch.where(ids == 151643, 0, ids)).cumsum(1) / 528,
                              head_amp=nullcontext)
    ids = [i % 11 for i in range(528)]
    batch = packed(ids, range(1, 528), note_positions=(), width=544, target_slots=544)
    result = plain_control_ce(backend, **batch, source_cut=cut)
    logits = backend.hidden(batch['input_ids'][:, :-1])[0, :527] @ head.weight.T
    losses = F.cross_entropy(logits, torch.tensor(ids[1:]), reduction='none')
    source = losses.mean()
    continuation = losses[cut - 1:cut + 15].mean()
    expected = continuation + .5 * source
    assert torch.allclose(result.continuation_loss, continuation, atol=1e-6, rtol=0)
    assert torch.allclose(result.source_loss, source, atol=1e-6, rtol=0)
    assert torch.allclose(result.loss, expected, atol=1e-6, rtol=0)
    # The production head uses reentrant checkpointing, so exercise its actual
    # backward API instead of torch.autograd.grad(), which it does not support.
    result.loss.backward()
    actual_grad = (embedding.weight.grad.clone(), head.weight.grad.clone())
    expected_grad = torch.autograd.grad(expected, (embedding.weight, head.weight))
    for a, b in zip(actual_grad, expected_grad):
        assert torch.allclose(a, b, atol=1e-6, rtol=0)


def test_plain_control_rejects_an_altered_source_label():
    batch = packed([3] * 528, range(1, 528), note_positions=(), width=544, target_slots=544)
    batch['labels'][0, 199] = 4
    with pytest.raises(ValueError, match='original source tokens'):
        plain_control_ce(None, **batch, source_cut=200)


@pytest.mark.parametrize('past,future,width,slots', [
    ([3] * 127, [4] * 16, 160, 16),
    ([3] * 513, [4] * 16, 544, 16),
    ([3] * 128, [4] * 15, 160, 16),
    ([3] * 128, [4] * 16, 143, 16),
    ([3] * 128, [4] * 16, 160, 32),
    ([True] * 128, [4] * 16, 160, 16),
])
def test_bad_horizon_or_truncation_rejected(past, future, width, slots):
    with pytest.raises(ValueError):
        pack_raw_continuation(past, future, width=width, target_slots=slots)
