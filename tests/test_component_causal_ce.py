import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.masked_causal_ce import masked_component_causal_ce
from train.neuron_backend import NeuronCausalBackend


def test_three_component_loss_and_gradients_equal_dense_causal_reference():
    torch.manual_seed(23)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=19, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attn_implementation='eager')).eval()
    dense = copy.deepcopy(model)
    ids = torch.randint(0, 19, (2, 12))
    positions = torch.tensor([[1, 3, 5, 7, 8, 10], [2, 4, 6, 8, 9, 11]])
    labels = ids.gather(1, positions)
    masks = {
        'source': torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool),
        'answer': torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]], dtype=torch.bool),
        'selection': torch.tensor([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)}
    weights = dict(source=.5, answer=.5, selection=.1)
    actual = masked_component_causal_ce(NeuronCausalBackend(model, head_chunk=4),
        ids, positions, labels, masks, weights)
    logits = dense(ids[:, :-1]).logits.float()
    all_ce = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction='none')
    expected = torch.zeros(())
    for name, mask in masks.items():
        values = torch.stack([all_ce[i, positions[i, mask[i]] - 1].mean() for i in range(2)])
        assert torch.allclose(actual.per_example[name], values, atol=1e-6)
        assert torch.equal(actual.counts[name], mask.sum(1).float())
        expected = expected + weights[name] * values.mean()
    assert torch.allclose(actual.loss, expected, atol=1e-6)
    actual.loss.backward()
    expected.backward()
    for (name, parameter), (other_name, reference) in zip(model.named_parameters(), dense.named_parameters()):
        assert name == other_name
        assert torch.allclose(parameter.grad, reference.grad, atol=1e-6, rtol=1e-5), name


class IndependentLogits:
    """Isolate normalization coefficients, not a claimed language-model behavior."""
    def __init__(self, logits):
        self.device = torch.device('cpu')
        self.head_chunk = 3
        self.logits = torch.nn.Parameter(logits.clone())
        self.model = SimpleNamespace(config=SimpleNamespace(vocab_size=7), lm_head=torch.nn.Identity())

    def hidden(self, ids):
        return self.logits

    def head_amp(self):
        return contextlib.nullcontext()


def test_longer_auxiliary_sequence_cannot_dilute_final_answer_coefficients():
    torch.manual_seed(24)
    logits = torch.randn(1, 11, 7)
    gradients = []
    for selection_length in (0, 2, 6):
        backend = IndependentLogits(logits)
        ids = torch.ones((1, 12), dtype=torch.long)
        positions = torch.arange(1, 11)[None]
        labels = torch.ones_like(positions)
        masks = {name: torch.zeros_like(positions, dtype=torch.bool)
                 for name in ('source', 'answer', 'selection')}
        masks['source'][0, :2] = True
        masks['answer'][0, 2:4] = True
        masks['selection'][0, 4:4 + selection_length] = True
        result = masked_component_causal_ce(backend, ids, positions, labels,
            masks, dict(source=.5, answer=.5, selection=.1))
        result.loss.backward()
        gradients.append(backend.logits.grad[0, 2:4].clone())
        if selection_length == 0:
            assert result.losses['selection'].item() == 0
    assert all(torch.equal(gradients[0], other) for other in gradients[1:])


@pytest.mark.parametrize('change', ['overlap', 'weight_keys', 'negative_weight', 'integer_mask'])
def test_malformed_components_fail_before_backend_work(change):
    ids = torch.ones((1, 4), dtype=torch.long)
    positions = torch.tensor([[1, 2]])
    masks = dict(source=torch.tensor([[True, False]]), answer=torch.tensor([[False, True]]))
    weights = dict(source=.5, answer=.5)
    if change == 'overlap':
        masks['answer'][0, 0] = True
    elif change == 'weight_keys':
        weights['other'] = .1
    elif change == 'negative_weight':
        weights['answer'] = -.1
    else:
        masks['answer'] = masks['answer'].long()
    with pytest.raises(ValueError):
        masked_component_causal_ce(SimpleNamespace(), ids, positions,
                                   torch.ones_like(positions), masks, weights)
