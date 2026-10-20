"""V29 causal/source/normalization tests; these do not qualify hardware."""
import copy
from collections import Counter

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train import real_cpt_scaffold as scaffold
from train.masked_causal_ce import masked_component_causal_ce
from train.neuron_backend import NeuronCausalBackend
from train.useful_note_sft_data import EOS


class Tokenizer:
    eos_token_id = EOS
    all_special_ids = [EOS, 151668]
    def __len__(self): return 151669
    def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)


def example(kind='calculation', document='one'):
    tok = Tokenizer()
    prefix = tok.encode('Source context: ' * 9 + '\n(2 + 3) * 4 = ')
    future = tok.encode('20. More text...')
    if kind == 'ordinary': prefix = tok.encode('a' * 512)
    row = dict(document_id=document, split='train', kind=kind, prefix_ids=prefix,
               continuation16_ids=future, stop_id=EOS)
    if kind == 'calculation':
        text, ids = scaffold.verified_work(prefix, tok)
        row.update(full_calculation_text=text, full_calculation_ids=ids)
    return row


def selected(pack, component, key='labels'):
    return pack[key][pack['component_masks'][component]].tolist()


def test_clean_pass_is_identical_across_arms_and_includes_unmodified_future():
    row = example()
    views = [scaffold.training_views(row, Tokenizer(), arm) for arm in scaffold.ARMS]
    for v in views:
        assert selected(v.clean, 'source') == (row['prefix_ids'] + row['continuation16_ids'])[1:]
        assert selected(v.clean, 'source', 'target_positions') == list(range(1, len(row['prefix_ids'])+16))
        assert selected(v.auxiliary, 'continuation') == row['continuation16_ids']
        assert EOS not in selected(v.clean, 'source')
    for v in views[1:]:
        for key in ('input_ids', 'target_positions', 'labels'):
            assert torch.equal(v.clean[key], views[0].clean[key])
    assert not views[0].auxiliary['component_masks']['work'].any()
    assert selected(views[1].auxiliary, 'work') == Tokenizer().encode('Answer:\n')
    assert selected(views[2].auxiliary, 'work') == row['full_calculation_ids'] + Tokenizer().encode('Answer:\n')


def test_writer_computation_cannot_read_future_annotations():
    a = example()
    b = copy.deepcopy(a)
    b.update(continuation16_ids=[99]*16, numeric_value='BAD', numeric_surface_text='BAD')
    x = scaffold.training_views(a, Tokenizer(), 'computation')
    y = scaffold.training_views(b, Tokenizer(), 'computation')
    assert selected(x.auxiliary, 'work') == selected(y.auxiliary, 'work')
    cut = len(a['prefix_ids'])
    assert torch.equal(x.clean['input_ids'][:, :cut], y.clean['input_ids'][:, :cut])
    n = selected(x.auxiliary, 'continuation', 'target_positions')[0]
    assert torch.equal(x.auxiliary['input_ids'][:, :n], y.auxiliary['input_ids'][:, :n])


def test_ordinary_documents_receive_only_clean_source_supervision():
    for arm in scaffold.ARMS:
        view = scaffold.training_views(example('ordinary'), Tokenizer(), arm)
        assert len(selected(view.clean, 'source')) == 527
        assert view.auxiliary is None


def test_pinned_native_metadata_is_not_requeried_per_token_or_view():
    class CountingTokenizer(Tokenizer):
        calls = 0
        def __len__(self):
            self.calls += 1
            return super().__len__()
    tok = CountingTokenizer()
    for arm in scaffold.ARMS:
        for _ in range(3):
            scaffold.training_views(example(), tok, arm)
    assert tok.calls == 1


@pytest.mark.parametrize('change', [dict(split='dev'), dict(split='test'),
    dict(full_calculation_ids=[9]), dict(continuation16_ids=[EOS]*16),
    dict(continuation16_ids=[151668]*16), dict(stop_id=1)])
def test_reject_eval_leakage_corrupt_teachers_and_special_source_tokens(change):
    row = example(); row.update(change)
    with pytest.raises(ValueError): scaffold.training_views(row, Tokenizer(), 'computation')


def test_no_silent_truncation():
    with pytest.raises(ValueError):
        scaffold.training_views(example(), Tokenizer(), 'computation', width=150)


def test_grouped_order_preserves_all_sources_and_exact_auxiliary_fraction():
    rows = [example(document=f'c{i}') for i in range(8)]
    rows += [example('ordinary', document=f'o{i}') for i in range(24)]
    a = scaffold.grouped_order(rows, seed=29, passes=2)
    b = scaffold.grouped_order(rows, seed=30, passes=2)
    assert a != b and Counter(a) == Counter({i:2 for i in range(32)})
    for start in range(0, len(a), 8):
        assert sum(rows[i]['kind']=='calculation' for i in a[start:start+8]) == 2
    assert scaffold.auxiliary_weights(calculation_examples=2, source_examples=8) == {
        'work': .025, 'continuation': .125}
    rows[0]['document_id'] = rows[-1]['document_id']
    with pytest.raises(ValueError): scaffold.grouped_order(rows, seed=29, passes=2)


@pytest.mark.parametrize('arm', scaffold.ARMS)
def test_real_two_forward_objective_matches_dense_loss_and_gradients(arm):
    torch.manual_seed(29)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=19, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=0, attn_implementation='eager')).eval()
    reference = copy.deepcopy(model)
    backend = NeuronCausalBackend(model, head_chunk=32, checkpoint_trunk=True)
    views = scaffold.training_views(example(), Tokenizer(), arm)
    actual, expected = [], []
    weights = [{'source': 1.}, scaffold.auxiliary_weights(
        calculation_examples=1, source_examples=4)]
    for original, component_weights in zip([views.clean, views.auxiliary], weights):
        p = copy.deepcopy(original)
        p['input_ids'] %= 19
        active = torch.stack(list(p['component_masks'].values())).any(0)
        p['labels'] = torch.where(active, p['labels'] % 19, -100)
        actual.append(masked_component_causal_ce(backend, **p,
                       component_weights=component_weights).loss)
        x = p['input_ids']
        logits = reference(x[:, :-1]).logits.float()
        ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction='none')
        total = ce.sum() * 0.
        for name, mask in p['component_masks'].items():
            positions = p['target_positions'][mask] - 1
            if positions.numel(): total = total + component_weights[name] * ce[0, positions].mean()
        expected.append(total)
    loss, dense = sum(actual), sum(expected)
    assert torch.allclose(loss, dense, atol=1e-6)
    # Two actual backward passes accumulate before the ONE optimizer step.
    for part in actual: part.backward()
    dense.backward()
    for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        assert torch.allclose(p.grad, q.grad, atol=2e-6, rtol=2e-5), name
