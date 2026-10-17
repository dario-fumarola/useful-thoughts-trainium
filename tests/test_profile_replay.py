import math

import pytest
import torch

from train.profile_replay import action_bucket, choose_cases, gradient_geometry, geometry_summary


def test_buckets_keep_real_stop_and_reject_hidden_tail():
    assert action_bucket([7], {7}) == 1
    assert action_bucket([3] * 32 + [7], {7}) == 64
    assert action_bucket([3] * 1024, {7}) == 1024
    with pytest.raises(ValueError, match="after their first stop"):
        action_bucket([3, 7, 3], {7})
    with pytest.raises(ValueError, match="real stop"):
        action_bucket([3], {7})


def test_cases_use_frozen_order_and_length_never_scores():
    selection = {"split": "dev", "item_ids": ["first", "short", "later"]}
    notes = {key: {"status": "ok", "split": "dev", "actions": actions}
             for key, actions in [("first", [2] * 1024), ("short", [3, 7]), ("later", [4, 7])]}
    cases = choose_cases(selection, notes, {7})
    assert cases[0]["item_id"] == "first"
    assert cases[0]["actions"] == [7]
    assert cases[1]["item_id"] == "short"
    assert cases[1]["actions"] == [3, 7]
    with pytest.raises(ValueError, match="DEV"):
        choose_cases({**selection, "split": "test"}, notes, {7})
    notes["first"]["status"] = "error"
    with pytest.raises(ValueError, match="drop"):
        choose_cases(selection, notes, {7})


def test_full_gradient_geometry_detects_local_corruption_and_sign():
    reference = torch.tensor([3., 4., 0.])
    actual = torch.tensor([3., 4., 1.])
    result = geometry_summary(gradient_geometry(reference, actual))
    assert result["elements"] == 3
    assert result["relative_l2"] == pytest.approx(.2)
    assert result["cosine"] == pytest.approx(5 / math.sqrt(26))
    negative = geometry_summary(gradient_geometry(reference, -reference))
    assert negative["relative_l2"] == 2
    assert negative["cosine"] == -1
    with pytest.raises(ValueError, match="Nonfinite"):
        gradient_geometry(reference, torch.tensor([3., float("nan"), 0.]))


def test_actual_causal_backend_truncation_preserves_signed_loss_and_all_gradients():
    # Exercise the existing backend's checkpointed vocabulary loss, including
    # the sampled stop and its sequence normalization, with deterministic causal
    # hidden states. This is CPU semantics only, not a Neuron/B0 qualification.
    from types import SimpleNamespace
    from train.neuron_backend import NeuronCausalBackend
    from weekend_thinking.core import Format, action_mask

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3", layer_types=["full_attention"])
            self.embedding = torch.nn.Embedding(11, 4)
            self.lm_head = torch.nn.Linear(4, 11, bias=False)

    torch.manual_seed(101)
    model = Model()
    backend = NeuronCausalBackend(model, head_chunk=2)
    backend.hidden = lambda ids: model.embedding(ids).cumsum(1)
    fmt = Format(header_ids=(1,), open_ids=(2,), close_ids=(3,), stop_id=9, pad_id=0)
    prefix = torch.tensor([[4, 5]])
    for sign in (1., -1.):
        records = []
        for tokens in ([6, 9, 0, 0, 0, 0], [6, 9]):
            model.zero_grad(set_to_none=True)
            actions = torch.tensor([tokens])
            loss = backend.replay_one_loss(prefix, actions, action_mask(actions, 9),
                torch.tensor(sign), fmt=fmt, temperature=.7, divisor=16)
            loss.backward()
            records.append((loss.detach(), {n:p.grad.clone() for n,p in model.named_parameters()}))
        assert torch.allclose(records[0][0], records[1][0], atol=1e-7, rtol=1e-6)
        for name, grad in records[0][1].items():
            assert torch.allclose(grad, records[1][1][name], atol=1e-7, rtol=1e-6)
