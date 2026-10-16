"""Fixed experiment and real driver control flow on CPU-only runtime fixtures.

These tests execute worker(), replacing the accelerator/runtime boundary and
tiny model/backend. They do not qualify Neuron numerics or device execution.
"""
from __future__ import annotations

import copy
import importlib.machinery
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import train.neuron_driver as driver
import train.runtime_compat as compatibility
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, Record
from weekend_thinking.data import write_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def proposal():
    return json.loads((ROOT / "configs/weekend.json").read_text())


def test_current_frozen_recipe_accepts_both_predeclared_seeds(proposal):
    for seed in (1234, 5678):
        driver.validate_fixed_recipe(proposal, seed)


@pytest.mark.parametrize("section,field,changed", [
    ("thought_arm", "algorithm", "best_of_only_policy_update"),
    ("thought_arm", "ntp_auxiliary_weight", 0.1),
    ("thought_arm", "temperature", 1.0),
    ("thought_arm", "normalization", "fixed_cap"),
    ("optimization", "learning_rate", 2e-6),
])
def test_changed_scientific_recipe_is_rejected_before_launch(proposal, section, field, changed):
    changed_proposal = copy.deepcopy(proposal)
    changed_proposal[section][field] = changed
    with pytest.raises(ValueError, match=f"{section}.{field}"):
        driver.validate_fixed_recipe(changed_proposal, 1234)


def test_seed_outside_the_frozen_plan_is_rejected(proposal):
    with pytest.raises(ValueError, match="seed.*absent"):
        driver.validate_fixed_recipe(proposal, 999)


def cpu_worker_fixture(tmp_path, monkeypatch, *, condition="finite", arm="vanilla"):
    """Keep worker's actual branch/guard ordering; inject only its boundaries."""
    events = SimpleNamespace(optimizer_steps=0, replay_calls=0, checkpoints=[], reductions=[])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.config = SimpleNamespace(tie_word_embeddings=True, vocab_size=13)
            if condition == "nonfinite_gradient":
                self.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

        def tie_weights(self):
            pass

        def get_input_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        get_output_embeddings = get_input_embeddings

    model = Model()

    class Backend:
        def __init__(self, actual_model, **kwargs):
            assert actual_model is model

        def vanilla_loss(self, prefix, future):
            loss = model.weight.square().sum()
            return loss * float("nan") if condition == "nonfinite_loss" else loss

        def sample(self, prefix, **kwargs):
            assert arm == "thought", "Vanilla branch called the thought sampler"
            actions = torch.tensor([[[7, 5]] * 4])
            return Rollouts(actions, torch.ones_like(actions, dtype=torch.bool),
                            torch.zeros_like(actions, dtype=torch.float32), 0)

        def score(self, prefix, future, rollout, **kwargs):
            assert arm == "thought", "Vanilla branch called the reward reader"
            return torch.full((1, 4), float("nan") if condition == "nonfinite_reward" else 0.0)

        def clean_score(self, prefix, future):
            assert arm == "thought", "Vanilla branch called the clean reader"
            return torch.zeros(1)

        def replay_one_loss(self, *args, **kwargs):
            events.replay_calls += 1
            return model.weight.square().sum()

    class Optimizer:
        def __init__(self, parameters, optimizer_class, **kwargs):
            self.parameters = list(parameters)
            self.inner = torch.optim.SGD(self.parameters, lr=kwargs["lr"])
            self.grad_norm = torch.tensor(0.0)

        def zero_grad(self, **kwargs):
            self.inner.zero_grad(**kwargs)

        def step(self, **kwargs):
            events.optimizer_steps += 1
            self.grad_norm = torch.stack([p.grad.square().sum() for p in self.parameters]).sum().sqrt()
            self.inner.step()

    def module(name):
        value = ModuleType(name)
        value.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        value.__path__ = []
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, value)
        return value

    xla = module("torch_xla")
    xla.device = lambda: torch.device("cpu")
    xla.sync = lambda **kwargs: None
    module("torch_xla.amp")
    syncfree = module("torch_xla.amp.syncfree")
    syncfree.AdamW = torch.optim.AdamW
    module("torch_xla.core")
    xm = module("torch_xla.core.xla_model")
    xm.REDUCE_MAX, xm.REDUCE_SUM = "max", "sum"
    xm.rendezvous = lambda *args: None
    xm.set_rng_state = lambda *args: None

    def reduce(operation, value):
        events.reductions.append((operation, value.detach().clone()))
        if operation == "max" and condition == "remote_nonfinite_gradient":
            return torch.ones_like(value)
        return value

    xm.all_reduce = reduce
    xm.all_gather = lambda value: value
    runtime = module("torch_xla.runtime")
    runtime.global_ordinal, runtime.world_size = lambda: 0, lambda: 1
    module("torch_xla.distributed")
    module("torch_xla.distributed.xla_backend")
    zero = module("torch_xla.distributed.zero_redundancy_optimizer")
    zero.ZeroRedundancyOptimizer = Optimizer
    module("torch_xla.debug")
    metrics = module("torch_xla.debug.metrics")
    metrics.metrics_report = lambda: "CPU control-flow fixture; no accelerator execution"
    monkeypatch.setattr(torch, "xla", xla, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(compatibility, "configure_runtime", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(driver, "NeuronCausalBackend", Backend)
    tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(13)},
                                decode=lambda values: " ".join(map(str, values)))
    monkeypatch.setattr(driver, "load_base", lambda path: (model, tokenizer))
    monkeypatch.setattr(driver, "source_fingerprint", lambda path: {"cpu_fixture": True})
    monkeypatch.setattr(driver, "save_checkpoint", lambda *args, **kwargs: events.checkpoints.append(args[3]))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "ASSET_MANIFEST.json").write_text("{}")
    records, formatted = tmp_path / "records.jsonl", tmp_path / "format.json"
    write_records(records, [Record("r", "d", "train", (2, 3), (4,))])
    fmt = Format((2,), (3,), 5, 0)
    formatted.write_text(json.dumps({"format": dict(open_ids=fmt.open_ids, close_ids=fmt.close_ids,
                                                  stop_id=fmt.stop_id, pad_id=fmt.pad_id)}))
    args = SimpleNamespace(qualify=True, model_dir=model_dir, records=records, format=formatted,
        max_steps=1, global_prefixes=1, diagnostic_cap=2, precision="fp32", arm=arm,
        seed=1234, processes=1, resume=None, output=tmp_path / "run", stop_file=None,
        checkpoint_interval_seconds=43200)
    return args, model, events


def test_vanilla_step_does_not_access_thought_only_variables(tmp_path, monkeypatch):
    args, model, events = cpu_worker_fixture(tmp_path, monkeypatch)
    driver.worker(0, args)
    assert events.optimizer_steps == 1 and events.replay_calls == 0
    assert model.weight.item() < 1.0
    log = json.loads((args.output / "metrics.jsonl").read_text())
    assert log["counters"] == {"source_records": 1, "source_tokens": 3, "sampled_actions": 0}
    assert log["metrics"]["loss"] == 1.0
    for key in ('clean', 'thought', 'gain', 'capped', 'abs_advantage', 'mean_signed_length'):
        assert log['metrics'][key] is None
    assert log['max_replay_logp_error'] is None
    assert log['replay_audit_performed'] is False


@pytest.mark.parametrize("condition,error", [
    ("nonfinite_loss", "Nonfinite continuation loss before backward"),
    ("nonfinite_gradient", "Nonfinite gradient detected before optimizer update"),
])
def test_nonfinite_vanilla_failure_preserves_weights_before_optimizer_step(tmp_path, monkeypatch, condition, error):
    args, model, events = cpu_worker_fixture(tmp_path, monkeypatch, condition=condition)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match=error):
        driver.worker(0, args)
    assert events.optimizer_steps == 0 and events.checkpoints == []
    assert torch.equal(model.weight, before)
    if condition == "nonfinite_loss":
        assert model.weight.grad is None
    else:
        assert torch.isinf(model.weight.grad).all()
        assert any(operation == "max" and bool(value) for operation, value in events.reductions)


def test_nonfinite_reader_reward_stops_before_policy_replay(tmp_path, monkeypatch):
    args, model, events = cpu_worker_fixture(tmp_path, monkeypatch, condition="nonfinite_reward", arm="thought")
    with pytest.raises(RuntimeError, match="Nonfinite reward before policy replay"):
        driver.worker(0, args)
    assert events.replay_calls == events.optimizer_steps == 0
    assert model.weight.grad is None and model.weight.item() == 1.0


def test_other_rank_nonfinite_flag_prevents_finite_local_update(tmp_path, monkeypatch):
    args, model, events = cpu_worker_fixture(tmp_path, monkeypatch, condition="remote_nonfinite_gradient")
    with pytest.raises(RuntimeError, match="Nonfinite gradient detected before optimizer update"):
        driver.worker(0, args)
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.item() == 1.0 and events.optimizer_steps == 0
    assert any(operation == "max" and not bool(value) for operation, value in events.reductions)
