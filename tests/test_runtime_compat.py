"""The compatibility change must preserve successful calls and real errors."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from train.runtime_compat import _supply_wrapped_output_path, _uncached_device_state_transfer


def test_missing_compiler_artifact_path_is_supplied_without_altering_other_args(tmp_path):
    calls = []

    def original(output_neff, entry, retry=0, output_wrapped_neff=None):
        calls.append((output_neff, entry, retry, output_wrapped_neff))
        return b"identical compiler result"

    wrapped = _supply_wrapped_output_path(original)
    neff = str(tmp_path / "model.neff")
    assert wrapped(neff, "cache entry", retry=3) == b"identical compiler result"
    assert calls[-1] == (neff, "cache entry", 3, str(tmp_path / "wrapped_neff.hlo"))
    supplied = str(tmp_path / "caller-supplied.hlo")
    wrapped(neff, "cache entry", 2, supplied)
    assert calls[-1] == (neff, "cache entry", 2, supplied)
    wrapped(neff, "cache entry", 1, None)
    assert Path(calls[-1][-1]).name == "wrapped_neff.hlo"


def test_real_compiler_errors_propagate():
    def failing(output_neff, entry, output_wrapped_neff=None):
        raise RuntimeError("real compiler failure")

    with pytest.raises(RuntimeError, match="real compiler failure"):
        _supply_wrapped_output_path(failing)("/tmp/model.neff", object())


def test_unknown_compiler_signature_fails_closed():
    with pytest.raises(RuntimeError, match="unexpected signature"):
        _supply_wrapped_output_path(lambda output_neff: output_neff)


def test_configure_runtime_is_idempotent_and_rejects_unqualified_source(monkeypatch, tmp_path):
    import libneuronxla.neuron_cc_wrapper as compiler_wrapper
    from train.runtime_compat import configure_runtime

    first = configure_runtime()
    patched = compiler_wrapper.compile_cache_entry
    assert configure_runtime() == first
    assert compiler_wrapper.compile_cache_entry is patched
    assert first["installed_files_changed"] is False
    unknown_source = tmp_path / "unknown_wrapper.py"
    unknown_source.write_text("# Changed source must be requalified.\n")
    monkeypatch.setattr(compiler_wrapper, "__file__", str(unknown_source))
    with pytest.raises(RuntimeError, match="source hash"):
        configure_runtime()


@pytest.mark.parametrize('previous', [True, False])
def test_equal_mutable_counters_are_independent_and_cache_setting_is_restored(previous):
    import torch

    enabled = [previous]
    cache = {}
    events = []

    def transfer(value):
        if enabled[0]:
            return cache.setdefault(float(value), value.clone())
        return value.clone()

    def synchronize(**kwargs):
        assert enabled[0] is False
        events.append(kwargs)

    runtime = SimpleNamespace(_XLAC=SimpleNamespace(
        _get_xla_enable_device_data_cache=lambda: enabled[0],
        _set_xla_enable_device_data_cache=lambda value: enabled.__setitem__(0, value)),
        sync=synchronize)
    with _uncached_device_state_transfer(runtime):
        counters = [transfer(torch.tensor(2.)) for _ in range(3)]
    counters[0].add_(1)
    assert [float(v) for v in counters] == [3., 2., 2.]
    assert enabled[0] is previous
    assert events == [{'wait': True}]


@pytest.mark.parametrize('failure', ['transfer', 'synchronize'])
def test_device_transfer_cache_is_restored_after_failure(failure):
    enabled = [True]

    def synchronize(**kwargs):
        raise RuntimeError('synchronize failed')

    runtime = SimpleNamespace(_XLAC=SimpleNamespace(
        _get_xla_enable_device_data_cache=lambda: enabled[0],
        _set_xla_enable_device_data_cache=lambda value: enabled.__setitem__(0, value)),
        sync=synchronize)
    with pytest.raises(RuntimeError, match=failure+' failed'):
        with _uncached_device_state_transfer(runtime):
            if failure == 'transfer':
                raise RuntimeError('transfer failed')
    assert enabled[0] is True
