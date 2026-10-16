"""The compatibility change must preserve successful calls and real errors."""

from pathlib import Path

import pytest

from train.runtime_compat import _supply_wrapped_output_path


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
