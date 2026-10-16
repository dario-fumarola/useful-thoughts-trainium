"""Independent local-audit semantics; tiny CPU models, no hardware claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.evaluate import shard_questions
from train.evaluate_local import (HORIZONS, MODES, SCHEMA, construct_controls,
    generate_note, item_seed, load_note_bundles, load_selected, main,
    read_selection, score_record, summarize)
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def model():
    torch.manual_seed(194)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=3,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attn_implementation="eager")).eval()


@pytest.fixture
def fmt():
    return Format((2,), (3, 4), 5, 0, header_ids=(9,), additional_stop_ids=(0,))


def selection(ids=("a", "b", "c"), *, split="dev", records_sha256="0"*64):
    return dict(item_ids=list(ids), document_ids=["document:"+i for i in ids],
        count=len(ids), item_ids_sha256=canonical_hash(list(ids)), split=split,
        records_sha256=records_sha256)


def note(prefix, content, fmt, *, error=False):
    if error:
        return dict(status="error", actions=[], content_ids=[], n_actions=0,
                    prefix_sha256=canonical_hash(prefix))
    actions = list(content)+[fmt.stop_id]
    return dict(status="ok", actions=actions, content_ids=list(content),
        behavior_logps=[-2.]*len(actions), n_actions=len(actions),
        prefix_sha256=canonical_hash(prefix))


def test_selection_uses_existing_global_ids_before_sharding_and_never_allows_test_qualification(tmp_path):
    dev, test = selection(), selection(("t1", "t2"), split="test")
    manifest = {"records_sha256": "0"*64, "splits": {"dev": dev, "test": test}}
    path = tmp_path/"ids.json"
    path.write_text(json.dumps(manifest))
    result = read_selection(path, qualification=True, split="dev")
    assert result["item_ids"] == ["a", "b", "c"]
    parts = [shard_questions(result["item_ids"], rank, 2) for rank in range(2)]
    assert parts == [["a", "c"], ["b"]]
    with pytest.raises(ValueError, match="never final test"):
        read_selection(path, qualification=True, split="test")
    with pytest.raises(ValueError, match="dev512/test2048"):
        read_selection(path, qualification=False, split="dev")
    bad = dict(dev, item_ids_sha256="0"*64)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="ordered-ID hash"):
        read_selection(path, qualification=True)


def test_generation_loading_never_accesses_future_ids_and_checks_exact_global_coverage(tmp_path):
    path = tmp_path/"records.jsonl"
    rows = [dict(id=key, document_id="document:"+key, split="dev", prefix=[10, 11, 12],
                 future="deliberately invalid: generation must not read this") for key in ("c", "a", "b")]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    selected = selection(records_sha256=sha256_file(path))
    actual = load_selected(path, selected, include_future=False)
    assert [row["item_id"] for row in actual] == ["a", "b", "c"]
    assert all("future" not in row for row in actual)
    with pytest.raises(ValueError, match="16-token future"):
        load_selected(path, selected, include_future=True)
    # Even syntactically broken future fields are never decoded for generation.
    broken = path.read_text().replace('"future": "deliberately invalid: generation must not read this"', '"future": UNDECODABLE')
    path.write_text(broken)
    assert len(load_selected(path, dict(selected, records_sha256=sha256_file(path)), include_future=False)) == 3
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete"):
        load_selected(path, selection(("a", "missing"), records_sha256=sha256_file(path)), include_future=False)
    path.write_text(path.read_text()+json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError, match="frozen selection asset"):
        load_selected(path, selected, include_future=False)


def test_one_sample_is_reproducible_per_record_independent_of_shard_order(fmt):
    backend = NeuronCausalBackend(model(), banned_ids=(29, 30), checkpoint_trunk=False,
                                  stop_check_interval=1)
    prefixes = {"a": [10, 11, 12], "b": [13, 14, 15], "c": [16, 17, 18]}
    def run(ids):
        return {key: generate_note(backend, prefixes[key], fmt=fmt, cap=5, policy="sample",
                    seed=item_seed(1729, key), set_seed=torch.manual_seed) for key in ids}
    whole = run(["a", "b", "c"])
    sharded = {**run(["b"]), **run(["a", "c"])}
    assert whole == sharded
    assert len({row["item_seed"] for row in whole.values()}) == 3
    for row in whole.values():
        assert not set(row["actions"]) & {29, 30}
        assert len(row["actions"]) == len(row["behavior_logps"])
        assert row["content_ids"] == row["actions"][:len(row["content_ids"])]
        assert row["stop_type"] == "length_cap" or row["actions"][-1] in {5, 0}


def test_control_randomization_is_global_content_only_and_matches_all_recipient_lengths(fmt):
    selected = selection()
    notes = {"a": note([10, 11], [7, 8, 9], fmt), "b": note([12, 13], [], fmt),
             "c": note([14, 15], [20, 21], fmt)}
    controls = construct_controls(selected, notes, filler_ids=[22, 23], seed=17)
    assert controls == construct_controls(selected, notes, filler_ids=[22, 23], seed=17)
    for key in selected["item_ids"]:
        values = controls[key]["content_ids"]
        assert values["raw"] is None and values["empty"] == []
        assert sorted(values["shuffled"]) == sorted(values["own"])
        assert all(controls[key]["matched_lengths"].values())
        assert all(len(values[mode]) == len(values["own"]) for mode in ("filler", "shuffled", "other"))
    assert controls["a"]["donor_item_id"] == "b"
    assert controls["a"]["content_ids"]["other"] == [22, 23, 22]  # Empty donor uses fixed filler.
    assert controls["c"]["donor_item_id"] == "a"
    assert controls["c"]["content_ids"]["other"] == [7, 8]
    parts = [shard_questions(selected["item_ids"], rank, 2) for rank in range(2)]
    assert {key: controls[key] for part in parts for key in part} == controls


def test_all_conditions_and_horizons_match_independent_full_logit_reader_reference(fmt):
    reader = model()
    backend = NeuronCausalBackend(reader, checkpoint_trunk=False)
    prefix, future = [10, 11, 12], [13+(i % 10) for i in range(16)]
    notes = {"a": note(prefix, [7, 8], fmt), "b": note([15, 16, 17], [20], fmt)}
    controls = construct_controls(selection(("a", "b")), notes, filler_ids=[24], seed=17)
    record = dict(item_id="a", document_id="document:a", split="dev", prefix=prefix, future=future)
    rows = score_record(record, controls["a"], notes["a"], backend=backend, fmt=fmt, total_width=30)
    assert len(rows) == len(MODES)*len(HORIZONS) == 18
    for row in rows:
        assert row["status"] == "ok"
        content = controls["a"]["content_ids"][row["mode"]]
        context = prefix if content is None else list(fmt.header_ids)+prefix+list(fmt.open_ids)+content+list(fmt.close_ids)
        with torch.no_grad():
            logits = reader(torch.tensor([context+future])).logits[0, len(context)-1:-1]
            lp = logits.log_softmax(-1).gather(-1, torch.tensor(future)[:, None])[:, 0]
        expected = lp[:row["horizon"]]
        assert row["log_likelihood"] == pytest.approx(float(expected.sum()), abs=2e-5)
        assert row["token_log_likelihoods"] == pytest.approx(expected.tolist(), abs=1e-5)
        assert row["target_token_ids"] == future[:row["horizon"]]
        raw = next(r for r in rows if r["mode"] == "raw" and r["horizon"] == row["horizon"])
        assert row["gain_over_raw_nats"] == pytest.approx(row["log_likelihood"]-raw["log_likelihood"])


def test_failed_notes_retain_every_condition_without_imputing_zero_likelihood(fmt):
    backend = NeuronCausalBackend(model(), checkpoint_trunk=False)
    prefix = [10, 11, 12]
    selected = selection(("a", "b"))
    notes = {"a": note(prefix, [], fmt, error=True), "b": note(prefix, [7], fmt)}
    controls = construct_controls(selected, notes, filler_ids=[24], seed=17)
    rows = []
    for key in selected["item_ids"]:
        rows += score_record(dict(item_id=key, document_id="document:"+key, split="dev",
            prefix=prefix, future=[13]*16), controls[key], notes[key], backend=backend, fmt=fmt, total_width=29)
    assert len(rows) == 36
    for row in rows:
        failed = (row["item_id"] == "a" and row["mode"] not in {"raw", "empty"}) or (row["item_id"] == "b" and row["mode"] == "other")
        assert row["status"] == ("error" if failed else "ok")
        if failed:
            assert row["log_likelihood"] is None and row["gain_over_raw_nats"] is None
    result = summarize(rows, ["a", "b"])
    assert result["items"] == 2 and not result["failure_free"]
    assert result["conditions"]["own_K1"]["items"] == 2
    assert result["conditions"]["own_K1"]["successful_items"] == 1
    assert result["paired_document_contrasts"]["own_minus_raw_K1"]["result"] is None
    assert summarize(rows, ["a", "b"], policy="greedy")["primary"] is None
    with pytest.raises(ValueError, match="exactly once"):
        summarize(rows[:-1], ["a", "b"])


def write_bundle(directory, rows, selected, fmt, *, checkpoint=None):
    directory.mkdir()
    path = directory/"notes.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    protocol = dict(schema=SCHEMA, command="generate", selection_item_ids_sha256=selected["item_ids_sha256"],
        records_sha256=selected["records_sha256"], thought_format=json.loads(json.dumps(fmt.__dict__)), thought_cap=4)
    report = dict(status="complete", protocol=protocol, model=dict(sha256="base"), checkpoint=checkpoint,
                  notes_sha256=sha256_file(path))
    (directory/"run.json").write_text(json.dumps(report))


def test_saved_note_bundles_require_complete_coverage_exact_hashes_and_one_writer_identity(tmp_path, fmt):
    selected = selection(("a", "b"))
    rows = [dict(item_id=key, document_id="document:"+key, split="dev", **note([10, 11], [7], fmt)) for key in ("a", "b")]
    left, right = tmp_path/"left", tmp_path/"right"
    write_bundle(left, rows[:1], selected, fmt)
    write_bundle(right, rows[1:], selected, fmt)
    notes, identity = load_note_bundles([left, right], selected, fmt)
    assert set(notes) == {"a", "b"} and len(identity["bundles"]) == 2
    with pytest.raises(ValueError, match="every globally selected"):
        load_note_bundles([left], selected, fmt)
    with pytest.raises(ValueError, match="Duplicate"):
        load_note_bundles([left, right, left], selected, fmt)
    (left/"notes.jsonl").write_text((left/"notes.jsonl").read_text()+"\n")
    with pytest.raises(ValueError, match="content hash"):
        load_note_bundles([left, right], selected, fmt)
    (left/"notes.jsonl").write_text(json.dumps(rows[0])+"\n")
    report = json.loads((right/"run.json").read_text())
    report["checkpoint"] = dict(sha256="different-writer")
    (right/"run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="mix writer"):
        load_note_bundles([left, right], selected, fmt)


def test_final_protocol_gate_runs_before_any_selected_record_or_future_loading(tmp_path, fmt, monkeypatch):
    import train.evaluate_local as module
    selected = selection(tuple(f"test:{i}" for i in range(2048)), split="test")
    manifest, formatted = tmp_path/"selection.json", tmp_path/"format.json"
    manifest.write_text(json.dumps(selected))
    formatted.write_text(json.dumps(dict(format=fmt.__dict__)))
    monkeypatch.setattr(module, "asset_provenance", lambda _: dict(files={}))
    monkeypatch.setattr(module, "source_hashes", lambda: {})
    loaded = []
    monkeypatch.setattr(module, "load_selected", lambda *a, **kw: loaded.append(kw))
    monkeypatch.setattr(sys, "argv", ["evaluate_local", "generate", "--model-dir", str(tmp_path),
        "--records", str(tmp_path/"records.jsonl"), "--selection", str(manifest), "--format", str(formatted),
        "--output", str(tmp_path/"output"), "--arm", "B0"])
    with pytest.raises(ValueError, match="frozen-protocol"):
        main()
    assert loaded == [] and not (tmp_path/"output").exists()


def test_real_tiny_asset_cli_saves_notes_and_crosses_to_a_different_reader_without_resampling(tmp_path, fmt, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    assets = tmp_path/"model"
    reader = model()
    reader.save_pretrained(assets)
    tokenizer = Tokenizer(WordLevel({f"tok{i}": i for i in range(30)}, unk_token="tok1"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="tok0", unk_token="tok1")
    tokenizer.save_pretrained(assets)
    records, manifest, formatted = tmp_path/"records.jsonl", tmp_path/"ids.json", tmp_path/"format.json"
    records.write_text("".join(json.dumps(dict(id=key, document_id="document:"+key, split="dev",
        prefix=[10, 11, 12], future=[13, 14, 15, 16]*4))+"\n" for key in ("a", "b")))
    manifest.write_text(json.dumps(selection(("a", "b"), records_sha256=sha256_file(records))))
    formatted.write_text(json.dumps(dict(format=fmt.__dict__)))
    output_notes, output_scores = tmp_path/"notes", tmp_path/"scores"
    shared = ["--model-dir", str(assets), "--records", str(records), "--selection", str(manifest),
              "--format", str(formatted), "--device", "cpu", "--qualification", "--thought-cap", "4"]
    monkeypatch.setattr(sys, "argv", ["evaluate_local", "generate", *shared, "--output", str(output_notes)])
    main()
    notes_hash = sha256_file(output_notes/"notes.jsonl")
    generation = json.loads((output_notes/"run.json").read_text())
    assert generation["status"] == "complete" and generation["summary"]["statuses"] == {"ok": 2}
    generated = [json.loads(line) for line in (output_notes/"notes.jsonl").read_text().splitlines()]
    assert all("future" not in row and "first_target_token_id" not in row for row in generated)
    # A distinct, explicitly named reader checkpoint uses exactly the saved notes.
    with torch.no_grad():
        reader.model.layers[0].mlp.down_proj.weight.add_(.01)
    checkpoint = tmp_path/"reader.pt"
    torch.save(dict(model=reader.state_dict(), step=7, run_id="fixture-reader"), checkpoint)
    monkeypatch.setattr(sys, "argv", ["evaluate_local", "score", *shared, "--output", str(output_scores),
        "--notes", str(output_notes), "--checkpoint", str(checkpoint), "--filler-text", "tok24"])
    main()
    report = json.loads((output_scores/"run.json").read_text())
    assert report["status"] == "complete" and report["summary"]["failure_free"]
    assert report["summary"]["items"] == 2 and report["checkpoint"]["run_id"] == "fixture-reader"
    assert report["crossed_writer_reader"] and report["summary"]["primary"] is None
    contrast = report["summary"]["paired_document_contrasts"]["own_minus_raw_K1"]
    assert contrast["items"] == contrast["documents"] == 2 and len(contrast["ci95"]) == 2
    assert report["writer"]["checkpoint"] is None
    assert len((output_scores/"scores.jsonl").read_text().splitlines()) == 36
    scored = [json.loads(line) for line in (output_scores/"scores.jsonl").read_text().splitlines()]
    assert all(row["first_target_token_id"] == 13 and row["first_target_token_text"] == "tok13" for row in scored)
    assert sha256_file(output_notes/"notes.jsonl") == notes_hash
    mixed = [dict(row) for row in scored]
    mixed[0]["writer_model_identity_sha256"] = "another-writer"
    with pytest.raises(ValueError, match="mixed or partially tagged"):
        summarize(mixed, ["a", "b"])
    next(row for row in scored if row["mode"] == "own" and row["horizon"] == 1)["log_likelihood"] = float("nan")
    invalid = summarize(scored, ["a", "b"])
    assert not invalid["failure_free"] and invalid["conditions"]["own_K1"]["invalid_numeric_outcomes"] == 1
    assert invalid["paired_document_contrasts"]["own_minus_raw_K1"]["result"] is None
