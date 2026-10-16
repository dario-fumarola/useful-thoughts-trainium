"""Secondary likelihood semantics on CPU fixtures; no accelerator claims."""
from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.evaluate import Decoded
from train.evaluate_likelihood import (ARC_DEVELOPMENT_SHA256, arc_context,
    evaluate_item, lambada_parts, prepare_arc_development, read_gold, read_items,
    score_continuations, summarize)
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import sha256_file
from weekend_thinking.core import Format


def tiny_model():
    torch.manual_seed(125)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=3,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attn_implementation="eager")).eval()


def test_static_padding_scores_every_real_token_for_unequal_continuations():
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=False)
    prefix, targets = [10, 11, 12], [[13], [14, 15, 16], [17, 18, 19, 20, 21]]
    actual = score_continuations(backend, prefix, targets, total_width=15,
                                future_width=7, pad_id=0, head_chunk=2)
    with torch.no_grad():
        for values, result in zip(targets, actual):
            ids = torch.tensor([prefix+values])
            logits = model(ids).logits[0, len(prefix)-1:len(prefix)+len(values)-1]
            reference = logits.log_softmax(-1).gather(-1, torch.tensor(values)[:, None])[:, 0]
            assert torch.allclose(torch.tensor(result.token_log_likelihoods), reference, atol=1e-5)
            assert result.log_likelihood == pytest.approx(float(reference.sum()), abs=1e-5)
            assert result.mean_token_log_likelihood == pytest.approx(float(reference.mean()), abs=1e-5)
            assert result.greedy_token_ids == logits.argmax(-1).tolist()
    # Different padding widths do not enter any real continuation's causal past.
    padded = score_continuations(backend, prefix, targets, total_width=19,
                                future_width=9, pad_id=0, head_chunk=2)
    for first, second in zip(actual, padded):
        assert first.log_likelihood == pytest.approx(second.log_likelihood, abs=1e-5)


def test_empty_or_truncated_targets_and_reduced_precision_are_rejected():
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=False)
    for target in ([], [10, 11, 12, 13]):
        with pytest.raises(ValueError, match="continuation"):
            score_continuations(backend, [10], [target], total_width=7, future_width=3, pad_id=0)
    model.half()
    with pytest.raises(ValueError, match="FP32"):
        score_continuations(backend, [10], [[11]], total_width=7, future_width=3, pad_id=0)


class Tokenizer:
    def __init__(self, values):
        self.values = values

    def encode(self, text, **kwargs):
        return self.values[text]


def test_lambada_accuracy_requires_all_subtokens_not_just_the_first():
    model = tiny_model()
    prefix = [10, 11]
    with torch.no_grad():
        first = int(model(torch.tensor([prefix])).logits[0, -1].argmax())
        second = int(model(torch.tensor([prefix+[first]])).logits[0, -1].argmax())
    target = [first, (second+1) % 31]
    row = evaluate_item(dict(item_id="fixture", task="lambada", split="synthetic", context="prefix"),
        dict(target=" multi"), backend=NeuronCausalBackend(model, checkpoint_trunk=False), decoder=None,
        tokenizer=Tokenizer({"prefix": prefix, " multi": target}), fmt=Format((2,), (3,), 5, 0),
        mode="direct", max_context_tokens=8, thought_cap=4, total_width=12, future_width=4)
    assert row["status"] == "ok" and row["correct"] == 0
    assert row["n_target_tokens"] == 2 and row["correct_target_tokens"] == 1
    assert len(row["continuations"][0]["token_log_likelihoods"]) == 2
    assert lambada_parts("The final word has punctuation.") == ("The final word has", " punctuation.")


def test_arc_reports_raw_and_token_mean_rankings_when_they_disagree():
    model = tiny_model()
    prefix = [10, 11]
    with torch.no_grad():
        initial = model(torch.tensor([prefix])).logits[0, -1]
        short = [int(initial.argmin())]
        long = []
        for _ in range(3):
            long.append(int(model(torch.tensor([prefix+long])).logits[0, -1].argmax()))
    result = evaluate_item(dict(item_id="arc:fixture", task="arc_challenge", split="train",
        context="question", choices=[" short", " longer"]), dict(gold=1),
        backend=NeuronCausalBackend(model, checkpoint_trunk=False), decoder=None,
        tokenizer=Tokenizer({"question": prefix, " short": short, " longer": long}),
        fmt=Format((2,), (3,), 5, 0), mode="direct", max_context_tokens=8,
        thought_cap=4, total_width=12, future_width=4)
    assert result["status"] == "ok"
    assert result["prediction"] == 0 and result["prediction_norm"] == 1
    assert result["correct"] == 0 and result["correct_norm"] == 1


class ScriptedDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, prefix, **kwargs):
        self.calls.append((list(prefix), kwargs))
        return Decoded([7, 5], [7], "note", "note", "token:5")


def test_one_prefix_only_note_is_fixed_before_any_candidate_or_gold_scoring():
    model = tiny_model()
    backend, decoder = NeuronCausalBackend(model, checkpoint_trunk=False), ScriptedDecoder()
    tokenizer = Tokenizer({"question": [10, 11], " first": [12], " second": [13, 14]})
    fmt = Format((2,), (3, 4), 5, 0, header_ids=(9,))
    results = []
    for choices, gold in [([" first", " second"], 0), ([" second", " first"], 1)]:
        results.append(evaluate_item(dict(item_id="fixture", task="arc_challenge", split="train",
            context="question", choices=choices), dict(gold=gold), backend=backend, decoder=decoder,
            tokenizer=tokenizer, fmt=fmt, mode="one_thought", max_context_tokens=8,
            thought_cap=4, total_width=20, future_width=4))
    assert len(decoder.calls) == 2  # One per question, never one per choice.
    assert decoder.calls[0][0] == decoder.calls[1][0] == [9, 10, 11, 2]
    assert all(row["status"] == "ok" for row in results)
    assert results[0]["scoring_context_ids"] == results[1]["scoring_context_ids"] == [9, 10, 11, 2, 7, 3, 4]
    assert results[0]["continuations"][0] == results[1]["continuations"][1]


def test_question_gold_and_official_lambada_test_boundaries(tmp_path):
    path = tmp_path/"questions.jsonl"
    arc = dict(item_id="arc:1", task="arc_challenge", split="train", context="q", choices=[" a", " b"])
    path.write_text(json.dumps(arc)+"\n")
    assert read_items(path, task="arc_challenge", qualification=True) == [arc]
    path.write_text(json.dumps(dict(arc, gold=0))+"\n")
    with pytest.raises(ValueError, match="gold separate"):
        read_items(path, task="arc_challenge", qualification=True)
    path.write_text(json.dumps(dict(item_id="lambada:1", task="lambada", split="test", context="prefix"))+"\n")
    with pytest.raises(ValueError, match="synthetic"):
        read_items(path, task="lambada", qualification=True)
    gold = tmp_path/"gold.jsonl"
    gold.write_text(json.dumps(dict(item_id="wrong-id", gold=0))+"\n")
    with pytest.raises(ValueError, match="item sets differ"):
        read_gold(gold, [arc], task="arc_challenge")


def test_execution_failures_remain_in_accuracy_denominator():
    successful = dict(item_id="ok", status="ok", correct=1, correct_norm=1,
                      gold_log_likelihood=-2.5, n_thought_tokens=0)
    failure = evaluate_item(dict(item_id="failure", task="lambada", split="synthetic", context="prefix"),
        dict(target=" oversize"), backend=NeuronCausalBackend(tiny_model(), checkpoint_trunk=False), decoder=None,
        tokenizer=Tokenizer({"prefix": [10, 11], " oversize": [12, 13, 14]}),
        fmt=Format((2,), (3,), 5, 0), mode="direct", max_context_tokens=8,
        thought_cap=4, total_width=10, future_width=2)
    assert failure["status"] == "error" and failure["failed_stage"] == "likelihood"
    summary = summarize([successful, failure])
    assert summary["items"] == 2 and summary["successful_items"] == 1
    assert summary["accuracy"] == summary["accuracy_token_mean_normalized"] == 0.5
    assert summary["mean_gold_log_likelihood_successful_only"] == -2.5


def test_arc_development_projection_preserves_labels_and_leading_space(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = tmp_path/"train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist([dict(id="fixture", question="What is the answer?",
        choices={"label": ["1", "2", "3"], "text": ["one", "two", "three"]}, answerKey="2")]), source)
    # A fixture hash is explicitly substituted; no fixture is claimed upstream data.
    monkeypatch.setitem(ARC_DEVELOPMENT_SHA256, "train", sha256_file(source))
    output = tmp_path/"projection"
    prepare_arc_development(source, output, split="train")
    questions = [json.loads(line) for line in (output/"questions.jsonl").read_text().splitlines()]
    gold = json.loads((output/"gold.jsonl").read_text())
    assert questions[0]["context"] == arc_context("What is the answer?")
    assert questions[0]["choices"] == [" one", " two", " three"]
    assert not {"gold", "answerKey", "target"} & questions[0].keys()
    assert gold["gold"] == 1
    with pytest.raises(ValueError, match="restricted"):
        prepare_arc_development(tmp_path/"test-00000-of-00001.parquet", tmp_path/"forbidden", split="test")
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="pinned Parquet hash"):
        prepare_arc_development(source, tmp_path/"tampered", split="train")
