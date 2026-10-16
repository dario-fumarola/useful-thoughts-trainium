"""Meaningful evaluator semantics on CPU fixtures; no accelerator claim."""
import json
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.evaluate import (Decoded, GenerationFailure, GreedyDecoder, first_answer_line_end,
                            generate_item, load_own_checkpoint, question_prompt, read_questions,
                            require_frozen_protocol, validate_prompt_spec)
from weekend_thinking.core import Format, policy_log_probs
from weekend_thinking.evaluation import score_predictions


def tiny_model():
    torch.manual_seed(92)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=3,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        attn_implementation="eager")).eval()


class Tokenizer:
    def encode(self, text, **kwargs):
        return [10, 11, 12]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def test_fixed_padded_cache_matches_uncached_contiguous_generation():
    model = tiny_model()
    decoder = GreedyDecoder(model, Tokenizer(), banned_ids=(29, 30))
    result = decoder.decode([10, 11, 12], max_new_tokens=5, prefill_width=9,
                            pad_id=0, stop_ids=(), answer_line_stop=False)
    prefix = [10, 11, 12]
    expected = []
    with torch.no_grad():
        for _ in range(5):
            logits = model(torch.tensor([prefix])).logits[:, -1]
            value = int(policy_log_probs(logits, 1.0, (29, 30)).argmax(-1)[0])
            expected.append(value)
            prefix.append(value)
    assert result.actions == expected
    assert not set(result.actions) & {29, 30}
    assert result.stop_type == "length_cap"
    # Actual prompt length changes, while fixed prefill/decode widths remain.
    second = decoder.decode([10, 11, 12, 13], max_new_tokens=5, prefill_width=9,
                            pad_id=0, stop_ids=(), answer_line_stop=False)
    unpadded = decoder.decode([10, 11, 12, 13], max_new_tokens=5, prefill_width=4,
                              pad_id=0, stop_ids=(), answer_line_stop=False)
    assert second.actions == unpadded.actions


def test_sampled_stop_is_counted_and_excluded_from_content():
    model = tiny_model()
    decoder = GreedyDecoder(model, Tokenizer())
    first = decoder.decode([10, 11], max_new_tokens=1, prefill_width=5,
                           pad_id=0, stop_ids=(), answer_line_stop=False).actions[0]
    stopped = decoder.decode([10, 11], max_new_tokens=5, prefill_width=5,
                             pad_id=0, stop_ids=(first,), answer_line_stop=False)
    assert stopped.actions == [first]
    assert stopped.content_ids == [] and stopped.text == ""
    assert stopped.stop_type == f"token:{first}"


def test_answer_line_stop_waits_for_complete_number_and_does_not_use_gold():
    assert first_answer_line_end("Compute.\n#### 1") is None
    assert first_answer_line_end("Compute.\n#### 1.25") is None
    text = "Compute.\n#### 1.25\n\nQuestion: another problem"
    assert text[:first_answer_line_end(text)] == "Compute.\n#### 1.25\n"
    assert first_answer_line_end("#### 2/3\n") is None
    assert first_answer_line_end("### 42\n") is None


def test_question_input_and_demonstrations_enforce_gold_boundary(tmp_path):
    path = tmp_path / "questions.jsonl"
    q = dict(item_id="gsm8k:train:00002", task="gsm8k", split="train", question="How many?")
    path.write_text(json.dumps(q) + "\n")
    assert read_questions(path, qualification=True)[0]["question"] == "How many?"
    with pytest.raises(ValueError, match="test"):
        read_questions(path, qualification=False)
    path.write_text(json.dumps(dict(q, answer="#### 3")) + "\n")
    with pytest.raises(ValueError, match="questions only"):
        read_questions(path, qualification=True)
    spec = dict(instruction="Finish with #### <number>.", demonstrations=[
        dict(item_id="gsm8k:train:00000", split="train", question="One plus one?", answer="1+1=2.\n#### 2")])
    validate_prompt_spec(spec, {q["item_id"]})
    text = question_prompt(q["question"], spec)
    assert text.endswith("Question: How many?\nAnswer:")
    assert "1+1=2.\n#### 2" in text
    with pytest.raises(ValueError, match="disjoint"):
        validate_prompt_spec(spec, {"gsm8k:train:00000"})
    spec["demonstrations"][0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        validate_prompt_spec(spec, {q["item_id"]})


class ScriptedDecoder:
    tokenizer = Tokenizer()

    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def decode(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def item_call(decoder, mode="direct"):
    return generate_item(dict(item_id="gsm8k:train:00002", document_id="d2", split="train", question="How many?"),
        decoder=decoder, prompt_spec=dict(instruction="Give #### <number>.", demonstrations=[]),
        fmt=Format((2,), (3, 4), 5, 0, header_ids=(9,), additional_stop_ids=(0,)),
        mode=mode, max_answer_tokens=8, thought_cap=4,
        question_prefill_width=3, answer_prefill_width=11 if mode == "one_thought" else 3)


def test_one_thought_compacts_content_then_close_and_keeps_raw_ids():
    decoder = ScriptedDecoder([Decoded([7, 5], [7], "calculation", "calculation", "token:5"),
                               Decoded([20, 21], [20, 21], "#### 8\n", "#### 8\n", "answer_line")])
    row = item_call(decoder, "one_thought")
    assert decoder.calls[0][0] == [9, 10, 11, 12, 2]
    assert decoder.calls[1][0] == [9, 10, 11, 12, 2, 7, 3, 4, 10, 11, 12]
    assert row["reader_suffix"].endswith("Question: How many?\nAnswer:")
    assert row["thought_actions"] == [7, 5] and row["answer_actions"] == [20, 21]
    assert row["n_generated_tokens"] == 4 and row["n_thought_tokens"] == 2
    assert row["status"] == "ok" and row["generated_text"] == "#### 8\n"


def test_empty_format_retains_interface_without_generating_thought_actions():
    decoder = ScriptedDecoder([Decoded([20], [20], "#### 8\n", "#### 8\n", "answer_line")])
    row = item_call(decoder, "empty_format")
    assert len(decoder.calls) == 1
    assert decoder.calls[0][0] == [9, 10, 11, 12, 2, 3, 4, 10, 11, 12]
    assert row["thought_actions"] == [] and row["thought_stop_type"] == "empty_control"
    assert row["n_thought_tokens"] == row["thought_cap"] == 0


def test_truncation_and_execution_failures_are_preserved_for_strict_scoring():
    capped = item_call(ScriptedDecoder([Decoded([7], [7], "#### 8", "#### 8", "length_cap")]))
    failure = GenerationFailure(RuntimeError("device failure"), Decoded([7], [7], "partial", "partial", "error"))
    errored = item_call(ScriptedDecoder([failure]))
    malformed = item_call(ScriptedDecoder([Decoded([0], [], "", "", "token:0")]))
    assert [capped["status"], errored["status"], malformed["status"]] == ["truncated", "error", "format_error"]
    assert errored["answer_actions"] == [7] and errored["n_generated_tokens"] == 1
    predictions = []
    for index, row in enumerate((capped, errored, malformed)):
        row["item_id"] = str(index)
        predictions.append(row)
    scores, summary = score_predictions([dict(item_id=str(i), answer="#### 8") for i in range(3)], predictions)
    assert len(scores) == summary["items"] == 3 and summary["accuracy"] == 0.0


def test_full_own_checkpoint_loading_and_conflicting_ties(tmp_path):
    model = tiny_model()
    saved = {k: t.clone() for k, t in model.state_dict().items()}
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight.add_(1.0)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(model=saved, step=5, run_id="own-run"), checkpoint)
    provenance = load_own_checkpoint(model, checkpoint)
    assert provenance["step"] == 5 and provenance["sha256"]
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved[key])
    saved["lm_head.weight"].add_(1.0)
    torch.save(dict(model=saved), checkpoint)
    with pytest.raises(ValueError, match="conflicting tied"):
        load_own_checkpoint(model, checkpoint)


def test_final_protocol_receipt_must_match_exact_configuration(tmp_path):
    protocol = dict(mode="one_thought", thought_cap=1024, max_answer_tokens=512)
    with pytest.raises(ValueError, match="frozen-protocol"):
        require_frozen_protocol(None, protocol)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(dict(status="frozen", frozen_utc="2026-10-16T00:00:00Z", protocols=[protocol])))
    require_frozen_protocol(path, protocol)
    with pytest.raises(ValueError, match="does not match"):
        require_frozen_protocol(path, dict(protocol, thought_cap=2048))
