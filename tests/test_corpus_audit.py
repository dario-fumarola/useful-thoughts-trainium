"""Corpus contamination and split-boundary regressions, without network access."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from weekend_thinking.corpus_audit import Components, NearDuplicateIndex, QuestionScreen, page_identity, shingle_hashes, words

ROOT = Path(__file__).resolve().parents[1]


def test_question_exclusion_handles_punctuation_and_short_questions():
    screen = QuestionScreen([
        {"task": "long", "question": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"},
        {"task": "short", "question": "How many blue marbles are there?"},
        {"task": "underspecified", "question": "Why?"},
    ])
    assert screen.matches(words("A page. ALPHA, beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu!")) == {"long"}
    assert screen.matches(words("Exercise: how many blue marbles are there? Explain your reasoning.")) == {"short"}
    assert screen.matches(words("How many red marbles are there?")) == set()
    assert screen.counts["questions_below_minimum_short_words"] == 1
    with pytest.raises(ValueError):
        QuestionScreen([])


def test_near_duplicate_index_verifies_real_jaccard_and_rejects_distinct_text():
    original = [f"word{i}" for i in range(500)]
    slightly_changed = original[:-20] + [f"replacement{i}" for i in range(20)]
    unrelated = [f"unrelated{i}" for i in range(500)]
    index = NearDuplicateIndex(0.8)
    assert index.add(0, shingle_hashes(original)) == []
    matches = index.add(1, shingle_hashes(slightly_changed))
    assert len(matches) == 1 and matches[0][0] == 0 and matches[0][1] > 0.9
    assert index.add(2, shingle_hashes(unrelated)) == []


def test_components_preserve_transitive_identity_before_splitting():
    groups = Components()
    for _ in range(5):
        groups.add()
    groups.union(3, 4)
    groups.union(1, 2)
    groups.union(2, 4)
    assert [groups.find(i) for i in range(5)] == [0, 1, 1, 1, 1]
    assert page_identity("https://EXAMPLE.org/page/?utm_source=mail", "a") == page_identity("http://example.org/page", "b")


def test_content_bearing_query_parameters_distinguish_documents():
    assert page_identity("https://example.org/display.cgi?id=1", "same") != page_identity("https://example.org/display.cgi?id=2", "same")
    assert page_identity("https://example.org/wiki?title=Alpha", "same") != page_identity("https://example.org/wiki?title=Beta", "same")
    assert page_identity("https://example.org/thread?t=1&page=2&utm_campaign=x", "a") == page_identity("http://EXAMPLE.org/thread?page=2&t=1", "b")


def test_lambada_exclusion_projection_omits_final_target():
    spec = importlib.util.spec_from_file_location("fetch_exclusion_questions", ROOT / "scripts/fetch_exclusion_questions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.question_projection("The target word is hidden.  ", lambada=True) == "The target word is"
    with pytest.raises(ValueError):
        module.question_projection("target", lambada=True)


def test_full_pipeline_excludes_every_version_of_contaminated_page(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0, "[SPECIAL]": 1, **{f"term{i}": i+2 for i in range(100)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=["[SPECIAL]"])
    token_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(token_dir)
    question = " ".join(f"term{i}" for i in range(13))
    contaminated = question + " " + " ".join(f"term{i}" for i in range(13, 30))
    alternate = " ".join(f"term{i}" for i in range(30, 60))
    clean = " ".join(f"term{i}" for i in range(60, 90))
    docs = [
        {"text": contaminated, "url": "https://example.org/contaminated?utm_source=one"},
        {"text": alternate, "url": "https://example.org/contaminated?utm_source=two"},
        {"text": clean, "url": "https://example.org/clean"},
        {"text": clean, "url": "https://another.example/duplicate"},
        {"text": alternate + " [SPECIAL]", "url": "https://example.org/special"},
    ]
    raw, questions, out = tmp_path / "raw.jsonl", tmp_path / "questions.jsonl", tmp_path / "records.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in docs))
    questions.write_text(json.dumps({"question": question, "task": "fixture"}) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_audited_corpus.py"), "--documents", str(raw),
        "--benchmark-questions", str(questions), "--out", str(out), "--tokenizer-dir", str(token_dir),
        "--tokenizer-identity", "fixture-only", "--source-revision", "fixture-only", "--workers", "1",
        "--prefix-length", "8", "--future-length", "2",
    ], check=True, capture_output=True, text=True, timeout=60)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert len(records) == 1
    assert manifest["counts"]["benchmark_excluded_component_documents"] == 3
    assert manifest["counts"]["exact_duplicate_edges"] == 1
    assert len(records[0]["prefix"]) == 8 and len(records[0]["future"]) == 2
    assert manifest["benchmark_screen"]["outcomes_read"] is False
