"""Independent exact arithmetic and projection checks; no model evidence."""
import ast
from collections import Counter
import json
from fractions import Fraction
import re
from unittest.mock import patch

import pytest

from train import controlled_arithmetic_data as data
from weekend_thinking.artifacts import canonical_hash, sha256_file


def independent_steps(text):
    steps = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        assert isinstance(node, ast.BinOp)
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add):
            op, value = "+", left + right
        elif isinstance(node.op, ast.Sub):
            op, value = "-", left - right
        else:
            assert isinstance(node.op, ast.Mult)
            op, value = "*", left * right
        steps.append([int(left), op, int(right), int(value)])
        return value

    value = visit(ast.parse(text, mode="eval").body)
    return int(value), steps


@pytest.fixture(scope="module")
def rows():
    return data.build_rows()[0]


def test_complete_balanced_seed_deterministic_grammars_and_signs(rows):
    again, attempts = data.build_rows()
    assert rows == again and attempts
    assert len(rows) == len({r["item_id"] for r in rows}) == 256
    assert Counter(r["family"] for r in rows) == {family: 64 for family in data.FAMILIES}
    assert [r["family"] for r in rows[:8]] == list(data.FAMILIES) * 2
    for family in data.FAMILIES[1:3]:
        values = [r["value"] for r in rows if r["family"] == family]
        assert [v < 0 for v in values] == [False, True] * 32


def test_every_expression_and_postorder_step_independently_exact(rows):
    for row in rows:
        value, steps = independent_steps(row["expression"])
        assert (value, steps) == (row["value"], row["steps"])
        assert abs(value) >= 1000 and all(abs(s[-1]) > 1 for s in steps)
        assert len(steps) == (5 if row["family"] == "signed_three_products" else 3)
        tree = ast.parse(row["expression"], mode="eval")
        assert all(10 <= n.value <= 999 for n in ast.walk(tree) if isinstance(n, ast.Constant))
        assert row["target_text"] == f" {value}\n"
        assert row["prefix_text"] == f"Calculate {row['expression']} ="


def test_partial_omits_root_and_wrong_mutates_only_last_included_result(rows):
    pattern = re.compile(r"(-?\d+) ([+*-]) (-?\d+) = (-?\d+)\.")
    for row in rows:
        for name, text in row["supplied_notes"].items():
            partial, wrong = name.startswith("partial"), name.endswith("wrong") or name == "wrong"
            expected = row["steps"][:-1] if partial else row["steps"]
            actual = [[int(a), op, int(b), int(v)] for a, op, b, v in pattern.findall(text)]
            assert len(actual) == len(expected)
            assert actual[:-1] == expected[:-1] and actual[-1][:3] == expected[-1][:3]
            assert actual[-1][-1] == (data.wrong_value(expected[-1][-1]) if wrong else expected[-1][-1])
            if partial:
                assert "Result:" not in text and str(row["value"]) not in text
                assert all(step[-1] != row["value"] for step in actual)
            else:
                assert text.endswith(f"Result: {actual[-1][-1]}.")


@pytest.mark.parametrize("value", [2, 9, 10, 99, 100, 999, -2, -9, -10, -99, -100, -999])
def test_corruption_is_off_by_one_same_sign_same_digit_width(value):
    wrong = data.wrong_value(value)
    assert abs(wrong - value) == 1 and (wrong < 0) == (value < 0)
    assert len(str(abs(wrong))) == len(str(abs(value)))


def test_ast_exclusion_replaces_collision_without_changing_family_balance(rows):
    excluded = {data.canonical_expression(row["expression"]) for row in rows[:8]}
    alternate, _ = data.build_rows(excluded)
    assert not excluded & {data.canonical_expression(row["expression"]) for row in alternate}
    assert Counter(r["family"] for r in alternate) == Counter(r["family"] for r in rows)
    assert data.canonical_expression(" (12 × 20) − 3 =") == data.canonical_expression("(12 * 20) - 3")


def test_impossible_acceptance_fails_at_finite_construction_bound():
    with patch.object(data, "canonical_expression", return_value="excluded"), patch.object(data, "MAX_ATTEMPTS_PER_FAMILY", 7):
        with pytest.raises(ValueError, match="Fixed construction attempt limit"):
            data.build_rows({"excluded"})


class CharacterTokenizer:
    all_special_ids = [0]

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, **_):
        return list(text.encode("ascii"))

    def __call__(self, text, **_):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_native_preparation_keeps_all_surfaces_masks_and_future_free_projection(tmp_path):
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"split": "train", "reanchor_text": "(12*34)+56 ="}) + "\n")
    output = tmp_path / "prepared"
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
        manifest = data.prepare(output, tmp_path / "model", training)
    prepared = json.loads((output / "reader-inputs.json").read_text())["rows"]
    projection = [json.loads(line) for line in (output / "prefixes.jsonl").read_text().splitlines()]
    assert len(prepared) == len(projection) == manifest["items"] == 256
    assert manifest["reader_inputs_sha256"] == sha256_file(output / "reader-inputs.json")
    assert manifest["prefixes_sha256"] == sha256_file(output / "prefixes.jsonl")
    assert manifest["numeric_only_supported_items"] == 256
    for row, past in zip(prepared, projection):
        assert set(past) == {"item_id", "document_id", "split", "prefix_ids", "reanchor_ids"}
        assert past == {key: row[key] for key in past}
        assert row["item_id"] == canonical_hash(dict(namespace=data.SCHEMA, expression=row["expression"]))
        for name in ("prefix", "reanchor", "target"):
            assert row[name + "_ids"] == list(row[name + "_text"].encode("ascii"))
        kept = bytes(t for t, active in zip(row["target_ids"], row["numeric_only_mask"]) if active).decode()
        assert kept == str(abs(row["value"]))
        assert row["numeric_only_supported"] and len(row["target_token_offsets"]) == len(row["target_ids"])
        assert row["supplied_note_ids"] == {k: list(v.encode("ascii")) for k, v in row["supplied_notes"].items()}


def test_nontrain_exclusion_payload_rejected_before_publication(tmp_path):
    training = tmp_path / "nontrain.jsonl"
    training.write_text(json.dumps({"split": "dev", "reanchor_text": "1+2 ="}) + "\n")
    with pytest.raises(ValueError, match="Only approved TRAIN"):
        data.prepare(tmp_path / "out", tmp_path / "model", training)
    assert not (tmp_path / "out").exists()
