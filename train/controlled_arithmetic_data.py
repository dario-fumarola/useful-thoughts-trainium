"""Prospective synthetic arithmetic diagnostic, separate from real-data endpoints.

No model scores enter construction or acceptance. Supplied complete notes are
answer-copy controls. Partial notes omit the root operation and its answer.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import random

from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

SCHEMA = "controlled-compound-arithmetic256-v1"
SEED, N = 20261018, 256
MAX_ATTEMPTS_PER_FAMILY = 100_000
FAMILIES = ("sum_products", "product_sum_difference", "signed_three_products", "nested_product")


def expression(node):
    if isinstance(node, int):
        return str(node)
    op, left, right = node
    return f"({expression(left)} {op} {expression(right)})"


def solve(node):
    """Return the exact integer and postorder (left, op, right, result) steps."""
    if isinstance(node, int):
        return node, []
    op, left, right = node
    a, before = solve(left)
    b, after = solve(right)
    result = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[op]()
    return result, before + after + [(a, op, b, result)]


def tree(family, leaves):
    a, b, c, d, e, f = leaves
    return (
        ("+", ("*", a, b), ("*", c, d)),
        ("*", ("+", a, b), ("-", c, d)),
        ("+", ("-", ("*", a, b), ("*", c, d)), ("*", e, f)),
        ("*", ("+", ("*", a, b), c), d),
    )[family]


def wrong_value(value):
    """Off-by-one, preserving sign and the number of decimal digits."""
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    return sign * (magnitude + (1 if magnitude % 10 < 9 else -1))


def note(steps, *, partial=False, corrupt=False):
    selected = list(steps[:-1] if partial else steps)
    if not selected:
        raise ValueError("Partial notes require a proper intermediate operation")
    if corrupt:
        a, op, b, result = selected[-1]
        selected[-1] = (a, op, b, wrong_value(result))
    text = "Compute: " + " ".join(f"{a} {op} {b} = {value}." for a, op, b, value in selected)
    return text if partial else text + f" Result: {selected[-1][-1]}."


def canonical_expression(text):
    text = text.strip().removesuffix("=").strip().replace("×", "*").replace("−", "-")
    try:
        return ast.dump(ast.parse(text, mode="eval"), include_attributes=False)
    except (SyntaxError, ValueError):
        return None


def build_rows(excluded=()):
    """64 per fixed grammar; signs alternate in the two signed families."""
    rng = random.Random(SEED)
    groups = [[] for _ in FAMILIES]
    seen = set(excluded)
    attempts = Counter()
    for family in range(4):
        while len(groups[family]) < 64:
            attempts[FAMILIES[family]] += 1
            if attempts[FAMILIES[family]] > MAX_ATTEMPTS_PER_FAMILY:
                raise ValueError(f"Fixed construction attempt limit reached: {FAMILIES[family]}")
            leaves = [rng.randint(10, 99) if rng.randrange(2) == 0 else rng.randint(100, 999) for _ in range(6)]
            node = tree(family, leaves)
            rendered = expression(node)
            value, steps = solve(node)
            key = canonical_expression(rendered)
            partial = note(steps, partial=True)
            if (key in seen or abs(value) < 1000 or any(abs(s[-1]) <= 1 for s in steps)
                    or str(value) in partial):
                continue
            if family in (1, 2) and (value < 0) != bool(len(groups[family]) % 2):
                continue
            seen.add(key)
            groups[family].append(dict(
                family=FAMILIES[family], expression=rendered, value=value, steps=[list(s) for s in steps],
                prefix_text=f"Calculate {rendered} =", reanchor_text=f"{rendered} =", target_text=f" {value}\n",
                supplied_notes={"correct": note(steps), "wrong": note(steps, corrupt=True),
                                "partial_correct": partial, "partial_wrong": note(steps, partial=True, corrupt=True)},
            ))
    rows = [groups[family][i] for i in range(64) for family in range(4)]
    for row in rows:
        row["item_id"] = row["document_id"] = canonical_hash(dict(namespace=SCHEMA, expression=row["expression"]))
        row["split"] = "synthetic_dev"
    return rows, dict(attempts)


def prefix_projection(row):
    return {key: row[key] for key in ("item_id", "document_id", "split", "prefix_ids", "reanchor_ids")}


def prepare(output, model_dir, training_rows):
    from transformers import AutoTokenizer
    output, model_dir, training_rows = map(Path, (output, model_dir, training_rows))
    excluded = set()
    with training_rows.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("Only approved TRAIN expressions may be excluded")
            value = canonical_expression(row["reanchor_text"])
            if value is not None:
                excluded.add(value)
    rows, attempts = build_rows(excluded)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    for row in rows:
        for name in ("prefix", "reanchor", "target"):
            row[name + "_ids"] = enc(row[name + "_text"])
        if row["prefix_ids"] + row["target_ids"] != enc(row["prefix_text"] + row["target_text"]):
            raise ValueError("Exact full-surface tokenization boundary required")
        native = tok(row["target_text"], add_special_tokens=False, return_offsets_mapping=True)
        if native["input_ids"] != row["target_ids"]:
            raise ValueError("Target offset tokenization must match native target IDs")
        row["target_token_offsets"] = [list(span) for span in native["offset_mapping"]]
        digits_start = 1 + int(row["value"] < 0)
        digits_end = len(row["target_text"]) - 1
        row["numeric_only_mask"] = [
            digits_start <= start < end <= digits_end
            for start, end in row["target_token_offsets"]
        ]
        row["numeric_only_supported"] = any(row["numeric_only_mask"])
        row["supplied_note_ids"] = {key: enc(text) for key, text in row["supplied_notes"].items()}
        if any(any(tok_id in tok.all_special_ids for tok_id in ids) for ids in row["supplied_note_ids"].values()):
            raise ValueError("Ordinary native note text only")
    if len(rows) != N or len({row["item_id"] for row in rows}) != N:
        raise ValueError("Exactly256 unique frozen expressions required")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "reader-inputs.json", {"rows": rows})
    with (output / "prefixes.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(prefix_projection(row)) + "\n")
    manifest = dict(
        schema=SCHEMA, status="FROZEN_BEFORE_ANY_MODEL_OUTCOME", created_utc=utc_now(), seed=SEED,
        items=N, families=dict(Counter(row["family"] for row in rows)),
        sign_counts=dict(Counter("negative" if row["value"] < 0 else "positive" for row in rows)),
        construction_attempts=attempts, parsed_training_expression_exclusions=len(excluded),
        construction_attempt_limit_per_family=MAX_ATTEMPTS_PER_FAMILY,
        numeric_only_supported_items=sum(row["numeric_only_supported"] for row in rows),
        source_files={"train/controlled_arithmetic_data.py": sha256_file(__file__)},
        input_hashes={str(training_rows.resolve()): sha256_file(training_rows)},
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()), model_dir=str(model_dir.resolve()),
        reader_inputs_sha256=sha256_file(output / "reader-inputs.json"),
        prefixes_sha256=sha256_file(output / "prefixes.jsonl"),
        ordered_item_ids=[row["item_id"] for row in rows],
        limitations=[
            "Synthetic secondary diagnostic; does not replace real-document DEV or GSM8K TRAIN endpoints.",
            "No model-based hardness filter or replacement; actual raw headroom must be reported for every item.",
            "Full notes contain the answer: answer-copy plumbing, not self-generated computation evidence.",
            "Partial notes omit the root operation; corrupt notes introduce a deterministic off-by-one result.",
            "Numeric-only mask is descriptive: only native tokens wholly inside the integer digit span; sign and boundary bytes are excluded.",
            "Fixed grammar/seed sampling uncertainty is not natural-document or training-seed uncertainty.",
        ],
    )
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model-dir", "training-rows"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.model_dir, args.training_rows), indent=2))


if __name__ == "__main__":
    main()
