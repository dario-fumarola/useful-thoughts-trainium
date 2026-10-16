#!/usr/bin/env python3
"""Project the pinned GSM8K test source after training configuration freeze.

Writes canonical question/gold files and the scoring manifest to add to the
final evaluation-freeze-v2 receipt before any official generation or scoring.
No answer values are printed. Never run this for prompt qualification.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.evaluation_integrity import PARSER, evaluation_dependencies
from weekend_thinking.artifacts import atomic_bytes, atomic_json, canonical_hash, sha256_file
from weekend_thinking.numeric_projection import GSM8K_REVISION, GSM8K_TEST_SHA256, jsonl_bytes, project_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-source", type=Path, required=True)
    parser.add_argument("--frozen-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.frozen_experiment.read_text())
    if (spec.get("status") != "FROZEN_ASSETS_NOT_HARDWARE_QUALIFIED" or not spec.get("spec_hash")
            or canonical_hash({k:v for k,v in spec.items() if k != "spec_hash"}) != spec["spec_hash"]):
        parser.error("A valid pre-existing frozen training specification is required before test gold projection")
    if sha256_file(args.canonical_source) != GSM8K_TEST_SHA256:
        parser.error("Canonical source must be the exact pinned main/test GSM8K Parquet")
    import pyarrow.parquet as pq
    questions, gold = project_rows(pq.read_table(args.canonical_source, columns=["question", "answer"]).to_pylist())
    args.output.mkdir(parents=True, exist_ok=False)
    q, g = args.output/"questions.jsonl", args.output/"gold.jsonl"
    atomic_bytes(q, jsonl_bytes(questions))
    atomic_bytes(g, jsonl_bytes(gold))
    manifest = dict(schema="numeric-scoring-v1", task="gsm8k", task_revision=GSM8K_REVISION,
        split="test", canonical_source_file=str(args.canonical_source.resolve()), canonical_source_sha256=GSM8K_TEST_SHA256,
        question_file=str(q.resolve()), question_file_sha256=sha256_file(q), gold_file_sha256=sha256_file(g),
        item_ids_sha256=canonical_hash([row["item_id"] for row in questions]), items=len(questions),
        parser=PARSER, dependencies=evaluation_dependencies(), training_spec_hash=spec["spec_hash"])
    atomic_json(args.output/"scoring-manifest.json", manifest)
    print(json.dumps(dict(items=len(questions), manifest=str(args.output/"scoring-manifest.json"))))


if __name__ == "__main__":
    main()
