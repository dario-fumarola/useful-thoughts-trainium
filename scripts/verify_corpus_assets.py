#!/usr/bin/env python3
"""Verify prepared corpus bindings, every record, and sampled source anchors.

This is a data integrity receipt. It neither runs a model nor chooses outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Record
from weekend_thinking.corpus_audit import page_identity
from weekend_thinking.data import digest, make_record, split_document


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--benchmark-questions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite a verification receipt.")
    from transformers import AutoTokenizer

    manifest_path = args.records.with_suffix(".manifest.json")
    membership_path = args.records.with_suffix(".documents.jsonl")
    manifest = json.loads(manifest_path.read_text())
    source = json.loads(args.documents.with_suffix(".source.json").read_text())
    questions = json.loads(args.benchmark_questions.with_suffix(".manifest.json").read_text())
    asset = json.loads((args.model_dir / "ASSET_MANIFEST.json").read_text())
    require(manifest["document_identity"].startswith("host_path_and_sorted_content_query_v2"), "Require corrected v2 URL identity.")
    require(source["resolved_revision"] == manifest["source_revision"], "Raw-source revision mismatch.")
    require(source["sha256"] == manifest["input_sha256"] == sha256_file(args.documents), "Raw-document hash mismatch.")
    require(manifest["output_sha256"] == sha256_file(args.records), "Record hash mismatch.")
    require(manifest["document_membership_sha256"] == sha256_file(membership_path), "Membership hash mismatch.")
    require(manifest["benchmark_questions_sha256"] == questions["sha256"] == sha256_file(args.benchmark_questions), "Question hash mismatch.")
    require(questions["answer_columns_read"] is False, "Question acquisition read answer columns.")
    require(manifest["tokenizer_identity"] == asset["model_id"] + "@" + asset["resolved_revision"], "Tokenizer revision mismatch.")
    tokenizer_hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        if name in asset["files"]:
            actual = sha256_file(args.model_dir / name)
            require(actual == asset["files"][name]["sha256"], f"Tokenizer file changed: {name}")
            tokenizer_hashes[name] = actual
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 2**60
    vocabulary_size = json.loads((args.model_dir / "config.json").read_text())["vocab_size"]
    counts, ids, document_ids, train_ids, split_ids, samples = Counter(), set(), set(), [], defaultdict(list), {}
    maximum_token_id = 0
    with args.records.open() as stream:
        for line in stream:
            row = json.loads(line)
            row["prefix"], row["future"] = tuple(row["prefix"]), tuple(row["future"])
            record = Record(**row)
            require(record.id not in ids and record.document_id not in document_ids, "Duplicate record/document identity.")
            ids.add(record.id)
            document_ids.add(record.document_id)
            require(record.split == split_document(record.document_id, manifest["salt"]), "Document split mismatch.")
            require(len(record.prefix) == manifest["prefix_length"] and len(record.future) == manifest["future_length"], "Record length mismatch.")
            tokens = record.prefix + record.future
            require(not set(tokens).intersection(tokenizer.all_special_ids), "Special token in a retained record.")
            require(max(tokens) < vocabulary_size, "Token ID outside model vocabulary.")
            maximum_token_id = max(maximum_token_id, max(tokens))
            counts[record.split] += 1
            split_ids[record.split].append(record.id)
            if record.split == "train":
                train_ids.append(record.id)
            if counts[record.split] <= 16:
                samples[record.id] = record
    require(sum(counts.values()) == manifest["records"], "Record count mismatch.")
    require(all(counts[split] == manifest["counts"][split] for split in ("train", "dev", "test")), "Split counts mismatch.")
    membership = {}
    with membership_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["record_id"] in samples:
                membership[row["source_line"]] = row
    require(len(membership) == len(samples), "Missing sampled record membership.")
    compared = 0
    with args.documents.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in membership:
                continue
            source_row, member = json.loads(line), membership[line_number]
            content_hash = digest(source_row["text"])
            require(content_hash == member["content_sha256"], "Sample source-content hash mismatch.")
            doc_id = page_identity(source_row.get("url", ""), content_hash)
            tokens = tokenizer.encode(source_row["text"], add_special_tokens=False)
            expected = make_record(tokens, doc_id, domain=source_row.get("domain", "mathematics"),
                                   prefix_length=manifest["prefix_length"], future_length=manifest["future_length"], salt=manifest["salt"])
            require(expected == samples[member["record_id"]], "Source anchor differs from prepared record.")
            compared += 1
    require(compared == len(samples), "Not every sampled source anchor was verified.")
    order_hashes = {}
    for seed in (1234, 5678):
        ordered = sorted(train_ids, key=lambda identifier: hashlib.sha256(f"order:{seed}:{identifier}".encode()).hexdigest())
        order_hashes[str(seed)] = canonical_hash(ordered)
    receipt = {
        "created_utc": utc_now(), "passed": True, "scope": "data_integrity_only_no_model_outcomes",
        "records_file": str(args.records.resolve()), "records_sha256": manifest["output_sha256"],
        "preparation_manifest_sha256": sha256_file(manifest_path), "raw_sha256": source["sha256"],
        "source_revision": source["resolved_revision"], "source_license": source["license"],
        "question_manifest_sha256": sha256_file(args.benchmark_questions.with_suffix(".manifest.json")),
        "tokenizer_files_sha256": tokenizer_hashes, "tokenizer_identity": manifest["tokenizer_identity"],
        "counts": dict(counts), "maximum_token_id": maximum_token_id, "vocabulary_size": vocabulary_size,
        "sampled_source_anchors_verified": compared, "seed_order_hashes": order_hashes,
        "maximum_updates_at_batch_128": counts["train"] // 128,
        "fixed_first_512_dev_ids_sha256": canonical_hash(split_ids["dev"][:512]),
        "fixed_first_2048_test_ids_sha256": canonical_hash(split_ids["test"][:2048]),
        "audit_limitations": manifest["limitations"],
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
