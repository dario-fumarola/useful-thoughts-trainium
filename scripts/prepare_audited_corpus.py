#!/usr/bin/env python3
"""Screen, cluster, tokenize, and prepare one fixed anchor per raw document.

All cluster members are grouped before splitting. Any component containing a
benchmark overlap or a literal tokenizer special token is excluded in full.
Raw data and tokenized records stay in ignored local storage.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.corpus_audit import Components, NearDuplicateIndex, QuestionScreen, page_identity, shingle_hashes, words
from weekend_thinking.data import digest, make_record, write_records

TOKENIZER = SCREEN = OPTIONS = None


def initialize(tokenizer_dir: str, questions_path: str, options: dict):
    global TOKENIZER, SCREEN, OPTIONS
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    # Full documents are tokenized only to select a 528-token window, never fed
    # to a model. Avoid misleading model-context warnings for long source pages.
    TOKENIZER.model_max_length = 2**60
    with open(questions_path) as source:
        SCREEN = QuestionScreen([json.loads(line) for line in source if line.strip()])
    OPTIONS = options


def features(item: tuple[int, str]) -> dict:
    index, line = item
    row = json.loads(line)
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"source_line": index + 1, "empty": True}
    normalized = words(text)
    if not normalized:
        return {"source_line": index + 1, "empty": True}
    content_hash = digest(text)
    document_id = page_identity(row.get("url", ""), content_hash)
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    specials = sorted(set(tokens).intersection(TOKENIZER.all_special_ids))
    record = make_record(tokens, document_id, domain=row.get("domain", "mathematics"),
                         prefix_length=OPTIONS["prefix_length"], future_length=OPTIONS["future_length"],
                         salt=OPTIONS["salt"])
    return {"source_line": index + 1, "document_id": document_id, "url": row.get("url", ""),
            "content_sha256": content_hash, "token_count": len(tokens), "word_count": len(normalized),
            "special_token_ids": specials, "benchmark_tasks": sorted(SCREEN.matches(normalized)),
            "record": record, "shingles": shingle_hashes(normalized)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--benchmark-questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-identity", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=512)
    parser.add_argument("--future-length", type=int, default=16)
    parser.add_argument("--salt", default="weekend-2026-10-16-v1")
    parser.add_argument("--near-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("Positive worker count required.")
    manifest_path = args.out.with_suffix(".manifest.json")
    membership_path = args.out.with_suffix(".documents.jsonl")
    if any(path.exists() for path in (args.out, manifest_path, membership_path)):
        raise SystemExit("Refusing to overwrite records or their audit artifacts.")
    options = {"prefix_length": args.prefix_length, "future_length": args.future_length, "salt": args.salt}
    start = time.monotonic()
    index, components = NearDuplicateIndex(args.near_jaccard), Components()
    counts, benchmark_counts = Counter(), Counter()
    docs, content_seen, page_seen = [], {}, {}
    with args.documents.open() as source, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(str(args.tokenizer_dir.resolve()), str(args.benchmark_questions.resolve()), options),
    ) as pool:
        # A bounded generator prevents Executor.map from loading the whole corpus
        # into memory as Future objects on Python versions without buffersize.
        while True:
            batch = []
            for _ in range(2048):
                line = source.readline()
                if not line:
                    break
                counts["input_documents"] += 1
                batch.append((counts["input_documents"] - 1, line))
            if not batch:
                break
            for doc in pool.map(features, batch, chunksize=16):
                if doc.get("empty"):
                    counts["empty_documents"] += 1
                    continue
                position = components.add()
                shingles = doc.pop("shingles")
                doc["index"] = position
                exact_previous = content_seen.get(doc["content_sha256"])
                if exact_previous is not None:
                    components.union(position, exact_previous)
                    counts["exact_duplicate_edges"] += 1
                else:
                    content_seen[doc["content_sha256"]] = position
                    for other, _ in index.add(position, shingles):
                        components.union(position, other)
                        counts["verified_near_duplicate_edges"] += 1
                previous_page = page_seen.setdefault(doc["document_id"], position)
                if previous_page != position:
                    components.union(position, previous_page)
                    counts["same_page_edges"] += 1
                if doc["record"] is None:
                    counts["too_short_documents"] += 1
                if doc["special_token_ids"]:
                    counts["special_token_documents"] += 1
                if doc["benchmark_tasks"]:
                    counts["benchmark_overlap_documents"] += 1
                    benchmark_counts.update(doc["benchmark_tasks"])
                docs.append(doc)
            print(json.dumps({"progress_documents": counts["input_documents"],
                              "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)

    grouped = defaultdict(list)
    for doc in docs:
        grouped[components.find(doc["index"])].append(doc)
    records, membership = [], []
    for root, members in grouped.items():
        if any(doc["benchmark_tasks"] for doc in members):
            counts["benchmark_excluded_components"] += 1
            counts["benchmark_excluded_component_documents"] += len(members)
            continue
        if any(doc["special_token_ids"] for doc in members):
            counts["special_token_excluded_components"] += 1
            counts["special_token_excluded_component_documents"] += len(members)
            continue
        eligible = [doc for doc in members if doc["record"] is not None]
        if not eligible:
            counts["short_only_components"] += 1
            continue
        selected = eligible[0]
        record = selected["record"]
        records.append(record)
        counts[record.split] += 1
        counts["redundant_eligible_documents_removed"] += len(eligible) - 1
        membership.append({key: selected[key] for key in (
            "source_line", "document_id", "url", "content_sha256", "token_count", "word_count")})
        membership[-1].update({"record_id": record.id, "split": record.split,
                               "component_size": len(members), "component_root_source_line": docs[root]["source_line"]})
    records.sort(key=lambda record: hashlib.sha256((args.salt+":order:"+record.id).encode()).hexdigest())
    count = write_records(args.out, records)
    member_temp = membership_path.with_name("." + membership_path.name + ".incomplete")
    try:
        with member_temp.open("x") as destination:
            for row in membership:
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(member_temp, membership_path)
    finally:
        member_temp.unlink(missing_ok=True)
    with args.benchmark_questions.open() as source:
        question_screen = QuestionScreen([json.loads(line) for line in source if line.strip()])
    manifest = {
        "created_utc": utc_now(), "records": count, "counts": dict(counts),
        "input_sha256": sha256_file(args.documents), "output_sha256": sha256_file(args.out),
        "document_membership_sha256": sha256_file(membership_path),
        "benchmark_questions_sha256": sha256_file(args.benchmark_questions),
        "benchmark_screen": {"word_ngram": 13, "minimum_short_question_words": 5,
                             "question_counts": dict(question_screen.counts),
                             "matched_documents_by_task": dict(benchmark_counts), "outcomes_read": False},
        "near_duplicate_audit": {"word_shingle": 5, "jaccard_threshold": args.near_jaccard,
                                 "permutations": 64, "bands": 16, "rows_per_band": 4,
                                 "seed": 20261016, "candidate_pairs": index.candidate_pairs,
                                 "exact_jaccard_comparisons": index.comparisons,
                                 "maximum_candidates_per_document": index.max_candidates,
                                 "components": len(grouped), "component_policy": "first_eligible_source_document",
                                 "contaminated_components_excluded_in_full": True},
        "document_identity": "host_path_and_sorted_content_query_v2; named_tracking_parameters_removed",
        "special_tokens": {"policy": "exclude_entire_component_if_any_member_contains_tokenizer_special_id"},
        "tokenizer_identity": args.tokenizer_identity, "source_revision": args.source_revision,
        **options, "elapsed_seconds": round(time.monotonic()-start, 2),
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "numpy")},
        "limitations": ["MinHash/LSH candidate discovery is probabilistic; exact Jaccard verifies discovered edges only.",
                        "Normalized word overlap does not detect arbitrary paraphrases or semantic duplicates.",
                        "Questions shorter than five words are not independently screened in full.",
                        "Filtering changes the source distribution; retained records require at least prefix+future tokens.",
                        "No statement about benchmark contamination in the pretrained model."]}
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
