"""Deterministic canonical GSM8K projection; no model or outcome selection."""
from __future__ import annotations

import json

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"


def project_rows(rows: list[dict], *, split: str = "test") -> tuple[list[dict], list[dict]]:
    if split not in {"train", "test"}:
        raise ValueError("Expected canonical GSM8K train/test split")
    questions, gold = [], []
    for index, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("Canonical GSM8K rows require question and worked answer")
        item_id = f"gsm8k:{split}:{index:05d}"
        questions.append(dict(item_id=item_id, document_id=item_id, task="gsm8k", split=split,
                              question=row["question"]))
        gold.append(dict(item_id=item_id, document_id=item_id, task="gsm8k", split=split,
                         answer=row["answer"]))
    if not questions:
        raise ValueError("Empty canonical GSM8K source")
    return questions, gold


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n"
                   for row in rows).encode()
