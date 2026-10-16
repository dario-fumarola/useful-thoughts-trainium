"""Strict numeric answer scoring and paired downstream summaries.

No model calls, answer-aware generation, or automatic alternative extraction.
The parser must be locked using development data before official final testing.
"""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
import re
from .stats import paired_cluster_bootstrap

NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
PATTERN = re.compile(r"####\s*(" + NUMBER + r")\s*\.?\s*$")


def numeric_answer(text: str) -> str | None:
    match = PATTERN.search(text.strip())
    if not match:
        return None
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        value = Decimal(0)
    return format(value.normalize(), "f")


def score_predictions(gold: list[dict], predictions: list[dict]) -> tuple[list[dict], dict]:
    def indexed(rows):
        out = {}
        for row in rows:
            key = str(row["item_id"])
            if key in out:
                raise ValueError(f"Duplicate item: {key}")
            out[key] = row
        return out
    g, p = indexed(gold), indexed(predictions)
    if not g or set(g) != set(p):
        raise ValueError("Gold and prediction item sets must match exactly; missing results are not dropped.")
    out = []
    for key, row in g.items():
        truth = numeric_answer(row["answer"])
        if truth is None:
            raise ValueError(f"Invalid gold format for {key}; fix data before evaluation.")
        pred = p[key]
        status = pred.get("status", "ok")
        if status not in {"ok", "truncated", "error", "format_error"}:
            raise ValueError(f"Unknown prediction status: {status}")
        answer = numeric_answer(pred.get("generated_text", ""))
        if status == "ok" and answer is None:
            status = "format_error"
        # A recorded truncation or execution error counts as a failure, even if
        # an earlier partial buffer happens to contain a correct-looking marker.
        correct = status == "ok" and answer == truth
        out.append({"item_id": key, "document_id": str(row.get("document_id", key)),
                    "score": float(correct), "status": status,
                    "n_generated_tokens": int(pred.get("n_generated_tokens", 0))})
    counts = {s: sum(r["status"] == s for r in out) for s in ("ok", "truncated", "error", "format_error")}
    return out, {"items": len(out), "accuracy": sum(r["score"] for r in out) / len(out),
                 "statuses": counts, "parser": "strict_final_hash4_number_v1",
                 "qualification": "One generated path per item; failures retained in denominator."}


def downstream_contrasts(c_direct, c_thought, r_direct, r_thought, *, repeats=5000, seed=17):
    return {
        "primary_R_thought_minus_C_thought": paired_cluster_bootstrap(c_thought, r_thought, repeats=repeats, seed=seed),
        "training_R_direct_minus_C_direct": paired_cluster_bootstrap(c_direct, r_direct, repeats=repeats, seed=seed),
        "inference_R_thought_minus_R_direct": paired_cluster_bootstrap(r_direct, r_thought, repeats=repeats, seed=seed),
        "control_thinking_C_thought_minus_C_direct": paired_cluster_bootstrap(c_direct, c_thought, repeats=repeats, seed=seed),
    }
