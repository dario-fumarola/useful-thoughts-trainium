"""Unrestricted hindsight bound from released V30 greedy/B0 scores only.

This reads existing scores, never models. The token choices see the true target
and are not a deployable selector. No sampled-candidate or coefficient search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(evidence, manifest):
    scores = {}
    pins = {}
    for tag, wanted in (("score-B0", "B0_raw"), ("score-T", "greedy")):
        for shard in range(4):
            name = f"outputs/{tag}-{shard}.jsonl"
            path = evidence / name
            digest = sha256(path)
            if digest != manifest["files"][name]:
                raise ValueError(f"Released score SHA mismatch: {name}")
            pins[name] = digest
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["mode"] != wanted:
                    continue
                key = (row["item_id"], wanted)
                if key in scores:
                    raise ValueError("Duplicate score identity")
                values = row["token_logps"]
                if (row["status"] != "ok" or row["tokens"] != 16
                        or len(values) != 16
                        or any(type(v) not in (float, int)
                               or not math.isfinite(v) or v > 1e-6 for v in values)
                        or not math.isclose(math.fsum(values), row["full_logp"],
                                            abs_tol=1e-6, rel_tol=0)):
                    raise ValueError("Incomplete/nonfinite/inconsistent token scores")
                target = row["omitted_source_arrays"]["target_ids"]
                if target["length"] != 16:
                    raise ValueError("Original sixteen target tokens required")
                scores[key] = row
    return scores, pins


def analyze(scores, *, expected_documents=256):
    ids = sorted({item for item, _ in scores})
    if len(ids) != expected_documents or len(scores) != 2 * expected_documents:
        raise ValueError("Complete paired cohort required; no item omissions")
    documents = set()
    rows = []
    for item in ids:
        raw, assisted = scores[(item, "B0_raw")], scores[(item, "greedy")]
        target = raw["omitted_source_arrays"]["target_ids"]
        if (raw["document_id"] != assisted["document_id"]
                or target != assisted["omitted_source_arrays"]["target_ids"]):
            raise ValueError("Paired target/document mismatch")
        if raw["document_id"] in documents:
            raise ValueError("One question per distinct document required")
        documents.add(raw["document_id"])
        delta = [b - a for a, b in zip(raw["token_logps"], assisted["token_logps"])]
        positive = math.fsum(max(0, v) for v in delta)
        harm = -math.fsum(min(0, v) for v in delta)
        rows.append(dict(
            item_id=item, document_id=raw["document_id"], target=target,
            token_greedy_minus_B0=delta,
            greedy_minus_B0_nats=math.fsum(delta),
            unrestricted_hindsight_gain_nats=positive,
            avoidable_harm_nats=harm,
            tokens_where_assisted_is_better=sum(v > 0 for v in delta),
            tokens_where_B0_is_better=sum(v < 0 for v in delta)))
    gains = [r["unrestricted_hindsight_gain_nats"] for r in rows]
    summary = dict(
        documents=len(rows), tokens=16 * len(rows),
        mean_upper_nats_per16=statistics.mean(gains),
        median_upper_nats_per16=statistics.median(gains),
        minimum_upper_nats_per16=min(gains), maximum_upper_nats_per16=max(gains),
        documents_with_positive_upper=sum(v > 0 for v in gains),
        documents_with_upper_at_least_point25=sum(v >= .25 for v in gains),
        mean_greedy_minus_B0_nats_per16=statistics.mean(
            r["greedy_minus_B0_nats"] for r in rows),
        mean_avoidable_harm_nats_per16=statistics.mean(
            r["avoidable_harm_nats"] for r in rows),
        assisted_better_tokens=sum(r["tokens_where_assisted_is_better"] for r in rows),
        B0_better_tokens=sum(r["tokens_where_B0_is_better"] for r in rows))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    scores, pins = read_scores(args.evidence, manifest)
    rows, summary = analyze(scores)
    result = dict(
        status="EXPLORATORY_SAVED_GREEDY_TOKEN_ORACLE_NOT_CAUSAL_OR_CONFIRMATORY",
        formula="U_d=sum_t max(0,logp_greedy(d,t)-logp_B0_raw(d,t))",
        derivation="For fixed target y, (1-g)*p0(y)+g*p1(y) is affine in g. "
                   "Its maximum on[0,1] is max(p0(y),p1(y)). Taking logs and "
                   "summing gives the per-document tokenwise hindsight bound.",
        modes_used=["B0_raw", "greedy"],
        summary=summary, rows=rows, source_pins=pins,
        release_manifest_sha256=sha256(args.manifest),
        analysis_source_sha256=sha256(Path(__file__)),
        limits=[
            "Already-inspected V30 TRAIN panel and fixed checkpoint/greedy notes; "
            "not fresh confirmation or a revised V30 endpoint.",
            "Every token choice sees the actual target. A positive bound cannot "
            "show that a causal gate can recognize helpful predictions.",
            "Only one original greedy note per document. No best-of-eight, "
            "candidate reselection, coefficient fit, model call or learning.",
            "Does not measure ordinary-prefix opportunity/retention or natural "
            "corpus prevalence, serving latency, internalization or CPT benefit."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
