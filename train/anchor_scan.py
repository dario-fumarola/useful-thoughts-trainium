"""Prepare a TRAIN-only ID pool, then separately scan raw B0 surprisal.

Preparation decodes record headers only. A later score command observes TRAIN
targets to select possible training anchors; any writer must still receive only
their prefixes. DEV is used solely for one fixed numerical width comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from train.evaluate import asset_provenance
from train.evaluate_local import RECORD_HEADER, load_selected, read_selection, setup_model
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "anchors-train1024-v1"
SEED, POOL, TAKE, WIDTH, THRESHOLD = 20261017, 1024, 128, 544, 1.7
RECORDS = ROOT/"local/data/finemath/production-500000.audit-v2.records.jsonl"
MODEL = ROOT/"local/assets/qwen3-1.7b-base"


def order_key(item_id: str) -> str:
    return canonical_hash(dict(seed=SEED, item_id=item_id))


def train_pool(path: Path, expected_sha256: str, *, count: int = POOL) -> dict:
    """Hash the file and inspect headers; never JSON-decode prefix/future arrays."""
    digest = hashlib.sha256()
    candidates, seen = [], set()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            header = RECORD_HEADER.match(raw.decode())
            if header is None:
                raise ValueError("Expected canonical ID/document/split headers")
            item_id, document_id, split = header.groups()
            if split != "train":
                continue
            if item_id in seen:
                raise ValueError("Duplicate TRAIN item ID")
            seen.add(item_id)
            candidates.append((order_key(item_id), item_id, document_id))
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Records differ from their pinned manifest")
    selected, documents = [], set()
    for _, item_id, document_id in sorted(candidates):
        if document_id in documents:
            continue
        documents.add(document_id)
        selected.append((item_id, document_id))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient distinct TRAIN documents; no fallback")
    ids = [item for item, _ in selected]
    return dict(records_file=str(path.resolve()), records_sha256=expected_sha256,
        split="train", count=count, item_ids=ids,
        document_ids=[document for _, document in selected], item_ids_sha256=canonical_hash(ids),
        eligible_train_items=len(candidates), selection_seed=SEED,
        selection_rule="Ascending SHA256(canonical JSON {seed:20261017,item_id}); first item per document, then first1024",
        pool_preparation_arrays_decoded=False)


def sources() -> dict:
    return {**evaluation_dependencies(), "train/anchor_scan.py": sha256_file(Path(__file__))}


def prepare(output: Path, *, records=RECORDS, model_dir=MODEL,
            selection_file=ROOT/"local/data/finemath/fixed-local-evaluation-v1.ids.json",
            parity_scores=ROOT/"local/nvme/evaluation/local-dev512-v1-B0-001/scores/scores.jsonl") -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest_path = records.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["prefix_length"] != 512 or manifest["future_length"] != 16:
        raise ValueError("This scan requires the existing prefix512/future16 records")
    pool = train_pool(records, manifest["output_sha256"])
    pool["source_revision"] = manifest["source_revision"]
    dev = read_selection(selection_file, qualification=True, split="dev")
    if dev["records_sha256"] != pool["records_sha256"]:
        raise ValueError("Width comparison and TRAIN pool need the same fixed record asset")
    parity = dict(dev, count=1, item_ids=dev["item_ids"][:1], document_ids=dev["document_ids"][:1],
                  item_ids_sha256=canonical_hash(dev["item_ids"][:1]))
    if set(pool["document_ids"]) & set(parity["document_ids"]):
        raise ValueError("DEV width-reference document overlaps the TRAIN pool")
    model_manifest = model_dir/"ASSET_MANIFEST.json"
    model = json.loads(model_manifest.read_text())
    if model["model_id"] != "Qwen/Qwen3-1.7B-Base":
        raise ValueError("Use the unchanged released B0")
    protocol = dict(schema=SCHEMA, experiment_id=SCHEMA, created_utc=utc_now(),
        purpose="TRAIN anchor preparation only; no candidate generation or training",
        model_dir=str(model_dir.resolve()), model_id=model["model_id"],
        model_revision=model["resolved_revision"], model_manifest_sha256=sha256_file(model_manifest),
        records_manifest=str(manifest_path.resolve()), records_manifest_sha256=sha256_file(manifest_path),
        source_sha256=sources(), pool_items=POOL, threshold_surprisal_nats=THRESHOLD,
        threshold_comparison=">=", requested_generation_items=TAKE,
        generation_selection_rule="First up to128 qualifying IDs in frozen pool order; no fallback",
        selector_uses_observed_TRAIN_target=True, writer_inputs="prefix tokens only",
        horizons=[1, 16], width=WIDTH, pad_id=151643, compute="fp32", checkpoint=None,
        parity=dict(selection=parity, selection_file=str(selection_file.resolve()),
            selection_file_sha256=sha256_file(selection_file), scores_file=str(parity_scores.resolve()),
            scores_sha256=sha256_file(parity_scores), atol=.002,
            role="Numerical width comparison only; never eligible for TRAIN anchor selection"))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"pool.ids.json", pool)
    protocol["pool_sha256"] = sha256_file(output/"pool.ids.json")
    atomic_json(output/"protocol.json", protocol)
    return protocol


def reference_logps(path: Path, item_id: str) -> list[float]:
    """Read saved likelihoods for the one predeclared DEV item, not new targets."""
    values = []
    with path.open() as stream:
        for line in stream:
            if item_id not in line:
                continue
            row = json.loads(line)
            if row["item_id"] == item_id and row["mode"] == "raw" and row["horizon"] == 16:
                if row["status"] != "ok":
                    raise ValueError("Failed DEV raw reference")
                values.append(row["token_log_likelihoods"])
    if (len(values) != 1 or len(values[0]) != 16
            or not all(math.isfinite(v) for v in values[0])):
        raise ValueError("Need one complete finite raw reference")
    return values[0]


def raw_score(backend, record: dict, pad_id: int) -> dict:
    result = score_continuations(backend, record["prefix"], [record["future"][:16]],
        total_width=WIDTH, future_width=16, pad_id=pad_id)[0]
    return dict(status="ok", token_log_likelihoods=result.token_log_likelihoods,
        raw_K1_surprisal_nats=-result.token_log_likelihoods[0],
        raw_K1_log_likelihood=result.token_log_likelihoods[0],
        raw_K16_log_likelihood=math.fsum(result.token_log_likelihoods))


def summarize(pool: dict, rows: list[dict]) -> tuple[dict, dict | None]:
    index = {row["item_id"]: row for row in rows}
    if len(rows) != len(index) or set(index) != set(pool["item_ids"]):
        raise ValueError("Retain the entire fixed pool exactly once")
    documents = dict(zip(pool["item_ids"], pool["document_ids"]))
    for key, row in index.items():
        if row["split"] != "train" or row["document_id"] != documents[key]:
            raise ValueError("TRAIN pool document/split mismatch")
        if row["status"] == "ok":
            values = row.get("token_log_likelihoods", [])
            if (len(values) != 16 or not all(math.isfinite(v) for v in values)
                    or row["raw_K1_surprisal_nats"] != -values[0]):
                raise ValueError("Invalid raw likelihood row")
    failures = sum(row["status"] != "ok" for row in rows)
    summary = dict(status="incomplete" if failures else "complete", expected_items=pool["count"],
        rows=len(rows), failures=failures, threshold_surprisal_nats=THRESHOLD,
        requested_generation_items=TAKE, generation_selection_available=not failures,
        failure_policy="Any unknown earlier qualifier withholds selection; no complete-case fallback")
    if failures:
        return summary, None
    qualifying = [key for key in pool["item_ids"] if index[key]["raw_K1_surprisal_nats"] >= THRESHOLD]
    chosen = qualifying[:TAKE]
    selected = dict(pool, count=len(chosen), item_ids=chosen,
        document_ids=[documents[key] for key in chosen], item_ids_sha256=canonical_hash(chosen),
        parent_pool_item_ids_sha256=pool["item_ids_sha256"], threshold_surprisal_nats=THRESHOLD,
        selection_rule="First up to128 TRAIN items with raw_K1_surprisal_nats>=1.7 in frozen pool order; no fallback",
        selector_uses_observed_TRAIN_target=True, writer_inputs="prefix tokens only")
    summary.update(qualifying_items=len(qualifying), selected_items=len(chosen),
        fewer_than_requested=len(chosen)<TAKE,
        mean_raw_K1_surprisal_nats=statistics.mean(row["raw_K1_surprisal_nats"] for row in rows),
        mean_raw_K16_log_likelihood=statistics.mean(row["raw_K16_log_likelihood"] for row in rows))
    return summary, selected


def score(output: Path, device: str) -> None:
    protocol = json.loads((output/"protocol.json").read_text())
    pool = read_selection(output/"pool.ids.json", qualification=True, split="train")
    if (protocol["schema"] != SCHEMA or protocol["source_sha256"] != sources()
            or protocol["pool_sha256"] != sha256_file(output/"pool.ids.json")
            or pool["count"] != POOL or pool["selection_seed"] != SEED
            or protocol["width"] != WIDTH or protocol["pool_items"] != POOL
            or protocol["requested_generation_items"] != TAKE or protocol["horizons"] != [1, 16]
            or protocol["pad_id"] != 151643 or protocol["parity"]["atol"] != .002
            or protocol["threshold_surprisal_nats"] != THRESHOLD or protocol["compute"] != "fp32"):
        raise ValueError("Anchor protocol/source/selection differs from preparation")
    pins = [(protocol["records_manifest"], protocol["records_manifest_sha256"]),
            (Path(protocol["model_dir"])/"ASSET_MANIFEST.json", protocol["model_manifest_sha256"]),
            (protocol["parity"]["selection_file"], protocol["parity"]["selection_file_sha256"]),
            (protocol["parity"]["scores_file"], protocol["parity"]["scores_sha256"])]
    if any(sha256_file(path) != digest for path, digest in pins):
        raise ValueError("Changed pinned metadata or saved reference")
    directory = output/"scoring"
    directory.mkdir(exist_ok=False)
    report = dict(schema=SCHEMA, status="running", started_utc=utc_now(),
                  protocol_sha256=sha256_file(output/"protocol.json"), expected_items=POOL)
    atomic_json(directory/"run.json", report)
    try:
        assets = asset_provenance(Path(protocol["model_dir"]))
        backend, _, checkpoint, signature, _ = setup_model(SimpleNamespace(
            model_dir=Path(protocol["model_dir"]), checkpoint=None, device=device))
        report.update(model_assets=assets, model=signature, checkpoint=checkpoint)
        parity = protocol["parity"]
        record = load_selected(Path(pool["records_file"]), parity["selection"], include_future=True)[0]
        actual = raw_score(backend, record, protocol["pad_id"])
        expected = reference_logps(Path(parity["scores_file"]), record["item_id"])
        error = max(abs(a-b) for a,b in zip(actual["token_log_likelihoods"], expected))
        atomic_json(directory/"parity.json", dict(item_id=record["item_id"], split="dev",
            role=parity["role"], max_token_error=error, atol=parity["atol"],
            comparison_status="ok" if math.isfinite(error) and error<=parity["atol"] else "error", **actual))
        if not math.isfinite(error) or error > parity["atol"]:
            raise ValueError(f"Raw width parity failed: {error}")
        report["DEV_width_parity_max_token_error"] = error
        records = load_selected(Path(pool["records_file"]), pool, include_future=True)
        rows = []
        with (directory/"scores.jsonl").open("x") as stream:
            for index, record in enumerate(records):
                started = time.monotonic()
                row = {key:record[key] for key in ("item_id", "document_id", "split")}
                try:
                    row.update(raw_score(backend, record, protocol["pad_id"]))
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                row["wall_seconds"] = time.monotonic()-started
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
                print(json.dumps(dict(items_complete=index+1, expected_items=POOL,
                    failures=sum(r["status"]!="ok" for r in rows), utc=utc_now())), flush=True)
        summary, selected = summarize(pool, rows)
        summary.update(protocol_sha256=report["protocol_sha256"], scores_sha256=sha256_file(directory/"scores.jsonl"))
        atomic_json(directory/"summary.json", summary)
        report.update(rows=len(rows), failures=summary["failures"], summary_sha256=sha256_file(directory/"summary.json"))
        if selected is None:
            raise RuntimeError("Retained failed TRAIN scores; generation selection withheld")
        atomic_json(output/"candidate-selection.ids.json", selected)
        report.update(status="complete", finished_utc=utc_now(),
            candidate_selection_sha256=sha256_file(output/"candidate-selection.ids.json"))
    except BaseException as exc:
        report.update(status="failed", finished_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(directory/"run.json", report, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--device", choices=("cpu", "xla"), default="xla")
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = prepare(args.output, records=args.records, model_dir=args.model_dir)
        print(json.dumps(dict(status="prepared_only", protocol=str(args.output/"protocol.json"),
                              schema=protocol["schema"])), flush=True)
    else:
        score(args.output, args.device)


if __name__ == "__main__":
    main()
