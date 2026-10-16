"""Fail-closed final-evaluation bindings; qualification does not call these gates.

This verifies local trusted artifacts, not the scientific truth of a receipt.
It neither reads held-out targets nor imports the accelerator runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.numeric_projection import GSM8K_REVISION, GSM8K_TEST_SHA256, jsonl_bytes, project_rows


FREEZE_SCHEMA = "evaluation-freeze-v2"
PARSER = "strict_final_hash4_number_v1"


def evaluation_dependencies(root: Path | None = None) -> dict[str, str]:
    """A common conservative closure, including helpers imported by other entrypoints."""
    root = root or Path(__file__).resolve().parents[1]
    names = ["train/evaluate.py", "train/evaluate_likelihood.py",
             "train/evaluation_integrity.py", "train/neuron_backend.py", "train/runtime_compat.py",
             "weekend_thinking/artifacts.py", "weekend_thinking/core.py",
             "weekend_thinking/backend_contract.py", "weekend_thinking/pretrained.py",
             "weekend_thinking/evaluation.py", "weekend_thinking/stats.py",
             "weekend_thinking/order.py", "weekend_thinking/numeric_projection.py",
             "scripts/score_numeric_answers.py"]
    # These new entrypoints share this closure once present in the active source.
    for name in ("train/evaluate_local.py", "scripts/prepare_numeric_scoring.py"):
        if (root/name).exists():
            names.append(name)
    return {name: sha256_file(root/name) for name in sorted(names)}


def read_evaluation_freeze(path: Path | None) -> dict:
    if path is None:
        raise ValueError("Final evaluation requires --frozen-protocol")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != FREEZE_SCHEMA or receipt.get("status") != "frozen"
            or not receipt.get("frozen_utc")):
        raise ValueError("Expected a complete evaluation-freeze-v2 receipt")
    spec = receipt.get("training_spec", {})
    if not spec.get("spec_hash") or canonical_hash({k: v for k, v in spec.items()
                                                   if k != "spec_hash"}) != spec["spec_hash"]:
        raise ValueError("Frozen training specification hash mismatch")
    if not isinstance(receipt.get("models"), list) or not receipt["models"]:
        raise ValueError("Freeze must enumerate the exact permitted final models")
    return receipt


def training_order_hash(spec: dict, seed: int) -> str:
    """Match order.seed_order using record headers only, without decoding futures."""
    path = Path(spec["assets"]["records_file"])
    if path.is_symlink() or sha256_file(path) != spec["assets"]["records_sha256"]:
        raise ValueError("Frozen training records changed")
    pattern = re.compile(r'^\{"id":\s*"([^"\\]+)",\s*"document_id":\s*"[^"\\]+",\s*"split":\s*"(train|dev|test)"')
    ids = []
    with path.open() as stream:
        for line in stream:
            match = pattern.match(line)
            if not match:
                raise ValueError("Expected canonical record headers before prefix/future fields")
            if match[2] == "train":
                ids.append(match[1])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Training order requires nonempty unique IDs")
    return canonical_hash(sorted(ids, key=lambda item: hashlib.sha256(
        f"order:{seed}:{item}".encode()).hexdigest()))


def final_model_identity(checkpoint: Path | None, assets: dict, spec: dict, *,
                         arm: str, seed: int | None) -> dict:
    """Construct and verify a final identity before putting it in a freeze receipt.

    C/R checks include every rank shard even though inference only loads model.pt.
    The model dictionary's internal step/run/spec is checked separately on load.
    """
    base = spec["assets"]
    if (not assets.get("manifest_sha256") or assets["manifest_sha256"] != base["model_manifest_sha256"]
            or assets.get("model_id") != base["model_id"] or assets.get("revision") != base["model_revision"]):
        raise ValueError("Base assets differ from the frozen pretrained identity")
    common = dict(arm=arm, seed=seed, base_assets_sha256=assets["sha256"],
                  model_manifest_sha256=assets["manifest_sha256"], model_id=assets["model_id"],
                  model_revision=assets["revision"])
    if arm == "B0":
        if checkpoint is not None or seed is not None:
            raise ValueError("B0 requires unchanged pretrained assets and no checkpoint/seed")
        return common
    if arm not in {"vanilla", "thought"} or type(seed) is not int or seed not in spec["proposal"]["optimization"]["seeds"]:
        raise ValueError("Final trained model arm/seed is not frozen")
    if checkpoint is None or checkpoint.name != "model.pt" or checkpoint.is_symlink() or checkpoint.parent.is_symlink():
        raise ValueError("Final trained model requires complete checkpoint model.pt")
    directory = checkpoint.parent
    complete = directory/"COMPLETE.json"
    if complete.is_symlink():
        raise ValueError("Symlink checkpoint manifest is not accepted")
    manifest = json.loads(complete.read_text())
    step, world = spec["terminal_steps"], spec["site"]["world_size_per_arm"]
    expected = {"model.pt"} | {f"rank-{rank:05d}.pt" for rank in range(world)}
    if (manifest.get("schema") != "neuron-v1" or type(manifest.get("step")) is not int
            or manifest["step"] != step or manifest.get("world_size") != world
            or manifest.get("spec_hash") != spec["spec_hash"] or set(manifest.get("files", {})) != expected
            or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]):
        raise ValueError("Final checkpoint must be complete at the exact frozen terminal step/spec/topology")
    metadata = manifest.get("metadata", {})
    batch = spec["proposal"]["optimization"]["global_unique_prefixes_per_update"]
    order_hash = training_order_hash(spec, seed)
    for key, value in dict(spec_hash=spec["spec_hash"], arm=arm, seed=seed, order_hash=order_hash,
                           global_batch=batch, accumulation=batch//world, parameter_dtype="fp32",
                           compute_dtype=spec["proposal"]["optimization"]["compute_dtype"]).items():
        if metadata.get(key) != value:
            raise ValueError(f"Final checkpoint lineage mismatch: {key}")
    if metadata.get("counters", {}).get("source_records") != step*batch:
        raise ValueError("Final checkpoint source exposure differs from terminal step")
    for name, entry in manifest["files"].items():
        path = directory/name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry.get("bytes")
                or sha256_file(path) != entry.get("sha256")):
            raise ValueError(f"Final checkpoint integrity mismatch: {name}")
    return dict(common, spec_hash=spec["spec_hash"], step=step, run_id=manifest["run_id"],
                order_hash=order_hash, model_sha256=manifest["files"]["model.pt"]["sha256"],
                complete_sha256=sha256_file(complete))


def require_final_model(frozen_path: Path | None, checkpoint: Path | None, assets: dict, *,
                        arm: str, seed: int | None) -> dict:
    receipt = read_evaluation_freeze(frozen_path)
    identity = final_model_identity(checkpoint, assets, receipt["training_spec"], arm=arm, seed=seed)
    if identity not in receipt["models"]:
        raise ValueError("Model identity is not in the frozen final checkpoint inventory")
    return identity


def validate_loaded_checkpoint(saved: dict, identity: dict) -> None:
    for key in ("step", "run_id", "spec_hash"):
        if saved.get(key) != identity.get(key):
            raise ValueError(f"Loaded checkpoint metadata differs from final identity: {key}")


def require_numeric_scoring_manifest(frozen_path: Path | None, manifest_path: Path,
                                     gold_path: Path) -> tuple[dict, dict]:
    """Validate locked hashes before the caller parses a single official gold answer."""
    frozen = read_evaluation_freeze(frozen_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest not in frozen.get("scoring_manifests", []):
        raise ValueError("Numeric scoring manifest is not locked in the evaluation freeze")
    if (manifest.get("schema") != "numeric-scoring-v1" or manifest.get("task") != "gsm8k"
            or manifest.get("task_revision") != GSM8K_REVISION or manifest.get("split") != "test"
            or manifest.get("parser") != PARSER or manifest.get("dependencies") != evaluation_dependencies()
            or manifest.get("training_spec_hash") != frozen["training_spec"]["spec_hash"]
            or manifest.get("gold_file_sha256") != sha256_file(gold_path)):
        raise ValueError("Numeric gold/parser/revision/dependency binding mismatch")
    source = Path(manifest["canonical_source_file"])
    if manifest.get("canonical_source_sha256") != GSM8K_TEST_SHA256 or sha256_file(source) != GSM8K_TEST_SHA256:
        raise ValueError("Canonical GSM8K source changed")
    questions = Path(manifest["question_file"])
    if sha256_file(questions) != manifest.get("question_file_sha256"):
        raise ValueError("Canonical question projection changed")
    # Run only after the freeze gate above. Regenerate hashes, not hand-picked labels.
    import pyarrow.parquet as pq
    projected_questions, projected_gold = project_rows(pq.read_table(
        source, columns=["question", "answer"]).to_pylist())
    if (hashlib.sha256(jsonl_bytes(projected_questions)).hexdigest() != manifest["question_file_sha256"]
            or hashlib.sha256(jsonl_bytes(projected_gold)).hexdigest() != manifest["gold_file_sha256"]
            or canonical_hash([row["item_id"] for row in projected_questions]) != manifest.get("item_ids_sha256")):
        raise ValueError("Question/gold files are not the complete canonical benchmark projection")
    return frozen, manifest


def validate_numeric_predictions(frozen: dict, manifest: dict, predictions_path: Path) -> tuple[list[dict], dict]:
    """Accept one original complete generation shard, tied to its run receipt."""
    run = json.loads(predictions_path.with_name("run.json").read_text())
    protocol, identity = run.get("protocol", {}), run.get("model_identity")
    if (run.get("qualification") is not False or run.get("status") != "complete"
            or protocol not in frozen.get("protocols", []) or identity not in frozen["models"]
            or protocol.get("task") != "gsm8k" or protocol.get("task_revision") != manifest["task_revision"]
            or protocol.get("task_file_sha256") != manifest["question_file_sha256"]
            or protocol.get("item_ids_sha256") != manifest["item_ids_sha256"]
            or protocol.get("dependencies") != evaluation_dependencies()
            or run.get("protocol_sha256") != canonical_hash(protocol)
            or run.get("predictions_sha256") != sha256_file(predictions_path)):
        raise ValueError("Predictions are not bound to a complete frozen generation run")
    if (run.get("model", {}).get("sha256") != identity["base_assets_sha256"]
            or (run.get("checkpoint") or {}).get("sha256") != identity.get("model_sha256")):
        raise ValueError("Prediction run model/checkpoint identity mismatch")
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    if len(rows) != run.get("expected_items"):
        raise ValueError("Prediction shard count differs from generation receipt")
    for row in rows:
        if (row.get("split") != "test" or row.get("run_id") != run.get("run_id")
                or row.get("protocol_sha256") != run["protocol_sha256"]
                or row.get("model_sha256") != identity["base_assets_sha256"]
                or row.get("checkpoint_sha256") != identity.get("model_sha256")
                or row.get("model_identity_sha256") != canonical_hash(identity)):
            raise ValueError("Prediction row provenance differs from the frozen run")
    return rows, run
