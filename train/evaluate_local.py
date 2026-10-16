"""Held-out content controls with saved, prefix-only writer notes and FP32 readers.

Generation and scoring are separate commands so the same fixed notes can be
scored by initial and trained readers. No reward or future enters generation,
control construction, item selection, or note selection. One stochastic sample
per document is primary; greedy generation is an explicitly separate diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import time

import torch

from train.evaluate import (asset_provenance, load_own_checkpoint,
                            require_frozen_protocol, shard_questions)
from train.evaluate_likelihood import score_continuations
from train.evaluation_integrity import evaluation_dependencies, require_final_model
from train.neuron_backend import NeuronCausalBackend
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions, matched_controls
from weekend_thinking.pretrained import load_base, sampled_fingerprint
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = "local_content_audit_v1"
MODES = ("raw", "empty", "own", "filler", "shuffled", "other")
HORIZONS = (1, 4, 16)
RECORD_HEADER = re.compile(r'^\{\s*"id"\s*:\s*"([^"\\]+)"\s*,\s*"document_id"\s*:\s*"([^"\\]+)"\s*,\s*"split"\s*:\s*"(train|dev|test)"\s*,\s*"prefix"\s*:\s*')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_selection(path: Path, *, qualification: bool, split: str | None = None) -> dict:
    """A globally fixed list is validated before any rank takes a slice."""
    selection = json.loads(path.read_text())
    if "splits" in selection:
        if split not in selection["splits"]:
            raise ValueError("Choose a declared --split from the global ID manifest")
        selection = {**{key: value for key, value in selection.items() if key != "splits"},
                     **selection["splits"][split]}
    elif split is not None and selection.get("split") != split:
        raise ValueError("Requested split differs from the selection")
    ids, documents = selection.get("item_ids"), selection.get("document_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Selection needs nonempty unique ordered item_ids")
    if (not isinstance(documents, list) or len(documents) != len(ids)
            or any(not isinstance(i, str) or not i for i in documents)
            or len(documents) != len(set(documents))):
        raise ValueError("Selection requires one distinct document per item")
    if selection.get("count") != len(ids) or selection.get("item_ids_sha256") != canonical_hash(ids):
        raise ValueError("Selection count or ordered-ID hash mismatch")
    split = selection.get("split")
    if qualification:
        if split not in {"train", "dev"}:
            raise ValueError("Qualification accepts TRAIN/dev only, never final test")
    elif split not in {"dev", "test"} or len(ids) != {"dev": 512, "test": 2048}[split]:
        raise ValueError("Scientific local audit requires the frozen global dev512/test2048 selection")
    if not isinstance(selection.get("records_sha256"), str) or len(selection["records_sha256"]) != 64:
        raise ValueError("Selection must bind the complete records-file hash")
    return selection


def load_selected(path: Path, selection: dict, *, include_future: bool) -> list[dict]:
    """Read only declared record fields; generation never accesses future IDs."""
    if sha256_file(path) != selection["records_sha256"]:
        raise ValueError("Records differ from the frozen selection asset")
    expected = dict(zip(selection["item_ids"], selection["document_ids"]))
    rows = {}
    decoder = json.JSONDecoder()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            header = RECORD_HEADER.match(line)
            if header is None:
                raise ValueError("Expected canonical id/document_id/split/prefix record headers")
            identifier, document_id, split = header.group(1, 2, 3)
            if identifier not in expected:
                continue
            if identifier in rows or document_id != expected[identifier] or split != selection["split"]:
                raise ValueError("Selected record duplicate/document/split mismatch")
            # Decode exactly one array; no selected or unselected future is
            # parsed by the writer. Canonical write_records puts prefix first.
            prefix, _ = decoder.raw_decode(line, header.end())
            if not isinstance(prefix, list) or not prefix or any(type(t) is not int or t < 0 for t in prefix):
                raise ValueError("Invalid selected prefix")
            row = dict(item_id=identifier, document_id=document_id, split=split, prefix=prefix)
            if include_future:
                obj = json.loads(line)
                future = obj.get("future")
                if not isinstance(future, list) or len(future) < max(HORIZONS) or any(type(t) is not int or t < 0 for t in future):
                    raise ValueError("Every local record needs the complete 16-token future")
                row["future"] = future[:max(HORIZONS)]
            rows[identifier] = row
    if set(rows) != set(expected):
        raise ValueError("Selected record set is incomplete; never drop missing items")
    result = [rows[identifier] for identifier in selection["item_ids"]]
    if len({len(row["prefix"]) for row in result}) != 1:
        raise ValueError("Local writer prefixes must have one fixed length")
    return result


def item_seed(seed: int, item_id: str) -> int:
    """Record-local RNG identity is invariant to process count and sharding."""
    return int(canonical_hash(dict(seed=seed, item_id=item_id))[:15], 16)


def generate_note(backend, prefix: list[int], *, fmt: Format, cap: int,
                  policy: str, seed: int, set_seed) -> dict:
    """This interface accepts no record, continuation, score, or gold target."""
    if policy not in {"sample", "greedy"}:
        raise ValueError("Expected one sample or one greedy diagnostic")
    set_seed(seed)
    rollout = backend.sample(torch.tensor([prefix], device=backend.device), fmt=fmt,
        group_size=1, max_actions=cap, temperature=.7, policy_version=0, greedy=policy == "greedy")
    backend.sync()
    mask = rollout.action_mask.cpu()[0, 0]
    actions = rollout.actions.cpu()[0, 0][mask].tolist()
    logps = rollout.behavior_logps.cpu()[0, 0][mask].tolist()
    if not actions or not all(math.isfinite(value) for value in logps):
        raise RuntimeError("Empty or nonfinite generated note")
    stops = {fmt.stop_id, *fmt.additional_stop_ids}
    content = content_from_actions(actions, fmt.stop_id, fmt.additional_stop_ids)
    return dict(status="ok", actions=actions, content_ids=content, behavior_logps=logps,
        stop_type=f"token:{actions[-1]}" if actions[-1] in stops else "length_cap",
        n_actions=len(actions), n_content_tokens=len(content), item_seed=seed,
        decode_iterations=rollout.decode_iterations, prefix_sha256=canonical_hash(prefix))


def load_note_bundles(paths: list[Path], selection: dict, fmt: Format) -> tuple[dict[str, dict], dict]:
    """All writer shards must cover the global selection exactly once."""
    notes, identity, receipts = {}, None, []
    for directory in paths:
        report = json.loads((directory/"run.json").read_text())
        protocol = report["protocol"]
        if (report.get("status") != "complete" or protocol.get("schema") != SCHEMA
                or protocol.get("command") != "generate"
                or protocol.get("selection_item_ids_sha256") != selection["item_ids_sha256"]
                or protocol.get("records_sha256") != selection["records_sha256"]
                or protocol.get("thought_format") != json.loads(json.dumps(fmt.__dict__))):
            raise ValueError("Note bundle is incomplete or belongs to another selection/format")
        if sha256_file(directory/"notes.jsonl") != report.get("notes_sha256"):
            raise ValueError("Saved note bundle content hash mismatch")
        current = dict(protocol=protocol, model=report["model"], checkpoint=report.get("checkpoint"),
                       final_model_identity=report.get("final_model_identity"))
        if identity is None:
            identity = current
        elif identity != current:
            raise ValueError("Cannot mix writer checkpoints or generation protocols")
        for row in read_jsonl(directory/"notes.jsonl"):
            key = row.get("item_id")
            if key in notes or key not in selection["item_ids"]:
                raise ValueError("Duplicate or unexpected note item")
            if row.get("status") not in {"ok", "error"}:
                raise ValueError("Unknown note status")
            index = selection["item_ids"].index(key)
            if row.get("document_id") != selection["document_ids"][index] or row.get("split") != selection["split"]:
                raise ValueError("Saved note document/split mismatch")
            if row["status"] == "ok":
                if (not row["actions"] or any(type(token) is not int or token < 0 for token in row["actions"])
                        or len(row.get("behavior_logps", [])) != len(row["actions"])
                        or not all(math.isfinite(value) for value in row["behavior_logps"])):
                    raise ValueError("Saved note action/logp mismatch")
                content = content_from_actions(row["actions"], fmt.stop_id, fmt.additional_stop_ids)
                if content != row["content_ids"] or len(row["actions"]) > protocol["thought_cap"]:
                    raise ValueError("Saved note content/action/stop mismatch")
                if len(content) < len(row["actions"])-1:
                    raise ValueError("Saved actions after the first sampled terminator")
                if len(content) == len(row["actions"]) and len(content) != protocol["thought_cap"]:
                    raise ValueError("A short saved action sequence needs its actual sampled terminator")
            notes[key] = row
        receipts.append(dict(directory=str(directory.resolve()), notes_sha256=report["notes_sha256"]))
    if set(notes) != set(selection["item_ids"]):
        raise ValueError("Writer shards must cover every globally selected item")
    return notes, dict(**identity, bundles=receipts)


def construct_controls(selection: dict, notes: dict[str, dict], *, filler_ids: list[int], seed: int) -> dict[str, dict]:
    """Global content-only randomization, performed before reading any future."""
    ids = selection["item_ids"]
    if len(ids) < 2:
        raise ValueError("Unrelated-note controls need at least two selected documents")
    contents = [notes[key]["content_ids"] if notes[key]["status"] == "ok" else [] for key in ids]
    matched = matched_controls(contents, filler_ids=filler_ids, seed=seed)
    controls = {}
    for index, key in enumerate(ids):
        donor = ids[(index+1) % len(ids)]
        values = dict(raw=None, empty=[], own=contents[index], filler=matched["filler"][index],
                      shuffled=matched["shuffled"][index], other=matched["other"][index])
        failures = {}
        if notes[key]["status"] != "ok":
            failures.update({mode: "recipient note generation failed" for mode in ("own", "filler", "shuffled", "other")})
        if notes[donor]["status"] != "ok":
            failures["other"] = "donor note generation failed"
        controls[key] = dict(content_ids=values, failures=failures, donor_item_id=donor,
            donor_original_tokens=len(contents[(index+1) % len(ids)]), recipient_tokens=len(contents[index]),
            donor_fit="cyclic_repeat_or_trim; empty donor uses fixed filler",
            matched_lengths={mode: len(values[mode]) == len(contents[index]) for mode in ("filler", "shuffled", "other")})
    return controls


def model_binding(assets: dict, checkpoint: dict | None, final_identity: dict | None) -> dict:
    """Stable identities without output paths, shared by all inference shards."""
    return final_identity if final_identity is not None else dict(
        base_assets_sha256=assets["sha256"], checkpoint_sha256=(checkpoint or {}).get("sha256"))


def score_record(record: dict, controls: dict, note: dict, *, backend, fmt: Format,
                 total_width: int, first_target_text: str | None = None) -> list[dict]:
    """Every requested condition/horizon gets a row, including execution failures."""
    rows = []
    for mode in MODES:
        started = time.monotonic()
        common = dict(item_id=record["item_id"], document_id=record["document_id"],
            split=record["split"], mode=mode, note_status=note["status"],
            first_target_token_id=record["future"][0], first_target_token_text=first_target_text,
            n_note_actions=note.get("n_actions", 0), n_content_tokens=len(controls["content_ids"][mode] or []))
        try:
            if canonical_hash(record["prefix"]) != note.get("prefix_sha256"):
                raise ValueError("Writer note and reader prefix hashes disagree")
            if mode in controls["failures"]:
                raise RuntimeError(controls["failures"][mode])
            context = list(record["prefix"])
            content = controls["content_ids"][mode]
            if content is not None:
                context = list(fmt.header_ids)+context+list(fmt.open_ids)+content+list(fmt.close_ids)
            score = score_continuations(backend, context, [record["future"][:max(HORIZONS)]],
                total_width=total_width, future_width=max(HORIZONS), pad_id=fmt.pad_id)[0]
            for horizon in HORIZONS:
                values = score.token_log_likelihoods[:horizon]
                rows.append(dict(**common, horizon=horizon, status="ok", log_likelihood=sum(values),
                    token_log_likelihoods=values, n_target_tokens=horizon))
        except Exception as error:
            for horizon in HORIZONS:
                rows.append(dict(**common, horizon=horizon, status="error", log_likelihood=None,
                    n_target_tokens=horizon, error=f"{type(error).__name__}: {error}"))
        for row in rows[-len(HORIZONS):]:
            row.update(condition_wall_seconds=time.monotonic()-started,
                       target_token_ids=record["future"][:row["horizon"]])
    raw = {row["horizon"]: row for row in rows if row["mode"] == "raw"}
    for row in rows:
        baseline = raw[row["horizon"]]
        row["gain_over_raw_nats"] = (row["log_likelihood"]-baseline["log_likelihood"]
            if row["status"] == baseline["status"] == "ok" else None)
    return rows


def summarize(rows: list[dict], expected_items: list[str], *, policy: str = "sample", crossed: bool = False) -> dict:
    expected = {(key, mode, horizon) for key in expected_items for mode in MODES for horizon in HORIZONS}
    keys = [(row["item_id"], row["mode"], row["horizon"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Every selected item/condition/horizon must be present exactly once")
    for field in ("protocol_sha256", "checkpoint_sha256", "model_identity_sha256", "writer_model_identity_sha256"):
        present = [field in row for row in rows]
        if (any(present) and not all(present)) or (all(present) and len({canonical_hash(row[field]) for row in rows}) != 1):
            raise ValueError(f"Cannot combine mixed or partially tagged local conditions: {field}")
    groups = defaultdict(list)
    def valid(row):
        value = row.get("log_likelihood")
        return row["status"] == "ok" and isinstance(value, (int, float)) and math.isfinite(value)
    for row in rows:
        groups[(row["mode"], row["horizon"])].append(row)
    output = {}
    for (mode, horizon), values in groups.items():
        successful = [r["log_likelihood"] for r in values if valid(r)]
        gains = [r["gain_over_raw_nats"] for r in values if valid(r) and isinstance(r.get("gain_over_raw_nats"), (int, float))
                 and math.isfinite(r["gain_over_raw_nats"])]
        output[f"{mode}_K{horizon}"] = dict(items=len(values), statuses=dict(Counter(r["status"] for r in values)),
            invalid_numeric_outcomes=sum(r["status"] == "ok" and not valid(r) for r in values),
            successful_items=len(successful), paired_successful_items=len(gains),
            mean_log_likelihood_successful_only=sum(successful)/len(successful) if successful else None,
            mean_gain_over_raw_paired_successful_only=sum(gains)/len(gains) if gains else None)
    contrasts = {}
    for horizon in HORIZONS:
        own = {row["item_id"]: row for row in groups[("own", horizon)]}
        for baseline in ("raw", "empty", "filler", "shuffled", "other"):
            controls = {row["item_id"]: row for row in groups[(baseline, horizon)]}
            complete = all(valid(own[key]) and valid(controls[key]) for key in expected_items)
            name = f"own_minus_{baseline}_K{horizon}"
            if not complete or len(expected_items) < 2:
                contrasts[name] = dict(items=len(expected_items), result=None,
                    reason="Incomplete pair or fewer than two documents; no failed item was dropped for a CI")
                continue
            def values(condition):
                return [dict(item_id=key, document_id=condition[key]["document_id"],
                             score=condition[key]["log_likelihood"]) for key in expected_items]
            contrasts[name] = paired_cluster_bootstrap(values(controls), values(own), repeats=5000, seed=17)
    return dict(items=len(expected_items), conditions=output,
        paired_document_contrasts=contrasts,
        primary=("own K=1: independent one-sample mean utility, not an oracle best-of statistic"
                 if policy == "sample" and not crossed else None),
        crossed_writer_reader=crossed,
        writer_policy=policy, greedy_note="Greedy is a separate diagnostic, never the stochastic primary" if policy == "greedy" else None,
        failure_free=all(valid(row) for row in rows),
        failure_note="No numerical value is imputed for execution failures. All failures remain rows/counts; likelihood means condition on successful execution. An incomplete audit cannot establish unconditional primary utility.")


def source_hashes() -> dict:
    return evaluation_dependencies()


def setup_model(args, final_identity: dict | None = None):
    model, tokenizer = load_base(args.model_dir)
    checkpoint = load_own_checkpoint(model, args.checkpoint, final_identity=final_identity) if args.checkpoint else None
    signature = sampled_fingerprint(model)
    sync, set_seed = (lambda: None), torch.manual_seed
    if args.device == "xla":
        from train.runtime_compat import configure_runtime
        configure_runtime()
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = torch_xla.device()
        model.to(device)
        if model.config.tie_word_embeddings:
            model.tie_weights()
        sync = lambda: torch_xla.sync(wait=True)
        sync()
        set_seed = lambda seed: xm.set_rng_state(seed, device)
        if signature != sampled_fingerprint(model):
            raise RuntimeError("Local evaluator model fingerprint changed on transfer")
    defined = set(tokenizer.get_vocab().values())
    if not defined or min(defined) < 0 or max(defined) >= model.config.vocab_size:
        raise ValueError("Tokenizer IDs outside model vocabulary")
    banned = tuple(sorted(set(range(model.config.vocab_size))-defined))
    backend = NeuronCausalBackend(model, sync=sync, banned_ids=banned,
                                  checkpoint_trunk=False, head_amp=nullcontext)
    return backend, tokenizer, checkpoint, signature, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "score"))
    for name in ("model-dir", "records", "selection", "format", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=("B0", "vanilla", "thought"))
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--notes", nargs="+", type=Path)
    parser.add_argument("--policy", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--control-seed", type=int, default=17)
    parser.add_argument("--filler-text", default=" ")
    parser.add_argument("--thought-cap", type=int, default=1024)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "xla"), default="xla")
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--frozen-protocol", type=Path)
    args = parser.parse_args()
    if not 1 <= args.thought_cap <= 1024:
        parser.error("Local audit thought cap must be 1..1024")
    if not args.qualification and args.thought_cap != 1024:
        parser.error("The scientific local audit uses the frozen 1024-action cap")
    if args.device == "cpu" and not args.qualification:
        parser.error("CPU execution is qualification only")
    if args.command == "score" and not args.notes:
        parser.error("Score requires saved --notes writer bundles")
    if not args.qualification and args.arm is None:
        parser.error("Scientific evaluation needs an explicit --arm model identity")
    selection = read_selection(args.selection, qualification=args.qualification, split=args.split)
    ids = selection["item_ids"]
    local_ids = shard_questions(ids, args.shard_index, args.num_shards)
    if not local_ids:
        parser.error("Requested shard is empty")
    fmt = Format.from_dict(json.loads(args.format.read_text())["format"])
    assets = asset_provenance(args.model_dir)
    protocol = dict(schema=SCHEMA, command=args.command, split=selection["split"],
        selection_sha256=sha256_file(args.selection), selection_item_ids_sha256=selection["item_ids_sha256"],
        records_sha256=selection["records_sha256"], thought_format=fmt.__dict__,
        thought_cap=args.thought_cap, compute="fp32", head_precision="fp32",
        model_id=assets.get("model_id"), model_revision=assets.get("revision"),
        tokenizer_and_config_sha256=canonical_hash({name:value for name,value in assets["files"].items()
            if not name.endswith((".safetensors", ".bin"))}), dependencies=source_hashes())
    notes = controls = writer = None
    if args.command == "generate":
        protocol.update(selection_policy=args.policy, temperature=.7, seed=args.seed,
            seed_rule="record-local item_seed(seed,item_id); invariant to shards", support="tokenizer_defined_ids_only")
    else:
        notes, writer = load_note_bundles(args.notes, selection, fmt)
        protocol.update(modes=list(MODES), horizons=list(HORIZONS), filler_text=args.filler_text,
            control_seed=args.control_seed, writer_protocol=writer["protocol"],
            writer_model_identity=model_binding(writer["model"], writer["checkpoint"], writer["final_model_identity"]),
            other_note="next global selected item, cyclic repeat/trim to recipient content length; empty donor uses filler")
    protocol = json.loads(json.dumps(protocol))
    final_identity = None
    if not args.qualification:
        require_frozen_protocol(args.frozen_protocol, protocol)
        final_identity = require_final_model(args.frozen_protocol, args.checkpoint, assets,
                                             arm=args.arm, seed=args.model_seed)
        if writer is not None:
            writer_identity = writer["final_model_identity"]
            if not isinstance(writer_identity, dict):
                raise ValueError("Scientific reader needs a qualified frozen writer identity")
            writer_checkpoint = writer["checkpoint"]
            verified_writer = require_final_model(args.frozen_protocol,
                Path(writer_checkpoint["path"]) if writer_checkpoint else None,
                writer["model"], arm=writer_identity["arm"], seed=writer_identity["seed"])
            if verified_writer != writer_identity:
                raise ValueError("Saved writer identity differs from the frozen inventory")
    # Futures are read later, after model validation and global control creation.
    records = load_selected(args.records, selection, include_future=False)
    by_id = {row["item_id"]: row for row in records}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_utc=utc_now(), qualification=args.qualification,
        protocol=protocol, protocol_sha256=canonical_hash(protocol), model=assets,
        global_expected_items=len(ids), expected_items=len(local_ids),
        shard=dict(index=args.shard_index, count=args.num_shards), writer=writer,
        final_model_identity=final_identity)
    atomic_json(args.output/"run.json", report)
    try:
        backend, tokenizer, checkpoint, signature, set_seed = setup_model(args, final_identity)
        report.update(checkpoint=checkpoint, model_fingerprint=signature,
                      visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"))
        reader_identity = model_binding(assets, checkpoint, final_identity)
        writer_identity = protocol.get("writer_model_identity")
        crossed = writer_identity is not None and writer_identity != reader_identity
        report["crossed_writer_reader"] = crossed
        defined = set(tokenizer.get_vocab().values())
        format_ids = (*fmt.header_ids, *fmt.open_ids, *fmt.close_ids, fmt.stop_id, fmt.pad_id, *fmt.additional_stop_ids)
        if not set(format_ids) <= defined:
            raise ValueError("Thought format contains undefined token IDs")
        prefix_width = len(records[0]["prefix"])
        width = prefix_width+len(fmt.header_ids)+len(fmt.open_ids)+args.thought_cap+len(fmt.close_ids)+max(HORIZONS)
        if width > backend.model.config.max_position_embeddings:
            raise ValueError("Local audit exceeds model context")
        if args.command == "score":
            filler = tokenizer.encode(args.filler_text, add_special_tokens=False)
            if not filler or not set(filler) <= defined or set(filler) & {fmt.stop_id, *fmt.additional_stop_ids}:
                raise ValueError("Filler needs nonempty tokenizer-defined, non-terminating tokens")
            # Controls depend only on the complete fixed note bundle and RNG.
            controls = construct_controls(selection, notes, filler_ids=filler, seed=args.control_seed)
            report.update(filler_ids=filler, scoring_width=width)
            if writer["protocol"]["thought_cap"] != args.thought_cap:
                raise ValueError("Reader cap must match the saved writer protocol")
            if writer["protocol"]["tokenizer_and_config_sha256"] != protocol["tokenizer_and_config_sha256"]:
                raise ValueError("Saved writer and reader tokenizers differ")
            # Only the reader now obtains targets, after prefix-only controls.
            by_id = {row["item_id"]: row for row in load_selected(args.records, selection, include_future=True)}
        atomic_json(args.output/"run.json", report, overwrite=True)
        output = args.output/("notes.jsonl" if args.command == "generate" else "scores.jsonl")
        rows, started = [], time.monotonic()
        with output.open("x") as stream:
            for key in local_ids:
                record = by_id[key]
                if args.command == "generate":
                    note_started = time.monotonic()
                    try:
                        result = generate_note(backend, record["prefix"], fmt=fmt, cap=args.thought_cap,
                            policy=args.policy, seed=item_seed(args.seed, key), set_seed=set_seed)
                        result["text"] = tokenizer.decode(result["content_ids"], skip_special_tokens=False,
                                                           clean_up_tokenization_spaces=False)
                    except Exception as error:
                        result = dict(status="error", error=f"{type(error).__name__}: {error}",
                            actions=[], content_ids=[], prefix_sha256=canonical_hash(record["prefix"]),
                            n_actions=0, n_content_tokens=0, item_seed=item_seed(args.seed, key))
                    result.update(item_id=key, document_id=record["document_id"], split=record["split"])
                    result["wall_seconds"] = time.monotonic()-note_started
                    current = [result]
                else:
                    current = score_record(record, controls[key], notes[key], backend=backend, fmt=fmt, total_width=width,
                        first_target_text=tokenizer.decode([record["future"][0]], skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False))
                for row in current:
                    row.update(protocol_sha256=report["protocol_sha256"],
                        checkpoint_sha256=(checkpoint or {}).get("sha256"),
                        model_identity_sha256=canonical_hash(reader_identity))
                    if args.command == "score":
                        row["writer_model_identity_sha256"] = canonical_hash(writer_identity)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.extend(current)
                print(json.dumps(dict(item_id=key, statuses=dict(Counter(r["status"] for r in current)))), flush=True)
        if args.command == "score":
            report["summary"] = summarize(rows, local_ids, policy=writer["protocol"]["selection_policy"], crossed=crossed)
            with (args.output/"controls.jsonl").open("x") as stream:
                for key in local_ids:
                    stream.write(json.dumps(dict(item_id=key, **controls[key]))+"\n")
            report["controls_sha256"] = sha256_file(args.output/"controls.jsonl")
        else:
            report["summary"] = dict(items=len(rows), statuses=dict(Counter(r["status"] for r in rows)),
                sampled_actions=sum(r["n_actions"] for r in rows))
        report.update(status="complete", finished_utc=utc_now(), wall_seconds=time.monotonic()-started,
            **{("notes_sha256" if args.command == "generate" else "scores_sha256"): sha256_file(output)})
    except BaseException as error:
        report.update(status="failed", finished_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output/"run.json", report, overwrite=True)


if __name__ == "__main__":
    main()
