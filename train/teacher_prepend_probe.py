"""Score every closed TRAIN16 teacher note under the unchanged B0 prepend reader.

Hindsight writing observes future16. Its measured utility is training-label
information, never evidence for a deployable prefix-only writer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from train import anchor_scan, hindsight_teacher as teacher
from train.evaluate_local import setup_model
from train.evaluate_likelihood import score_continuations
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions, matched_controls
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "teacher-prepend-train16-v1"
TEACHER = ROOT/"local/nvme/diagnostics/hindsight-teacher-train16-v1"
CONTROL = ROOT/"local/diagnostics/hindsight-teacher-train16-v1"
ANCHORS = ROOT/"local/nvme/diagnostics/anchors-train1024-v1"
MODES = ("empty", "own", "filler", "shuffled", "other")
CONDITIONS = teacher.CONDITIONS


def sources():
    return {**teacher.sources(), "train/anchor_scan.py": sha256_file(anchor_scan.__file__),
            "train/teacher_prepend_probe.py": sha256_file(__file__)}


def read(path):
    return json.loads(Path(path).read_text())


def finite_vector(values, length):
    return (isinstance(values, list) and len(values) == length
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


def validate_notes(p, inputs, report, all_notes, defined):
    """Validate full discrete support, causal prompts, stops, coverage and both gates."""
    ids, documents = p["selection"]["item_ids"], p["selection"]["document_ids"]
    if (p["schema"] != teacher.SCHEMA or p["source_files"] != teacher.sources()
            or p["conditions"] != list(CONDITIONS) or p["selection"]["split"] != "train"
            or p["selection"]["count"] != 16 or len(ids) != 16 or len(set(ids)) != 16
            or len(documents) != 16 or len(set(documents)) != 16
            or p["selection"]["item_ids_sha256"] != canonical_hash(ids)
            or p["future_exposure"] != dict(prefix_only=False, hindsight=True)
            or p["total_notes"] != 32 or p["examples_per_condition"] != 16
            or p["batch_size"] != 4 or p["group_size"] != 1
            or p["cap"] != 256 or p["temperature"] != .7 or p["compute"] != "fp32"
            or p["baseline_atol"] != .002 or p["replay_atol"] != .001
            or not p["native_vocab"]["equal"] or len(defined) != p["native_vocab"]["entries"]):
        raise ValueError("Changed teacher recipe, selection, native support or source")
    if (set(all_notes) != set(CONDITIONS)
            or Format.from_dict(p["format"]) != teacher.teacher_format(inputs["template"]["token_segments"][-1][-1])):
        raise ValueError("Teacher stop format or condition set changed")
    parent = p["parent_selection"]
    expected_selection = dict(parent, count=16, item_ids=parent["item_ids"][:16],
                              document_ids=parent["document_ids"][:16], item_ids_sha256=canonical_hash(ids))
    if parent["count"] != 128 or p["selection"] != expected_selection:
        raise ValueError("Teacher must retain exactly the first16 fixed TRAIN128 anchors")
    if (report["status"] != "complete" or report["notes"] != 32
            or set(report["forward_checks"]) != set(CONDITIONS)
            or set(report["replay_checks"]) != set(CONDITIONS)):
        raise ValueError("All32 closed teacher notes and both condition gates are required")
    for field, tolerance in (("forward_checks", .002), ("replay_checks", .001)):
        for condition in CONDITIONS:
            gate = report[field][condition]
            if gate["atol"] != tolerance or not finite_vector([gate["max_error"]], 1) or not 0 <= gate["max_error"] <= tolerance:
                raise ValueError("Teacher CPU/XLA or cached/replay gate failed")
    rows = inputs["rows"]
    if (len(rows) != 16 or [r["item_id"] for r in rows] != ids
            or [r["document_id"] for r in rows] != documents
            or any(r["split"] != "train" or len(r["prefix"]) != 512 or len(r["future"]) != 16
                   or any(type(v) is not int or v not in defined for v in r["prefix"]+r["future"]) for r in rows)):
        raise ValueError("Teacher full input coverage differs from the fixed TRAIN16")
    for condition in CONDITIONS:
        notes = all_notes[condition]
        if len(notes) != 16 or [n["item_id"] for n in notes] != ids:
            raise ValueError("Every condition must retain all16 ordered notes")
        for i, (row, note) in enumerate(zip(rows, notes)):
            prompt = teacher.build_prompt(row["prefix"], row["future"] if condition == "hindsight" else None,
                                          inputs["template"]["token_segments"])
            actions, mask, lp = note["actions"], note["action_mask"], note["behavior_logps"]
            if (inputs["prompts"][condition][i] != prompt or len(prompt) != p["prompt_width"]
                    or note["prompt_sha256"] != canonical_hash(prompt)
                    or note["document_id"] != row["document_id"] or note["split"] != "train"
                    or note["condition"] != condition or note["future_exposed"] != (condition == "hindsight")
                    or note["batch"] != i//4 or note["batch_row"] != i%4
                    or note["batch_seed"] != teacher.batch_seed(ids[(i//4)*4:(i//4+1)*4], i//4)
                    or note["batch_seed"] != p["batch_seeds"][i//4]
                    or not 1 <= len(actions) <= 256 or any(type(v) is not int or v not in defined for v in actions)
                    or not finite_vector(lp, len(actions)) or len(mask) != 256
                    or any(type(v) is not bool for v in mask) or mask != [True]*len(actions)+[False]*(256-len(actions))):
                raise ValueError("Invalid note support, native prompt, provenance, mask or logps")
            stopped = actions[-1] in (151645, 151643)
            if (any(v in (151645, 151643) for v in actions[:-1])
                    or (not stopped and len(actions) != 256)
                    or note["stop_type"] != (f"token:{actions[-1]}" if stopped else "length_cap")
                    or note["content_ids"] != content_from_actions(actions, 151645, (151643,))
                    or type(note["decode_iterations"]) is not int or not len(actions) <= note["decode_iterations"] <= 256
                    or note["words"] != len(note["text"].split())
                    or note["instruction_word_limit_met"] != (note["words"] <= 150)):
                raise ValueError("Teacher stop/content/decode/word accounting changed")
            gate = report["replay_checks"][condition]["max_error"]
            if (note["first_batch_replay_max_error"] != (gate if i < 4 else None)
                    or not finite_vector([note["sampling_batch_wall_seconds"], note["replay_batch_wall_seconds"]], 2)
                    or note["sampling_batch_wall_seconds"] < 0 or note["replay_batch_wall_seconds"] < 0):
                raise ValueError("Teacher replay/cost accounting changed")
        for first in range(0, 16, 4):
            for field in ("sampling_batch_wall_seconds", "replay_batch_wall_seconds", "decode_iterations"):
                if len({n[field] for n in notes[first:first+4]}) != 1:
                    raise ValueError("A shared B4 teacher cost must be recorded once per batch")


def closed_teacher(directory, control):
    """Small-file closure only; released weight bytes were pinned by teacher loading."""
    paths = dict(teacher_protocol=directory/"protocol.json", teacher_inputs=directory/"inputs.json",
        teacher_run=directory/"generation/run.json", cpu_reference=directory/"cpu-reference/reference.json",
        cpu_logps=directory/"cpu-reference/next-logps.pt", cpu_loading=directory/"cpu-reference/loading-info.pt",
        teacher_release=control/"release.json", teacher_exit=control/"exit.json", teacher_launcher=control/"launch.sh",
        **{f"notes_{c}": directory/"generation"/f"{c}.jsonl" for c in CONDITIONS})
    report = read(paths["teacher_run"])
    if (report["status"] != "complete" or not paths["teacher_exit"].is_file()
            or read(paths["teacher_exit"])["exit_code"] != 0):
        raise ValueError("Require fully closed teacher generation and successful process exit")
    p, inputs = [read(paths[k]) for k in ("teacher_protocol", "teacher_inputs")]
    paths["teacher_selection"] = Path(p["selection_path"])
    reference, release = read(paths["cpu_reference"]), read(paths["teacher_release"])
    if (report["protocol_sha256"] != sha256_file(paths["teacher_protocol"])
            or sha256_file(paths["teacher_inputs"]) != p["inputs_sha256"]
            or sha256_file(paths["teacher_selection"]) != p["selection_sha256"]
            or release["protocol_sha256"] != report["protocol_sha256"]
            or release["cpu_reference_sha256"] != sha256_file(paths["cpu_reference"])
            or reference["status"] != "complete" or reference["protocol_sha256"] != report["protocol_sha256"]
            or reference["logps_sha256"] != sha256_file(paths["cpu_logps"])
            or reference["loading_info_sha256"] != sha256_file(paths["cpu_loading"])
            or any(reference["loading_info"].values()) or not reference["tied"] or reference["dtype"] != "float32"
            or reference["signature"] != report.get("signature") or reference["native_vocab"] != p["native_vocab"]
            or reference["input_item_ids"] != p["selection"]["item_ids"][:4]
            or reference["prompt_hashes"] != {c: canonical_hash(inputs["prompts"][c][:4]) for c in CONDITIONS}):
        raise ValueError("Teacher success/CPU reference/input/release hash binding is incomplete")
    for key, path in (("teacher", Path(p["teacher_dir"])/"ASSET_MANIFEST.json"),
                      ("base", Path(p["base_dir"])/"ASSET_MANIFEST.json")):
        if sha256_file(path) != p[f"{key}_manifest_sha256"]:
            raise ValueError("Released teacher/base asset manifest changed")
        paths[f"{key}_manifest"] = path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p["teacher_dir"], local_files_only=True, trust_remote_code=False)
    vocab = tok.get_vocab()
    if canonical_hash(vocab) != p["native_vocab"]["mapping_sha256"]:
        raise ValueError("Teacher tokenizer-defined support changed")
    config = read(Path(p["teacher_dir"])/"config.json")
    if (reference["banned_ids"] != sorted(set(range(config["vocab_size"]))-set(vocab.values()))
            or reference["shapes"] != {c:[4,config["vocab_size"]] for c in CONDITIONS}):
        raise ValueError("CPU teacher support or reference shapes changed")
    notes = {}
    for condition in CONDITIONS:
        path = paths[f"notes_{condition}"]
        if sha256_file(path) != report["notes_sha256"][condition]:
            raise ValueError("Changed closed teacher note file")
        notes[condition] = [json.loads(line) for line in path.read_text().splitlines()]
        if any(n["text"] != tok.decode(n["content_ids"], skip_special_tokens=False) for n in notes[condition]):
            raise ValueError("Teacher text differs from native content IDs")
    validate_notes(p, inputs, report, notes, set(vocab.values()))
    return p, inputs, notes, report, paths


def raw_references(directory, ids, base_sha):
    p, run, summary = [read(directory/name) for name in ("protocol.json", "scoring/run.json", "scoring/summary.json")]
    paths = dict(anchor_protocol=directory/"protocol.json", anchor_run=directory/"scoring/run.json",
                 anchor_summary=directory/"scoring/summary.json", anchor_scores=directory/"scoring/scores.jsonl")
    if (p["schema"] != anchor_scan.SCHEMA or p["source_sha256"] != anchor_scan.sources()
            or p["model_manifest_sha256"] != base_sha or run["status"] != "complete"
            or summary["status"] != "complete" or run["failures"] != 0 or summary["failures"] != 0
            or run["protocol_sha256"] != sha256_file(paths["anchor_protocol"])
            or summary["protocol_sha256"] != run["protocol_sha256"]
            or run["summary_sha256"] != sha256_file(paths["anchor_summary"])
            or summary["scores_sha256"] != sha256_file(paths["anchor_scores"])):
        raise ValueError("Require complete source-bound original B0 TRAIN raw scores")
    values = {}
    for line in paths["anchor_scores"].read_text().splitlines():
        r = json.loads(line)
        if r["item_id"] in ids:
            if r["item_id"] in values or r["status"] != "ok" or not finite_vector(r.get("token_log_likelihoods"), 16):
                raise ValueError("Invalid or duplicate TRAIN raw reference")
            values[r["item_id"]] = r["token_log_likelihoods"]
    if set(values) != set(ids):
        raise ValueError("Missing fixed TRAIN16 raw reference")
    return values, run["model"], paths


def build_controls(ids, notes, filler_ids, seed):
    rows = []
    for condition in CONDITIONS:
        contents = [n["content_ids"] for n in notes[condition]]
        condition_seed = int(canonical_hash(dict(schema=SCHEMA, seed=seed, condition=condition))[:15], 16)
        controls = matched_controls(contents, filler_ids=filler_ids, seed=condition_seed)
        for i, key in enumerate(ids):
            rows.append(dict(item_id=key, condition=condition, donor_item_id=ids[(i+1)%len(ids)],
                control_seed=condition_seed, donor_rule="next fixed global item within same teacher condition; cyclic repeat/trim; empty donor uses filler",
                content_ids=dict(empty=[], own=contents[i], **{mode:controls[mode][i] for mode in controls})))
    return rows


def prepare(output, teacher_dir=TEACHER, teacher_control=CONTROL, anchors=ANCHORS):
    old, inputs, notes, _, paths = closed_teacher(teacher_dir, teacher_control)
    references, model, raw_paths = raw_references(anchors, old["selection"]["item_ids"], old["base_manifest_sha256"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(old["base_dir"], local_files_only=True, trust_remote_code=False)
    if tok.encode("\n\n", add_special_tokens=False) != [271]:
        raise ValueError("Measured prepend separator changed")
    filler = tok.encode("The passage continues.", add_special_tokens=False)
    controls = build_controls(old["selection"]["item_ids"], notes, filler, 20261018)
    paths.update(raw_paths)
    p = dict(schema=SCHEMA, experiment_id=output.name, prepared_utc=utc_now(), source_files=sources(),
        teacher_dir=str(teacher_dir.resolve()), teacher_control=str(teacher_control.resolve()), anchors=str(anchors.resolve()),
        paths={k:str(v.resolve()) for k,v in paths.items()}, hashes={k:sha256_file(v) for k,v in paths.items()},
        selection=old["selection"], model_dir=old["base_dir"], model=model, checkpoint=None,
        model_manifest_sha256=old["base_manifest_sha256"], conditions=list(CONDITIONS), modes=list(MODES),
        separator_ids=[271], filler_ids=filler, control_seed=20261018, interface="prepend_plain",
        serialization="content + [271] + original512prefix; raw has original512prefix only",
        horizons=[1,16], width=1024, compute="fp32", raw_reference_atol=.002,
        expected_items=16, expected_notes=32, expected_score_rows=176, bootstrap_repeats=5000, bootstrap_seed=17,
        limitations=["TRAIN target-surprisal selected anchors; no held-out utility claim.",
            "Hindsight teacher receives truefuture16 during writing: label-generation information, never deployment evidence.",
            "Prefix-only teacher is external 4B assistance, not 1.7B self-generated reasoning.",
            "All32 notes retained, including empty/capped/over150word notes; no filtering or resampling.",
            "Copying statistics are descriptive and do not change denominators or label selection."])
    output.mkdir(parents=True, exist_ok=False)
    (output/"controls.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in controls))
    p["controls_sha256"] = sha256_file(output/"controls.jsonl")
    atomic_json(output/"protocol.json", p)
    return p


def load_inputs(output):
    p = read(output/"protocol.json")
    if (p["schema"] != SCHEMA or p["source_files"] != sources() or p["compute"] != "fp32"
            or p["width"] != 1024 or p["horizons"] != [1,16] or p["separator_ids"] != [271]
            or p["conditions"] != list(CONDITIONS) or p["modes"] != list(MODES)
            or p["raw_reference_atol"] != .002 or p["expected_score_rows"] != 176
            or p["bootstrap_repeats"] != 5000 or p["bootstrap_seed"] != 17):
        raise ValueError("Teacher reader source or fixed recipe changed")
    if any(sha256_file(path) != p["hashes"][key] for key,path in p["paths"].items()):
        raise ValueError("Closed teacher/input/raw reference changed")
    old, inputs, notes, teacher_run, _ = closed_teacher(Path(p["teacher_dir"]), Path(p["teacher_control"]))
    refs, model, _ = raw_references(Path(p["anchors"]), p["selection"]["item_ids"], p["model_manifest_sha256"])
    controls = [json.loads(line) for line in (output/"controls.jsonl").read_text().splitlines()]
    if (old["selection"] != p["selection"] or model != p["model"]
            or sha256_file(output/"controls.jsonl") != p["controls_sha256"]
            or controls != build_controls(p["selection"]["item_ids"], notes, p["filler_ids"], p["control_seed"])):
        raise ValueError("Teacher selection/B0/global controls changed")
    return p, inputs, notes, teacher_run, {(r["item_id"],r["condition"]):r for r in controls}, refs


def contexts(prefix, key, controls):
    result = [("raw", "raw", list(prefix))]
    result += [(condition, mode, controls[key,condition]["content_ids"][mode]+[271]+list(prefix))
               for condition in CONDITIONS for mode in MODES]
    return result


def score(output, device):
    p, inputs, _, _, controls, refs = load_inputs(output)  # Complete all32 barrier precedes reader setup.
    backend, _, checkpoint, model, _ = setup_model(SimpleNamespace(model_dir=Path(p["model_dir"]), checkpoint=None, device=device))
    if checkpoint is not None or model != p["model"]:
        raise ValueError("Reader must be unchanged released B0")
    d = output/"scoring"; d.mkdir(exist_ok=False)
    report = dict(schema=SCHEMA, status="running", started_utc=utc_now(), model=model, checkpoint=None,
                  protocol_sha256=sha256_file(output/"protocol.json"), rows=0, failures=0)
    atomic_json(d/"run.json", report); maximum = 0.; raw_count = 0
    try:
        with (d/"scores.jsonl").open("x") as stream:
            for record in inputs["rows"]:
                for condition, mode, context in contexts(record["prefix"], record["item_id"], controls):
                    before = time.monotonic()
                    row = dict(item_id=record["item_id"], document_id=record["document_id"], split="train",
                               interface="raw" if condition == "raw" else "prepend_plain",
                               condition=condition, mode=mode, context_tokens=len(context),
                               context_sha256=canonical_hash(context), prefix_sha256=canonical_hash(record["prefix"]),
                               future_sha256=canonical_hash(record["future"]))
                    try:
                        values = score_continuations(backend, context, [record["future"]], total_width=1024,
                                                     future_width=16, pad_id=151643)[0].token_log_likelihoods
                        if not finite_vector(values, 16):
                            raise ValueError("Require sixteen finite token likelihoods")
                        row.update(status="ok", token_log_likelihoods=values)
                        if condition == "raw":
                            error = max(abs(a-b) for a,b in zip(values,refs[record["item_id"]]))
                            maximum = max(maximum,error); raw_count += 1; row["raw_reference_max_token_error"] = error
                            if error > .002:
                                raise ValueError("Original B0 raw parity failed")
                    except Exception as exc:
                        row.update(status="error",error=f"{type(exc).__name__}: {exc}")
                        report["failures"] += 1
                    row["wall_seconds"] = time.monotonic()-before
                    stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush(); report["rows"] += 1
                print(json.dumps(dict(items=report["rows"]//11, rows=report["rows"], failures=report["failures"], utc=utc_now())), flush=True)
        if report["rows"] != 176 or raw_count != 16 or report["failures"]:
            raise RuntimeError("Complete teacher reader attempted; failed rows retained, no retry/filtering")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        report.update(finished_utc=utc_now(), raw_reference_rows=raw_count, raw_reference_max_token_error=maximum,
                      scores_sha256=sha256_file(d/"scores.jsonl") if (d/"scores.jsonl").exists() else None)
        atomic_json(d/"run.json", report, overwrite=True)


def copying(content, future):
    """Token-exact future overlap; never a retention criterion."""
    longest = 0; previous = [0]*(len(future)+1)
    for token in content:
        current = [0]
        for j, target in enumerate(future):
            value = previous[j]+1 if token == target else 0
            current.append(value); longest = max(longest,value)
        previous = current
    result = dict(longest_contiguous_copied_span=longest)
    for size in (4,8):
        grams = {tuple(future[i:i+size]) for i in range(len(future)-size+1)}
        count = sum(tuple(content[i:i+size]) in grams for i in range(max(0,len(content)-size+1)))
        result[f"future_{size}gram_note_occurrences"] = count
        result[f"has_future_{size}gram"] = bool(count)
    return result


def reduce_rows(p, rows, inputs, notes):
    ids = p["selection"]["item_ids"]; documents = dict(zip(ids,p["selection"]["document_ids"]))
    expected = {(key,"raw","raw") for key in ids} | {(key,c,m) for key in ids for c in CONDITIONS for m in MODES}
    index = {(r["item_id"],r["condition"],r["mode"]):r for r in rows}
    if (len(index) != len(rows) or set(index) != expected
            or any(r["document_id"] != documents[r["item_id"]] or r["split"] != "train" for r in rows)):
        raise ValueError("Require exact all16 by all11 reader coverage; no duplicate/filtering")
    failures = sum(r["status"] != "ok" for r in rows)
    if any(r["status"] == "ok" and not finite_vector(r.get("token_log_likelihoods"),16) for r in rows):
        raise ValueError("Nonfinite/incomplete successful teacher score")
    def values(condition, mode, horizon):
        return [math.fsum(index[key,condition,mode]["token_log_likelihoods"][:horizon]) for key in ids]
    def contrast(a,b):
        return paired_cluster_bootstrap([dict(item_id=k,document_id=documents[k],score=v) for k,v in zip(ids,a)],
            [dict(item_id=k,document_id=documents[k],score=v) for k,v in zip(ids,b)], repeats=5000, seed=17)
    absolute, comparisons, perdoc = {}, {}, []
    for i,key in enumerate(ids):
        record = inputs["rows"][i]
        for condition in CONDITIONS:
            note = notes[condition][i]
            perdoc.append(dict(item_id=key, document_id=documents[key], condition=condition,
                future_exposed=condition=="hindsight", content_tokens=len(note["content_ids"]), actions=len(note["actions"]),
                words=note["words"], instruction_word_limit_met=note["instruction_word_limit_met"],
                stop_type=note["stop_type"], empty=not note["content_ids"], copying=copying(note["content_ids"],record["future"]),
                log_likelihoods={f"{mode}/K{h}":math.fsum(index[key,condition,mode]["token_log_likelihoods"][:h])
                    if index[key,condition,mode]["status"]=="ok" else None for mode in MODES for h in (1,16)}))
    if not failures:
        for h in (1,16):
            raw = values("raw","raw",h)
            absolute[f"raw/K{h}"] = statistics.mean(raw)
            for c in CONDITIONS:
                for m in MODES:
                    absolute[f"{c}/{m}/K{h}"] = statistics.mean(values(c,m,h))
                for treatment, baseline in (("empty","raw"), *(("own",m) for m in ("raw","empty","filler","shuffled","other"))):
                    a = raw if baseline == "raw" else values(c,baseline,h)
                    comparisons[f"{c}/{treatment}_minus_{baseline}/K{h}"] = contrast(a,values(c,treatment,h))
            for baseline in ("raw","empty","filler","shuffled","other"):
                delta = {}
                for c in CONDITIONS:
                    a = raw if baseline == "raw" else values(c,baseline,h)
                    delta[c] = [b-v for b,v in zip(values(c,"own",h),a)]
                comparisons[f"hindsight_minus_prefix_only/own_minus_{baseline}/K{h}"] = contrast(delta["prefix_only"],delta["hindsight"])
            comparisons[f"hindsight_minus_prefix_only/own_absolute/K{h}"] = contrast(values("prefix_only","own",h),values("hindsight","own",h))
    descriptions = {}
    for c in CONDITIONS:
        subset = [r for r in perdoc if r["condition"] == c]
        descriptions[c] = dict(items=16, empty=sum(r["empty"] for r in subset), stops=dict(Counter(r["stop_type"] for r in subset)),
            content_tokens_sum=sum(r["content_tokens"] for r in subset), mean_content_tokens=statistics.mean(r["content_tokens"] for r in subset),
            mean_words=statistics.mean(r["words"] for r in subset), over150words=sum(not r["instruction_word_limit_met"] for r in subset),
            documents_copying_4gram=sum(r["copying"]["has_future_4gram"] for r in subset),
            documents_copying_8gram=sum(r["copying"]["has_future_8gram"] for r in subset),
            maximum_contiguous_copied_span=max(r["copying"]["longest_contiguous_copied_span"] for r in subset))
    return dict(schema=SCHEMA, experiment_id=p["experiment_id"], status="incomplete" if failures else "complete",
        split="train", expected_items=16, expected_notes=32, expected_rows=176, rows=len(rows), failures=failures,
        absolute_mean_log_likelihoods=absolute, paired_document_contrasts=comparisons, descriptions=descriptions, per_document=perdoc,
        failure_policy="All16 remain denominators; no complete-case contrasts if any row fails.",
        interpretation="External teacher TRAIN diagnostic. Hindsight saw target16 during note generation; never deployed-policy evidence.")


def summarize(output):
    p, inputs, notes, teacher_run, _, _ = load_inputs(output)
    d = output/"scoring"; run = read(d/"run.json"); path = d/"scores.jsonl"
    if (run["schema"] != SCHEMA or run["status"] not in ("complete","failed") or run["model"] != p["model"]
            or run["checkpoint"] is not None or run["protocol_sha256"] != sha256_file(output/"protocol.json")
            or run["scores_sha256"] != sha256_file(path)):
        raise ValueError("Require closed, hash-bound actual B0 score output")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = reduce_rows(p,rows,inputs,notes)
    if result["failures"] != run["failures"] or len(rows) != run["rows"]:
        raise ValueError("Closed score row/failure counters disagree")
    batches = {(c,n["batch"]):n for c in CONDITIONS for n in notes[c]}
    result.update(completed_utc=utc_now(), protocol_sha256=sha256_file(output/"protocol.json"),
        scores_sha256=sha256_file(path), run_sha256=sha256_file(d/"run.json"),
        teacher_run_sha256=p["hashes"]["teacher_run"], raw_reference_rows=run["raw_reference_rows"],
        raw_reference_max_token_error=run["raw_reference_max_token_error"],
        cost=dict(new_sampling_seconds=0, reused_teacher_batches=len(batches),
            reused_sampling_batch_seconds_counted_once=sum(n["sampling_batch_wall_seconds"] for n in batches.values()),
            reused_replay_audit_batch_seconds_counted_once=sum(n["replay_batch_wall_seconds"] for n in batches.values()),
            reused_forward_gate_seconds=sum(g["wall_seconds"] for g in teacher_run["forward_checks"].values()),
            new_reader_row_wall_seconds_sum=sum(r["wall_seconds"] for r in rows),
            new_reader_process_seconds=(datetime.fromisoformat(run["finished_utc"])-datetime.fromisoformat(run["started_utc"])).total_seconds()))
    atomic_json(output/"summary.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","score","summarize")); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--teacher",type=Path,default=TEACHER); parser.add_argument("--teacher-control",type=Path,default=CONTROL)
    parser.add_argument("--anchors",type=Path,default=ANCHORS); parser.add_argument("--device",choices=("cpu","xla"),default="xla")
    a = parser.parse_args()
    if a.command == "prepare":
        p = prepare(a.output,a.teacher,a.teacher_control,a.anchors)
        print(json.dumps(dict(status="prepared_only",items=16,notes=32,score_contexts=p["expected_score_rows"])))
    elif a.command == "score": score(a.output,a.device)
    else: summarize(a.output)


if __name__ == "__main__":
    main()
