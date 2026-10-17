"""Fixed TRAIN16 external-teacher pilot; no reader utility or student training.

Preparation and native HF CPU reference are separate from accelerator generation.
Only hindsight labels expose the observed continuation; both conditions are saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.evaluate_local import load_selected, read_selection, setup_model
from train.evaluation_integrity import evaluation_dependencies
from train.neuron_backend import CausalCache
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.core import Format, content_from_actions, policy_log_probs
from weekend_thinking.pretrained import sampled_fingerprint

SCHEMA = "hindsight-teacher-train16-v1"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CONDITIONS = ("prefix_only", "hindsight")
PREFIX, FUTURE = "__NATIVE_PREFIX_53a0e0__", "__NATIVE_FUTURE_3f8e5b__"
SYSTEM = ("Write a compact mathematical reasoning note, at most 150 words, explaining useful "
          "relationships and deductions for continuing the passage. The continuation field may "
          "be blank: infer useful deductions from the passage if blank, and explain the supplied "
          "continuation if provided. Do not restate the question, quote the continuation, add "
          "boilerplate, or address the user. Return only the reasoning note.")


def sources():
    result = evaluation_dependencies()
    root = Path(__file__).resolve().parents[1]
    for name in ("train/hindsight_teacher.py", "weekend_thinking/runtime_policy.py"):
        result[name] = sha256_file(root/name)
    return result


def native_vocab(base, teacher):
    vocab = teacher.get_vocab()
    if base.get_vocab() != vocab:
        raise ValueError("Native base/teacher vocab mappings differ; cannot insert source token IDs")
    if teacher.encode("\n", add_special_tokens=False) != [198]:
        raise ValueError("The fixed blank continuation must be sixteen newline tokens (198)")
    if (teacher.eos_token_id, teacher.pad_token_id) != (151645, 151643):
        raise ValueError("Teacher EOS/pad identity changed")
    return dict(equal=True, entries=len(vocab), mapping_sha256=canonical_hash(vocab))


def prompt_parts(tokenizer):
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=
        f"Passage:\n{PREFIX}\nObserved continuation (may be blank):\n{FUTURE}\nWrite only the reasoning note.")]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if rendered.count(PREFIX) != 1 or rendered.count(FUTURE) != 1:
        raise ValueError("Chat template must preserve exactly one ordered pair of placeholders")
    first, rest = rendered.split(PREFIX)
    if FUTURE not in rest:
        raise ValueError("Chat template changed prefix/future order")
    middle, last = rest.split(FUTURE)
    parts = [tokenizer.encode(text, add_special_tokens=False) for text in (first, middle, last)]
    if not all(parts):
        raise ValueError("Nonempty constant chat segments required")
    return dict(messages=messages, rendered=rendered, chat_template=tokenizer.chat_template,
                text_segments=[first, middle, last], token_segments=parts,
                template_arguments=dict(enable_thinking=False, add_generation_prompt=True))


def build_prompt(prefix, future, parts):
    """None inserts only fixed blanks; this function accepts no row or hidden future."""
    if len(prefix) != 512 or (future is not None and len(future) != 16):
        raise ValueError("Native prompt requires prefix512 and optional future16")
    a, b, c = parts
    continuation = [198]*16 if future is None else list(future)
    return list(a)+list(prefix)+list(b)+continuation+list(c)


def batch_seed(ids, index, seed=20261018):
    # Matched across teacher conditions; batch row streams remain distinct.
    return int(canonical_hash(dict(schema=SCHEMA, seed=seed, batch=index, item_ids=ids))[:15], 16)


def teacher_format(last_token):
    return Format((last_token,), (151645,), 151645, 151643, additional_stop_ids=(151643,))


def next_logp_error(actual, reference, defined):
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[0] != 4:
        raise ValueError("Teacher reference requires matching B4/vocabulary matrices")
    selected = actual[:, defined]-reference[:, defined]
    if not torch.isfinite(selected).all():
        raise RuntimeError("Nonfinite defined-vocabulary teacher logps")
    return selected.abs().max().item()


def verify_teacher_assets(directory):
    manifest = json.loads((directory/"ASSET_MANIFEST.json").read_text())
    if manifest["model_id"] != "Qwen/Qwen3-4B" or manifest["resolved_revision"] != REVISION:
        raise ValueError("Teacher snapshot identity changed")
    for name, record in manifest["files"].items():
        if sha256_file(directory/name) != record["sha256"]:
            raise ValueError(f"Changed released teacher asset: {name}")


def prepare(output, teacher_dir, base_dir, selection_path):
    from transformers import AutoTokenizer
    verify_teacher_assets(teacher_dir)
    tok = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True, trust_remote_code=False)
    base = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    vocab, parts = native_vocab(base, tok), prompt_parts(tok)
    selection = read_selection(selection_path, qualification=True, split="train")
    subset = {**selection, "item_ids": selection["item_ids"][:16],
              "document_ids": selection["document_ids"][:16], "count": 16}
    subset["item_ids_sha256"] = canonical_hash(subset["item_ids"])
    if selection["count"] != 128 or len(subset["item_ids"]) != 16:
        raise ValueError("Use the first sixteen of the fixed TRAIN128 pool")
    rows = load_selected(Path(selection["records_file"]), subset, include_future=True)
    prompts = {condition: [build_prompt(row["prefix"], row["future"] if condition == "hindsight" else None,
                         parts["token_segments"]) for row in rows] for condition in CONDITIONS}
    width = len(prompts["prefix_only"][0])
    if any(len(x) != width for values in prompts.values() for x in values):
        raise ValueError("All teacher prompts must have identical width")
    start = len(parts["token_segments"][0]); future_start = start+512+len(parts["token_segments"][1])
    role_markers = [dict(position=offset+i, following_token=segment[i+1])
        for segment, offset in zip(parts["token_segments"], (0,start+512,future_start+16))
        for i, token in enumerate(segment[:-1]) if token == 151644]
    inputs = dict(rows=rows, prompts=prompts, template=parts,
                  constant_chat_role_markers=role_markers,
                  spans=dict(prefix=[start, start+512], continuation=[future_start, future_start+16]))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"inputs.json", inputs)
    protocol = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(),
        teacher_dir=str(teacher_dir.resolve()), base_dir=str(base_dir.resolve()),
        teacher_manifest_sha256=sha256_file(teacher_dir/"ASSET_MANIFEST.json"),
        base_manifest_sha256=sha256_file(base_dir/"ASSET_MANIFEST.json"),
        source_files=sources(), selection_path=str(selection_path.resolve()),
        selection_sha256=sha256_file(selection_path), parent_selection=selection, selection=subset,
        inputs_sha256=sha256_file(output/"inputs.json"), native_vocab=vocab, prompt_width=width,
        conditions=list(CONDITIONS), future_exposure=dict(prefix_only=False, hindsight=True),
        examples_per_condition=16, total_notes=32, batch_size=4, group_size=1,
        cap=256, temperature=.7, seed=20261018, compute="fp32",
        format=teacher_format(parts["token_segments"][-1][-1]).__dict__,
        batch_seeds=[batch_seed(subset["item_ids"][i:i+4], i//4) for i in range(0,16,4)],
        support="full tokenizer-defined support; no top-k/top-p or repetition penalty",
        baseline_atol=.002, replay_atol=.001, max_words_instruction=150,
        limitations=["TRAIN pool was target-surprisal selected, not held-out utility.",
            "Hindsight teacher sees future16 for label creation; student generation must not.",
            "Prefix-only 4B teacher is external assistance, not 1.7B self-generated reasoning.",
            "Sampling differs from official recommended teacher settings; no utility claim.",
            "Keep every note, empty/capped/overlength included; no outcome selection or retry."])
    atomic_json(output/"protocol.json", protocol)
    return protocol


def read_protocol(output):
    p = json.loads((output/"protocol.json").read_text())
    if p["schema"] != SCHEMA or p["source_files"] != sources() or p["conditions"] != list(CONDITIONS):
        raise ValueError("Teacher source/protocol changed")
    for file, expected in ((output/"inputs.json", p["inputs_sha256"]),
        (Path(p["selection_path"]), p["selection_sha256"]),
        (Path(p["teacher_dir"])/"ASSET_MANIFEST.json", p["teacher_manifest_sha256"]),
        (Path(p["base_dir"])/"ASSET_MANIFEST.json", p["base_manifest_sha256"])):
        if sha256_file(file) != expected:
            raise ValueError(f"Changed teacher preparation input: {file}")
    verify_teacher_assets(Path(p["teacher_dir"]))
    return p, json.loads((output/"inputs.json").read_text())


@torch.no_grad()
def cpu_reference(output, threads):
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    if not 1 <= threads <= 16:
        raise ValueError("CPU reference is bounded to sixteen threads")
    torch.set_num_threads(threads)
    p, inputs = read_protocol(output); started = time.monotonic()
    directory = output/"cpu-reference"; directory.mkdir(exist_ok=False)
    print(json.dumps(dict(phase="cpu_teacher_load", utc=utc_now(), threads=threads)), flush=True)
    model, info = AutoModelForCausalLM.from_pretrained(p["teacher_dir"], local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Non-exact teacher load: {info}")
    torch.save(info, directory/"loading-info.pt")  # Preserve installed set/list types exactly.
    model.eval(); model.tie_weights()
    signature = sampled_fingerprint(model)
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if not tied or any(t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError("Teacher must have tied FP32 released parameters")
    tok = AutoTokenizer.from_pretrained(p["teacher_dir"], local_files_only=True, trust_remote_code=False)
    banned = sorted(set(range(model.config.vocab_size))-set(tok.get_vocab().values()))
    values = {}; timings = {}
    for condition in CONDITIONS:
        before = time.monotonic(); x = torch.tensor(inputs["prompts"][condition][:4])
        hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
        values[condition] = policy_log_probs(model.lm_head(hidden[:, -1]), .7, banned).cpu()
        if not torch.isfinite(values[condition]).all():
            raise ValueError("Nonfinite CPU teacher next logps")
        timings[condition] = time.monotonic()-before
        print(json.dumps(dict(phase="cpu_next_logps", condition=condition, seconds=timings[condition], utc=utc_now())), flush=True)
    torch.save(values, directory/"next-logps.pt")
    atomic_json(directory/"reference.json", dict(status="complete", finished_utc=utc_now(),
        protocol_sha256=sha256_file(output/"protocol.json"), torch_version=torch.__version__,
        transformers_version=__version__, threads=threads,
        loading_info={key:sorted(value) if isinstance(value,set) else value for key,value in info.items()},
        loading_info_types={key:type(value).__name__ for key,value in info.items()},
        loading_info_sha256=sha256_file(directory/"loading-info.pt"), signature=signature,
        tied=tied, parameter_tensors=len(list(model.parameters())), dtype="float32", native_vocab=p["native_vocab"],
        input_item_ids=p["selection"]["item_ids"][:4], prompt_hashes={c:canonical_hash(inputs["prompts"][c][:4]) for c in CONDITIONS},
        logps_sha256=sha256_file(directory/"next-logps.pt"), shapes={c:list(v.shape) for c,v in values.items()},
        banned_ids=banned, forward_wall_seconds=timings, total_wall_seconds=time.monotonic()-started,
        method="ordinary native HF eager causal trunk; last-position head only; T=.7/full defined support"))


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= require_utc_deadline(deadline):
        raise RuntimeError("Teacher finite deadline reached; no automatic retry")


@torch.no_grad()
def generate(output, deadline):
    check_deadline(deadline); p, inputs = read_protocol(output)
    reference = json.loads((output/"cpu-reference/reference.json").read_text())
    if (reference["status"] != "complete" or reference["protocol_sha256"] != sha256_file(output/"protocol.json")
            or reference["logps_sha256"] != sha256_file(output/"cpu-reference/next-logps.pt")):
        raise ValueError("Complete, hash-bound CPU reference required before accelerator work")
    baseline = torch.load(output/"cpu-reference/next-logps.pt", weights_only=True)
    directory = output/"generation"; directory.mkdir(exist_ok=False)
    report = dict(status="running", started_utc=utc_now(), protocol_sha256=sha256_file(output/"protocol.json"),
                  deadline_utc=deadline, notes=0, forward_checks={}, replay_checks={})
    atomic_json(directory/"run.json", report)
    try:
        backend, tok, _, signature, set_seed = setup_model(SimpleNamespace(
            model_dir=Path(p["teacher_dir"]), checkpoint=None, device="xla"))
        if signature != reference["signature"]:
            raise ValueError("Teacher CPU/device sampled weight identity changed")
        fmt = Format.from_dict(p["format"]); defined = sorted(tok.get_vocab().values())
        # Both new-architecture forward checks precede ANY generated note.
        for condition in CONDITIONS:
            check_deadline(deadline); before = time.monotonic()
            x = torch.tensor(inputs["prompts"][condition][:4], device=backend.device)
            cache = CausalCache(backend.model.config.num_hidden_layers, x.shape[-1]+256)
            lp = policy_log_probs(backend._next(x, torch.arange(x.shape[-1], device=backend.device), cache), .7, backend.banned_ids)
            backend.sync(); actual = lp.cpu()
            error = next_logp_error(actual, baseline[condition], defined)
            report["forward_checks"][condition] = dict(max_error=error, atol=.002, wall_seconds=time.monotonic()-before)
            atomic_json(directory/"run.json", report, overwrite=True)
            if not math.isfinite(error) or error > .002:
                raise RuntimeError("Teacher CPU/XLA next-logp parity failed before generation")
            del cache, lp, x
        for condition in CONDITIONS:
            with (directory/f"{condition}.jsonl").open("x") as stream:
                for first in range(0,16,4):
                    check_deadline(deadline); prompts = inputs["prompts"][condition][first:first+4]
                    prefix = torch.tensor([v[:-1] for v in prompts], device=backend.device)
                    backend.sync(); set_seed(p["batch_seeds"][first//4]); before = time.monotonic()
                    rollout = backend.sample(prefix, fmt=fmt, group_size=1, max_actions=256, temperature=.7, policy_version=0)
                    backend.sync(); sample_seconds = time.monotonic()-before
                    actions, masks, logps = [v.cpu()[:,0] for v in (rollout.actions, rollout.action_mask, rollout.behavior_logps)]
                    error = None; audit_seconds = 0.
                    if first == 0:
                        before = time.monotonic(); replay = backend.replay(prefix, rollout, fmt=fmt, temperature=.7)
                        backend.sync(); error = torch.where(rollout.action_mask, (replay-rollout.behavior_logps).abs(), 0.).max().cpu().item()
                        audit_seconds = time.monotonic()-before
                        report["replay_checks"][condition] = dict(max_error=error if math.isfinite(error) else repr(error),
                                                                 atol=.001, wall_seconds=audit_seconds)
                    for index, (a, mask, lp) in enumerate(zip(actions, masks, logps)):
                        active = a[mask].tolist(); content = content_from_actions(active, fmt.stop_id, fmt.additional_stop_ids)
                        text = tok.decode(content, skip_special_tokens=False); row = inputs["rows"][first+index]
                        note = dict(item_id=row["item_id"], document_id=row["document_id"], split="train", condition=condition,
                            future_exposed=condition=="hindsight", batch=first//4, batch_row=index,
                            batch_seed=p["batch_seeds"][first//4], prompt_sha256=canonical_hash(prompts[index]),
                            actions=active, action_mask=mask.tolist(),
                            behavior_logps=[v if math.isfinite(v) else repr(v) for v in lp[mask].tolist()], content_ids=content,
                            text=text, words=len(text.split()), instruction_word_limit_met=len(text.split())<=150,
                            stop_type=f"token:{active[-1]}" if active[-1] in (151645,151643) else "length_cap",
                            decode_iterations=rollout.decode_iterations, sampling_batch_wall_seconds=sample_seconds,
                            first_batch_replay_max_error=error if error is None or math.isfinite(error) else repr(error),
                            replay_batch_wall_seconds=audit_seconds)
                        stream.write(json.dumps(note, allow_nan=False)+"\n"); stream.flush(); report["notes"] += 1
                    atomic_json(directory/"run.json", report, overwrite=True)
                    if not torch.isfinite(logps[masks]).all() or (error is not None and (not math.isfinite(error) or error>.001)):
                        raise RuntimeError("Teacher sampled logps/replay failed; partial notes preserved")
                    print(json.dumps(dict(condition=condition, notes=report["notes"], batch_wall_seconds=sample_seconds, utc=utc_now())), flush=True)
        report.update(status="complete", signature=signature, notes_sha256={c:sha256_file(directory/f"{c}.jsonl") for c in CONDITIONS})
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_utc"] = utc_now(); atomic_json(directory/"run.json", report, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "cpu-reference", "generate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-dir", type=Path, default=Path("local/nvme/assets/qwen3-4b-teacher"))
    parser.add_argument("--base-dir", type=Path, default=Path("local/assets/qwen3-1.7b-base"))
    parser.add_argument("--selection", type=Path, default=Path("local/nvme/diagnostics/anchors-train1024-v1/candidate-selection.ids.json"))
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.teacher_dir, args.base_dir, args.selection)
    elif args.command == "cpu-reference":
        cpu_reference(args.output, args.cpu_threads)
    else:
        generate(args.output, args.deadline_utc)


if __name__ == "__main__":
    main()
