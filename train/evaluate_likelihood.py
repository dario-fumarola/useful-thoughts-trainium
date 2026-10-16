"""FP32 secondary ARC/LAMBADA evaluation, with gold separate from writer inputs.

ARC keeps the legacy Question/Answer prompt and leading-space answer choices.
It reports sum-log-likelihood ranking and TOKEN-mean normalization explicitly.
LAMBADA scores every token of its final word and checks every teacher-forced
greedy token, correcting the legacy first-token-only accuracy implementation.
Official test files require an exact frozen protocol; qualification uses ARC
train/validation or explicitly synthetic LAMBADA examples only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

import torch

from train.evaluate import (GenerationFailure, GreedyDecoder, asset_provenance,
                            load_own_checkpoint, require_frozen_protocol, shard_questions)
from train.neuron_backend import NeuronCausalBackend
from train.evaluation_integrity import evaluation_dependencies, require_final_model
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format
from weekend_thinking.pretrained import load_base, sampled_fingerprint

SCHEMA = "secondary_fp32_likelihood_v1"
REVISIONS = {"arc_challenge": "210d026faf9955653af8916fad021475a3f00453",
             "lambada": "900124bf3b8235c6daf21033af9948b3f07346c4"}
ARC_DEVELOPMENT_SHA256 = {
    "train": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
    "validation": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",
}
CONVENTIONS = {
    "arc_challenge": "Question: {question}\\nAnswer:; choices retain one added leading space; raw=sum token logP; normalized=raw/token_count; first index wins exact ties",
    "lambada": "context=text.rsplit(' ',1)[0]; target=' '+last word; sum all target token logP; exact requires every greedy-on-gold-prefix token to match",
    "tokenization": "context and leading-space continuation encoded separately, add_special_tokens=False, matching legacy benchmark definitions",
    "thought": "one greedy prefix-only note, fixed before any continuation scoring; header+context+open+content+close; no target-dependent note selection",
    "likelihood_support": "raw full released-model vocabulary; writer alone masks model-only tokenizer-undefined rows",
    "padding": "contiguous real context+target, right padded; fixed-width prediction positions; mask every non-target continuation slot",
}


def arc_context(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def lambada_parts(text: str) -> tuple[str, str]:
    """Preserve the legacy literal-space split and leading-space target."""
    parts = text.rsplit(" ", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("Expected LAMBADA context and a nonempty final word")
    return parts[0], " " + parts[1]


def read_items(path: Path, *, task: str, qualification: bool) -> list[dict]:
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if {"answer", "answerKey", "gold", "target", "solution"} & row.keys():
            raise ValueError("Task input must be question/context only; keep gold separate")
        if row.get("task") != task or not isinstance(row.get("context"), str) or not row["context"].strip():
            raise ValueError("Task/context mismatch")
        allowed = ({"train", "validation"} if task == "arc_challenge" else {"synthetic"}) if qualification else {"test"}
        if row.get("split") not in allowed:
            raise ValueError(f"Allowed {task} splits for this purpose: {sorted(allowed)}")
        identifier = row.get("item_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Missing or duplicate item ID")
        if task == "arc_challenge":
            choices = row.get("choices")
            if not isinstance(choices, list) or len(choices) < 2 or any(not isinstance(c, str) or not c.strip() for c in choices):
                raise ValueError("ARC needs at least two nonempty candidate continuations")
        seen.add(identifier)
        rows.append(row)
    if not rows:
        raise ValueError("Empty task input")
    return rows


def read_gold(path: Path, items: list[dict], *, task: str) -> dict[str, dict]:
    gold = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = row.get("item_id")
        if not identifier or identifier in gold:
            raise ValueError("Missing or duplicate gold item ID")
        gold[identifier] = row
    if set(gold) != {row["item_id"] for row in items}:
        raise ValueError("Gold/question item sets differ; never drop missing outcomes")
    for item in items:
        row = gold[item["item_id"]]
        if task == "arc_challenge":
            if type(row.get("gold")) is not int or not 0 <= row["gold"] < len(item["choices"]):
                raise ValueError("ARC gold index outside its candidate list")
        elif not isinstance(row.get("target"), str) or not row["target"].strip():
            raise ValueError("LAMBADA needs a nonempty target in its separate gold file")
    return gold


@dataclass
class ContinuationScore:
    log_likelihood: float
    mean_token_log_likelihood: float
    token_log_likelihoods: list[float]
    greedy_token_ids: list[int]
    target_token_ids: list[int]


@torch.no_grad()
def score_continuations(backend, context_ids: list[int], continuations: list[list[int]], *,
                        total_width: int, future_width: int, pad_id: int,
                        head_chunk: int = 16) -> list[ContinuationScore]:
    """All K target tokens, one fixed-size padded graph per candidate.

    Prefix and candidate lengths become tensor data, not shape-changing slices
    on device. Padding enters no valid target's causal past. Only selected token
    logps and argmax IDs cross to the host, never vocabulary-sized tensors.
    """
    if not context_ids or not continuations or min(future_width, head_chunk) < 1:
        raise ValueError("Nonempty context/continuations and positive widths required")
    if len(context_ids) + future_width > total_width:
        raise ValueError("Context plus fixed future width exceeds static scoring width")
    if any(not values or len(values) > future_width for values in continuations):
        raise ValueError("Empty or over-width continuation; never silently truncate targets")
    if any(parameter.dtype != torch.float32 for parameter in backend.model.parameters() if parameter.is_floating_point()):
        raise ValueError("This evaluator requires FP32 model parameters")
    output = []
    for target in continuations:
        ids = torch.full((1, total_width), pad_id, dtype=torch.long)
        ids[0, :len(context_ids)+len(target)] = torch.tensor(context_ids+target)
        targets = torch.full((1, future_width), pad_id, dtype=torch.long)
        targets[0, :len(target)] = torch.tensor(target)
        positions = (torch.arange(future_width) + len(context_ids) - 1)[None]
        active = (torch.arange(future_width) < len(target))[None]
        ids, targets, positions, active = (value.to(backend.device) for value in (ids, targets, positions, active))
        hidden = backend.hidden(ids)
        if hidden.dtype != torch.float32:
            raise ValueError("This evaluator requires FP32 trunk computation")
        selected = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
        logps, greedy = [], []
        for start in range(0, future_width, head_chunk):
            stop = min(start+head_chunk, future_width)
            # FP32 only: no backend autocast or reduced-precision head context.
            logits = backend.model.lm_head(selected[:, start:stop]).float()
            lp = logits.log_softmax(-1).gather(-1, targets[:, start:stop, None])[:, :, 0]
            logps.append(torch.where(active[:, start:stop], lp, 0.0))
            greedy.append(logits.argmax(-1))
        token_logps = torch.cat(logps, -1)
        prediction = torch.cat(greedy, -1)
        backend.sync()
        raw = token_logps.cpu()[0, :len(target)].tolist()
        predicted = prediction.cpu()[0, :len(target)].tolist()
        if not all(math.isfinite(value) for value in raw):
            raise RuntimeError("Nonfinite continuation log likelihood")
        total = sum(raw)
        output.append(ContinuationScore(total, total/len(target), raw, predicted, list(target)))
    return output


def condition_on_note(prefix_ids: list[int], *, decoder, fmt: Format, mode: str,
                      max_context_tokens: int, thought_cap: int) -> tuple[list[int], dict]:
    """Only prefix IDs are accepted; gold and candidate continuations are absent."""
    if mode == "direct":
        return list(prefix_ids), {"thought_actions": [], "thought_content_ids": [],
                                  "thought_stop_type": "not_requested", "n_thought_tokens": 0}
    if mode != "one_thought":
        raise ValueError("Expected direct or one_thought")
    writer = list(fmt.header_ids) + list(prefix_ids) + list(fmt.open_ids)
    thought = decoder.decode(writer, max_new_tokens=thought_cap,
        prefill_width=max_context_tokens+len(fmt.header_ids)+len(fmt.open_ids), pad_id=fmt.pad_id,
        stop_ids=(fmt.stop_id, *fmt.additional_stop_ids), answer_line_stop=False)
    context = writer + thought.content_ids + list(fmt.close_ids)
    return context, {"writer_prompt_ids": writer, "thought_actions": thought.actions,
        "thought_content_ids": thought.content_ids, "thought_text": thought.text,
        "thought_stop_type": thought.stop_type, "n_thought_tokens": len(thought.actions)}


def evaluate_item(item: dict, gold: dict, *, backend, decoder, tokenizer, fmt: Format,
                  mode: str, max_context_tokens: int, thought_cap: int,
                  total_width: int, future_width: int) -> dict:
    row = {key: item[key] for key in ("item_id", "task", "split")}
    row.update(document_id=item.get("document_id", item["item_id"]), mode=mode, status="error",
               correct=0, correct_norm=0, n_thought_tokens=0, thought_actions=[])
    started, stage = time.monotonic(), "context"
    try:
        prefix = tokenizer.encode(item["context"], add_special_tokens=False)
        row.update(original_context_tokens=len(prefix), truncated_context_tokens=max(0, len(prefix)-max_context_tokens))
        prefix = prefix[-max_context_tokens:]
        row["context_ids"] = prefix
        stage = "thought"
        context, note = condition_on_note(prefix, decoder=decoder, fmt=fmt, mode=mode,
                                         max_context_tokens=max_context_tokens, thought_cap=thought_cap)
        row.update(note)
        row["scoring_context_ids"] = context
        # The note is already fixed. Only now are continuations/gold targets
        # introduced to the reader. ARC's gold label is used after all scoring.
        stage = "likelihood"
        texts = item["choices"] if item["task"] == "arc_challenge" else [gold["target"]]
        targets = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        scores = score_continuations(backend, context, targets, total_width=total_width,
                                    future_width=future_width, pad_id=fmt.pad_id)
        row["continuations"] = [asdict(score) for score in scores]
        if item["task"] == "arc_challenge":
            raw = [score.log_likelihood for score in scores]
            normalized = [score.mean_token_log_likelihood for score in scores]
            predicted = max(range(len(scores)), key=raw.__getitem__)
            predicted_norm = max(range(len(scores)), key=normalized.__getitem__)
            label = gold["gold"]
            row.update(gold=label, prediction=predicted, prediction_norm=predicted_norm,
                correct=int(predicted == label), correct_norm=int(predicted_norm == label),
                gold_log_likelihood=raw[label], n_target_tokens=len(targets[label]))
        else:
            score = scores[0]
            exact = score.greedy_token_ids == score.target_token_ids
            row.update(correct=int(exact), correct_norm=int(exact),
                gold_log_likelihood=score.log_likelihood, n_target_tokens=len(score.target_token_ids),
                correct_target_tokens=sum(a == b for a, b in zip(score.greedy_token_ids, score.target_token_ids)),
                greedy_metric="all_target_tokens_on_gold_prefix_exact")
        row["status"] = "ok"
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}", failed_stage=stage)
        if isinstance(error, GenerationFailure):
            row.update(thought_actions=error.partial.actions, thought_content_ids=error.partial.content_ids,
                       n_thought_tokens=len(error.partial.actions))
    row["wall_seconds"] = time.monotonic()-started
    return row


def summarize(rows: list[dict]) -> dict:
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Need nonempty unique item outcomes")
    successful = [row for row in rows if row["status"] == "ok"]
    return {"items": len(rows), "successful_items": len(successful),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "accuracy": sum(row["correct"] for row in rows)/len(rows),
        "accuracy_token_mean_normalized": sum(row["correct_norm"] for row in rows)/len(rows),
        "mean_gold_log_likelihood_successful_only": (sum(row["gold_log_likelihood"] for row in successful)/len(successful)) if successful else None,
        "nll_denominator_note": "Likelihood mean conditions on successful execution; every error remains an accuracy failure and is reported.",
        "thought_actions": sum(row["n_thought_tokens"] for row in rows)}


def prepare_arc_development(parquet: Path, output: Path, *, split: str) -> dict:
    """Export only canonical development columns; no official test extraction."""
    if split not in {"train", "validation"} or not parquet.name.startswith(split+"-"):
        raise ValueError("Preparation is restricted to ARC train/validation Parquet")
    source_hash = sha256_file(parquet)
    if source_hash != ARC_DEVELOPMENT_SHA256[split]:
        raise ValueError("ARC development source differs from the pinned Parquet hash")
    import pyarrow.parquet as pq
    table = pq.read_table(parquet, columns=["id", "question", "choices", "answerKey"])
    output.mkdir(parents=True, exist_ok=False)
    question_path, gold_path = output/"questions.jsonl", output/"gold.jsonl"
    count = 0
    with question_path.open("x") as questions, gold_path.open("x") as gold:
        for source in table.to_pylist():
            labels = source["choices"]["label"]
            require = source["answerKey"] in labels and len(labels) == len(set(labels))
            if not require:
                raise ValueError("Invalid ARC label; refusing to silently omit an item")
            identifier = f"arc_challenge:{split}:{source['id']}"
            row = dict(item_id=identifier, task="arc_challenge", split=split,
                context=arc_context(source["question"]), choices=[" "+text for text in source["choices"]["text"]])
            questions.write(json.dumps(row, ensure_ascii=False)+"\n")
            gold.write(json.dumps(dict(item_id=identifier, gold=labels.index(source["answerKey"])))+"\n")
            count += 1
    manifest = dict(task="arc_challenge", source_revision=REVISIONS["arc_challenge"],
        source_file=str(parquet.resolve()), source_sha256=source_hash, split=split, items=count,
        questions_sha256=sha256_file(question_path), gold_sha256=sha256_file(gold_path), conventions=CONVENTIONS)
    atomic_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-arc", help="Export pinned local ARC development Parquet only")
    prepare.add_argument("--parquet", type=Path, required=True)
    prepare.add_argument("--split", choices=("train", "validation"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--task", choices=tuple(REVISIONS), required=True)
    run.add_argument("--task-file", type=Path, required=True)
    run.add_argument("--gold-file", type=Path, required=True)
    run.add_argument("--task-revision", required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--arm", choices=("B0", "vanilla", "thought"), default="B0")
    run.add_argument("--seed", type=int)
    run.add_argument("--format", type=Path, required=True)
    run.add_argument("--mode", choices=("direct", "one_thought"), default="direct")
    run.add_argument("--max-context-tokens", type=int, default=768)
    run.add_argument("--future-width", type=int, default=128)
    run.add_argument("--thought-cap", type=int, default=1024)
    run.add_argument("--device", choices=("cpu", "xla"), default="xla")
    run.add_argument("--qualification", action="store_true")
    run.add_argument("--frozen-protocol", type=Path)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-arc":
        result = prepare_arc_development(args.parquet, args.output, split=args.split)
        print(json.dumps({key: result[key] for key in ("items", "split", "source_revision")}))
        return
    if min(args.max_context_tokens, args.future_width, args.thought_cap) < 1:
        parser.error("Positive fixed widths/token budgets required")
    if args.device == "cpu" and not args.qualification:
        parser.error("CPU execution is qualification only")
    synthetic = args.qualification and args.task == "lambada" and args.task_revision == "synthetic-lambada-v1"
    if not synthetic and args.task_revision != REVISIONS[args.task]:
        parser.error("Use the pinned canonical task revision, or the declared synthetic LAMBADA fixture identity")
    items = read_items(args.task_file, task=args.task, qualification=args.qualification)
    local = shard_questions(items, args.shard_index, args.num_shards)
    if not local:
        parser.error("Requested shard is empty")
    fmt = Format.from_dict(json.loads(args.format.read_text())["format"])
    assets = asset_provenance(args.model_dir)
    protocol = dict(schema=SCHEMA, task=args.task, task_revision=args.task_revision,
        task_file_sha256=sha256_file(args.task_file), gold_file_sha256=sha256_file(args.gold_file),
        item_ids_sha256=canonical_hash([item["item_id"] for item in items]), mode=args.mode,
        max_context_tokens=args.max_context_tokens, future_width=args.future_width,
        thought_cap=args.thought_cap if args.mode == "one_thought" else 0,
        thought_format=asdict(fmt) if args.mode == "one_thought" else None,
        compute="fp32", head_precision="fp32", conventions=CONVENTIONS,
        model_id=assets.get("model_id"), model_revision=assets.get("revision"),
        tokenizer_and_config_sha256=canonical_hash({name: value for name, value in assets["files"].items()
            if not name.endswith((".safetensors", ".bin"))}),
        dependencies=evaluation_dependencies(), evaluator_sha256=sha256_file(Path(__file__)),
        backend_sha256=sha256_file(Path(__file__).with_name("neuron_backend.py")))
    protocol = json.loads(json.dumps(protocol))
    final_identity = None
    if not args.qualification:
        require_frozen_protocol(args.frozen_protocol, protocol)
        final_identity = require_final_model(args.frozen_protocol, args.checkpoint, assets,
                                             arm=args.arm, seed=args.seed)
    # Gold is parsed only after the official-test freeze gate, and never passed
    # into the writer. Hashing the file above does not inspect its answers.
    gold = read_gold(args.gold_file, items, task=args.task)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_utc=utc_now(), qualification=args.qualification,
        protocol=protocol, protocol_sha256=canonical_hash(protocol), model=assets, model_identity=final_identity,
        expected_items=len(local), global_expected_items=len(items),
        shard=dict(index=args.shard_index, count=args.num_shards), device=args.device)
    atomic_json(args.output/"run.json", report)
    try:
        model, tokenizer = load_base(args.model_dir)
        report["checkpoint"] = load_own_checkpoint(model, args.checkpoint, final_identity=final_identity) if args.checkpoint else None
        signature = sampled_fingerprint(model)
        sync = lambda: None
        if args.device == "xla":
            from train.runtime_compat import configure_runtime
            report["runtime_compatibility"] = configure_runtime()
            import torch_xla
            device = torch_xla.device()
            model.to(device)
            if model.config.tie_word_embeddings:
                model.tie_weights()
            sync = lambda: torch_xla.sync(wait=True)
            sync()
            if signature != sampled_fingerprint(model):
                raise RuntimeError("Model fingerprint changed during device transfer")
            report["visible_cores"] = os.environ.get("NEURON_RT_VISIBLE_CORES")
        report["model_fingerprint"] = signature
        defined = set(tokenizer.get_vocab().values())
        if not defined or min(defined) < 0 or max(defined) >= model.config.vocab_size:
            raise ValueError("Tokenizer IDs exceed released model support")
        if not set((*fmt.header_ids, *fmt.open_ids, *fmt.close_ids, fmt.pad_id,
                    fmt.stop_id, *fmt.additional_stop_ids)) <= defined:
            raise ValueError("Thought/padding format IDs need tokenizer definitions")
        banned = tuple(sorted(set(range(model.config.vocab_size))-defined))
        decoder = GreedyDecoder(model, tokenizer, sync=sync, banned_ids=banned)
        backend = NeuronCausalBackend(model, sync=sync, checkpoint_trunk=False, head_amp=nullcontext)
        width = args.max_context_tokens+args.future_width
        if args.mode == "one_thought":
            width += len(fmt.header_ids)+len(fmt.open_ids)+len(fmt.close_ids)+args.thought_cap
        if width > model.config.max_position_embeddings:
            raise ValueError("Scoring width exceeds released model context")
        report["static_scoring_width"] = width
        rows, started = [], time.monotonic()
        with (args.output/"scores.jsonl").open("x") as stream:
            for item in local:
                result = evaluate_item(item, gold[item["item_id"]], backend=backend, decoder=decoder,
                    tokenizer=tokenizer, fmt=fmt, mode=args.mode, max_context_tokens=args.max_context_tokens,
                    thought_cap=args.thought_cap, total_width=width, future_width=args.future_width)
                result.update(protocol_sha256=report["protocol_sha256"],
                              checkpoint_sha256=(report["checkpoint"] or {}).get("sha256"))
                rows.append(result)
                stream.write(json.dumps(result, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({key: result[key] for key in ("item_id", "status", "wall_seconds")}), flush=True)
        report.update(status="complete", finished_utc=utc_now(), summary=summarize(rows),
            evaluation_wall_seconds=time.monotonic()-started, scores_sha256=sha256_file(args.output/"scores.jsonl"))
        if args.device == "xla":
            from torch_xla.debug import metrics
            (args.output/"xla_metrics.txt").write_text(metrics.metrics_report())
    except BaseException as error:
        report.update(status="failed", finished_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        atomic_json(args.output/"run.json", report, overwrite=True)
        raise
    atomic_json(args.output/"run.json", report, overwrite=True)


if __name__ == "__main__":
    main()
