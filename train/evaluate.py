"""Question-only, single-path GSM8K generation with a static causal cache.

Gold scoring is a separate process (`scripts/score_numeric_answers.py`). Train
items may qualify prompts before freeze; test generation requires a matching
frozen protocol. This is the same path on CPU fixtures and the real accelerator.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Callable

import torch

from train.neuron_backend import CausalCache, causal_mask
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, policy_log_probs
from weekend_thinking.evaluation import numeric_answer
from weekend_thinking.pretrained import load_base, sampled_fingerprint


SCHEMA = "gsm8k_single_greedy_v1"
ANSWER_STOP = "first_complete_hash4_numeric_line_or_eos_v1"
DEFAULT_INSTRUCTION = (
    "Solve the following math word problem. Show your calculation and finish "
    "with a line in the form #### <number>, followed by a newline."
)
READER_SUFFIX_TEMPLATE = (
    "\n\nUse the scratch work to answer the following question. Finish with a "
    "line in the form #### <number>, followed by a newline.\nQuestion: {question}\nAnswer:"
)


def read_questions(path: Path, *, qualification: bool) -> list[dict]:
    """Reject gold-bearing inputs rather than merely hiding fields downstream."""
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if {"answer", "solution", "target", "gold"} & row.keys():
            raise ValueError("Generation task files must contain questions only; keep gold separate")
        if row.get("task") != "gsm8k" or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Expected a nonempty GSM8K question")
        expected_split = "train" if qualification else "test"
        if row.get("split") != expected_split:
            raise ValueError(f"Expected only {expected_split} questions for this evaluation purpose")
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in seen:
            raise ValueError("Missing or duplicate task item ID")
        seen.add(item_id)
        rows.append({"item_id": item_id, "question": row["question"], "split": expected_split,
                     "task": "gsm8k", "document_id": str(row.get("document_id", item_id))})
    if not rows:
        raise ValueError("Empty task file")
    return rows


def shard_questions(questions: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    """Disjoint deterministic slices; validate the whole task before calling."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard index < positive number of shards")
    return questions[shard_index::num_shards]


def validate_prompt_spec(spec: dict, question_ids: set[str]) -> None:
    if not isinstance(spec.get("instruction"), str) or not spec["instruction"].strip():
        raise ValueError("Prompt spec needs an explicit instruction")
    demos = spec.get("demonstrations")
    if not isinstance(demos, list):
        raise ValueError("Prompt demonstrations must be an explicit list")
    seen = set()
    for demo in demos:
        item_id = demo.get("item_id")
        if (demo.get("split") != "train" or not item_id or item_id in question_ids or item_id in seen
                or not isinstance(demo.get("question"), str) or not demo["question"].strip()
                or not isinstance(demo.get("answer"), str) or numeric_answer(demo["answer"]) is None):
            raise ValueError("Demonstrations must be disjoint, uniquely identified TRAIN worked examples")
        seen.add(item_id)


def question_prompt(question: str, spec: dict) -> str:
    """This function accepts no target for the current question."""
    chunks = [spec["instruction"].strip()]
    for demo in spec["demonstrations"]:
        chunks.append(f"Question: {demo['question']}\nAnswer: {demo['answer'].rstrip()}")
    chunks.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(chunks)


def reader_suffix(question: str) -> str:
    """Repeat only prefix information after thought close, never a target."""
    return READER_SUFFIX_TEMPLATE.format(question=question)


def first_answer_line_end(text: str) -> int | None:
    """A format stop, using generated text only; no numerical correctness test."""
    end = 0
    for line in text.splitlines(keepends=True):
        end += len(line)
        if line.endswith(("\n", "\r")) and numeric_answer(text[:end]) is not None:
            return end
    return None


@dataclass
class Decoded:
    actions: list[int]          # Includes a generated stop token, if any.
    content_ids: list[int]      # Excludes that token; never includes padding.
    raw_text: str
    text: str
    stop_type: str


class GenerationFailure(RuntimeError):
    def __init__(self, error: Exception, partial: Decoded):
        super().__init__(f"{type(error).__name__}: {error}")
        self.partial = partial


class GreedyDecoder:
    """Fixed prefill and one-token decode shapes; vocabulary never leaves device.

    Prefill padding is after the real prefix. Decode writes at its actual end,
    overwriting padded cache rows with consecutive real tokens. Scalar token
    transfers permit prompt/format-only early stopping and are included in time.
    """
    def __init__(self, model, tokenizer, *, amp: Callable = nullcontext,
                 sync: Callable = lambda: None, banned_ids: tuple[int, ...] = (),
                 head_amp: Callable | None = None):
        if model.config.model_type != "qwen3" or any(t != "full_attention" for t in model.config.layer_types):
            raise ValueError("Evaluator requires ordinary causal Qwen3")
        self.model, self.tokenizer, self.amp, self.sync = model.eval(), tokenizer, amp, sync
        self.head_amp = amp if head_amp is None else head_amp
        self.banned_ids = banned_ids

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _next(self, ids, positions, cache, selected_position=None):
        cache.set_positions(positions)
        with self.amp():
            hidden = self.model.model(input_ids=ids, position_ids=positions[None],
                attention_mask={"full_attention": causal_mask(positions, cache.holder["width"])},
                past_key_values=cache, use_cache=True).last_hidden_state
            if selected_position is None:
                hidden = hidden[:, -1]
            else:
                index = selected_position.reshape(1, 1, 1).expand(1, 1, hidden.shape[-1])
                hidden = hidden.gather(1, index)[:, 0]
        with self.head_amp():
            return self.model.lm_head(hidden).float()

    @torch.no_grad()
    def decode(self, prompt_ids: list[int], *, max_new_tokens: int, prefill_width: int,
               pad_id: int, stop_ids: tuple[int, ...], answer_line_stop: bool) -> Decoded:
        if not prompt_ids or len(prompt_ids) > prefill_width or max_new_tokens < 1:
            raise ValueError("Invalid prompt/prefill/decode bounds")
        actions, content = [], []
        raw_text = ""
        try:
            # Construct on host before transfer: XLA slice-assignment shapes
            # otherwise depend on each item's unpadded prompt length.
            prompt = torch.full((1, prefill_width), pad_id, dtype=torch.long)
            prompt[0, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
            prompt = prompt.to(self.device)
            cache = CausalCache(self.model.config.num_hidden_layers, prefill_width + max_new_tokens)
            position = torch.tensor([len(prompt_ids)], device=self.device, dtype=torch.long)
            logits = self._next(prompt, torch.arange(prefill_width, device=self.device), cache, position - 1)
            for step in range(max_new_tokens):
                token = policy_log_probs(logits, 1.0, self.banned_ids).argmax(-1)
                self.sync()
                value = int(token.cpu()[0])
                actions.append(value)
                if value in stop_ids:
                    return Decoded(actions, content, raw_text, raw_text, f"token:{value}")
                content.append(value)
                raw_text = self.tokenizer.decode(content, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)
                end = first_answer_line_end(raw_text) if answer_line_stop else None
                if end is not None:
                    return Decoded(actions, content, raw_text, raw_text[:end], "answer_line")
                if step + 1 < max_new_tokens:
                    logits = self._next(token[:, None], position, cache)
                    position = position + 1
            return Decoded(actions, content, raw_text, raw_text, "length_cap")
        except Exception as error:
            raise GenerationFailure(error, Decoded(actions, content, raw_text, raw_text, "error")) from error


def generate_item(question: dict, *, decoder, prompt_spec: dict, fmt: Format,
                  mode: str, max_answer_tokens: int, thought_cap: int,
                  question_prefill_width: int, answer_prefill_width: int) -> dict:
    """One path, with all generation failures represented in the denominator."""
    row = dict(item_id=question["item_id"], document_id=question["document_id"],
               task="gsm8k", split=question["split"], mode=mode,
               selection_policy="single_greedy", status="error", generated_text="",
               raw_generated_text="", answer_actions=[], thought_actions=[],
               n_generated_tokens=0, n_answer_tokens=0, n_thought_tokens=0,
               answer_cap=max_answer_tokens, thought_cap=thought_cap if mode == "one_thought" else 0)
    started = time.monotonic()
    stage = "prompt"
    try:
        prompt_text = question_prompt(question["question"], prompt_spec)
        prefix = decoder.tokenizer.encode(prompt_text, add_special_tokens=False)
        row.update(question_prompt=prompt_text, question_prompt_ids=prefix)
        answer_prefix = prefix
        if mode not in {"direct", "one_thought", "empty_format"}:
            raise ValueError("Unknown evaluation mode")
        if mode != "direct":
            stage = "thought"
            writer = list(fmt.header_ids) + prefix + list(fmt.open_ids)
            row["writer_prompt_ids"] = writer
            if mode == "one_thought":
                thought = decoder.decode(writer, max_new_tokens=thought_cap,
                    prefill_width=question_prefill_width + len(fmt.header_ids) + len(fmt.open_ids),
                    pad_id=fmt.pad_id, stop_ids=(fmt.stop_id, *fmt.additional_stop_ids), answer_line_stop=False)
            else:
                thought = Decoded([], [], "", "", "empty_control")
            row.update(thought_actions=thought.actions, thought_content_ids=thought.content_ids,
                       thought_text=thought.text, thought_stop_type=thought.stop_type,
                       n_thought_tokens=len(thought.actions))
            suffix = reader_suffix(question["question"])
            suffix_ids = decoder.tokenizer.encode(suffix, add_special_tokens=False)
            row.update(reader_suffix=suffix, reader_suffix_ids=suffix_ids)
            answer_prefix = writer + thought.content_ids + list(fmt.close_ids) + suffix_ids
        row["answer_prompt_ids"] = answer_prefix
        stage = "answer"
        answer = decoder.decode(answer_prefix, max_new_tokens=max_answer_tokens,
            prefill_width=answer_prefill_width, pad_id=fmt.pad_id,
            stop_ids=tuple(dict.fromkeys((fmt.pad_id, *fmt.additional_stop_ids))), answer_line_stop=True)
        status = "truncated" if answer.stop_type == "length_cap" else "ok"
        if status == "ok" and numeric_answer(answer.text) is None:
            status = "format_error"
        row.update(status=status, generated_text=answer.text, raw_generated_text=answer.raw_text,
                   answer_actions=answer.actions, answer_content_ids=answer.content_ids,
                   answer_stop_type=answer.stop_type, n_answer_tokens=len(answer.actions))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        row["failed_stage"] = stage
        if isinstance(error, GenerationFailure):
            partial = error.partial
            row[f"{stage}_actions"] = partial.actions
            row[f"{stage}_content_ids"] = partial.content_ids
            row[f"n_{stage}_tokens"] = len(partial.actions)
            if stage == "answer":
                row.update(generated_text=partial.text, raw_generated_text=partial.raw_text)
            else:
                row["thought_text"] = partial.text
    row["n_generated_tokens"] = row["n_answer_tokens"] + row["n_thought_tokens"]
    row["wall_seconds"] = time.monotonic() - started
    return row


def asset_provenance(directory: Path) -> dict:
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "merges.txt", "vocab.json", "generation_config.json"}
    paths = sorted(p for p in directory.iterdir() if p.is_file() and
                   (p.name in names or p.suffix == ".safetensors" or
                    p.name.endswith(".safetensors.index.json") or p.name.startswith("pytorch_model")))
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in paths}
    if "config.json" not in files or not any(n.endswith((".safetensors", ".bin")) for n in files):
        raise ValueError("Expected local model config and released weight files")
    manifest = directory / "ASSET_MANIFEST.json"
    result = {"directory": str(directory.resolve()), "files": files, "sha256": canonical_hash(files)}
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        for name, value in files.items():
            if recorded.get("files", {}).get(name, {}).get("sha256") != value["sha256"]:
                raise ValueError(f"Model asset differs from pinned manifest: {name}")
        result.update(manifest_sha256=sha256_file(manifest),
                      model_id=recorded.get("model_id"), revision=recorded.get("resolved_revision"))
    return result


def load_own_checkpoint(model, path: Path) -> dict:
    """Only the documented full-model dictionary, never partial or guessed keys."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
        raise ValueError("Expected a trusted own checkpoint dictionary with full 'model' state")
    weights = saved["model"]
    if model.config.tie_word_embeddings:
        left, right = weights.get("model.embed_tokens.weight"), weights.get("lm_head.weight")
        if left is not None and right is not None and not torch.equal(left, right):
            raise ValueError("Checkpoint contains conflicting tied embedding/head weights")
    model.load_state_dict(weights, strict=True)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            **{k: saved[k] for k in ("run_id", "step", "spec_hash") if k in saved}}


def require_frozen_protocol(path: Path | None, protocol: dict) -> None:
    if path is None:
        raise ValueError("Official test generation requires --frozen-protocol")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "frozen" or not receipt.get("frozen_utc") or protocol not in receipt.get("protocols", []):
        raise ValueError("Frozen receipt does not match the exact proposed evaluation protocol")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--task-revision", required=True)
    parser.add_argument("--prompt-spec", type=Path, required=True)
    parser.add_argument("--format", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "one_thought", "empty_format"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--thought-cap", type=int, default=1024)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--device", choices=("xla", "cpu"), default="xla")
    parser.add_argument("--compute", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--head-precision", choices=("fp32", "bf16"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    args.head_precision = args.head_precision or args.compute
    if min(args.max_answer_tokens, args.thought_cap) < 1:
        parser.error("Token budgets must be positive")
    if args.device == "cpu" and not args.qualification:
        parser.error("CPU mode is for qualification only")
    if args.compute == "fp32" and args.head_precision == "bf16":
        parser.error("A BF16 head requires BF16 compute context")
    if len(args.task_revision) != 40 or any(c not in "0123456789abcdef" for c in args.task_revision):
        parser.error("Task revision must be an explicit 40-character pinned commit")
    questions = read_questions(args.task_file, qualification=args.qualification)
    local_questions = shard_questions(questions, args.shard_index, args.num_shards)
    if not local_questions:
        parser.error("Requested shard contains no task items")
    spec = json.loads(args.prompt_spec.read_text())
    validate_prompt_spec(spec, {q["item_id"] for q in questions})
    fmt = Format.from_dict(json.loads(args.format.read_text())["format"])
    assets = asset_provenance(args.model_dir)
    tokenizer_files = {name: value for name, value in assets["files"].items()
                       if name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                                   "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json"}}
    protocol = dict(schema=SCHEMA, task="gsm8k", task_revision=args.task_revision,
        task_file_sha256=sha256_file(args.task_file), item_ids_sha256=canonical_hash([q["item_id"] for q in questions]),
        prompt_spec=spec, mode=args.mode, selection_policy="single_greedy",
        thought_format=fmt.__dict__ if args.mode != "direct" else None,
        thought_reader_suffix_template=READER_SUFFIX_TEMPLATE if args.mode != "direct" else None,
        thought_cap=args.thought_cap if args.mode == "one_thought" else 0,
        max_answer_tokens=args.max_answer_tokens, answer_stop=ANSWER_STOP,
        answer_stop_ids=list(dict.fromkeys((fmt.pad_id, *fmt.additional_stop_ids))),
        tokenizer_and_config_sha256=canonical_hash(tokenizer_files),
        support="tokenizer_defined_ids_only", compute=args.compute, head_precision=args.head_precision,
        evaluator_sha256=sha256_file(Path(__file__)), backend_sha256=sha256_file(Path(__file__).with_name("neuron_backend.py")))
    # JSON-normalize tuples so a written/read receipt compares identically.
    protocol = json.loads(json.dumps(protocol))
    if not args.qualification:
        require_frozen_protocol(args.frozen_protocol, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(run_id=args.output.name, status="running", started_utc=utc_now(), qualification=args.qualification,
                  protocol=protocol, protocol_sha256=canonical_hash(protocol), expected_items=len(local_questions),
                  global_expected_items=len(questions),
                  shard=dict(index=args.shard_index, count=args.num_shards,
                             expected_items=len(local_questions),
                             item_ids_sha256=canonical_hash([q["item_id"] for q in local_questions])),
                  model=assets, parser="strict_final_hash4_number_v1",
                  note="Question-only generation. No gold scoring or answer-based selection in this process.")
    atomic_json(args.output / "run.json", report)
    try:
        model, tokenizer = load_base(args.model_dir)
        report["checkpoint"] = load_own_checkpoint(model, args.checkpoint) if args.checkpoint else None
        cpu_signature = sampled_fingerprint(model)
        if tokenizer.eos_token_id is None or fmt.pad_id != tokenizer.eos_token_id:
            raise ValueError("Resolved format pad must be the qualified tokenizer EOS")
        defined_ids = set(tokenizer.get_vocab().values())
        if not defined_ids or min(defined_ids) < 0 or max(defined_ids) >= model.config.vocab_size:
            raise ValueError("Tokenizer IDs exceed released model support")
        if not set((*fmt.header_ids, *fmt.open_ids, *fmt.close_ids, fmt.pad_id,
                    fmt.stop_id, *fmt.additional_stop_ids)) <= defined_ids:
            raise ValueError("Thought format IDs must have tokenizer definitions")
        banned = tuple(sorted(set(range(model.config.vocab_size)) - defined_ids))
        report.update(cpu_fingerprint=cpu_signature, banned_ids=list(banned),
                      banned_ids_sha256=canonical_hash(banned))
        sync, amp = (lambda: None), nullcontext
        if args.device == "xla":
            from train.runtime_compat import configure_runtime
            report["runtime_compatibility"] = configure_runtime()
            import torch_xla
            import torch_xla.amp
            device = torch_xla.device()
            model.to(device)
            if model.config.tie_word_embeddings:
                model.tie_weights()
            sync = lambda: torch_xla.sync(wait=True)
            sync()
            if args.compute == "bf16":
                amp = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)
            report.update(device=str(device), visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
                          device_fingerprint=sampled_fingerprint(model))
            if cpu_signature != report["device_fingerprint"]:
                raise RuntimeError("Model fingerprint/tie changed on accelerator transfer")
        else:
            report["device"] = "cpu"
            if args.compute == "bf16":
                amp = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
        decoder = GreedyDecoder(model, tokenizer, amp=amp, sync=sync, banned_ids=banned,
                                head_amp=nullcontext if args.head_precision == "fp32" else amp)
        widths = [len(tokenizer.encode(question_prompt(q["question"], spec), add_special_tokens=False)) for q in questions]
        question_width = max(widths)
        answer_width = question_width
        if args.mode != "direct":
            answer_width += len(fmt.header_ids) + len(fmt.open_ids) + len(fmt.close_ids)
            if args.mode == "one_thought":
                answer_width += args.thought_cap
            answer_width += max(len(tokenizer.encode(reader_suffix(q["question"]), add_special_tokens=False)) for q in questions)
        if answer_width + args.max_answer_tokens > model.config.max_position_embeddings:
            raise ValueError("Evaluation budget exceeds released model context")
        report.update(question_prefill_width=question_width, answer_prefill_width=answer_width)
        atomic_json(args.output / "run.json", report, overwrite=True)
        counts = {key: 0 for key in ("ok", "format_error", "truncated", "error")}
        started = time.monotonic()
        with (args.output / "predictions.jsonl").open("x") as stream:
            for question in local_questions:
                row = generate_item(question, decoder=decoder, prompt_spec=spec, fmt=fmt,
                    mode=args.mode, max_answer_tokens=args.max_answer_tokens, thought_cap=args.thought_cap,
                    question_prefill_width=question_width, answer_prefill_width=answer_width)
                row.update(run_id=report["run_id"], protocol_sha256=report["protocol_sha256"],
                           model_sha256=report["model"]["sha256"],
                           checkpoint_sha256=(report["checkpoint"] or {}).get("sha256"))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                counts[row["status"]] += 1
                print(json.dumps({k: row[k] for k in ("item_id", "mode", "status", "n_generated_tokens", "wall_seconds")}), flush=True)
        report.update(status="complete", finished_utc=utc_now(), statuses=counts,
                      generation_wall_seconds=time.monotonic() - started,
                      predictions_sha256=sha256_file(args.output / "predictions.jsonl"))
        if args.device == "xla":
            from torch_xla.debug import metrics
            (args.output / "xla_metrics.txt").write_text(metrics.metrics_report())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc_now())
        raise
    finally:
        atomic_json(args.output / "run.json", report, overwrite=True)


if __name__ == "__main__":
    main()
