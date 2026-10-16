"""Bounded fixed-action cache/replay diagnosis; never samples or trains.

Teacher-force a previously failing rollout through both cached and full-sequence
paths. This prevents a numerical repair from passing merely by drawing different
actions. The writer input contains only the previously recorded prefix.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch

from train.neuron_backend import CausalCache, NeuronCausalBackend
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, action_mask, policy_log_probs
from weekend_thinking.pretrained import load_base


@torch.no_grad()
def cached_forced_logps(backend, prefix, rollout, *, fmt, temperature=.7):
    b, group, cap = rollout.actions.shape
    prompt = backend._prompt(prefix, fmt).repeat_interleave(group, 0)
    prompt_width = prompt.shape[-1]
    cache = CausalCache(backend.model.config.num_hidden_layers, prompt_width + cap)
    logits = backend._next(prompt, torch.arange(prompt_width, device=backend.device), cache)
    actions = rollout.actions.reshape(b * group, cap)
    position = torch.tensor([prompt_width], device=backend.device, dtype=torch.long)
    index = position - prompt_width
    selected = []
    for step in range(cap):
        # Data-valued gather indices avoid compiling one slice graph per step.
        action = actions.index_select(-1, index)[:, 0]
        lp = policy_log_probs(logits, temperature, backend.banned_ids)
        selected.append(lp.gather(-1, action[:, None])[:, 0])
        backend.sync()
        if step % 128 == 0 or step + 1 == cap:
            print(f"{utc_now()} forced action {step + 1}/{cap}", flush=True)
        if step + 1 < cap:
            logits = backend._next(action[:, None], position, cache)
            position = position + 1
            index = index + 1
    return torch.stack(selected, -1).reshape(b, group, cap)


def error_summary(cached, replay, mask, actions):
    difference = replay - cached
    selected = difference[mask]
    absolute = selected.abs()
    maximum = torch.where(mask, difference.abs(), -1.0).flatten().argmax()
    index = tuple(int(v) for v in torch.unravel_index(maximum, difference.shape))
    return dict(active_actions=int(mask.sum()), max_abs=float(absolute.max()),
        mean_abs=float(absolute.mean()), mean_signed=float(selected.mean()),
        rmse=float(selected.square().mean().sqrt()),
        absolute_quantiles={str(q): float(torch.quantile(absolute, q)) for q in (.5, .9, .95, .99, .999)},
        above_0_1=int((absolute > .1).sum()), above_0_2=int((absolute > .2).sum()),
        per_completion_sum_difference=(difference * mask).sum(-1).tolist(),
        maximum_at=dict(batch=index[0], group=index[1], action=index[2],
                        token=int(actions[index]), cached=float(cached[index]), replay=float(replay[index])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--format", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--source-trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head-precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--trunk-precision", choices=("fp32", "bf16"), default="bf16")
    args = p.parse_args()
    if args.trunk_precision == "fp32" and args.head_precision != "fp32":
        p.error("Full FP32 probe requires FP32 head")
    source = json.loads(args.source_report.read_text())
    trace = torch.load(args.source_trace, map_location="cpu", weights_only=True)
    actions = trace["actions"]
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] != 4 or not 1 <= actions.shape[2] <= 1024:
        raise ValueError("Probe requires one saved group of four, at most 1024 actions")
    prefix = torch.tensor([source["prefix_ids"]], dtype=torch.long)
    if not 1 <= prefix.shape[-1] <= 512:
        raise ValueError("Probe prefix must be recorded and bounded to 512 tokens")
    fmt = Format.from_dict(json.loads(args.format.read_text())["format"])
    mask = trace["mask"].bool()
    if not torch.equal(action_mask(actions, fmt.stop_id, fmt.additional_stop_ids), mask):
        raise ValueError("Source mask differs from its first-stop semantics")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="fixed_action_numerical_probe", status="running", started_utc=utc_now(),
        trunk_precision=args.trunk_precision, parameter_precision="fp32", head_precision=args.head_precision,
        replay_logp_atol=.001 if args.trunk_precision == "fp32" else .2,
        temperature=.7, actions_shape=list(actions.shape),
        lengths=mask.sum(-1).tolist(), source_report_sha256=sha256_file(args.source_report),
        source_trace_sha256=sha256_file(args.source_trace), format_sha256=sha256_file(args.format),
        model_manifest_sha256=sha256_file(args.model_dir / "ASSET_MANIFEST.json"),
        probe_source_sha256=sha256_file(Path(__file__)),
        backend_source_sha256=sha256_file(Path(__file__).with_name("neuron_backend.py")),
        no_sampling=True, no_future_inputs=True, no_optimizer=True)
    def save():
        atomic_json(args.output / "probe.json", report, overwrite=True)
    save()
    try:
        model, tokenizer = load_base(args.model_dir)
        import torch_xla
        import torch_xla.amp
        from train.runtime_compat import configure_runtime
        report["runtime_compatibility"] = configure_runtime()
        device = torch_xla.device()
        model.to(device)
        if model.config.tie_word_embeddings:
            model.tie_weights()
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise RuntimeError("Released weight tie was lost")
        model.eval()
        valid_ids = set(tokenizer.get_vocab().values())
        banned = tuple(i for i in range(model.config.vocab_size) if i not in valid_ids)
        amp = (nullcontext if args.trunk_precision == "fp32"
               else lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16))
        backend = NeuronCausalBackend(model, amp=amp,
            head_amp=nullcontext if args.head_precision == "fp32" else amp,
            sync=lambda: torch_xla.sync(wait=True), banned_ids=banned)
        rollout = Rollouts(actions.to(device), mask.to(device),
                           trace["behavior_logps"].to(device), policy_version=0)
        prefix = prefix.to(device)
        torch_xla.sync(wait=True)
        started = time.monotonic()
        with torch.no_grad():
            cached = cached_forced_logps(backend, prefix, rollout, fmt=fmt).cpu()
            report["cached_seconds"] = time.monotonic() - started
            report["status"] = "cached_complete_replay_running"
            save()
            print(f"{utc_now()} full-sequence replay", flush=True)
            replay = backend.replay(prefix, rollout, fmt=fmt, temperature=.7).cpu()
        summary = error_summary(cached, replay, mask, actions)
        torch.save(dict(actions=actions, mask=mask, cached_logps=cached, replay_logps=replay,
                        differences=replay-cached, old_behavior_logps=trace["behavior_logps"],
                        old_replay_logps=trace["replay_logps"]), args.output / "every_token.pt")
        report.update(summary=summary, complete_seconds=time.monotonic() - started,
                      status="passed" if summary["max_abs"] <= report["replay_logp_atol"] else "failed_bound",
                      finished_utc=utc_now())
        # Preserve the old maximum's exact action independently of the new max.
        old_difference = trace["replay_logps"] - trace["behavior_logps"]
        old_max = torch.where(mask, old_difference.abs(), -1.0).flatten().argmax()
        at = tuple(int(v) for v in torch.unravel_index(old_max, old_difference.shape))
        report["old_maximum_action"] = dict(index=list(at), token=int(actions[at]),
            old_difference=float(old_difference[at]), new_difference=float((replay-cached)[at]))
        from torch_xla.debug import metrics
        (args.output / "xla_metrics.txt").write_text(metrics.metrics_report())
        save()
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "passed":
            raise RuntimeError(f"Fixed-action probe exceeds unchanged {report['replay_logp_atol']} bound")
    except BaseException as error:
        if report["status"] != "failed_bound":
            report["status"] = "failed"
        report.update(error=f"{type(error).__name__}: {error}", finished_utc=utc_now())
        save()
        raise


if __name__ == "__main__":
    main()
