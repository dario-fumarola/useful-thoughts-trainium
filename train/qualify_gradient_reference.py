"""Independent CPU gradient reference for one saved long policy trajectory.

Uses the native released Qwen trunk and a single vocabulary projection, without
the production cache, checkpoint wrappers, or chunked gradient implementation.
This is a bounded numerical diagnostic, not scientific training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import transformers

from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.core import Format, policy_log_probs
from weekend_thinking.pretrained import load_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--format', type=Path, required=True)
    parser.add_argument('--prefix-report', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--advantage', type=float, default=.25)
    parser.add_argument('--divisor', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.divisor < 1 or not math.isfinite(args.advantage):
        parser.error('Finite advantage and positive divisor required.')
    args.out.mkdir(parents=True, exist_ok=False)
    criteria = dict(scope='one_saved_long_trajectory_gradient_reference',
        created_utc=utc_now(), accelerator_execution=False,
        compare_norm_atol=1e-5, compare_norm_rtol=5e-4,
        criterion='abs(device_norm - CPU_float64_reduction_norm) <= atol + rtol * CPU_norm',
        qualification='Reference and criterion only; hardware result must be compared separately.')
    atomic_json(args.out / 'criteria.json', criteria)
    source = Path(__file__)
    report = dict(**criteria, status='running', source_sha256=sha256_file(source),
        core_sha256=sha256_file(source.parents[1] / 'weekend_thinking/core.py'),
        model_manifest_sha256=sha256_file(args.model_dir / 'ASSET_MANIFEST.json'),
        format_sha256=sha256_file(args.format), prefix_report_sha256=sha256_file(args.prefix_report),
        trace_sha256=sha256_file(args.trace), group=args.group,
        advantage=args.advantage, divisor=args.divisor, temperature=.7,
        torch_version=torch.__version__, transformers_version=transformers.__version__)
    atomic_json(args.out / 'reference.json', report)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    actions, mask = trace['actions'], trace['mask']
    if (actions.ndim != 3 or actions.shape[0] != 1 or not 0 <= args.group < actions.shape[1]
            or not 1 <= actions.shape[2] <= 1024 or mask.shape != actions.shape):
        raise ValueError('Expected a bounded single-prefix saved rollout.')
    actions, mask = actions[:, args.group], mask[:, args.group]
    prefix = json.loads(args.prefix_report.read_text())['prefix_ids']
    fmt = Format.from_dict(json.loads(args.format.read_text())['format'])
    prompt = torch.tensor([list(fmt.header_ids) + prefix + list(fmt.open_ids)])
    x = torch.cat((prompt, actions[:, :-1]), dim=1)
    model, tokenizer = load_base(args.model_dir)
    model.eval()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError('CPU reference lost the released embedding tie.')
    valid = set(tokenizer.get_vocab().values())
    banned = tuple(i for i in range(model.config.vocab_size) if i not in valid)
    if any(int(token) not in valid for token in actions[mask]):
        raise ValueError('Saved active action lies outside the writer support.')
    started = time.monotonic()
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state[:, prompt.shape[1]-1:]
    logits = model.lm_head(hidden)
    lp = policy_log_probs(logits, .7, banned).gather(-1, actions[:, :, None])[:, :, 0]
    loss = -args.advantage * (lp * mask).sum() / mask.sum().clamp_min(1) / args.divisor
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite reference loss.')
    loss.backward()
    norms, squared_total = {}, 0.0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise ValueError(f'Missing/nonfinite reference gradient: {name}')
        flat = gradient.detach().reshape(-1)
        squared = 0.0
        for start in range(0, flat.numel(), 8*1024*1024):
            squared += float(flat[start:start+8*1024*1024].double().square().sum())
        norms[name] = math.sqrt(squared)
        squared_total += squared
    report.update(status='completed', loss=float(loss.detach()),
        full_gradient_norm_float64=math.sqrt(squared_total), parameter_gradient_norms_float64=norms,
        unique_parameters=sum(p.numel() for p in model.parameters()),
        active_actions=int(mask.sum()), input_tokens=x.shape[1],
        elapsed_seconds=time.monotonic()-started, finished_utc=utc_now())
    atomic_json(args.out / 'reference.json', report, overwrite=True)
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'parameter_gradient_norms_float64'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
