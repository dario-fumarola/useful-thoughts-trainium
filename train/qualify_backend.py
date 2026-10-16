"""Bounded cached-rollout/replay qualification with real released weights."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch

from weekend_thinking.artifacts import atomic_json, utc_now
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, action_mask
from weekend_thinking.pretrained import load_base
from train.neuron_backend import NeuronCausalBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--format', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cap', type=int, default=8)
    p.add_argument('--prefix-tokens', type=int, default=512)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--backward', action='store_true')
    a = p.parse_args()
    if not 1 <= a.cap <= 1024 or not 1 <= a.repeat <= 10:
        p.error('Bounded cap 1..1024 and repeat 1..10 required')
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(stage='P2', status='running', started_utc=utc_now(),
                  cap=a.cap, precision=a.precision, group_size=4,
                  replay_logp_atol=0.001 if a.precision == 'fp32' else 0.2)
    def save():
        atomic_json(a.out/'backend.json', report, overwrite=True)
    save()
    try:
        model, tok = load_base(a.model_dir)
        fmt = Format.from_dict(json.loads(a.format.read_text())['format'])
        text = ('For a rectangle, the area equals width times length. '
                'A rectangle has width 7 and length 9, so its area is 63 square units. ')
        ids = tok.encode(text * (a.prefix_tokens//8 + 2), add_special_tokens=False)[:a.prefix_tokens]
        prefix = torch.tensor([ids])
        future = torch.tensor([tok.encode(' The perimeter is 32.', add_special_tokens=False)[:2]])
        valid = set(tok.get_vocab().values())
        banned = tuple(i for i in range(model.config.vocab_size) if i not in valid)
        report['banned_model_only_rows'] = list(banned)
        report['prefix_ids'] = ids
        import torch_xla
        import torch_xla.amp
        import torch_xla.core.xla_model as xm
        from torch_xla.debug import metrics
        from train.runtime_compat import configure_runtime
        report['runtime_compatibility'] = configure_runtime()
        if not hasattr(torch, 'xla'):
            torch._register_device_module('xla', torch_xla)
        device = torch_xla.device()
        model.to(device)
        if model.config.tie_word_embeddings:
            model.tie_weights()
        model.eval()
        amp = (nullcontext if a.precision == 'fp32'
               else lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16))
        backend = NeuronCausalBackend(model, amp=amp, sync=lambda: torch_xla.sync(),
                                     banned_ids=banned)
        prefix, future = prefix.to(device), future.to(device)
        torch_xla.sync(wait=True)
        runs = []
        for run in range(a.repeat):
            xm.set_rng_state(1234 + run, device)
            print(f'Run {run}: sampling cap={a.cap}', flush=True)
            started = time.monotonic()
            with torch.no_grad():
                rollout = backend.sample(prefix, fmt=fmt, group_size=4, max_actions=a.cap,
                                         temperature=0.7, policy_version=0)
                torch_xla.sync(wait=True)
                sampled_seconds = time.monotonic() - started
                print(f'Run {run}: replay', flush=True)
                replay = backend.replay(prefix, rollout, fmt=fmt, temperature=0.7)
                error = (replay - rollout.behavior_logps).abs()
                maximum = torch.where(rollout.action_mask, error, 0.0).max().cpu().item()
                print(f'Run {run}: reward scoring', flush=True)
                thought = backend.score(prefix, future, rollout, fmt=fmt, horizon=1).cpu()
                clean = backend.clean_score(prefix, future).cpu()
                actions = rollout.actions.cpu()
                mask = rollout.action_mask.cpu()
                torch.save({'actions': actions, 'mask': mask,
                            'behavior_logps': rollout.behavior_logps.cpu(),
                            'replay_logps': replay.cpu(), 'thought_log_evidence': thought,
                            'clean_log_evidence': clean}, a.out/f'trace-{run}.pt')
            row = dict(run=run, sample_seconds=sampled_seconds,
                       complete_trace_seconds=time.monotonic()-started,
                       max_replay_logp_error=maximum, sampled_actions=mask.sum(-1).tolist(),
                       thought_log_evidence=thought.tolist(), clean_log_evidence=clean.tolist(),
                       samples=[tok.decode(actions[0,g][mask[0,g]].tolist(), skip_special_tokens=False)
                                for g in range(4)])
            runs.append(row)
            report['runs'] = runs
            save()
            (a.out/f'xla_metrics-{run}.txt').write_text(metrics.metrics_report())
            if maximum > report['replay_logp_atol']:
                raise RuntimeError(f'Cache/replay mismatch {maximum} exceeds fixed tolerance')
        if a.backward:
            print('Bounded single-thought backward; no optimizer update', flush=True)
            model.zero_grad(set_to_none=True)
            loss = backend.replay_one_loss(prefix, rollout.actions[:,0],
                rollout.action_mask[:,0], torch.tensor(.25, device=device),
                fmt=fmt, temperature=.7, divisor=4)
            loss.backward()
            torch_xla.sync(wait=True)
            norm = torch.stack([p.grad.float().square().sum() for p in model.parameters()
                                if p.grad is not None]).sum().sqrt().cpu().item()
            report['backward_gradient_norm'] = norm
            if not 0 < norm < 1e8:
                raise RuntimeError('Backward gradient is zero or nonfinite')
        report['status'] = 'passed_trace'
        report['limitations'] = ['No distributed update/resume qualification.',
            'Short traces do not qualify the 1024-action cap.',
            'Fixed non-evaluation development passage; no downstream claim.']
        save()
        print(json.dumps({'status': report['status'], 'runs': runs}, indent=2), flush=True)
    except BaseException as e:
        report.update(status='failed', error=f'{type(e).__name__}: {e}')
        save()
        raise


if __name__ == '__main__':
    main()
