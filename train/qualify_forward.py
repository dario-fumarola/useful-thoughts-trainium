"""Bounded real-weight qualification. No optimizer or scientific training.

Compare CPU and accelerator selected-token distributions without projecting a
whole sequence to the vocabulary. Save all measured values, including failures.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.pretrained import load_base, sampled_fingerprint


def selected_logps(model, ids, targets, amp):
    device = next(model.parameters()).device
    x = ids.to(device)
    n = x.shape[-1]
    pos = torch.arange(n, device=device)
    mask = torch.where(pos[None, :] <= pos[:, None], 0.0, -1e9)[None, None]
    with torch.no_grad(), amp():
        h = model.model(input_ids=x, position_ids=pos[None],
                        attention_mask={"full_attention": mask}, use_cache=False).last_hidden_state
        logits = model.lm_head(h[:, -1:])[:, 0].float()
    return logits.log_softmax(-1).index_select(-1, targets.to(device))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-only', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = {'stage': 'P1', 'started_utc': utc_now(), 'status': 'running',
              'model_manifest_sha256': sha256_file(a.model_dir / 'ASSET_MANIFEST.json'),
              # Declared before any accelerator observation; never auto-relaxed.
              'tolerances': {'fp32_logp_atol': 0.001, 'bf16_logp_atol': 0.15}}
    def save():
        atomic_json(a.out / 'forward.json', report, overwrite=True)
    save()
    try:
        print('Loading exact pretrained checkpoint on CPU', flush=True)
        started = time.monotonic()
        model, tok = load_base(a.model_dir)
        model.eval()
        assert model.config.model_type == 'qwen3'
        assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 2048
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        report['load_seconds'] = time.monotonic() - started
        report['cpu_fingerprint'] = sampled_fingerprint(model)
        text = 'A rectangle has width 7 and length 9. Its area is'
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
        target_ids = sorted(set(tok.encode(' 63 square units. 16 49 0', add_special_tokens=False)))
        targets = torch.tensor(target_ids)
        report.update(input_text=text, input_ids=ids.tolist(), target_ids=target_ids,
                      intended_tie=True, cpu_tie=True, parameter_dtype=str(next(model.parameters()).dtype))
        cpu = selected_logps(model, ids, targets, nullcontext).cpu()
        cpu_bf = selected_logps(model, ids, targets,
                                lambda: torch.autocast('cpu', dtype=torch.bfloat16)).cpu()
        report.update(cpu_fp32_logps=cpu.tolist(), cpu_bf16_logps=cpu_bf.tolist(),
                      cpu_precision_max_difference=float((cpu - cpu_bf).abs().max()))
        save()
        if a.cpu_only:
            report['status'] = 'cpu_passed_device_pending'
            save()
            return
        print('Transferring to XLA; compiling short FP32 forward', flush=True)
        import torch_xla
        import torch_xla.amp
        import torch_xla.core.xla_model as xm
        from train.runtime_compat import configure_runtime
        report['runtime_compatibility'] = configure_runtime()
        device = torch_xla.device()
        model.to(device)
        # XLA transfer creates distinct Parameters for a shared CPU object.
        # Restore only the tie declared by the released checkpoint config.
        if model.config.tie_word_embeddings:
            model.tie_weights()
        torch_xla.sync(wait=True)
        report['device'] = str(device)
        report['device_tie'] = model.get_input_embeddings().weight is model.get_output_embeddings().weight
        report['device_fingerprint'] = sampled_fingerprint(model)
        save()
        t = time.monotonic()
        actual = selected_logps(model, ids, targets, nullcontext).cpu()
        report['device_fp32_seconds_cold'] = time.monotonic() - t
        report['device_fp32_logps'] = actual.tolist()
        report['fp32_max_error'] = float((cpu - actual).abs().max())
        save()
        print('Compiling short BF16-autocast forward', flush=True)
        actual_bf = selected_logps(model, ids, targets,
                                  lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)).cpu()
        report['device_bf16_logps'] = actual_bf.tolist()
        report['bf16_max_error'] = float((cpu_bf - actual_bf).abs().max())
        from torch_xla.debug import metrics
        (a.out / 'xla_metrics.txt').write_text(metrics.metrics_report())
        try:
            report['memory'] = xm.get_memory_info(device)
        except (RuntimeError, AttributeError) as e:
            report['memory_query_error'] = str(e)
        report['passed'] = (report['device_tie'] and report['cpu_fingerprint'] == report['device_fingerprint']
                            and report['fp32_max_error'] <= report['tolerances']['fp32_logp_atol']
                            and report['bf16_max_error'] <= report['tolerances']['bf16_logp_atol'])
        report['status'] = 'passed_short_forward' if report['passed'] else 'failed'
        save()
        if not report['passed']:
            raise RuntimeError('P1 qualification failed; see saved measurements')
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as e:
        report.update(status='failed', error=f'{type(e).__name__}: {e}')
        save()
        raise


if __name__ == '__main__':
    main()
