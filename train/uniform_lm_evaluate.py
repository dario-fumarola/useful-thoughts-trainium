"""Full-support original-text likelihood for the fixed uniform-LM comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from train import partial_state_driver as driver
from train.neuron_backend import NeuronCausalBackend
from train.useful_note_precision import cast_parameters
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.pretrained import load_base, sampled_fingerprint


def token_logps(backend, ids):
    """Score every shifted target, preserving native special tokens."""
    if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[1] < 2:
        raise ValueError('Original complete native windows required')
    x = ids.to(backend.device)
    hidden = backend.hidden(x[:, :-1])
    backend.sync()
    result = []
    for first in range(0, hidden.shape[1], backend.head_chunk):
        h = hidden[:, first:first+backend.head_chunk]
        y = x[:, first+1:first+1+backend.head_chunk]
        logits = backend.model.lm_head(h).float()
        result.append(-F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1), reduction='none').reshape(y.shape))
        del logits
        # Reuse two small head graphs (full chunk and remainder), rather than
        # compiling every vocabulary projection into one large fused graph.
        backend.sync()
    return torch.cat(result, 1)


def validate(p, release, output):
    if (p['schema'] != 'uniform-lm-likelihood-v1' or p['world_size'] != 8
            or p['micro_batch'] != 4 or p['visible_cores'] != '32-39'
            or p['source_files'] != driver.source_files()
            or sha256_file(Path(__file__)) != p['evaluator_sha256']):
        raise ValueError('Fixed evaluator, shape and source required')
    if sha256_file(p['data_file']) != p['data_sha256']:
        raise ValueError('Evaluation sources differ')
    cohort = p['cohort']
    if cohort not in ('QUALIFICATION', 'DEV', 'CONFIRM', 'CALCULATION'):
        raise ValueError('Named evaluation cohort required')
    if cohort != 'CALCULATION':
        manifest_file = Path(p['data_manifest_file'])
        if sha256_file(manifest_file) != p['data_manifest_sha256']:
            raise ValueError('Pinned uniform-LM source manifest differs')
        source = json.loads(manifest_file.read_text())['files'][cohort]
        if source['sha256'] != p['data_sha256'] or source['rows'] != p['items']:
            raise ValueError('Named source cohort does not match the protocol')
    if cohort != 'QUALIFICATION':
        terminal = release['all_terminal_models']
        if len(terminal) != 4 or any(v['step'] != 128 for v in terminal.values()):
            raise ValueError('Freeze all four terminal models before development scoring')
    if cohort == 'CONFIRM':
        selection = release['selection']
        if sha256_file(selection['file']) != selection['sha256']:
            raise ValueError('Freeze development choice before confirmation')
        selected = json.loads(Path(selection['file']).read_text())
        if p['model_name'] not in ('B0', selected['recipe']):
            raise ValueError('Only B0 and the frozen choice may score confirmation')
    if (p['checkpoint'] is None) != (p['model_name'] == 'B0'):
        raise ValueError('B0 and trained model identities must match their actual paths')
    if p['model_name'] != 'B0' and p['checkpoint_sha256'] != release[
            'all_terminal_models'][p['model_name']]['checkpoint_sha256']:
        raise ValueError('Named trained model differs from the frozen terminal inventory')
    if p['checkpoint'] is not None and sha256_file(p['checkpoint']) != p['checkpoint_sha256']:
        raise ValueError('Exact frozen model checkpoint required')
    if str(output) not in release['jobs']:
        raise ValueError('Explicit finite evaluation job required')


def worker(_index, args, p, rows):
    import torch_xla
    import torch_xla.runtime as xr
    from train.runtime_compat import configure_runtime
    from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy
    configure_runtime()
    if not hasattr(torch, 'xla'):
        torch._register_device_module('xla', torch_xla)
    require_runtime_policy(p['runtime_policy'], runtime_policy())
    rank, world = xr.global_ordinal(), xr.world_size()
    if world != 8:
        raise RuntimeError('Eight assigned native workers required')
    device = torch_xla.device()
    model, _ = load_base(p['model_dir'])
    if sampled_fingerprint(model) != p['base_signature']:
        raise ValueError('Fresh original base load differs')
    if p['checkpoint'] is not None:
        state = torch.load(p['checkpoint'], weights_only=True, map_location='cpu', mmap=True)
        if state['step'] != 128:
            raise ValueError('Terminal model required')
        model.load_state_dict(state['model'], strict=True)
        del state
    cast_parameters(model)
    expected_signature = sampled_fingerprint(model)
    model.to(device)
    model.tie_weights()
    if sampled_fingerprint(model) != expected_signature:
        raise RuntimeError('Actual device weights differ from the loaded model')
    backend = NeuronCausalBackend(model, head_chunk=32, checkpoint_trunk=False,
                                 sync=lambda: torch_xla.sync(wait=True))
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    batch_seconds = []
    started = time.monotonic()
    deadline = datetime.fromisoformat(args.deadline_utc)
    with (args.output/f'rank-{rank:05d}.jsonl').open('x') as out, torch.no_grad():
        for first in range(rank*4, len(rows), world*4):
            batch_started = time.monotonic()
            if args.stop_file.exists() or datetime.now(timezone.utc) >= deadline:
                raise RuntimeError('Finite evaluation resource boundary reached')
            batch = rows[first:first+4]
            if len(batch) != 4:
                raise ValueError('Complete fixed native microbatch required')
            lp = token_logps(backend, torch.tensor([r['ids'] for r in batch]))
            torch_xla.sync(wait=True)
            lp = lp.cpu()
            if lp.shape != (4, 527) or not torch.isfinite(lp).all():
                raise RuntimeError('Finite complete 527-token likelihood required')
            for row, values in zip(batch, lp.tolist()):
                record = dict(item_id=row['item_id'], document_id=row['document_id'],
                              model=p['model_name'], token_logps=values)
                out.write(json.dumps(record, allow_nan=False)+'\n')
                completed += 1
            out.flush()
            batch_seconds.append(time.monotonic()-batch_started)
    atomic_json(args.output/f'complete-rank-{rank:05d}.json',
                dict(rank=rank, rows=completed, wall_seconds=time.monotonic()-started,
                     batch_seconds=batch_seconds, actual_model_signature=expected_signature,
                     protocol_sha256=sha256_file(args.protocol),
                     sha256=sha256_file(args.output/f'rank-{rank:05d}.jsonl'), utc=utc_now()))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'execution-release', 'output', 'stop-file'):
        a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--deadline-utc', required=True)
    args = a.parse_args()
    p = json.loads(args.protocol.read_text())
    release = json.loads(args.execution_release.read_text())
    if release['jobs'][str(args.output)]['protocol_sha256'] != sha256_file(args.protocol):
        raise ValueError('Frozen evaluation protocol differs')
    validate(p, release, args.output)
    rows = [json.loads(s) for s in Path(p['data_file']).open()]
    if (len(rows) != p['items'] or len(rows) % 32
            or len({r['document_id'] for r in rows}) != len(rows)
            or any(len(r['ids']) != 528 for r in rows)):
        raise ValueError('Complete unique native source cohort required')
    if args.output.exists():
        raise ValueError('Fresh evaluation output required')
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(args, p, rows), nprocs=None)


if __name__ == '__main__':
    main()
