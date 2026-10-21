"""Reconstruct the uniform-LM native qualification from saved full states."""
import argparse
import json
import math
from pathlib import Path
import shlex

import torch

from train import partial_state_driver as driver, uniform_lm
from train.neuron_driver import verify_checkpoint
from train.useful_note_recovery import optimizer_digest, tensor_digest
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--release', type=Path, required=True)
    args = a.parse_args()
    release = json.loads(args.release.read_text())
    control = Path(release['control'])
    jobs = release['named_jobs']
    fresh, replay = Path(jobs['qualification']['output']), Path(jobs['qualification-replay']['output'])
    protocol = Path(jobs['qualification']['protocol'])
    p = json.loads(protocol.read_text())
    rows = [json.loads(s) for s in Path(p['data_file']).open()]
    order = driver.order_for(p, rows)
    closures = [json.loads((control/'jobs'/name/'closure.json').read_text())
                for name in ('qualification', 'qualification-replay')]
    require(all(v['exit_code'] == 0 and v['failure'] is None
                and v['actual_eight_observed'] and v['owned_session_absent']
                and v['assigned_devices_idle'] for v in closures), 'Native process closure failed')
    cost = math.fsum(v['reserved_chip_hours'] for v in closures)
    require(cost <= release['maximum_reserved_chip_hours'], 'Qualification resource ceiling exceeded')
    metrics = [[json.loads(s) for s in (out/'metrics.jsonl').open()] for out in (fresh, replay)]
    for history in metrics:
        require([m['step'] for m in history] == [1, 2, 3, 4], 'Four completed updates required')
        for step, m in enumerate(history):
            rate = uniform_lm.learning_rate(p, step)
            require(m['learning_rate'] == rate
                    and all(v == rate for v in m['actual_base_optimizer_lrs'])
                    and m['gradient_reductions'] == 1, 'Actual LR/reducer mismatch')
            require(m['ordered_item_ids'] == [rows[k]['item_id'] for k in
                    order[step*256:(step+1)*256]], 'Data order changed')
            require(m['counters'] == dict(expressions=(step+1)*256,
                    supervised_tokens=(step+1)*134912), 'Accumulated target counts changed')
            require(math.isfinite(m['loss']) and math.isfinite(m['gradient_norm'])
                    and m['gradient_norm'] > 0 and m['actual_master_and_replica_audit'],
                    'Native update/master/replica audit failed')
    ref = json.loads(Path(release['initial_loss_reference']).read_text())
    error = abs(metrics[0][0]['loss']-ref['global_loss'])
    require(error <= ref['maximum_mean_abs_error'], 'Initial uniform loss differs from CPU B0')
    for left, right in zip(metrics[0][2:], metrics[1][2:]):
        require(all(left[k] == right[k] for k in
                    ('loss', 'gradient_norm', 'learning_rate', 'actual_base_optimizer_lrs',
                     'gradient_reductions', 'counters', 'ordered_item_ids')),
                'Updates 3/4 differ after full-state restoration')
    manifests = {}
    for out, steps in ((fresh, (2, 4)), (replay, (2, 4))):
        for step in steps:
            path = out/f'step-{step:07d}'
            verify_checkpoint(path, canonical_hash(p), 8)
            manifests[str(path)] = sha256_file(path/'COMPLETE.json')
    states = [torch.load(out/'step-0000004/model.pt', weights_only=True,
                         map_location='cpu', mmap=True)['model'] for out in (fresh, replay)]
    require(states[0].keys() == states[1].keys(), 'Model tensor inventories differ')
    model_digests = {}
    for name, value in states[0].items():
        require(torch.equal(value.view(torch.uint8), states[1][name].view(torch.uint8)),
                'Recovered terminal model tensor differs: '+name)
        model_digests[name] = tensor_digest(value)
    del states
    shards = []
    for rank in range(8):
        name = f'rank-{rank:05d}.pt'
        pair = [torch.load(out/'step-0000004'/name, weights_only=True,
                           map_location='cpu', mmap=True) for out in (fresh, replay)]
        digests = [optimizer_digest(s['optimizer']) for s in pair]
        require(digests[0] == digests[1], 'Recovered full optimizer state differs')
        restored = json.loads((replay/'recoveries/reproduce-3-4'/f'restored-rank-{rank:05d}.json').read_text())
        require(restored['full_active_model_bytes_equal'] and restored['full_optimizer_bytes_equal']
                and restored['tied_parameter_objects_preserved'], 'Actual device restore failed')
        shards.append(dict(rank=rank, terminal_optimizer_digest=canonical_hash(digests[0])))
        del pair, digests
    flags = shlex.split(p['runtime_policy']['environment']['NEURON_CC_FLAGS'])
    require(flags[:2] == ['--model-type=transformer', '--auto-cast=none']
            and all(v.startswith('--compile_workdir=') for v in flags[2:]),
            'Explicit precision contract differs')
    result = dict(status='PASS_UNIFORM_LM_NATIVE_QUALIFICATION', utc=utc_now(),
                  source_files=p['source_files'], protocol_sha256=sha256_file(protocol),
                  accumulation=8, world_size=8, width=528, actual_optimizer_updates=4,
                  replayed_optimizer_updates=2, initial_loss_absolute_error=error,
                  checks={k: True for k in ('initial_loss_reference', 'single_gradient_reducer',
                      'fp32_master_update', 'actual_optimizer_lr', 'replica_agreement',
                      'full_state_recovery', 'explicit_precision')},
                  full_terminal_model_digest=canonical_hash(model_digests),
                  all_optimizer_shards=shards, complete_manifests=manifests,
                  reserved_chip_hours=cost, closures=closures,
                  scope='Hardware mechanics and exact deterministic recovery, not predictive improvement')
    atomic_json(control/'native-qualification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_files', 'complete_manifests')}))


if __name__ == '__main__':
    main()
