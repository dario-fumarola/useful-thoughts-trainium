"""One-core, no-training full16 scorer parity/throughput qualification.

The closed B0 CPU reference remains authoritative. No model or scientific
endpoint is selected by this engineering probe, and it never samples a note.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import statistics
import subprocess
import time
from types import SimpleNamespace
import torch

from train.real_trace_evaluation import span_logprobs
from train.real_trace_contexts import reader_context
from train.real_trace_notes import ROOT, read, rows, sources
from train.evaluate_local import setup_model
from train.evaluate import asset_provenance
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.runtime_policy import runtime_policy, require_runtime_policy

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/real-trace-eval-v10'
assert ROOT == W
D = R/'local/diagnostics/real-span-XLA-B0-v14-v1'
O = R/'local/nvme-cpt/diagnostics/real-span-XLA-B0-v14-v1'
plan = read(D/'plan.json')
assert sha256_file(__file__) == plan['profile_sha256']
assert os.environ['NEURON_RT_VISIBLE_CORES'] == '0' and os.environ['NEURONCORE_NUM_DEVICES'] == '1'
require_runtime_policy(plan['runtime_policy'], runtime_policy())
topology = json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls', '--json-output'], text=True))
assert all(not d['neuron_processes'] for d in topology if d['neuron_device'] == 0)
O.mkdir(exist_ok=False)
atomic_json(D/'launch.json', dict(utc=utc_now(), pid=os.getpid(), plan_sha256=sha256_file(D/'plan.json'),
                                 profile_sha256=sha256_file(__file__)))
for name, digest in plan['fixed_files'].items():
    assert sha256_file(name) == digest
qs, targets, supplied = (rows(Path(plan[name])) for name in ('questions', 'targets', 'oracle_controls'))
bundle = read(plan['notes_bundle'])
assert bundle['status'] == 'COMPLETE_REAL_PREFIX_ONLY_NOTE_BUNDLE' and bundle['model_id'] == 'B0'
ids = [v['item_id'] for v in qs]
assert len(ids) == len(set(ids)) == 256
assert ids == [v['item_id'] for v in targets] == [v['item_id'] for v in supplied] == bundle['ordered_item_ids']
assert [v['item_id'] for v in bundle['controls']] == ids
for name, digest in bundle['source_pins'].items():
    assert sha256_file(name) == digest
reference_run = read(plan['CPU_reference_run'])
assert reference_run['status'] == 'COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS'
assert reference_run['model_id'] == 'B0' and reference_run['documents'] == 256
reference = {(v['item_id'], v['mode']): v for v in rows(Path(plan['CPU_reference_scores']))}
assert len(reference) == 1280
began = time.monotonic()
result = dict(status='running', started_utc=utc_now(), scope='Frozen B0 DEV_AST256,all5modes,one logical core,FP32,no training',
              rows=0, profile_sha256=sha256_file(__file__), source_files=sources(),
              CPU_reference_run_sha256=sha256_file(plan['CPU_reference_run']))
atomic_json(O/'run.json', result)
timings, errors = [], []
try:
    backend, tokenizer, loaded, signature, _ = setup_model(SimpleNamespace(
        model_dir=Path(plan['model_dir']), checkpoint=None, device='xla'))
    assert loaded is None and signature == reference_run['model_signature']
    assert asset_provenance(Path(plan['model_dir'])) == reference_run['assets']
    assert all(p.dtype == torch.float32 for p in backend.model.parameters())
    assert backend.device.type == 'xla'
    modes = ('raw', 'empty', 'oracle', 'own', 'other')
    ordered = sorted(range(256), key=lambda i: len(qs[i]['prefix_ids']))
    with (O/'scores.jsonl').open('x') as stream:
        for mode in modes:
            for start in range(0, 256, 4):
                available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                                     if l.startswith('MemAvailable:')))*1024
                if available < 512*2**30:
                    raise RuntimeError('Host availability floor; no retry')
                contexts, batch = [], ordered[start:start+4]
                for i in batch:
                    note = (None if mode == 'raw' else [] if mode == 'empty' else
                            supplied[i]['oracle_ids'] if mode == 'oracle' else
                            bundle['controls'][i][mode]['content_ids'])
                    contexts.append(reader_context(qs[i], tokenizer, note))
                before = time.monotonic()
                actual = span_logprobs(backend, contexts, [targets[i]['target16'] for i in batch], width=672)
                wall = time.monotonic()-before
                timings.append(wall)
                for i, lp in zip(batch, actual.tolist()):
                    expected = reference[ids[i], mode]['logps16']
                    delta = [a-b for a, b in zip(lp, expected)]
                    errors.append(delta)
                    value = dict(item_id=ids[i], document_id=qs[i]['document_id'], mode=mode,
                                 logps16=lp, CPU_logps16=expected, batch_wall_seconds=wall,
                                 maximum_token_error=max(map(abs, delta)))
                    stream.write(json.dumps(value, allow_nan=False)+'\n')
                    result['rows'] += 1
                stream.flush()
                if max(abs(v) for row in errors[-4:] for v in row) > .001:
                    raise RuntimeError('Declared full16 token parity exceeded; preserve and no adoption')
                if len(timings) == 1:
                    topo = json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls', '--json-output'], text=True))
                    active = [d for d in topo if d['neuron_device'] == 0]
                    assert len(active) == 1 and len(active[0]['neuron_processes']) == 1
                    atomic_json(O/'actual-single-core.json', dict(utc=utc_now(), topology=active))
                if start % 64 == 0:
                    print(json.dumps(dict(utc=utc_now(), mode=mode, rows=result['rows'], last_batch_seconds=wall)), flush=True)
    assert result['rows'] == 1280 and len(timings) == 320
    mean_span_drift = abs(sum(sum(v) for v in errors)/1280)
    warm_median = statistics.median(timings[1:])
    warm_panel_estimate = statistics.mean(timings[1:])*320
    speedup = reference_run['wall_seconds']/warm_panel_estimate
    numerical = mean_span_drift <= .001
    result.update(status='COMPLETE_FIXED_REFERENCE_PARITY_AND_THROUGHPUT_PROFILE',
                  max_token_error=max(abs(v) for row in errors for v in row),
                  absolute_mean_span_drift=mean_span_drift, numerical_gate_pass=numerical,
                  first_batch_seconds=timings[0], warm_median_batch4_seconds=warm_median,
                  estimated_warm_panel_seconds=warm_panel_estimate,
                  canonical_CPU16thread_panel_seconds=reference_run['wall_seconds'],
                  estimated_warm_speedup_vs_CPU=speedup,
                  declared_performance_gate_pass=bool(numerical and speedup >= 4),
                  scores_sha256=sha256_file(O/'scores.jsonl'),
                  limits=['CPU reference is unchanged and authoritative for the live scientific study.',
                          'One logical Trainium core versus16 CPU threads; not equal resource cost or service latency.',
                          'Warm estimate excludes first compilation batch; first-batch and total wall are separately retained.',
                          'Actual frozen B0 weights and all256documents×5modes; no new independent scientific observations.',
                          'This does not qualify an RL step, BF16 sampling, weight synchronization or gradients.'])
except BaseException as error:
    result.update(status='FAILED_ENGINEERING_PROFILE_NO_ADOPTION', error=f'{type(error).__name__}: {error}')
    raise
finally:
    result.update(finished_utc=utc_now(), total_wall_seconds=time.monotonic()-began)
    atomic_json(O/'run.json', result, overwrite=True)
print(json.dumps({k: v for k, v in result.items() if k != 'source_files'}, indent=2), flush=True)
