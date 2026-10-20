"""Prepare one bounded TRAIN-only qualification and native initial-loss reference.

CPU only. This does not launch hardware or inspect a held model outcome.
"""
from pathlib import Path
import copy
import json
import math
import os
import subprocess

import torch
from transformers import AutoTokenizer

from train import partial_state_driver as driver
from train import real_cpt_scaffold as scaffold
from train.evaluate import asset_provenance
from train.masked_causal_ce import masked_component_causal_ce
from train.neuron_backend import NeuronCausalBackend
from train.useful_note_precision import cast_parameters
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.pretrained import load_base, sampled_fingerprint

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/real-cpt-scaffold-v29'
C = R/'local/diagnostics/real-cpt-scaffold-v29-qualification-v1'
D = R/'local/data/real-cpt-scaffold-v29-cpu-v2'

assert os.environ.get('USE_TORCH_XLA') == '0'
assert subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=W,text=True).strip() == 'e7c2508'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert sha256_file(D/'TRAIN.jsonl') == driver.SCAFFOLD_PARENT_SHA
assert sha256_file(D/'manifest.json') == driver.SCAFFOLD_MANIFEST_SHA
parent = [json.loads(line) for line in (D/'TRAIN.jsonl').open()]
indices = scaffold.grouped_order(parent, seed=2026102029, passes=1)[:128]
rows = [parent[i] for i in indices]
data_path = C/'TRAIN128.jsonl'
with data_path.open('x') as f:
    for row in rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
template = R/'local/diagnostics/real-source-balanced-v17-study-v1/protocol-source_cot-seed2026101913.json'
p = copy.deepcopy(json.loads(template.read_text()))
for key in ('task_loss_weight','hardware_qualification_file','hardware_qualification_sha256',
            'held_primary_manifest','held_primary_manifest_sha256','prospective_design_sha256'):
    p.pop(key,None)
p.update(schema=driver.SCAFFOLD_SCHEMA, experiment_id='real-cpt-scaffold-v29',
    purpose='real_cpt_scaffold_qualification', items=128, steps=8, save_every=4,
    seed=2026102029, lr=1e-5, world_size=8, micro_batch=4, global_batch=32,
    accumulation=1, auxiliary_micro_batch=1, width=640, target_slots=640,
    source_loss_weight=1., continuation_loss_weight=.5, work_loss_weight=1.,
    auxiliary_batch_fraction=.25, source_files=driver.source_files(), visible_cores='0-7',
    view_schedule='clean_plus_raw_format_computation_cycle',
    loss='clean_LM_plus_fraction_weighted_auxiliary_continuation_work',
    parent_data_file=str(D/'TRAIN.jsonl'), parent_data_sha256=driver.SCAFFOLD_PARENT_SHA,
    data_file=str(data_path), data_sha256=sha256_file(data_path),
    data_manifest_file=str(D/'manifest.json'), data_manifest_sha256=driver.SCAFFOLD_MANIFEST_SHA,
    checkpoint_export_cleanup=True, checkpoint_memory_diagnostic=True)
driver.validate(p)
assets = asset_provenance(Path(p['model_dir']))
base = json.loads(Path(p['base_protocol']).read_text())
assert sha256_file(p['base_protocol']) == p['base_protocol_sha256']
assert assets['manifest_sha256'] == base['model_manifest_sha256']
tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
order = driver.order_for(p, rows)
atomic_json(C/'CPU-reference-plan.json',dict(utc=utc_now(), status='CPU_TRAIN_ONLY_REFERENCE_PREPARED_BEFORE_MODEL_LOAD',
    data_sha256=p['data_sha256'], source_files=p['source_files'], assets=assets,
    reference='Rank-average source/work/continuation means of actual world8 step0 raw view',
    comparisons_fixed_before_device_calls=dict(max_component_mean_abs_error=.02,max_total_abs_error=.025),
    CPU_threads=torch.get_num_threads(), source_script_sha256=sha256_file(__file__)))
model, _ = load_base(p['model_dir'])
assert sampled_fingerprint(model) == p['base_signature']
conversion = cast_parameters(model)
assert all(t.dtype == torch.bfloat16 for t in model.parameters())
assert all(not t.is_floating_point() or t.dtype == torch.float32 for _,t in model.named_buffers())
assert sampled_fingerprint(model) == p['base_signature']
backend = NeuronCausalBackend(model, head_chunk=32, checkpoint_trunk=False)
rank_values = []
for rank in range(8):
    means = dict(source=0.,work=0.,continuation=0.)
    counts = dict(source=0,work=0,continuation=0)
    total = 0.
    with torch.no_grad():
        for branch, batch, weights in driver.scaffold_passes_at(rows,tok,p,order,
                driver.SCAFFOLD_QUALIFICATION_ARM,0,rank):
            ce = masked_component_causal_ce(backend,**batch,component_weights=weights)
            total += float(ce.loss)
            for name in ce.losses:
                means[name] += float(ce.losses[name])
                counts[name] += int(ce.counts[name].sum())
            del ce, batch
    assert math.isfinite(total) and all(math.isfinite(v) for v in means.values())
    record = dict(rank=rank,total_loss=total,means=means,counts=counts,
        ordered_item_ids=[rows[i]['item_id'] for i in order[rank*4:rank*4+4]])
    rank_values.append(record)
    print(json.dumps(record),flush=True)
reference = dict(status='CPU_NATIVE_BF16_INITIAL_STEP_REFERENCE_NOT_HARDWARE_PASS',utc=utc_now(),
    data_sha256=p['data_sha256'],parent_data_sha256=p['parent_data_sha256'],
    source_files=p['source_files'],preparer_sha256=sha256_file(__file__),
    source_commit='e7c2508',assets=assets,base_signature=p['base_signature'],
    parameter_dtype='bf16',floating_buffer_dtype='fp32',logsoftmax_dtype='fp32',
    per_rank=rank_values,global_means={name:math.fsum(v['means'][name] for v in rank_values)/8
        for name in ('source','work','continuation')},
    global_total_loss=math.fsum(v['total_loss'] for v in rank_values)/8,
    global_counts={name:sum(v['counts'][name] for v in rank_values) for name in ('source','work','continuation')},
    comparisons_fixed_before_device_calls=dict(max_component_mean_abs_error=.02,max_total_abs_error=.025),
    scope='Initial released B0 TRAIN-only CPU forward reference; no optimization, held outcome or accelerator call')
atomic_json(C/'CPU-initial-reference.json',reference)
p.update(initial_loss_reference_file=str(C/'CPU-initial-reference.json'),
         initial_loss_reference_sha256=sha256_file(C/'CPU-initial-reference.json'))
driver.validate(p)
atomic_json(C/'protocol.json',p)
atomic_json(C/'CPU-preparation-complete.json',dict(status='QUALIFICATION_PROTOCOL_AND_NATIVE_CPU_REFERENCE_READY',
    utc=utc_now(),protocol_sha256=sha256_file(C/'protocol.json'),spec_hash=canonical_hash(p),
    data_sha256=p['data_sha256'],reference_sha256=p['initial_loss_reference_sha256'],
    hardware_launched=False,source_commit='e7c2508'))
