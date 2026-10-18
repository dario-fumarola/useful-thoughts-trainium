"""Finite W/Cprime/T supervised diagnostic; fresh B0 and one ZeRO reducer.

W=writer mean+reader natural16 mean+0.5 source mean.
Cprime=raw natural16 mean+0.5 source mean, in one ordinary source pass.
T=ordinary all-token CE on note-augmented source, NOT an exact TPT recipe.
Full-state recovery uses original protocols, FP32 masters and an actual byte
audit before RNG reset. Hardware recovery qualification is separate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'useful-note-sft-v5'
WEIGHTS = {'W': {'writer': 1., 'reader': 1., 'source': .5},
           'Cprime': {'continuation': 1., 'source': .5}, 'T': {'augmented': 1.}}
COMPONENTS = ('source', 'writer', 'reader', 'augmented')
METRIC_COMPONENTS = COMPONENTS + ('continuation',)


def planned_steps(p):
    """Recipe exposure; --qualify caps the64-item recipe at3 actual updates."""
    if p.get('purpose') == 'qualification':
        if p['items'] != 64:
            raise ValueError('Qualification requires the declared first64-item subset')
        return p['items']*p['epochs']//p['global_batch']
    if p.get('purpose') != 'science' or p['items'] < 5000:
        raise ValueError('Science requires at least5000 items; qualification is separate')
    exposures = p['items'] * p['epochs']; batch = p['global_batch']
    if p.get('tail_policy') == 'exact':
        if exposures % batch: raise ValueError('Exact epochs require a whole global batch')
        return exposures // batch
    if p.get('tail_policy') == 'ceil_repeat': return (exposures+batch-1) // batch
    if p.get('tail_policy') == 'drop_tail': return exposures // batch
    raise ValueError('Declare exact, ceil_repeat or drop_tail exposure policy')


def source_files():
    from train.filter_symbolic_notes import sources
    from train.general_hindsight_filtered_data import sources as general_sources
    names = ('train/useful_note_sft_driver.py', 'train/useful_note_sft_data.py',
             'train/masked_causal_ce.py', 'train/neuron_driver.py',
             'train/warmstart_driver.py', 'train/useful_note_precision.py',
             'train/useful_note_controls.py', 'train/prepare_reader_learning_data.py',
             'train/prepare_general_sft_data.py',
             'train/useful_note_recovery.py',
             'weekend_thinking/runtime_policy.py',
             'weekend_thinking/checkpoints.py')
    return {**sources(), **general_sources(), **{name: sha256_file(ROOT/name) for name in names}}


def validate_protocol(p):
    from train.useful_note_precision import contract
    from weekend_thinking.runtime_policy import POLICY_ENVIRONMENT
    fixed = dict(schema=SCHEMA, epochs=2,
        global_batch=32, lr=1e-5, warmup_steps=20, betas=[.9,.95], eps=1e-8,
        weight_decay=.1, clip=1., loss_weights=WEIGHTS, parameter_dtype='bf16',
        gradient_dtype='bf16', precision_contract=contract(), compute_dtype='bf16',
        optimizer_dtype='fp32', schedule='linear_warmup_then_cosine',
        checkpoint_policy={'development_model_only': [64], 'full_state': 'terminal_or_STOP'})
    if any(isinstance(p.get(k), bool) or p.get(k) != v for k,v in fixed.items()):
        raise ValueError('Freeze the two-epoch W/Cprime/T BF16-storage recipe')
    if any(type(p[k]) is not int for k in ('seed','items','epochs','steps','global_batch','warmup_steps','world_size','accumulation')):
        raise ValueError('Integer recipe counts required')
    if not 0 <= p['seed'] < 2**63 or p['items'] < 1 or p['steps'] != planned_steps(p):
        raise ValueError('Explicit seed/item count/actual update bound required')
    if (p['purpose']=='qualification' and p.get('qualification_state_policy') not in ('small_slices','full_checkpoint')):
        raise ValueError('Declare small nonresumable qualification slices or one full checkpoint')
    if (p['world_size'], p['accumulation']) not in ((8,4),(32,1)):
        raise ValueError('World8/accum4 or world32/accum1 must realize global32')
    if (p['world_size'],p['accumulation'])==(8,4) and p.get('topology_alternative')!='explicit_world8_BF16_accumulation':
        raise ValueError('World32/accum1 primary; world8 needs an explicit alternate identity')
    if p.get('interface') != 'I3':
        raise ValueError('Ordinary I3 reader-learning fallback and compute precision required; I6 is evaluation only')
    if set(p.get('packing',{})) != set(COMPONENTS):
        raise ValueError('All four component shapes must be frozen')
    for c, shape in p['packing'].items():
        if (set(shape) != {'width','target_slots'} or any(type(v) is not int or v <= 0 for v in shape.values())
                or shape['width'] % 32 or (shape['target_slots'] != 16 if c == 'reader' else shape['target_slots'] % 32)):
            raise ValueError('Compact ceil32 shapes; reader has16 exact slots')
    policy = p.get('runtime_policy',{})
    if (policy.get('schema') != 1 or set(policy.get('environment',{})) != set(POLICY_ENVIRONMENT)
            or policy['environment'].get('XLA_COMPILATION_CACHE_SIZE') != '2'
            or policy['environment'].get('PJRT_DEVICE') != 'NEURON'
            or any(policy['environment'].get(k) is not None for k in ('XLA_USE_BF16','XLA_DOWNCAST_BF16','NEURON_NUM_RECENT_MODELS_TO_KEEP'))
            or p.get('allocator_environment') != {'NEURON_RT_CONTIGUOUS_SCRATCHPAD':'0'}):
        raise ValueError('Bind LRU2 and the qualified noncontiguous allocator')
    if p.get('source_files') != source_files():
        raise ValueError('SFT source/packing closure changed')


def learning_rate(step, p):
    """1-based science updates: warmup20, positive finalLR, zero afterward.

    Cosine denominator=steps-warmup+1. The last actual optimizer update remains
    positive; the hypothetical next rate is0. Qualification uses only the first
    three warmup rates and need not complete the scientific schedule.
    """
    if type(step) is not int or not 1 <= step <= p['steps']+1:
        raise ValueError('LR requires an actual1-based planned update')
    if step <= p['warmup_steps']:
        return p['lr'] * step / p['warmup_steps']
    return p['lr'] * .5 * (1 + math.cos(math.pi * (step-p['warmup_steps']) / (p['steps']-p['warmup_steps']+1)))


def batch_indices(step, rank, p, count=None, order=None):
    from train.warmstart_driver import order_index
    if not 0 <= rank < p['world_size'] or not 0 <= step < p['steps']:
        raise ValueError('Invalid ordered step/rank')
    positions=[step*p['global_batch']+micro*p['world_size']+rank for micro in range(p['accumulation'])]
    return ([order[pos] for pos in positions] if order is not None else
        [order_index(pos,p['items'] if count is None else count,p['seed']) for pos in positions])


def inventory_contract(manifest):
    if manifest.get('schema') == 'useful-note-supervised-train-v3':
        return 'teacher_note_kind', 'exact-postorder-v1', False
    if manifest.get('schema') == 'general-native-supervised-train-v1':
        return 'teacher_kind', 'future-aware-general-firstline-native-v1', True
    raise ValueError('Explicit symbolic or GENERAL native training schema required')


def validate_inventory(p,manifest,rows):
    """Unique item/cut identities; repeated TRAIN documents remain explicit."""
    ids=[r['item_id'] for r in rows];docs=[r['document_id'] for r in rows]
    kind_field, kind, likelihood = inventory_contract(manifest)
    cut_field='native_window_cut' if likelihood else 'absolute_source_cut'
    cuts=[r.get(cut_field) for r in rows]
    if (manifest.get('status') != 'complete_CPU_inventory_NOT_Neuron_training_qualification'
            or len(rows) != p['items'] or len(set(ids)) != len(rows)
            or any(type(c) is not int or c < 0 for c in cuts)
            or len(set(zip(docs,cuts))) != len(rows)
            or any(r.get('split') != 'train' or r.get(kind_field) != kind for r in rows)
            or manifest.get('items') != p['items'] or manifest.get('unique_documents') != len(set(docs))
            or manifest.get('purpose') != p['purpose'] or manifest.get('split') != 'train'
            or manifest.get('training_sha256') != p['data_sha256']
            or manifest.get('selection_uses_likelihood') is not likelihood
            or manifest.get('ordered_item_ids_sha256') != canonical_hash(ids)
            or manifest.get('ordered_document_ids_sha256') != canonical_hash(docs)
            or any(manifest.get(k) != p[k] for k in ('packing','interface','parts'))):
        raise ValueError('Complete fixed TRAIN item/document/cut inventory and native packing required')
    if likelihood and (manifest.get('teacher_kind') != kind
            or p['items'] != (64 if p['purpose']=='qualification' else 5120)
            or len(set(docs)) != len(rows) or any(c != 512 for c in cuts)
            or any(len(r['prefix_ids']) != 512 or len(r['natural_future16_ids']) != 16
                or r['original_source_ids'] != r['prefix_ids']+r['natural_future16_ids']
                or r.get('selection_uses_likelihood') is not True for r in rows)):
        raise ValueError('GENERAL requires unique documents, original native512+16 and honest gain selection')
    return len(set(docs))


def load_inputs(p):
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    for key in ('data','dataset'):
        if sha256_file(p[f'{key}_file']) != p[f'{key}_sha256']:
            raise ValueError('Frozen training/inventory bytes changed')
    manifest = json.loads(Path(p['dataset_file']).read_text())
    rows = [json.loads(s) for s in Path(p['data_file']).read_text().splitlines() if s.strip()]
    validate_inventory(p,manifest,rows)
    if p['purpose']=='qualification':
        parent=Path(manifest['parent_science_manifest'])
        if (sha256_file(parent)!=manifest['parent_science_manifest_sha256']
                or manifest['preparation_source_sha256']!=p['source_files']['train/useful_note_sft_driver.py']):
            raise ValueError('Qualification subset preparation/parent changed')
        science=json.loads(parent.read_text());data=Path(science['training_file'])
        if (science['purpose']!='science' or science['schema']!=manifest['schema']
                or science['items']<5000 or sha256_file(data)!=manifest['parent_science_training_sha256']):
            raise ValueError('Qualification requires the fixed new science parent')
        if manifest['schema']=='general-native-supervised-train-v1':
            if (science['items']!=5120 or science['unique_documents']!=5120
                    or science['training_sha256']!=manifest['parent_science_training_sha256']):
                raise ValueError('GENERAL qualification requires the actual complete5120-document parent')
            parent_rows=[json.loads(line) for line in data.read_text().splitlines()]
            parent_protocol=dict(p,items=5120,purpose='science',data_sha256=science['training_sha256'])
            validate_inventory(parent_protocol,science,parent_rows)
        with data.open('rb') as f: first=b''.join(next(f) for _ in range(64))
        if Path(p['data_file']).read_bytes()!=first:
            raise ValueError('Qualification must preserve the exact first64 science rows')
    for name,digest in {**manifest['filter_input_hashes'], **manifest['source_files']}.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        if sha256_file(path) != digest or (name in manifest['source_files'] and p['source_files'].get(name) != digest):
            raise ValueError('Original filter or inventory source changed')
    assets = asset_provenance(Path(p['model_dir']))
    if assets['model_id'] != 'Qwen/Qwen3-1.7B-Base' or assets['manifest_sha256'] != p['model_manifest_sha256']:
        raise ValueError('Released B0 asset manifest required')
    tok = AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    from train.useful_note_sft_data import writer_stop_ids
    if (canonical_hash(tok.get_vocab()) != manifest['tokenizer_vocab_sha256']
            or list(writer_stop_ids(p['interface'],p['parts'])) != manifest['writer_stop_ids']):
        raise ValueError('Native vocabulary/first-close stop contract changed')
    return rows,tok,assets,manifest


def prepare_qualification_inventory(science_dir,output,model_dir,deadline):
    """First64 of the immutable science inventory; preserve its global shapes.

    CPU packing only, no likelihood selection or model load. This subset does
    not satisfy the >=5000 science quota and is never a scientific training arm.
    """
    from transformers import AutoTokenizer
    from train.useful_note_sft_data import pack_example
    science_dir=Path(science_dir);output=Path(output)
    def check():
        if datetime.now(timezone.utc)>=require_utc_deadline(deadline):raise TimeoutError('Qualification inventory deadline')
    check();path=science_dir/'manifest.json';m=json.loads(path.read_text())
    data=Path(m['training_file'])
    _,_,uses_likelihood=inventory_contract(m)
    if (output.exists() or m['purpose']!='science'
            or m['status']!='complete_CPU_inventory_NOT_Neuron_training_qualification'
            or m['items']<5000 or m['selection_uses_likelihood'] is not uses_likelihood
            or sha256_file(data)!=m['training_sha256']):
        raise ValueError('Fresh qualification output and complete new science inventory required')
    if uses_likelihood and (m['items']!=5120 or m['unique_documents']!=5120):
        raise ValueError('GENERAL qualification requires the actual complete5120-document quota')
    lines=data.read_bytes().splitlines(keepends=True)
    rows=[json.loads(line) for line in lines]
    if (len(rows)!=m['items'] or canonical_hash([r['item_id'] for r in rows])!=m['ordered_item_ids_sha256']
            or canonical_hash([r['document_id'] for r in rows])!=m['ordered_document_ids_sha256']):
        raise ValueError('Parent science inventory/order changed')
    for name,digest in {**m['input_hashes'],**m['source_files']}.items():
        if sha256_file(Path(name) if Path(name).is_absolute() else ROOT/name)!=digest:
            raise ValueError('Parent science provenance changed')
    selected=rows[:64];tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    if canonical_hash(tok.get_vocab())!=m['tokenizer_vocab_sha256']:raise ValueError('Native vocabulary changed')
    counts={c:dict(examples=0,active_labels=0,note_close_labels=0,non_note_labels=0) for c in COMPONENTS}
    for row in selected:
        check()
        for c in COMPONENTS:
            batch=pack_example(row,tok,c,interface=m['interface'],parts=m['parts'],**m['packing'][c]);packed_counts(batch,c)
            n=int(batch['note_close_mask'].sum());f=int(batch['future_mask'].sum())
            for key,value in dict(examples=1,active_labels=n+f,note_close_labels=n,non_note_labels=f).items():counts[c][key]+=value
    check();output.mkdir(parents=True);target=output/'training.jsonl';target.write_bytes(b''.join(lines[:64]))
    keys=('interface','parts','packing','tokenizer_vocab_sha256','writer_stop_ids','mask_semantics','reader_target',
          'input_hashes','filter_input_hashes','source_files','selection_uses_likelihood')
    manifest={k:m[k] for k in keys}
    if uses_likelihood:manifest['teacher_kind']=m['teacher_kind']
    manifest.update(schema=m['schema'],status=m['status'],purpose='qualification',split='train',
        items=64,unique_documents=len({r['document_id'] for r in selected}),component_counts=counts,
        training_file=str(target.resolve()),training_sha256=sha256_file(target),created_utc=utc_now(),
        ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in selected]),
        ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in selected]),
        selection='First64 unchanged science rows; no outcome selection, three actual global32 updates, NOT science quota.',
        parent_science_manifest=str(path.resolve()),parent_science_manifest_sha256=sha256_file(path),
        parent_science_training_sha256=m['training_sha256'],preparation_source_sha256=sha256_file(__file__))
    atomic_json(output/'manifest.json',manifest)
    return manifest


def load_native_reference(p,assets):
    import torch
    from train.useful_note_precision import SCHEMA as NATIVE_SCHEMA,contract,sources
    output=Path(p['native_reference_output'])
    for name,key in (('reference.json','native_reference_sha256'),('next-logps.pt','native_logps_sha256')):
        if sha256_file(output/name)!=p[key]:raise ValueError('New BF16 native reference changed')
    report=json.loads((output/'reference.json').read_text())
    if (report.get('schema')!=NATIVE_SCHEMA or report.get('precision')!=contract()
            or report.get('status')!='complete_CPU_FP32_canonical_reference_for_BF16_device'
            or report['assets']!=assets or report['base_signature']!=p['base_signature']
            or report['source_files']!=sources() or report['width']!=1024):
        raise ValueError('Require the new BF16-storage reference, not old FP32 parity')
    if sha256_file(report['contexts_file'])!=report['contexts_sha256']:
        raise ValueError('Native causal contexts changed')
    values=torch.load(output/'next-logps.pt',map_location='cpu',weights_only=True)
    if values.dtype!=torch.float32 or values.shape!=(4,151936) or not torch.isfinite(values).all():
        raise ValueError('Four finite full released-head CPU reference vectors required')
    return report['qualification_refs'],values


def publish_native_gate(output,rank,reference_index,actual,reference,hidden_dtype,head_dtype,p,protocol_hash,*,raise_on_failure=True):
    """Persist the actual vector and honest finite/KL result before any raise.

    Called before ZeRO construction. An optimizer initialization failure must
    not erase a completed forward diagnostic; NaN/Inf stay in the tensor file,
    while the JSON contains finite counts/status rather than invalid numbers.
    """
    import torch
    from train.neuron_driver import stream_save
    from train.useful_note_precision import compare_distributions
    actual=actual.detach().cpu();path=output/f'forward-rank-{rank:05d}.pt'
    stream_save(path,actual)
    report=dict(rank=rank,reference_index=reference_index,created_utc=utc_now(),
        protocol_sha256=protocol_hash,precision_contract=p['precision_contract'],
        parameter_dtype='bf16',actual_hidden_dtype=hidden_dtype,actual_head_dtype=head_dtype,
        forward_sha256=sha256_file(path),actual_nonfinite_values=int((~torch.isfinite(actual)).sum()),
        native_check=None,status='FAILED_NATIVE_GATE')
    try:
        report['native_check']=compare_distributions(reference,actual)[0]
        if (report['native_check']['passed'] and hidden_dtype==head_dtype=='torch.bfloat16'):
            report['status']='PASS'
        else:report['failure']='Bidirectional KL or actual BF16 dtype does not meet the declared native contract'
    except (ValueError,RuntimeError) as error:
        report['failure']=str(error)
    atomic_json(output/f'qualification-rank-{rank:05d}.json',report)
    if report['status']!='PASS' and raise_on_failure:
        raise RuntimeError(f'New BF16 native gate failed; actual evidence saved in {output}')
    return report


def native_gate_consensus(output,rank,world,local_report,xm,device):
    """Every rank reaches the same decision before ZeRO or a local raise.

    Persisted numerical/dtype failures are ordinary consensus participants.
    Fatal device or I/O failures are still bounded by the external deadline.
    """
    import torch
    flag=torch.tensor([float(local_report['status']!='PASS')],device=device)
    values=xm.all_gather(flag).reshape(-1).cpu().tolist()
    if len(values)!=world or any(v not in (0.,1.) for v in values):
        raise RuntimeError('Invalid native failure consensus')
    failed=[i for i,value in enumerate(values) if value]
    report=dict(status='FAIL' if failed else 'PASS',rank=rank,world_size=world,
        created_utc=utc_now(),failed_ranks=failed,all_rank_failure_flags=values,
        local_gate_sha256=sha256_file(output/f'qualification-rank-{rank:05d}.json'),
        optimizer_construction_started=False)
    atomic_json(output/f'native-consensus-rank-{rank:05d}.json',report)
    xm.rendezvous('native-consensus-published')
    if failed:raise RuntimeError(f'Native gate failed on ranks {failed}; all ranks stop before ZeRO')
    return report


def packed_counts(batch,component):
    import torch
    note = batch['note_close_mask']; future = batch['future_mask']; active = note|future
    n,f = int(note.sum()),int(future.sum()); labels = n+f
    if ((note&future).any() or not labels
            or (component == 'writer' and (not 2 <= n <= 129 or f))
            or (component != 'writer' and n)
            or (component == 'source' and f != 527) or (component == 'reader' and f != 16)
            or component not in COMPONENTS
            or not torch.equal(batch['labels'][active],batch['input_ids'].gather(1,batch['target_positions'])[active])):
        raise ValueError('Declared component and native causal labels disagree')
    real = int(batch['target_positions'][active].max())+1
    return dict(labels=labels,real_input_tokens=real,trunk_slots=batch['input_ids'].numel()-1,
                head_slots=batch['target_positions'].numel())


def replicas_bitwise_equal(replicas):
    import torch
    values=replicas.contiguous()
    return all(torch.equal(v.view(torch.uint8),values[0].view(torch.uint8)) for v in values)


def dense_matmul_proxy(values,parameters,hidden_size,vocab_size):
    """Checkpointed dense operation proxy, not measured FLOPs.

    Approximate8 operations/weight/token includes recomputation/backward. This
    excludes quadratic attention, lookup, normalization, optimizer and transfers.
    """
    head=hidden_size*vocab_size;trunk=parameters-head
    if trunk<=0:raise ValueError('Invalid tied-model dimensions')
    return 8*sum(trunk*v['trunk_slots']+head*v['head_slots'] for v in values.values())


def inspect_optimizer(optimizer,expected_step,expected_parameters):
    """Full finite scans of actual FP32 shards/moments, one small host result."""
    import torch
    state=optimizer.state_dict();masters=state.get('sharded_master_weights',{});moments=state.get('base_state',{})
    if len(masters)!=expected_parameters or set(masters)!=set(moments) or not masters:
        raise RuntimeError('Missing initialized ZeRO masters or Adam moments')
    checked=[];steps=[];inventory=[]
    for key,master in masters.items():
        s=moments[key]
        if (not {'exp_avg','exp_avg_sq','step'}<=set(s) or master.dtype!=torch.float32 or any(s[name].dtype!=torch.float32 or s[name].shape!=master.shape
                for name in ('exp_avg','exp_avg_sq')) or s['step'].numel()!=1):
            raise RuntimeError('Actual master/moment dtype or shape differs from FP32')
        checked.extend((master,s['exp_avg'],s['exp_avg_sq']));steps.append(s['step'].reshape(()))
        inventory.append(dict(index=key,shape=list(master.shape),dtype=str(master.dtype),numel=master.numel()))
    with torch.no_grad():
        finite=torch.stack([torch.isfinite(t).all() for t in checked]).all()
        nonnegative=torch.stack([s['exp_avg_sq'].ge(0).all() for s in moments.values()]).all()
        result=torch.cat((torch.stack(steps).float(),torch.stack((finite,nonnegative)).float())).cpu()
    if (not bool(result[-2]) or not bool(result[-1]) or not result[:-2].eq(expected_step).all()):
        raise RuntimeError('Nonfinite optimizer state, negative second moment or wrong actual Adam counter')
    return dict(full_master_and_moment_finite=True,second_moments_nonnegative=True,adam_step=expected_step,
        initialized_parameter_shards=len(masters),master_numel=sum(t.numel() for t in masters.values()),
        master_moment_dtype='torch.float32',inventory=inventory),state


def qualification_slices(model,optimizer,step,state,rank,world):
    """Bounded samples after full finite verification; these cannot resume."""
    import torch
    report,actual=inspect_optimizer(optimizer,step,len(list(model.parameters())))
    keys=list(actual['sharded_master_weights']);selected=[keys[i*(len(keys)-1)//min(7,len(keys)-1)] for i in range(min(8,len(keys)))] if len(keys)>1 else keys
    tensors=[];layout=[]
    for key in selected:
        fields={'master':actual['sharded_master_weights'][key],**{name:actual['base_state'][key][name] for name in ('exp_avg','exp_avg_sq')}}
        for name,value in fields.items():
            sample=value.detach().reshape(-1)[:32];layout.append(dict(index=key,field=name,values=sample.numel()))
            tensors.append(sample)
    samples=torch.cat(tensors).cpu()
    model_samples=torch.cat([t.detach().reshape(-1)[:32] for t in model.parameters()]).cpu()
    return dict(schema='useful-note-qualification-slices-v1',resumable=False,step=step,rank=rank,world_size=world,
        metadata=state,optimizer_full_state_checks=report,optimizer_sample_layout=layout,optimizer_samples=samples,
        model_parameter_names=[name for name,_ in model.named_parameters()],model_first32_samples=model_samples)


def save_qualification_slices(output,model,optimizer,step,state,rank,world,xm,device):
    from train.neuron_driver import stream_save
    from weekend_thinking.checkpoints import capture_cpu_rng,restore_cpu_rng
    rng=dict(cpu=capture_cpu_rng(),device=int(xm.get_rng_state(device)));d=output/'qualification-final'
    d.mkdir(parents=True,exist_ok=True);xm.rendezvous('qualification-slices-start')
    payload=qualification_slices(model,optimizer,step,state,rank,world)
    stream_save(d/f'rank-{rank:05d}.pt',payload);xm.rendezvous('qualification-slices-written')
    if rank==0:
        paths=[d/f'rank-{r:05d}.pt' for r in range(world)]
        atomic_json(d/'COMPLETE.json',dict(schema='useful-note-qualification-slices-v1',resumable=False,
            step=step,world_size=world,run_id=output.name,spec_hash=state['spec_hash'],created_utc=utc_now(),
            files={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in paths}))
    xm.rendezvous('qualification-slices-complete');restore_cpu_rng(rng['cpu']);xm.set_rng_state(rng['device'],device)
    return d


def backward_components(row,tok,arm,backend,p,sync):
    """Accumulate means; never zero, clip or reduce gradients in this helper."""
    from train.useful_note_sft_data import pack_example
    from train.masked_causal_ce import masked_causal_ce
    if arm not in WEIGHTS: raise ValueError('Unknown W/Cprime/T arm')
    if arm=='Cprime':
        from train.useful_note_controls import plain_control_ce
        started=time.monotonic()
        batch=pack_example(row,tok,'source',interface=p['interface'],parts=p['parts'],**p['packing']['source'])
        counts=packed_counts(batch,'source')
        ce=plain_control_ce(backend,**batch,source_cut=len(row['prefix_ids']))
        values=[float(t.detach().cpu()) for t in (ce.loss,ce.source_loss,ce.continuation_loss)]
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite plain control CE')
        (ce.loss/p['accumulation']).backward();sync()
        return dict(source=dict(loss=values[1],**counts,rank_seconds=time.monotonic()-started),
            continuation=dict(loss=values[2],labels=16,real_input_tokens=0,trunk_slots=0,head_slots=0,rank_seconds=0.))
    result = {}
    for component,weight in WEIGHTS[arm].items():
        started = time.monotonic()
        batch = pack_example(row,tok,component,interface=p['interface'],parts=p['parts'],**p['packing'][component])
        counts = packed_counts(batch,component)
        ce = masked_causal_ce(backend,**batch); value = float(ce.loss.detach().cpu())
        if not math.isfinite(value): raise RuntimeError(f'Nonfinite {component} CE before backward')
        (ce.loss * (weight/p['accumulation'])).backward(); sync()
        result[component] = dict(loss=value,**counts,rank_seconds=time.monotonic()-started)
        del ce,batch
    return result


def save_development_model(output,model,step,state,rank,xm,device):
    """Atomic model-only milestone, explicitly nonresumable; RNG-transparent."""
    from train.neuron_driver import cpu_state, stream_save
    from weekend_thinking.checkpoints import capture_cpu_rng, restore_cpu_rng
    rng = dict(cpu=capture_cpu_rng(),device=int(xm.get_rng_state(device)))
    directory = output/'development'/f'step-{step:07d}'
    xm.rendezvous(f'development-{step}-start')
    if rank == 0:
        directory.mkdir(parents=True,exist_ok=False); weights = cpu_state(model.state_dict())
        weights['lm_head.weight'] = weights['model.embed_tokens.weight']
        stream_save(directory/'model.pt',dict(model=weights,step=step,run_id=output.name,spec_hash=state['spec_hash']))
        del weights
        path=directory/'model.pt'
        atomic_json(directory/'COMPLETE.json',dict(schema='useful-note-model-only-v1',resumable=False,
            step=step,run_id=output.name,spec_hash=state['spec_hash'],created_utc=utc_now(),metadata=state,
            files={'model.pt':dict(bytes=path.stat().st_size,sha256=sha256_file(path))}))
    xm.rendezvous(f'development-{step}-complete')
    restore_cpu_rng(rng['cpu']); xm.set_rng_state(rng['device'],device)
    return directory


def prepare_resume(output,checkpoint,attempt,p,arm,protocol_hash,rows,order):
    """Hash full state and preserve abandoned metric tails before device work.

    A resumed attempt uses the original immutable run/protocol/ordered data.
    Every old metric byte and completion receipt is retained in its segment.
    The finite launcher must establish that the original process group is gone.
    """
    import re
    from train.neuron_driver import verify_checkpoint
    from weekend_thinking.artifacts import atomic_bytes
    output=Path(output);checkpoint=Path(checkpoint)
    if (output.is_symlink() or checkpoint.is_symlink() or (checkpoint/'COMPLETE.json').is_symlink()
            or not output.is_dir() or checkpoint.parent.resolve()!=output.resolve()
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',attempt)):
        raise ValueError('Original ordinary run/checkpoint and safe unique recovery attempt required')
    manifest=verify_checkpoint(checkpoint,canonical_hash(p),p['world_size'])
    step=manifest['step'];run=json.loads((output/'run.json').read_text())
    expected=dict(schema=SCHEMA,run_id=output.name,arm=arm,protocol_sha256=protocol_hash,
        spec_hash=canonical_hash(p),order_hash=canonical_hash([rows[i]['item_id'] for i in order]),
        world_size=p['world_size'],accumulation=p['accumulation'],purpose='science',
        items=p['items'],scientific_steps=p['steps'],qualification=False)
    if (any(run.get(k)!=v for k,v in expected.items()) or manifest['run_id']!=output.name
            or type(step)is not int or not 1<=step<p['steps'] or checkpoint.name!=f'step-{step:07d}'
            or manifest.get('metadata',{}).get('qualification') is not False):
        raise ValueError('Unfinished original science identity and full checkpoint required')
    if any(manifest['metadata'].get(k)!=expected[k] for k in
            ('spec_hash','order_hash','arm','protocol_sha256','purpose','items')):
        raise ValueError('Full checkpoint metadata differs from original run')
    metrics=output/'metrics.jsonl';raw=metrics.read_bytes();lines=raw.splitlines(keepends=True)
    if len(lines)<step:raise ValueError('Checkpoint metrics are incomplete')
    for i,line in enumerate(lines[:step]):
        value=json.loads(line)
        ids=[rows[k]['item_id'] for k in order[i*p['global_batch']:(i+1)*p['global_batch']]]
        if (value['step']!=i+1 or value['ordered_item_ids']!=ids
                or value['counters']['source_records']!=(i+1)*p['global_batch']):
            raise ValueError('Checkpoint metrics/order/counters changed')
    abandoned=[]
    for parent in (output,output/'development'):
        for directory in parent.glob('step-*'):
            if not re.fullmatch(r'step-[0-9]{7}',directory.name):
                raise ValueError('Unexpected checkpoint directory name')
            if int(directory.name[5:])<=step:continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError('Ordinary checkpoint directory required')
            if parent==output and (directory/'COMPLETE.json').exists():
                raise ValueError('Resume requires the latest complete full checkpoint')
            abandoned.append(directory)
    completion=output/'completion.json'
    if completion.exists() and json.loads(completion.read_text()).get('status')=='complete':
        raise ValueError('Completed science run cannot be resumed')
    segment=output/'recoveries'/attempt
    if segment.exists():raise ValueError('Recovery segment must be fresh')
    segment.mkdir(parents=True)
    atomic_bytes(segment/'previous-metrics.jsonl',raw)
    if sha256_file(segment/'previous-metrics.jsonl')!=sha256_file(metrics):
        raise RuntimeError('Original metrics were not preserved byte exactly')
    if completion.exists():
        atomic_bytes(segment/'previous-completion.json',completion.read_bytes())
    moved=[]
    for directory in abandoned:
        relative=directory.relative_to(output);destination=segment/'abandoned'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);directory.rename(destination)
        moved.append(str(relative))
    # COMPLETE at stepS certifies only the firstS updates. Preserve all later
    # bytes, including a possibly interrupted JSON line, as an abandoned tail.
    atomic_bytes(metrics,b''.join(lines[:step]),overwrite=True)
    receipt=dict(status='CPU_INTEGRITY_AND_METRIC_PRESERVATION_ONLY',created_utc=utc_now(),
        checkpoint=str(checkpoint.resolve()),checkpoint_complete_sha256=sha256_file(checkpoint/'COMPLETE.json'),
        protocol_sha256=protocol_hash,run_sha256=sha256_file(output/'run.json'),start_step=step,
        previous_metrics_sha256=sha256_file(segment/'previous-metrics.jsonl'),
        continued_metrics_sha256=sha256_file(metrics),abandoned_metric_bytes=sum(map(len,lines[step:])),
        preserved_abandoned_checkpoint_directories=moved,
        limitations='No model/optimizer restore or next-update qualification follows from CPU integrity.')
    atomic_json(segment/'prepared.json',receipt)
    return manifest,segment


def recovery_consensus(output,rank,world,error,xm,device,stage):
    """Uniform refusal on payload/readback errors; retain all local evidence."""
    import torch
    if stage not in ('payload','restored'):raise ValueError('Unknown recovery barrier')
    local=output/f'{stage}-local-rank-{rank:05d}.json'
    atomic_json(local,dict(rank=rank,stage=stage,status='FAIL' if error else 'PASS',error=error))
    flags=xm.all_gather(torch.tensor([float(bool(error))],device=device)).reshape(-1).cpu().tolist()
    if len(flags)!=world or any(v not in (0.,1.) for v in flags):raise RuntimeError('Invalid recovery consensus')
    failed=[i for i,value in enumerate(flags) if value]
    report=dict(rank=rank,stage=stage,status='FAIL' if failed else 'PASS',failed_ranks=failed,
        local_receipt_sha256=sha256_file(local))
    atomic_json(output/f'{stage}-consensus-rank-{rank:05d}.json',report)
    xm.rendezvous(f'recovery-{stage}-consensus-published')
    if failed:raise RuntimeError(f'Recovery {stage} refused by ranks {failed}; all ranks stop')
    return report


def resume_boundary(stopped,rng,xm,device,sync):
    """Finish the collective stop check before resetting continuation streams."""
    from train.neuron_driver import restore_rng_after_sync
    run_updates=not stopped()
    restore_rng_after_sync(rng,xm,device,sync)
    return run_updates


def worker(_index,args,p,rows,tok,assets,inventory,refs,native,protocol_hash,order):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa: F401
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
    from train.runtime_compat import configure_runtime
    from train.neuron_backend import NeuronCausalBackend
    from train.neuron_driver import save_checkpoint
    from train.warmstart_driver import stop_due
    from weekend_thinking.pretrained import load_base,sampled_fingerprint
    from weekend_thinking.runtime_policy import require_runtime_policy,runtime_policy
    from train.useful_note_precision import cast_parameters,released_bf16_header,shard_expectations,master_samples,active_shard_samples,compare_distributions

    configure_runtime()
    if not hasattr(torch,'xla'): torch._register_device_module('xla',torch_xla)
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    device=torch_xla.device();rank,world=xr.global_ordinal(),xr.world_size()
    if world != p['world_size']: raise ValueError('Actual world differs from frozen topology')
    dist.init_process_group('xla',init_method='xla://')
    if any(os.environ.get(k)!=v for k,v in p['allocator_environment'].items()):raise ValueError('Actual allocator differs')
    random.seed(p['seed']+rank);np.random.seed((p['seed']+rank)%2**32);torch.manual_seed(p['seed'])
    header=released_bf16_header(p['model_dir'])
    model,_=load_base(p['model_dir']);signature=sampled_fingerprint(model)
    if signature != p['base_signature']: raise ValueError('CPU initialization differs from B0')
    conversion=cast_parameters(model);expected_masters=shard_expectations(model,rank,world)
    if sampled_fingerprint(model)!=signature:raise ValueError('Initial BF16 values differ from released weights')
    model.to(device);model.tie_weights()
    if (any(t.dtype != torch.bfloat16 for t in model.parameters())
            or any(t.is_floating_point() and t.dtype!=torch.float32 for t in model.buffers())
            or model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or sampled_fingerprint(model) != signature):
        raise ValueError('Actual tied BF16 parameters and original FP32 buffers required')
    for name,value in model.named_buffers():
        if canonical_hash(value.detach().cpu().reshape(-1).tolist())!=conversion['buffers'][name]['sha256']:
            raise ValueError('Original rotary buffer values changed on transfer')
    sync=lambda:torch_xla.sync(wait=True)
    backend=NeuronCausalBackend(model,sync=sync)
    evidence=args.output if args.resume is None else args.recovery_dir/'initialization'
    if rank==0:evidence.mkdir(parents=True,exist_ok=False)
    xm.rendezvous('native-output-root-ready');sync()
    ref_index=rank%len(refs);ref=refs[ref_index]
    with torch.no_grad():
        ids=torch.full((1,1024),151643,dtype=torch.long);ids[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
        h=backend.hidden(ids.to(device));logits=model.lm_head(h[:,len(ref['context_ids'])-1]);lp=logits.float().log_softmax(-1)
        sync();actual=lp.cpu()
        local_gate=publish_native_gate(evidence,rank,ref_index,actual,native[ref_index:ref_index+1],
                            str(h.dtype),str(logits.dtype),p,protocol_hash,raise_on_failure=False)
        del h,logits,lp,ids
    native_gate_consensus(evidence,rank,world,local_gate,xm,device)
    dtypes={}
    def record(name):
        def hook(_module,inputs,output):
            value=dict(input=str(inputs[0].dtype),output=str(output.dtype))
            if name in dtypes and dtypes[name] != value: raise RuntimeError('Compute dtype changed within run')
            dtypes[name]=value
        return hook
    hooks=[model.model.layers[0].self_attn.q_proj.register_forward_hook(record('first_q_proj')),
           model.lm_head.register_forward_hook(record('head_before_float'))]
    optimizer=ZeroRedundancyOptimizer(model.parameters(),AdamW,optimizer_dtype=torch.float32,
        grad_clipping=True,max_norm=p['clip'],save_master_weights=True,lr=p['lr'],
        betas=tuple(p['betas']),eps=p['eps'],weight_decay=p['weight_decay'],
        use_grad_acc_hook=False,higher_cc_precision=False)
    initial_masters=master_samples(optimizer,expected_masters)
    if not torch.equal(initial_masters.contiguous().view(torch.uint8),expected_masters['values'].contiguous().view(torch.uint8)):
        raise RuntimeError('Standard FP32 masters differ from exact released CPU shard samples')
    found_inf=torch.zeros((),device=device);spec_hash=canonical_hash(p)
    order_hash=canonical_hash([rows[i]['item_id'] for i in order]);steps=3 if args.qualify else p['steps']
    if rank==0 and args.resume is None:
        atomic_json(args.output/'run.json',dict(schema=SCHEMA,run_id=args.output.name,started_utc=utc_now(),
            arm=args.arm,protocol_sha256=protocol_hash,spec_hash=spec_hash,order_hash=order_hash,
            world_size=world,accumulation=p['accumulation'],planned_steps=steps,scientific_steps=p['steps'],
            purpose=p['purpose'],items=p['items'],unique_documents=inventory['unique_documents'],
            tail_policy=p.get('tail_policy'),deadline_utc=args.deadline_utc,
            qualification=bool(args.qualify),assets=assets,base_signature=signature,runtime_policy=runtime_policy(),
            allocator_environment=p['allocator_environment'],component_weights=WEIGHTS[args.arm],packing=p['packing'],
            precision_contract=p['precision_contract'],released_weight_header=header,parameter_conversion=conversion,
            zero_use_grad_acc_hook=False,zero_higher_cc_precision=False,
            one_epoch_component_counts=inventory['component_counts'],
            resume_implemented=True,
            limitations=['Recovery implementation requires separate hardware qualification; no uninterrupted-parity claim.','Same documents/updates do not match labels, gradient dose, tokens or FLOPs.',
                        'BF16 native KL gate is a forward-distribution check, not tail CE/gradient equivalence.','T is ordinary augmented CE, not exact TPT.']))
    xm.rendezvous('useful-note-output-ready');sync()
    torch.save(dict(expected_cpu=expected_masters['values'],actual_fp32_masters=initial_masters),
               evidence/f'initial-master-rank-{rank:05d}.pt')
    atomic_json(evidence/f'initial-master-rank-{rank:05d}.json',dict(rank=rank,compute_dtype=p['compute_dtype'],
        parameter_dtype='bf16',initial_master_samples_exact=True,master_inventory=expected_masters['entries'],
        base_signature=signature,protocol_sha256=protocol_hash))
    del actual
    deadline=require_utc_deadline(args.deadline_utc)
    def stopped():
        value=torch.tensor(float(stop_due(args.stop_file,deadline)),device=device)
        return bool(xm.all_reduce(xm.REDUCE_MAX,value).cpu())
    def sample():return torch.cat([t.detach().reshape(-1)[:32] for t in model.parameters()]).cpu()
    counters=dict(source_records=0,source_tokens=0,sampled_actions=0,labels={c:0 for c in METRIC_COMPONENTS})
    actual_steps=0;save_walls=[]
    if args.resume is not None:
        from train.useful_note_recovery import validate_payload,restore_and_verify
        error=None
        try:
            weights=torch.load(args.resume/'model.pt',weights_only=True,map_location='cpu')
            saved=torch.load(args.resume/f'rank-{rank:05d}.pt',weights_only=True,map_location='cpu')
            actual_steps=validate_payload(args.resume_manifest,weights,saved,model,protocol=p,
                protocol_sha256=protocol_hash,arm=args.arm,rank=rank,run_id=args.output.name,
                order_hash=order_hash,expected_lr=learning_rate(args.resume_manifest['step'],p))
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        recovery_consensus(args.recovery_dir,rank,world,error,xm,device,'payload')
        counters=json.loads(json.dumps(saved['counters']))
        error=None
        try:restored=restore_and_verify(model,optimizer,weights,saved,xm=xm,device=device,sync=torch_xla.sync)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        recovery_consensus(args.recovery_dir,rank,world,error,xm,device,'restored')
        atomic_json(args.recovery_dir/f'restored-rank-{rank:05d}.json',dict(restored,
            rank=rank,step=actual_steps,checkpoint_complete_sha256=sha256_file(args.resume/'COMPLETE.json'),
            protocol_sha256=protocol_hash,deadline_utc=args.deadline_utc))
        # The local helper reset preceded this collective audit. Reset AGAIN
        # after every recovery observation so the continuation has the exact
        # saved stream. zero_grad next clears the ZeRO loader's dummy gradients.
        run_updates=resume_boundary(stopped,saved['rng'],xm,device,torch_xla.sync)
        del weights,saved
    else:run_updates=not stopped()
    if run_updates:
        for step in range(actual_steps,steps):
            start=time.monotonic();optimizer.zero_grad(set_to_none=True);before=sample()
            master_before=master_samples(optimizer,expected_masters);lr=learning_rate(step+1,p)
            for group in optimizer.param_groups:group['lr']=lr
            local={c:dict(loss=0.,labels=0,real_input_tokens=0,trunk_slots=0,head_slots=0,rank_seconds=0.) for c in METRIC_COMPONENTS}
            for index in batch_indices(step,rank,p,order=order):
                for c,v in backward_components(rows[index],tok,args.arm,backend,p,sync).items():
                    for k,z in v.items():local[c][k]+=z
            if any(t.grad is not None and t.grad.dtype!=torch.bfloat16 for t in model.parameters()):
                raise RuntimeError('Actual parameter-gradient storage must remain BF16')
            finite=torch.stack([torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None]).all()
            if bool(xm.all_reduce(xm.REDUCE_MAX,(~finite).float()).cpu()):raise RuntimeError('Nonfinite accumulated gradient')
            opt_start=time.monotonic();optimizer.step(found_inf=found_inf);sync();opt_seconds=time.monotonic()-opt_start
            norm=float(optimizer.grad_norm.cpu());after=sample();delta=after.float()-before.float()
            master_after=master_samples(optimizer,expected_masters);master_delta=master_after-master_before
            active=active_shard_samples(model,expected_masters)
            if not torch.equal(active.contiguous().view(torch.uint8),master_after.bfloat16().contiguous().view(torch.uint8)):
                raise RuntimeError('Active BF16 model samples differ from preserved FP32 master cast')
            replicas=xm.all_gather(after.to(device)).reshape(world,-1).cpu().contiguous()
            equal=replicas_bitwise_equal(replicas)
            if not math.isfinite(norm) or norm<=0 or not torch.isfinite(after).all() or not equal:
                raise RuntimeError('Nonfinite norm/weights or sampled replica disagreement')
            if not torch.isfinite(master_after).all() or not master_delta.ne(0).any():
                raise RuntimeError('Real FP32 master samples must change; BF16 visible rounding may be unchanged')
            expected_dtype='torch.'+('bfloat16' if p['compute_dtype']=='bf16' else 'float32')
            if any(dtypes.get(name,{}).get('output')!=expected_dtype for name in ('first_q_proj','head_before_float')):
                raise RuntimeError('Actual matmul/head dtype before float differs from declared compute')
            if step in (1,2):
                initialized,_=inspect_optimizer(optimizer,step+1,len(list(model.parameters())))
                atomic_json(evidence/f'optimizer-after{step+1}-rank-{rank:05d}.json',initialized)
            keys=('loss','labels','real_input_tokens','trunk_slots','head_slots','rank_seconds')
            totals=xm.all_reduce(xm.REDUCE_SUM,torch.tensor([local[c][k] for c in METRIC_COMPONENTS for k in keys],device=device)).cpu().tolist()
            values={c:dict(zip(keys,totals[i*len(keys):(i+1)*len(keys)])) for i,c in enumerate(METRIC_COMPONENTS)}
            for c,v in values.items():
                v['loss']/=p['global_batch']
                for k in keys[1:-1]:v[k]=int(v[k])
                counters['labels'][c]+=v['labels']
            counters['source_records']+=p['global_batch'];counters['source_tokens']+=528*p['global_batch']
            actual_steps=step+1
            opt_rank_seconds=float(xm.all_reduce(xm.REDUCE_SUM,torch.tensor(opt_seconds,device=device)).cpu())
            metric=dict(step=actual_steps,time_utc=utc_now(),elapsed_seconds=time.monotonic()-start,lr=lr,
                components=values,gradient_norm=norm,sampled_update_l2=float(delta.norm()),
                sampled_changed_fraction=float(delta.ne(0).float().mean()),sampled_replicas_bitwise_equal=equal,
                sampled_fp32_master_update_l2=float(master_delta.norm()),
                sampled_fp32_master_changed_fraction=float(master_delta.ne(0).float().mean()),
                sampled_active_model_matches_cast_master=True,
                parameter_dtype='bf16',gradient_dtype='bf16',master_moment_dtype='fp32',
                optimizer_rank_seconds=opt_rank_seconds,
                compute_dtypes_before_float=dtypes,counters=json.loads(json.dumps(counters)),
                dense_checkpointed_matmul_proxy_operations=dense_matmul_proxy(values,signature['parameters'],model.config.hidden_size,model.config.vocab_size),
                proxy_excludes='Quadratic attention, lookup/normalization, optimizer, collectives and data/cache I/O; not measured FLOPs.',
                complete_ordered_passes=counters['source_records']//len(rows),
                ordered_item_ids=[rows[i]['item_id'] for i in order[step*p['global_batch']:(step+1)*p['global_batch']]])
            if rank==0:
                with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps(metric),flush=True)
            stop=stopped();state=dict(spec_hash=spec_hash,order_hash=order_hash,arm=args.arm,seed=p['seed'],
                global_batch=p['global_batch'],accumulation=p['accumulation'],protocol_sha256=protocol_hash,
                data_sha256=p['data_sha256'],dataset_sha256=p['dataset_sha256'],counters=counters,
                parameter_dtype='bf16',gradient_dtype='bf16',optimizer_dtype='fp32',
                precision_contract=p['precision_contract'],compute_dtype=p['compute_dtype'],runtime_policy=runtime_policy(),
                allocator_environment=p['allocator_environment'],qualification=bool(args.qualify),resume_implemented=True,
                purpose=p['purpose'],items=p['items'],unique_documents=inventory['unique_documents'])
            if actual_steps==64 or actual_steps==steps or stop:
                if shutil.disk_usage(args.output).free<35*1024**3:raise RuntimeError('Less than35GiB before checkpoint')
                saving=time.monotonic()
                if actual_steps==64:save_development_model(args.output,model,actual_steps,state,rank,xm,device)
                if actual_steps==steps or stop:
                    if args.qualify:
                        save_qualification_slices(args.output,model,optimizer,actual_steps,state,rank,world,xm,device)
                    if not args.qualify or p['qualification_state_policy']=='full_checkpoint':
                        save_checkpoint(args.output,model,optimizer,actual_steps,state,rank,world,xm,device)
                save_walls.append(dict(step=actual_steps,rank0_wall_seconds=time.monotonic()-saving))
            if stop:break
    if rank==0:atomic_json(args.output/'completion.json',dict(status='complete' if actual_steps==steps else 'stopped',
        finished_utc=utc_now(),actual_steps=actual_steps,planned_steps=steps,qualification=bool(args.qualify),
        counters=counters,checkpoint_walls=save_walls,protocol_sha256=protocol_hash,resume_implemented=True,
        recovery_attempt=args.attempt_id if args.resume is not None else None),overwrite=args.resume is not None)
    for hook in hooks:hook.remove()
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','output','stop-file'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=tuple(WEIGHTS),required=True);parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--resume',type=Path);parser.add_argument('--attempt-id')
    parser.add_argument('--deadline-utc',required=True);args=parser.parse_args()
    p=json.loads(args.protocol.read_text());validate_protocol(p)
    if args.qualify != (p['purpose']=='qualification'):
        raise ValueError('Qualification flag must match the separately declared subset purpose')
    if ((args.resume is None and args.output.exists()) or args.stop_file.is_symlink() or (args.stop_file.exists() and not args.stop_file.is_file())
            or require_utc_deadline(args.deadline_utc)<=datetime.now(timezone.utc)):
        raise ValueError('Fresh output, ordinary STOP and future finite deadline required')
    if (os.environ.get('NEURONCORE_NUM_DEVICES')!=str(p['world_size'])
            or os.environ.get('NEURON_RT_CONTIGUOUS_SCRATCHPAD')!='0'
            or any(not os.environ.get(k) for k in ('NEURON_RT_VISIBLE_CORES','NEURON_RT_ROOT_COMM_ID','MASTER_ADDR','MASTER_PORT'))):
        raise ValueError('Explicit qualified allocator/topology/host and Neuron endpoints required')
    if (args.resume is not None and (args.qualify or not args.attempt_id)
            or args.resume is None and args.attempt_id is not None):
        raise ValueError('Recovery requires an original science checkpoint and a fresh attempt ID')
    # Advisory lock survives process exceptions without a stale-file decision.
    # Hold it across CPU preparation and the full distributed worker lifetime.
    import fcntl
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (args.output.parent/f'.{args.output.name}.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args,p)


def execute(args,p):
    rows,tok,assets,inventory=load_inputs(p);refs,native=load_native_reference(p,assets)
    # Construct the shared order once on host, rather than shuffling the entire
    # dataset independently on every worker/component call.
    from train.warmstart_driver import order_index
    order=[order_index(i,len(rows),p['seed']) for i in range(p['steps']*p['global_batch'])]
    if args.resume is not None:
        args.resume_manifest,args.recovery_dir=prepare_resume(args.output,args.resume,args.attempt_id,
            p,args.arm,sha256_file(args.protocol),rows,order)
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker,args=(args,p,rows,tok,assets,inventory,refs,native,sha256_file(args.protocol),order),nprocs=None)


if __name__=='__main__':main()
