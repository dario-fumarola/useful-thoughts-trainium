"""Finite joint calculation warmstart and matched full-source CE control.

Fresh released B0 in each arm. The joint arm accumulates three separately
averaged component losses before ONE ZeRO-owned reduction/clip/optimizer step.
The control receives exactly the same ordered original source chunks. Additional
note/reader labels and computation are treatment, not resource matching.
"""
from __future__ import annotations

import argparse
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
SCHEMA = 'calculation-joint-warmstart-v1'
ARMS = ('joint', 'source_control')
COMPONENTS = ('source_only', 'note_only', 'reader_only')


def components(arm):
    if arm not in ARMS:
        raise ValueError('Unknown calculation learning arm')
    return COMPONENTS if arm == 'joint' else COMPONENTS[:1]


def source_files():
    from train.calculation_boundary_probe import sources
    names = ('train/calculation_warmstart_driver.py', 'train/calculation_warmstart_data.py',
             'train/masked_causal_ce.py',
             'train/warmstart_driver.py', 'train/neuron_driver.py',
             'weekend_thinking/runtime_policy.py')
    return {**sources(), **{name: sha256_file(ROOT/name) for name in names}}


def validate_protocol(p):
    from weekend_thinking.runtime_policy import POLICY_ENVIRONMENT
    expected = dict(schema=SCHEMA, seed=20261018, world_size=8, items=512,
        steps=128, epochs=2, global_batch=8, accumulation=1, precision='fp32',
        lr=1e-5, betas=[.9,.95], eps=1e-8, weight_decay=.1, clip=1.,
        loss_weights={name:1. for name in COMPONENTS})
    if any(isinstance(p.get(k),bool) or p.get(k)!=v for k,v in expected.items()):
        raise ValueError('Require the frozen 512-document/two-epoch/FP32 recipe')
    if any(type(p[k]) is not int for k in ('seed','world_size','items','steps','epochs','global_batch','accumulation')):
        raise ValueError('Integer training counts required')
    policy=p.get('runtime_policy',{})
    if (policy.get('schema')!=1 or set(policy.get('environment',{}))!=set(POLICY_ENVIRONMENT)
            or policy['environment'].get('XLA_COMPILATION_CACHE_SIZE')!='2'):
        raise ValueError('Bind the complete actual LRU2 runtime policy')
    if p.get('source_files')!=source_files():
        raise ValueError('Training and causal packing source closure changed')


def supervision_counts(packed, component):
    note=int(packed['note_close_mask'].sum()); source=int(packed['future_mask'].sum())
    if ((component=='source_only' and (note,source)!=(0,527))
            or (component=='reader_only' and (note,source)!=(0,16))
            or (component=='note_only' and (not 2<=note<=129 or source!=0))
            or component not in COMPONENTS):
        raise ValueError('Declared component and actual supervised labels disagree')
    return note if component=='note_only' else source


def load_inputs(p):
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.calculation_warmstart_data import pack_example
    for key in ('data','dataset'):
        if sha256_file(p[f'{key}_file'])!=p[f'{key}_sha256']:
            raise ValueError(f'Changed {key} bytes')
    assets=asset_provenance(Path(p['model_dir']))
    if assets['model_id']!='Qwen/Qwen3-1.7B-Base' or assets['manifest_sha256']!=p['model_manifest_sha256']:
        raise ValueError('Require the exact released B0 assets')
    rows=[json.loads(s) for s in Path(p['data_file']).read_text().splitlines() if s.strip()]
    manifest=json.loads(Path(p['dataset_file']).read_text()); ids=[r['item_id'] for r in rows]
    if (len(rows)!=512 or len(set(ids))!=512 or len({r['document_id'] for r in rows})!=512
            or any(r['split']!='train' for r in rows) or manifest['split']!='train'
            or manifest['items']!=512 or manifest['examples_sha256']!=p['data_sha256']
            or manifest['item_ids_sha256']!=canonical_hash(ids)):
        raise ValueError('Complete fixed 512 unique TRAIN documents required')
    sealed=json.loads(Path(manifest['prospective_ids_file']).read_text())
    if (sha256_file(manifest['prospective_ids_file'])!=manifest['prospective_ids_sha256']
            or sealed['ordered_item_ids']!=ids):
        raise ValueError('Training rows differ from prospectively sealed IDs/order')
    if any(p['source_files'].get(k)!=v for k,v in manifest['source_files'].items()):
        raise ValueError('Preparation source differs from frozen training closure')
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    counts={name:0 for name in COMPONENTS}
    for row in rows:
        for name in COMPONENTS:
            counts[name]+=supervision_counts(pack_example(row,tok,name),name)
    return rows,tok,assets,counts


def load_native_reference(p,assets):
    """Reuse numerical references only, never scientific scores or answers."""
    from train.calculation_boundary_probe import load,reference
    directory=Path(p['native_reference_output'])
    for name,key in (('protocol.json','native_protocol_sha256'),
                     ('cpu-reference/reference.json','native_reference_sha256'),
                     ('cpu-reference/next-logps.pt','native_logps_sha256')):
        if sha256_file(directory/name)!=p[key]:
            raise ValueError('Bound native numerical reference changed')
    protocol,inputs=load(directory); report,values=reference(directory,protocol,inputs)
    if protocol['assets']!=assets or report['signature']!=p['base_signature']:
        raise ValueError('Numerical reference belongs to different base weights')
    return inputs['qualification_refs'],values


def backward_components(row, tokenizer, arm, backend, sync):
    """No update/zero/reduction here: accumulate all component means once."""
    from train.calculation_warmstart_data import pack_example
    from train.masked_causal_ce import masked_causal_ce
    losses={name:0. for name in COMPONENTS}; counts={name:0 for name in COMPONENTS}; seconds={}
    for name in components(arm):
        started=time.monotonic(); packed=pack_example(row,tokenizer,name)
        counts[name]=supervision_counts(packed,name)
        ce=masked_causal_ce(backend,**packed)
        loss=ce.loss; value=float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError(f'Nonfinite {name} CE before backward')
        loss.backward(); sync(); losses[name]=value
        # Free the finished graph before constructing the next component.
        del loss,ce,packed
        seconds[name]=time.monotonic()-started
    return losses,counts,seconds


def worker(_index,args,p,rows,tok,assets,counts,native_refs,native_values,protocol_hash):
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
    from train.warmstart_driver import order_index,stop_due
    from weekend_thinking.pretrained import load_base,sampled_fingerprint
    from weekend_thinking.runtime_policy import require_runtime_policy,runtime_policy

    configure_runtime()
    if not hasattr(torch,'xla'): torch._register_device_module('xla',torch_xla)
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    device=torch_xla.device(); rank,world=xr.global_ordinal(),xr.world_size()
    if world!=8: raise ValueError('Actual eight-rank topology required')
    dist.init_process_group('xla',init_method='xla://')
    random.seed(p['seed']+rank); np.random.seed(p['seed']+rank); torch.manual_seed(p['seed'])
    model,_=load_base(p['model_dir']); signature=sampled_fingerprint(model)
    if signature!=p['base_signature']: raise ValueError('Actual CPU base fingerprint differs')
    model.to(device); model.tie_weights()
    if (any(t.dtype!=torch.float32 for t in model.parameters())
            or model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or sampled_fingerprint(model)!=signature):
        raise ValueError('Released tied FP32 base changed during device transfer')
    backend=NeuronCausalBackend(model,sync=lambda:torch_xla.sync(wait=True))
    ref_index=rank%len(native_refs); ref=native_refs[ref_index]
    with torch.no_grad():
        ids=torch.full((1,1024),151643,dtype=torch.long)
        ids[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
        hidden=backend.hidden(ids.to(device))
        if hidden.dtype!=torch.float32:
            raise RuntimeError('Actual FP32 trunk required by this initial pilot')
        actual=model.lm_head(hidden[:,len(ref['context_ids'])-1]).float().log_softmax(-1)
        torch_xla.sync(wait=True); actual=actual.cpu()
        error=float((actual-native_values[ref_index:ref_index+1]).abs().max())
        if actual.dtype!=torch.float32 or not bool(torch.isfinite(actual).all()) or not math.isfinite(error) or error>.002:
            raise RuntimeError('Actual eight-rank native CPU/device full-head gate failed')
        del hidden,ids
    optimizer=ZeroRedundancyOptimizer(model.parameters(),AdamW,optimizer_dtype=torch.float32,
        grad_clipping=True,max_norm=p['clip'],save_master_weights=True,lr=p['lr'],
        betas=tuple(p['betas']),eps=p['eps'],weight_decay=p['weight_decay'])
    found_inf=torch.zeros((),device=device)
    order=[order_index(i,len(rows),p['seed']) for i in range(p['steps']*world)]
    order_hash=canonical_hash([rows[i]['item_id'] for i in order]); spec_hash=canonical_hash(p)
    steps=2 if args.qualify else p['steps']; deadline=require_utc_deadline(args.deadline_utc)
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'run.json',dict(schema=SCHEMA,run_id=args.output.name,started_utc=utc_now(),
            arm=args.arm,seed=p['seed'],protocol_sha256=protocol_hash,spec_hash=spec_hash,
            order_hash=order_hash,world_size=world,planned_steps=steps,scientific_steps=p['steps'],
            qualification=bool(args.qualify),assets=assets,base_signature=signature,
            runtime_policy=runtime_policy(),component_order=list(components(args.arm)),
            one_epoch_component_labels={k:counts[k] for k in components(args.arm)},
            source_tokens_definition='Original source input tokens528 per document; source CE supervises527. Reader and note labels are additional.',
            limitations=['Extra treatment labels/gradient dose/compute are not matched.',
                'Two-epoch interface pilot, not broad reasoning pretraining or spontaneous emergence.',
                'No exact resume is implemented in this finite driver; partial checkpoints preserve state only.']))
    xm.rendezvous('calculation-output-ready'); torch_xla.sync(wait=True)
    gate_path=args.output/f'forward-rank-{rank:05d}.pt'; torch.save(actual,gate_path)
    atomic_json(args.output/f'qualification-rank-{rank:05d}.json',dict(rank=rank,world_size=world,
        native_reference_index=ref_index,max_full_head_logp_error=error,atol=.002,
        values_sha256=sha256_file(gate_path),base_signature=signature,tied=True,precision='fp32',
        protocol_sha256=protocol_hash,created_utc=utc_now()))
    del actual
    xm.set_rng_state(p['seed']+100003*rank,device)
    counters=dict(source_records=0,source_tokens=0,sampled_actions=0,source_labels=0,note_labels=0,reader_labels=0)
    def synchronized_stop():
        value=torch.tensor(int(stop_due(args.stop_file,deadline)),device=device)
        return bool(xm.all_reduce(xm.REDUCE_MAX,value).cpu())
    def sample_parameters():
        return torch.cat([t.detach().reshape(-1)[:32] for t in model.parameters()]).cpu()
    actual_steps=0
    if not synchronized_stop():
        for step in range(steps):
            started=time.monotonic(); optimizer.zero_grad(set_to_none=True); before=sample_parameters()
            row=rows[order[step*world+rank]]
            losses,label_counts,seconds=backward_components(row,tok,args.arm,backend,lambda:torch_xla.sync(wait=True))
            finite=torch.stack([torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None]).all()
            if bool(xm.all_reduce(xm.REDUCE_MAX,(~finite).float()).cpu()):
                raise RuntimeError('Nonfinite accumulated gradient before sole ZeRO step')
            optimizer_start=time.monotonic(); optimizer.step(found_inf=found_inf); torch_xla.sync(wait=True)
            optimizer_seconds=time.monotonic()-optimizer_start
            norm=float(optimizer.grad_norm.cpu()); after=sample_parameters(); delta=after-before
            replicas=xm.all_gather(after.to(device)).reshape(world,-1).cpu()
            if (not math.isfinite(norm) or norm<=0 or not bool(torch.isfinite(after).all())
                    or not bool(delta.ne(0).any()) or not torch.equal(replicas,replicas[:1].expand_as(replicas))):
                raise RuntimeError('Nonfinite/unchanged sampled weights or replica mismatch')
            totals=xm.all_reduce(xm.REDUCE_SUM,torch.tensor(
                [losses[k] for k in COMPONENTS]+[label_counts[k] for k in COMPONENTS],device=device)).cpu().tolist()
            # Timings are summed rank-seconds; elapsed_seconds is rank0 wall time.
            times=xm.all_reduce(xm.REDUCE_SUM,torch.tensor(
                [seconds.get(k,0.) for k in COMPONENTS]+[optimizer_seconds],device=device)).cpu().tolist()
            counters['source_records']+=world; counters['source_tokens']+=528*world
            for key,value in zip(('source_labels','note_labels','reader_labels'),totals[3:]): counters[key]+=int(value)
            actual_steps=step+1
            metric=dict(step=actual_steps,time_utc=utc_now(),arm=args.arm,elapsed_seconds=time.monotonic()-started,
                component_losses={k:v/world for k,v in zip(COMPONENTS,totals[:3])},
                component_labels={k:int(v) for k,v in zip(COMPONENTS,totals[3:])},
                executed_trunk_token_slots={k:world*width for k,width in
                    (('source_only',543),('note_only',831),('reader_only',1023)) if k in components(args.arm)},
                component_rank_seconds=dict(zip((*COMPONENTS,'optimizer'),times)),gradient_norm=norm,
                sampled_update_l2=float(delta.norm()),sampled_changed_fraction=float(delta.ne(0).float().mean()),
                sampled_replicas_bitwise_equal=True,counters=counters.copy(),
                complete_ordered_passes=counters['source_records']//len(rows),
                next_pass_records=counters['source_records']%len(rows),
                ordered_item_ids=[rows[i]['item_id'] for i in order[step*world:(step+1)*world]])
            if rank==0:
                with (args.output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metric,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(metric),flush=True)
            stop=synchronized_stop()
            if actual_steps in (16,steps) or stop:
                if shutil.disk_usage(args.output).free<35*1024**3:
                    raise RuntimeError('Less than35GiB free before complete checkpoint')
                save_checkpoint(args.output,model,optimizer,actual_steps,
                    dict(spec_hash=spec_hash,order_hash=order_hash,arm=args.arm,seed=p['seed'],global_batch=world,
                        accumulation=1,protocol_sha256=protocol_hash,data_sha256=p['data_sha256'],
                        dataset_sha256=p['dataset_sha256'],objective='joint_note_reader_fullsource' if args.arm=='joint' else 'fullsource_ce',
                        counters=counters.copy(),parameter_dtype='fp32',compute_dtype='fp32',runtime_policy=runtime_policy(),
                        qualification=bool(args.qualify),stopped_early=stop and actual_steps<steps),rank,world,xm,device)
            if stop: break
    if rank==0:
        atomic_json(args.output/'completion.json',dict(finished_utc=utc_now(),arm=args.arm,
            actual_steps=actual_steps,planned_steps=steps,qualification=bool(args.qualify),
            status='complete' if actual_steps==steps else 'stopped',counters=counters,
            protocol_sha256=protocol_hash))
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True); parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--deadline-utc',required=True); parser.add_argument('--stop-file',type=Path,required=True)
    args=parser.parse_args(); p=json.loads(args.protocol.read_text()); validate_protocol(p)
    if (args.output.exists() or args.stop_file.is_symlink()
            or (args.stop_file.exists() and not args.stop_file.is_file())
            or require_utc_deadline(args.deadline_utc)<=datetime.now(timezone.utc)):
        raise ValueError('Require fresh output, ordinary STOP and future finite deadline')
    if (os.environ.get('NEURONCORE_NUM_DEVICES')!='8'
            or os.environ.get('NEURON_RT_CONTIGUOUS_SCRATCHPAD')!='0'
            or any(not os.environ.get(k) for k in ('NEURON_RT_VISIBLE_CORES','NEURON_RT_ROOT_COMM_ID','MASTER_ADDR','MASTER_PORT'))):
        raise ValueError('Require eight ranks, qualified noncontiguous allocator and dedicated explicit endpoints')
    rows,tok,assets,counts=load_inputs(p)
    native_refs,native_values=load_native_reference(p,assets)
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker,args=(args,p,rows,tok,assets,counts,native_refs,native_values,sha256_file(args.protocol)),nprocs=None)


if __name__=='__main__': main()
