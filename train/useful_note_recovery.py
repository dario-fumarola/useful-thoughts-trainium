"""Strict full-state recovery for the BF16-model/FP32-master SFT path.

The caller owns protocol/source and COMPLETE-file integrity checks. This module
checks the actual loaded payload and verifies the resulting model and optimizer
state before resetting RNG. A model-only milestone is never resumable.
"""
from __future__ import annotations

import hashlib
import math

from weekend_thinking.artifacts import canonical_hash


def tensor_digest(value):
    """Hash actual CPU bytes, including BF16 without converting its values."""
    import torch
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
    h.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def optimizer_digest(state):
    """All masters, moments and Adam counters, with no sampled-only claim."""
    masters=state['sharded_master_weights'];moments=state['base_state']
    return dict(param_groups=canonical_hash(state['param_groups']),
        tensors={str(i):dict(master=tensor_digest(master),
            **{key:tensor_digest(moments[i][key]) for key in ('exp_avg','exp_avg_sq','step')})
            for i,master in masters.items()})


def validate_payload(manifest,weights,saved,model,*,protocol,protocol_sha256,
                     arm,rank,run_id,order_hash,expected_lr):
    """Reject mixed runs/arms/steps/topologies or rounded/missing masters.

    The producer protocol can be an explicitly qualified earlier schema when a
    diagnostic caller supplies its exact identity. Production resumes must use
    their original immutable protocol and source closure.
    """
    import torch
    p=protocol;step=manifest.get('step');world=p['world_size'];spec=canonical_hash(p)
    if (manifest.get('schema')!='neuron-v1' or manifest.get('run_id')!=run_id
            or manifest.get('world_size')!=world or manifest.get('spec_hash')!=spec
            or type(step)is not int or not 1<=step<=p['steps'] or not 0<=rank<world
            or set(manifest.get('files',{}))!={'model.pt',*[f'rank-{r:05d}.pt' for r in range(world)]}):
        raise ValueError('Complete original full-state run/spec/topology required')
    if any(weights.get(k)!=v for k,v in dict(step=step,run_id=run_id,spec_hash=spec).items()):
        raise ValueError('Model belongs to a different checkpoint')
    identity=dict(spec_hash=spec,order_hash=order_hash,arm=arm,seed=p['seed'],
        global_batch=p['global_batch'],accumulation=p['accumulation'],
        protocol_sha256=protocol_sha256,data_sha256=p['data_sha256'],dataset_sha256=p['dataset_sha256'],
        parameter_dtype='bf16',gradient_dtype='bf16',optimizer_dtype='fp32',compute_dtype='bf16',
        precision_contract=p['precision_contract'],runtime_policy=p['runtime_policy'],
        allocator_environment=p['allocator_environment'],purpose=p['purpose'],items=p['items'])
    if any(saved.get(k)!=v or manifest.get('metadata',{}).get(k)!=v for k,v in identity.items()):
        raise ValueError('Checkpoint arm/order/data/precision/runtime identity mismatch')
    if saved.get('step')!=step or saved.get('rank')!=rank or saved.get('world_size')!=world:
        raise ValueError('Checkpoint shard step/rank/topology mismatch')
    counters=saved.get('counters',{})
    if (counters!=manifest['metadata'].get('counters')
            or counters.get('source_records')!=step*p['global_batch']
            or counters.get('source_tokens')!=528*step*p['global_batch']
            or counters.get('sampled_actions')!=0
            or set(counters.get('labels',{}))!={'source','writer','reader','augmented','continuation'}
            or any(type(v)is not int or v<0 for v in counters['labels'].values())):
        raise ValueError('Checkpoint counters disagree with completed updates')
    if (any(t.dtype!=torch.bfloat16 for t in model.parameters())
            or any(t.is_floating_point() and t.dtype!=torch.float32 for t in model.buffers())):
        raise ValueError('Actual BF16 parameters and original FP32 buffers required')
    expected=model.state_dict();actual=weights.get('model',{})
    if set(expected)!=set(actual):raise ValueError('Full active model tensor inventory required')
    for name,value in actual.items():
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=expected[name].dtype or value.shape!=expected[name].shape
                or not torch.isfinite(value).all()):
            raise ValueError('Saved active-model dtype/shape/finite state differs')
    if (model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or not torch.equal(actual['model.embed_tokens.weight'].view(torch.uint8),actual['lm_head.weight'].view(torch.uint8))):
        raise ValueError('Saved tied active BF16 model values disagree')
    opt=saved.get('optimizer',{});parameters=list(model.parameters())
    ids=list(range(len(parameters)));masters=opt.get('sharded_master_weights',{});moments=opt.get('base_state',{})
    if list(masters)!=ids or set(moments)!=set(ids):
        raise ValueError('Complete ordered FP32 masters and moments required')
    groups=opt.get('param_groups',[])
    if len(groups)!=1 or groups[0].get('params')!=ids:
        raise ValueError('Original single optimizer parameter group required')
    group=groups[0]
    if (not math.isfinite(expected_lr) or expected_lr<=0 or group.get('lr')!=expected_lr
            or tuple(group.get('betas',()))!=tuple(p['betas']) or group.get('eps')!=p['eps']
            or group.get('weight_decay')!=p['weight_decay']):
        raise ValueError('Saved optimizer hyperparameters/LR differ')
    for i,param in enumerate(parameters):
        shape=((param.shape[0]+world-1)//world,*param.shape[1:]);master=masters[i];state=moments[i]
        if (not isinstance(master,torch.Tensor) or master.device.type!='cpu'
                or master.dtype!=torch.float32 or tuple(master.shape)!=shape
                or not torch.isfinite(master).all() or set(state)!={'exp_avg','exp_avg_sq','step'}):
            raise ValueError('Full FP32 master shard inventory/values invalid')
        for name in ('exp_avg','exp_avg_sq'):
            value=state[name]
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float32
                    or tuple(value.shape)!=shape or not torch.isfinite(value).all()):
                raise ValueError('Full FP32 Adam moment inventory/values invalid')
        value=state['step']
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=torch.float32 or value.numel()!=1 or not torch.isfinite(value).all() or float(value)!=step
                or (state['exp_avg_sq']<0).any()):
            raise ValueError('Adam counter or second moment invalid')
    rng=saved.get('rng',{})
    if set(rng)!={'cpu','device'} or type(rng['device'])is not int or set(rng['cpu'])!={'python','torch','numpy'}:
        raise ValueError('Saved CPU and device RNG states required')
    # Validate without changing the caller's live random streams.
    import random
    import numpy as np
    try:
        random.Random().setstate(rng['cpu']['python'])
        torch.Generator().set_state(rng['cpu']['torch'])
        n=rng['cpu']['numpy']
        np.random.RandomState().set_state((n['algorithm'],n['keys'].numpy().astype(np.uint32),
            n['position'],n['has_gauss'],n['cached_gaussian']))
    except (KeyError,TypeError,ValueError,RuntimeError,AttributeError) as error:
        raise ValueError('Invalid saved CPU RNG payload') from error
    return step


def restore_and_verify(model,optimizer,weights,saved,*,xm,device,sync):
    """Use the supported ZeRO loader, then compare all actual restored bytes.

    Observations occur BEFORE the final RNG reset. No extra synchronization may
    be inserted after this function and before the ordinary continuation.
    """
    from train.runtime_compat import restore_syncfree_device_state
    from train.neuron_driver import restore_rng_after_sync
    expected_optimizer=optimizer_digest(saved['optimizer'])
    model.load_state_dict(weights['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restore_syncfree_device_state(optimizer)
    sync(wait=True)
    actual=model.state_dict()
    if (set(actual)!=set(weights['model'])
            or any(tensor_digest(actual[k])!=tensor_digest(v) for k,v in weights['model'].items())):
        raise RuntimeError('Actual restored BF16 model differs from saved active model')
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError('Restored embedding/head Parameter tie lost')
    actual_optimizer=optimizer_digest(optimizer.state_dict())
    if actual_optimizer!=expected_optimizer:
        raise RuntimeError('Actual restored FP32 masters/moments/counters differ')
    report=dict(full_active_model_bytes_equal=True,full_optimizer_bytes_equal=True,
        tied_parameter_objects_preserved=True,optimizer_digest=actual_optimizer,
        scope='Actual state readback before RNG reset; next-update/uninterrupted parity requires separate qualification.')
    restore_rng_after_sync(saved['rng'],xm,device,sync)
    return report
