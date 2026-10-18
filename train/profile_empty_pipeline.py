"""One-core diagnostic of fixed replay cost; NO optimizer or policy training.

Each trace is forced to native EOS on its first action. This is instrumentation,
not a sample from B0. The old padded decode loop, reader and signed replay run
unchanged. Same active action at three static caps isolates padding work. BF16
autocast keeps parameters/gradients FP32. This does not qualify a complete RL step.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from weekend_thinking.artifacts import atomic_json,require_utc_deadline,sha256_file,utc_now

CAPS=(1024,256,32)
PRECISIONS=('fp32','bf16')
REPEATS=3


def forced_eos(logits,temperature,banned_ids):
    from weekend_thinking.core import policy_log_probs
    lp=policy_log_probs(logits,temperature,banned_ids)
    actions=torch.full((logits.shape[0],),151643,device=logits.device,dtype=torch.long)
    return actions,lp[:,151643]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-output',type=Path,required=True)
    parser.add_argument('--format',type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    args=parser.parse_args(); deadline=require_utc_deadline(args.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite profiler deadline; no automatic retry')
    check()
    if args.output.exists():raise FileExistsError(args.output)
    from train.calculation_boundary_probe import load,reference,qualified_reader
    from train.neuron_backend import NeuronCausalBackend
    from weekend_thinking.core import Format
    from weekend_thinking.pretrained import sampled_fingerprint
    from weekend_thinking.runtime_policy import runtime_policy
    protocol,inputs=load(args.reference_output);native_report,native=reference(args.reference_output,protocol,inputs)
    fmt=Format.from_dict(json.loads(args.format.read_text())['format'])
    args.output.mkdir(parents=True)
    report=dict(schema='forced-empty-pipeline-profile-v1',status='started',started_utc=utc_now(),
        deadline_utc=args.deadline_utc,reference_protocol_sha256=sha256_file(args.reference_output/'protocol.json'),
        source_sha256=sha256_file(__file__),format_sha256=sha256_file(args.format),
        caps=list(CAPS),precisions=list(PRECISIONS),repeats=REPEATS,group_size=4,
        runtime_policy=runtime_policy(),limitations=[
            'Forced EOS, not B0 behavior: real model prefill/decode and logp support, without Gumbel draws.',
            'One real TRAIN prefix, no optimizer, collectives or parameter updates; not a complete RL step or throughput gate.',
            'The first repetition can include compilation; later timings retain executable reload costs.',
            'Traces/masks are materialized and retransferred as device inputs outside timers, to avoid constant-folding the forced trace.',
            'Stored parameter/gradient signatures sample slices, not full tensors.'])
    atomic_json(args.output/'run.json',report)
    try:
        import torch_xla
        if not hasattr(torch,'xla'):
            torch._register_device_module('xla',torch_xla)
        # This uses only the already-completed experiment's numerical reference,
        # released assets and known inputs; no scientific score/answer rows open.
        with torch.no_grad():
            backend,tok,_=qualified_reader(args.reference_output,protocol,inputs,native_report,native,0,args.output,report)
        model=backend.model;before=sampled_fingerprint(model)
        source=next(r for r in inputs['rows'] if r['split']=='train')
        prefix=torch.tensor([source['original_prefix_ids']],device=backend.device)
        future=torch.tensor([source['original_future_ids']],device=backend.device)
        results=[]
        for precision in PRECISIONS:
            if precision=='bf16':
                import torch_xla.amp
                amp=lambda:torch_xla.amp.autocast(backend.device,dtype=torch.bfloat16)
            else:amp=nullcontext
            work=NeuronCausalBackend(model,amp=amp,head_amp=amp,sync=backend.sync,
                banned_ids=backend.banned_ids,checkpoint_trunk=True)
            for cap in CAPS:
                for repeat in range(REPEATS):
                    check(); model.zero_grad(set_to_none=True)
                    record=dict(precision=precision,cap=cap,repeat=repeat,started_utc=utc_now(),stage_seconds={})
                    def timed(name,fn):
                        check();work.sync();start=time.monotonic()
                        print(json.dumps(dict(utc=utc_now(),precision=precision,cap=cap,repeat=repeat,stage=name,event='start')),flush=True)
                        stage=dict(precision=precision,cap=cap,repeat=repeat,stage=name,started_utc=utc_now())
                        try:
                            result=fn();work.sync();stage['status']='complete'
                        except BaseException as error:
                            stage.update(status='failed',error=f'{type(error).__name__}: {error}')
                            raise
                        finally:
                            stage.update(wall_seconds=time.monotonic()-start,finished_utc=utc_now())
                            record['stage_seconds'][name]=stage['wall_seconds']
                            with (args.output/'stages.jsonl').open('a') as stream:
                                stream.write(json.dumps(stage,allow_nan=False)+'\n');stream.flush()
                        return result
                    with patch('train.neuron_backend.sample_actions',forced_eos):
                        rollout=timed('forced_empty_cached_decode',lambda:work.sample(prefix,fmt=fmt,group_size=4,
                            max_actions=cap,temperature=.7,policy_version=0))
                    actions=rollout.actions.cpu(); masks=rollout.action_mask.cpu()
                    if not torch.equal(masks.sum(-1),torch.ones((1,4),dtype=torch.long)) or not actions[:,:,0].eq(151643).all():
                        raise RuntimeError('Forced-empty trace was not exactly one active EOS per note')
                    rollout.actions=actions.clone().to(work.device)
                    rollout.action_mask=masks.clone().to(work.device)
                    work.sync()
                    clean=timed('clean16',lambda:work.clean_score(prefix,future,horizon=16)).cpu()
                    thought=timed('reader16',lambda:work.score(prefix,future,rollout,fmt=fmt,horizon=16)).cpu()
                    losses=[]
                    for g in range(4):
                        def backward():
                            loss=work.replay_one_loss(prefix,rollout.actions[:,g],rollout.action_mask[:,g],
                                torch.tensor(1.,device=work.device),fmt=fmt,temperature=.7,divisor=4)
                            value=float(loss.detach().cpu())
                            if not math.isfinite(value):raise RuntimeError('Nonfinite replay loss')
                            loss.backward();return value
                        losses.append(timed(f'replay_backward_{g}',backward))
                    sampled=torch.cat([p.grad.detach().reshape(-1)[:32] for p in model.parameters() if p.grad is not None]).cpu()
                    finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                    if not bool(finite.cpu()):raise RuntimeError('Nonfinite full gradient')
                    if not torch.isfinite(sampled).all():raise RuntimeError('Nonfinite sampled gradient')
                    if any(p.dtype!=torch.float32 or (p.grad is not None and p.grad.dtype!=torch.float32) for p in model.parameters()):
                        raise RuntimeError('FP32 parameter/gradient storage lost')
                    gradient_file=args.output/f'gradient-{precision}-{cap}-{repeat}.pt';torch.save(sampled,gradient_file)
                    record.update(active_actions=int(masks.sum()),decode_iterations=rollout.decode_iterations,
                        clean16=clean.tolist(),thought16=thought.tolist(),loss=sum(losses),
                        gradient_slice_l2=float(sampled.norm()),gradient_slice_sha256=sha256_file(gradient_file),
                        sum_stage_seconds=sum(record['stage_seconds'].values()),finished_utc=utc_now())
                    results.append(record)
                    with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps(record,allow_nan=False),flush=True)
                    del rollout,sampled
        model.zero_grad(set_to_none=True);backend.sync();after=sampled_fingerprint(model)
        if before!=after:raise RuntimeError('Profiler changed model weights')
        report.update(status='complete',rows=len(results),parameter_signature_before=before,parameter_signature_after=after,
            metrics_sha256=sha256_file(args.output/'metrics.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['finished_utc']=utc_now();atomic_json(args.output/'run.json',report,overwrite=True)


if __name__=='__main__':main()
