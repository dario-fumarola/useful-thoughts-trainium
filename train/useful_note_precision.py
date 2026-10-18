"""BF16 parameter storage, FP32 RoPE/master contract and native reference.

No whole-module dtype conversion: Qwen's nonpersistent rotary buffers must
retain their original FP32 values. The full released head is used here,
including tokenizer-undefined rows. This is a forward-distribution check;
tail CE and gradient fidelity are separate questions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

import torch

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = 'useful-note-bf16-storage-native-v1'


def contract():
    return dict(schema=SCHEMA, parameter_dtype='bf16', gradient_dtype='bf16',
        optimizer_dtype='fp32', floating_buffer_dtype='fp32', compute_dtype='bf16',
        reduce_scatter_input_dtype='bf16', higher_cc_precision=False, use_grad_acc_hook=False,
        runtime_collective_accumulation_precision='not_established_by_Python_contract',
        autocast=False, support='full_released_head', log_softmax_dtype='fp32',
        normalization='independent_FP64_logsumexp', max_bidirectional_kl_nats=.01,
        diagnostics_only=['max_logp_error','rms_logp_error','relative_l2_logp_error','argmax_equal'])


def sources():
    root=Path(__file__).resolve().parents[1]
    names=('train/useful_note_precision.py','train/neuron_backend.py',
           'weekend_thinking/pretrained.py','weekend_thinking/artifacts.py')
    return {name:sha256_file(root/name) for name in names}


def released_bf16_header(model_dir):
    """Check every released tensor's dtype without loading another payload."""
    paths=sorted(Path(model_dir).glob('*.safetensors'))
    if not paths: raise ValueError('Released safetensors required')
    count=0;parameters=0;headers={}
    for path in paths:
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if not 0 < length < 16*1024**2: raise ValueError('Invalid safetensors header')
            header=json.loads(f.read(length))
        tensors=[value for name,value in header.items() if name!='__metadata__']
        if any(value['dtype']!='BF16' for value in tensors):
            raise ValueError('Exact standard master initialization requires all released weights BF16')
        import math
        count+=len(tensors);parameters+=sum(math.prod(t['shape']) for t in tensors)
        headers[path.name]=dict(header_bytes=length,tensors=len(tensors))
    return dict(dtype='BF16',tensors=count,parameters=parameters,headers=headers)


def cast_parameters(model):
    """CPU only; preserve tied Parameter objects and byte-exact FP32 buffers."""
    parameters=list(model.parameters());identities=[id(p) for p in parameters]
    if not parameters or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in parameters):
        raise ValueError('Cast released CPU FP32 parameters before any device transfer')
    buffers={name:t.detach().clone() for name,t in model.named_buffers()}
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in buffers.values()):
        raise ValueError('Original floating buffers must remain FP32')
    with torch.no_grad():
        for p in parameters: p.data=p.data.to(torch.bfloat16)
    if [id(p) for p in model.parameters()]!=identities:
        raise RuntimeError('Parameter identity changed during cast')
    for name,t in model.named_buffers():
        old=buffers[name]
        if t.dtype!=old.dtype or not torch.equal(t.contiguous().view(torch.uint8),old.contiguous().view(torch.uint8)):
            raise RuntimeError('CPU parameter cast altered a buffer')
    return dict(parameters=len(parameters),parameter_dtype='torch.bfloat16',
        buffers={name:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=canonical_hash(t.reshape(-1).tolist()))
                 for name,t in buffers.items()},parameter_objects_preserved=True)


def compare_distributions(reference,actual):
    """Each row independently gates BOTH KL directions after FP64 normalization."""
    if reference.shape!=actual.shape or reference.ndim!=2 or reference.shape[1]<2:
        raise ValueError('Aligned complete [references,vocabulary] LP vectors required')
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise RuntimeError('Nonfinite full-head native vector')
    r=reference.detach().cpu().double();a=actual.detach().cpu().double()
    r=r-r.logsumexp(-1,keepdim=True);a=a-a.logsumexp(-1,keepdim=True)
    difference=a-r
    forward=(r.exp()*(r-a)).sum(-1);reverse=(a.exp()*(a-r)).sum(-1)
    # Tiny negative roundoff is reported as zero; a gross negative is invalid.
    if (forward < -1e-12).any() or (reverse < -1e-12).any(): raise RuntimeError('Invalid KL reduction')
    rows=[]
    for i in range(len(r)):
        f=max(0.,float(forward[i]));b=max(0.,float(reverse[i]))
        rows.append(dict(reference_to_device_kl_nats=f,device_to_reference_kl_nats=b,
            max_bidirectional_kl_nats=max(f,b),passed=max(f,b)<=.01,
            max_logp_error=float(difference[i].abs().max()),rms_logp_error=float(difference[i].square().mean().sqrt()),
            relative_l2_logp_error=float(difference[i].norm()/r[i].norm()),
            argmax_equal=int(r[i].argmax())==int(a[i].argmax())))
    return rows


def shard_expectations(model,rank,world):
    """Small exact initial CPU samples; mirror documented dim0 padding only."""
    if not 0<=rank<world: raise ValueError('Invalid shard rank')
    entries=[];values=[]
    for name,p in model.named_parameters():
        if p.device.type!='cpu' or p.dtype!=torch.bfloat16 or p.ndim<1:
            raise ValueError('CPU BF16 parameter inventory required')
        width=(p.shape[0]+world-1)//world;start=rank*width
        shape=[width,*p.shape[1:]];count=min(32,width*p[0].numel())
        flat=p.detach()[start:min(start+width,p.shape[0])].reshape(-1)
        sample=torch.zeros(count,dtype=torch.float32)
        sample[:min(count,len(flat))]=flat[:count].float()
        entries.append(dict(name=name,shape=shape,values=count,start_row=start));values.append(sample)
    return dict(entries=entries,values=torch.cat(values))


def master_samples(optimizer,expected):
    masters=optimizer.state_dict().get('sharded_master_weights',{})
    if list(masters)!=list(range(len(expected['entries']))):
        raise RuntimeError('Standard ZeRO master order/inventory differs')
    samples=[]
    for i,row in enumerate(expected['entries']):
        t=masters[i]
        if t.dtype!=torch.float32 or list(t.shape)!=row['shape']:
            raise RuntimeError('FP32 master shard shape/dtype differs from CPU parameter inventory')
        samples.append(t.detach().reshape(-1)[:row['values']])
    return torch.cat(samples).cpu()


def active_shard_samples(model,expected):
    """Corresponding active-model samples, including ZeRO's zero padding."""
    parameters=dict(model.named_parameters());samples=[]
    for row in expected['entries']:
        p=parameters[row['name']];start=row['start_row'];width=row['shape'][0]
        value=p.detach()[start:min(start+width,p.shape[0])].reshape(-1)[:row['values']]
        sample=torch.cat((value,torch.zeros(row['values']-value.numel(),device=p.device,dtype=p.dtype)))
        samples.append(sample)
    return torch.cat(samples).cpu()


def cpu_reference(contexts_file,model_dir,output,deadline,cpu_threads=16):
    from train.evaluate import asset_provenance
    from train.neuron_backend import NeuronCausalBackend
    from weekend_thinking.pretrained import load_base,sampled_fingerprint
    output=Path(output)
    if not 1<=cpu_threads<=16 or output.exists():raise ValueError('Fresh output and <=16 CPU threads required')
    def check():
        if datetime.now(timezone.utc)>=require_utc_deadline(deadline):raise TimeoutError('Native reference deadline')
    check();torch.set_num_threads(cpu_threads)
    refs=json.loads(Path(contexts_file).read_text())['qualification_refs']
    if len(refs)!=4 or any(not 1<=len(r['context_ids'])<=1024 for r in refs):
        raise ValueError('Four prospectively fixed causal native contexts required')
    assets=asset_provenance(Path(model_dir));header=released_bf16_header(model_dir)
    model,_=load_base(model_dir);signature=sampled_fingerprint(model);conversion=cast_parameters(model)
    if sampled_fingerprint(model)!=signature or model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError('BF16 initial values/tie differ from released B0')
    model.eval();backend=NeuronCausalBackend(model,checkpoint_trunk=False);values=[];dtypes=[]
    with torch.no_grad():
        for ref in refs:
            check();ids=torch.full((1,1024),151643,dtype=torch.long)
            ids[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
            hidden=backend.hidden(ids);logits=model.lm_head(hidden[:,len(ref['context_ids'])-1]);lp=logits.float().log_softmax(-1)
            if hidden.dtype!=torch.bfloat16 or logits.dtype!=torch.bfloat16 or not torch.isfinite(lp).all():
                raise RuntimeError('Canonical native BF16 compute failed')
            values.append(lp);dtypes.append(dict(hidden=str(hidden.dtype),head=str(logits.dtype),logps=str(lp.dtype)))
    check();output.mkdir(parents=True)
    torch.save(torch.cat(values),output/'next-logps.pt')
    report=dict(schema=SCHEMA,status='complete_CPU_BF16_parameter_FP32_RoPE_reference',created_utc=utc_now(),
        precision=contract(),source_files=sources(),assets=assets,base_signature=signature,released_header=header,
        conversion=conversion,contexts_file=str(Path(contexts_file).resolve()),contexts_sha256=sha256_file(contexts_file),
        qualification_refs=refs,width=1024,dtypes=dtypes,cpu_threads=cpu_threads,
        next_logps_sha256=sha256_file(output/'next-logps.pt'),limitations='Forward-distribution check, not tail CE or gradient equivalence.')
    atomic_json(output/'reference.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('contexts','model-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--cpu-threads',type=int,default=16)
    a=p.parse_args();cpu_reference(a.contexts,a.model_dir,a.output,a.deadline_utc,a.cpu_threads)


if __name__=='__main__':main()
