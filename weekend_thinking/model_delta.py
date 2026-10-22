"""Lossless BF16 tensor deltas; no quantization or optimizer-state claim."""
from __future__ import annotations
from collections import OrderedDict
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def bits(tensor):
    if tensor.dtype!=torch.bfloat16 or tensor.device.type!='cpu':
        raise ValueError('Exact CPU BF16 tensors required; never silently cast model weights')
    return tensor.detach().contiguous().view(torch.uint16).numpy()


def tensor_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def encode(base,trained,path:Path,*,provenance):
    if set(base)!=set(trained) or not base:raise ValueError('Identical complete model key sets required')
    if path.exists():raise FileExistsError(path)
    entries=[];arrays={};seen={}
    for name in sorted(base):
        old,new=bits(base[name]),bits(trained[name])
        if old.shape!=new.shape:raise ValueError('Model tensor shape differs: '+name)
        old_hash,new_hash=tensor_hash(old),tensor_hash(new)
        entry={'name':name,'shape':list(old.shape),'base_tensor_sha256':old_hash,'model_tensor_sha256':new_hash}
        identity=(tuple(old.shape),old_hash,new_hash)
        if identity in seen:entry['alias_of']=seen[identity]
        else:
            key=f'tensor_{len(arrays):04d}';entry['array']=key;arrays[key]=np.bitwise_xor(old,new);seen[identity]=name
        entries.append(entry)
    manifest={'schema':'lossless-bf16-xor-model-v1','entries':entries,'provenance':provenance,'qualification':'Exact tensor bits relative to the required base; not quantized weights or an optimizer checkpoint.'}
    arrays['manifest']=np.frombuffer(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    with path.open('xb') as output:np.savez_compressed(output,**arrays)
    return manifest


def decode(base,path:Path):
    restored=OrderedDict()
    with np.load(path,allow_pickle=False) as archive:
        manifest=json.loads(archive['manifest'].tobytes())
        if manifest.get('schema')!='lossless-bf16-xor-model-v1':raise ValueError('Unsupported model-delta schema')
        entries=manifest['entries'];names=[v['name'] for v in entries]
        if len(set(names))!=len(names) or set(names)!=set(base):raise ValueError('Complete unique model tensor set required')
        expected={'manifest'}|{v['array'] for v in entries if 'array' in v}
        if set(archive.files)!=expected:raise ValueError('Unexpected or missing delta arrays')
        for entry in entries:
            name=entry['name'];old=bits(base[name])
            if list(old.shape)!=entry['shape'] or tensor_hash(old)!=entry['base_tensor_sha256']:raise ValueError('Wrong base tensor: '+name)
            if ('array' in entry)==('alias_of' in entry):raise ValueError('A tensor requires exactly one payload or preceding alias')
            if 'alias_of' in entry:
                if entry['alias_of'] not in restored:raise ValueError('Alias must name a preceding restored tensor')
                new=bits(restored[entry['alias_of']])
            else:
                delta=archive[entry['array']]
                if delta.dtype!=np.uint16 or delta.shape!=old.shape:raise ValueError('Invalid delta shape or dtype')
                new=np.bitwise_xor(old,delta)
            if list(new.shape)!=entry['shape'] or tensor_hash(new)!=entry['model_tensor_sha256']:raise ValueError('Restored model tensor differs: '+name)
            restored[name]=(restored[entry['alias_of']] if 'alias_of' in entry else torch.from_numpy(np.asarray(new)).view(torch.bfloat16))
    return restored,manifest


def qwen_base(directory:Path):
    """The pinned local Qwen3-1.7B base, including its tied output alias."""
    from safetensors.torch import load_file
    directory=Path(directory);config=json.loads((directory/'config.json').read_text())
    if config.get('model_type')!='qwen3' or not config.get('tie_word_embeddings'):raise ValueError('This wrapper requires the qualified tied Qwen3 base')
    state=load_file(str(directory/'model.safetensors'),device='cpu')
    if 'lm_head.weight' in state:raise ValueError('Expected the pinned shared-head serialization')
    state['lm_head.weight']=state['model.embed_tokens.weight']
    return state
