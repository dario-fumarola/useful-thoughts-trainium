"""FP32 span likelihood on fixed QA fields and ordinary document continuations.

This reuses the qualified ordinary causal trunk. Targets are scoring-only;
the generation entrypoint never imports or opens these scoring examples.
"""
import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def score_spans(backend, contexts, targets, *, width, target_slots=32, pad_id=151643):
    """Score every supplied target token, with explicit causal alignment.

    Fixed-width right padding never enters a target. Each output contains the
    actual native-token vector, its full sum and first-token diagnostic. A
    caller must retain an exception as an execution failure,never silently
    omit an example or shorten its context/answer.
    """
    if (not contexts or len(contexts)!=len(targets) or type(width) is not int
            or width<2 or type(target_slots) is not int or target_slots<1):
        raise ValueError('Nonempty paired fixed-width spans required')
    vocab=backend.model.config.vocab_size
    if type(pad_id) is not int or not 0<=pad_id<vocab:
        raise ValueError('Valid native padding token required')
    n=len(contexts);x=torch.full((n,width),pad_id,dtype=torch.long)
    positions=torch.zeros((n,target_slots),dtype=torch.long)
    labels=torch.zeros_like(positions);lengths=[]
    for i,(past,future) in enumerate(zip(contexts,targets)):
        if (not past or not future or len(future)>target_slots
                or len(past)+len(future)>width
                or any(type(v) is not int or not 0<=v<vocab for v in [*past,*future])):
            raise ValueError('Complete bounded native context/target required;never truncate')
        x[i,:len(past)+len(future)]=torch.tensor([*past,*future])
        positions[i,:len(future)]=torch.arange(len(past)-1,len(past)+len(future)-1)
        labels[i,:len(future)]=torch.tensor(future);lengths.append(len(future))
    h=backend.hidden(x.to(backend.device)[:,:-1])
    selected=h.gather(1,positions.to(backend.device)[:,:,None].expand(-1,-1,h.shape[-1]))
    parts=[]
    for start in range(0,target_slots,backend.head_chunk):
        sl=slice(start,start+backend.head_chunk)
        with backend.head_amp():z=backend.model.lm_head(selected[:,sl]).float()
        lp=F.log_softmax(z,-1).gather(-1,labels[:,sl,None].to(backend.device))[:,:,0]
        parts.append(lp.cpu())
    values=torch.cat(parts,1);result=[]
    for i,length in enumerate(lengths):
        native=values[i,:length].tolist()
        if not all(math.isfinite(v) for v in native):raise RuntimeError('Nonfinite full-support log probabilities')
        result.append(dict(token_logps=native,full_logp=sum(native),K1_logp=native[0],tokens=length))
    return result
