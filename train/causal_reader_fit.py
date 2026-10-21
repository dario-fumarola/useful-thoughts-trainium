"""CPU-only fixed head fitting and the TRAIN-optimal scalar mixture control.

No model loader, source-data reader, note generator or DEV selection lives here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from train.causal_reader_gate import SelectiveHead


def fixed_order(documents, seed):
    if documents!=5120 or seed not in (2026102131,2026102132):
        raise ValueError('Frozen5120-source/two-order protocol required')
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.permutation(documents) for _ in range(10)])


def constant_logps(raw,assisted,weight):
    if weight==0:return raw
    if weight==1:return assisted
    if not 0<weight<1:raise ValueError('Convex probability mixture required')
    return torch.logaddexp(math.log1p(-weight)+raw,math.log(weight)+assisted)


def fit_constant(raw,assisted):
    """One-dimensional convex TRAIN loss; endpoints explicitly compete.

    Root-find the monotone CE derivative rather than selecting a DEV grid.
    The posterior-responsibility form avoids exp(logp1-logp0) overflow.
    """
    raw,assisted=raw.detach().double(),assisted.detach().double()
    if raw.shape!=assisted.shape or not torch.isfinite(raw).all() or not torch.isfinite(assisted).all():
        raise ValueError('Finite paired TRAIN scores required')
    lower,upper=0.,1.
    for _ in range(64):
        weight=(lower+upper)/2
        if weight in (lower,upper):break
        mixture=constant_logps(raw,assisted,weight)
        first=(math.log1p(-weight)+raw-mixture).exp()
        second=(math.log(weight)+assisted-mixture).exp()
        derivative=(first/(1-weight)-second/weight).mean().item()
        if derivative>0:upper=weight
        else:lower=weight
    candidate=(lower+upper)/2
    options=[(float(-constant_logps(raw,assisted,w).mean()),w) for w in (0.,candidate,1.)]
    loss,weight=min(options)
    return dict(weight=weight,TRAIN_token_NLL=loss,method='convex derivative bisection plus endpoints')


def fit_position_constants(raw,assisted):
    """Secondary control:16 TRAIN-optimal weights; no hidden/context inputs."""
    if raw.ndim!=2 or raw.shape[1]!=16 or raw.shape!=assisted.shape:
        raise ValueError('Complete TRAIN sixteen-token score arrays required')
    fitted=[fit_constant(raw[:,position],assisted[:,position]) for position in range(16)]
    return dict(weights=[r['weight'] for r in fitted],
                TRAIN_token_NLL=sum(r['TRAIN_token_NLL'] for r in fitted)/16,
                method='independent convex TRAIN fit per fixed position; secondary control')


def fit_head(features,raw,assisted,*,order,deadline_monotonic,emit=None,probe=None,checkpoint=None):
    """Exactly800 updates; callers supply only TRAIN features and scores."""
    if (features.ndim!=3 or features.shape[:2]!=(5120,16)
            or raw.shape!=(5120,16) or assisted.shape!=raw.shape
            or len(order)!=51200 or features.dtype!=torch.float64):
        raise ValueError('Frozen complete CPU TRAIN tensors/order required')
    for epoch in range(10):
        if not np.array_equal(np.sort(order[epoch*5120:(epoch+1)*5120]),np.arange(5120)):
            raise ValueError('Each epoch must expose each source exactly once')
    if not all(torch.isfinite(v).all() for v in (features,raw,assisted)):
        raise ValueError('Nonfinite frozen feature/score cache')
    head=SelectiveHead(features.shape[-1])
    optimizer=torch.optim.AdamW([{'params':[head.weight],'weight_decay':.0001},
                                {'params':[head.bias],'weight_decay':0}],
                               lr=.01,betas=(.9,.999),eps=1e-8)
    raw=raw.detach().double();assisted=assisted.detach().double()
    for step in range(800):
        if time.monotonic()>deadline_monotonic:raise TimeoutError('Finite CPU head-fit deadline')
        indices=torch.tensor(order[step*64:(step+1)*64],dtype=torch.long)
        x=features[indices];lp0=raw[indices];lp1=assisted[indices]
        optimizer.zero_grad(set_to_none=True)
        before=torch.cat((head.weight.detach(),head.bias.detach()[None])).clone()
        loss=head.loss(x,lp0,lp1)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite head loss')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(gradient):raise RuntimeError('Nonfinite head gradient')
        optimizer.step()
        with torch.no_grad():
            after=torch.cat((head.weight,head.bias[None]))
            values=head.gate(head(x))
            if not torch.isfinite(after).all():raise RuntimeError('Nonfinite head parameters')
            positive=values[values>0]
            record=dict(step=step+1,CE=loss.item(),preclip_data_gradient_norm=gradient.item(),
                optimizer_step_norm=torch.linalg.vector_norm(after-before).item(),
                closed_fraction=(values==0).double().mean().item(),
                positive_gate_quantiles=torch.quantile(positive,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()
                    if positive.numel() else [])
        if emit:emit(record)
        if probe and (step+1)%80==0:probe(head,step+1)
        if checkpoint and (step+1)%200==0:
            checkpoint(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),step=step+1))
    return head
