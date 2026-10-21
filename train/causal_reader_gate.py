"""Causal frozen-reader features and a small selective head; no launcher.

Target log probabilities are returned separately from gate inputs. This module
does not load models, update backbones, choose notes or inspect source labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalFeatures:
    hidden: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class ReaderScores:
    features: CausalFeatures
    target_logps: torch.Tensor


@torch.no_grad()
def extract(backend, contexts, targets, *, width=672, pad_id=151643):
    """Rows immediately before targets; no target-specific gate features.

    Right padding and future targets cannot attend backward through the causal
    trunk. Actual-model qualification is still required beyond CPU fixtures.
    """
    if not contexts or len(contexts) != len(targets):
        raise ValueError('Nonempty paired documents required')
    size=backend.model.config.vocab_size
    x=torch.full((len(contexts),width),pad_id,dtype=torch.long)
    positions=torch.empty((len(contexts),16),dtype=torch.long)
    labels=torch.empty_like(positions)
    for index,(past,future) in enumerate(zip(contexts,targets)):
        if (not past or len(future)!=16 or len(past)+16>width
                or any(type(t) is not int or not 0<=t<size for t in [*past,*future])):
            raise ValueError('Exact complete native sixteen-token span required')
        x[index,:len(past)+16]=torch.tensor([*past,*future])
        positions[index]=torch.arange(len(past)-1,len(past)+15)
        labels[index]=torch.tensor(future)
    hidden=backend.hidden(x.to(backend.device)[:,:-1])
    chosen=hidden.gather(1,positions.to(backend.device)[:,:,None].expand(-1,-1,hidden.shape[-1]))
    backend.sync()
    # Export just the selected hidden rows, not full-vocabulary logits.
    causal_hidden=chosen.float().cpu().detach()
    entropies=[];target_logps=[]
    for start in range(0,16,backend.head_chunk):
        sl=slice(start,start+backend.head_chunk)
        with backend.head_amp():
            logits=backend.model.lm_head(chosen[:,sl]).float()
        logp=F.log_softmax(logits,-1)
        entropy=-(logp.exp()*logp).sum(-1)
        target=logp.gather(-1,labels[:,sl,None].to(backend.device))[:,:,0]
        backend.sync()
        entropies.append(entropy.cpu());target_logps.append(target.cpu())
    entropy=torch.cat(entropies,1);logps=torch.cat(target_logps,1)
    if not all(torch.isfinite(v).all() for v in (causal_hidden,entropy,logps)):
        raise RuntimeError('Nonfinite reader feature or target score')
    return ReaderScores(CausalFeatures(causal_hidden,entropy),logps)


def gate_inputs(raw: CausalFeatures, assisted: CausalFeatures, *,
                note_actions, capped, vocabulary_size):
    """No target IDs, target logps, correctness or source-kind argument."""
    if (raw.hidden.shape!=assisted.hidden.shape or raw.hidden.ndim!=3
            or raw.hidden.shape[1]!=16 or raw.entropy.shape!=raw.hidden.shape[:2]
            or assisted.entropy.shape!=raw.entropy.shape):
        raise ValueError('Paired complete causal hidden/entropy features required')
    n=raw.hidden.shape[0]
    counts=torch.as_tensor(note_actions,dtype=torch.float64)
    caps=torch.as_tensor(capped,dtype=torch.float64)
    if (counts.shape!=(n,) or caps.shape!=(n,) or (counts<0).any()
            or (counts>128).any() or not (counts==counts.floor()).all()
            or not ((caps==0)|(caps==1)).all()
            or type(vocabulary_size) is not int or vocabulary_size<2):
        raise ValueError('Actual bounded note metadata required')
    def normalized(value):
        value=value.detach().to(dtype=torch.float64,device='cpu')
        return value/torch.sqrt(value.square().mean(-1,keepdim=True)+1e-6)
    features=torch.cat((normalized(raw.hidden),normalized(assisted.hidden),
        (raw.entropy.detach().double()/math.log(vocabulary_size))[:,:,None],
        (assisted.entropy.detach().double()/math.log(vocabulary_size))[:,:,None],
        (counts/128)[:,None,None].expand(n,16,1),
        caps[:,None,None].expand(n,16,1),
        torch.arange(16,dtype=torch.float64)[None,:,None].expand(n,-1,-1)/16),-1)
    if not torch.isfinite(features).all():
        raise ValueError('Finite causal gate inputs required')
    return features.detach()


class SelectiveHead(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(dimensions,dtype=torch.float64))
        self.bias=nn.Parameter(torch.tensor(math.log(.0199/.9801),dtype=torch.float64))
        self.tau=.01
        self.closing_logit=math.log(self.tau/(1-self.tau))

    def forward(self, features):
        # Features are causal and detached; only these two head parameters learn.
        return features.detach().to(dtype=torch.float64)@self.weight+self.bias

    def log_weights(self, logits):
        active=logits>self.closing_logit
        distance=logits[active]-self.closing_logit
        # Shift around the closure boundary instead of subtracting two nearly
        # identical sigmoid probabilities. This remains finite near zero and
        # for saturated positive logits, with identical train/predict decisions.
        denominator=torch.logaddexp(torch.full_like(distance,math.log1p(-self.tau)),
                                    math.log(self.tau)+distance)
        log_g=math.log(self.tau)+distance+torch.log(-torch.expm1(-distance))-denominator
        return active,log_g,-denominator

    def gate(self, logits):
        active,log_g,_=self.log_weights(logits)
        values=torch.zeros_like(logits)
        values[active]=log_g.exp()
        return values

    def target_logps(self, logits, raw, assisted):
        """Exact raw values on closed rows; stable mixture elsewhere.

        The two predictive score arrays belong only to the likelihood loss,
        never to forward(features). Closed rows have zero local head gradient.
        """
        if logits.shape!=raw.shape or assisted.shape!=raw.shape:
            raise ValueError('Paired complete per-target scores required')
        lp0=raw.detach().to(dtype=torch.float64)
        lp1=assisted.detach().to(dtype=torch.float64)
        active,log_g,log_raw_weight=self.log_weights(logits)
        if not active.any():
            return lp0
        mixed=lp0.clone()
        mixed[active]=torch.logaddexp(log_raw_weight+lp0[active],log_g+lp1[active])
        return mixed

    def predict_logps(self, logits, raw, assisted=None, *, enabled=True):
        """Normalized full-vocabulary probability mixture; exact raw bypass.

        Callers can bypass writer/assisted work when disabled. This method does
        not choose a token or change the original vocabulary support.
        """
        if not enabled:
            return raw
        if logits.shape!=raw.shape[:-1] or assisted is None or assisted.shape!=raw.shape:
            raise ValueError('Paired full-vocabulary predictions required')
        active,log_g,log_raw_weight=self.log_weights(logits)
        if not active.any():
            return raw
        mixed=raw.detach().double().clone()
        mixed[active]=torch.logaddexp(log_raw_weight[:,None]+raw[active].detach().double(),
                                    log_g[:,None]+assisted[active].detach().double())
        return mixed

    def loss(self, features, raw, assisted):
        logits=self(features)
        mixture=self.target_logps(logits,raw,assisted)
        # Keep all-closed batches differentiable with exactly zero data gradient.
        return -mixture.mean(-1).mean()+0*logits.sum()
