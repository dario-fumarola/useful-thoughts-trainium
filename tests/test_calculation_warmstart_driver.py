"""CPU gradient checks for sequential component accumulation; no hardware claim."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from train import calculation_warmstart_driver as driver


class TinyBackend:
    device=torch.device('cpu')
    head_chunk=32
    head_amp=staticmethod(nullcontext)
    def __init__(self):
        self.model=torch.nn.Module()
        self.model.config=SimpleNamespace(vocab_size=17)
        self.model.embedding=torch.nn.Embedding(17,6)
        self.model.lm_head=torch.nn.Linear(6,17,bias=False)
        self.model.lm_head.weight=self.model.embedding.weight
    def hidden(self,ids):
        # Prefix-sensitive causal trunk; labels are shifted in the CE helper.
        return self.model.embedding(ids).cumsum(1)/32


def packed(component):
    count={'source_only':527,'note_only':5,'reader_only':16}[component]
    offset={'source_only':0,'note_only':3,'reader_only':6}[component]
    ids=((torch.arange(544)+offset)%17).reshape(1,-1).long()
    positions=torch.arange(1,count+1).reshape(1,-1)
    note=torch.full((1,count),component=='note_only',dtype=torch.bool)
    return dict(input_ids=ids,target_positions=positions,labels=ids.gather(1,positions),
        note_close_mask=note,future_mask=~note)


@pytest.mark.parametrize('arm',['joint','source_control'])
def test_sequential_backward_matches_dense_sum_and_preserves_existing_gradient(monkeypatch,arm):
    from train import calculation_warmstart_data
    torch.manual_seed(91); backend=TinyBackend(); reference=copy.deepcopy(backend)
    # A nonzero preexisting gradient catches accidental zero_grad inside a pass.
    for a,b in zip(backend.model.parameters(),reference.model.parameters()):
        a.grad=torch.full_like(a,.125); b.grad=torch.full_like(b,.125)
    calls=[]; syncs=[]
    def pack(row,tok,name):
        calls.append(name); return packed(name)
    monkeypatch.setattr(calculation_warmstart_data,'pack_example',pack)
    losses,counts,times=driver.backward_components({},None,arm,backend,lambda:syncs.append(1))
    dense=[]
    for name in driver.components(arm):
        batch=packed(name); hidden=reference.hidden(batch['input_ids'][:,:-1])
        positions=batch['target_positions']-1
        selected=hidden.gather(1,positions[:,:,None].expand(-1,-1,6))
        loss=F.cross_entropy(reference.model.lm_head(selected).reshape(-1,17),batch['labels'].flatten())
        assert losses[name]==pytest.approx(float(loss.detach()),abs=2e-6)
        dense.append(loss)
    sum(dense).backward()
    for a,b in zip(backend.model.parameters(),reference.model.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-6)
    assert calls==list(driver.components(arm)) and len(syncs)==len(calls)
    assert counts['source_only']==527
    assert counts['note_only']==(5 if arm=='joint' else 0)
    assert counts['reader_only']==(16 if arm=='joint' else 0)
    assert all(value>=0 for value in times.values())


def test_nonfinite_component_stops_before_any_backward(monkeypatch):
    from train import calculation_warmstart_data
    backend=TinyBackend()
    with torch.no_grad(): backend.model.embedding.weight.fill_(float('nan'))
    monkeypatch.setattr(calculation_warmstart_data,'pack_example',lambda *args:packed('source_only'))
    with pytest.raises(RuntimeError,match='Nonfinite source_only'):
        driver.backward_components({},None,'joint',backend,lambda:None)
    assert all(p.grad is None for p in backend.model.parameters())


@pytest.mark.parametrize('component,wrong_mask',[
    ('source_only','note_close_mask'),('note_only','future_mask'),('reader_only','note_close_mask')])
def test_components_refuse_changed_supervision_roles(component,wrong_mask):
    batch=packed(component); batch[wrong_mask][0,0]=True
    with pytest.raises(ValueError,match='supervised labels'):
        driver.supervision_counts(batch,component)
