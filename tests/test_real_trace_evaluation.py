from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from train.real_trace_evaluation import span_logprobs


class Backend:
    def __init__(self):
        torch.manual_seed(71)
        self.embedding=torch.nn.Embedding(13,7)
        self.model=SimpleNamespace(config=SimpleNamespace(vocab_size=13),
                                   lm_head=torch.nn.Linear(7,13))
        self.device=torch.device('cpu')
        self.head_amp=nullcontext
        self.sync=lambda:None

    def hidden(self,x):
        return self.embedding(x).cumsum(1)  # A strictly causal reference.


def test_heterogeneous_contexts_equal_independent_prefix_only_prediction_loop():
    backend=Backend();contexts=[[2,3],[4,5,6,7,8]]
    targets=[[1,2,3,4]*4,[2,6,4,1]*4]
    got=span_logprobs(backend,contexts,targets,width=32,pad_id=0)
    expected=[]
    with torch.no_grad():
        for c,y in zip(contexts,targets):
            values=[]
            for j,t in enumerate(y):
                # Each independent prediction sees ONLY its own causal past.
                h=backend.hidden(torch.tensor([c+y[:j]]))[:,-1]
                values.append(backend.model.lm_head(h).log_softmax(-1)[0,t])
            expected.append(torch.stack(values))
    torch.testing.assert_close(got,torch.stack(expected),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(got,span_logprobs(backend,contexts,targets,width=40,pad_id=12))


def test_a_later_actual_target_cannot_change_earlier_likelihoods():
    backend=Backend();context=[[3,4,5]];target=[[2]*16]
    before=span_logprobs(backend,context,target,width=24,pad_id=0)
    target[0][-1]=7
    after=span_logprobs(backend,context,target,width=24,pad_id=0)
    assert torch.equal(before[:,:15],after[:,:15])


@pytest.mark.parametrize('contexts,targets,width',[([],[],32),([[]],[[1]*16],32),
    ([[1]],[[2]*15],32),([[1]*17],[[2]*16],32),([[99]],[[2]*16],32)])
def test_incomplete_or_overwidth_inputs_are_rejected_not_dropped(contexts,targets,width):
    with pytest.raises(ValueError):span_logprobs(Backend(),contexts,targets,width=width,pad_id=0)
