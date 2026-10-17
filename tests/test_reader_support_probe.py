"""Same-logit support mass identity and exact fixed interface contexts."""
import torch

from train.reader_support_probe import contexts, support_from_logits
from weekend_thinking.core import Format


def test_support_mass_identity_matches_independent_restricted_softmax():
    logits = torch.tensor([[[1.,2.,-1.,4.],[3.,-2.,0.,1.]]])
    before = logits.clone()
    defined = torch.tensor([True,False,True,True])
    targets = torch.tensor([[3,0]])
    full,restricted,mass = support_from_logits(logits,targets,defined)
    selected = logits[...,defined].log_softmax(-1)
    expected = selected.gather(-1,torch.tensor([[[2],[0]]]))[...,0]
    independent_mass = logits.log_softmax(-1)[...,defined].logsumexp(-1)
    torch.testing.assert_close(restricted,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(mass,independent_mass,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(restricted-full,-mass,atol=1e-6,rtol=1e-6)
    assert torch.equal(logits,before) and (mass<=0).all()
    f,d,m=support_from_logits(logits,targets,torch.ones(4,dtype=torch.bool))
    assert torch.equal(f,d) and torch.equal(m,torch.zeros_like(m))


def test_all_controls_keep_same_support_context_and_plain_reference_bytes():
    fmt=Format((7,),(8,),9,0,header_ids=(6,))
    control=dict(content_ids=dict(empty=[],own=[30,31],filler=[40,40],shuffled=[31,30],other=[50,50]))
    prefix=list(range(100,140))
    rows={(i,m):v for i,m,v in contexts(prefix,control,fmt,[198])}
    assert len(rows)==16 and rows['raw','raw']==prefix
    assert rows['original','empty']==[6]+prefix+[7,8]
    assert rows['reanchor32','own']==[6]+prefix+[7,30,31,8]+prefix[-32:]
    assert rows['plain_reanchor32','empty']==prefix+[198,198]+prefix[-32:]
    assert rows['plain_reanchor32','other']==prefix+[198,50,50,198]+prefix[-32:]
