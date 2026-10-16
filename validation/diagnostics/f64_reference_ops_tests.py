"""Tiny CPU math/dtype fixtures; never a full-Qwen hardware qualification."""
from types import SimpleNamespace
import torch
from transformers.models.qwen3 import modeling_qwen3 as qwen
import f64_reference_ops as ops

torch.set_num_threads(1)


def test_rms_native_float32_formula_and_gradients_bitwise():
    torch.manual_seed(11)
    layer=qwen.Qwen3RMSNorm(8)
    x=torch.randn(2,5,8,requires_grad=True)
    original=layer(x);og=torch.autograd.grad(original.square().sum(),(x,layer.weight))
    proposed=ops.rms_reference(layer,x);pg=torch.autograd.grad(proposed.square().sum(),(x,layer.weight))
    assert torch.equal(original,proposed)
    assert all(torch.equal(a,b) for a,b in zip(og,pg))


def test_eager_native_float32_formula_mask_and_gradients_bitwise():
    torch.manual_seed(12)
    module=SimpleNamespace(num_key_value_groups=2,training=False)
    tensors=[torch.randn(2,h,7,8,requires_grad=True) for h in (4,2,2)]
    mask=torch.where(torch.arange(7)[None,:]<=torch.arange(7)[:,None],0.,float('-inf'))[None,None]
    original=qwen.eager_attention_forward(module,*tensors,mask,8**-.5)
    og=torch.autograd.grad(original[0].square().sum(),tensors)
    proposed=ops.eager_attention_reference(module,*tensors,mask,8**-.5)
    pg=torch.autograd.grad(proposed[0].square().sum(),tensors)
    assert all(torch.equal(a,b) for a,b in zip(original,proposed))
    assert all(torch.equal(a,b) for a,b in zip(og,pg))


def test_references_preserve_float64_through_norm_softmax_and_backward():
    seen={}
    ops.OBSERVER=lambda name,value:seen.setdefault(name,set()).add(str(value.dtype))
    try:
        layer=qwen.Qwen3RMSNorm(8).double()
        x=torch.randn(2,7,8,dtype=torch.float64,requires_grad=True)
        normalized=ops.rms_reference(layer,x)
        tensors=[torch.randn(2,h,7,8,dtype=torch.float64,requires_grad=True) for h in (4,2,2)]
        output,_=ops.eager_attention_reference(SimpleNamespace(num_key_value_groups=2,training=False),
            *tensors,None,8**-.5)
        (normalized.square().sum()+output.square().sum()).backward()
        assert all(values=={'torch.float64'} for values in seen.values())
        assert x.grad.dtype==torch.float64
        assert all(t.grad.dtype==torch.float64 for t in tensors)
        assert 'attention.softmax' in seen and 'rms.variance' in seen
    finally:
        ops.OBSERVER=None


def test_tiny_complete_model_freezes_native_rope_and_keeps_fp64_components():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(13)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=37,hidden_size=24,intermediate_size=40,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,attn_implementation='eager')).eval()
    ids=torch.tensor([[1,2,4,5,6,7,8]])
    positions=torch.arange(7)[None,:]
    cos,sin=model.model.rotary_emb(torch.zeros(1,7,24),positions)
    model.double();old_attention=qwen.eager_attention_forward
    seen={}
    ops.OBSERVER=lambda name,value:seen.setdefault(name,set()).add(str(value.dtype))
    try:
        info=ops.install(model,cos,sin)
        a,b=model.model.rotary_emb(torch.zeros(1,7,24,dtype=torch.float64),positions)
        assert torch.equal(a,cos.double()) and torch.equal(b,sin.double())
        assert info['rms_modules']==9
        logits=model(ids,use_cache=False).logits
        assert logits.dtype==torch.float64
        torch.nn.functional.cross_entropy(logits.reshape(-1,37),torch.tensor([2,4,5,6,7,8,9])).backward()
        assert all(p.grad.dtype==torch.float64 and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        assert all(dtypes=={'torch.float64'} for dtypes in seen.values())
        assert {'rms.variance','attention.softmax','rope.frozen_cos'}<=set(seen)
    finally:
        qwen.eager_attention_forward=old_attention
        ops.OBSERVER=None
