"""Process-local diagnostic formulas; no edits to installed or production code."""
import types
import torch
from torch.nn import functional as F

OBSERVER = None


def observe(name, value):
    if OBSERVER is not None:
        OBSERVER(name, value)


def rms_reference(self, hidden_states):
    # Same native RMS formula, preserving the diagnostic input precision.
    observe('rms.input',hidden_states)
    variance=hidden_states.pow(2).mean(-1,keepdim=True)
    observe('rms.variance',variance)
    hidden_states=hidden_states*torch.rsqrt(variance+self.variance_epsilon)
    result=self.weight*hidden_states
    observe('rms.output',result)
    return result


def eager_attention_reference(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    key_states=repeat_kv(key,module.num_key_value_groups)
    value_states=repeat_kv(value,module.num_key_value_groups)
    weights=torch.matmul(query,key_states.transpose(2,3))*scaling
    if attention_mask is not None:
        weights=weights+attention_mask
    observe('attention.pre_softmax',weights)
    weights=F.softmax(weights,dim=-1,dtype=query.dtype).to(query.dtype)
    observe('attention.softmax',weights)
    weights=F.dropout(weights,p=dropout,training=module.training)
    result=torch.matmul(weights,value_states).transpose(1,2).contiguous()
    observe('attention.output',result)
    return result,weights


def install(model,cos_f32,sin_f32):
    """Fix native F32 positional constants, promote differentiable arithmetic."""
    from transformers.models.qwen3 import modeling_qwen3 as qwen
    if cos_f32.dtype!=torch.float32 or sin_f32.dtype!=torch.float32:
        raise ValueError('RoPE constants must be the original native FP32 values')
    cos,sin=cos_f32.double(),sin_f32.double()
    positions=torch.arange(cos.shape[-2])[None,:]

    def frozen_rotary(self,x,position_ids):
        if (position_ids.shape[-1]!=positions.shape[-1]
                or not torch.equal(position_ids,positions.expand_as(position_ids))):
            raise ValueError('Unexpected RoPE positions; reference is fixed to0..511')
        if x.dtype!=torch.float64:raise ValueError('Differentiable hidden state is not FP64')
        observe('rope.frozen_cos',cos);observe('rope.frozen_sin',sin)
        return cos.expand(position_ids.shape[0],-1,-1),sin.expand(position_ids.shape[0],-1,-1)

    count=0
    for module in model.modules():
        if isinstance(module,qwen.Qwen3RMSNorm):
            module.forward=types.MethodType(rms_reference,module);count+=1
    model.model.rotary_emb.forward=types.MethodType(frozen_rotary,model.model.rotary_emb)
    qwen.eager_attention_forward=eager_attention_reference
    return dict(rms_modules=count,rope_mode='Frozen native-CPU FP32 cos/sin constants cast exactly toFP64',
                attention='Native eager formula with dtype-preserving softmax',
                installed_files_changed=False)
