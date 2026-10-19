import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from train.document_qa_decoder import QADecoder


class Tok:
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)


def test_online_stop_retains_every_native_action_and_uses_no_gold():
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=256,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        eos_token_id=0,pad_token_id=0))
    class Scripted(QADecoder):
        def __init__(self,text):super().__init__(model,Tok());self.ids=list(text.encode());self.i=0
        def _next(self,*args,**kwargs):
            logits=torch.full((1,256),-100.)
            logits[0,self.ids[self.i]]=100.;self.i+=1
            return logits
    for mode,text,expected in [('direct','Lyon\nUnseen tail','Lyon\n'),
                               ('prior','Step 1: fact\nFinal answer:\nLyon\nUnseen tail','Step 1: fact\nFinal answer:\nLyon\n')]:
        r=Scripted(text).decode([1,2],mode=mode,max_new_tokens=80,prefill_width=8,pad_id=0,stop_ids=(0,))
        assert r.stop_type=='qa_answer_line' and r.text==expected
        assert r.actions==list(expected.encode()) and r.content_ids==r.actions


def test_cache_native_ids_match_ordinary_causal_greedy_tokens():
    torch.manual_seed(21)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        eos_token_id=0,pad_token_id=0,attn_implementation='eager')).eval()
    prefix=[2,3,4];sequence=list(prefix);actions=[]
    for _ in range(5):
        with torch.no_grad():v=int(model(torch.tensor([sequence])).logits[0,-1].argmax())
        sequence.append(v);actions.append(v)
    actual=QADecoder(model,Tok()).decode(prefix,mode='prior',max_new_tokens=5,
        prefill_width=8,pad_id=0,stop_ids=(999,))
    assert actual.actions==actions and actual.stop_type=='length_cap'
