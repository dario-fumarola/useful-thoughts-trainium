import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from train.permutation_program_decoder import ProgramDecoder


class Tok:
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)


def test_online_symbol_stop_preserves_actions_and_ignores_later_text():
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=256,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        eos_token_id=0,pad_token_id=0))
    class Scripted(ProgramDecoder):
        def __init__(self,text):super().__init__(model,Tok());self.ids=list(text.encode());self.i=0
        def _next(self,*args,**kwargs):
            logits=torch.full((1,256),-100.);logits[0,self.ids[self.i]]=100.;self.i+=1
            return logits
    examples=[('direct','A\n'),('pause',' b\n'),('empty','Final answer:\nC\n'),
              ('trace','r: A -> B\nFinal answer:\nB\n')]
    for mode,text in examples:
        actual=Scripted(text+'Not generated').decode([1,2],mode=mode,max_new_tokens=100,
            prefill_width=8,pad_id=0,stop_ids=(0,))
        assert actual.text==text and actual.actions==list(text.encode())
        assert actual.stop_type=='qa_answer_line'
