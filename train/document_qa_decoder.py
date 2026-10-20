"""Existing causal cache/greedy policy with a gold-independent QA field stop."""
import torch

from train.document_qa_answers import parse_answer
from train.evaluate import Decoded, GenerationFailure, GreedyDecoder
from train.neuron_backend import CausalCache
from weekend_thinking.core import policy_log_probs


class QADecoder(GreedyDecoder):
    @torch.no_grad()
    def decode(self, prompt_ids, *, mode, max_new_tokens=256,
               prefill_width=3584, pad_id=151643, stop_ids=(151643,151645),
               answer_parser=None):
        if mode not in ('direct','prior') or not prompt_ids or len(prompt_ids)>prefill_width or max_new_tokens<1:
            raise ValueError('Declared complete QA prompt and decoding bounds required')
        actions, content, raw = [], [], ''
        try:
            prompt=torch.full((1,prefill_width),pad_id,dtype=torch.long)
            prompt[0,:len(prompt_ids)]=torch.tensor(prompt_ids,dtype=torch.long)
            cache=CausalCache(self.model.config.num_hidden_layers,prefill_width+max_new_tokens)
            position=torch.tensor([len(prompt_ids)],device=self.device,dtype=torch.long)
            logits=self._next(prompt.to(self.device),torch.arange(prefill_width,device=self.device),cache,position-1)
            for step in range(max_new_tokens):
                token=policy_log_probs(logits,1.,self.banned_ids).argmax(-1)
                self.sync();value=int(token.cpu()[0]);actions.append(value)
                if value in stop_ids:
                    return Decoded(actions,content,raw,raw,f'token:{value}')
                content.append(value)
                raw=self.tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                parsed=(parse_answer if answer_parser is None else answer_parser)(raw,mode)
                if parsed.status=='complete':
                    return Decoded(actions,content,raw,raw[:parsed.end],'qa_answer_line')
                if step+1<max_new_tokens:
                    logits=self._next(token[:,None],position,cache);position=position+1
            return Decoded(actions,content,raw,raw,'length_cap')
        except Exception as error:
            raise GenerationFailure(error,Decoded(actions,content,raw,raw,'error')) from error
