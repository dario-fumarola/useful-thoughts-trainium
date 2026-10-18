from dataclasses import asdict
from decimal import Decimal

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.evaluate import GreedyDecoder
from train.partial_state_answers import NumberLineDecoder, first_number_line_end, parsed_number, score_answer


@pytest.mark.parametrize('text,expected', [('1925\n','1925'), (' −1,925.0\n','-1925'),
    ('Answer: 1.925e3\n','1925'), ('\n\n+0005.\n','5'), ('Answer:\n -0\n','0'),
    ('  Answer: 1925\n','1925')])
def test_complete_numeric_surfaces(text, expected):
    assert first_number_line_end(text) is not None
    assert parsed_number(text) == Decimal(expected)


def test_no_later_number_rescue_and_cot_uses_first_answer_label():
    assert parsed_number('Not sure.\n1925\n') is None
    assert parsed_number('work 12 * 3 = 36\nAnswer:\n1925\n', cot=True) == 1925
    assert parsed_number('work 12 * 3 = 36\n  Answer:\n1925\n', cot=True) == 1925
    assert parsed_number('Answer: unsure\nAnswer: 1925\n', cot=True) is None
    assert parsed_number('1925\n', cot=True) is None
    assert parsed_number('1,92\n') is None
    assert parsed_number('NaN\n') is None


def test_eos_closes_number_but_cap_or_execution_failure_cannot_be_rescued():
    p = dict(text='1925', stop_type='token:151643')
    assert score_answer(p,1925)['score'] == 1
    assert score_answer(dict(p,stop_type='length_cap'),1925)['status'] == 'truncated'
    assert score_answer(dict(p,status='error'),1925)['score'] == 0
    assert score_answer(dict(p,status='format_error'),1925)['score'] == 0
    assert score_answer(dict(text='1925\nextra',stop_type='number_line'),1925)['score'] == 1
    # Parsing huge exponents does not expand them to giant decimal strings.
    assert score_answer(dict(text='1e999999999999\n',stop_type='number_line'),1925)['score'] == 0


def test_unknown_generation_outcome_or_nonfinite_gold_is_rejected():
    with pytest.raises(ValueError):
        score_answer(dict(text='1925\n',stop_type='missing'),1925)
    with pytest.raises(ValueError):
        score_answer(dict(text='1925\n',stop_type='number_line'),'NaN')


class NonNumericTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join('t'+str(i) for i in ids)


def model():
    torch.manual_seed(42)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()


def test_actual_cpu_qwen_cached_actions_match_original_when_number_stop_is_absent():
    m=model();tok=NonNumericTokenizer();old=GreedyDecoder(m,tok);new=NumberLineDecoder(m,tok)
    kwargs=dict(prompt_ids=[2,3,4],max_new_tokens=6,prefill_width=8,pad_id=0,stop_ids=(0,))
    a=old.decode(**kwargs,answer_line_stop=False);b=new.decode_number(**kwargs)
    assert asdict(a)==asdict(b)


def test_host_stopping_preserves_full_native_token_when_it_crosses_newline():
    class Tok:
        def decode(self, ids, **kwargs):
            return ''.join({1:'19',2:'25\nextra'}[i] for i in ids)
    class Fixed(NumberLineDecoder):
        def _next(self, ids, positions, cache, selected_position=None):
            step=getattr(self,'step',0);self.step=step+1
            logits=torch.full((1,19),-100.);logits[0,1 if step==0 else 2]=100.;return logits
    out=Fixed(model(),Tok()).decode_number([3],max_new_tokens=4,prefill_width=4,pad_id=0,stop_ids=(0,))
    assert out.actions==[1,2] and out.content_ids==[1,2]
    assert out.raw_text=='1925\nextra' and out.text=='1925\n' and out.stop_type=='number_line'
