import pytest
from train.real_trace_contexts import (question_projection,writer_prompt,
    reader_context,boundary_stop_ids,EOS)


class PrefixOnly(dict):
    def __getitem__(self,key):
        if key not in ('item_id','document_id','prefix_ids'):
            raise AssertionError('Writer accessed offline label: '+key)
        return super().__getitem__(key)


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        return {'\nCalculation:\n':[7,8,9],'Answer:\n':[10,11,9]}[text]
    def decode(self,ids,**kwargs):
        assert ids==[10,11,9]
        return 'Answer:\n'


def test_question_reads_only_prefix_projection_and_copies_native_ids():
    source=PrefixOnly(item_id='x',document_id='d',prefix_ids=[23]*128,
                      continuation16_ids=[91]*16,full_calculation_ids=[92])
    q=question_projection(source)
    source['prefix_ids'][0]=24
    assert q['prefix_ids']==[23]*128
    assert set(q)=={'item_id','document_id','prefix_ids'}


def test_raw_empty_and_note_have_exact_native_contexts():
    q=dict(prefix_ids=[23]*128);tok=Tokenizer()
    assert reader_context(q,tok)==q['prefix_ids']
    assert reader_context(q,tok,[])==q['prefix_ids']+[7,8,9,10,11,9]
    assert reader_context(q,tok,[44,45])==q['prefix_ids']+[7,8,9,44,45,10,11,9]
    assert writer_prompt(q,tok)==q['prefix_ids']+[7,8,9]
    assert boundary_stop_ids(tok)==(EOS,10)


def test_eos_cannot_be_silently_inserted_in_note_content():
    with pytest.raises(ValueError):reader_context(dict(prefix_ids=[23]*128),Tokenizer(),[EOS])
