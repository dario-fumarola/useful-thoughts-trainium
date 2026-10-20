"""Causal boundaries and complete-attempt accounting for the V29 readout."""
from copy import deepcopy
from fractions import Fraction

import pytest

from train.real_cpt_readout import (controls, corrupt_work, donors,
    equal_length_batches, numeric_masks, validate_note)
from train.real_trace_contexts import question_projection, writer_prompt


class Tokenizer:
    eos_token_id=151643
    all_special_ids=[151643]
    def __len__(self):return 151669
    def get_vocab(self):return {chr(i):i for i in range(128)} | {' -':300,'EOS':151643}
    def encode(self,text,add_special_tokens=False):
        result=[]
        while text:
            if text.startswith(' -'):result.append(300);text=text[2:]
            else:result.append(ord(text[0]));text=text[1:]
        return result
    def decode(self,ids,**kwargs):return ''.join(' -' if i==300 else chr(i) for i in ids)


def example():
    tok=Tokenizer()
    return dict(item_id='a',document_id='doc-a',prefix_ids=tok.encode('source '*20+'(2 + 3) * 4 = '))


def test_writer_and_verified_controls_ignore_all_future_fields():
    q=example();tok=Tokenizer()
    class FutureForbidden(dict):
        def __getitem__(self,key):
            if key not in ('item_id','document_id','prefix_ids'):raise AssertionError('future accessed')
            return super().__getitem__(key)
    row=FutureForbidden(q,continuation16_ids=[999],full_calculation_text='SECRET',numeric_value='SECRET')
    assert question_projection(row)==q
    assert writer_prompt(row,tok)==writer_prompt(q,tok)
    note=dict(content_ids=tok.encode('5 * 4 = 20\n'))
    first=controls(row,tok,note,note)
    assert first==controls(q,tok,note,note)
    assert first['raw']==q['prefix_ids']
    assert first['own'][-8:]==tok.encode('Answer:\n')
    text,_=corrupt_work(row,tok)
    assert text=='2 + 3 = 5\n5 * 4 = 21\n'


def test_donor_is_intact_stable_derangement_without_outcomes():
    ids=['id-'+str(i) for i in range(17)]
    a=donors(ids)
    assert a==donors(list(reversed(ids)))
    assert set(a.values())==set(ids)
    assert all(k!=v for k,v in a.items())
    with pytest.raises(ValueError):donors(['a','a'])


def test_equal_length_grouping_preserves_every_real_item_without_padding():
    qs=[dict(prefix_ids=[1]*n) for n in [128]*13+[131]*7+[512]*16]
    batches=list(equal_length_batches(qs))
    assert sorted(i for batch in batches for i in batch)==list(range(len(qs)))
    assert [len(b) for b in batches]==[8,4,1,4,1,1,1,8,8]
    assert all(len({len(qs[i]['prefix_ids']) for i in b})==1 for b in batches)


def test_partial_error_and_cap_remain_original_observations():
    tok=Tokenizer();q=example();a=[ord('a')]*128
    row=dict(item_id=q['item_id'],document_id=q['document_id'],prompt_ids=writer_prompt(q,tok),
        status='ok',actions=a,content_ids=a,text='a'*128,raw_text='a'*128,stop_type='length_cap')
    assert validate_note(row,q,tok,stop_ids=(151643,65))==row
    partial=deepcopy(row);partial.update(status='error',actions=a[:3],content_ids=a[:3],
        text='aaa',raw_text='aaa',stop_type='error')
    assert validate_note(partial,q,tok,stop_ids=(151643,65))==partial
    broken=deepcopy(partial);broken['content_ids']=[]
    with pytest.raises(ValueError):validate_note(broken,q,tok,stop_ids=(151643,65))


@pytest.mark.parametrize('tail,literal,value,boundary',[
    (' -16/2 = -8 more text beyond','-16/2','-8','='),
    (' 25\n= 5 + 20 some more text','25','25','='),
    (' 20. More ordinary text here','20','20','.'),
])
def test_native_numeric_masks_handle_merged_signs_and_boundary_whitespace(tail,literal,value,boundary):
    tok=Tokenizer();q=example();target=tok.encode(tail)[:16]
    q.update(continuation16_ids=target,numeric_surface_text=literal,numeric_value=value,
             closing_boundary_ids=tok.encode(boundary))
    masks=numeric_masks(q,tok)
    assert all(len(mask)==16 for mask in masks.values())
    numeric=tok.decode([t for t,use in zip(target,masks['numeral']) if use]).strip()
    assert Fraction(numeric)==Fraction(value)
    assert all(sum((masks['numeral'][i],masks['boundary_and_whitespace'][i],masks['surrounding'][i]))==1 for i in range(16))
    assert masks['numeral_and_boundary']==[a or b for a,b in zip(masks['numeral'],masks['boundary_and_whitespace'])]
