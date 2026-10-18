"""CPU-only packing semantics; no corpus preparation, model or training evidence."""
import copy

import pytest
import torch

from train import calculation_warmstart_data as data


class NativeFixture:
    eos_token_id=data.PAD
    def __init__(self,*,drop_newline=False,unsupported=False):
        self.drop_newline=drop_newline;self.unsupported=unsupported
    def get_vocab(self):return {**{str(i):i for i in range(128)},'fused-dot-newline':999,'EOS':data.PAD}
    def encode(self,text,**kwargs):
        if self.drop_newline:text=text.replace('\n','')
        result=[]
        while text:
            if text.startswith('.\n'):result.append(999);text=text[2:]
            else:result.append(ord(text[0]));text=text[1:]
        return result+[1000] if self.unsupported else result
    def decode(self,ids,**kwargs):
        return ''.join('.\n' if t==999 else '<EOS>' if t==data.PAD else chr(t) for t in ids)


def example(tok=None):
    tok=tok or NativeFixture();prefix='A source paragraph. '*7+'(3*7)+4 =';note='Compute 3*7 = 21.'
    prefix_ids=tok.encode(prefix);note_ids=tok.encode(note)
    source=prefix_ids+tok.encode(' 25. Different original following source. ')
    source += [32]*(528-len(source))
    return dict(split='train',prefix_text=prefix,prefix_ids=prefix_ids,note_text=note,note_ids=note_ids,
        boundary=dict(note=dict(correct_text=note,correct_ids=note_ids)),ordinary_source_ids=source,
        verified_numeric_rhs=dict(value='25',text='25'))


def assert_ce_metadata(batch,width,slots):
    assert set(batch)=={'input_ids','target_positions','labels','note_close_mask','future_mask'}
    assert batch['input_ids'].shape==(1,width)
    assert all(t.device.type=='cpu' for t in batch.values())
    assert batch['input_ids'].dtype==batch['labels'].dtype==batch['target_positions'].dtype==torch.long
    for name in ('labels','target_positions','note_close_mask','future_mask'):assert batch[name].shape==(1,slots)
    assert batch['note_close_mask'].dtype==batch['future_mask'].dtype==torch.bool
    note,source=batch['note_close_mask'],batch['future_mask'];active=note|source
    assert not (note&source).any()
    positions=batch['target_positions'][active]
    assert (positions>=1).all() and (positions<width).all() and len(positions)==len(positions.unique())
    assert torch.equal(batch['labels'][active],batch['input_ids'].gather(1,batch['target_positions'])[active])
    assert (batch['labels'][~active]==-100).all() and (batch['target_positions'][~active]==0).all()


def test_exact_shared_prompt_two_checked_demos_and_future_absence():
    tok=NativeFixture();row=example(tok);prompt=data.writer_prompt(row['prefix_text'],tok)
    rendered=tok.decode(prompt)
    assert '(17+28)*9 =\nCalculation:\nCompute 17+28 = 45.' in rendered
    assert '(36/4)+23 =\nCalculation:\nCompute 36/4 = 9.' in rendered
    assert rendered.endswith('Text:\n'+row['prefix_text']+'\nCalculation:\n')
    assert 'Different original following source' not in rendered
    packed=data.pack_example(row,tok,'note_only')
    assert torch.equal(packed['input_ids'][0,:len(prompt)],torch.tensor(prompt))
    assert_ce_metadata(packed,832,129)
    assert not packed['future_mask'].any()


def test_note_canonical_newline_is_encoded_jointly_with_fused_punctuation():
    tok=NativeFixture();row=example(tok);packed=data.pack_example(row,tok,'note_only')
    active=packed['note_close_mask'];labels=packed['labels'][active].tolist()
    assert labels==tok.encode(row['note_text']+'\n')
    assert labels[-1]==999 and labels!=tok.encode(row['note_text'])+tok.encode('\n')
    assert tok.decode(labels)==row['note_text']+'\n'
    assert packed['target_positions'][active][0]==len(data.writer_prompt(row['prefix_text'],tok))


def test_changing_future_suffix_or_rhs_cannot_change_any_writer_tensor():
    tok=NativeFixture();row=example(tok);other=copy.deepcopy(row)
    cut=len(row['prefix_ids']);other['ordinary_source_ids'][cut:]=[88]*(528-cut)
    other['verified_numeric_rhs']={'value':'1234567','text':'source RHS never accessed'}
    before=data.pack_example(row,tok,'note_only');after=data.pack_example(other,tok,'note_only')
    assert all(torch.equal(before[k],after[k]) for k in before)
    # Writer packing also works without any future-bearing source/target field.
    del other['ordinary_source_ids'];del other['verified_numeric_rhs']
    no_future=data.pack_example(other,tok,'note_only')
    assert all(torch.equal(before[k],no_future[k]) for k in before)


def test_source_is_exact_original528_and_all527_targets_including_last_and_real_eos():
    tok=NativeFixture();row=example(tok);row['ordinary_source_ids'][300]=data.PAD;row['ordinary_source_ids'][-1]=90
    packed=data.pack_example(row,tok,'source_only');assert_ce_metadata(packed,544,527)
    assert packed['target_positions'].tolist()==[list(range(1,528))]
    assert packed['labels'].tolist()==[row['ordinary_source_ids'][1:]]
    assert packed['labels'][0,-1]==90 and packed['future_mask'].all() and not packed['note_close_mask'].any()
    assert packed['labels'][0,299]==data.PAD  # True source EOS is supervised, never inferred as padding.
    assert packed['input_ids'][0,:528].tolist()==row['ordinary_source_ids']
    assert packed['input_ids'][0,528:].tolist()==[data.PAD]*16
    assert tok.decode(packed['input_ids'][0,:528].tolist())==tok.decode(row['ordinary_source_ids'])
    minimal=dict(split='train',ordinary_source_ids=row['ordinary_source_ids'])
    assert all(torch.equal(packed[k],data.pack_example(minimal,tok,'source_only')[k]) for k in packed)


@pytest.mark.parametrize('length',[1,128])
def test_note_content_and_closure_bounds_allow_exact_cap(length):
    tok=NativeFixture();row=example(tok);row['note_text']='X'*length;row['note_ids']=tok.encode(row['note_text'])
    row['boundary']['note']=dict(correct_text=row['note_text'],correct_ids=row['note_ids'])
    batch=data.pack_example(row,tok,'note_only');assert_ce_metadata(batch,832,129)
    assert batch['note_close_mask'].sum()==length+1


@pytest.mark.parametrize('damage',['dev','test','missing_split','unknown','empty','long_note','prefix_ids','proof_text',
                                  'proof_ids','note_ids','short_prefix','unfinished_prefix','overwidth','source_length','source_bool','source_undefined'])
def test_packer_rejects_invalid_or_overbound_inputs_without_truncation(damage):
    tok=NativeFixture();row=example(tok);component='note_only'
    if damage in ('dev','test'):row['split']=damage
    elif damage=='missing_split':row.pop('split')
    elif damage=='unknown':component='joint'
    elif damage in ('empty','long_note'):
        row['note_text']='' if damage=='empty' else 'X'*129;row['note_ids']=tok.encode(row['note_text'])
        row['boundary']['note']=dict(correct_text=row['note_text'],correct_ids=row['note_ids'])
    elif damage=='prefix_ids':row['prefix_ids'][0]+=1
    elif damage=='proof_text':row['boundary']['note']['correct_text']='unverified note'
    elif damage=='proof_ids':row['boundary']['note']['correct_ids']=[]
    elif damage=='note_ids':row['note_ids']=[]
    elif damage in ('short_prefix','unfinished_prefix','overwidth'):
        row['prefix_text']={'short_prefix':'3*7 =','unfinished_prefix':'X'*128,'overwidth':'X'*511+'='}[damage]
        row['prefix_ids']=tok.encode(row['prefix_text'])
    else:
        component='source_only'
        if damage=='source_length':row['ordinary_source_ids'].pop()
        elif damage=='source_bool':row['ordinary_source_ids'][10]=True
        else:row['ordinary_source_ids'][10]=1000
    with pytest.raises(ValueError):data.pack_example(row,tok,component)


@pytest.mark.parametrize('bad_tokenizer',[NativeFixture(drop_newline=True),NativeFixture(unsupported=True)])
def test_missing_native_closure_or_undefined_encoded_ids_fail_closed(bad_tokenizer):
    with pytest.raises(ValueError):data.pack_example(example(),bad_tokenizer,'note_only')


def test_missing_only_the_note_closure_is_rejected_after_valid_prompt():
    class MissingClosure(NativeFixture):
        def encode(self,text,**kwargs):
            if text.startswith('Compute ') and text.endswith('\n'):text=text[:-1]
            return super().encode(text,**kwargs)
    tok=MissingClosure();row=example()
    assert tok.decode(data.writer_prompt(row['prefix_text'],tok)).endswith(data.SUFFIX)
    with pytest.raises(ValueError,match='roundtrip'):data.pack_example(row,tok,'note_only')
