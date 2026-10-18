"""CPU-only packing semantics; no corpus preparation, model or training evidence."""
import copy
from datetime import datetime, timedelta, timezone
import json

import pytest
import torch

from train import calculation_warmstart_data as data
from weekend_thinking.artifacts import canonical_hash, sha256_file


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


class ReaderFixture(NativeFixture):
    def get_vocab(self):return {**super().get_vocab(),'double-newline':271}
    def encode(self,text,**kwargs):
        parts=text.split('\n\n');result=[]
        for i,part in enumerate(parts):
            if i:result.append(271)
            result.extend(super().encode(part,**kwargs))
        return result
    def decode(self,ids,**kwargs):
        return ''.join('\n\n' if t==271 else super(ReaderFixture,self).decode([t],**kwargs) for t in ids)


def reader_example(tok=None,*,item='item',doc='doc'):
    tok=tok or ReaderFixture();row=example(tok);row.update(item_id=item,document_id=doc,
        writer_prefix_ids=list(row['prefix_ids']),writer_prefix_text=row['prefix_text'])
    row['boundary']['expression']='(3*7)+4';row['corrupted_note_ids']=tok.encode('Compute 3*7 = 22.')
    cut=len(row['prefix_ids']);row['natural_future16_ids']=row['ordinary_source_ids'][cut:cut+16]
    return row


def test_reader_exact_shared_context_and_only16_causal_targets_with_real_eos():
    from train.calculation_boundary_probe import make_contexts
    tok=ReaderFixture();row=reader_example(tok);cut=len(row['prefix_ids'])
    row['ordinary_source_ids'][cut+8]=data.PAD;row['natural_future16_ids']=row['ordinary_source_ids'][cut:cut+16]
    expected=make_contexts([row],interface='after_note_reanchor_lhs',tokenizer=tok)[row['item_id']]['values']['correct']
    assert data.reader_context(row,tok)==expected
    packed=data.pack_example(row,tok,'reader_only');assert_ce_metadata(packed,1024,16)
    assert not packed['note_close_mask'].any() and packed['future_mask'].all()
    assert packed['target_positions'].tolist()==[list(range(len(expected),len(expected)+16))]
    assert packed['labels'].tolist()==[row['natural_future16_ids']]
    assert packed['labels'][0,8]==data.PAD  # A real source EOS stays a target.
    assert packed['input_ids'][0,:len(expected)].tolist()==expected
    assert packed['input_ids'][0,len(expected):len(expected)+16].tolist()==row['natural_future16_ids']
    assert packed['input_ids'][0,len(expected)+16:].tolist()==[data.PAD]*(1024-len(expected)-16)


def test_reader_context_never_reads_future_rhs_or_normalized_expression():
    tok=ReaderFixture();row=reader_example(tok);expected=data.reader_context(row,tok)
    row.pop('natural_future16_ids');row.pop('ordinary_source_ids');row.pop('verified_numeric_rhs')
    row['boundary']['normalized_expression']='WRONG normalized expression must not be used'
    assert data.reader_context(row,tok)==expected
    # Source bytes AFTER the16 target tokens cannot enter the packed reader.
    row=reader_example(tok);before=data.pack_example(row,tok,'reader_only');cut=len(row['prefix_ids'])
    row['ordinary_source_ids'][cut+16:]=[88]*(528-cut-16)
    after=data.pack_example(row,tok,'reader_only')
    assert all(torch.equal(before[k],after[k]) for k in before)


@pytest.mark.parametrize('damage',['short16','target_mismatch','prefix_alias','note_proof','normalized_lhs','overwidth'])
def test_reader_rejects_changed_target_context_or_overwidth_without_truncation(damage):
    tok=ReaderFixture();row=reader_example(tok)
    if damage=='short16':row['natural_future16_ids'].pop()
    elif damage=='target_mismatch':row['natural_future16_ids'][0]=88
    elif damage=='prefix_alias':row['writer_prefix_text']='Changed causal context='
    elif damage=='note_proof':row['boundary']['note']['correct_ids']=[]
    elif damage=='normalized_lhs':row['boundary']['expression']='(3 * 7) + 4'
    else:
        lhs='1+'*255+'1';past=lhs+'=';row.update(prefix_text=past,prefix_ids=tok.encode(past),writer_prefix_text=past,writer_prefix_ids=tok.encode(past))
        row['boundary']['expression']=lhs;row['ordinary_source_ids']=row['prefix_ids']+[32]*16;row['natural_future16_ids']=[32]*16
    with pytest.raises(ValueError):data.pack_example(row,tok,'reader_only')


def preparation_fixture(tmp_path,monkeypatch):
    import transformers
    tok=ReaderFixture();directory=tmp_path/'dataset';directory.mkdir();source=tmp_path/'source';source.write_bytes(b'Frozen original CPU fixture bytes\n')
    rows=[reader_example(tok,item='item'+str(i),doc='doc'+str(i)) for i in range(3)]
    lines={r['item_id']:json.dumps(r,separators=(', ',': '))+'\n' for r in rows};accepted=directory/'accepted.jsonl'
    accepted.write_text(''.join(lines.values()));curator='train/calculation_boundaries.py'
    manifest=dict(schema='finemath-arithmetic-boundaries-train-dev-v2',accepted_sha256=sha256_file(accepted),
        source_file=str(source),source_sha256=sha256_file(source),source_files={curator:sha256_file(data.Path(__file__).resolve().parents[1]/curator)},
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),selection={'train':dict(selected_item_ids=['item1'],reserve_item_ids=['item0','item2'])})
    original=directory/'manifest.json';original.write_text(json.dumps(manifest));ids=['item1','item0'];docs=['doc1','doc0']
    planned=dict(status='SEALED_PROSPECTIVE_IDS_NOT_TRAINING_RELEASE',count=2,split='train',ordered_item_ids=ids,ordered_item_ids_sha256=canonical_hash(ids),
        ordered_document_ids=docs,ordered_document_ids_sha256=canonical_hash(docs),manifest_file=str(original),manifest_sha256=sha256_file(original),
        accepted_file=str(accepted),accepted_sha256=sha256_file(accepted),curator_source_files=manifest['source_files'],source_file=str(source),source_sha256=sha256_file(source))
    prospective=tmp_path/'ids.json';prospective.write_text(json.dumps(planned))
    monkeypatch.setattr(data,'TRAIN_ITEMS',2);monkeypatch.setattr(data,'PROSPECTIVE_IDS_SHA256',sha256_file(prospective))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    monkeypatch.setattr(data,'preparation_sources',lambda:{'train/calculation_warmstart_data.py':sha256_file(data.__file__)})
    return directory,prospective,tok,rows,lines


def test_preparation_keeps_whole_row_bytes_and_sealed_order_with_component_totals(tmp_path,monkeypatch):
    directory,ids,tok,rows,lines=preparation_fixture(tmp_path,monkeypatch);output=tmp_path/'prepared'
    deadline=(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat()
    manifest=data.prepare_training512(directory,ids,tmp_path/'fake-tokenizer',output,deadline)
    assert (output/'examples.jsonl').read_text()==lines['item1']+lines['item0']
    assert manifest['items']==2 and manifest['split']=='train' and manifest['no_filter_or_replacement']
    assert manifest['component_label_totals']['source_only']==2*527
    assert manifest['component_label_totals']['reader_only']==2*16
    assert manifest['component_label_totals']['note_only']==sum(len(tok.encode(r['note_text']+'\n')) for r in rows[:2])
    assert manifest['component_counts']['reader_only']==dict(examples=2,active_labels=32,note_close_labels=0,future_labels=32)
    assert manifest['examples_sha256']==sha256_file(output/'examples.jsonl')


@pytest.mark.parametrize('damage',['sealed_bytes','accepted_bytes','original_source_bytes','reordered_ids','missing_id'])
def test_preparation_rejects_identity_drift_instead_of_filtering_or_replacing(tmp_path,monkeypatch,damage):
    directory,ids,tok,rows,lines=preparation_fixture(tmp_path,monkeypatch);output=tmp_path/'prepared';planned=json.loads(ids.read_text())
    if damage=='sealed_bytes':ids.write_text(ids.read_text()+' ')
    elif damage=='accepted_bytes':(directory/'accepted.jsonl').write_text((directory/'accepted.jsonl').read_text()+' ')
    elif damage=='original_source_bytes':data.Path(planned['source_file']).write_bytes(b'changed original')
    else:
        planned['ordered_item_ids']=['item0','item1'] if damage=='reordered_ids' else ['item1','missing']
        planned['ordered_item_ids_sha256']=canonical_hash(planned['ordered_item_ids']);ids.write_text(json.dumps(planned));monkeypatch.setattr(data,'PROSPECTIVE_IDS_SHA256',sha256_file(ids))
    with pytest.raises(ValueError):data.prepare_training512(directory,ids,tmp_path/'fake-tokenizer',output,(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat())
    assert not (output/'manifest.json').exists()


def test_preparation_packing_failure_retains_partial_and_does_not_use_reserve(tmp_path,monkeypatch):
    directory,ids,tok,rows,lines=preparation_fixture(tmp_path,monkeypatch);output=tmp_path/'prepared'
    # This pins an intentionally malformed synthetic accepted publication; the
    # second sealed item fails packing even though a good reserve item exists.
    rows[0]['note_text']='';lines['item0']=json.dumps(rows[0])+'\n';accepted=directory/'accepted.jsonl';accepted.write_text(''.join(lines.values()))
    m=json.loads((directory/'manifest.json').read_text());m['accepted_sha256']=sha256_file(accepted);(directory/'manifest.json').write_text(json.dumps(m))
    planned=json.loads(ids.read_text());planned.update(accepted_sha256=sha256_file(accepted),manifest_sha256=sha256_file(directory/'manifest.json'));ids.write_text(json.dumps(planned));monkeypatch.setattr(data,'PROSPECTIVE_IDS_SHA256',sha256_file(ids))
    with pytest.raises(ValueError):data.prepare_training512(directory,ids,tmp_path/'fake-tokenizer',output,(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat())
    assert not (output/'manifest.json').exists() and (output/'preparation.failed.json').is_file()
    assert (output/'examples.jsonl').read_text()==lines['item1']
