"""Scientific selection, exact source provenance and four-component masks."""
import copy
import json

import pytest
import torch

from train import prepare_reader_learning_data as data
from train import useful_note_sft_data as packing
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Tokenizer:
    eos_token_id = 151643

    def encode(self, text, **kwargs):
        return list(text.encode('ascii'))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode('ascii')


def candidate(key, doc, *, ops=2, representative=True, cut=140):
    expression = '2+3*4' if ops == 2 else '3*4'
    value = '14' if ops == 2 else '12'
    prefix = ('Original mathematical context. ' * 8)[:cut-len(expression)-2] + expression + ' ='
    original = (prefix+' '+value+'.\nMore ordinary source text.').ljust(528, 'x')
    ids = list(original.encode('ascii')); past = list(prefix.encode('ascii'))
    source = dict(item_id='original-'+key, document_id=doc, split='train',
        boundary=dict(expression=expression), numeric_value=value, numeric_surface_text=value,
        original_source_ids=ids, original_source_text=original, prefix_ids=past, prefix_text=prefix,
        natural_future16_ids=ids[len(past):len(past)+16], target_ids=ids[len(past):len(past)+4],
        numeric_token_mask=[False,True,True,False], reanchor_ids=list((expression+' =').encode()),
        reanchor_text=expression+' =')
    wrapper = dict(candidate_id=key, document_id=doc, split='train', absolute_source_cut=cut,
        binary_ops=ops, representative=representative, source_payload=source,
        source_payload_sha256=canonical_hash(source))
    row = dict(source, item_id=key, original_item_id=source['item_id'], absolute_source_cut=cut,
        teacher_note_kind='exact-postorder-v1',
        candidate_provenance={k:v for k,v in wrapper.items() if k!='source_payload'})
    row['teacher_note_text'] = packing.symbolic_note(row)
    row['teacher_note_ids'] = Tokenizer().encode(row['teacher_note_text'])
    return row, wrapper


def publication(tmp_path, rows):
    p=tmp_path/'candidates.jsonl'
    p.write_bytes(b''.join((json.dumps(r, separators=(', ', ': '))+'\n').encode() for r in rows))
    return p


def test_selection_is_publication_order_not_gain_or_other_cuts(tmp_path):
    one,_=candidate('one','D0',ops=1)
    extra,_=candidate('extra','D1',representative=False)
    a,_=candidate('first','D1',cut=141); b,_=candidate('second','D2')
    a['diagnostic_gain']=-100; b['diagnostic_gain']=100
    p=publication(tmp_path,[one,extra,a,b]); original=p.read_bytes().splitlines(keepends=True)
    selected,inventory=data.select_representatives(p,2)
    assert [r['item_id'] for _,_,r in selected]==['first','second']
    assert [line for _,line,_ in selected]==original[2:]
    assert inventory['selected_publication_lines']==[3,4]
    assert inventory['eligible_compound_representatives']==2
    a['diagnostic_gain']=1000; b['diagnostic_gain']=-1000
    p=publication(tmp_path,[one,extra,a,b])
    again,_=data.select_representatives(p,2)
    assert [r['item_id'] for _,_,r in again]==['first','second']


def test_shortfall_cannot_use_one_op_or_nonrepresentative_fallback(tmp_path):
    a,_=candidate('a','A'); b,_=candidate('b','B',representative=False); c,_=candidate('c','C',ops=1)
    with pytest.raises(ValueError,match='shortfall.*no fallback'):
        data.select_representatives(publication(tmp_path,[a,b,c]),2)


@pytest.mark.parametrize('change', ['repeat_doc','repeat_cut','math','provenance','split'])
def test_full_inventory_rejects_invalid_rows_even_after_quota(tmp_path,change):
    a,_=candidate('a','A'); b,_=candidate('b','B')
    if change=='repeat_doc':
        b['document_id']='A';b['candidate_provenance']['document_id']='A'
        b['absolute_source_cut']=141;b['candidate_provenance']['absolute_source_cut']=141
    elif change=='repeat_cut':
        b['document_id']='A';b['candidate_provenance']['document_id']='A'
    elif change=='math':b['numeric_value']='99'
    elif change=='provenance':b['candidate_provenance']['candidate_id']='elsewhere'
    else:b['split']='dev';b.pop('candidate_provenance') # split rejected before payload access
    with pytest.raises(ValueError):data.select_representatives(publication(tmp_path,[a,b]),1)


def test_exact_original_source_payload_is_verified_not_just_math(tmp_path):
    row,wrapper=candidate('a','A'); selected=[(1,b'original bytes',row)]
    union=tmp_path/'union.jsonl';union.write_text(json.dumps(wrapper)+'\n')
    data.verify_source_rows(selected,union)
    altered=copy.deepcopy(row);altered['natural_future16_ids'][-1]+=1
    with pytest.raises(ValueError,match='changed original source'):
        data.verify_source_rows([(1,b'',altered)],union)
    wrapper['source_payload']['prefix_text']='rewritten'
    union.write_text(json.dumps(wrapper)+'\n')
    with pytest.raises(ValueError,match='payload hash'):
        data.verify_source_rows(selected,union)


def test_upstream_missing_selected_candidate_is_not_replaced(tmp_path):
    row,wrapper=candidate('a','A');union=tmp_path/'union.jsonl';union.write_text('')
    with pytest.raises(ValueError,match='missing'):
        data.verify_source_rows([(1,b'',row)],union)


def test_native_serialization_future_and_symbolic_note_must_remain_exact():
    row,_=candidate('a','A');tok=Tokenizer()
    assert data.validate_native(row,tok)==row['teacher_note_ids']
    for field in ('teacher_note_text','prefix_text','original_source_text','reanchor_text'):
        altered=copy.deepcopy(row);altered[field]+=' '
        with pytest.raises(ValueError):data.validate_native(altered,tok)
    altered=copy.deepcopy(row);altered['natural_future16_ids'][-1]+=1
    with pytest.raises(ValueError):data.validate_native(altered,tok)


def test_all_four_real_causal_components_have_declared_label_roles():
    row,_=candidate('a','A');tok=Tokenizer();parts=dict(old_open=[198,151667],old_close=[151668,198])
    shapes,counts,lengths=data.measure_packing([row],tok,parts)
    assert counts['source']['active_labels']==527
    assert counts['reader']['active_labels']==16
    assert counts['writer']['note_close_labels']==len(row['teacher_note_ids'])+1
    assert counts['writer']['non_note_labels']==0
    assert counts['augmented']['active_labels']>527
    for c in packing.COMPONENTS:
        assert counts[c]['examples']==1
        assert counts[c]['note_close_labels']+counts[c]['non_note_labels']==counts[c]['active_labels']
    original=packing.pack_example(row,tok,'writer',interface='I3',parts=parts,**shapes['writer'])
    changed=copy.deepcopy(row);cut=len(row['prefix_ids'])
    changed['original_source_ids'][cut:]=[ord('z')]*(528-cut)
    changed['natural_future16_ids']=[ord('z')]*16
    alternate=packing.pack_example(changed,tok,'writer',interface='I3',parts=parts,**shapes['writer'])
    assert all(torch.equal(original[k],alternate[k]) for k in original)
    assert lengths['max']<=128


def test_changed_input_or_expired_deadline_fails_closed(tmp_path):
    path=tmp_path/'pinned';path.write_text('original');digest=sha256_file(path)
    assert data.pinned(path,digest)==path.resolve()
    path.write_text('changed')
    with pytest.raises(ValueError,match='Immutable input'):data.pinned(path,digest)
    with pytest.raises(TimeoutError,match='partial evidence retained'):
        data.check_deadline('2000-01-01T00:00:00Z')
