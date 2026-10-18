"""Fixed token-only truncation, legacy provenance bridge and full denominators."""
import copy
import json
from pathlib import Path

import pytest

from train import firstline_note_probe as e
from weekend_thinking.artifacts import sha256_file,canonical_hash

class Tokenizer:
    def __init__(self,pieces):self.pieces=pieces
    def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)
    def encode(self,*args,**kwargs):raise AssertionError('Truncation must never re-encode or search a target')


def test_leading_whitespace_and_entire_merged_newline_token_are_preserved():
    t=Tokenizer({1:'\n\n  ',2:'A calculation',3:' = 4.\nNext article',4:' ignored'})
    out=e.truncate_content([1,2,3,4],t)
    assert out['content_ids']==[1,2,3]
    assert out['content_text']=='\n\n  A calculation = 4.\nNext article'
    assert out['boundary_token_index']==2 and out['reason']=='completed_content_line'


def test_first_nonwhitespace_line_not_first_blank_line_and_no_numeric_search():
    t=Tokenizer({1:' \n\t',2:'words without arithmetic',3:'\n',4:'1234'})
    assert e.truncate_content([1,2,3,4],t)['content_ids']==[1,2,3]
    t=Tokenizer({1:'999',2:'words',3:'\n',4:'target not examined'})
    assert e.truncate_content([1,2,3,4],t)['content_ids']==[1,2,3]


def test_cap_empty_and_original_short_notes_stay_exact_native_prefixes():
    t=Tokenizer({1:'x',2:'\n',3:' '})
    out=e.truncate_content([1]*128,t)
    assert out['content_ids']==[1]*64 and out['reason']=='content_cap'
    assert e.truncate_content([],t)['content_ids']==[]
    assert e.truncate_content([3,2,1],t)['content_ids']==[3,2,1]
    assert e.truncate_content([3,2,1],t)['reason']=='original_end'
    with pytest.raises(ValueError):e.truncate_content([1,True],t)


def test_only_unused_trained_identity_function_can_differ_in_legacy_bridge():
    old='CONST=1\ndef trained_identity(x): return x\ndef complete_notes(x): return x+1\n'
    allowed='CONST=1\ndef trained_identity(x): return (x, "Cprime")\ndef complete_notes(x): return x+1\n'
    assert e.same_data_api(old,allowed)
    assert not e.same_data_api(old,allowed.replace('return x+1','return x+2'))
    assert not e.same_data_api(old,allowed.replace('CONST=1','CONST=2'))


def test_archived_source_bytes_and_actual_data_bytes_are_both_required(tmp_path):
    source=tmp_path/'source.py';source.write_text('old source')
    saved=tmp_path/'payload/source.py';saved.parent.mkdir();saved.write_bytes(source.read_bytes())
    digest=sha256_file(source);archive=dict(entries=[dict(source=str(source.resolve()),relative='payload/source.py',sha256=digest)])
    assert e.archive_entry(tmp_path/'ARCHIVE.json',archive,source,digest)==saved
    source.write_text('new identity function')
    assert e.archive_entry(tmp_path/'ARCHIVE.json',archive,source,digest,current=False)==saved
    with pytest.raises(ValueError):e.archive_entry(tmp_path/'ARCHIVE.json',archive,source,digest)
    saved.write_text('changed archived proof')
    with pytest.raises(ValueError):e.archive_entry(tmp_path/'ARCHIVE.json',archive,source,digest,current=False)


def test_archive_path_cannot_escape_closed_inventory(tmp_path):
    f=tmp_path/'source';f.write_text('data');h=sha256_file(f)
    archive=dict(entries=[dict(source=str(f.resolve()),relative='../elsewhere',sha256=h)])
    with pytest.raises(ValueError,match='escaped'):e.archive_entry(tmp_path/'ARCHIVE.json',archive,f,h)


def test_global_donors_and_every_context_change_only_supplied_note_prefix():
    row=dict(item_id='a',prefix_ids=[10,11],reanchor_ids=[11]);parts=dict(old_open=[20],old_close=[30])
    notes={'a':dict(content_ids=[1,2,3]),'b':dict(content_ids=[])}
    shorts={'a':dict(content_ids=[1]),'b':dict(content_ids=[])}
    donors={'a':'b','b':'a'}
    ctx=e.make_contexts(row,parts,notes,shorts,donors)
    assert ctx['raw']==[10,11]
    assert ctx['full_own']==[10,11,20,1,2,3,30,11]
    assert ctx['short_own']==[10,11,20,1,30,11]
    assert ctx['full_other']==ctx['short_other']==ctx['empty']==[10,11,20,30,11]
    other=e.make_contexts(dict(row,item_id='b'),parts,notes,shorts,donors)
    assert other['short_other']==ctx['short_own']
    assert notes['a']['content_ids']==[1,2,3] #original full notes immutable


def fixture_values():
    ids=['a','b','c'];docs=['D1','D2','D3'];values=[]
    for key,doc in zip(ids,docs):
        for path in e.PATHS:
            for mode in e.MODES:
                value=2. if mode=='short_own' else 0.
                values.append(dict(item_id=key,document_id=doc,path=path,mode=mode,status='ok',
                    **{m:value for m in e.parent.METRICS}))
    return ids,docs,values


def test_both_primary_paired_document_contrasts_required_and_directions_correct():
    ids,docs,values=fixture_values();result=e.reduce_rows(values,ids,docs)
    for path in e.PATHS:
        assert result[path]['exploratory_primary_pass']
        for comparison in ('short_own_minus_raw','short_own_minus_short_other'):
            c=result[path]['contrasts'][comparison+'/natural16_ll']
            assert c['difference_b_minus_a']==2 and c['ci95']==[2,2] and c['documents']==3
    for v in values:
        if v['mode']=='short_other':v.update(**{m:3. for m in e.parent.METRICS})
    assert not any(v['exploratory_primary_pass'] for v in e.reduce_rows(values,ids,docs).values())


def test_error_blocks_complete_pair_statistics_but_keeps_all_item_denominators():
    ids,docs,values=fixture_values()
    v=next(v for v in values if v['path']=='I3' and v['mode']=='short_other')
    v.clear();v.update(item_id='a',document_id='D1',path='I3',mode='short_other',status='error')
    result=e.reduce_rows(values,ids,docs)['I3']
    assert not result['exploratory_primary_pass']
    assert result['absolute']['short_other/natural16_ll']==dict(items=3,errors=1,mean=None)
    assert result['contrasts']['short_own_minus_short_other/natural16_ll']['available'] is False
    assert result['absolute']['raw/natural16_ll']['items']==3


@pytest.mark.parametrize('defect',['missing','duplicate','wrong_doc','status','nonfinite'])
def test_matrix_validator_never_filters_or_ignores_bad_paths(defect):
    ids,docs,values=fixture_values()
    if defect=='missing':values.pop()
    elif defect=='duplicate':values[-1]=values[0]
    elif defect=='wrong_doc':values[0]['document_id']='wrong'
    elif defect=='status':values[0]['status']='ignored'
    else:values[0]['natural16_ll']=float('nan')
    with pytest.raises(ValueError):e.reduce_rows(values,ids,docs)


def test_all_shard_receipts_checked_before_any_new_outcome_vectors_are_parsed(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'SHARDS',2)
    p=dict(ordered_item_ids=['a','b'],ordered_document_ids=['A','B'])
    x=dict(rows=[dict(item_id='a'),dict(item_id='b')])
    monkeypatch.setattr(e,'load',lambda output:(p,x))
    (tmp_path/'protocol.json').write_text('{}');d=tmp_path/'scores/shard-0';d.mkdir(parents=True)
    (d/'scores.jsonl').write_text('not opened as outcomes')
    (d/'run.json').write_text(json.dumps(dict(status='complete_attempts',rows=12,
        protocol_sha256=sha256_file(tmp_path/'protocol.json'),scores_sha256=sha256_file(d/'scores.jsonl'),
        reader=dict(id='B0',checkpoint=None))))
    monkeypatch.setattr(e.parent,'rows',lambda _:pytest.fail('Partial outcome vector opened'))
    with pytest.raises(FileNotFoundError):e.summarize(tmp_path)


def test_original_generation_costs_are_not_replaced_by_kept_token_counts():
    notes={'a':dict(actions=[1]*128,content_ids=[1]*128,wall_seconds=12.)}
    values=[dict(timing_group='a/raw',wall_seconds=2.,shared_raw_forward_seconds=1.,shared_by_paths=list(e.PATHS)) for _ in e.PATHS]
    result=e.parent.saved_costs(notes,values)
    assert result['total_generated_actions']==128 and result['note_generation_seconds_sum']==12.
    assert result['scoring_cell_seconds_sum']==2. #not double-counted acrossI3/I6
