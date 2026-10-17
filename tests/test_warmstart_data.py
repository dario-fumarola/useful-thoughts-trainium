"""Synthetic TRAIN labels only: selection semantics, provenance and causal masks."""
import copy
import json

import pytest
import torch

from train import warmstart_data as data
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def fmt():
    # EOS=pad; explicit rendered close has two tokens, one is the stop token.
    return Format((2,), (3,4), 3, 0, header_ids=(17,18), additional_stop_ids=(0,))


def example(note=None, stop='token:0'):
    content = [7,8] if note is None else note
    return dict(item_id='train:0', document_id='doc:0', split='train',
        prefix=([10,11]*256), future=[0]+[12]*15, content_ids=content,
        sampled_actions=content+([] if stop=='length_cap' else [int(stop.split(':')[1])]),
        sampled_stop_type=stop)


def candidate_fixture(ids=('train:0','train:1')):
    p = dict(selection=dict(item_ids=list(ids), split='train'))
    scores, notes = [], {}
    for key in ids:
        def add(interface, mode, index, total):
            row = dict(item_id=key, split='train', interface=interface, mode=mode,
                candidate_index=index, status='ok', token_log_likelihoods=[total]+[0.]*15)
            scores.append(row)
        add('reanchor32','empty',None,-20.)
        add('raw','raw',None,-1.); add('raw','repeat32',None,-2.)
        for i in range(16):
            notes[key,i] = dict(content_ids=[7+i])
            for mode in ('own','filler','shuffled','other'):
                add('reanchor32',mode,i,(-10. if i==2 else -15.) if mode=='own' else -25.)
    return p, scores, notes


def score(scores, key='train:0', mode='own', index=2):
    return next(r for r in scores if r['item_id']==key and r['mode']==mode and r['candidate_index']==index)


def test_winner_is_own_k16_in_frozen_document_order_without_raw_win_requirement():
    p,scores,notes = candidate_fixture()
    # K1 would prefer candidate2, but the sum over16 prefers candidate5.
    score(scores,index=2)['token_log_likelihoods'] = [-.1]+[-1.]*15
    score(scores,index=5)['token_log_likelihoods'] = [-5.]+[0.]*15
    chosen,audit = data.choose_candidates(p,list(reversed(scores)),notes)
    assert chosen == [('train:0',5),('train:1',2)]
    assert audit[0]['contrasts']['own_minus_raw'] == -4.
    assert audit[0]['contrasts']['own_minus_raw_repeat32'] == -3.
    assert audit[0]['selected'] and audit[0]['nonempty']


@pytest.mark.parametrize('mode', ['empty','filler','shuffled','other'])
def test_any_matched_control_tie_or_win_rejects_highest_own_without_fallback(mode):
    p,scores,notes = candidate_fixture(ids=('train:0',))
    # Candidate1 would pass; never fall back after the highest-own candidate fails.
    score(scores,mode=mode,index=None if mode=='empty' else 2)['token_log_likelihoods'] = [-10.]+[0.]*15
    chosen,audit = data.choose_candidates(p,scores,notes)
    assert chosen == [] and audit[0]['candidate_index']==2 and not audit[0]['selected']


def test_empty_winner_and_ties_keep_deterministic_first_index_without_fallback():
    p,scores,notes = candidate_fixture(ids=('train:0',))
    score(scores,index=1)['token_log_likelihoods'] = [-10.]+[0.]*15
    notes['train:0',1]['content_ids'] = []
    chosen,audit = data.choose_candidates(p,scores,notes)
    assert chosen == [] and audit[0]['candidate_index']==1 and not audit[0]['nonempty']


@pytest.mark.parametrize('bad', ['dev_protocol','dev_row','duplicate','failure','nan_unselected_control'])
def test_selection_rejects_dev_duplicates_and_any_failed_or_nonfinite_score(bad):
    p,scores,notes = candidate_fixture()
    if bad=='dev_protocol': p['selection']['split']='dev'
    if bad=='dev_row': scores[0]['split']='dev'
    if bad=='duplicate': scores.append(copy.deepcopy(scores[0]))
    if bad=='failure': score(scores,index=0)['status']='error'
    if bad=='nan_unselected_control': score(scores,mode='filler',index=0)['token_log_likelihoods'][0]=float('nan')
    with pytest.raises(ValueError): data.choose_candidates(p,scores,notes)


@pytest.mark.parametrize('length,stop', [(2,'token:0'),(256,'length_cap')])
def test_packing_labels_only_content_canonical_close_and_future_with_exact_positions(length,stop):
    row = example(note=[7]*length,stop=stop); before=copy.deepcopy(row)
    packed=data.pack_example(row,fmt(),'warmstart')
    ids,positions,labels=(packed[k][0] for k in ('input_ids','target_positions','labels'))
    n,f=packed['note_close_mask'][0],packed['future_mask'][0]
    assert ids.shape==(864,) and positions.shape==(274,)
    assert int(n.sum())==length+2 and int(f.sum())==16 and not (n&f).any()
    prompt=list(fmt().header_ids)+row['prefix']+list(fmt().open_ids)
    rendered=prompt+row['content_ids']+[3,4]+row['prefix'][-32:]+row['future']
    assert ids[:len(rendered)].tolist()==rendered
    assert positions[n].tolist()==list(range(len(prompt),len(prompt)+length+2))
    assert labels[n].tolist()==row['content_ids']+[3,4]
    assert positions[f].tolist()==list(range(len(rendered)-16,len(rendered)))
    assert labels[f].tolist()==row['future'] and labels[f][0]==fmt().pad_id
    assert torch.equal(labels[n|f],ids[positions[n|f]])
    assert labels[~(n|f)].eq(-100).all()
    conditioning=set(range(len(prompt)))|set(range(len(prompt)+length+2,len(rendered)-16))
    assert not set(positions[n|f].tolist()) & conditioning
    assert row==before  # Source EOS/cap provenance remains untouched.


def test_raw_control_uses_identical_selected_future_and_no_note_header_reanchor_labels():
    row=example(); packed=data.pack_example(row,fmt(),'raw_control')
    assert packed['input_ids'].shape==(1,544)
    assert packed['input_ids'][0,:528].tolist()==row['prefix']+row['future']
    assert packed['target_positions'].tolist()==[list(range(512,528))]
    assert packed['labels'].tolist()==[row['future']]
    assert not packed['note_close_mask'].any() and packed['future_mask'].all()


@pytest.mark.parametrize('bad', ['dev','long_note','short_prefix','short_future','embedded_stop'])
def test_packing_rejects_malformed_or_nontrain_supervision(bad):
    row=example()
    if bad=='dev': row['split']='dev'
    if bad=='long_note': row['content_ids']=[7]*257
    if bad=='short_prefix': row['prefix'].pop()
    if bad=='short_future': row['future'].pop()
    if bad=='embedded_stop': row['content_ids']=[7,0]
    with pytest.raises(ValueError): data.pack_example(row,fmt(),'warmstart')


def completed_metadata(tmp_path,monkeypatch):
    """Small fixture replaces heavy corpus validation, never actual outcomes."""
    directory=tmp_path/'corpus'; directory.mkdir()
    p,scores,notes=candidate_fixture(ids=('train:0',))
    p['selection']['records_file']='synthetic-unopened-records'
    (directory/'protocol.json').write_text(json.dumps(p))
    (directory/'global').mkdir()
    barrier=dict(model={'fixture':'B0'})
    (directory/'global/COMPLETE.json').write_text(json.dumps(barrier))
    summary=dict(status='complete',failures=0,rows=17024,
        protocol_sha256=sha256_file(directory/'protocol.json'),scoring=[])
    for shard in range(8):
        d=directory/'scoring'/f'shard-{shard:02d}';d.mkdir(parents=True)
        (d/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores) if shard==0 else '')
        run=dict(status='complete',schema=data.corpus.SCHEMA,shard=shard,
            protocol_sha256=summary['protocol_sha256'],
            barrier_sha256=sha256_file(directory/'global/COMPLETE.json'),model=barrier['model'],checkpoint=None)
        (d/'run.json').write_text(json.dumps(run))
        summary['scoring'].append(dict(shard=shard,run_sha256=sha256_file(d/'run.json'),
                                      scores_sha256=sha256_file(d/'scores.jsonl')))
    (directory/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(data.corpus,'read_protocol',lambda _: (p,fmt()))
    monkeypatch.setattr(data.corpus,'global_inputs',lambda *a:(notes,{},barrier))
    monkeypatch.setattr(data.corpus,'reduce_rows',lambda *a:dict(failures=0,rows=17024))
    record=example(); record['document_id']='doc:0'
    notes['train:0',2].update(prefix_sha256=canonical_hash(record['prefix']),
                             actions=[9,0],stop_type='token:0')
    return directory,summary,record


@pytest.mark.parametrize('bad', ['incomplete','protocol','barrier','model'])
def test_preparation_refuses_wrong_completed_corpus_before_canonical_future_read(tmp_path,monkeypatch,bad):
    directory,summary,_=completed_metadata(tmp_path,monkeypatch)
    if bad=='incomplete': summary['status']='running'
    elif bad=='protocol': summary['protocol_sha256']='0'*64
    else:
        d=directory/'scoring/shard-00'; run=json.loads((d/'run.json').read_text())
        if bad=='barrier': run['barrier_sha256']='0'*64
        else: run['model']={'fixture':'different'}
        (d/'run.json').write_text(json.dumps(run))
        summary['scoring'][0]['run_sha256']=sha256_file(d/'run.json')
    (directory/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(data,'load_selected',lambda *a,**k:pytest.fail('Invalid provenance read futures'))
    with pytest.raises(ValueError): data.prepare(tmp_path/'out',directory)
    assert not (tmp_path/'out').exists()


def test_prepared_dataset_keeps_exact_same_ordered_subset_for_both_arms(tmp_path,monkeypatch):
    directory,_,record=completed_metadata(tmp_path,monkeypatch)
    monkeypatch.setattr(data,'load_selected',lambda *a,**k:[record])
    output=tmp_path/'out';data.prepare(output,directory)
    rows=[json.loads(line) for line in (output/'examples.jsonl').read_text().splitlines()]
    manifest=json.loads((output/'dataset.json').read_text())
    assert [r['item_id'] for r in rows]==['train:0'] and rows[0]['source_candidate_index']==2
    assert rows[0]['sampled_actions']==[9,0] and rows[0]['sampled_stop_type']=='token:0'
    assert rows[0]['content_ids']==[9] and rows[0]['split']=='train'
    assert manifest['examples_sha256']==sha256_file(output/'examples.jsonl')
    assert manifest['item_ids_sha256']==canonical_hash(['train:0'])
    for arm in ('warmstart','raw_control'):
        packed=data.pack_example(rows[0],fmt(),arm)
        assert packed['labels'][packed['future_mask']].tolist()==record['future']
