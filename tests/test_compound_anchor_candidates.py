"""Selection/provenance and fail-closed materialization of already-curated rows."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from train import compound_anchor_candidates as data
from train import computable_equations as base
from train import computable_equations_supervised as supervised
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Characters:
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)


def examples(doc='doc', expressions=('2*3+4 = 10;',), raw=False):
    text = 'Context words. '*12+'\nCalculation: '+('\nMore context: '.join(expressions))+'\n tail'*120
    tok = Characters(); original = tok.encode(text)[:528]
    assert len(original) == 528
    record = dict(id=doc+':512', document_id=doc, split='train', prefix=original[:512], future=original[512:])
    result = []
    for cut, char in enumerate(text[:512], 1):
        if char != '=': continue
        chosen = supervised.candidate_at_cut(original[:cut], cut, tok)
        row, status = base.curate(record, chosen, tok, 'fixture-source', 'fixture-source-sha')
        assert status['status'] == 'accepted'
        row['item_id'] = f'{doc}:fixture:{cut}'
        if raw:
            row.update(absolute_source_cut=cut, source_chunk_token_start=0,
                       serialization_variant='raw', item_id=f'{doc}:raw-fixture:{cut}')
        result.append(row)
    return result


def write_rows(path, rows, opaque_nontrain=False):
    with path.open('wb') as stream:
        if opaque_nontrain:
            stream.write(b'{"item_id": "dev", "document_id": "dev-doc", "split": "dev", INVALID FUTURE\n')
            stream.write(b'{"item_id": "test", "document_id": "test-doc", "split": "test", INVALID FUTURE\n')
        for row in rows: stream.write((json.dumps(row)+'\n').encode())


def inputs(tmp_path, old, raw, opaque_nontrain=False):
    paths=[]
    for index, rows in enumerate((old, raw)):
        path=tmp_path/f'rows-{index}.jsonl';write_rows(path,rows,opaque_nontrain)
        spec={**data.SPECS[index], 'accepted_sha256':sha256_file(path)}
        paths.append((path,spec))
    return paths


def test_nontrain_payloads_are_opaque_before_json_decode(tmp_path):
    row=examples()[0];paths=inputs(tmp_path,[row],[],opaque_nontrain=True)
    actual, representatives, audit, duplicates=data.assemble(paths)
    assert len(actual)==len(representatives)==len(audit)==1 and not duplicates
    assert actual[0]['source_payload']==row
    assert audit[0]['line_number']==3


def test_same_cut_dedup_preserves_all_origins_and_prefers_raw_in_either_input_order(tmp_path):
    old=examples()[0];raw=examples(raw=True)[0]
    paths=inputs(tmp_path,[old],[raw])
    for sequence in (paths,list(reversed(paths))):
        rows,ids,audit,duplicates=data.assemble(sequence)
        assert len(rows)==len(ids)==len(duplicates)==1 and len(audit)==2
        assert rows[0]['source_payload']==raw and rows[0]['source_payload_sha256']==canonical_hash(raw)
        assert len(rows[0]['origins'])==2 and rows[0]['retained_publication']==data.SPECS[1]['publication']
        for origin in rows[0]['origins']:
            line=(paths[0][0] if origin['publication']==data.SPECS[0]['publication'] else paths[1][0]).read_bytes()
            assert origin['original_line_sha256']==data.hashlib.sha256(line).hexdigest()


def test_same_cut_native_future_disagreement_is_not_silently_resolved_by_priority(tmp_path):
    old=examples()[0];raw=examples(raw=True)[0]
    raw['original_source_ids'][raw['boundary']['cut']+15]+=1
    raw['natural_future16_ids'][-1]+=1
    with pytest.raises(ValueError,match='Same native cut disagrees'):
        data.assemble(inputs(tmp_path,[old],[raw]))


def test_changed_native_target_is_rejected_before_publication(tmp_path):
    row=examples()[0];row['natural_future16_ids'][0]+=1
    with pytest.raises(ValueError,match='Natural16'):
        data.assemble(inputs(tmp_path,[row],[]))


def test_closing_certificate_mismatch_is_rejected(tmp_path):
    row=examples()[0];row['closing_boundary_ids']=[]
    with pytest.raises(ValueError,match='boundary identity'):
        data.assemble(inputs(tmp_path,[row],[]))


def test_distinct_source_cuts_in_one_window_remain_distinct_examples(tmp_path):
    old=examples(expressions=('2*3+4 = 10;','2*4+3 = 11;'))
    before=deepcopy(old);rows,ids,audit,duplicates=data.assemble(inputs(tmp_path,old,[]))
    assert len(rows)==2 and len(ids)==1 and len(audit)==2 and not duplicates
    assert len({r['absolute_source_cut'] for r in rows})==2
    assert old==before
    groups=data.duplicate_groups(rows,lambda r:canonical_hash(r['source_payload']['original_source_ids']))
    assert len(groups)==1 and len(groups[0]['candidate_ids'])==2


def test_order_and_representative_depend_on_frozen_hashes_not_input_order(tmp_path):
    old=examples('a',('2*3+4 = 10;','2*4+3 = 11;'))+examples('b')
    rows,ids,_,_=data.assemble(inputs(tmp_path,old,[]))
    other,other_ids,_,_=data.assemble(inputs(tmp_path,list(reversed(old)),[]))
    assert [r['candidate_id'] for r in rows]==[r['candidate_id'] for r in other] and ids==other_ids
    for doc in ('a','b'):
        selected=[r for r in rows if r['document_id']==doc]
        assert selected[0]['representative'] and sum(r['representative'] for r in selected)==1
        assert selected[0]['candidate_hash']==min(r['candidate_hash'] for r in selected)


def test_compound_exclusion_is_explicit_without_replacing_document(tmp_path):
    one=examples('one',('6+4 = 10;',))
    rows,ids,audit,_=data.assemble(inputs(tmp_path,one+examples('two'),[]))
    assert len(rows)==len(ids)==1
    assert CounterLike(audit)=={'one_op_not_in_compound_inventory':1,'candidate_input':1}


def CounterLike(rows):
    return {status:sum(r['status']==status for r in rows) for status in {r['status'] for r in rows}}


def test_all_op_fallback_keeps_every_cut_and_prefers_compound_per_document(tmp_path):
    old=examples('mixed',('6+4 = 10;','2*4+3 = 11;'))+examples('one',('4+5 = 9;',))
    paths=inputs(tmp_path,old,[])
    compound,cids,_,_=data.assemble(paths)
    rows,ids,audit,_=data.assemble(paths,compound_priority=True)
    assert len(compound)==1 and len(rows)==3 and len(ids)==2
    reps=[r for r in rows if r['representative']]
    assert [(r['document_id'],r['binary_ops']) for r in reps]==[('mixed',2),('one',1)]
    assert compound[0]['candidate_hash']==reps[0]['candidate_hash']
    assert all(r['status']=='candidate_input' for r in audit)


def test_absolute_cut_uses_original_anchor_offset_without_reconstructing_source():
    row=examples()[0];row['source_record_id']='doc:1512'
    assert data.absolute_cut(row,data.SPECS[0]['publication'])==1000+row['boundary']['cut']
    row['source_record_id']='another-doc:1512'
    with pytest.raises(ValueError,match='native anchor identity'):
        data.absolute_cut(row,data.SPECS[0]['publication'])


def test_raw_cut_must_match_unchanged_window_and_local_cut():
    row=examples(raw=True)[0];row['absolute_source_cut']+=1
    with pytest.raises(ValueError,match='absolute/local native cuts'):
        data.absolute_cut(row,data.SPECS[1]['publication'])


def pinned_directories(tmp_path):
    specs=[];directories=[]
    for index,rows in enumerate((examples(),examples(raw=True))):
        directory=tmp_path/f'publication-{index}';directory.mkdir();directories.append(directory)
        accepted=directory/'accepted.jsonl';write_rows(accepted,rows,opaque_nontrain=True)
        manifest=dict(schema=data.SPECS[index]['publication'],accepted_sha256=sha256_file(accepted))
        (directory/'manifest.json').write_text(json.dumps(manifest))
        specs.append({**data.SPECS[index], 'manifest_sha256':sha256_file(directory/'manifest.json'),
                      'accepted_sha256':sha256_file(accepted)})
    return directories,specs


def deadline():return (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()


def test_complete_materialization_binds_exact_files_and_keeps_inputs_unchanged(tmp_path):
    directories,specs=pinned_directories(tmp_path);out=tmp_path/'output'
    manifest=data.prepare(*directories,out,deadline(),specs=specs,expected=(1,1))
    assert manifest['candidates']==manifest['documents']==1 and manifest['same_cut_duplicates']==1
    assert (out/'closure.json').exists() and not (out/'materialization.failed.json').exists()
    for name,entry in manifest['output_files'].items():assert sha256_file(out/name)==entry['sha256']
    row=json.loads((out/'candidates.jsonl').read_text())
    assert row['source_payload']==examples(raw=True)[0]
    for directory,spec in zip(directories,specs):assert sha256_file(directory/'accepted.jsonl')==spec['accepted_sha256']
    with pytest.raises(FileExistsError):data.prepare(*directories,out,deadline(),specs=specs,expected=(1,1))


def test_input_mutation_during_materialization_preserves_failure_without_final_manifest(tmp_path,monkeypatch):
    directories,specs=pinned_directories(tmp_path);out=tmp_path/'output';original=data.assemble
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        with (directories[0]/'accepted.jsonl').open('ab') as f:f.write(b'\n')
        return result
    monkeypatch.setattr(data,'assemble',changed)
    with pytest.raises(ValueError,match='inputs changed'):
        data.prepare(*directories,out,deadline(),specs=specs,expected=(1,1))
    assert (out/'materialization.failed.json').exists() and not (out/'manifest.json').exists()
    assert not (out/'closure.json').exists()


def test_wrong_pinned_manifest_fails_before_any_payload_decode(tmp_path,monkeypatch):
    directories,specs=pinned_directories(tmp_path);specs[0]['manifest_sha256']='0'*64
    def forbidden(*args,**kwargs):raise AssertionError('TRAIN payload loader reached')
    monkeypatch.setattr(data,'assemble',forbidden)
    with pytest.raises(ValueError,match='input hash differs'):
        data.prepare(*directories,tmp_path/'output',deadline(),specs=specs,expected=(1,1))


def test_wrong_expected_full_count_does_not_publish_smaller_inventory(tmp_path):
    directories,specs=pinned_directories(tmp_path);out=tmp_path/'output'
    with pytest.raises(ValueError,match='full candidate/document counts'):
        data.prepare(*directories,out,deadline(),specs=specs,expected=(2,2))
    assert (out/'materialization.failed.json').exists() and not (out/'manifest.json').exists()
