"""Strict closed-file matching and paired document differences without model execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from train import warmstart_compare as compare
from weekend_thinking.artifacts import canonical_hash,sha256_file


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


@pytest.fixture
def artifacts(tmp_path,monkeypatch):
    e=compare.evaluate;ids=[f'item{i:02d}' for i in range(64)];manifest='f'*64
    selected=dict(split='dev',count=64,item_ids=ids,document_ids=['doc:'+k for k in ids],item_ids_sha256=canonical_hash(ids),strata={k:f'Q{i//16+1}' for i,k in enumerate(ids)})
    base_id=dict(**e.original.model_identity('B0'),loaded_checkpoint=None,model={'parameters':3,'sampled_parameter_sha256':'b'*64},base_manifest_sha256=manifest)
    reference=tmp_path/'baseline/raw-reference.json';write(reference,{'items':ids,'reader':base_id})
    raw={k:[-3.]*16 for k in ids};registry={};directories={}
    old_common=dict(selection=selected,selection_sha256=canonical_hash(selected),model_dir=str(tmp_path/'base'),model_manifest_sha256=manifest,
        format_sha256='d'*64,cap=256,temperature=.7,seed=1729,batch_size=4,groups_per_prefix=1,shards=4,shard_size=16,
        batch_seed_rule='fixed ordered-batch hash',controls_seed=17,filler_text=' ',first_batch_replay_atol=.001,source_sha256=e.original.sources())
    notes={k:dict(n_actions=3,n_content_tokens=2,stop_type='token:5',prefix_sha256=canonical_hash([i]*512)) for i,k in enumerate(ids)}
    controls={k:dict(content_ids={m:[] if m=='empty' else [8,9] for m in e.MODES[1:]}) for k in ids}
    generation=[dict(batch_sampling_wall_seconds_total=4.,batch_replay_audit_wall_seconds_total=1.) for _ in range(4)]
    for role,own in [('B0',-4.),('writerBC',-2.),('raw_control',-3.5)]:
        directory=tmp_path/role;directory.mkdir();directories[role]=directory
        if role=='B0':writer=deepcopy(base_id)
        else:
            model=tmp_path/'models'/role/'model.pt'
            writer=dict(**e.original.model_identity(role,model,('a' if role=='writerBC' else 'c')*64),
                loaded_checkpoint={'path':str(model.resolve()),'sha256':('a' if role=='writerBC' else 'c')*64,'run_id':role+'-train','step':64,'spec_hash':'s'*64},
                model={'parameters':3,'sampled_parameter_sha256':('a' if role=='writerBC' else 'c')*64},base_manifest_sha256=manifest)
        p=dict(schema=e.SCHEMA,experiment_id=role,bundle=str((tmp_path/'bundles'/role).resolve()),baseline=str(reference.parent.resolve()),
            writer=writer,reader=e.original.model_identity('B0'),baseline_reader=base_id,selection=selected,
            model_dir=old_common['model_dir'],model_manifest_sha256=manifest,source_files=e.sources(),input_hashes={str(reference.resolve()):sha256_file(reference)},
            interface='prepend_plain',serialization='content_ids + [271] + full original512 prefix; raw=prefix',separator_ids=[271],
            modes=list(e.MODES),width=1024,horizons=[1,16],compute='fp32',shards=4,shard_size=16,expected_shard_rows=96,expected_rows=384,
            raw_reference_atol=.002,raw_reference_rule='Only unchanged B0 reader must match existing B0 raw token logps',
            primary='Fixed B0 reader, K16 fresh G1 own-minus-empty/filler/shuffled/other; K1 secondary',bootstrap=dict(unit='document',repeats=5000,seed=17))
        bundle_path=Path(p['bundle']);writer_paths=[bundle_path/'controls.jsonl',bundle_path/'notes/shard-0/notes.jsonl']
        write_rows(writer_paths[0],list(controls.values()));write_rows(writer_paths[1],[notes[k] for k in ids[:16]])
        p['input_hashes'].update({str(file.resolve()):sha256_file(file) for file in writer_paths})
        write(directory/'protocol.json',p)
        registry[p['bundle']]=(deepcopy(old_common),None,deepcopy(notes),writer,deepcopy(controls),deepcopy(generation),writer_paths)
        receipts=[];values=[]
        scores=dict(raw=-3.,empty=-3.1,own=own,filler=-5.,shuffled=-4.,other=-4.5)
        for shard in range(4):
            d=directory/'scoring'/f'shard-{shard}';local=[]
            for k in ids[shard*16:(shard+1)*16]:
                for m in e.MODES:
                    row=dict(item_id=k,document_id='doc:'+k,split='dev',shard=shard,mode=m,status='ok',token_log_likelihoods=[scores[m]]*16,
                        writer_identity_sha256=canonical_hash(writer),reader_identity_sha256=canonical_hash(base_id),writer_checkpoint_sha256=writer['checkpoint_sha256'],reader_checkpoint_sha256=None,
                        interface='raw' if m=='raw' else 'prepend_plain',context_tokens=512 if m=='raw' else 513+(0 if m=='empty' else 2),wall_seconds=.125)
                    if m=='raw':row['raw_reference_max_token_error']=0.
                    local.append(row)
            write_rows(d/'scores.jsonl',local);values.extend(local)
            run=dict(status='complete',schema=e.SCHEMA,shard=shard,writer=writer,reader=base_id,rows=96,expected_rows=96,item_ids=ids[shard*16:(shard+1)*16],
                protocol_sha256=sha256_file(directory/'protocol.json'),scores_sha256=sha256_file(d/'scores.jsonl'),raw_reference_rows=16,raw_reference_max_token_error=0.,
                started_utc='2026-10-18T00:00:00+00:00',finished_utc='2026-10-18T00:00:03+00:00')
            write(d/'run.json',run);receipts.append(dict(shard=shard,run_sha256=sha256_file(d/'run.json'),scores_sha256=sha256_file(d/'scores.jsonl')))
        perdoc=[dict(item_id=k,document_id='doc:'+k,quartile=selected['strata'][k],n_actions=3,n_content_tokens=2,stop_type='token:5',
                    absolute_log_likelihoods={f'K{h}':{m:sum([scores[m]]*h) for m in e.MODES} for h in (1,16)}) for k in ids]
        summary=dict(schema=e.SCHEMA,status='complete',experiment_id=role,documents=64,rows=384,failures=0,fixed_B0_reader_primary=True,
            protocol_sha256=sha256_file(directory/'protocol.json'),primary=p['primary'],writer=writer,reader=base_id,scoring=receipts,per_document=perdoc,
            note_context=dict(empty=0,caps=0,content_length_counts={'2':64},actions_total=192,content_tokens_total=128,content_tokens_mean=2,content_tokens_median=2.,stop_counts={'token:5':64}),
            cost=dict(new_sampling_seconds=0,reused_sampling_batches=16,reused_sampling_batch_seconds_counted_once=16.,reused_replay_audit_batch_seconds_counted_once=4.,
                new_score_context_seconds=48.,new_score_process_seconds=12.,stage_span_seconds=3.,qualification='Batch generation counted once; isolated reader/parallel process walls are not training throughput'),raw_reference_max_token_error=0.)
        write(directory/'summary.json',summary)
    monkeypatch.setattr(e,'closed_bundle',lambda directory:registry[str(directory.resolve())])
    monkeypatch.setattr(e,'raw_baseline',lambda directory:(deepcopy(old_common),raw,base_id,[reference]))
    return directories,registry


def reseal(directory,shard=0):
    e=compare.evaluate;d=directory/'scoring'/f'shard-{shard}';run=e.read(d/'run.json')
    run['scores_sha256']=sha256_file(d/'scores.jsonl');write(d/'run.json',run)
    s=e.read(directory/'summary.json');s['scoring'][shard].update(run_sha256=sha256_file(d/'run.json'),scores_sha256=run['scores_sha256']);write(directory/'summary.json',s)


def test_complete_inputs_and_actual_paired_document_direction_and_horizons(artifacts):
    paths,_=artifacts;arms={role:compare.load_complete(path/'summary.json') for role,path in paths.items()}
    result=compare.compare(arms)
    assert result['documents']==64 and len(result['per_document'])==64 and result['primary']=='K16' and result['secondary']=='K1'
    assert set(result['groups'])=={'all','Q1','Q2','Q3','Q4'}
    for label,group in result['groups'].items():
        for h in (1,16):
            value=group['horizons'][f'K{h}'];ci=value['paired_comparisons']['writerBC_minus_B0']['own_minus_empty']
            assert ci['difference_b_minus_a']==pytest.approx(2*h) and ci['ci95']==pytest.approx([2*h,2*h])
            assert ci['items']==ci['documents']==(64 if label=='all' else 16)
            assert ci['bootstrap_draws']==5000 and ci['bootstrap_seed']==17
            assert value['paired_comparisons']['writerBC_minus_raw_control']['own_minus_other']['difference_b_minus_a']==pytest.approx(1.5*h)
            assert value['arms']['writerBC']['absolute_mean_nats']['own']==-2*h
    assert result['arms']['B0']['note_context']['caps']==0 and result['arms']['writerBC']['cost']['new_score_context_seconds']==48.
    assert result['arms']['raw_control']['writer']['loaded_checkpoint']['step']==64
    assert result['arms']['writerBC']['training_checkpoint_metadata']==dict(run_id='writerBC-train',step=64,spec_hash='s'*64)


@pytest.mark.parametrize('damage',['partial','adapted','summary_score','cost','length','missing_receipt'])
def test_rejects_partial_adapted_or_changed_summary_metadata(artifacts,damage):
    paths,_=artifacts;d=paths['writerBC'];s=compare.evaluate.read(d/'summary.json')
    if damage=='partial':s['status']='incomplete'
    elif damage=='adapted':
        p=compare.evaluate.read(d/'protocol.json');p['reader']=p['writer'];write(d/'protocol.json',p)
        s['protocol_sha256']=sha256_file(d/'protocol.json')
    elif damage=='summary_score':s['per_document'][3]['absolute_log_likelihoods']['K16']['own']+=1
    elif damage=='cost':s['cost']['new_score_context_seconds']+=1
    elif damage=='length':s['note_context']['caps']=1
    else:s['scoring'].pop()
    write(d/'summary.json',s)
    with pytest.raises(ValueError):compare.load_complete(d)


@pytest.mark.parametrize('damage',['failed','short','nan','duplicate','document','reader_identity','raw_mismatch','context'])
def test_resealed_invalid_rows_cannot_hide_behind_summary_or_hashes(artifacts,damage):
    paths,_=artifacts;d=paths['writerBC'];file=d/'scoring/shard-0/scores.jsonl';rows=compare.evaluate.rows(file)
    if damage=='failed':rows[2]['status']='error'
    elif damage=='short':rows[2]['token_log_likelihoods'].pop()
    elif damage=='nan':rows[2]['token_log_likelihoods'][0]=float('nan')
    elif damage=='duplicate':rows[2]=deepcopy(rows[1])
    elif damage=='document':rows[2]['document_id']='other-document'
    elif damage=='reader_identity':rows[2]['reader_identity_sha256']='changed'
    elif damage=='raw_mismatch':rows[0]['token_log_likelihoods'][0]+=.003
    else:rows[2]['context_tokens']+=1
    write_rows(file,rows);reseal(d)
    with pytest.raises(ValueError):compare.load_complete(d)


def test_unresealed_source_change_is_rejected(artifacts):
    paths,_=artifacts;d=paths['writerBC'];file=d/'scoring/shard-0/scores.jsonl';file.write_text(file.read_text()+'\n')
    with pytest.raises(ValueError,match='changed score'):compare.load_complete(d)


@pytest.mark.parametrize('damage',['quartile','order','document','policy','reader','raw_reference','baseline_path','writer_alias','prefix','weights_alias','early_checkpoint'])
def test_cross_arm_matching_is_exact_and_never_intersects_or_filters(artifacts,damage):
    paths,_=artifacts;arms={role:compare.load_complete(path) for role,path in paths.items()};arm=arms['writerBC']
    if damage=='quartile':arm['protocol']['selection']=deepcopy(arm['protocol']['selection']);arm['protocol']['selection']['strata']['item00']='Q2'
    elif damage=='order':arm['per_document']=list(reversed(arm['per_document']))
    elif damage=='document':arm['per_document'][0]['document_id']='other'
    elif damage=='policy':arm['bundle_policy']['seed']=1730
    elif damage=='reader':arm['reader']=dict(arm['reader'],model={'changed':True})
    elif damage=='raw_reference':arm['baseline_pins']={k:'changed' for k in arm['baseline_pins']}
    elif damage=='baseline_path':arm['protocol']['baseline']='/different/baseline'
    elif damage=='prefix':arm['prefix_hashes']['item00']='changed'
    elif damage=='weights_alias':arm['writer']=dict(arm['writer'],checkpoint_sha256=arms['raw_control']['writer']['checkpoint_sha256'])
    elif damage=='early_checkpoint':arm['writer']['loaded_checkpoint']['step']=16
    else:arm['writer']=deepcopy(arms['raw_control']['writer'])
    with pytest.raises(ValueError):compare.compare(arms)


@pytest.mark.parametrize('kind',['controls','notes'])
def test_writer_note_and_control_paths_must_be_explicitly_pinned(artifacts,kind):
    paths,_=artifacts;d=paths['writerBC'];p=compare.evaluate.read(d/'protocol.json')
    target=Path(p['bundle'])/('controls.jsonl' if kind=='controls' else 'notes/shard-0/notes.jsonl')
    p['input_hashes'].pop(str(target.resolve()));write(d/'protocol.json',p)
    s=compare.evaluate.read(d/'summary.json');s['protocol_sha256']=sha256_file(d/'protocol.json');write(d/'summary.json',s)
    with pytest.raises(ValueError,match='Writer bundle/raw-reference files'):compare.load_complete(d)
