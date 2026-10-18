"""Fixed GENERAL TRAIN pool/projections and native hindsight-label decisions.

CPU-only commands never load a model. This separate future-aware supervised
branch does not change the closed TRAIN256 pilot or compound5120 artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import time

from train import general_hindsight_notes as pilot
from train import useful_note_evaluation as shared
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'general-hindsight-filtered-train5120-v1'
SALT = SCHEMA
RAW_N, TEACHER_N, LABEL_N, SHARDS = 131072, 24576, 5120, 16
THRESHOLD = 1.7
DESIGN_SHA = '4fa7c23ec8c037166c76e36e215d7e2eb8f2d96377fc9b56b7b3b25f02674889'
SOURCE_SHA = 'bd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff'
PILOT_PROTOCOL_SHA = 'b2c9951d16738201af24c19a0fcb945c916439d3a2b327b3e59bea8c584053c9'
HEADER = re.compile(rb'^\{\s*"id"\s*:\s*"([^"\\]+)"\s*,\s*"document_id"\s*:\s*"([^"\\]+)"\s*,\s*"split"\s*:\s*"(train|dev|test)"\s*,\s*"prefix"\s*:\s*')


def sources():
    return {**pilot.sources(), **{name:sha256_file(ROOT/name) for name in
        ('train/general_hindsight_filtered_data.py','train/general_hindsight_filtered_batch.py')}}


def normalized_whitespace(text):return ' '.join(text.split())


@lru_cache(maxsize=4)
def defined_ids(tok):return frozenset(tok.get_vocab().values())


def pool_order(rows):
    """One native anchor per document; selection never consumes token payloads."""
    if len({r['document_id'] for r in rows})!=len(rows) or len({r['item_id'] for r in rows})!=len(rows):
        raise ValueError('Unique approved TRAIN documents/anchors required')
    if any(r['split']!='train' for r in rows):raise ValueError('TRAIN metadata only')
    return sorted(rows,key=lambda r:(canonical_hash(dict(salt=SALT,role='raw_pool',document_id=r['document_id'],item_id=r['item_id'])),r['document_id'],r['item_id']))


def write_rows(file,rows):
    with Path(file).open('x') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def phase_file(output,phase):return Path(output)/f'{phase}.manifest.json'


def shard_file(output,phase,shard):return Path(output)/f'{phase}-records'/f'shard-{shard:02d}.jsonl'


def validate_native(row,defined):
    if (row['split']!='train' or len(row['prefix_ids'])!=512 or len(row['natural_future16_ids'])!=16
            or row['reanchor_ids']!=row['prefix_ids'][-24:]
            or any(type(t)is not int or t not in defined for t in row['prefix_ids']+row['natural_future16_ids'])
            or row['source528_sha256']!=canonical_hash(row['prefix_ids']+row['natural_future16_ids'])):
        raise ValueError('Exact defined native512+16 TRAIN source required; never truncate/filter')


def prepare_raw(output,source,membership,source_manifest,design,pilot_output,until):
    from transformers import AutoTokenizer
    shared.check(until);output=Path(output);source=Path(source).resolve();membership=Path(membership).resolve()
    source_manifest=Path(source_manifest).resolve();design=Path(design).resolve();pilot_output=Path(pilot_output).resolve()
    shared.verify_pins({str(design):DESIGN_SHA,str(pilot_output/'protocol.json'):PILOT_PROTOCOL_SHA})
    sm=shared.read(source_manifest);old=shared.read(pilot_output/'protocol.json')
    if sm['output_sha256']!=SOURCE_SHA or sha256_file(source)!=SOURCE_SHA or sha256_file(membership)!=sm['document_membership_sha256']:
        raise ValueError('Exact approved audited source/membership required')
    tok=AutoTokenizer.from_pretrained(old['model_dir'],local_files_only=True,trust_remote_code=False);defined=set(tok.get_vocab().values())
    # Membership JSON has provenance only. Non-TRAIN lines remain opaque.
    available=[]
    for line in membership.open('rb'):
        match=re.search(rb'"split"\s*:\s*"(train|dev|test)"',line)
        if match is None:raise ValueError('Audited membership split missing')
        if match[1]!=b'train':continue
        m=json.loads(line);available.append(dict(item_id=m['record_id'],document_id=m['document_id'],split='train'))
    ordered=pool_order(available)
    if len(ordered)<RAW_N:raise ValueError('Fixed131072 unique TRAIN quota shortfall; no smaller pool')
    selected=ordered[:RAW_N];frozen=shared.read(design)['raw_pool']
    if (frozen['documents']!=RAW_N or frozen['salt']!=SALT
            or canonical_hash([r['item_id'] for r in selected])!=frozen['ordered_item_ids_sha256']
            or canonical_hash([r['document_id'] for r in selected])!=frozen['ordered_document_ids_sha256']):
        raise ValueError('Prospectively declared raw-pool identity/order changed')
    wanted={r['item_id']:(i,r['document_id']) for i,r in enumerate(selected)}
    output.mkdir(parents=True,exist_ok=False);(output/'raw-records').mkdir();started=time.monotonic()
    atomic_json(output/'preparation.json',dict(status='PREPARING_CPU_ONLY',started_utc=utc_now(),records_written=0))
    streams=[shard_file(output,'raw',i).open('x') for i in range(SHARDS)];found={};h=hashlib.sha256();counts=Counter()
    try:
        for line_number,line in enumerate(source.open('rb'),1):
            h.update(line);match=HEADER.match(line)
            if match is None:raise ValueError('Canonical source header missing')
            key,doc,split=(v.decode() for v in match.groups());counts[split]+=1
            if split!='train' or key not in wanted:continue # Never JSON-decode TEST/DEV or unselected arrays.
            shared.check(until);index,expected=wanted[key]
            if doc!=expected or key in found:raise ValueError('Selected native source duplicate/membership mismatch')
            r=json.loads(line)
            row=dict(item_id=key,document_id=doc,split='train',pool_index=index,prefix_ids=r['prefix'],natural_future16_ids=r['future'],
                reanchor_ids=r['prefix'][-24:],source_line=line_number,source528_sha256=canonical_hash(r['prefix']+r['future']))
            validate_native(row,defined);found[index]=row
            if len(found)%8192==0:atomic_json(output/'preparation.json',dict(status='PREPARING_CPU_ONLY',records_written=len(found),elapsed_seconds=time.monotonic()-started),overwrite=True)
        if h.hexdigest()!=SOURCE_SHA or set(found)!=set(range(RAW_N)) or len({r['document_id'] for r in found.values()})!=RAW_N:
            raise ValueError('Complete selected source/hash identity required')
        # Write fixed order within each shard regardless of canonical file order.
        for index in range(RAW_N):streams[index%SHARDS].write(json.dumps(found[index],sort_keys=True)+'\n')
    except BaseException as exc:
        atomic_json(output/'preparation.json',dict(status='FAILED_CPU_PREPARATION_PARTIALS_RETAINED',records_found=len(found),error=f'{type(exc).__name__}: {exc}'),overwrite=True);raise
    finally:
        for stream in streams:stream.close()
    order=[dict(pool_index=i,**r) for i,r in enumerate(selected)];write_rows(output/'raw-order.jsonl',order)
    base_keys=('model_dir','assets','model_manifest_sha256','base_signature','native_reference','native_inputs','writer','vocab_size','banned_ids','teacher_parts','parts','writer_prefill_width','teacher_policy')
    config={k:old[k] for k in base_keys}
    forbidden=sorted(set(tok.all_special_ids)|{151667,151668})
    manifest=dict(schema=SCHEMA,status='COMPLETE_CPU_RAW_PROJECTIONS_NOT_MODEL_RELEASE',created_utc=utc_now(),raw_items=RAW_N,
        teacher_items=TEACHER_N,label_items=LABEL_N,shards=SHARDS,threshold_surprisal_nats=THRESHOLD,salt=SALT,
        input_hashes=shared.pin([source,membership,source_manifest,design,pilot_output/'protocol.json']),source_files=sources(),
        source_counts=dict(counts),approved_unique_TRAIN_documents=len(ordered),raw_order_file=str((output/'raw-order.jsonl').resolve()),
        raw_order_sha256=sha256_file(output/'raw-order.jsonl'),ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in selected]),
        ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in selected]),
        raw_shards={str(i):dict(file=str(shard_file(output,'raw',i).resolve()),sha256=sha256_file(shard_file(output,'raw',i)),items=RAW_N//SHARDS) for i in range(SHARDS)},
        native_config=config,forbidden_content_ids=forbidden,tokenizer_special_ids=sorted(tok.all_special_ids),
        teacher_selection='First24576 finite successful raw K1>=1.7 in raw order; unknown earlier eligibility blocks publication, no quota fallback.',
        label_selection='First5120 STRICT I3 teacher-minus-empty>0, nonempty stripped text/native-special-free and no literal completefuture16 copy. No likelihood ranking or I6 fallback.',
        copy_rule='Contiguous future16 native token tuple OR full decodedfuture16 substring after whitespace-only collapse/strip; no case/Unicode/punctuation normalization.',
        limitations=['Explicit future-aware TRAIN label curation informed by closed pilot; no deployed prefix-only utility claim.','16 independent workers read only native own-shard projections.','Plain-text/copy gates do not certify semantic utility or exclude paraphrased target hints.'])
    shared.verify_pins(manifest['input_hashes']);atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'preparation.json',dict(status=manifest['status'],records_written=RAW_N,elapsed_seconds=time.monotonic()-started,
        manifest_sha256=sha256_file(output/'manifest.json'),finished_utc=utc_now()),overwrite=True)
    return manifest


def finalize(output,until,control_dir):
    """Bind the actual finite model deadline AFTER CPU projection preparation."""
    output=Path(output);m=shared.read(output/'manifest.json');shared.check(until)
    if m['schema']!=SCHEMA or m['status']!='COMPLETE_CPU_RAW_PROJECTIONS_NOT_MODEL_RELEASE' or m['source_files']!=sources():
        raise ValueError('Settled complete CPU projections/source required')
    p=dict(m['native_config'],schema=SCHEMA,source_files=sources(),raw_items=RAW_N,teacher_items=TEACHER_N,label_items=LABEL_N,shards=SHARDS,
        manifest_file=str((output/'manifest.json').resolve()),manifest_sha256=sha256_file(output/'manifest.json'),
        paths=list(pilot.PATHS),modes=list(pilot.MODES),width=1024,future_width=16,precision='fp32',max_actions=64,stop_ids=[151643],
        mixer_weight=.375,deadline_utc=until,control_dir=str(Path(control_dir).resolve()),forbidden_content_ids=m['forbidden_content_ids'],
        purpose='Independent GENERAL future-aware filtered supervised-label branch; not a training release.',bootstrap=dict(repeats=5000,seed=17))
    atomic_json(output/'protocol.json',p);return p


def load(output):
    output=Path(output);p=shared.read(output/'protocol.json');m=shared.read(p['manifest_file'])
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or m['source_files']!=sources() or sha256_file(p['manifest_file'])!=p['manifest_sha256']
            or (p['raw_items'],p['teacher_items'],p['label_items'],p['shards'])!=(RAW_N,TEACHER_N,LABEL_N,SHARDS)
            or p['writer']!=pilot.B0 or p['paths']!=list(pilot.PATHS) or p['modes']!=list(pilot.MODES) or p['max_actions']!=64 or p['mixer_weight']!=.375):
        raise ValueError('Frozen GENERAL branch source/recipe changed')
    # Native arrays are pinned per shard; never reparse/hash all131k in every model worker.
    shared.verify_pins(p['native_inputs']);return p,m


def read_shard(output,phase,shard):
    p,m=load(output)
    if type(shard)is not int or shard not in range(SHARDS):raise ValueError('Fixed16 shards required')
    manifest=m if phase=='raw' else shared.read(phase_file(output,phase))
    if phase!='raw' and (manifest['schema']!=SCHEMA or manifest['protocol_sha256']!=sha256_file(Path(output)/'protocol.json')
            or manifest['status']!={'teacher':'COMPLETE_FIXED_TEACHER_SELECTION','reader':'COMPLETE_GLOBAL_INTACT_DONORS'}[phase]):
        raise ValueError('Complete source-bound stage projection required')
    info=manifest[f'{phase}_shards'][str(shard)];shared.verify_pins({info['file']:info['sha256']});rows=shared.rows(info['file'])
    field='pool_index' if phase=='raw' else 'teacher_index';n=RAW_N if phase=='raw' else TEACHER_N
    if len(rows)!=n//SHARDS or [r[field] for r in rows]!=list(range(shard,n,SHARDS)):
        raise ValueError('Exact own-shard native order/count required')
    return p,rows


def require_closure(output,p,phase):
    path=Path(p['control_dir'])/f'{phase}.closure.json';c=shared.read(path)
    if (c['status']!='COMPLETE_PHYSICALLY_CLOSED' or c['protocol_sha256']!=sha256_file(Path(output)/'protocol.json')
            or not c['all16_exit0'] or not c['all48_owned_PID_start_identities_absent'] or not c['all_sessions_absent']
            or not c['assigned_devices_empty'] or c['owned_group_survivors']):
        raise ValueError('Actual full16-job physical closure required')
    return shared.pin([path])


def complete_scores(output,p,phase,n,multiplier=1):
    """Read all receipts FIRST; only then expose complete saved outcomes."""
    pins={};reports=[];identity=None
    manifest=Path(output)/'manifest.json' if phase=='raw' else phase_file(output,phase)
    for shard in range(SHARDS):
        d=Path(output)/phase/f'shard-{shard:02d}';r=shared.read(d/'run.json')
        if (r['status']!='complete_attempts' or r['rows']!=n//SHARDS*multiplier
                or r['protocol_sha256']!=sha256_file(Path(output)/'protocol.json') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or r['record_manifest_sha256']!=sha256_file(manifest) or r['records_sha256']!=sha256_file(shard_file(output,phase,shard))
                or any(r['reader'].get(k)!=v for k,v in pilot.B0.items()) or identity is not None and identity!=r['reader']):
            raise ValueError('All16 complete source-bound B0 score shards required')
        identity=r['reader'];pins.update(shared.pin([d/'run.json',d/'scores.jsonl']));reports.append(r)
    values=[r for shard in range(SHARDS) for r in shared.rows(Path(output)/phase/f'shard-{shard:02d}'/'scores.jsonl')]
    return values,pins,reports


def choose_teachers(scores,order):
    index={r['pool_index']:r for r in scores}
    if len(index)!=len(scores) or set(index)!=set(range(RAW_N)) or [r['pool_index'] for r in order]!=list(range(RAW_N)):
        raise ValueError('All131072 raw attempts/ordered metadata required')
    selected=[];unknown=[]
    for i,o in enumerate(order):
        r=index[i]
        if r['item_id']!=o['item_id'] or r['document_id']!=o['document_id'] or r['status'] not in ('ok','error'):
            raise ValueError('Raw score identity/status mismatch')
        if r['status']=='error':unknown.append(i);continue
        pilot.likelihood(r['token_logps'])
        if r['raw_K1_surprisal_nats']!=-r['token_logps'][0]:raise ValueError('K1 is not original first-token LP')
        if r['raw_K1_surprisal_nats']>=THRESHOLD:selected.append(i)
    chosen=selected[:TEACHER_N]
    if len(chosen)<TEACHER_N:return None,dict(status='FAILED_FIXED_TEACHER_QUOTA',qualifiers=len(selected),required=TEACHER_N,raw_errors=len(unknown))
    if any(i<=chosen[-1] for i in unknown):return None,dict(status='FAILED_UNKNOWN_EARLIER_ELIGIBILITY',qualifiers=len(selected),raw_errors=len(unknown))
    return chosen,dict(status='COMPLETE_FIXED_TEACHER_SELECTION',qualifiers=len(selected),selected=TEACHER_N,raw_errors=len(unknown),last_selected_pool_index=chosen[-1])


def select_teachers(output):
    output=Path(output);p,m=load(output);closed=require_closure(output,p,'raw');values,pins,reports=complete_scores(output,p,'raw',RAW_N)
    shared.verify_pins({m['raw_order_file']:m['raw_order_sha256']});order=shared.rows(m['raw_order_file'])
    chosen,audit=choose_teachers(values,order)
    if chosen is None:
        atomic_json(output/'teacher-selection.audit.json',dict(audit,raw_closure=closed));raise RuntimeError(audit['status'])
    target={v:i for i,v in enumerate(chosen)};selected={};raw_index={v['pool_index']:v for v in values}
    for shard in range(SHARDS):
        _,rows=read_shard(output,'raw',shard)
        for row in rows:
            v=raw_index[row['pool_index']]
            if v['target_ids']!=row['natural_future16_ids'] or v['context_sha256']!=canonical_hash(row['prefix_ids']):
                raise ValueError('Raw decision is not the exact original prefix/target pair')
            if row['pool_index'] in target:selected[target[row['pool_index']]]=dict(row,teacher_index=target[row['pool_index']])
    if set(selected)!=set(range(TEACHER_N)) or len({r['document_id'] for r in selected.values()})!=TEACHER_N:
        raise ValueError('Exact selected unique documents required')
    atomic_json(output/'teacher-selection.audit.json',dict(audit,raw_closure=closed))
    (output/'teacher-records').mkdir()
    for shard in range(SHARDS):write_rows(shard_file(output,'teacher',shard),[selected[i] for i in range(shard,TEACHER_N,SHARDS)])
    write_rows(output/'teacher-order.jsonl',[{k:selected[i][k] for k in ('teacher_index','pool_index','item_id','document_id')} for i in range(TEACHER_N)])
    result=dict(schema=SCHEMA,status=audit['status'],protocol_sha256=sha256_file(output/'protocol.json'),raw_score_pins=pins,raw_closure=closed,
        teacher_order_file=str((output/'teacher-order.jsonl').resolve()),teacher_order_sha256=sha256_file(output/'teacher-order.jsonl'),
        teacher_shards={str(i):dict(file=str(shard_file(output,'teacher',i).resolve()),sha256=sha256_file(shard_file(output,'teacher',i)),items=TEACHER_N//SHARDS) for i in range(SHARDS)},audit=audit)
    atomic_json(phase_file(output,'teacher'),result);return result


def label_eligibility(note,future,tok,forbidden):
    ids=note['content_ids'];text=pilot.decode(tok,ids);target=pilot.decode(tok,future);reasons=[]
    if note['status']!='ok':reasons.append('teacher_error')
    if not text.strip():reasons.append('empty_stripped_text')
    if not 1<=len(ids)<=64:reasons.append('native_length_not1to64')
    if any(type(t)is not int or t not in defined_ids(tok) for t in ids):reasons.append('undefined_or_invalid_native_id')
    if set(ids)&set(forbidden):reasons.append('native_special_or_thought_marker')
    token_copy=any(ids[i:i+16]==future for i in range(max(0,len(ids)-15)))
    literal=normalized_whitespace(target);char_copy=bool(literal and literal in normalized_whitespace(text))
    if token_copy:reasons.append('complete_future16_token_copy')
    if char_copy:reasons.append('complete_future16_whitespace_only_character_copy')
    return dict(eligible=not reasons,reasons=reasons,full16_token_copy=token_copy,full16_character_copy=char_copy)


def project_readers(output):
    from transformers import AutoTokenizer
    output=Path(output);p,_=load(output);closed=require_closure(output,p,'teacher');tm=shared.read(phase_file(output,'teacher'))
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False);notes={};rows={};pins={};reports=[];identity=None
    # Every receipt/hash is checked before using any teacher label.
    for shard in range(SHARDS):
        d=output/'teacher'/f'shard-{shard:02d}';r=shared.read(d/'run.json')
        if (r['status']!='complete_attempts' or r['rows']!=TEACHER_N//SHARDS or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['notes_sha256']!=sha256_file(d/'notes.jsonl') or any(r['writer'].get(k)!=v for k,v in pilot.B0.items())
                or r['record_manifest_sha256']!=sha256_file(phase_file(output,'teacher'))
                or r['records_sha256']!=sha256_file(shard_file(output,'teacher',shard)) or identity is not None and identity!=r['writer']):
            raise ValueError('All24576 complete teacher attempts required')
        identity=r['writer'];reports.append(r);pins.update(shared.pin([d/'run.json',d/'notes.jsonl']))
    for shard in range(SHARDS):
        _,source=read_shard(output,'teacher',shard);vv=shared.rows(output/'teacher'/f'shard-{shard:02d}'/'notes.jsonl')
        if [r['teacher_index'] for r in vv]!=[r['teacher_index'] for r in source]:raise ValueError('Teacher order changed')
        for note,row in zip(vv,source):
            if note['pool_index']!=row['pool_index']:raise ValueError('Original raw-pool ancestry changed')
            pilot.validate_note(note,row,p,tok);notes[row['teacher_index']]=note;rows[row['teacher_index']]=row
    projected=global_reader_projection(rows,notes)
    (output/'reader-records').mkdir()
    for shard in range(SHARDS):
        write_rows(shard_file(output,'reader',shard),[projected[i] for i in range(shard,TEACHER_N,SHARDS)])
    result=dict(schema=SCHEMA,status='COMPLETE_GLOBAL_INTACT_DONORS',protocol_sha256=sha256_file(output/'protocol.json'),teacher_closure=closed,
        teacher_manifest_sha256=sha256_file(phase_file(output,'teacher')),note_pins=pins,teacher_stage_costs=reports,
        other='Next teacher_index modulo24576, intact native first-line note; errors/empty donors retained; no matching/repeat/trim.',
        reader_shards={str(i):dict(file=str(shard_file(output,'reader',i).resolve()),sha256=sha256_file(shard_file(output,'reader',i)),items=TEACHER_N//SHARDS) for i in range(SHARDS)})
    atomic_json(phase_file(output,'reader'),result);return result


def global_reader_projection(rows,notes):
    if (set(rows)!=set(range(TEACHER_N)) or set(notes)!=set(rows)
            or len({r['document_id'] for r in rows.values()})!=TEACHER_N):
        raise ValueError('All unique global teacher documents/attempts required')
    return {i:dict(rows[i],teacher_note=notes[i],other_note=notes[(i+1)%TEACHER_N],other_item_id=rows[(i+1)%TEACHER_N]['item_id']) for i in range(TEACHER_N)}


def native_contexts(row,p):
    from train.interface_oracle_probe import reader_context
    return dict(raw=list(row['prefix_ids']),**{mode:reader_context(row,note,'I3',p['parts']) for mode,note in
        (('empty',[]),('teacher',row['teacher_note']['content_ids']),('other',row['other_note']['content_ids']))})


def pack_native(row,tok,component,*,parts,width,target_slots):
    """Native label adapter; no symbolic proof/string construction or shared edits."""
    from train.useful_note_sft_data import pack_writer,packed
    from train.interface_oracle_probe import reader_context
    past=row['prefix_ids'];future=row['natural_future16_ids'];note=row['teacher_note_ids'];original=row['original_source_ids']
    if row['split']!='train' or len(past)!=512 or len(future)!=16 or original!=past+future:
        raise ValueError('Exact native source528 and original16 targets required')
    check=label_eligibility(dict(status='ok',content_ids=note),future,tok,set(tok.all_special_ids)|{151667,151668})
    if not check['eligible']:raise ValueError('Unsafe native teacher label')
    if component=='writer':return pack_writer(past,note,interface='I3',parts=parts,width=width,target_slots=target_slots)
    if component=='source':ids=original;positions=list(range(1,528))
    elif component in ('reader','augmented'):
        context=reader_context(row,note,'I3',parts);ids=context+future
        positions=list(range(len(context),len(ids))) if component=='reader' else list(range(1,len(ids)))
    else:raise ValueError('Declared native source/writer/reader/augmented component required')
    return packed(ids,positions,note_positions=[],width=width,target_slots=target_slots)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare-raw','finalize','select-teachers','project-readers'))
    a.add_argument('--output',type=Path,required=True)
    for name in ('source','membership','source-manifest','design','pilot-output','control-dir'):a.add_argument('--'+name,type=Path)
    a.add_argument('--deadline-utc');v=a.parse_args()
    if v.command=='prepare-raw':r=prepare_raw(v.output,v.source,v.membership,v.source_manifest,v.design,v.pilot_output,v.deadline_utc)
    elif v.command=='finalize':r=finalize(v.output,v.deadline_utc,v.control_dir)
    elif v.command=='select-teachers':r=select_teachers(v.output)
    else:r=project_readers(v.output)
    print(json.dumps(dict(status=r.get('status','COMPLETE_FINITE_PROTOCOL'),schema=SCHEMA)))


if __name__=='__main__':main()
