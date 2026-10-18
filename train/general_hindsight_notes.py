"""Exploratory GENERAL TRAIN256 hindsight labels; never prefix-only utility.

One frozen B0 greedy cap64 teacher attempt per document sees the real future.
All raw actions survive target-independent first-line extraction. All attempts
physically close before intact global donors and the fixed I3/I6 reader batch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from train import useful_note_evaluation as shared
from train.firstline_note_probe import truncate_content
from train.interface_oracle_probe import reader_context
from train.teacher_prepend_probe import copying
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'general-hindsight-notes-train256-v1'
DECLARATION_SHA = '6060b9cf1740d34abb48d490ef97e90ab3156b7d49cc2fdec75b0d2de51483bf'
SELECTION_SHA = 'eedf86f96615eb72cc7cd411867bc81681ec6193307f4d9137b437f2b976cd4c'
N, SHARDS, CAP, WIDTH = 256, 4, 64, 1024
PATHS = ('I3', 'I6w0.375')
MODES = ('raw', 'empty', 'teacher', 'other')
B0 = dict(id='B0', checkpoint=None, checkpoint_sha256=None)


def sources():
    names = ('train/general_hindsight_notes.py', 'train/firstline_note_probe.py', 'train/teacher_prepend_probe.py')
    return {**shared.sources(), **{name: sha256_file(ROOT/name) for name in names}}


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def validate_selection(s, rows):
    ids, docs = s['item_ids'], s['document_ids']
    if (s['split'] != 'train' or s['count'] != N or len(ids) != N or len(set(ids)) != N
            or len(docs) != N or len(set(docs)) != N or canonical_hash(ids) != s['item_ids_sha256']
            or canonical_hash(docs) != s['document_ids_sha256'] or len(s['origins']) != N
            or s['threshold_surprisal_nats'] != 1.7 or s['threshold_comparison'] != '>='
            or s['likelihood_ranking'] is not False or s['held256_document_overlap'] != 0
            or s['priorDEV572_document_overlap'] != 0):
        raise ValueError('Exact fixed unique TRAIN256 selection and threshold required')
    if ([r['item_id'] for r in rows] != ids or [r['document_id'] for r in rows] != docs
            or any(r['split'] != 'train' or len(r['prefix']) != 512 or len(r['future']) != 16 for r in rows)
            or any(not math.isfinite(v['raw_K1_surprisal_nats']) or v['raw_K1_surprisal_nats'] < 1.7 for v in s['origins'])):
        raise ValueError('No selection replacement, shortened record or below-threshold item')


def teacher_prompt(row, parts):
    """Native source IDs stay unchanged; future access is explicit hindsight."""
    return parts['initial'] + row['prefix_ids'] + parts['middle'] + row['natural_future16_ids'] + parts['final']


def prepare(output, declaration_file, records_file, recovery_file, native_reference):
    from transformers import AutoTokenizer
    from train.interface_oracle_probe import load as native_load, reference
    declaration_file = Path(declaration_file).resolve()
    shared.verify_pins({str(declaration_file): DECLARATION_SHA})
    declaration = shared.read(declaration_file); shared.check(declaration['proposed_runtime']['deadline_utc'])
    selection_file = Path(declaration['selection_file']); shared.verify_pins({str(selection_file): SELECTION_SHA})
    s = shared.read(selection_file); shared.verify_pins(s['source_input_hashes'])
    recovery_file = Path(recovery_file).resolve(); records_file = Path(records_file).resolve()
    recovery = shared.read(recovery_file)
    if (recovery['status'] != 'COMPLETE_CPU_ONLY_NO_MODEL' or recovery['selection_sha256'] != SELECTION_SHA
            or recovery['original_records_sha256'] != s['records_sha256'] or recovery['items'] != N
            or recovery['recovered_file'] != str(records_file) or recovery['recovered_sha256'] != sha256_file(records_file)):
        raise ValueError('Actual allowed-TRAIN native source recovery receipt required')
    rows = shared.rows(records_file); validate_selection(s, rows)
    native_reference = Path(native_reference).resolve(); old, native = native_load(native_reference)
    ref, _ = reference(native_reference, old, native)
    if old['assets']['manifest_sha256'] != declaration['teacher']['base_manifest_sha256']:
        raise ValueError('Actual released B0 assets must match the declaration')
    tok = AutoTokenizer.from_pretrained(old['model_dir'], local_files_only=True, trust_remote_code=False)
    defined = set(tok.get_vocab().values()); enc = lambda text: tok.encode(text, add_special_tokens=False)
    parts = {name: enc(declaration['teacher'][field]) for name, field in
             (('initial', 'fixed_initial_text'), ('middle', 'middle_text'), ('final', 'final_text'))}
    data = [dict(item_id=r['item_id'], document_id=r['document_id'], split='train', prefix_ids=r['prefix'],
                 natural_future16_ids=r['future'], reanchor_ids=r['prefix'][-24:]) for r in rows]
    if any(type(t) is not int or t not in defined for r in data for t in r['prefix_ids'] + r['natural_future16_ids']):
        raise ValueError('Original defined native source tokens required')
    widths = [len(teacher_prompt(r, parts)) for r in data]
    shared.check(declaration['proposed_runtime']['deadline_utc']); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inputs.json', dict(rows=data))
    pins = {**s['source_input_hashes'], **shared.pin([declaration_file, selection_file, records_file, recovery_file])}
    native_pins = shared.pin([native_reference/'protocol.json', native_reference/'cpu-reference/reference.json',
                              native_reference/'cpu-reference/next-logps.pt'])
    p = dict(schema=SCHEMA, created_utc=utc_now(), source_files=sources(), input_hashes=pins,
        inputs_sha256=sha256_file(output/'inputs.json'), items=N, shards=SHARDS, paths=list(PATHS), modes=list(MODES),
        ordered_item_ids=s['item_ids'], ordered_document_ids=s['document_ids'], declaration_file=str(declaration_file),
        model_dir=old['model_dir'], assets=old['assets'], model_manifest_sha256=old['assets']['manifest_sha256'],
        base_signature=ref['signature'], native_reference=str(native_reference), native_inputs=native_pins,
        writer=B0, vocab_size=old['vocab_size'], banned_ids=sorted(set(range(old['vocab_size']))-defined),
        teacher_parts=parts, parts=native['parts'], max_actions=CAP, stop_ids=[151643], selection_policy='single_greedy',
        writer_prefill_width=((max(widths)+31)//32)*32, teacher_prompt_tokens=dict(min=min(widths), max=max(widths)),
        width=WIDTH, future_width=16, precision='fp32', mixer_weight=.375, deadline_utc=declaration['proposed_runtime']['deadline_utc'],
        control_dir=str(declaration_file.parent), bootstrap=dict(repeats=5000, seed=17),
        general_reanchor='Exact last24 original prefix tokens, identical to existing GENERAL I3; no arithmetic LHS search.',
        purpose='Exploratory TRAIN hindsight-label feasibility, including previously inspected TRAIN anchors; no deployed prefix-only utility.',
        limitations=declaration['limitations'], teacher_policy=declaration['teacher'])
    atomic_json(output/'protocol.json', p); return p


def load(output):
    output = Path(output); p = shared.read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or p['items'] != N or p['shards'] != SHARDS
            or p['paths'] != list(PATHS) or p['modes'] != list(MODES) or p['writer'] != B0
            or p['max_actions'] != CAP or p['stop_ids'] != [151643] or p['selection_policy'] != 'single_greedy'
            or p['width'] != WIDTH or p['future_width'] != 16 or p['precision'] != 'fp32' or p['mixer_weight'] != .375):
        raise ValueError('Frozen hindsight source/recipe changed')
    shared.verify_pins(p['input_hashes']); shared.verify_pins(p['native_inputs'])
    shared.verify_pins({str(output/'inputs.json'): p['inputs_sha256']})
    data = shared.read(output/'inputs.json')['rows']
    if ([r['item_id'] for r in data] != p['ordered_item_ids'] or [r['document_id'] for r in data] != p['ordered_document_ids']
            or len(data) != N or len(set(p['ordered_document_ids'])) != N
            or any(r['split'] != 'train' or len(r['prefix_ids']) != 512 or len(r['natural_future16_ids']) != 16
                   or r['reanchor_ids'] != r['prefix_ids'][-24:] for r in data)):
        raise ValueError('Complete original TRAIN native records required')
    return p, data


def require_closure(p, output, phase):
    file = Path(p['control_dir'])/f'{phase}.closure.json'; c = shared.read(file)
    if (c['status'] != 'COMPLETE_PHYSICALLY_CLOSED' or not c['all4_exit0']
            or not c['all12_owned_PID_start_identities_absent'] or not c['all_sessions_absent']
            or not c['assigned_devices_empty'] or c['owned_group_survivors']
            or c['protocol_sha256'] != sha256_file(Path(output)/'protocol.json')):
        raise ValueError('Actual whole-stage physical closure required')
    return shared.pin([file])


def generate(output, shard, until, device='xla'):
    import torch
    from train.evaluate import GreedyDecoder, GenerationFailure
    shared.check(until); output = Path(output); p, data = load(output)
    if type(shard) is not int or shard not in range(SHARDS) or until != p['deadline_utc']:
        raise ValueError('Fixed shard/deadline required')
    d = output/'notes'/f'shard-{shard}'; d.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    report = dict(status='started', rows=0, shard=shard, started_utc=utc_now(), protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json', report)
    try:
        before = time.monotonic(); backend, tok, identity = shared.load_model(p, B0, device)
        report.update(writer=identity, model_setup_seconds=time.monotonic()-before); before=time.monotonic()
        report['gate'] = shared.base_gate(backend, p, shard, d, identity); report['native_gate_seconds']=time.monotonic()-before
        decoder=GreedyDecoder(backend.model, tok, sync=backend.sync, banned_ids=backend.banned_ids)
        with (d/'notes.jsonl').open('x') as stream:
            for r in data[shard::SHARDS]:
                shared.check(until); prompt=teacher_prompt(r,p['teacher_parts']); before=time.monotonic()
                row=dict(item_id=r['item_id'],document_id=r['document_id'],status='error',actions=[],raw_content_ids=[],
                    raw_text='',prefix_sha256=canonical_hash(r['prefix_ids']),future_sha256=canonical_hash(r['natural_future16_ids']),
                    teacher_prompt_sha256=canonical_hash(prompt))
                try:
                    with torch.no_grad(): value=decoder.decode(prompt,max_new_tokens=CAP,prefill_width=p['writer_prefill_width'],
                        pad_id=151643,stop_ids=(151643,),answer_line_stop=False)
                    row.update(status='ok',**asdict(value)); row['raw_content_ids']=row.pop('content_ids')
                except Exception as exc:
                    if isinstance(exc,GenerationFailure):
                        row.update(**asdict(exc.partial)); row['raw_content_ids']=row.pop('content_ids')
                    row['error']=f'{type(exc).__name__}: {exc}'
                row.update(wall_seconds=time.monotonic()-before,n_actions=len(row['actions']))
                row['extraction']=truncate_content(row['raw_content_ids'],tok)
                row.update(content_ids=row['extraction']['content_ids'],content_text=row['extraction']['content_text'])
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json',report,overwrite=True)


def validate_note(row, source, p, tok):
    a=row['actions']; stopped=bool(a and a[-1]==151643); raw=a[:-1] if stopped else a
    if (row['status'] not in ('ok','error') or row['item_id']!=source['item_id'] or row['document_id']!=source['document_id']
            or row['prefix_sha256']!=canonical_hash(source['prefix_ids']) or row['future_sha256']!=canonical_hash(source['natural_future16_ids'])
            or row['teacher_prompt_sha256']!=canonical_hash(teacher_prompt(source,p['teacher_parts']))
            or len(a)>CAP or any(type(t)is not int or not 0<=t<p['vocab_size'] or t in p['banned_ids'] for t in a)
            or row['raw_content_ids']!=raw or 151643 in raw or row['n_actions']!=len(a)
            or row['raw_text']!=decode(tok,raw) or row['extraction']!=truncate_content(raw,tok)
            or row['content_ids']!=row['extraction']['content_ids'] or row['content_text']!=row['extraction']['content_text']):
        raise ValueError('Genuine original actions/first-line extraction/source contract changed')
    if row['status']=='ok' and (not a or (row['stop_type']!='token:151643' if stopped else row['stop_type']!='length_cap' or len(a)!=CAP)):
        raise ValueError('Successful teacher path must end at EOS or cap64')
    if not math.isfinite(row['wall_seconds']) or row['wall_seconds']<0:raise ValueError('Actual nonnegative teacher cost required')


def complete_notes(output,p,data):
    from transformers import AutoTokenizer
    output=Path(output); tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    notes={};pins={};identity=None
    for shard in range(SHARDS):
        d=output/'notes'/f'shard-{shard}'; r=shared.read(d/'run.json'); values=shared.rows(d/'notes.jsonl'); expected=data[shard::SHARDS]
        if (r['status']!='complete_attempts' or r['rows']!=len(expected) or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['notes_sha256']!=sha256_file(d/'notes.jsonl') or any(r['writer'].get(k)!=v for k,v in B0.items())
                or identity is not None and identity!=r['writer'] or [v['item_id'] for v in values]!=[q['item_id'] for q in expected]):
            raise ValueError('All256 original B0 teacher attempts required before controls')
        for row,q in zip(values,expected):validate_note(row,q,p,tok);notes[row['item_id']]=row
        identity=r['writer'];pins.update(shared.pin([d/'run.json',d/'notes.jsonl']))
    return notes,pins,identity


def donors(ids):return {key:ids[(i+1)%len(ids)] for i,key in enumerate(ids)}


def controls(output):
    output=Path(output);p,data=load(output);closed=require_closure(p,output,'generate');notes,pins,identity=complete_notes(output,p,data)
    atomic_json(output/'controls.json',dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),
        generation_closure=closed,note_pins=pins,writer=identity,other_donors=donors(p['ordered_item_ids']),
        other_content_ids={key:list(notes[donor]['content_ids']) for key,donor in donors(p['ordered_item_ids']).items()}))


def reader_inputs(output,p,data):
    closed=require_closure(p,output,'generate');notes,pins,identity=complete_notes(output,p,data);c=shared.read(Path(output)/'controls.json')
    donor=donors(p['ordered_item_ids'])
    if (c['status']!='complete' or c['protocol_sha256']!=sha256_file(Path(output)/'protocol.json') or c['generation_closure']!=closed
            or c['note_pins']!=pins or c['writer']!=identity or c['other_donors']!=donor
            or c['other_content_ids']!={key:list(notes[d]['content_ids']) for key,d in donor.items()}):
        raise ValueError('Complete intact GLOBAL teacher donor barrier changed')
    return notes,pins,identity,c


def contexts(row,p,notes,control):
    key=row['item_id']; payloads=dict(empty=[],teacher=notes[key]['content_ids'],other=control['other_content_ids'][key])
    result=dict(raw=list(row['prefix_ids']),**{mode:reader_context(row,ids,'I3',p['parts']) for mode,ids in payloads.items()})
    if any(len(v)+16>WIDTH for v in result.values()):raise ValueError('Original contexts must fit without truncation')
    return result


def likelihood(lp):
    if len(lp)!=16 or any(type(v)not in (float,int) or not math.isfinite(v) or v>0 for v in lp):
        raise ValueError('All16 finite nonpositive full-support token LPs required')
    return sum(lp)


def score(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    shared.check(until);output=Path(output);p,data=load(output);notes,pins,identity,control=reader_inputs(output,p,data)
    if type(shard)is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    d=output/'scores'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'),controls_sha256=sha256_file(output/'controls.json'))
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,_,actual=shared.load_model(p,B0,device)
        report.update(reader=actual,writer=identity,model_setup_seconds=time.monotonic()-before);before=time.monotonic()
        report['gate']=shared.base_gate(backend,p,shard,d,actual);report.update(native_gate_seconds=time.monotonic()-before,shared_raw_forward_seconds=0.)
        loop=time.monotonic()
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in data[shard::SHARDS]:
                shared.check(until);key=row['item_id'];ctx=contexts(row,p,notes,control);before=time.monotonic();raw=raw_lp=y=None;raw_error=None
                try:
                    raw=full_logits(backend,ctx['raw'],row['natural_future16_ids']);y=torch.tensor([row['natural_future16_ids']],device=backend.device)
                    raw_lp=token_logps(raw,y);backend.sync();raw_values=raw_lp.cpu()[0].tolist();likelihood(raw_values)
                except Exception as exc:raw_error=f'{type(exc).__name__}: {exc}'
                raw_seconds=time.monotonic()-before;report['shared_raw_forward_seconds']+=raw_seconds
                for mode in MODES:
                    shared.check(until);before=time.monotonic();cell=dict(item_id=key,document_id=row['document_id'],mode=mode,
                        target_ids=row['natural_future16_ids'],context_sha256=canonical_hash(ctx[mode]))
                    try:
                        if raw_error:raise RuntimeError(raw_error)
                        donor=control['other_donors'][key]
                        if mode=='teacher' and notes[key]['status']!='ok' or mode=='other' and notes[donor]['status']!='ok':
                            raise ValueError('Retained failed teacher/donor; no partial-note substitution')
                        if mode=='raw':vectors=[raw_values,raw_values]
                        else:
                            note=full_logits(backend,ctx[mode],row['natural_future16_ids']);mixed,_,end=mixed_token_logps(raw,note,y,[.375],raw_lp)
                            backend.sync();vectors=[end.cpu()[0].tolist(),mixed.cpu()[0].tolist()]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,natural16_ll=likelihood(lp)) for path,lp in zip(PATHS,vectors)]
                    except Exception as exc:emitted=[dict(cell,path=path,status='error',error=f'{type(exc).__name__}: {exc}') for path in PATHS]
                    elapsed=time.monotonic()-before+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{key}/{mode}',shared_by_paths=list(PATHS),shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.)
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp,y
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),scoring_loop_seconds=time.monotonic()-loop)
    except BaseException as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def reduce_rows(values,ids,documents):
    index={(v['item_id'],v['path'],v['mode']):v for v in values}
    if len(values)!=len(index) or set(index)!={(key,path,mode) for key in ids for path in PATHS for mode in MODES}:
        raise ValueError('Every document/path/condition required exactly once')
    result={}
    for path in PATHS:
        absolute={};good={};contrasts={}
        for mode in MODES:
            rr=[index[key,path,mode] for key in ids]
            if any(v['document_id']!=doc or v['status'] not in ('ok','error') for v,doc in zip(rr,documents)):
                raise ValueError('Document/status changed')
            errors=sum(v['status']!='ok' for v in rr)
            if not errors:
                if any(not math.isfinite(v['natural16_ll']) for v in rr):raise ValueError('Nonfinite score')
                good[mode]=[dict(item_id=key,document_id=doc,score=v['natural16_ll']) for key,doc,v in zip(ids,documents,rr)]
            absolute[mode]=dict(items=len(ids),errors=errors,mean=None if errors else sum(v['natural16_ll'] for v in rr)/len(ids))
        for a,b in (('raw','teacher'),('empty','teacher'),('other','teacher'),('raw','empty')):
            name=f'{b}_minus_{a}'
            valid=[(index[key,path,a],index[key,path,b]) for key in ids]
            contrasts[name]=(paired_cluster_bootstrap(good[a],good[b],repeats=5000,seed=17) if a in good and b in good
                else dict(available=False,items=len(ids),reason='Failures retained; no complete-case mean/CI or imputation'))
            contrasts[name]['strict_positive_documents']=sum(x['status']==y['status']=='ok' and y['natural16_ll']>x['natural16_ll'] for x,y in valid)
            contrasts[name]['unavailable_documents']=sum(x['status']!='ok' or y['status']!='ok' for x,y in valid)
            contrasts[name]['denominator']=len(ids)
        result[path]=dict(absolute=absolute,contrasts=contrasts,acceptance='Per-document STRICT teacher-minus-empty>0; no ranking/margin/replacement')
    return result


def summarize(output):
    output=Path(output);p,data=load(output);closed=require_closure(p,output,'score');notes,note_pins,writer,c=reader_inputs(output,p,data)
    reports=[];pins={};values=[];reader=None
    for shard in range(SHARDS):
        d=output/'scores'/f'shard-{shard}';r=shared.read(d/'run.json')
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*len(MODES)*len(PATHS)
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['controls_sha256']!=sha256_file(output/'controls.json')
                or r['scores_sha256']!=sha256_file(d/'scores.jsonl') or any(r['reader'].get(k)!=v for k,v in B0.items())
                or r['writer']!=writer or reader is not None and reader!=r['reader']):
            raise ValueError('All four complete unchanged B0 reader shards required before outcomes')
        reader=r['reader'];reports.append(r);pins.update(shared.pin([d/'run.json',d/'scores.jsonl']))
    for shard in range(SHARDS):
        vv=shared.rows(output/'scores'/f'shard-{shard}'/'scores.jsonl');selected=data[shard::SHARDS]
        if [(v['item_id'],v['path'],v['mode']) for v in vv]!=[(r['item_id'],path,mode) for r in selected for mode in MODES for path in PATHS]:
            raise ValueError('Complete fixed row order required')
        byid={r['item_id']:r for r in selected}
        for v in vv:
            r=byid[v['item_id']]
            if (v['document_id']!=r['document_id'] or v['target_ids']!=r['natural_future16_ids']
                    or v['context_sha256']!=canonical_hash(contexts(r,p,notes,c)[v['mode']])
                    or v['status']=='ok' and v['natural16_ll']!=likelihood(v['token_logps'])):
                raise ValueError('Actual source/context/target/metric mismatch')
        values.extend(vv)
    groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids']);costs=shared.saved_costs(notes,values)
    overlap={r['item_id']:{kind:dict(copying(note,r['natural_future16_ids']),full_future16_copy=copying(note,r['natural_future16_ids'])['longest_contiguous_copied_span']>=16)
        for kind,note in (('raw',notes[r['item_id']]['raw_content_ids']),('extracted',notes[r['item_id']]['content_ids']))} for r in data}
    costs.update(raw_content_tokens=sum(len(v['raw_content_ids']) for v in notes.values()),
        notes_shortened=sum(len(v['content_ids'])<len(v['raw_content_ids']) for v in notes.values()),
        measured_generation_latency_savings=None)
    counts=dict(teacher_attempts=N,empty=sum(v['status']=='ok' and not v['content_ids'] for v in notes.values()),
        caps=sum(v['status']=='ok' and v['stop_type']=='length_cap' for v in notes.values()),
        eos=sum(v['status']=='ok' and v['stop_type']=='token:151643' for v in notes.values()),
        completed_content_lines=sum(v['extraction']['reason']=='completed_content_line' for v in notes.values()),
        teacher_errors=sum(v['status']=='error' for v in notes.values()),scoring_errors=sum(v['status']=='error' for v in values))
    copy_counts={kind:{field:sum(v[kind][field] for v in overlap.values()) for field in
        ('has_future_4gram','has_future_8gram','full_future16_copy')} for kind in ('raw','extracted')}
    result=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),items=N,cells=len(values),counts=counts,
        groups=groups,costs=costs,copying_counts=copy_counts,copying_by_document=overlap,writer=writer,reader=reader,stage_costs=reports,
        protocol_sha256=sha256_file(output/'protocol.json'),input_hashes={**pins,**note_pins,**closed,**shared.pin([output/'controls.json'])},
        limitations=p['limitations']+['Target-aware teacher labels may convey copied or paraphrased hints; token overlap cannot rule that out.',
            'All256 documents remain in denominators; an error blocks a complete primary mean/CI. No positive-label quota is implied.',
            'I3 and I6 share each full-logit forward; actual score time is counted once per timing group. Raw cap64 teacher costs stay unchanged.'])
    shared.verify_pins(result['input_hashes']);atomic_json(output/'summary.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','controls','score','summarize'))
    a.add_argument('--output',type=Path,required=True)
    for name in ('declaration','records','recovery','native-reference'):a.add_argument('--'+name,type=Path)
    a.add_argument('--shard',type=int);a.add_argument('--deadline-utc');a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':prepare(v.output,v.declaration,v.records,v.recovery,v.native_reference)
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    elif v.command=='controls':controls(v.output)
    elif v.command=='score':score(v.output,v.shard,v.deadline_utc,v.device)
    else:print(json.dumps(summarize(v.output),indent=2))


if __name__=='__main__':main()
