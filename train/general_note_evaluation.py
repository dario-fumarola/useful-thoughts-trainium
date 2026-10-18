"""Fresh prefix-only notes on the sealed GENERAL high-surprisal DEV256.

Hindsight target copies are reader plumbing controls, never writer inputs.
Every note attempt physically closes before intact global donors or futures.
Each score job loads one reader; B0 and trained-self publications are separate.
"""
from dataclasses import asdict
from pathlib import Path
import argparse, json, math, time

from train import useful_note_evaluation as shared
from train.interface_oracle_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='general-note-held256-evaluation-v1'
HELD_SHA='d2c48a1f1825c4455489e4ab7c1496cc162c27c5643305eaab45630d0f310669'
N,SHARDS,CAP,WIDTH=256,4,128,1024
MODES=('raw','empty','own','other','target_copy','corrupt_copy')
PATHS=('I3','I6w0.375')
B0=dict(id='B0',checkpoint=None,checkpoint_sha256=None)


def sources():
    return {**shared.sources(),'train/general_note_evaluation.py':sha256_file(__file__)}


def validate_prefixes(prefixes,ids,docs):
    fields={'item_id','document_id','split','prefix_ids','reanchor_ids'}
    if (len(prefixes)!=N or len(ids)!=N or len(set(ids))!=N or len(docs)!=N or len(set(docs))!=N
            or [r['item_id'] for r in prefixes]!=ids or [r['document_id'] for r in prefixes]!=docs
            or any(set(r)!=fields or r['split']!='dev' or len(r['prefix_ids'])!=512
                   or r['reanchor_ids']!=r['prefix_ids'][-24:] for r in prefixes)):
        raise ValueError('Exact future-free GENERAL DEV256 native prefixes required')


def prepare(output,held,until,writer,control_dir):
    from train.useful_note_sft_data import writer_prompt,writer_stop_ids
    shared.check(until);held=Path(held).resolve();s=shared.read(held/'selection.json');old=shared.read(held/'protocol.json')
    shared.verify_pins({str(held/'selection.json'):HELD_SHA,str(held/'protocol.json'):s['protocol_sha256'],
        str(held/'held-prefixes.jsonl'):s['prefixes_sha256']})
    cohort_closure=ROOT/'local/diagnostics'/held.name/'close.json'
    shared.verify_pins({str(cohort_closure):s['closure_sha256']})
    if (s['status']!='SEALED_BEFORE_SFT_OUTCOMES' or s['count']!=N or s['raw_documents']!=2048
            or s['raw_errors'] or s['threshold_K1_surprisal']!=1.7 or s['comparison']!='>='
            or canonical_hash(s['item_ids'])!=s['item_ids_sha256']):
        raise ValueError('Original prospectively sealed GENERAL selection required')
    prefixes=shared.rows(held/'held-prefixes.jsonl');validate_prefixes(prefixes,s['item_ids'],s['document_ids'])
    if writer_stop_ids('I3',old['parts'])!=(151668,151643):raise ValueError('Native first-close/EOS contract changed')
    if writer['id']!='B0':shared.verify_pins(writer['provenance'])
    elif writer!=B0:raise ValueError('Exact B0 identity required')
    widths=[len(writer_prompt(r['prefix_ids'],'I3',old['parts'])) for r in prefixes]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'prefixes.jsonl').open('x') as stream:
        for row in prefixes:stream.write(json.dumps(row)+'\n')
    inherited={k:old[k] for k in ('model_dir','assets','model_manifest_sha256','base_signature','native_reference','native_inputs','parts','vocab_size','banned_ids')}
    p=dict(schema=SCHEMA,created_utc=utc_now(),**inherited,source_files=sources(),writer=writer,
        input_hashes=shared.pin([held/'selection.json',held/'protocol.json',held/'held-prefixes.jsonl',cohort_closure]),
        reader_inputs_file=str(held/'held-inputs.json'),reader_inputs_sha256=s['inputs_sha256'],
        prefixes_sha256=sha256_file(output/'prefixes.jsonl'),ordered_item_ids=s['item_ids'],ordered_document_ids=s['document_ids'],
        items=N,shards=SHARDS,max_actions=CAP,writer_prefill_width=((max(widths)+31)//32)*32,
        interface='I3',selection_policy='single_greedy',width=WIDTH,modes=list(MODES),paths=list(PATHS),
        precision='fp32',writer_stop_ids=[151668,151643],generation_support='tokenizer_defined',scoring_support='full_released_head_T1',
        deadline_utc=until,control_dir=str(Path(control_dir).resolve()),bootstrap=dict(repeats=5000,seed=17),
        primary='I3 self-reader natural16 sum primary: own-minus-raw AND own-minus-intact-other lower95%>0. Fixed I6=.375 secondary control; frozen-B0 reader reported separately.',
        limitations=['B0-surprisal-conditioned developmental DEV, not unselected GENERAL or official TEST.',
            'Cap128 TOTAL actions includes sampled native first-close/EOS; capped content is never truncated.',
            'Other is an intact global cyclic own-note permutation; per-item lengths differ, population lengths are preserved.',
            'target_copy is exact native future16; corrupt_copy is next-document future16, not a certified mathematically false note.',
            'Target-copy headroom is plumbing, not useful prefix-only writer evidence. No arithmetic teacher or numeric mask.',
            'Fixed I6=.375 uses full logits; no fit or policy-training change. Shared score costs count once per timing group.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    output=Path(output);p=shared.read(output/'protocol.json')
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or p['items']!=N or p['shards']!=SHARDS
            or p['max_actions']!=CAP or p['interface']!='I3' or p['width']!=WIDTH
            or p['modes']!=list(MODES) or p['paths']!=list(PATHS) or p['selection_policy']!='single_greedy'
            or p['bootstrap']!=dict(repeats=5000,seed=17) or p['precision']!='fp32'
            or p['writer_stop_ids']!=[151668,151643]):
        raise ValueError('Frozen GENERAL schema/source/recipe required')
    shared.verify_pins(p['input_hashes']);shared.verify_pins(p['native_inputs'])
    shared.verify_pins({str(output/'prefixes.jsonl'):p['prefixes_sha256']})
    prefixes=shared.rows(output/'prefixes.jsonl');validate_prefixes(prefixes,p['ordered_item_ids'],p['ordered_document_ids'])
    return p,prefixes


def closure(output,p,stage):
    file=Path(p['control_dir'])/f'{stage}.closure.json';c=shared.read(file)
    if (c['status']!='COMPLETE_PHYSICALLY_CLOSED' or not c['all4_exit0']
            or not c['all12_owned_PID_start_identities_absent'] or not c['all_sessions_absent']
            or not c['assigned_devices_empty'] or c['owned_group_survivors']
            or c['protocol_sha256']!=sha256_file(Path(output)/'protocol.json')):
        raise ValueError('Original complete physical stage closure required')
    return shared.pin([file])


def generate(output,shard,until,device='xla'):
    import torch
    from train.evaluate import GreedyDecoder,GenerationFailure
    from train.useful_note_sft_data import writer_prompt,writer_stop_ids
    shared.check(until);output=Path(output);p,prefixes=load(output)
    if shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    d=output/'notes'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,tok,identity=shared.load_model(p,p['writer'],device)
        report.update(writer=identity,model_setup_seconds=time.monotonic()-before);before=time.monotonic()
        report['gate']=shared.base_gate(backend,p,shard,d,identity);report['native_gate_seconds']=time.monotonic()-before
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        with (d/'notes.jsonl').open('x') as stream:
            for r in prefixes[shard::SHARDS]:
                shared.check(until);before=time.monotonic();prompt=writer_prompt(r['prefix_ids'],'I3',p['parts'])
                row=dict(item_id=r['item_id'],document_id=r['document_id'],status='error',actions=[],content_ids=[],
                    prefix_sha256=canonical_hash(r['prefix_ids']),writer_prompt_sha256=canonical_hash(prompt))
                try:
                    with torch.no_grad():value=decoder.decode(prompt,max_new_tokens=CAP,prefill_width=p['writer_prefill_width'],
                        pad_id=151643,stop_ids=writer_stop_ids('I3',p['parts']),answer_line_stop=False)
                    row.update(status='ok',**asdict(value))
                except Exception as error:
                    if isinstance(error,GenerationFailure):row.update(**asdict(error.partial))
                    row['error']=f'{type(error).__name__}: {error}'
                row.update(wall_seconds=time.monotonic()-before,n_actions=len(row['actions']),n_content_tokens=len(row['content_ids']))
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def notes_and_controls(output,p,prefixes,create=False):
    closed=closure(output,p,'generate');notes,pins,writer=shared.complete_notes(output,p,prefixes)
    expected=dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),generation_closure=closed,
        note_pins=pins,writer=writer,**shared.other_controls(p,notes))
    file=output/'controls.json'
    if create:atomic_json(file,expected)
    elif shared.read(file)!=expected:raise ValueError('Intact GLOBAL note/control publication changed')
    return notes,pins,writer,expected


def reader_rows(p,prefixes):
    # Caller must establish complete note/physical/control barriers FIRST.
    shared.verify_pins({p['reader_inputs_file']:p['reader_inputs_sha256']})
    rows=shared.read(p['reader_inputs_file'])['rows']
    fields={'item_id','document_id','split','prefix_ids','reanchor_ids','natural_future16_ids'}
    if ([shared.prefix_row(r) for r in rows]!=prefixes or any(set(r)!=fields or len(r['natural_future16_ids'])!=16
            or any(type(t)is not int or not 0<=t<p['vocab_size'] or t in p['banned_ids'] for t in r['natural_future16_ids']) for r in rows)):
        raise ValueError('Exact native GENERAL256 reader targets required')
    return rows


def contexts(row,p,notes,control,byid):
    key=row['item_id'];donor=control['other_donors'][key]
    payloads=dict(empty=[],own=notes[key]['content_ids'],other=control['other_content_ids'][key],
        target_copy=row['natural_future16_ids'],corrupt_copy=byid[donor]['natural_future16_ids'])
    result=dict(raw=list(row['prefix_ids']),**{m:reader_context(row,v,'I3',p['parts']) for m,v in payloads.items()})
    if any(len(c)+16>WIDTH for c in result.values()):raise ValueError('No context truncation or positional gap allowed')
    return result


def likelihood(lp):
    if len(lp)!=16 or any(type(v)not in (float,int) or not math.isfinite(v) or v>0 for v in lp):
        raise ValueError('All16 finite nonpositive full-support token LPs required')
    return sum(lp)


def reader_identity(p,reader):
    if reader not in ('B0','self') or reader=='self' and p['writer']['id']=='B0':
        raise ValueError('Separate reader B0/self; B0 self is a duplicate publication')
    return B0 if reader=='B0' else p['writer']


def score(output,shard,reader,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    shared.check(until);output=Path(output);p,prefixes=load(output);expected=reader_identity(p,reader)
    if shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    notes,pins,writer,control=notes_and_controls(output,p,prefixes);data=reader_rows(p,prefixes);byid={r['item_id']:r for r in data}
    d=output/'scores'/reader/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,reader_role=reader,started_utc=utc_now(),
        protocol_sha256=sha256_file(output/'protocol.json'),controls_sha256=sha256_file(output/'controls.json'))
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,_,actual=shared.load_model(p,expected,device)
        report.update(reader=actual,writer=writer,model_setup_seconds=time.monotonic()-before);before=time.monotonic()
        report['gate']=shared.base_gate(backend,p,shard,d,actual);report.update(native_gate_seconds=time.monotonic()-before,shared_raw_forward_seconds=0.)
        loop=time.monotonic()
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in data[shard::SHARDS]:
                shared.check(until);key=row['item_id'];ctx=contexts(row,p,notes,control,byid);before=time.monotonic();raw=raw_lp=y=None;raw_error=None
                try:
                    raw=full_logits(backend,ctx['raw'],row['natural_future16_ids']);y=torch.tensor([row['natural_future16_ids']],device=backend.device)
                    raw_lp=token_logps(raw,y);backend.sync();raw_values=raw_lp.cpu()[0].tolist();likelihood(raw_values)
                except Exception as error:raw_error=f'{type(error).__name__}: {error}'
                raw_seconds=time.monotonic()-before;report['shared_raw_forward_seconds']+=raw_seconds
                for mode in MODES:
                    shared.check(until);before=time.monotonic();cell=dict(item_id=key,document_id=row['document_id'],mode=mode,
                        target_ids=row['natural_future16_ids'],context_sha256=canonical_hash(ctx[mode]))
                    try:
                        if raw_error:raise RuntimeError(raw_error)
                        if mode=='own' and notes[key]['status']!='ok' or mode=='other' and notes[control['other_donors'][key]]['status']!='ok':
                            raise ValueError('Failed actual note/donor retained; no partial substitution')
                        if mode=='raw':vectors=[raw_values,raw_values]
                        else:
                            note=full_logits(backend,ctx[mode],row['natural_future16_ids']);mixed,_,end=mixed_token_logps(raw,note,y,[.375],raw_lp)
                            backend.sync();vectors=[end.cpu()[0].tolist(),mixed.cpu()[0].tolist()]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,natural16_ll=likelihood(lp)) for path,lp in zip(PATHS,vectors)]
                    except Exception as error:emitted=[dict(cell,path=path,status='error',error=f'{type(error).__name__}: {error}') for path in PATHS]
                    elapsed=time.monotonic()-before+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{key}/{mode}',shared_by_paths=list(PATHS),shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.)
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp,y
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),scoring_loop_seconds=time.monotonic()-loop)
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def reduce_rows(values,ids,docs):
    index={(v['item_id'],v['path'],v['mode']):v for v in values}
    if (not ids or len(ids)!=len(docs) or len(set(ids))!=len(ids) or len(set(docs))!=len(docs)
            or len(values)!=len(index) or set(index)!={(key,path,mode) for key in ids for path in PATHS for mode in MODES}):
        raise ValueError('Complete GENERAL item/path/mode coverage exactly once required')
    result={}
    for path in PATHS:
        absolute={};good={};contrasts={}
        for mode in MODES:
            rr=[index[key,path,mode] for key in ids]
            if any(v['document_id']!=doc or v['status'] not in ('ok','error') for v,doc in zip(rr,docs)):
                raise ValueError('Document/status changed')
            errors=sum(v['status']!='ok' for v in rr)
            if not errors:
                if any(not math.isfinite(v['natural16_ll']) for v in rr):raise ValueError('Nonfinite LL')
                good[mode]=[dict(item_id=key,document_id=doc,score=v['natural16_ll']) for key,doc,v in zip(ids,docs,rr)]
            absolute[mode]=dict(items=len(ids),errors=errors,mean=None if errors else sum(v['natural16_ll'] for v in rr)/len(ids))
        for a,b in (('raw','own'),('empty','own'),('other','own'),('raw','target_copy'),('own','target_copy'),('raw','empty'),('corrupt_copy','target_copy')):
            contrasts[f'{b}_minus_{a}']=(paired_cluster_bootstrap(good[a],good[b],repeats=5000,seed=17) if a in good and b in good
                else dict(available=False,items=len(ids),reason='Errors retained; no complete-case mean/CI or imputation'))
        lower=lambda key:contrasts[key].get('ci95',[None])[0]
        positive=lambda key:lower(key)is not None and lower(key)>0
        tax=contrasts['empty_minus_raw'].get('difference_b_minus_a')
        result[path]=dict(absolute=absolute,contrasts=contrasts,gates=dict(
            own_over_raw=positive('own_minus_raw'),own_over_other=positive('own_minus_other'),
            useful_content=positive('own_minus_raw') and positive('own_minus_other'),
            target_copy_headroom=positive('target_copy_minus_own'),empty_tax_qualified=tax is not None and tax>=-.5))
    return result


def summarize(output,reader):
    output=Path(output);p,prefixes=load(output);expected=reader_identity(p,reader);closed=closure(output,p,f'score-{reader}')
    notes,note_pins,writer,control=notes_and_controls(output,p,prefixes);data=reader_rows(p,prefixes);byid={r['item_id']:r for r in data}
    values=[];pins={};reports=[];actual_reader=None
    for shard in range(SHARDS):
        d=output/'scores'/reader/f'shard-{shard}';r=shared.read(d/'run.json');vv=shared.rows(d/'scores.jsonl');selected=data[shard::SHARDS]
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*len(MODES)*len(PATHS)
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['controls_sha256']!=sha256_file(output/'controls.json')
                or r['scores_sha256']!=sha256_file(d/'scores.jsonl') or r['reader_role']!=reader
                or any(r['reader'].get(k)!=v for k,v in expected.items()) or r['writer']!=writer
                or actual_reader is not None and r['reader']!=actual_reader):raise ValueError('Complete unchanged reader/writer shards required')
        if [(v['item_id'],v['mode'],v['path']) for v in vv]!=[(q['item_id'],mode,path) for q in selected for mode in MODES for path in PATHS]:
            raise ValueError('Original full shard order required')
        for v in vv:
            row=byid[v['item_id']]
            if (v['document_id']!=row['document_id'] or v['target_ids']!=row['natural_future16_ids']
                    or v['context_sha256']!=canonical_hash(contexts(row,p,notes,control,byid)[v['mode']])
                    or v['status']=='ok' and v['natural16_ll']!=likelihood(v['token_logps'])):raise ValueError('Actual context/target/vector changed')
        values.extend(vv);reports.append(r);actual_reader=r['reader'];pins.update(shared.pin([d/'run.json',d/'scores.jsonl']))
    groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids'])
    counts=dict(notes=N,empty=sum(v['status']=='ok' and not v['content_ids'] for v in notes.values()),
        caps=sum(v['status']=='ok' and v['stop_type']=='length_cap' for v in notes.values()),
        first_close=sum(v['status']=='ok' and v['stop_type']=='token:151668' for v in notes.values()),
        eos=sum(v['status']=='ok' and v['stop_type']=='token:151643' for v in notes.values()),
        target_copy_equals_corrupt_copy=sum(r['natural_future16_ids']==byid[control['other_donors'][r['item_id']]]['natural_future16_ids'] for r in data),
        generation_errors=sum(v['status']=='error' for v in notes.values()),scoring_errors=sum(v['status']=='error' for v in values))
    result=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),items=N,cells=len(values),counts=counts,
        writer=writer,reader=actual_reader,reader_role=reader,groups=groups,costs=shared.saved_costs(notes,values),score_stages=reports,
        generation_stages=[shared.read(output/'notes'/f'shard-{s}'/'run.json') for s in range(SHARDS)],
        bootstrap='5000 paired DOCUMENT draws seed17; conditional on checkpoint/panel, not training-seed uncertainty.',
        protocol_sha256=sha256_file(output/'protocol.json'),input_hashes={**pins,**note_pins,**closed,**shared.pin([output/'controls.json'])},
        limitations=p['limitations'])
    shared.verify_pins(result['input_hashes']);atomic_json(output/f'summary-{reader}.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','controls','score','summarize'));a.add_argument('--output',type=Path,required=True)
    for name in ('held','control-dir','checkpoint','training-protocol'):a.add_argument('--'+name,type=Path)
    for name in ('checkpoint-sha256','complete-sha256','deadline-utc'):a.add_argument('--'+name)
    a.add_argument('--shard',type=int);a.add_argument('--reader',choices=('B0','self'),default='B0');a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':
        writer=B0 if v.checkpoint is None else shared.trained_identity(v.checkpoint,v.checkpoint_sha256,v.complete_sha256,v.training_protocol)
        prepare(v.output,v.held,v.deadline_utc,writer,v.control_dir)
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    elif v.command=='controls':
        p,prefixes=load(v.output);notes_and_controls(v.output,p,prefixes,create=True)
    elif v.command=='score':score(v.output,v.shard,v.reader,v.deadline_utc,v.device)
    else:print(json.dumps(summarize(v.output,v.reader),indent=2))


if __name__=='__main__':main()
