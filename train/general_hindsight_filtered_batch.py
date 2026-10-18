"""Finite independent B0 shards for the declared GENERAL filtered-label branch.

Composition of the qualified decoder/native FP32 gate/full-logit mixer. No
sampler, model, training objective or shared evaluation dependency is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from train import general_hindsight_filtered_data as data
from train import general_hindsight_notes as pilot
from train import useful_note_evaluation as shared
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now


def start(output,phase,shard,until,device):
    shared.check(until);output=Path(output);p,rows=data.read_shard(output,phase,shard)
    if until!=p['deadline_utc']:raise ValueError('Actual fixed batch deadline required')
    if phase!='raw':
        data.require_closure(output,p,'raw')
        if phase=='reader':data.require_closure(output,p,'teacher')
    directory=output/phase/f'shard-{shard:02d}';directory.mkdir(parents=True,exist_ok=False)
    manifest=output/'manifest.json' if phase=='raw' else data.phase_file(output,phase)
    report=dict(status='started',phase=phase,shard=shard,rows=0,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'),
        record_manifest_sha256=sha256_file(manifest),records_sha256=sha256_file(data.shard_file(output,phase,shard)))
    atomic_json(directory/'run.json',report)
    return output,p,rows,directory,report


def setup(p,shard,directory,report,device):
    before=time.monotonic();backend,tok,identity=shared.load_model(p,pilot.B0,device)
    report.update(model_setup_seconds=time.monotonic()-before,reader=identity);before=time.monotonic()
    report['gate']=shared.base_gate(backend,p,shard%4,directory,identity)
    report.update(native_gate_seconds=time.monotonic()-before,native_reference_index=shard%4)
    return backend,tok,identity


def finish(directory,report,started,error=None):
    if error is not None:report.update(status='failed',error=f'{type(error).__name__}: {error}')
    report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started)
    atomic_json(directory/'run.json',report,overwrite=True)


def raw(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,token_logps
    output,p,rows,d,report=start(output,'raw',shard,until,device);started=time.monotonic();error=None
    try:
        backend,_,_=setup(p,shard,d,report,device);loop=time.monotonic()
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in rows:
                shared.check(until);before=time.monotonic();cell={k:row[k] for k in ('item_id','document_id','pool_index')}
                cell.update(status='error',context_sha256=canonical_hash(row['prefix_ids']),target_ids=row['natural_future16_ids'])
                try:
                    logits=full_logits(backend,row['prefix_ids'],row['natural_future16_ids'])
                    target=torch.tensor([row['natural_future16_ids']],device=backend.device);v=token_logps(logits,target)
                    backend.sync();lp=v.cpu()[0].tolist();ll=pilot.likelihood(lp)
                    cell.update(status='ok',token_logps=lp,natural16_ll=ll,raw_K1_surprisal_nats=-lp[0])
                except Exception as exc:cell['error']=f'{type(exc).__name__}: {exc}'
                cell['wall_seconds']=time.monotonic()-before;stream.write(json.dumps(cell,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),loop_seconds=time.monotonic()-loop)
    except BaseException as exc:error=exc;raise
    finally:finish(d,report,started,error)


def generate_one(decoder,row,p,tok):
    """One attempt only; retain genuine partials and raw64 before extraction."""
    from train.evaluate import GenerationFailure
    prompt=pilot.teacher_prompt(row,p['teacher_parts']);started=time.monotonic()
    result={k:row[k] for k in ('item_id','document_id','pool_index','teacher_index')}
    result.update(status='error',actions=[],raw_content_ids=[],raw_text='',prefix_sha256=canonical_hash(row['prefix_ids']),
        future_sha256=canonical_hash(row['natural_future16_ids']),teacher_prompt_sha256=canonical_hash(prompt))
    try:
        value=decoder.decode(prompt,max_new_tokens=64,prefill_width=p['writer_prefill_width'],pad_id=151643,stop_ids=(151643,),answer_line_stop=False)
        result.update(status='ok',**asdict(value));result['raw_content_ids']=result.pop('content_ids')
    except Exception as exc:
        if isinstance(exc,GenerationFailure):result.update(**asdict(exc.partial));result['raw_content_ids']=result.pop('content_ids')
        result['error']=f'{type(exc).__name__}: {exc}'
    result.update(wall_seconds=time.monotonic()-started,n_actions=len(result['actions']))
    result['extraction']=pilot.truncate_content(result['raw_content_ids'],tok)
    result.update(content_ids=result['extraction']['content_ids'],content_text=result['extraction']['content_text'])
    return result


def teacher(output,shard,until,device='xla'):
    import torch
    from train.evaluate import GreedyDecoder
    output,p,rows,d,report=start(output,'teacher',shard,until,device);started=time.monotonic();error=None
    try:
        backend,tok,identity=setup(p,shard,d,report,device);report['writer']=identity;loop=time.monotonic()
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        with torch.no_grad(),(d/'notes.jsonl').open('x') as stream:
            for row in rows:
                shared.check(until);value=generate_one(decoder,row,p,tok)
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'),loop_seconds=time.monotonic()-loop)
    except BaseException as exc:error=exc;raise
    finally:finish(d,report,started,error)


def reader(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    output,p,rows,d,report=start(output,'reader',shard,until,device);started=time.monotonic();error=None
    try:
        backend,_,_=setup(p,shard,d,report,device);loop=time.monotonic();report['shared_raw_forward_seconds']=0.
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in rows:
                shared.check(until);contexts=data.native_contexts(row,p);before=time.monotonic();raw=raw_lp=y=None;raw_error=None
                try:
                    raw=full_logits(backend,contexts['raw'],row['natural_future16_ids']);y=torch.tensor([row['natural_future16_ids']],device=backend.device)
                    raw_lp=token_logps(raw,y);backend.sync();raw_values=raw_lp.cpu()[0].tolist();pilot.likelihood(raw_values)
                except Exception as exc:raw_error=f'{type(exc).__name__}: {exc}'
                raw_seconds=time.monotonic()-before;report['shared_raw_forward_seconds']+=raw_seconds
                for mode in pilot.MODES:
                    shared.check(until);before=time.monotonic();cell={k:row[k] for k in ('item_id','document_id','pool_index','teacher_index')}
                    cell.update(mode=mode,target_ids=row['natural_future16_ids'],context_sha256=canonical_hash(contexts[mode]))
                    try:
                        if raw_error:raise RuntimeError(raw_error)
                        if mode=='teacher' and row['teacher_note']['status']!='ok' or mode=='other' and row['other_note']['status']!='ok':
                            raise ValueError('Failed actual teacher/donor retained; no partial substitution')
                        if mode=='raw':vectors=[raw_values,raw_values]
                        else:
                            logits=full_logits(backend,contexts[mode],row['natural_future16_ids']);mixed,_,end=mixed_token_logps(raw,logits,y,[.375],raw_lp)
                            backend.sync();vectors=[end.cpu()[0].tolist(),mixed.cpu()[0].tolist()]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,natural16_ll=pilot.likelihood(lp)) for path,lp in zip(pilot.PATHS,vectors)]
                    except Exception as exc:emitted=[dict(cell,path=path,status='error',error=f'{type(exc).__name__}: {exc}') for path in pilot.PATHS]
                    elapsed=time.monotonic()-before+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{row["item_id"]}/{mode}',shared_by_paths=list(pilot.PATHS),shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.)
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp,y
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),loop_seconds=time.monotonic()-loop)
    except BaseException as exc:error=exc;raise
    finally:finish(d,report,started,error)


def decision_rows(values,records,p,tok):
    """All decisions, including errors/demo copies; never rank by the gain."""
    index={(r['teacher_index'],r['path'],r['mode']):r for r in values};n=len(records)
    if len(index)!=len(values) or set(index)!={(i,path,mode) for i in range(n) for path in pilot.PATHS for mode in pilot.MODES}:
        raise ValueError('All candidate/path/mode attempts required')
    demos={data.normalized_whitespace(v['note']) for v in p['teacher_policy']['synthetic_illustrative_examples']}
    result=[]
    for i,row in enumerate(records):
        teacher=row['teacher_note'];eligible=data.label_eligibility(teacher,row['natural_future16_ids'],tok,p['forbidden_content_ids'])
        per_path={}
        for path in pilot.PATHS:
            vv={mode:index[i,path,mode] for mode in pilot.MODES};complete=all(v['status']=='ok' for v in vv.values())
            ll={mode:v.get('natural16_ll') for mode,v in vv.items()};positive={mode:vv['teacher']['status']==vv[mode]['status']=='ok' and ll['teacher']>ll[mode] for mode in ('raw','empty','other')}
            per_path[path]=dict(statuses={mode:v['status'] for mode,v in vv.items()},absolute_ll=ll,strict_teacher_positive=positive,all_three=all(positive.values()),complete=complete)
        text=data.normalized_whitespace(teacher['content_text']);overlap={kind:pilot.copying(ids,row['natural_future16_ids']) for kind,ids in (('raw',teacher['raw_content_ids']),('extracted',teacher['content_ids']))}
        result.append(dict(teacher_index=i,pool_index=row['pool_index'],item_id=row['item_id'],document_id=row['document_id'],
            eligible_plain_label=eligible,accepted=bool(eligible['eligible'] and per_path['I3']['strict_teacher_positive']['empty']),
            paths=per_path,copying=overlap,demo_note_exact_whitespace_normalized_match=text in demos,
            contains_whole_demo_note=any(v and v in text for v in demos)))
    return result


def summarize(output):
    from collections import Counter
    from transformers import AutoTokenizer
    output=Path(output);p,m=data.load(output);closure=data.require_closure(output,p,'reader')
    values,pins,reports=data.complete_scores(output,p,'reader',data.TEACHER_N,len(pilot.PATHS)*len(pilot.MODES))
    records={};notes={}
    for shard in range(data.SHARDS):
        _,rows=data.read_shard(output,'reader',shard)
        expected=[(r['teacher_index'],path,mode) for r in rows for mode in pilot.MODES for path in pilot.PATHS]
        vv=[v for v in values if v['teacher_index']%data.SHARDS==shard]
        if [(v['teacher_index'],v['path'],v['mode']) for v in vv]!=expected:raise ValueError('Actual own-shard cell order changed')
        local={r['teacher_index']:r for r in rows}
        for v in vv:
            row=local[v['teacher_index']]
            if (v['item_id']!=row['item_id'] or v['document_id']!=row['document_id'] or v['target_ids']!=row['natural_future16_ids']
                    or v['context_sha256']!=canonical_hash(data.native_contexts(row,p)[v['mode']])
                    or v['status']=='ok' and v['natural16_ll']!=pilot.likelihood(v['token_logps'])):
                raise ValueError('Actual source/context/vector differs')
        records.update(local);notes.update({r['item_id']:r['teacher_note'] for r in rows})
    ordered=[records[i] for i in range(data.TEACHER_N)];ids=[r['item_id'] for r in ordered];docs=[r['document_id'] for r in ordered]
    if len(set(ids))!=data.TEACHER_N or len(set(docs))!=data.TEACHER_N:raise ValueError('Unique TRAIN documents required')
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    decisions=decision_rows(values,ordered,p,tok);data.write_rows(output/'acceptance.decisions.jsonl',decisions)
    selected=[v for v in decisions if v['accepted']][:data.LABEL_N];success=len(selected)==data.LABEL_N
    if success:
        training=[]
        for decision in selected:
            row=records[decision['teacher_index']];note=row['teacher_note'];item={k:v for k,v in row.items() if k not in ('teacher_note','other_note')}
            item.update(original_source_ids=row['prefix_ids']+row['natural_future16_ids'],teacher_note_ids=note['content_ids'],teacher_note_text=note['content_text'],
                teacher_prompt_sha256=note['teacher_prompt_sha256'],teacher_kind='future-aware-general-firstline-native-v1',selection_uses_likelihood=True,
                label_decision=decision,teacher_protocol_sha256=sha256_file(output/'protocol.json'))
            training.append(item)
        data.write_rows(output/'training.jsonl',training)
    counts=dict(raw_documents=data.RAW_N,teacher_attempts=data.TEACHER_N,reader_attempts=len(values),teacher_errors=sum(n['status']!='ok' for n in notes.values()),
        score_errors=sum(v['status']!='ok' for v in values),strict_I3_empty_positive=sum(v['paths']['I3']['strict_teacher_positive']['empty'] for v in decisions),
        eligible_positive=sum(v['accepted'] for v in decisions),selected=len(selected) if success else 0,requested=data.LABEL_N)
    diagnostics={path:{f'teacher_gt_{a}':sum(v['paths'][path]['strict_teacher_positive'][a] for v in decisions) for a in ('raw','empty','other')} for path in pilot.PATHS}
    for path in pilot.PATHS:diagnostics[path]['all_three']=sum(v['paths'][path]['all_three'] for v in decisions)
    diagnostics.update(demo_exact=sum(v['demo_note_exact_whitespace_normalized_match'] for v in decisions),demo_contained=sum(v['contains_whole_demo_note'] for v in decisions),
        rejection_reasons=dict(Counter(reason for v in decisions for reason in v['eligible_plain_label']['reasons'])))
    rm=shared.read(data.phase_file(output,'reader'));tm=shared.read(data.phase_file(output,'teacher'))
    result=dict(schema=data.SCHEMA,status='COMPLETE_FIXED5120_NATIVE_LABELS_NOT_TRAINING_RELEASE' if success else 'FAILED_FIXED_LABEL_QUOTA',finished_utc=utc_now(),
        protocol_sha256=sha256_file(output/'protocol.json'),source_files=data.sources(),counts=counts,diagnostics=diagnostics,
        groups=pilot.reduce_rows(values,ids,docs),costs=shared.saved_costs(notes,values),raw_stage_costs=[shared.read(output/'raw'/f'shard-{i:02d}'/'run.json') for i in range(data.SHARDS)],
        teacher_stage_costs=rm['teacher_stage_costs'],reader_stage_costs=reports,reader_closure=closure,
        decisions_file=str((output/'acceptance.decisions.jsonl').resolve()),decisions_sha256=sha256_file(output/'acceptance.decisions.jsonl'),
        training_file=str((output/'training.jsonl').resolve()) if success else None,training_sha256=sha256_file(output/'training.jsonl') if success else None,
        input_hashes={**pins,**shared.pin([output/'manifest.json',data.phase_file(output,'teacher'),data.phase_file(output,'reader')]),**rm['note_pins'],**tm['raw_score_pins']},
        limitations=m['limitations']+['Only I3 teacher-minus-empty>0 selects labels; teacher-minus-raw/other/demo matches are descriptive and never extra semantic gates.',
            'All24576 notes/196608 attempts and negative/error decisions remain; quota failure emits no training rows.','Full16 verbatim exclusion does not exclude shorter copying or paraphrased target hints.',
            'Costs include all raw64 actions; shorter extracted notes are not measured generation-latency savings.'])
    if success:
        atomic_json(output/'training.manifest.json',dict(schema='general-hindsight-native-labels-v1',status=result['status'],items=data.LABEL_N,split='train',purpose='supervised_labels_not_training_release',
            examples_file=result['training_file'],examples_sha256=result['training_sha256'],item_ids_sha256=canonical_hash([v['item_id'] for v in selected]),
            source_files=data.sources(),protocol_sha256=result['protocol_sha256'],decisions_sha256=result['decisions_sha256'],selection_uses_likelihood=True))
    shared.verify_pins(result['input_hashes']);atomic_json(output/'summary.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('raw','teacher','reader','summarize'));a.add_argument('--output',type=Path,required=True)
    a.add_argument('--shard',type=int);a.add_argument('--deadline-utc');a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='summarize':
        r=summarize(v.output);print(json.dumps(dict(status=r['status'],counts=r['counts'])))
    else:globals()[v.command](v.output,v.shard,v.deadline_utc,v.device)


if __name__=='__main__':main()
