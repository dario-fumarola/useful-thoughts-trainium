"""Exploratory first-line truncation of CLOSED cached B0 held256 notes.

No generation, numeric searching, re-tokenization or target-aware note choice.
The original all128-action paths and their costs are preserved, never replaced
by claims that keeping fewer cached tokens measured a faster writer.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import time

from train import useful_note_evaluation as parent
from train.interface_oracle_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import paired_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'postorder-firstline-note-held256-v1'
DECLARATION_SHA = 'f379835f8c56b766dfdca98ca985f156e611bd485730e8d1dd5a7aeb6a7813ba'
N, SHARDS, CONTENT_CAP = 256, 4, 64
PATHS = ('I3','I6w0.375')
MODES = ('raw','empty','full_own','full_other','short_own','short_other')
CONTRASTS = (('raw','short_own'),('short_other','short_own'),('empty','short_own'),
             ('full_own','short_own'),('raw','full_own'),('full_other','full_own'))


def sources():
    return {**parent.sources(), 'train/firstline_note_probe.py':sha256_file(__file__)}


def truncate_content(content_ids, tokenizer):
    """Only token IDs+tokenizer enter this policy. Keep an EXACT token prefix.

    Leading blank lines remain. A newline completes a line only after the first
    non-whitespace character. The whole token containing that newline remains,
    including any suffix it encodes; never split or re-encode that token.
    """
    ids = list(content_ids)
    if any(type(v) is not int or v < 0 for v in ids):
        raise ValueError('Original native content token IDs required')
    decode = lambda values:tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    stop = min(len(ids),CONTENT_CAP); boundary = None
    for end in range(1,stop+1):
        text = decode(ids[:end]); first = next((i for i,c in enumerate(text) if not c.isspace()),None)
        if first is not None and '\n' in text[first+1:]:
            stop = end; boundary = end-1; break
    kept = ids[:stop]
    return dict(content_ids=kept,content_text=decode(kept),original_tokens=len(ids),kept_tokens=len(kept),
        boundary_token_index=boundary,reason='completed_content_line' if boundary is not None else
        'content_cap' if len(ids)>CONTENT_CAP else 'original_end')


def same_data_api(archived_text, current_text):
    """The old module may differ ONLY in its unused trained-identity function."""
    def shape(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='trained_identity']
        return ast.dump(tree,include_attributes=False)
    return shape(archived_text)==shape(current_text)


def archive_entry(archive_file, archive, path, digest, *, current=True):
    path=Path(path).resolve();matches=[v for v in archive['entries'] if v['source']==str(path)]
    if len(matches)!=1 or matches[0]['sha256']!=digest:
        raise ValueError('Old artifact/source not bound by the declared archive')
    relative=Path(matches[0]['relative'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Archive relative path escaped its closed inventory')
    saved=Path(archive_file).parent/relative
    parent.verify_pins({str(saved):digest})
    if current:parent.verify_pins({str(path):digest})
    return saved


def closed_old(old_output, archive_file, declaration):
    """Validate old hashes/source lineage, explicitly WITHOUT parent.load()."""
    old_output=Path(old_output).resolve();archive_file=Path(archive_file).resolve();bounds=declaration['data']
    files={old_output/'protocol.json':bounds['original_protocol_sha256'],
           old_output/'summary-B0-with-mixer.json':bounds['original_complete_summary_sha256'],
           archive_file:bounds['original_EBS_archive_sha256']}
    parent.verify_pins({str(p):h for p,h in files.items()})
    p=parent.read(old_output/'protocol.json');summary=parent.read(old_output/'summary-B0-with-mixer.json');archive=parent.read(archive_file)
    if (archive['status']!='complete_SHA256_source_and_destination_readback_verified'
            or summary['status'] not in ('complete','complete_attempts_with_errors')
            or summary['protocol_sha256']!=bounds['original_protocol_sha256']
            or archive['protocol_sha256']!=bounds['original_protocol_sha256']
            or archive['summary_sha256']!=bounds['original_complete_summary_sha256']
            or p['writer']!=dict(id='B0',checkpoint=None,checkpoint_sha256=None)
            or p['items']!=N or p['max_actions']!=128 or p['shards']!=SHARDS):
        raise ValueError('Exact complete cached B0 held256 experiment required')
    bridge={}
    for name,digest in p['source_files'].items():
        actual=ROOT/name;saved=archive_entry(archive_file,archive,actual,digest,current=False)
        if sha256_file(actual)!=digest:
            if name!='train/useful_note_evaluation.py' or not same_data_api(saved.read_text(),actual.read_text()):
                raise ValueError('Legacy data/native/mixer helper changed beyond trained_identity')
            bridge[name]=dict(old_sha256=digest,current_sha256=sha256_file(actual),archived_file=str(saved),
                scope='Only trained_identity differs; every remaining module AST node unchanged. Old outcomes retain old source identity.')
    parent.verify_pins(p['input_hashes']);parent.verify_pins(p['native_inputs'])
    prefixes=parent.rows(old_output/'prefixes.jsonl')
    parent.verify_pins({str(old_output/'prefixes.jsonl'):p['prefixes_sha256']})
    if ([r['item_id'] for r in prefixes]!=p['ordered_item_ids']
            or [r['document_id'] for r in prefixes]!=p['ordered_document_ids']):
        raise ValueError('Original held prefix/document order changed')
    notes,note_pins,writer=parent.complete_notes(old_output,p,prefixes)
    control=parent.read(old_output/'controls.json');matched=parent.other_controls(p,notes)
    if (control['status']!='complete' or control['protocol_sha256']!=bounds['original_protocol_sha256']
            or control['note_pins']!=note_pins or control['writer']!=writer or any(control[k]!=v for k,v in matched.items())):
        raise ValueError('Original complete global notes/donor controls required')
    paths=dict(files);paths.update({Path(k):v for k,v in note_pins.items()});paths.update({Path(k):v for k,v in summary['input_hashes'].items()})
    paths[old_output/'controls.json']=sha256_file(old_output/'controls.json')
    paths[old_output/'prefixes.jsonl']=p['prefixes_sha256']
    control_dir=ROOT/'local/diagnostics'/old_output.name
    for phase in ('generate','score'):
        path=control_dir/f'{phase}.closure.json';closure=parent.read(path)
        if (closure['status']!='COMPLETE_PHYSICALLY_CLOSED' or not closure['all4_exit0']
                or not closure['all12_owned_PID_start_identities_absent'] or not closure['all_sessions_absent']
                or not closure['assigned_devices_empty'] or closure['owned_group_survivors']
                or closure['protocol_sha256']!=bounds['original_protocol_sha256']):
            raise ValueError('Both original physical closures required')
        paths[path]=sha256_file(path)
    for path,digest in paths.items():
        if path!=archive_file:archive_entry(archive_file,archive,path,digest)
    parent.verify_pins(summary['input_hashes'])
    values=[v for path in summary['input_hashes'] if Path(path).name=='scores.jsonl' for v in parent.rows(path)]
    expected={(key,path,mode) for key in p['ordered_item_ids'] for path in PATHS for mode in parent.MODES}
    if len(values)!=N*12 or len({(v['item_id'],v['path'],v['mode']) for v in values})!=len(values) or {(v['item_id'],v['path'],v['mode']) for v in values}!=expected:
        raise ValueError('All original3072 score attempts required')
    data,_=parent.selected_rows(parent.read(Path(p['held_file'])))
    if [parent.prefix_row(r) for r in data]!=prefixes:
        raise ValueError('Curated source and original writer prefixes differ')
    return p,data,notes,matched,values,{str(k):v for k,v in paths.items()},bridge


def make_contexts(row,parts,notes,shorts,donors):
    key=row['item_id'];donor=donors[key]
    payloads=dict(empty=[],full_own=notes[key]['content_ids'],full_other=notes[donor]['content_ids'],
                  short_own=shorts[key]['content_ids'],short_other=shorts[donor]['content_ids'])
    return dict(raw=list(row['prefix_ids']),**{mode:reader_context(row,content,'I3',parts) for mode,content in payloads.items()})


def prepare(output,old_output,archive_file,declaration_file):
    from transformers import AutoTokenizer
    declaration_file=Path(declaration_file).resolve();parent.verify_pins({str(declaration_file):DECLARATION_SHA})
    declaration=parent.read(declaration_file);until=declaration['boundaries']['deadline_utc'];parent.check(until)
    old,data,notes,control,values,pins,bridge=closed_old(old_output,archive_file,declaration)
    tok=AutoTokenizer.from_pretrained(old['model_dir'],local_files_only=True,trust_remote_code=False)
    shorts={key:truncate_content(note['content_ids'],tok) for key,note in notes.items()};hashes={}
    old_index={(v['item_id'],v['path'],v['mode']):v for v in values}
    for row in data:
        contexts=make_contexts(row,old['parts'],notes,shorts,control['other_donors'])
        if any(len(c)+16>1024 for c in contexts.values()):raise ValueError('No context truncation beyond the declared note policy')
        hashes[row['item_id']]={mode:canonical_hash(c) for mode,c in contexts.items()}
        for new,previous in (('raw','raw'),('empty','empty'),('full_own','own'),('full_other','other')):
            if any(old_index[row['item_id'],path,previous]['context_sha256']!=hashes[row['item_id']][new] for path in PATHS):
                raise ValueError('Unchanged full/raw/empty context differs from original score evidence')
    parent.check(until);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'inputs.json',dict(rows=data,notes=notes,short_notes=shorts,other_donors=control['other_donors'],context_hashes=hashes))
    p=dict(schema=SCHEMA,created_utc=utc_now(),purpose='REUSED exploratory DEV; no new generation/training or official TEST',
        items=N,shards=SHARDS,modes=list(MODES),paths=list(PATHS),mixer_weight=.375,content_cap=CONTENT_CAP,
        declaration_file=str(declaration_file),declaration_sha256=DECLARATION_SHA,source_files=sources(),legacy_source_bridge=bridge,
        input_hashes={**pins,str(declaration_file):DECLARATION_SHA,**old['input_hashes'],**old['native_inputs']},
        inputs_sha256=sha256_file(output/'inputs.json'),old_output=str(Path(old_output).resolve()),
        ordered_item_ids=old['ordered_item_ids'],ordered_document_ids=old['ordered_document_ids'],
        model_dir=old['model_dir'],assets=old['assets'],model_manifest_sha256=old['model_manifest_sha256'],
        base_signature=old['base_signature'],native_reference=old['native_reference'],native_inputs=old['native_inputs'],
        writer=old['writer'],parts=old['parts'],width=1024,precision='fp32',deadline_utc=until,
        policy=declaration['policy'],pass_rule=declaration['pass_rule'],bootstrap=dict(repeats=5000,seed=17),
        limitations=[declaration['rationale'],declaration['costs'],'No best-of/target-aware truncation, replacement or omitted note.',
            'Two paths share computation; score wall time is counted once per item/mode timing group.',
            'Both positive primary lower bounds within a path are required; any PASS remains exploratory on reused DEV.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    output=Path(output);p=parent.read(output/'protocol.json')
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or p['items']!=N or p['shards']!=SHARDS
            or p['paths']!=list(PATHS) or p['modes']!=list(MODES) or p['mixer_weight']!=.375
            or p['content_cap']!=64 or p['precision']!='fp32' or p['width']!=1024):
        raise ValueError('Frozen firstline source/policy changed')
    parent.verify_pins(p['input_hashes']);parent.verify_pins({str(output/'inputs.json'):p['inputs_sha256']})
    x=parent.read(output/'inputs.json')
    if [r['item_id'] for r in x['rows']]!=p['ordered_item_ids'] or [r['document_id'] for r in x['rows']]!=p['ordered_document_ids']:
        raise ValueError('All original256 items/documents required')
    return p,x


def score(output,shard,until,device='xla'):
    import torch
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    parent.check(until);output=Path(output);p,x=load(output)
    if type(shard)is not int or shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    d=output/'scores'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='started',rows=0,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,_,identity=parent.load_model(p,p['writer'],device)
        report.update(reader=identity,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic();report['gate']=parent.base_gate(backend,p,shard,d,identity)
        report.update(native_gate_seconds=time.monotonic()-before,shared_raw_forward_seconds=0.);loop=time.monotonic()
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in x['rows'][shard::SHARDS]:
                parent.check(until);key=row['item_id'];contexts=make_contexts(row,p['parts'],x['notes'],x['short_notes'],x['other_donors'])
                raw=raw_lp=y=None;before=time.monotonic();raw_error=None
                try:
                    raw=full_logits(backend,contexts['raw'],row['natural_future16_ids'])
                    y=torch.tensor([row['natural_future16_ids']],device=backend.device);raw_lp=token_logps(raw,y)
                    backend.sync();raw_values=raw_lp.cpu()[0].tolist()
                except Exception as exc:raw_error=f'{type(exc).__name__}: {exc}'
                raw_seconds=time.monotonic()-before;report['shared_raw_forward_seconds']+=raw_seconds
                for mode in MODES:
                    parent.check(until);before=time.monotonic();cell=dict(item_id=key,document_id=row['document_id'],mode=mode,
                        target_ids=row['natural_future16_ids'],context_sha256=canonical_hash(contexts[mode]))
                    try:
                        if raw_error:raise RuntimeError(raw_error)
                        donor=x['other_donors'][key]
                        if (mode.endswith('_own') and x['notes'][key]['status']!='ok'
                                or mode.endswith('_other') and x['notes'][donor]['status']!='ok'):
                            raise ValueError('Retained failed original note, no partial-note substitution')
                        if mode=='raw':vectors=[raw_values,raw_values]
                        else:
                            note=full_logits(backend,contexts[mode],row['natural_future16_ids'])
                            mixed,_,end=mixed_token_logps(raw,note,y,[.375],raw_lp);backend.sync()
                            vectors=[end.cpu()[0].tolist(),mixed.cpu()[0].tolist()]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,**parent.metric_values(lp,row)) for path,lp in zip(PATHS,vectors)]
                    except Exception as exc:emitted=[dict(cell,path=path,status='error',error=f'{type(exc).__name__}: {exc}') for path in PATHS]
                    elapsed=time.monotonic()-before+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{key}/{mode}',shared_by_paths=list(PATHS),
                            shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.)
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp,y
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),scoring_loop_seconds=time.monotonic()-loop)
    except BaseException as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started);atomic_json(d/'run.json',report,overwrite=True)


def reduce_rows(values,ids,documents):
    index={(r['item_id'],r['path'],r['mode']):r for r in values}
    if len(index)!=len(values) or set(index)!={(key,path,mode) for key in ids for path in PATHS for mode in MODES}:
        raise ValueError('Every item/path/condition exactly once; no filtered denominators')
    result={}
    for path in PATHS:
        absolute={};contrasts={}
        for metric in parent.METRICS:
            good={}
            for mode in MODES:
                rr=[index[key,path,mode] for key in ids]
                if any(r['document_id']!=doc or r['status'] not in ('ok','error') for r,doc in zip(rr,documents)):
                    raise ValueError('Document/status changed')
                errors=sum(r['status']=='error' for r in rr)
                if not errors and any(not math.isfinite(r[metric]) for r in rr):raise ValueError('Nonfinite metric')
                absolute[f'{mode}/{metric}']=dict(items=len(ids),errors=errors,mean=None if errors else sum(r[metric] for r in rr)/len(ids))
                if not errors:good[mode]=[dict(item_id=key,document_id=doc,score=r[metric]) for key,doc,r in zip(ids,documents,rr)]
            for a,b in CONTRASTS:
                contrasts[f'{b}_minus_{a}/{metric}']=(paired_cluster_bootstrap(good[a],good[b],repeats=5000,seed=17)
                    if a in good and b in good else dict(available=False,items=len(ids),reason='Error retained; no imputation/complete-case mean'))
        primary=[contrasts[f'short_own_minus_{a}/natural16_ll'] for a in ('raw','short_other')]
        result[path]=dict(absolute=absolute,contrasts=contrasts,
            exploratory_primary_pass=all(v.get('ci95',[None])[0] is not None and v['ci95'][0]>0 for v in primary))
    return result


def summarize(output):
    output=Path(output);p,x=load(output);values=[];pins={};identity=None;reports=[];byid={r['item_id']:r for r in x['rows']}
    # Check every completed shard receipt/hash BEFORE parsing new score vectors.
    for shard in range(SHARDS):
        d=output/'scores'/f'shard-{shard}';r=parent.read(d/'run.json')
        if (r['status']!='complete_attempts' or r['rows']!=len(x['rows'][shard::SHARDS])*len(PATHS)*len(MODES)
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or r['reader']['id']!='B0' or r['reader']['checkpoint'] is not None or identity is not None and identity!=r['reader']):
            raise ValueError('All four complete unchanged B0 score shards required before outcomes')
        identity=r['reader'];pins.update(parent.pin([d/'run.json',d/'scores.jsonl']));reports.append(r)
    for shard in range(SHARDS):
        vv=parent.rows(output/'scores'/f'shard-{shard}'/'scores.jsonl')
        expected=[(row['item_id'],path,mode) for row in x['rows'][shard::SHARDS] for mode in MODES for path in PATHS]
        if [(v['item_id'],v['path'],v['mode']) for v in vv]!=expected:raise ValueError('Every original cell in fixed shard order required')
        for v in vv:
            row=byid[v['item_id']]
            if (v['document_id']!=row['document_id'] or v['target_ids']!=row['natural_future16_ids']
                    or v['context_sha256']!=x['context_hashes'][v['item_id']][v['mode']]):raise ValueError('Cell source/context/target changed')
            if v['status']=='ok' and any(v[k]!=z for k,z in parent.metric_values(v['token_logps'],row).items()):raise ValueError('Metric differs from retained16vector')
        values.extend(vv)
    groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids']);costs=parent.saved_costs(x['notes'],values)
    costs.update(kept_content_tokens=sum(v['kept_tokens'] for v in x['short_notes'].values()),
        notes_shortened=sum(v['kept_tokens']<v['original_tokens'] for v in x['short_notes'].values()),
        completed_content_lines=sum(v['reason']=='completed_content_line' for v in x['short_notes'].values()),
        measured_generation_latency_savings=None)
    result=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),items=N,cells=len(values),
        protocol_sha256=sha256_file(output/'protocol.json'),input_hashes=pins,groups=groups,costs=costs,reader=identity,
        scoring_errors=sum(v['status']=='error' for v in values),legacy_source_bridge=p['legacy_source_bridge'],
        stage_costs=reports,limitations=p['limitations'])
    parent.verify_pins(pins);atomic_json(output/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','score','summarize'))
    parser.add_argument('--output',type=Path,required=True)
    for name in ('old-output','archive','declaration'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--shard',type=int);parser.add_argument('--deadline-utc');parser.add_argument('--device',choices=('cpu','xla'),default='xla')
    a=parser.parse_args()
    if a.command=='prepare':
        p=prepare(a.output,a.old_output,a.archive,a.declaration);print(json.dumps(dict(items=p['items'],cells=N*len(PATHS)*len(MODES))))
    elif a.command=='score':score(a.output,a.shard,a.deadline_utc,a.device)
    else:print(json.dumps(summarize(a.output),indent=2))


if __name__=='__main__':main()
