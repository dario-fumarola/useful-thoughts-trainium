"""Fresh greedy notes on the sealed raw-v3 held256; no training or final TEST.

Prepare projects only prefixes into the generation input. All256 attempted
paths must close before cyclic global donors or continuation scoring. A failed
note is retained and makes its own/donor LL unavailable; no complete-case or
zero-loss imputation is used. Symbolic teacher and literal target copy differ.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='useful-note-held256-evaluation-v1'
N,SHARDS,CAP,WIDTH=256,4,128,1024
HELD_SHA256='dbca1a7f42cf87f665652b1b5915478177fc0f3787c4db25ce186c382fdaa13a'
MODES=('raw','empty','own','other','symbolic_teacher','target_copy')
METRICS=('natural16_ll','scalar_boundary_ll','numeric_only_ll')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def check(until):
    if datetime.now(timezone.utc)>=require_utc_deadline(until):raise TimeoutError('Finite evaluation deadline; no retry')
def sources():
    from train.filter_symbolic_notes import sources as frozen
    return {**frozen(),**{name:sha256_file(ROOT/name) for name in
        ('train/useful_note_evaluation.py','train/warmstart_evaluate.py','train/interface_probe.py','train/interface_mixer_probe.py')}}
def pin(paths):return {str(Path(p).resolve()):sha256_file(p) for p in paths}
def verify_pins(values):
    for path,digest in values.items():
        if Path(path).is_symlink() or sha256_file(path)!=digest:raise ValueError(f'Bound input changed: {path}')


def trained_identity(checkpoint, expected_sha, complete_sha, training_protocol):
    """Verify the full terminal publication, including every master/moment file.

    This hashes optimizer shards without loading them. Their numerical checks
    belong to training; evaluator model loading remains strict/tied and FP32.
    """
    checkpoint=Path(checkpoint);training_protocol=Path(training_protocol)
    complete=checkpoint.parent/'COMPLETE.json';run_dir=checkpoint.parent.parent
    if checkpoint.name!='model.pt' or sha256_file(checkpoint)!=expected_sha or sha256_file(complete)!=complete_sha:
        raise ValueError('Exact terminal model/COMPLETE hashes required')
    c,r,end,p=(read(v) for v in (complete,run_dir/'run.json',run_dir/'completion.json',training_protocol))
    from train.useful_note_sft_driver import validate_protocol, SCHEMA as TRAINING_SCHEMA, WEIGHTS
    validate_protocol(p)
    spec=canonical_hash(p);step=p['steps'];world=p['world_size']
    if (p['schema']!=TRAINING_SCHEMA or p['purpose']!='science' or r['arm'] not in WEIGHTS
            or r['qualification'] or r['purpose']!='science' or r['spec_hash']!=spec
            or r['protocol_sha256']!=sha256_file(training_protocol) or r['scientific_steps']!=step
            or end['status']!='complete' or end['actual_steps']!=step or end['planned_steps']!=step
            or end['protocol_sha256']!=r['protocol_sha256'] or c['schema']!='neuron-v1'
            or (c['run_id'],c['step'],c['spec_hash'],c['world_size'])!=(run_dir.name,step,spec,world)
            or c['metadata']['arm']!=r['arm'] or c['metadata']['qualification']
            or c['metadata']['parameter_dtype']!=p['parameter_dtype'] or p['optimizer_dtype']!='fp32'
            or c['metadata']['order_hash']!=r['order_hash']
            or c['metadata']['counters']!=end['counters']
            or end['counters']['source_records']!=step*p['global_batch']):
        raise ValueError('Actual completed science lineage/counters required')
    expected={'model.pt'}|{f'rank-{i:05d}.pt' for i in range(world)}
    if set(c['files'])!=expected:raise ValueError('Complete model/master/moment shard inventory required')
    for name,entry in c['files'].items():
        file=checkpoint.parent/name
        if file.is_symlink() or file.stat().st_size!=entry['bytes'] or sha256_file(file)!=entry['sha256']:
            raise ValueError('Checkpoint model/master/moment integrity failed')
    verify_pins({str(ROOT/name):digest for name,digest in p['source_files'].items()})
    return dict(id=r['arm'],checkpoint=str(checkpoint.resolve()),checkpoint_sha256=expected_sha,
        step=step,run_id=run_dir.name,spec_hash=spec,seed=p['seed'],order_hash=r['order_hash'],
        provenance=pin([complete,run_dir/'run.json',run_dir/'completion.json',training_protocol]),
        checkpoint_files={str((checkpoint.parent/name).resolve()):entry['sha256'] for name,entry in c['files'].items()},
        training_source_files=p['source_files'])


def selected_rows(held):
    manifest=read(held['dataset_manifest_file'])
    if sha256_file(held['dataset_manifest_file'])!=held['dataset_manifest_sha256'] or manifest['schema']!='finemath-computable-equations-raw-v3':
        raise ValueError('Use the separately sealed raw-v3 held inventory')
    accepted=Path(held['dataset_manifest_file']).parent/'accepted.jsonl'
    if sha256_file(accepted)!=held['accepted_sha256']:raise ValueError('Held accepted publication changed')
    ids=held['selected_item_ids'];docs=held['selected_document_ids']
    if len(ids)!=N or len(set(ids))!=N or len(docs)!=N or len(set(docs))!=N or canonical_hash(ids)!=held['selected_item_ids_sha256']:
        raise ValueError('Exactly256 sealed unique documents, no replacement')
    wanted=set(ids);found={}
    for line in accepted.open():
        row=json.loads(line)
        if row['item_id'] in wanted:
            if row['item_id'] in found or row['split']!='dev':raise ValueError('Unique DEV held rows required')
            found[row['item_id']]=row
    if set(found)!=wanted:raise ValueError('Never drop unavailable held items')
    result=[found[key] for key in ids]
    if [r['document_id'] for r in result]!=docs:raise ValueError('Held document identity changed')
    return result,accepted


def prefix_row(row):
    return {k:row[k] for k in ('item_id','document_id','split','prefix_ids','reanchor_ids')}


def prepare(output, held_file, model_dir, native_reference, until, writer):
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.interface_oracle_probe import load as load_native, reference, serialization
    from train.useful_note_sft_data import writer_prompt,writer_stop_ids
    check(until)
    if sha256_file(held_file)!=HELD_SHA256:raise ValueError('Exact separately sealed NEW held256 required, not older DEV')
    held=read(held_file);data,accepted=selected_rows(held)
    verify_pins({held['exclusions_file']:held['exclusions_sha256'],held['phase1_manifest_file']:held['phase1_manifest_sha256']})
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    assets=asset_provenance(model_dir);old,native=load_native(native_reference);ref,_=reference(native_reference,old,native)
    if old['assets']!=assets:raise ValueError('Native reference must be actual base assets')
    fmt=read(ROOT/'local/resolved_format.json');published=read(ROOT/'configs/rlp_published_prompt.json')
    parts=serialization(tok,dict(fmt['format'],header_text=fmt['header_text']),published)
    if writer_stop_ids('I3',parts)!=(151668,151643):raise ValueError('Exact first-close/EOS stopping required')
    projected=[prefix_row(r) for r in data];lengths=[len(r['prefix_ids']) for r in projected]
    widths=[len(writer_prompt(r['prefix_ids'],'I3',parts)) for r in projected]
    defined=set(tok.get_vocab().values())
    if any(not r['prefix_ids'] or any(type(v)is not int or v not in defined for v in r['prefix_ids']+r['reanchor_ids']) for r in projected):
        raise ValueError('Complete original variable native prefixes required')
    check(until);output.mkdir(parents=True,exist_ok=False)
    with (output/'prefixes.jsonl').open('x') as stream:
        for row in projected:stream.write(json.dumps(row)+'\n')
    inputs=pin([held_file,accepted,Path(held['dataset_manifest_file']),Path(held['exclusions_file']),Path(held['phase1_manifest_file']),ROOT/'local/resolved_format.json',ROOT/'configs/rlp_published_prompt.json'])
    p=dict(schema=SCHEMA,created_utc=utc_now(),purpose='Nonfinal fresh held arithmetic development; no official TEST',
        held_file=str(Path(held_file).resolve()),items=N,ordered_item_ids=held['selected_item_ids'],ordered_document_ids=held['selected_document_ids'],
        source_files=sources(),input_hashes=inputs,prefixes_sha256=sha256_file(output/'prefixes.jsonl'),
        model_dir=str(Path(model_dir).resolve()),assets=assets,model_manifest_sha256=assets['manifest_sha256'],writer=writer,
        native_reference=str(Path(native_reference).resolve()),native_inputs=pin([native_reference/'protocol.json',native_reference/'cpu-reference/reference.json',native_reference/'cpu-reference/next-logps.pt']),
        base_signature=ref['signature'],parts=parts,interface='I3',max_actions=CAP,selection_policy='single_greedy',shards=SHARDS,
        vocab_size=old['vocab_size'],banned_ids=sorted(set(range(old['vocab_size']))-defined),
        prefix_lengths=dict(min=min(lengths),max=max(lengths),mean=sum(lengths)/N),writer_prefill_width=((max(widths)+31)//32)*32,
        width=WIDTH,modes=list(MODES),primary='Original natural16 sum; own-minus-raw and own-minus-other, paired by document',
        generation_support='Tokenizer-defined IDs only, existing cached greedy decoder',scoring_support='Full released FP32 head, unmasked T=1 token likelihoods',
        secondary='Complete original scalar+closing prefix and existing numeral-token overlap mask from the same16-vector',
        bootstrap=dict(repeats=5000,seed=17),deadline_utc=until,
        limitations=['CPU preparation reads curated rows solely to project fixed prefixes; generation reads only this prefix projection.',
            'Supervised raw-v3/exclusion inventory is fixed before outcomes; no future-dependent sample or held-item selection.',
            '128 TOTAL actions includes a sampled native close/EOS. A capped128-token content note stays unchanged and is flagged.',
            'Other is an intact global cyclic note permutation, with no repeat/trim/filler. Successful empty donors stay empty and failed donors remain unavailable. This preserves the population note-length distribution, while individual own/donor lengths can differ. Retain raw/empty contrasts alongside own-minus-other.',
            'Symbolic teacher is postorder; target_copy is the literal original scalar+boundary span.',
            'Optional I6 uses fixed.375 FULL-logit mixing, not a new fit or a policy-training path.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    p=read(output/'protocol.json')
    if (p['schema']!=SCHEMA or p['source_files']!=sources() or p['items']!=N or p['modes']!=list(MODES)
            or p['max_actions']!=CAP or p['interface']!='I3' or p['shards']!=SHARDS or p['width']!=WIDTH):
        raise ValueError('Frozen evaluator recipe/source changed')
    verify_pins(p['input_hashes']);verify_pins(p['native_inputs'])
    if sha256_file(output/'prefixes.jsonl')!=p['prefixes_sha256']:raise ValueError('Prefix projection changed')
    prefixes=rows(output/'prefixes.jsonl')
    if [r['item_id'] for r in prefixes]!=p['ordered_item_ids']:raise ValueError('Complete ordered prefixes required')
    return p,prefixes


def load_model(p,identity,device):
    from train.warmstart_evaluate import load_model as existing
    from train.evaluate import asset_provenance
    if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Actual base assets changed')
    if identity['id']!='B0':
        verify_pins(identity['provenance']);verify_pins(identity['checkpoint_files'])
        verify_pins({str(ROOT/k):v for k,v in identity['training_source_files'].items()})
    backend,tok,actual,seed=existing(p,identity,device)
    loaded=actual['loaded_checkpoint']
    if identity['id']=='B0':
        if loaded is not None or actual['model']!=p['base_signature']:raise ValueError('Actual B0 initialization changed')
    elif any(loaded.get(k)!=identity[k] for k in ('run_id','step','spec_hash')):
        raise ValueError('Loaded own checkpoint lineage differs')
    return backend,tok,actual


def base_gate(backend,p,shard,directory,identity):
    if identity['id']!='B0':return dict(scope='Own checkpoint hash/source/master-shard integrity; trained logits are not compared with B0')
    import torch
    from train.interface_oracle_probe import load as old_load,reference
    old,x=old_load(Path(p['native_reference']));_,native=reference(Path(p['native_reference']),old,x)
    q=x['qualification_refs'][shard];ids=torch.full((1,WIDTH),151643,dtype=torch.long);ids[0,:len(q['context_ids'])]=torch.tensor(q['context_ids'])
    with torch.no_grad():
        h=backend.hidden(ids.to(backend.device));v=backend.model.lm_head(h[:,len(q['context_ids'])-1]).float().log_softmax(-1);backend.sync();v=v.cpu()
    torch.save(v,directory/'forward-logps.pt');err=float((v-native[shard:shard+1]).abs().max())
    evidence=dict(scope='Initial FP32 B0 native full-head',max_error=err if math.isfinite(err) else None,atol=.002,sha256=sha256_file(directory/'forward-logps.pt'))
    atomic_json(directory/'native-check.json',evidence)
    if not math.isfinite(err) or err>.002:raise RuntimeError('Initial B0 native gate failed')
    return evidence


def generate(output,shard,until,device='xla'):
    stage_started=time.monotonic()
    import torch
    from train.evaluate import GreedyDecoder,GenerationFailure
    from train.useful_note_sft_data import writer_prompt,writer_stop_ids
    check(until);p,prefixes=load(output)
    if shard not in range(SHARDS) or until!=p['deadline_utc']:raise ValueError('Fixed shard/deadline required')
    d=output/'notes'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',rows=0,protocol_sha256=sha256_file(output/'protocol.json'),shard=shard,started_utc=utc_now())
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,tok,identity=load_model(p,p['writer'],device);report['writer']=identity
        report['model_setup_seconds']=time.monotonic()-before;before=time.monotonic()
        report['gate']=base_gate(backend,p,shard,d,identity);report['native_gate_seconds']=time.monotonic()-before
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        with (d/'notes.jsonl').open('x') as stream:
            for r in prefixes[shard::SHARDS]:
                check(until);before=time.monotonic();prompt=writer_prompt(r['prefix_ids'],'I3',p['parts'])
                row=dict(item_id=r['item_id'],document_id=r['document_id'],prefix_sha256=canonical_hash(r['prefix_ids']),writer_prompt_sha256=canonical_hash(prompt),status='error',actions=[],content_ids=[])
                try:
                    with torch.no_grad():value=decoder.decode(prompt,max_new_tokens=CAP,prefill_width=p['writer_prefill_width'],pad_id=151643,stop_ids=writer_stop_ids('I3',p['parts']),answer_line_stop=False)
                    row.update(status='ok',**asdict(value))
                except Exception as error:
                    if isinstance(error,GenerationFailure):row.update(**asdict(error.partial))
                    row['error']=f'{type(error).__name__}: {error}'
                row.update(wall_seconds=time.monotonic()-before,n_actions=len(row['actions']),n_content_tokens=len(row['content_ids']))
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();report['rows']+=1
        report.update(status='complete_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-stage_started);atomic_json(d/'run.json',report,overwrite=True)


def complete_notes(output,p,prefixes):
    from train.useful_note_sft_data import writer_prompt
    notes={};pins={};identity=None
    banned=set(p['banned_ids'])
    for shard in range(SHARDS):
        d=output/'notes'/f'shard-{shard}';r=read(d/'run.json');values=rows(d/'notes.jsonl')
        if r['status']!='complete_attempts' or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['rows']!=N//SHARDS or r['notes_sha256']!=sha256_file(d/'notes.jsonl'):
            raise ValueError('All256 complete attempted notes required before controls/futures')
        if (identity is not None and r['writer']!=identity) or any(r['writer'].get(k)!=v for k,v in p['writer'].items()):raise ValueError('Writer identity differs between shards/protocol')
        identity=r['writer'];expected=prefixes[shard::SHARDS]
        if [v['item_id'] for v in values]!=[v['item_id'] for v in expected]:raise ValueError('Note coverage/order mismatch')
        for row,q in zip(values,expected):
            a=row['actions'];stop=a[-1] if a and a[-1] in (151668,151643) else None
            content=a[:-1] if stop is not None else a
            if (row['status'] not in ('ok','error') or row['document_id']!=q['document_id'] or row['prefix_sha256']!=canonical_hash(q['prefix_ids'])
                    or row['writer_prompt_sha256']!=canonical_hash(writer_prompt(q['prefix_ids'],'I3',p['parts']))
                    or len(a)>CAP or any(type(v)is not int or not 0<=v<p['vocab_size'] or v in banned for v in a)
                    or row['content_ids']!=content or any(v in (151668,151643) for v in content)
                    or row['n_actions']!=len(a) or row['n_content_tokens']!=len(content)
                    or row['status']=='ok' and (not a or (row['stop_type']!=f'token:{stop}' if stop is not None else row['stop_type']!='length_cap' or len(a)!=CAP))):
                raise ValueError('Invalid retained note actions/stop contract')
            notes[row['item_id']]=row
        pins.update(pin([d/'run.json',d/'notes.jsonl']))
    return notes,pins,identity


def other_controls(p,notes):
    ids=p['ordered_item_ids']
    donors={key:ids[(i+1)%len(ids)] for i,key in enumerate(ids)}
    return dict(other_donors=donors,other_content_ids={key:list(notes[donor]['content_ids']) for key,donor in donors.items()})


def controls(output):
    p,prefixes=load(output);notes,pins,writer=complete_notes(output,p,prefixes)
    atomic_json(output/'controls.json',dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),note_pins=pins,writer=writer,**other_controls(p,notes),items=N))


def metric_values(lp,row):
    if len(lp)!=16 or any(type(v)not in (int,float) or not math.isfinite(v) or v>0 for v in lp):raise ValueError('Sixteen finite full-support token LPs required')
    n=len(row['target_ids'])
    if not 1<=n<=16 or row['target_ids']!=row['natural_future16_ids'][:n] or len(row['numeric_token_mask'])!=n:raise ValueError('Original scalar/boundary/number span required')
    return dict(natural16_ll=sum(lp),scalar_boundary_ll=sum(lp[:n]),numeric_only_ll=sum(v for v,on in zip(lp,row['numeric_token_mask']) if on))


def contexts(row,p,notes,other,tok):
    from train.interface_oracle_probe import reader_context
    from train.symbolic_note_labels import teacher_note,POSTORDER
    teacher=tok.encode(teacher_note(dict(row,teacher_note_kind=POSTORDER)),add_special_tokens=False)
    payloads=dict(empty=[],own=notes[row['item_id']]['content_ids'],other=other,symbolic_teacher=teacher,target_copy=row['target_ids'])
    return {'raw':list(row['prefix_ids']),**{m:reader_context(row,v,'I3',p['parts']) for m,v in payloads.items()}}


def score(output,shard,reader,until,device='xla',with_mixer=False):
    stage_started=time.monotonic()
    import torch
    from train.evaluate_likelihood import score_continuations
    from train.interface_mixer_probe import full_logits,mixed_token_logps,token_logps
    check(until);p,prefixes=load(output);notes,pins,writer=complete_notes(output,p,prefixes);control=read(output/'controls.json')
    matched=other_controls(p,notes);expected=matched['other_donors']
    if control['status']!='complete' or control['note_pins']!=pins or any(control[k]!=v for k,v in matched.items()) or control['writer']!=writer or control['protocol_sha256']!=sha256_file(output/'protocol.json'):
        raise ValueError('All-global note/control barrier required before future scoring')
    if shard not in range(SHARDS) or reader not in ('B0','self') or until!=p['deadline_utc']:raise ValueError('Fixed shard/reader/deadline required')
    identity=dict(id='B0',checkpoint=None,checkpoint_sha256=None) if reader=='B0' else p['writer']
    if reader=='self' and identity['id']=='B0':raise ValueError('B0 self reader is identical; use its B0 score publication')
    data,_=selected_rows(read(Path(p['held_file'])))
    if [prefix_row(r) for r in data]!=prefixes:raise ValueError('Original writer/reader prefixes changed')
    d=output/'scores'/reader/('with-mixer' if with_mixer else 'ordinary')/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    paths=('I3','I6w0.375') if with_mixer else ('I3',)
    report=dict(status='started',rows=0,reader_role=reader,paths=list(paths),protocol_sha256=sha256_file(output/'protocol.json'),controls_sha256=sha256_file(output/'controls.json'),started_utc=utc_now())
    atomic_json(d/'run.json',report)
    try:
        before=time.monotonic();backend,tok,actual=load_model(p,identity,device)
        report.update(reader=actual,writer=writer,model_setup_seconds=time.monotonic()-before)
        before=time.monotonic();report['gate']=base_gate(backend,p,shard,d,actual)
        report.update(native_gate_seconds=time.monotonic()-before,shared_raw_forward_seconds=0.)
        loop_started=time.monotonic()
        with torch.no_grad(),(d/'scores.jsonl').open('x') as stream:
            for row in data[shard::SHARDS]:
                key=row['item_id'];donor=expected[key];ctxs=contexts(row,p,notes,matched['other_content_ids'][key],tok);raw=None;raw_lp=None
                raw_seconds=0.
                if with_mixer:
                    raw_started=time.monotonic()
                    raw=full_logits(backend,row['prefix_ids'],row['natural_future16_ids']);y=torch.tensor([row['natural_future16_ids']],device=backend.device);raw_lp=token_logps(raw,y)
                    backend.sync();raw_values=raw_lp.cpu()[0].tolist()
                    raw_seconds=time.monotonic()-raw_started;report['shared_raw_forward_seconds']+=raw_seconds
                for mode in MODES:
                    check(until);started=time.monotonic();cell=dict(item_id=key,document_id=row['document_id'],mode=mode,context_sha256=canonical_hash(ctxs[mode]),target_ids=row['natural_future16_ids'])
                    try:
                        if mode=='own' and notes[key]['status']!='ok' or mode=='other' and notes[donor]['status']!='ok':raise ValueError('Actual failed writer path; no partial-note substitution')
                        if with_mixer:
                            if mode=='raw':lp=raw_values;mixed=lp
                            else:
                                logits=full_logits(backend,ctxs[mode],row['natural_future16_ids']);values,_,end=mixed_token_logps(raw,logits,y,[.375],raw_lp)
                                backend.sync();lp=end.cpu()[0].tolist();mixed=values.cpu()[0].tolist()
                            vectors=[lp,mixed]
                        else:vectors=[score_continuations(backend,ctxs[mode],[row['natural_future16_ids']],total_width=WIDTH,future_width=16,pad_id=151643)[0].token_log_likelihoods]
                        emitted=[dict(cell,path=path,status='ok',token_logps=lp,**metric_values(lp,row)) for path,lp in zip(paths,vectors)]
                    except Exception as error:emitted=[dict(cell,path=path,status='error',error=f'{type(error).__name__}: {error}') for path in paths]
                    elapsed=time.monotonic()-started+(raw_seconds if mode=='raw' else 0.)
                    for v in emitted:
                        v.update(wall_seconds=elapsed,timing_group=f'{key}/{mode}',shared_by_paths=list(paths),
                            shared_raw_forward_seconds=raw_seconds if mode=='raw' else 0.,
                            timing_scope='Actual cell computation shared by listed paths; sum once per timing_group, not once per emitted row. Raw cell includes the actual raw full-logit pass when mixing.')
                        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                del raw,raw_lp
        report.update(status='complete_attempts',scores_sha256=sha256_file(d/'scores.jsonl'),scoring_loop_seconds=time.monotonic()-loop_started)
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-stage_started);atomic_json(d/'run.json',report,overwrite=True)


def saved_costs(notes,values):
    """Observed saved-path costs, never warm-throughput/end-to-end estimates."""
    note_values=list(notes.values());groups={}
    for row in values:
        key=row['timing_group'];cost=(row['wall_seconds'],row['shared_raw_forward_seconds'],row['shared_by_paths'])
        if any(not math.isfinite(v) or v<0 for v in cost[:2]) or (key in groups and groups[key]!=cost):
            raise ValueError('Shared timing evidence differs between paths')
        groups[key]=cost
    return dict(total_generated_actions=sum(len(v['actions']) for v in note_values),
        total_content_tokens=sum(len(v['content_ids']) for v in note_values),
        mean_actions_per_attempt=sum(len(v['actions']) for v in note_values)/len(note_values),
        mean_content_tokens_per_attempt=sum(len(v['content_ids']) for v in note_values)/len(note_values),
        note_generation_seconds_sum=sum(v['wall_seconds'] for v in note_values),
        note_generation_seconds_mean=sum(v['wall_seconds'] for v in note_values)/len(note_values),
        scoring_shared_timing_groups=len(groups),scoring_cell_seconds_sum=sum(v[0] for v in groups.values()),
        raw_full_logits_seconds_sum=sum(v[1] for v in groups.values()),
        scope='Observed action/content and note latency include all saved attempts, including errors/partials. Score cell times count shared I3/I6 work once and include actual raw forward/softmax/transfer costs. Model setup/native gate/process-stage times are separate; no warm or end-to-end throughput claim.')


def reduce_rows(values,ids,documents,paths):
    from weekend_thinking.stats import paired_cluster_bootstrap
    index={(r['item_id'],r['path'],r['mode']):r for r in values}
    expected={(key,path,mode) for key in ids for path in paths for mode in MODES}
    if len(index)!=len(values) or set(index)!=expected:raise ValueError('Every held item/path/mode exactly once; no dropped failures')
    if any(v['status'] not in ('ok','error') for v in values):raise ValueError('Unknown attempt status')
    result={}
    for path in paths:
        absolute={};contrasts={}
        for metric in METRICS:
            good={}
            for mode in MODES:
                rr=[index[key,path,mode] for key in ids]
                if any(v['document_id']!=doc for v,doc in zip(rr,documents)):raise ValueError('Document identity mismatch')
                errors=sum(v['status']!='ok' for v in rr);available=errors==0
                absolute[f'{mode}/{metric}']=dict(items=len(ids),errors=errors,mean=sum(v[metric] for v in rr)/len(ids) if available else None)
                if available:
                    if any(not math.isfinite(v[metric]) for v in rr):raise ValueError('Nonfinite score')
                    good[mode]=[dict(item_id=key,document_id=doc,score=v[metric]) for key,doc,v in zip(ids,documents,rr)]
            for a,b in (('raw','own'),('other','own'),('empty','own'),('raw','empty'),('empty','symbolic_teacher'),('raw','target_copy')):
                contrasts[f'{b}_minus_{a}/{metric}']=paired_cluster_bootstrap(good[a],good[b],repeats=5000,seed=17) if a in good and b in good else dict(available=False,items=len(ids),reason='Missing/error path; no imputation or complete-case primary')
        result[path]=dict(absolute=absolute,contrasts=contrasts)
    return result


def summarize(output,reader,with_mixer=False):
    from transformers import AutoTokenizer
    p,prefixes=load(output);notes,note_pins,writer=complete_notes(output,p,prefixes);values=[];pins={};paths=['I3','I6w0.375'] if with_mixer else ['I3'];actual_reader=None
    control=read(output/'controls.json');matched=other_controls(p,notes)
    if control['note_pins']!=note_pins or any(control[k]!=v for k,v in matched.items()):raise ValueError('Global note/control publication changed')
    data,_=selected_rows(read(Path(p['held_file'])))
    if [prefix_row(r) for r in data]!=prefixes:raise ValueError('Scoring source/prefix mismatch')
    byid={r['item_id']:r for r in data};tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    expected_contexts={r['item_id']:contexts(r,p,notes,matched['other_content_ids'][r['item_id']],tok) for r in data}
    expected_identity=dict(id='B0',checkpoint=None,checkpoint_sha256=None) if reader=='B0' else p['writer']
    stage_costs=[]
    for shard in range(SHARDS):
        d=output/'scores'/reader/('with-mixer' if with_mixer else 'ordinary')/f'shard-{shard}';r=read(d/'run.json');vv=rows(d/'scores.jsonl')
        if (r['status']!='complete_attempts' or r['rows']!=N//SHARDS*len(MODES)*len(paths) or r['paths']!=paths
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or r['controls_sha256']!=sha256_file(output/'controls.json')):raise ValueError('Complete hash-bound reader shards required')
        if (actual_reader is not None and actual_reader!=r['reader']) or any(r['reader'].get(k)!=v for k,v in expected_identity.items()) or r['writer']!=writer:
            raise ValueError('Reader/writer differs between shards/protocol')
        for v in vv:
            row=byid[v['item_id']]
            if (v['document_id']!=row['document_id'] or v['target_ids']!=row['natural_future16_ids']
                    or v['context_sha256']!=canonical_hash(expected_contexts[v['item_id']][v['mode']])):
                raise ValueError('Scored identity/context/target changed')
            if v['status']=='ok' and any(v[k]!=z for k,z in metric_values(v['token_logps'],row).items()):raise ValueError('Saved metric differs from original token vector')
        actual_reader=r['reader'];values.extend(vv);pins.update(pin([d/'run.json',d/'scores.jsonl']))
        generation=read(output/'notes'/f'shard-{shard}'/'run.json')
        stage_costs.append(dict(shard=shard,generation={k:generation[k] for k in ('started_utc','finished_utc','model_setup_seconds','native_gate_seconds','stage_wall_seconds')},
            scoring={k:r[k] for k in ('started_utc','finished_utc','model_setup_seconds','native_gate_seconds','scoring_loop_seconds','shared_raw_forward_seconds','stage_wall_seconds')}))
    groups=reduce_rows(values,p['ordered_item_ids'],p['ordered_document_ids'],paths)
    counts=dict(notes=N,empty=sum(v['status']=='ok' and not v['content_ids'] for v in notes.values()),caps=sum(v['status']=='ok' and v.get('stop_type')=='length_cap' for v in notes.values()),errors=sum(v['status']=='error' for v in notes.values()),error_paths_with_zero_content=sum(v['status']=='error' and not v['content_ids'] for v in notes.values()),scoring_errors=sum(v['status']=='error' for v in values))
    summary=dict(status='complete_attempts_with_errors' if counts['scoring_errors'] else 'complete',counts=counts,costs=saved_costs(notes,values),stage_costs=stage_costs,writer=p['writer'],reader=actual_reader,groups=groups,protocol_sha256=sha256_file(output/'protocol.json'),input_hashes=pins,
        stage_cost_scope='Saved function-stage times include metadata checks/model setup/native gate and loop work, not interpreter startup or root orchestration. Loop time includes context construction and row serialization; shared cell times exclude file I/O. Shard sums are process-work observations, not elapsed parallel-stage latency or warm/end-to-end throughput.',
        primary='I3 natural16 own-minus-raw and own-minus-other; failures stay in256 denominator and block a complete-pair mean/CI.',
        bootstrap='5000 paired DOCUMENT draws, seed17; conditional on this checkpoint/held panel, not training-seed uncertainty.',
        limitations=p['limitations'],finished_utc=utc_now())
    atomic_json(output/f'summary-{reader}{"-with-mixer" if with_mixer else ""}.json',summary);return summary


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','controls','score','summarize'));a.add_argument('--output',type=Path,required=True)
    for name in ('held','model-dir','native-reference','checkpoint','training-protocol'):a.add_argument('--'+name,type=Path)
    for name in ('checkpoint-sha256','complete-sha256','deadline-utc'):a.add_argument('--'+name)
    a.add_argument('--shard',type=int);a.add_argument('--reader',choices=('B0','self'),default='B0');a.add_argument('--device',choices=('cpu','xla'),default='xla');a.add_argument('--with-mixer',action='store_true');v=a.parse_args()
    if v.command=='prepare':
        check(v.deadline_utc)
        identity=dict(id='B0',checkpoint=None,checkpoint_sha256=None) if v.checkpoint is None else trained_identity(v.checkpoint,v.checkpoint_sha256,v.complete_sha256,v.training_protocol)
        prepare(v.output,v.held,v.model_dir,v.native_reference,v.deadline_utc,identity)
    elif v.command=='generate':generate(v.output,v.shard,v.deadline_utc,v.device)
    elif v.command=='controls':controls(v.output)
    elif v.command=='score':score(v.output,v.shard,v.reader,v.deadline_utc,v.device,v.with_mixer)
    else:summarize(v.output,v.reader,v.with_mixer)


if __name__=='__main__':main()
