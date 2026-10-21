"""Finite, release-guarded frozen-reader generation/feature workers.

Nothing runs on import. Qualification/science each require their own explicit
protocol release; a CPU data/preparation plan cannot launch these workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from train.candidate_utility_v30_driver import DEPENDENCIES as PREVIOUS_SOURCES, sample_question
from train.causal_reader_gate import extract
from train.real_trace_contexts import question_projection,reader_context,writer_prompt
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file,utc_now

ROOT=Path(__file__).resolve().parents[1]
DEPENDENCIES=tuple(dict.fromkeys((*PREVIOUS_SOURCES,'train/causal_reader_gate.py',
    'train/causal_reader_fit.py','train/causal_reader_gate_driver.py',
    'train/causal_reader_head_driver.py','scripts/check_causal_reader_qualification.py',
    'scripts/causal_reader_cost_profile.py','scripts/freeze_causal_reader_protocol.py',
    'train/causal_reader_supervisor.py')))


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).open()]
def sources():return {name:sha256_file(ROOT/name) for name in DEPENDENCIES}


def require_frozen(p):
    if p['status']!='FROZEN_CAUSAL_READER_PROTOCOL' or p['sources']!=sources():
        raise ValueError('Exact frozen reader protocol/sources required')
    for filename,digest in ((p['design_file'],p['design_sha256']),
            (p['cost_plan_file'],p['cost_plan_sha256']),(p['decision_file'],p['decision_sha256']),
            (p['binding_file'],p['binding_sha256'])):
        if sha256_file(filename)!=digest:raise ValueError('Frozen reviewed execution contract changed')
    if (sha256_file(p['data_audit_file'])!=p['data_audit_sha256']
            or read(p['data_audit_file'])['status']!='PASS_ALL_SELECTED_SOURCE_NATIVE_EXCLUSION_AND_EXACT_EQUATION_CHECKS'):
        raise ValueError('Independent complete source/native cohort audit required')


def protocol(a):
    p=read(a.protocol);require_frozen(p)
    if a.phase=='qualification':
        if p['qualification_release']['status']!='RELEASED_BOUNDED_ACTUAL_MODEL_QUALIFICATION_ONLY':
            raise ValueError('Actual-model qualification is not released')
        if a.cohort!='qualification':raise ValueError('No science documents in qualification')
    else:
        if a.device!='xla':raise ValueError('Frozen science feature path is XLA; CPU is reference-only')
        require_science_release(p,a.protocol)
        if a.cohort=='qualification':raise ValueError('No qualification reuse in science')
    spec=p['data'][a.cohort]
    if a.num_shards!=p['partitions'][a.phase][a.command]:
        raise ValueError('Exact released partition geometry required')
    if sha256_file(spec['questions_file'])!=spec['questions_sha256']:
        raise ValueError('Frozen prefix-only questions changed')
    questions=rows(spec['questions_file'])
    if len(questions)!=spec['rows'] or any(set(q)!={'item_id','document_id','prefix_ids'} for q in questions):
        raise ValueError('Exact prefix-only question projection required')
    if not 0<=a.shard_index<a.num_shards or spec['rows']%a.num_shards:
        raise ValueError('Complete predeclared evenly divided shards required')
    return p,spec,[(i,question_projection(q)) for i,q in enumerate(questions) if i%a.num_shards==a.shard_index]


def require_science_release(p,path):
    release=read(p['science_release_file'])
    if (release['status']!='RELEASED_ONE_BOUNDED_FROZEN_READER_HEAD_EXPERIMENT'
            or release['protocol_sha256']!=sha256_file(path)
            or sha256_file(release['qualification_file'])!=release['qualification_sha256']
            or read(release['qualification_file'])['status']!='PASS_ACTUAL_READER_FEATURE_QUALIFICATION'
            or sha256_file(release['profile_file'])!=release['profile_sha256']
            or read(release['profile_file'])['status']!='PASS_FINITE_ACTUAL_PROFILE'
            or read(release['profile_file'])['protocol_sha256']!=sha256_file(path)
            or release['maximum_total_reserved_chip_hours']!=8):
        raise ValueError('Actual numerical/cost qualification and finite science release required')
    return release


def check_time(a):
    if datetime.now(timezone.utc)>=datetime.fromisoformat(a.deadline_utc):
        raise TimeoutError('Finite worker deadline; no automatic retry')
    if (a.output/'STOP').exists():raise TimeoutError('Explicit worker STOP')


def load_model(p,role,device):
    from train.evaluate_local import setup_model
    from train.real_cpt_readout_driver import check_model
    binding=p['models']['B0' if role=='B0_raw' else 'T']
    backend,tok,loaded,signature,set_seed=setup_model(SimpleNamespace(
        model_dir=Path(p['model_dir']),checkpoint=Path(binding['checkpoint']) if binding['checkpoint'] else None,
        device=device))
    check_model(loaded,binding)
    if (backend.model.config.hidden_size!=p['hidden_size']
            or backend.model.config.vocab_size!=p['vocabulary_size']
            or any(parameter.dtype!=torch.float32 for parameter in backend.model.parameters())):
        raise ValueError('Frozen actual FP32 model dimensions/precision required')
    for parameter in backend.model.parameters():parameter.requires_grad_(False)
    return backend,tok,signature,set_seed


def load_notes(a,p,spec):
    if a.notes is None:raise ValueError('Complete prefix-only greedy notes required')
    result={}
    for path in sorted(a.notes.glob('part-*/complete.json')):
        complete=read(path);note_file=path.parent/'notes.jsonl'
        if (complete['status']!='COMPLETE_FROZEN_GREEDY_NOTES'
                or complete['protocol_sha256']!=sha256_file(a.protocol)
                or complete['cohort']!=a.cohort
                or sha256_file(note_file)!=complete['notes_sha256']):
            raise ValueError('Greedy note identity mismatch')
        for row in rows(note_file):
            if row['item_id'] in result or row['mode']!='greedy' or row['status']!='ok':
                raise ValueError('One intact original greedy attempt/source required')
            result[row['item_id']]=row
    if len(result)!=spec['rows']:raise ValueError('All source notes must complete before feature extraction')
    return result


def generate(a,p,spec,selected):
    if a.role!='T_note' or a.device!='xla':
        raise ValueError('Only the fixed XLA T writer generates notes')
    backend,tok,signature,set_seed=load_model(p,a.role,a.device)
    started=time.monotonic()
    with (a.output/'notes.jsonl').open('x') as stream:
        for index,q in selected:
            check_time(a)
            note=sample_question(backend,tok,set_seed,q,greedy=True)[0]
            note['global_index']=index
            stream.write(json.dumps(note,allow_nan=False)+'\n');stream.flush()
            if (index//a.num_shards+1)%32==0:print('GREEDY_SOURCES',index//a.num_shards+1,flush=True)
        os.fsync(stream.fileno())
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_FROZEN_GREEDY_NOTES',
        cohort=a.cohort,rows=len(selected),protocol_sha256=sha256_file(a.protocol),
        notes_sha256=sha256_file(a.output/'notes.jsonl'),signature=signature,
        seconds=time.monotonic()-started,completed_utc=utc_now()))


@torch.no_grad()
def dense_reference(backend,context,target):
    """Ordinary unpadded HF causal forward, not the production gather/mask."""
    x=torch.tensor([[*context,*target]])
    h=backend.model.model(input_ids=x,use_cache=False).last_hidden_state[:,len(context)-1:-1]
    lp=backend.model.lm_head(h).float().log_softmax(-1)
    return (h[0].detach(),-(lp.exp()*lp).sum(-1)[0],
            lp[0].gather(-1,torch.tensor(target)[:,None])[:,0],lp[0,0])


@torch.no_grad()
def verify_greedy_CPU(backend,tok,q,note):
    prompt=writer_prompt(q,tok);actions=note['actions']
    x=torch.tensor([[*prompt,*actions]])
    h=backend.model.model(input_ids=x[:,:-1],use_cache=False).last_hidden_state[:,len(prompt)-1:]
    z=backend.model.lm_head(h).float()
    masked=z.clone()
    if backend.banned_ids:masked[:,:,list(backend.banned_ids)]=-torch.inf
    expected=masked[0].argmax(-1).tolist()
    if expected!=actions:raise ValueError('Native greedy actions do not match ordinary HF conditionals')
    from weekend_thinking.core import policy_log_probs
    logp=policy_log_probs(z,.7,backend.banned_ids)[0].gather(-1,torch.tensor(actions)[:,None])[:,0]
    error=max(abs(x-y) for x,y in zip(logp.tolist(),note['behavior_logps']))
    if error>.001:raise ValueError('Native greedy behavior scores differ from CPU reference')
    return dict(actions=len(actions),maximum_behavior_error=error)


def features(a,p,spec,selected):
    if a.command=='reference' and (a.phase!='qualification' or a.device!='cpu'):
        raise ValueError('Independent CPU reference is qualification-only')
    if a.command=='features' and a.device!='xla':
        raise ValueError('Actual native feature qualification/science requires XLA')
    if sha256_file(spec['labels_file'])!=spec['labels_sha256']:
        raise ValueError('Frozen full native source labels changed')
    labels=rows(spec['labels_file'])
    notes=load_notes(a,p,spec) if a.role=='T_note' else None
    backend,tok,signature,_=load_model(p,a.role,a.device)
    n=len(selected);hidden_size=backend.model.config.hidden_size;vocab=backend.model.config.vocab_size
    names={'hidden':(n,16,hidden_size),'entropy':(n,16),'target_logps':(n,16)}
    if a.phase=='qualification':names['first_logps']=(n,vocab)
    arrays={name:np.lib.format.open_memmap(a.output/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape)
            for name,shape in names.items()}
    started=time.monotonic();greedy_checks=[];processed=0;witness_error=0.;batch_timings=[]
    with (a.output/'index.jsonl').open('x') as stream:
        for start in range(0,n,4):
            check_time(a);batch=selected[start:start+4];contexts=[];targets=[];indices=[]
            for index,q in batch:
                label=labels[index]
                if (label['item_id']!=q['item_id'] or label['document_id']!=q['document_id']
                        or label['prefix_ids']!=q['prefix_ids']):
                    raise ValueError('Native label/source order mismatch')
                note=notes[q['item_id']] if notes else None
                if note and note['prompt_sha256']!=canonical_hash(writer_prompt(q,tok)):
                    raise ValueError('Note belongs to another prefix')
                context=reader_context(q,tok,note['content_ids'] if note else None)
                contexts.append(context);targets.append(label['continuation16_ids'])
                indices.append(dict(global_index=index,item_id=q['item_id'],document_id=q['document_id'],
                    context_length=len(context),context_sha256=canonical_hash(context),
                    target_sha256=canonical_hash(targets[-1]),
                    note_actions=len(note['actions']) if note else 0,
                    capped=note['stop_type']=='length_cap' if note else False))
                if a.command=='reference' and note:
                    greedy_checks.append(verify_greedy_CPU(backend,tok,q,note))
            if a.command=='reference':
                individual=[dense_reference(backend,c,t) for c,t in zip(contexts,targets)]
                values=dict(zip(('hidden','entropy','target_logps','first_logps'),
                                (torch.stack([v[i] for v in individual]) for i in range(4))))
            else:
                # This is the identical feature path used in science. Extra
                # qualification vectors are a separate witness forward only.
                batch_started=time.monotonic()
                scores=extract(backend,contexts,targets)
                batch_timings.append(dict(first_local_source=start,documents=len(batch),
                    science_path_seconds=time.monotonic()-batch_started))
                values=dict(hidden=scores.features.hidden,entropy=scores.features.entropy,
                            target_logps=scores.target_logps)
                if a.phase=='qualification':
                    witness=extract(backend,contexts,targets,capture_first=True)
                    for first,second in ((scores.features.hidden,witness.features.hidden),
                            (scores.features.entropy,witness.features.entropy),
                            (scores.target_logps,witness.target_logps)):
                        witness_error=max(witness_error,(first-second).abs().max().item())
                    if witness_error>.001:raise ValueError('Diagnostic forward differs from the science path')
                    values['first_logps']=witness.first_logps
            for name,value in values.items():
                if not torch.isfinite(value).all():raise RuntimeError('Nonfinite qualified field')
                arrays[name][start:start+len(batch)]=value.numpy()
            for row in indices:stream.write(json.dumps(row)+'\n')
            processed+=len(batch)
            if processed%64==0:print('FEATURE_SOURCES',a.role,processed,flush=True)
        stream.flush();os.fsync(stream.fileno())
    for value in arrays.values():value.flush()
    for name in arrays:
        with (a.output/(name+'.npy')).open('rb') as stream:os.fsync(stream.fileno())
    pins={name+'.npy':sha256_file(a.output/(name+'.npy')) for name in arrays}
    pins['index.jsonl']=sha256_file(a.output/'index.jsonl')
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_FROZEN_READER_FEATURES',
        command=a.command,phase=a.phase,cohort=a.cohort,role=a.role,rows=n,
        protocol_sha256=sha256_file(a.protocol),files=pins,shapes=names,signature=signature,
        execution_device=str(backend.device),native_witness_max_error=witness_error,
        greedy_CPU_checks=greedy_checks,batch_timings=batch_timings,
        seconds=time.monotonic()-started,completed_utc=utc_now()))


def causality(a,p,spec,selected):
    if a.phase!='qualification' or a.device!='xla':
        raise ValueError('Actual native causality check is qualification-only')
    if sha256_file(spec['labels_file'])!=spec['labels_sha256']:
        raise ValueError('Qualification target file changed')
    labels=rows(spec['labels_file']);notes=load_notes(a,p,spec) if a.role=='T_note' else None
    backend,tok,signature,_=load_model(p,a.role,a.device)
    allowed=sorted(set(tok.get_vocab().values())-set(tok.all_special_ids))
    by_index=dict(selected);results=[]
    # Preparation fixes16 arithmetic then16 ordinary qualification sources.
    for index in (0,16):
        check_time(a);q=by_index[index];future=labels[index]['continuation16_ids']
        note=notes[q['item_id']] if notes else None
        context=reader_context(q,tok,note['content_ids'] if note else None)
        variants=[future]
        for first in (0,7,15):
            variant=list(future)
            for position in range(first,16):
                variant[position]=allowed[(future[position]+17)%len(allowed)]
                if variant[position]==future[position]:variant[position]=allowed[(future[position]+18)%len(allowed)]
            variants.append(variant)
        scores=extract(backend,[context]*4,variants)
        for row,first in enumerate((0,7,15),1):
            h_error=(scores.features.hidden[0,:first+1]-scores.features.hidden[row,:first+1]).abs().max().item()
            e_error=(scores.features.entropy[0,:first+1]-scores.features.entropy[row,:first+1]).abs().max().item()
            if h_error>1e-6 or e_error>1e-6:
                raise ValueError('Current/future target changes pre-target native features')
            results.append(dict(global_index=index,perturb_from=first,hidden_error=h_error,entropy_error=e_error))
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_ACTUAL_NATIVE_CAUSALITY_CHECK',
        role=a.role,checks=results,signature=signature,execution_device=str(backend.device),
        protocol_sha256=sha256_file(a.protocol),completed_utc=utc_now()))


def interventions(a,p,spec,selected):
    """Fixed256-source supporting diagnostics; primary predictions unchanged."""
    if a.phase!='science' or a.cohort!='DEV_calculation' or a.role!='T_note' or a.device!='xla':
        raise ValueError('Only the frozen secondary calculation panel is released here')
    from train.real_cpt_readout import controls,donors
    chosen=p['secondary_interventions']['item_ids']
    if len(chosen)!=256 or len(set(chosen))!=256:raise ValueError('Fixed256 independent source diagnostic required')
    selected=[(i,q) for i,q in selected if q['item_id'] in set(chosen)]
    if len(selected)!=64:raise ValueError('Exactly64 fixed diagnostic sources per worker required')
    if sha256_file(spec['labels_file'])!=spec['labels_sha256']:raise ValueError('Frozen labels changed')
    labels=rows(spec['labels_file']);notes=load_notes(a,p,spec);mapping=donors(chosen)
    backend,tok,signature,_=load_model(p,a.role,a.device)
    h=backend.model.config.hidden_size;files={};started=time.monotonic()
    for mode in ('empty','other','oracle','corrupt'):
        root=a.output/mode;root.mkdir();n=len(selected)
        shapes={'hidden':(n,16,h),'entropy':(n,16),'target_logps':(n,16)}
        arrays={name:np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape)
                for name,shape in shapes.items()}
        with (root/'index.jsonl').open('x') as stream:
            for start in range(0,n,4):
                check_time(a);batch=selected[start:start+4];contexts=[];targets=[];indices=[]
                for index,q in batch:
                    label=labels[index];note=notes[q['item_id']]
                    if label['item_id']!=q['item_id'] or label['prefix_ids']!=q['prefix_ids']:
                        raise ValueError('Counterfactual native source mismatch')
                    context=controls(q,tok,note,notes[mapping[q['item_id']]])[mode]
                    contexts.append(context);targets.append(label['continuation16_ids'])
                    # Metadata is held at the original generated values, so
                    # this intervention changes context content alone.
                    indices.append(dict(global_index=index,item_id=q['item_id'],document_id=q['document_id'],
                        target_sha256=canonical_hash(targets[-1]),context_sha256=canonical_hash(context),
                        note_actions=len(note['actions']),capped=note['stop_type']=='length_cap'))
                result=extract(backend,contexts,targets)
                values=dict(hidden=result.features.hidden,entropy=result.features.entropy,
                            target_logps=result.target_logps)
                for name,value in values.items():arrays[name][start:start+4]=value.numpy()
                for row in indices:stream.write(json.dumps(row)+'\n')
            stream.flush();os.fsync(stream.fileno())
        for value in arrays.values():value.flush()
        for name in arrays:
            with (root/(name+'.npy')).open('rb') as stream:os.fsync(stream.fileno())
        for path in root.iterdir():files[str(path.relative_to(a.output))]=sha256_file(path)
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_FIXED_SECONDARY_NOTE_INTERVENTIONS',
        protocol_sha256=sha256_file(a.protocol),rows=len(selected),files=files,signature=signature,
        execution_device=str(backend.device),seconds=time.monotonic()-started,
        metadata_held_at_original_generated_values=True,
        limits='Supplied oracle/corrupt calculations are prefix-interpreter controls. Whole donor notes use a fixed derangement. No oracle is deployed or used in training; these are supporting diagnostics only.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('generate','features','reference','causality','interventions'))
    for name in ('protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('qualification','science'),required=True)
    parser.add_argument('--cohort',choices=('qualification','TRAIN','DEV_calculation','DEV_ordinary'),required=True)
    parser.add_argument('--role',choices=('B0_raw','T_raw','T_note'),required=True)
    parser.add_argument('--device',choices=('cpu','xla'),required=True)
    parser.add_argument('--notes',type=Path)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--deadline-utc',required=True)
    a=parser.parse_args();torch.set_num_threads(4)
    p,spec,selected=protocol(a);check_time(a);a.output.mkdir(parents=True,exist_ok=False)
    if a.command=='generate':generate(a,p,spec,selected)
    elif a.command=='causality':causality(a,p,spec,selected)
    elif a.command=='interventions':interventions(a,p,spec,selected)
    else:features(a,p,spec,selected)


if __name__=='__main__':main()
